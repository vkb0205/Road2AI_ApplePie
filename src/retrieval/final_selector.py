"""Article-first final selection for SME Legal QA (``final_selection.md``).

Final select is Stage 6 of the 7-stage pipeline. It receives the (noisy,
over-expanded) chunk pool produced by rerank + graph expansion and chooses the
**smallest set of evidence that still preserves the correct legal chain**.

Core principle (``final_selection.md`` §2, §6): selection is **article-first**.
Group chunks by article, score articles, select the article set by lane policy,
then pick supporting chunks *within* the selected articles.

    chunk pool → group by article → score articles → lane-budgeted selection
               → pick supporting chunks → compact ordered context

This module reuses the grader-faithful canonicalisation and soft
document-authority prior from :mod:`retrieval.article_select` (which the legacy
selector also used), but reorganises the flow around explicit per-lane budgets
and a redundancy penalty, per ``final_selection.md`` §7–§11.

Pure-Python (stdlib only) — no retrieval deps — so it is unit-testable offline
and drops directly into the Kaggle notebook after graph expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retrieval.article_select import (
    AuthorityConfig,
    authority_prior,
    canonical_dieu,
    is_provincial,
)

__all__ = [
    "LANE_BUDGETS",
    "ChunkEvidence",
    "ScoredArticle",
    "SelectionConfig",
    "SelectionResult",
    "FinalSelector",
]

# --------------------------------------------------------------------------- #
# Per-lane budgets (final_selection.md §10, §11)
# --------------------------------------------------------------------------- #
# (max_articles, max_chunks_per_article). Coverage widens for multi-hop /
# scenario lanes; direct lookups stay tight. These are POLICY numbers — no
# law/domain tables — and are the primary tuning surface for F2.
LANE_BUDGETS: Dict[str, Tuple[int, int]] = {
    "direct_lookup": (1, 2),
    "procedure_detail": (2, 3),
    "condition_requirement": (3, 2),
    "sanction": (2, 2),
    "cross_doc": (4, 2),
    "scenario": (6, 2),
}
# Fallback when an unknown lane label is passed.
_DEFAULT_BUDGET: Tuple[int, int] = (3, 2)


# --------------------------------------------------------------------------- #
# Input / output dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class ChunkEvidence:
    """One chunk in the post-expansion pool handed to final select.

    Attributes
    ----------
    row_idx:
        Canonical bundle join key (chunks table row).
    law_id, ten_van_ban, dieu_so:
        Article identity (``law_id|ten_van_ban|Điều X``).
    score:
        Rerank score (ideally normalised to ~[0, 1]).
    chunk_id:
        Optional chunk identifier.
    text:
        Optional chunk text (for the generation context).
    from_graph:
        ``True`` if this chunk was added by graph expansion (vs a rerank seed).
    """

    row_idx: int
    law_id: str
    ten_van_ban: str
    dieu_so: str
    score: float
    chunk_id: Optional[str] = None
    text: Optional[str] = None
    from_graph: bool = False


@dataclass
class ScoredArticle:
    """An article aggregated from its supporting chunks, with a final score."""

    dieu: str            # canonical "Điều X"
    law_id: str          # representative (highest-prior) source document
    ten_van_ban: str
    relevance: float     # best chunk score in the article
    support: float       # bounded bonus for multiple supporting chunks
    graph_gain: float    # bonus if discovered/strengthened by graph expansion
    authority: float     # soft document-authority prior
    redundancy: float    # penalty vs already-selected articles (filled at select time)
    final_score: float   # relevance + support + graph_gain + authority - redundancy
    chunks: List[ChunkEvidence] = field(default_factory=list)

    def to_relevant_string(self) -> str:
        return f"{self.law_id}|{self.ten_van_ban}|{self.dieu}"


@dataclass
class SelectionResult:
    """The compact evidence final select returns to the generator."""

    articles: List[ScoredArticle] = field(default_factory=list)
    chunks: List[ChunkEvidence] = field(default_factory=list)
    selection_metadata: Dict[str, Any] = field(default_factory=dict)

    def relevant_articles(self) -> List[str]:
        """Grader-ready ``law|ten|Điều`` strings, in selection order."""
        return [a.to_relevant_string() for a in self.articles]

    def relevant_docs(self) -> List[str]:
        """Distinct ``law_id|ten_van_ban`` strings, preserving order."""
        seen: set = set()
        out: List[str] = []
        for a in self.articles:
            key = f"{a.law_id}|{a.ten_van_ban}"
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out


@dataclass
class SelectionConfig:
    """Tunable weights + policy for article-first selection.

    Scoring (``final_selection.md`` §7):
        ``final = relevance + support + graph_gain + authority - redundancy``

    Weights are deliberately small relative to a normalised rerank score in
    ``[0, 1]`` so a strong semantic match dominates, with the priors only
    breaking near-ties (matching the legacy authority-prior philosophy).
    """

    authority: AuthorityConfig = field(default_factory=AuthorityConfig)
    drop_provincial: bool = True

    # Support: each extra useful chunk in an article adds ``support_per_chunk``
    # up to ``support_cap`` (multi-chunk articles are weak evidence of being
    # the governing article, not strong — keep the bonus small).
    support_per_chunk: float = 0.03
    support_cap: float = 0.09

    # Graph gain: bonus when an article was discovered/reinforced by graph
    # expansion (the multi-hop coverage signal). Applied once per article.
    graph_gain_bonus: float = 0.05

    # Redundancy penalty: subtracted from a candidate's score for each already
    # selected article that shares its representative document (suppresses
    # near-duplicate articles from the same văn bản crowding the budget).
    redundancy_penalty: float = 0.04

    # Recall floor: always keep at least this many articles when available,
    # even if the lane budget or margins would trim further.
    min_articles: int = 1

    # Admission margin: after the top article, admit a runner-up only if its
    # final_score is within ``rel_margin`` (fraction) of the top — keeps
    # precision without collapsing recall. Lanes with larger budgets still cap
    # at the budget; this only prevents admitting clearly-irrelevant tails.
    rel_margin: float = 0.45


# --------------------------------------------------------------------------- #
# Final selector
# --------------------------------------------------------------------------- #
class FinalSelector:
    """Article-first evidence selection with per-lane budgets.

    Parameters
    ----------
    config:
        :class:`SelectionConfig`.
    """

    def __init__(self, config: Optional[SelectionConfig] = None) -> None:
        self.config = config or SelectionConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def select(
        self,
        chunk_pool: Sequence[ChunkEvidence],
        lane: str = "direct_lookup",
    ) -> SelectionResult:
        """Select the minimal sufficient article set + supporting chunks.

        Steps (``final_selection.md`` §15):
          1. Group chunks by canonical article ``Điều X``.
          2. Score each article (relevance + support + graph_gain + authority).
          3. Rank articles; apply redundancy penalty + admission margin.
          4. Select up to the lane's article budget (recall floor honoured).
          5. Pick supporting chunks within each selected article.
          6. Build the compact ordered context.
        """
        max_articles, max_chunks = LANE_BUDGETS.get(lane, _DEFAULT_BUDGET)

        # 1-2: group + score.
        scored = self._score_articles(chunk_pool)
        if not scored:
            return SelectionResult(
                selection_metadata={"lane": lane, "reason": "empty pool"}
            )

        # 3-4: rank + redundancy-aware, margin-gated, budget-capped selection.
        selected = self._select_articles(scored, max_articles)

        # 5: pick supporting chunks within each selected article.
        out_chunks: List[ChunkEvidence] = []
        for art in selected:
            art_chunks = sorted(
                art.chunks, key=lambda c: c.score, reverse=True
            )[:max_chunks]
            art.chunks = art_chunks
            out_chunks.extend(art_chunks)

        return SelectionResult(
            articles=selected,
            chunks=out_chunks,
            selection_metadata={
                "lane": lane,
                "budget": {"max_articles": max_articles, "max_chunks": max_chunks},
                "n_candidates": len(scored),
                "n_selected": len(selected),
            },
        )

    # ------------------------------------------------------------------ #
    # Step 1-2: group chunks by article + score
    # ------------------------------------------------------------------ #
    def _score_articles(
        self, chunk_pool: Sequence[ChunkEvidence]
    ) -> List[ScoredArticle]:
        """Group chunks by canonical ``Điều X`` and compute per-article scores."""
        cfg = self.config
        acfg = cfg.authority

        # canonical dieu -> list of contributing chunks.
        buckets: Dict[str, List[ChunkEvidence]] = {}
        for c in chunk_pool:
            if cfg.drop_provincial and is_provincial(c.law_id, acfg):
                continue
            dieu = canonical_dieu(c.dieu_so)
            if not dieu.startswith("Điều"):
                continue
            buckets.setdefault(dieu, []).append(c)

        scored: List[ScoredArticle] = []
        for dieu, chunks in buckets.items():
            chunks_sorted = sorted(chunks, key=lambda c: c.score, reverse=True)
            best = chunks_sorted[0]

            # Representative document = highest authority-adjusted contributor.
            rep = max(
                chunks_sorted,
                key=lambda c: c.score + authority_prior(c.law_id, c.ten_van_ban, acfg),
            )

            relevance = best.score
            # Support: bounded bonus for additional useful chunks.
            n_extra = max(0, len(chunks_sorted) - 1)
            support = min(cfg.support_cap, n_extra * cfg.support_per_chunk)
            # Graph gain: applied if ANY contributing chunk came from expansion.
            graph_gain = (
                cfg.graph_gain_bonus
                if any(c.from_graph for c in chunks_sorted)
                else 0.0
            )
            authority = authority_prior(rep.law_id, rep.ten_van_ban, acfg)

            final = relevance + support + graph_gain + authority
            scored.append(
                ScoredArticle(
                    dieu=dieu,
                    law_id=rep.law_id,
                    ten_van_ban=rep.ten_van_ban,
                    relevance=relevance,
                    support=support,
                    graph_gain=graph_gain,
                    authority=authority,
                    redundancy=0.0,
                    final_score=final,
                    chunks=chunks_sorted,
                )
            )

        scored.sort(key=lambda a: a.final_score, reverse=True)
        return scored

    # ------------------------------------------------------------------ #
    # Step 3-4: redundancy-aware, margin-gated, budget-capped selection
    # ------------------------------------------------------------------ #
    def _select_articles(
        self, scored: List[ScoredArticle], max_articles: int
    ) -> List[ScoredArticle]:
        """Greedily admit articles up to budget, with redundancy + margin gates."""
        cfg = self.config
        if not scored:
            return []

        chosen: List[ScoredArticle] = [scored[0]]
        selected_docs: Dict[str, int] = {scored[0].law_id: 1}
        top = scored[0].final_score

        for art in scored[1:]:
            if len(chosen) >= max_articles:
                break
            # Redundancy penalty: shared representative document with a chosen
            # article suggests near-duplicate legal content.
            shared = selected_docs.get(art.law_id, 0)
            art.redundancy = shared * cfg.redundancy_penalty
            adjusted = art.final_score - art.redundancy

            # Admission margin (scale-aware): for normalised scores, admit when
            # within rel_margin fraction of the top. For raw/large scores the
            # fraction still behaves sensibly (relative gap).
            if top > 0:
                within_margin = adjusted >= top * (1.0 - cfg.rel_margin)
            else:
                within_margin = True

            if within_margin:
                chosen.append(art)
                selected_docs[art.law_id] = shared + 1
            # Do not break: a later article from a different doc may still be
            # within margin even if this one (penalised) was not. Budget caps
            # the loop. This favours recall (F2 weights recall 4× precision).

        # Recall floor.
        if len(chosen) < cfg.min_articles:
            chosen = scored[: cfg.min_articles]

        return chosen
