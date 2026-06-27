"""Article-level selection for G-LRAG retrieval (F2-optimised).

The official grader reduces every ``law_id|ten_van_ban|Điều X`` string to its
bare ``Điều X`` number and macro-averages precision / recall / F2 (β=2) over
questions. Two consequences drive this module:

1. **Recall is king.** F2 weights recall 4× precision, so the dominant failure
   mode (observed: 18/20 dev questions with zero correct articles) is *missing*
   the gold article, not over-returning. The selector must not trim
   aggressively.
2. **Document authority matters for ranking, not for scoring.** Because the
   grader is article-number-only, a correct number from *any* document scores.
   But authoritative central documents (Luật / Nghị định / Thông tư, recent)
   are far likelier to carry the gold article than provincial decisions
   (``QĐ-UBND``) or superseded decrees. We therefore use a soft authority prior
   to re-rank candidates *before* selecting, suppressing the provincial /
   superseded noise that crowds out the right article in the baseline.

This module is pure-Python (stdlib only) and has no retrieval dependencies, so
it is unit-testable offline and drops directly into the Kaggle notebook after
the cross-encoder rerank stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Canonicalisation (grader-faithful)
# --------------------------------------------------------------------------- #
_DIEU_RE = re.compile(r"Điều\s*(\d+[a-zA-Z]*(?:[.\-]\d+)?)", re.UNICODE)
# Year embedded in a law_id, e.g. "01/2021/NĐ-CP" -> 2021, "72/2002/QĐ-UB" -> 2002.
_YEAR_RE = re.compile(r"/(\d{4})/")


def canonical_dieu(dieu_so: str) -> str:
    """Canonicalise a ``Điều X`` string to ``"Điều <number>"`` (single space).

    Matches ``dev_set/eval.py._canonical_dieu`` exactly so selector output is
    scored consistently with the grader.
    """
    s = str(dieu_so).strip()
    m = re.match(r"^(Điều)\s*(.+)$", s, re.UNICODE)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    # Bare number or "Điều X" embedded elsewhere.
    m2 = _DIEU_RE.search(s)
    if m2:
        return f"Điều {m2.group(1)}"
    return s


# --------------------------------------------------------------------------- #
# Document authority prior
# --------------------------------------------------------------------------- #
@dataclass
class AuthorityConfig:
    """Tunable weights for the soft document-authority prior.

    The prior is added (in score space) to each candidate's retrieval/rerank
    score. Values are deliberately small relative to a normalised reranker
    score in ``[0, 1]`` so that a strong semantic match can still outrank a
    weak match from a more authoritative document.
    """

    # Additive bonus by document type (central statutes preferred).
    type_bonus: Dict[str, float] = field(
        default_factory=lambda: {
            "QH": 0.06,       # Luật (National Assembly) — top central statute
            "NĐ-CP": 0.05,    # Nghị định (Government decree)
            "ND-CP": 0.05,    # ASCII-folded variant
            "TT": 0.04,       # Thông tư (ministry circular)
            "QĐ-TTg": 0.03,   # Prime-minister decision
            "QĐ-TCHQ": 0.0,   # central agency decision (neutral)
            "QĐ-UBND": -0.08, # provincial decision — strong suppression
            "QĐ-UB": -0.08,   # legacy provincial decision
        }
    )
    # Per-year recency bonus; newer law preferred. Bounded by recency_cap.
    recency_per_year: float = 0.004
    recency_ref_year: int = 2000
    recency_cap: float = 0.10
    # Hard suppression: substrings in law_id that mark a provincial issuer.
    provincial_markers: Tuple[str, ...] = ("QĐ-UBND", "QĐ-UB")


def parse_law_type(law_id: str, ten_van_ban: str = "") -> str:
    """Best-effort document-type tag from a ``law_id`` (e.g. ``NĐ-CP``, ``QĐ-UBND``).

    Falls back to a coarse class parsed from the document title when the
    ``law_id`` suffix is unrecognised.
    """
    lid = str(law_id)
    # The type token is the segment after the last "/", e.g. "01/2021/NĐ-CP".
    tail = lid.rsplit("/", 1)[-1].strip()
    if tail:
        # QH13 / QH14 -> QH ; TT-BTC / TT-BKHĐT -> TT
        if tail.startswith("QH"):
            return "QH"
        if tail.startswith("TT"):
            return "TT"
        if tail in ("NĐ-CP", "ND-CP"):
            return tail
        if tail.startswith("QĐ-UBND") or tail.startswith("QĐ-UB"):
            return "QĐ-UBND" if "UBND" in tail else "QĐ-UB"
        if tail.startswith("QĐ-"):
            return tail
        return tail
    title = str(ten_van_ban)
    if title.startswith("Luật"):
        return "QH"
    if title.startswith("Nghị định"):
        return "NĐ-CP"
    if title.startswith("Thông tư"):
        return "TT"
    if title.startswith("Quyết định"):
        return "QĐ"
    return "?"


def parse_year(law_id: str) -> Optional[int]:
    m = _YEAR_RE.search(str(law_id))
    return int(m.group(1)) if m else None


def authority_prior(law_id: str, ten_van_ban: str, cfg: AuthorityConfig) -> float:
    """Compute the additive authority prior for one document."""
    law_type = parse_law_type(law_id, ten_van_ban)
    bonus = cfg.type_bonus.get(law_type, 0.0)
    year = parse_year(law_id)
    if year is not None:
        rec = (year - cfg.recency_ref_year) * cfg.recency_per_year
        rec = max(-cfg.recency_cap, min(cfg.recency_cap, rec))
        bonus += rec
    return bonus


def is_provincial(law_id: str, cfg: AuthorityConfig) -> bool:
    lid = str(law_id)
    return any(mark in lid for mark in cfg.provincial_markers)


# --------------------------------------------------------------------------- #
# Candidate + aggregation
# --------------------------------------------------------------------------- #
@dataclass
class ArticleCandidate:
    """One retrieved chunk's article, with the score from the rerank stage."""

    law_id: str
    ten_van_ban: str
    dieu_so: str
    score: float  # reranker score, ideally normalised to ~[0, 1]


@dataclass
class AggregatedArticle:
    """An article number aggregated across all chunks/documents that carry it."""

    dieu: str            # canonical "Điều X"
    law_id: str          # representative (highest-prior) source document
    ten_van_ban: str
    base_score: float    # best raw score among contributing candidates
    final_score: float   # base + authority prior + agreement bonus
    n_support: int       # how many distinct authoritative docs carried it

    def to_relevant_string(self) -> str:
        return f"{self.law_id}|{self.ten_van_ban}|{self.dieu}"


@dataclass
class SelectConfig:
    """Tunable selection policy (tuned on the dev set)."""

    authority: AuthorityConfig = field(default_factory=AuthorityConfig)
    drop_provincial: bool = True          # hard-drop provincial issuers
    max_k: int = 3                        # never return more than this
    min_k: int = 1                        # always return at least this
    # Add the n-th candidate (n>=2) only if its score is within `rel_margin`
    # (fraction of the top score) AND `abs_margin` of the top score.
    rel_margin: float = 0.18
    abs_margin: float = 0.12
    # Small bonus when several authoritative docs independently agree on a
    # number — multi-document agreement is weak evidence of correctness.
    agreement_bonus: float = 0.02
    agreement_cap: int = 3


def aggregate_articles(
    candidates: Sequence[ArticleCandidate], cfg: SelectConfig
) -> List[AggregatedArticle]:
    """Collapse chunk-level candidates into article-level entries.

    Grouping key is the canonical ``Điều X`` number (the grader is
    number-only). For each number we keep the highest authority-adjusted score
    and the most authoritative contributing document as the representative.
    """
    acfg = cfg.authority
    # number -> list of (adjusted_score, base_score, prior, law_id, ten, is_prov)
    buckets: Dict[str, List[Tuple[float, float, float, str, str, bool]]] = {}
    for c in candidates:
        prov = is_provincial(c.law_id, acfg)
        if cfg.drop_provincial and prov:
            continue
        dieu = canonical_dieu(c.dieu_so)
        if not dieu.startswith("Điều"):
            continue
        prior = authority_prior(c.law_id, c.ten_van_ban, acfg)
        adj = c.score + prior
        buckets.setdefault(dieu, []).append(
            (adj, c.score, prior, c.law_id, c.ten_van_ban, prov)
        )

    out: List[AggregatedArticle] = []
    for dieu, items in buckets.items():
        items.sort(key=lambda t: t[0], reverse=True)
        best_adj, best_base, _prior, law_id, ten, _prov = items[0]
        # Distinct supporting documents (by law_id).
        distinct_docs = {it[3] for it in items}
        n_support = len(distinct_docs)
        agree = min(n_support, cfg.agreement_cap) - 1
        final = best_adj + agree * cfg.agreement_bonus
        out.append(
            AggregatedArticle(
                dieu=dieu,
                law_id=law_id,
                ten_van_ban=ten,
                base_score=best_base,
                final_score=final,
                n_support=n_support,
            )
        )
    out.sort(key=lambda a: a.final_score, reverse=True)
    return out


def select_articles(
    candidates: Sequence[ArticleCandidate], cfg: Optional[SelectConfig] = None
) -> List[AggregatedArticle]:
    """Full pipeline: suppress → aggregate → F2-optimal K selection.

    Returns the chosen articles in rank order. The K policy always keeps the
    top article (recall floor), then admits further articles only when their
    score is close to the top (a confident second/third article), capped at
    ``max_k``. This matches the dev-set gold cardinality (16×1, 4×2) while
    protecting recall.
    """
    cfg = cfg or SelectConfig()
    ranked = aggregate_articles(candidates, cfg)
    if not ranked:
        return []

    top = ranked[0].final_score
    chosen = [ranked[0]]
    for art in ranked[1:]:
        if len(chosen) >= cfg.max_k:
            break
        within_rel = art.final_score >= top * (1.0 - cfg.rel_margin) if top > 0 else False
        within_abs = (top - art.final_score) <= cfg.abs_margin
        if within_rel and within_abs:
            chosen.append(art)
        else:
            break
    # Honour min_k even if margins were not met (recall floor).
    if len(chosen) < cfg.min_k:
        chosen = ranked[: cfg.min_k]
    return chosen


def select_relevant_strings(
    candidates: Sequence[ArticleCandidate], cfg: Optional[SelectConfig] = None
) -> List[str]:
    """Convenience wrapper returning grader-ready ``law|ten|Điều`` strings."""
    return [a.to_relevant_string() for a in select_articles(candidates, cfg)]
