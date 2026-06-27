"""Document-anchored, recall-first retrieval for G-LRAG.

Motivation (measured on the dev set):
  - BM25 article-level recall@200 ≈ 0.15, but **document-level** recall@50 ≈ 0.75.
  - The pipeline reliably finds the right *văn bản* yet buries the right *Điều*
    among hundreds of sibling chunks, and provincial/superseded documents crowd
    out the authoritative current article. 18/20 dev questions score zero.

Strategy: a two-stage, recall-first design.

  Stage A — Document anchoring
      Fuse the lexical (BM25/FTS) and dense (FAISS) chunk rankings into a
      *document* ranking via RRF over each document's best chunk rank, nudged by
      a soft authority prior (central + recent statutes up, provincial down).
      Keep the top-N candidate documents.

  Stage B — Within-document article harvesting
      From the union of lexical + dense hits, keep every chunk that belongs to
      an anchored document, collapse to the best chunk per ``Điều X``, and emit
      an article-candidate pool.

  Stage C — Article rerank (GPU, injected)
      A cross-encoder reranks (query, article-text) pairs. This is the step that
      lifts the right article above its siblings. It is injected as a callable
      so this module imports no torch/faiss and is unit-testable; on Kaggle the
      callable wraps ``BAAI/bge-reranker-v2-m3``.

The article pool this produces feeds :func:`retrieval.article_select.select_articles`
for the final F2-optimal K selection.

This module imports only stdlib + the in-repo pure-Python ``rrf`` and
``article_select`` helpers. Dense search, text fetch and rerank are injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from retrieval.article_select import (
    ArticleCandidate,
    AuthorityConfig,
    authority_prior,
    canonical_dieu,
    is_provincial,
)
from retrieval.rrf import rrf_score_only

# Injected callable types ---------------------------------------------------- #
# Lexical/dense search: query -> ranked list of hit dicts (best first). Each hit
# must carry: row_idx, doc_uid, law_id, ten_van_ban, dieu_so.
SearchFn = Callable[[str, int], List[Dict[str, Any]]]
# Text fetch: row_idxs -> {row_idx: chunk_text}.
TextFn = Callable[[Sequence[int]], Dict[int, str]]
# Rerank: (query, passages) -> one score per passage (higher = better).
RerankFn = Callable[[str, Sequence[str]], List[float]]


@dataclass
class DocAnchorConfig:
    """Tunable knobs for the document-anchoring retrieval stage."""

    top_bm25: int = 100          # lexical candidates pulled (wide for recall)
    top_dense: int = 100         # dense candidates pulled
    rrf_k: int = 60              # RRF smoothing constant
    top_docs: int = 10           # anchored documents kept
    doc_authority_weight: float = 1.0  # scale on authority prior at doc stage
    per_doc_articles: int = 6    # max distinct articles harvested per doc
    rerank_pool: int = 40        # max article candidates sent to the reranker
    authority: AuthorityConfig = field(default_factory=AuthorityConfig)


# --------------------------------------------------------------------------- #
# Stage A: document anchoring (pure)
# --------------------------------------------------------------------------- #
@dataclass
class AnchoredDoc:
    doc_uid: str
    law_id: str
    ten_van_ban: str
    fused_score: float   # RRF + authority
    rrf_score: float
    prior: float


def _doc_key(hit: Dict[str, Any]) -> str:
    """Stable per-document key (prefer doc_uid, fall back to law_id)."""
    duid = str(hit.get("doc_uid", "")).strip()
    return duid if duid else str(hit.get("law_id", "")).strip()


def _document_ranking(hits: Sequence[Dict[str, Any]]) -> List[str]:
    """Collapse a chunk ranking to a document ranking by first appearance.

    A document's rank is the rank of its best (earliest) chunk, which is what
    RRF needs as input.
    """
    seen: Dict[str, None] = {}
    for hit in hits:
        key = _doc_key(hit)
        if key and key not in seen:
            seen[key] = None
    return list(seen.keys())


def anchor_documents(
    lexical_hits: Sequence[Dict[str, Any]],
    dense_hits: Sequence[Dict[str, Any]],
    cfg: DocAnchorConfig,
) -> List[AnchoredDoc]:
    """Fuse lexical + dense chunk hits into a ranked, authority-adjusted doc list."""
    # Per-document metadata (first hit wins for display fields).
    meta: Dict[str, Dict[str, str]] = {}
    for hit in list(lexical_hits) + list(dense_hits):
        key = _doc_key(hit)
        if key and key not in meta:
            meta[key] = {
                "law_id": str(hit.get("law_id", "")),
                "ten_van_ban": str(hit.get("ten_van_ban", "")),
            }

    lex_doc_rank = _document_ranking(lexical_hits)
    den_doc_rank = _document_ranking(dense_hits)

    # RRF over document rankings. rrf_score_only keys on int, but our doc keys
    # are strings, so we fuse manually using the same 1/(k+rank) formula.
    scores: Dict[str, float] = {}
    for ranking in (lex_doc_rank, den_doc_rank):
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (cfg.rrf_k + rank)

    out: List[AnchoredDoc] = []
    for key, rrf in scores.items():
        m = meta.get(key, {"law_id": key, "ten_van_ban": ""})
        prior = authority_prior(m["law_id"], m["ten_van_ban"], cfg.authority)
        fused = rrf + cfg.doc_authority_weight * prior
        out.append(
            AnchoredDoc(
                doc_uid=key,
                law_id=m["law_id"],
                ten_van_ban=m["ten_van_ban"],
                fused_score=fused,
                rrf_score=rrf,
                prior=prior,
            )
        )
    out.sort(key=lambda d: (d.fused_score, d.rrf_score), reverse=True)
    return out[: cfg.top_docs]


# --------------------------------------------------------------------------- #
# Stage B: within-document article harvesting (pure)
# --------------------------------------------------------------------------- #
@dataclass
class HarvestedChunk:
    row_idx: int
    law_id: str
    ten_van_ban: str
    dieu_so: str
    base_score: float  # best available pre-rerank score (dense/lexical/rrf)


def _hit_score(hit: Dict[str, Any]) -> float:
    """Best available pre-rerank score from a hit dict."""
    for key in ("dense_score", "bm25_score", "rrf_score", "score"):
        if key in hit and hit[key] is not None:
            try:
                return float(hit[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def harvest_articles(
    lexical_hits: Sequence[Dict[str, Any]],
    dense_hits: Sequence[Dict[str, Any]],
    anchored: Sequence[AnchoredDoc],
    cfg: DocAnchorConfig,
) -> List[HarvestedChunk]:
    """Keep chunks belonging to anchored docs; best chunk per (doc, Điều X).

    Returns at most ``per_doc_articles`` articles per document, ordered by score
    within each document, then flattened.
    """
    anchored_keys = {d.doc_uid for d in anchored}
    # (doc_key, dieu) -> best HarvestedChunk
    best: Dict[Tuple[str, str], HarvestedChunk] = {}
    for hit in list(dense_hits) + list(lexical_hits):
        key = _doc_key(hit)
        if key not in anchored_keys:
            continue
        dieu = canonical_dieu(str(hit.get("dieu_so", "")))
        if not dieu.startswith("Điều"):
            continue
        score = _hit_score(hit)
        bkey = (key, dieu)
        cur = best.get(bkey)
        if cur is None or score > cur.base_score:
            best[bkey] = HarvestedChunk(
                row_idx=int(hit.get("row_idx", -1)),
                law_id=str(hit.get("law_id", "")),
                ten_van_ban=str(hit.get("ten_van_ban", "")),
                dieu_so=dieu,
                base_score=score,
            )

    # Cap articles per document.
    by_doc: Dict[str, List[HarvestedChunk]] = {}
    for (key, _dieu), hc in best.items():
        by_doc.setdefault(key, []).append(hc)
    harvested: List[HarvestedChunk] = []
    for key, items in by_doc.items():
        items.sort(key=lambda h: h.base_score, reverse=True)
        harvested.extend(items[: cfg.per_doc_articles])
    return harvested


# --------------------------------------------------------------------------- #
# Stage C: orchestration (dense/text/rerank injected)
# --------------------------------------------------------------------------- #
class DocAnchoredRetriever:
    """Recall-first retriever producing an article-candidate pool.

    Parameters
    ----------
    lexical_search, dense_search:
        ``(query, top_k) -> [hit dict]``. ``dense_search`` may be ``None`` for a
        lexical-only run (useful when the GPU dense leg is unavailable).
    fetch_text:
        ``(row_idxs) -> {row_idx: text}`` used to build rerank passages.
    rerank:
        ``(query, passages) -> [score]``. May be ``None`` to skip reranking
        (candidates then carry their pre-rerank ``base_score``).
    """

    def __init__(
        self,
        lexical_search: SearchFn,
        dense_search: Optional[SearchFn],
        fetch_text: Optional[TextFn] = None,
        rerank: Optional[RerankFn] = None,
        cfg: Optional[DocAnchorConfig] = None,
    ) -> None:
        self.lexical_search = lexical_search
        self.dense_search = dense_search
        self.fetch_text = fetch_text
        self.rerank = rerank
        self.cfg = cfg or DocAnchorConfig()

    def retrieve_pool(self, query: str) -> List[ArticleCandidate]:
        """Return reranked :class:`ArticleCandidate` list for one query."""
        cfg = self.cfg
        lex = self.lexical_search(query, cfg.top_bm25) or []
        den = self.dense_search(query, cfg.top_dense) if self.dense_search else []
        den = den or []

        anchored = anchor_documents(lex, den, cfg)
        harvested = harvest_articles(lex, den, anchored, cfg)
        if not harvested:
            return []

        # Limit how many candidates we rerank (cost control).
        harvested.sort(key=lambda h: h.base_score, reverse=True)
        harvested = harvested[: cfg.rerank_pool]

        scores = self._score(query, harvested)
        return [
            ArticleCandidate(
                law_id=h.law_id,
                ten_van_ban=h.ten_van_ban,
                dieu_so=h.dieu_so,
                score=s,
            )
            for h, s in zip(harvested, scores)
        ]

    def _score(self, query: str, harvested: List[HarvestedChunk]) -> List[float]:
        """Rerank passages if a reranker + text fetch are available."""
        if self.rerank is None or self.fetch_text is None:
            return [h.base_score for h in harvested]
        row_idxs = [h.row_idx for h in harvested if h.row_idx >= 0]
        texts = self.fetch_text(row_idxs) if row_idxs else {}
        passages = [texts.get(h.row_idx, h.dieu_so) for h in harvested]
        scores = self.rerank(query, passages)
        if len(scores) != len(harvested):
            # Defensive: fall back to base scores on a shape mismatch.
            return [h.base_score for h in harvested]
        return [float(s) for s in scores]


def pool_record(
    qid: int, question: str, candidates: Sequence[ArticleCandidate]
) -> Dict[str, Any]:
    """Serialise a query's candidates to the tune_selector pool schema."""
    return {
        "id": int(qid),
        "question": question,
        "candidates": [
            {
                "law_id": c.law_id,
                "ten_van_ban": c.ten_van_ban,
                "dieu_so": c.dieu_so,
                "score": round(float(c.score), 6),
            }
            for c in candidates
        ],
    }
