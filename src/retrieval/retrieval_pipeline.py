"""Unified 7-stage retrieval pipeline for SME Legal QA (``retrieval.md``).

A single, maintainable orchestrator that supersedes the legacy multi-pipeline
sprawl (``retriever.py`` / ``unified_pipeline.py`` / ``doc_anchor.py`` /
``kaggle_pipeline.py``) by composing the unit-tested building blocks:

    query
      → 1. Router.route            (lane + strategy switches)
      → 2. Decomposer.decompose    (2–5 sub-questions, only if should_decompose)
      → 3. hybrid retrieve         (per query: BM25 + FAISS → RRF; merge+dedup)
      → 4. rerank                  (cross-encoder on the ORIGINAL query)
      → 5. GraphExpander.expand    (CHUNK→ART→DOC→DOC→ART→CHUNK, lane-gated)
      → 6. FinalSelector.select    (article-first, per-lane budget)
      → 7. (answer — handled by the caller's generator)

The CRITICAL rule (``query_decomposition.md`` §5): **retrieve with the
sub-questions, but rerank with the original query**. Sub-questions broaden
recall; only the original query carries true user intent for relevance.

Every heavy dependency (FAISS / torch / transformers / NetworkX graph) is
reached only through injected callables / objects, so this module imports and
unit-tests on a CPU-only box with fake callables.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from retrieval.article_select import canonical_dieu
from retrieval.decomposer import Decomposer, DecompositionResult
from retrieval.final_selector import (
    ChunkEvidence,
    FinalSelector,
    SelectionResult,
)
from retrieval.router import Router, RoutingDecision
from retrieval.rrf import rrf_fuse

__all__ = [
    "RetrievalConfig",
    "RetrievalResult",
    "RetrievalPipeline",
    "merge_candidate_pools",
]

# Injected callable types ---------------------------------------------------- #
# Lexical/dense search: (query, top_k) -> ranked list of hit dicts (best first).
# Each hit MUST carry: row_idx, law_id, ten_van_ban, dieu_so. May carry chunk_id.
SearchFn = Callable[[str, int], List[Dict[str, Any]]]
# Text fetch: row_idxs -> {row_idx: chunk_text}.
TextFn = Callable[[Sequence[int]], Dict[int, str]]
# Rerank: (query, passages) -> one score per passage (higher = better).
RerankFn = Callable[[str, Sequence[str]], List[float]]

# Depth -> (top_bm25, top_dense, fused_pool) multipliers applied to the base
# config. Wider pools for complex / multi-hop lanes (recall-first).
_DEPTH_SCALE: Dict[str, float] = {
    "narrow": 0.6,
    "normal": 1.0,
    "wide": 1.5,
}
# Rerank strength -> rerank input pool size multiplier.
_RERANK_SCALE: Dict[str, float] = {
    "light": 0.6,
    "normal": 1.0,
    "strong": 1.3,
}


@dataclass
class RetrievalConfig:
    """Top-level knobs for the 7-stage pipeline (``retrieval.md`` §Policy)."""

    # Stage 3: Hybrid Retrieve (base values; scaled by retrieval_depth).
    top_bm25: int = 80
    top_dense: int = 80
    rrf_k: int = 60
    fused_pool_size: int = 120
    use_dense: bool = True

    # Stage 4: Rerank (base value; scaled by rerank_strength).
    rerank_input_size: int = 70
    rerank_text_truncate: int = 512  # chars per passage handed to reranker

    # Stage 5: Graph Expand.
    graph_expand_seeds: int = 8
    graph_expanded_top: int = 60

    # Stage 6: Final Select budgets come from the lane (see final_selector).


@dataclass
class RetrievalResult:
    """The pipeline's output for one query."""

    query: str
    routing: RoutingDecision
    decomposition: Optional[DecompositionResult]
    selection: SelectionResult
    stage_stats: Dict[str, Any] = field(default_factory=dict)

    # Convenience pass-throughs for the grader / generator.
    def relevant_articles(self) -> List[str]:
        return self.selection.relevant_articles()

    def relevant_docs(self) -> List[str]:
        return self.selection.relevant_docs()


def merge_candidate_pools(
    pools: Sequence[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Merge per-query hit pools, deduplicating by ``row_idx``.

    Keeps the FIRST occurrence's payload (original-query pool is passed first,
    so its metadata wins). A row's retained ordering is by best (smallest)
    rank across pools — but since the reranker re-scores everything on the
    original query downstream, the merge only needs to produce the *union* of
    candidates without losing any. Dedup is by ``row_idx``.
    """
    seen: Dict[int, Dict[str, Any]] = {}
    for pool in pools:
        for hit in pool:
            ridx = int(hit["row_idx"])
            if ridx not in seen:
                seen[ridx] = dict(hit)
    return list(seen.values())


class RetrievalPipeline:
    """Compose router, decomposer, hybrid retrieve, rerank, graph expand, select.

    Parameters
    ----------
    router:
        A built :class:`retrieval.router.Router`.
    decomposer:
        A built :class:`retrieval.decomposer.Decomposer`. Only invoked when the
        router sets ``should_decompose=True``.
    lexical_search:
        ``(query, top_k) -> [hit dict]`` (FTS5 BM25). Always required.
    final_selector:
        A built :class:`retrieval.final_selector.FinalSelector`.
    dense_search:
        Optional ``(query, top_k) -> [hit dict]`` (FAISS). Skipped when
        ``cfg.use_dense=False`` or ``None``.
    reranker:
        Optional ``(query, passages) -> [score]`` (cross-encoder). When
        ``None``, the fused RRF order is kept (no reranking).
    text_provider:
        Optional ``(row_idxs) -> {row_idx: text}`` to fetch chunk text for the
        reranker + the generation context. Required for reranking.
    graph_expander:
        Optional object with ``expand([(row_idx, score)], top_n) -> [dict]``
        (a :class:`retrieval.graph_expand.GraphExpander`). Invoked only when the
        router sets ``need_graph_expand=True``.
    cfg:
        :class:`RetrievalConfig`.
    """

    def __init__(
        self,
        router: Router,
        decomposer: Decomposer,
        lexical_search: SearchFn,
        final_selector: FinalSelector,
        dense_search: Optional[SearchFn] = None,
        reranker: Optional[RerankFn] = None,
        text_provider: Optional[TextFn] = None,
        graph_expander: Any = None,
        cfg: Optional[RetrievalConfig] = None,
    ) -> None:
        self.router = router
        self.decomposer = decomposer
        self.lexical_search = lexical_search
        self.dense_search = dense_search
        self.reranker = reranker
        self.text_provider = text_provider
        self.graph_expander = graph_expander
        self.final_selector = final_selector
        self.cfg = cfg or RetrievalConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str) -> RetrievalResult:
        """Run the full 7-stage pipeline for ``query``."""
        stats: Dict[str, Any] = {}
        q = (query or "").strip()

        # --- Stage 1: Router ------------------------------------------------ #
        routing = self.router.route(q)
        stats["lane"] = routing.lane
        stats["routing"] = routing.to_dict()

        # --- Stage 2: Decompose (conditional) ------------------------------ #
        decomposition: Optional[DecompositionResult] = None
        queries: List[str] = [q]
        if routing.should_decompose:
            decomposition = self.decomposer.decompose(q)
            if decomposition.should_decompose and decomposition.sub_questions:
                # Retrieve with [original] + sub-questions (original always kept).
                queries = [q] + [sq.text for sq in decomposition.sub_questions]
        stats["n_retrieval_queries"] = len(queries)

        # --- Stage 3: Hybrid retrieve (per query, merged) ------------------ #
        t0 = time.perf_counter()
        merged = self._hybrid_retrieve(queries, routing.retrieval_depth)
        stats["n_merged_candidates"] = len(merged)
        stats["ms_retrieve"] = round((time.perf_counter() - t0) * 1000, 1)

        # --- Stage 4: Rerank on the ORIGINAL query ------------------------- #
        t0 = time.perf_counter()
        reranked = self._rerank(q, merged, routing.rerank_strength)
        stats["n_reranked"] = len(reranked)
        stats["ms_rerank"] = round((time.perf_counter() - t0) * 1000, 1)

        # --- Stage 5: Graph expand (conditional) --------------------------- #
        t0 = time.perf_counter()
        expanded = self._graph_expand(reranked, routing.need_graph_expand)
        stats["n_expanded"] = len(expanded)
        stats["graph_expanded"] = bool(
            routing.need_graph_expand and self.graph_expander is not None
        )
        stats["ms_graph_expand"] = round((time.perf_counter() - t0) * 1000, 1)

        # --- Stage 6: Final select (article-first, lane-budgeted) ---------- #
        evidence = self._to_chunk_evidence(expanded)
        selection = self.final_selector.select(evidence, lane=routing.lane)
        stats["n_selected_articles"] = len(selection.articles)
        stats["n_selected_chunks"] = len(selection.chunks)

        return RetrievalResult(
            query=q,
            routing=routing,
            decomposition=decomposition,
            selection=selection,
            stage_stats=stats,
        )

    # ------------------------------------------------------------------ #
    # Stage 3: hybrid retrieve
    # ------------------------------------------------------------------ #
    def _hybrid_retrieve(
        self, queries: Sequence[str], depth: str
    ) -> List[Dict[str, Any]]:
        """Retrieve + RRF-fuse per query, then merge pools across queries.

        The original query's pool is fused first so its payload wins on dedup.
        """
        scale = _DEPTH_SCALE.get(depth, 1.0)
        top_bm25 = max(1, int(self.cfg.top_bm25 * scale))
        top_dense = max(1, int(self.cfg.top_dense * scale))
        fused_top = max(1, int(self.cfg.fused_pool_size * scale))

        per_query_pools: List[List[Dict[str, Any]]] = []
        for q in queries:
            lexical = self.lexical_search(q, top_bm25) or []
            rankings: List[List[Dict[str, Any]]] = [lexical]
            if self.cfg.use_dense and self.dense_search is not None:
                dense = self.dense_search(q, top_dense) or []
                rankings.append(dense)
            fused = rrf_fuse(rankings, k=self.cfg.rrf_k, fused_top=fused_top)
            # rrf_fuse returns {row_idx, rrf_score, source, payload} where
            # payload is the original hit dict. Flatten payload to top level so
            # downstream stages can read law_id / dieu_so / chunk_id / text.
            flattened = []
            for hit in fused:
                payload = hit.get("payload") or {}
                if isinstance(payload, dict):
                    flat = dict(payload)
                    flat["row_idx"] = hit["row_idx"]
                    flat["rrf_score"] = hit["rrf_score"]
                    flat.setdefault("score", hit["rrf_score"])
                    flattened.append(flat)
                else:
                    # Int row_idx only (no payload) → minimal dict.
                    flattened.append({
                        "row_idx": hit["row_idx"],
                        "rrf_score": hit["rrf_score"],
                        "score": hit["rrf_score"],
                    })
            per_query_pools.append(flattened)

        return merge_candidate_pools(per_query_pools)

    # ------------------------------------------------------------------ #
    # Stage 4: rerank on the original query
    # ------------------------------------------------------------------ #
    def _rerank(
        self, original_query: str, candidates: List[Dict[str, Any]], strength: str
    ) -> List[Dict[str, Any]]:
        """Cross-encoder rerank using the ORIGINAL query (the critical rule).

        When no reranker / text_provider is wired, returns the candidates in
        their fused (rrf_score) order, capped to the rerank pool size, with a
        ``score`` field copied from ``rrf_score`` so downstream stages have a
        uniform score key.
        """
        scale = _RERANK_SCALE.get(strength, 1.0)
        pool_size = max(1, int(self.cfg.rerank_input_size * scale))

        # Order the merged pool by fused score, keep the rerank pool.
        candidates = sorted(
            candidates,
            key=lambda h: float(h.get("rrf_score", h.get("score", 0.0))),
            reverse=True,
        )[:pool_size]

        if self.reranker is None or self.text_provider is None:
            for h in candidates:
                h["score"] = float(h.get("rrf_score", h.get("score", 0.0)))
            return candidates

        # Fetch text for the rerank pool.
        row_idxs = [int(h["row_idx"]) for h in candidates]
        texts = self.text_provider(row_idxs) or {}
        passages = [
            (texts.get(int(h["row_idx"]), "") or "")[: self.cfg.rerank_text_truncate]
            for h in candidates
        ]
        scores = self.reranker(original_query, passages) or []

        for h, s, txt in zip(candidates, scores, passages):
            h["score"] = float(s)
            if txt:
                h.setdefault("text", txt)
        candidates.sort(key=lambda h: h["score"], reverse=True)
        return candidates

    # ------------------------------------------------------------------ #
    # Stage 5: graph expand
    # ------------------------------------------------------------------ #
    def _graph_expand(
        self, reranked: List[Dict[str, Any]], need_expand: bool
    ) -> List[Dict[str, Any]]:
        """Expand via the KG when the lane needs multi-hop coverage.

        Seeds = top reranked chunks. The expander returns dicts
        ``{row_idx, score, source}`` where ``source`` marks graph-derived rows.
        We re-attach the original payload (law_id / ten_van_ban / dieu_so) for
        seed rows and flag expanded rows so final select can award graph_gain.
        """
        if not need_expand or self.graph_expander is None or not reranked:
            # No expansion: tag everything as a seed.
            for h in reranked:
                h["from_graph"] = False
            return reranked

        payload_by_row: Dict[int, Dict[str, Any]] = {
            int(h["row_idx"]): h for h in reranked
        }
        seeds: List[Tuple[int, float]] = [
            (int(h["row_idx"]), float(h.get("score", 0.0)))
            for h in reranked[: self.cfg.graph_expand_seeds]
        ]

        expanded = self.graph_expander.expand(
            seeds, top_n=self.cfg.graph_expanded_top
        )

        out: List[Dict[str, Any]] = []
        # Always retain the full reranked pool (seeds) at their rerank score.
        for h in reranked:
            h["from_graph"] = False
            out.append(h)
        seen = {int(h["row_idx"]) for h in out}

        for e in expanded:
            ridx = int(e["row_idx"])
            if ridx in seen:
                continue
            base = payload_by_row.get(ridx, {})
            row = dict(base)
            row["row_idx"] = ridx
            row["score"] = float(e.get("score", 0.0))
            row["from_graph"] = e.get("source", "candidate") != "candidate"
            out.append(row)
            seen.add(ridx)
        return out

    # ------------------------------------------------------------------ #
    # Stage 6 prep: dict pool -> ChunkEvidence
    # ------------------------------------------------------------------ #
    def _to_chunk_evidence(
        self, pool: List[Dict[str, Any]]
    ) -> List[ChunkEvidence]:
        """Convert the post-expansion dict pool into ``ChunkEvidence``.

        Rows missing article identity (law_id / dieu_so) are dropped — they
        cannot be grouped into an article and would be invisible to the grader.
        """
        evidence: List[ChunkEvidence] = []
        for h in pool:
            law_id = h.get("law_id")
            dieu_so = h.get("dieu_so")
            if not law_id or not dieu_so:
                continue
            evidence.append(
                ChunkEvidence(
                    row_idx=int(h["row_idx"]),
                    law_id=str(law_id),
                    ten_van_ban=str(h.get("ten_van_ban", "")),
                    dieu_so=str(dieu_so),
                    score=float(h.get("score", 0.0)),
                    chunk_id=h.get("chunk_id"),
                    text=h.get("text"),
                    from_graph=bool(h.get("from_graph", False)),
                )
            )
        return evidence
