"""Hybrid retriever for G-LRAG (PLAN.md task 7.1 — Retrieval Module).

Pipeline (online inference, per query)::

    query
      ├── lexical leg  (FTS5, top_bm25)      ─┐
      ├── dense leg    (FAISS | Qdrant, top_dense) ├→ RRF fuse (fused_top)
      │                                            │
      └────────────────────────────────────────────┴→ graph expand (expanded_top)
                                                      │
                                                cross-encoder rerank (Stage 7.4)
                                                      │
                                                  final top-K

Design constraints (from PLAN.md §7 + artifacts_guide.md)
----------------------------------------------------------
* **Lazy heavy imports.** ``faiss`` / ``FlagEmbedding`` / ``torch`` /
  ``qdrant_client`` are imported only inside the methods that need them, so
  this module imports & unit-tests on a CPU-only machine with none of those
  installed. Setting ``use_dense=False, use_reranker=False`` yields a pure
  lexical + graph-expansion CPU baseline (the 7.5 acceptance path).
* ``use_dense`` / ``use_reranker`` flags make a CPU lexical-only baseline
  possible without changing call sites.
* The dense backend is pluggable: pass a :class:`retrieval.faiss_index.
  FAISSIndex` **or** a :class:`retrieval.qdrant_index.QdrantIndex` — both
  share the ``search(query_vec, top_k)`` contract. A query encoder is only
  required when a dense leg is enabled.
* ``row_idx`` is the single join key across the whole pipeline (Stage 6
  bundle invariant), so RRF, graph expansion and ``make_relevant_lists``
  never need to translate ids.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retrieval.rrf import rrf_fuse
from retrieval.graph_expand import GraphExpander
from retrieval.debug import (
    RetrievalTrace,
    StageSnapshot,
    snapshot_items,
    format_trace as _format_trace,
)

__all__ = [
    "HybridRetriever",
    "RetrievalConfig",
    "Hit",
    "make_relevant_lists",
    "RetrievalTrace",
    "StageSnapshot",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_flag_reranker(model_name: str, use_fp16: bool, device: Optional[str]):
    """Construct a FlagReranker pinned to a single ``device`` (e.g. "cuda:0").

    Why pin: with >1 visible GPU and no device given, FlagEmbedding spawns
    multi-GPU worker processes on every ``compute_score`` call (re-initialising
    CUDA per child). For the handful of (query, passage) pairs reranked per
    sub-query, that spawn/IPC overhead dwarfs the actual compute and the GPU
    shows ~0% util while wall-clock explodes (the 580s single-query symptom).
    Pinning to one device keeps inference in-process.

    The device kwarg name differs across versions: 1.2.x uses ``device=``
    (single str), 1.3.x uses ``devices=``. Try the modern name, fall back to
    the legacy one, then fall back to no-device so an unexpected signature
    never hard-fails the load.
    """
    from FlagEmbedding import FlagReranker  # lazy

    if device is None:
        return FlagReranker(model_name, use_fp16=use_fp16)
    try:
        return FlagReranker(model_name, use_fp16=use_fp16, devices=device)
    except TypeError:
        pass
    try:
        return FlagReranker(model_name, use_fp16=use_fp16, device=device)
    except TypeError:
        return FlagReranker(model_name, use_fp16=use_fp16)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievalConfig:
    """Knobs for :class:`HybridRetriever` (defaults mirror config/default.yaml)."""

    use_dense: bool = False
    use_reranker: bool = False
    top_bm25: int = 50
    top_dense: int = 50
    rrf_k: int = 60
    fused_top: int = 30
    expanded_top: int = 50
    final_top_k: int = 5
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # Pin the cross-encoder reranker to ONE device. None => let FlagEmbedding
    # decide (its multi-GPU spawn path, slow with >1 visible GPU). Default to
    # cuda:0 so the reranker stays on the retrieval card, in-process.
    rerank_device: Optional[str] = "cuda:0"
    rerank_max_input_chars: int = 2000
    seed: int = 42

    # ---- Debug / observability (Stage 7 add-on) -------------------------
    # When True, retrieve() records a per-stage RetrievalTrace on
    # ``self.last_trace`` and per-stage wall-clock timings. Zero overhead
    # when False — the trace is never constructed and no snapshot dicts are
    # allocated on the hot path (PLAN 7.5 acceptance path is untouched).
    debug: bool = False
    # How many top items to keep per stage in the trace's ``top_items``.
    debug_top_n: int = 8
    # When True and debug is on, also print the formatted trace to stdout at
    # the end of retrieve() (handy for notebooks / one-off debugging).
    debug_print: bool = False


@dataclass
class Hit:
    """A single final retrieval hit, carrying the metadata needed for output.

    ``origin_subquery`` / ``aspect_id`` are populated by the decomposition
    layer (:class:`DecomposingHybridRetriever` in the colab/kaggle notebooks)
    when a query is split into per-aspect sub-queries. They let the downstream
    article-selection stage reason about aspect coverage directly from the
    fused Hit list, instead of re-deriving it from per-sub-query traces.

    - ``origin_subquery``: 0-based index of the sub-query that surfaced this
      Hit (``-1`` / unset when the hit did not come from a decomposed leg).
    - ``aspect_id``: the facet id this Hit was tagged with by the fusion layer
      (empty when no facet mapping is available).
    """

    row_idx: int
    score: float
    source: str
    law_id: str = ""
    ten_van_ban: str = ""
    dieu_so: str = ""
    chunk_id: str = ""
    doc_uid: str = ""
    chunk_text: str = ""
    origin_subquery: int = -1
    aspect_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_idx": self.row_idx,
            "score": self.score,
            "source": self.source,
            "law_id": self.law_id,
            "ten_van_ban": self.ten_van_ban,
            "dieu_so": self.dieu_so,
            "chunk_id": self.chunk_id,
            "doc_uid": self.doc_uid,
            "origin_subquery": self.origin_subquery,
            "aspect_id": self.aspect_id,
        }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def make_relevant_lists(hits: Sequence[Hit]) -> Tuple[List[str], List[str]]:
    """Collapse top-K hits into submission ``relevant_docs`` / ``relevant_articles``.

    - ``relevant_docs``: ``"{law_id}|{ten_van_ban}"`` for every hit that has a
      non-empty ``law_id`` and ``ten_van_ban`` (deduped, order preserved).
    - ``relevant_articles``: ``"{law_id}|{ten_van_ban}|{dieu_so}"`` only for
      hits whose ``dieu_so`` starts with ``"Điều"`` (deduped, order preserved).

    This mirrors ``artifacts_guide.md`` §"Tạo submission metadata" and the
    grader's ``Điều X`` normalisation in ``dev_set/eval.py``.
    """
    docs: List[str] = []
    articles: List[str] = []
    seen_docs: set = set()
    seen_articles: set = set()
    for h in hits:
        law_id = str(getattr(h, "law_id", "") or "").strip()
        ten = str(getattr(h, "ten_van_ban", "") or "").strip()
        dieu = str(getattr(h, "dieu_so", "") or "").strip()
        if law_id and ten:
            d = f"{law_id}|{ten}"
            if d not in seen_docs:
                seen_docs.add(d)
                docs.append(d)
        if law_id and ten and dieu and dieu.startswith("Điều"):
            a = f"{law_id}|{ten}|{dieu}"
            if a not in seen_articles:
                seen_articles.add(a)
                articles.append(a)
    return docs, articles


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Parallel lexical + dense retrieval with RRF, graph expansion & rerank.

    Parameters
    ----------
    fts:
        An opened :class:`retrieval.bm25_index.FTSIndex` (lexical leg).
    faiss_index:
        A dense backend exposing ``search(query_vec, top_k) -> list[dict]``.
        Either a :class:`retrieval.faiss_index.FAISSIndex` or a
        :class:`retrieval.qdrant_index.QdrantIndex`. ``None`` disables the
        dense leg (CPU lexical-only baseline).
    graph_expander:
        Optional :class:`retrieval.graph_expand.GraphExpander`. ``None``
        disables graph expansion (the 7.3 no-graph ablation).
    query_encoder:
        Optional callable ``query_str -> (1, dim) float32`` array (e.g.
        :class:`retrieval.faiss_index.BGEQueryEncoder`). Required only when
        ``config.use_dense`` is True.
    config:
        :class:`RetrievalConfig`.
    """

    def __init__(
        self,
        fts: Any,
        faiss_index: Optional[Any] = None,
        graph_expander: Optional[GraphExpander] = None,
        query_encoder: Optional[Any] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self.fts = fts
        self.faiss_index = faiss_index
        self.graph_expander = graph_expander
        self.query_encoder = query_encoder
        self.config = config or RetrievalConfig()
        # Cross-encoder reranker is lazily loaded ONCE and reused across
        # queries. Re-instantiating FlagReranker inside _rerank() on every
        # retrieve() call reloads ~568M weights each time — slow and a
        # repeated host-RAM spike that can OOM a 12 GB Colab instance
        # (especially the 20-question dev batch in the notebook). See _rerank().
        self._reranker: Any = None
        self._reranker_unavailable: bool = False
        # Debug trace from the most recent retrieve() call. ``None`` until
        # a retrieve() runs with config.debug=True (or debug_retrieve()).
        self.last_trace: Optional[RetrievalTrace] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        fetch_text: bool = False,
    ) -> List[Hit]:
        """Run the full hybrid pipeline and return the final top-K :class:`Hit`.

        When ``use_dense`` is False the dense leg is skipped (CPU baseline).
        When ``use_reranker`` is False the cross-encoder step is skipped and
        the post-expansion RRF scores are used directly.

        When ``config.debug`` is True, a :class:`RetrievalTrace` capturing one
        :class:`StageSnapshot` per stage (counts, top-N items with legal
        metadata, per-stage timing, skip reasons, fetch-stage missing rows)
        is stored on ``self.last_trace``. The trace has **zero overhead** when
        debug is False — it is never constructed on that path.
        """
        cfg = self.config
        trace: Optional[RetrievalTrace] = None
        t_start: Optional[float] = None
        if cfg.debug:
            trace = RetrievalTrace(
                query=query,
                config={
                    "use_dense": cfg.use_dense,
                    "use_reranker": cfg.use_reranker,
                    "graph_expander": self.graph_expander is not None,
                    "top_bm25": cfg.top_bm25,
                    "top_dense": cfg.top_dense,
                    "fused_top": cfg.fused_top,
                    "expanded_top": cfg.expanded_top,
                    "final_top_k": cfg.final_top_k,
                    "rrf_k": cfg.rrf_k,
                    "seed": cfg.seed,
                },
            )
            t_start = time.perf_counter()

        # ---- 1. lexical leg ------------------------------------------------
        if trace is not None:
            t0 = time.perf_counter()
            lexical_hits = self._lexical_leg(query, cfg.top_bm25)
            ms = (time.perf_counter() - t0) * 1000.0
            trace.add(StageSnapshot(
                name="lexical",
                count=len(lexical_hits),
                top_items=snapshot_items(
                    lexical_hits, cfg.debug_top_n,
                    score_key="bm25_score",
                    extra_keys=("law_id", "ten_van_ban", "dieu_so"),
                ),
                elapsed_ms=ms,
                skip=None if lexical_hits else "no lexical hits",
            ))
        else:
            lexical_hits = self._lexical_leg(query, cfg.top_bm25)

        # ---- 2. dense leg (optional) --------------------------------------
        dense_hits: List[Dict[str, Any]] = []
        if trace is not None:
            t0 = time.perf_counter()
            if cfg.use_dense and self.faiss_index is not None:
                dense_hits = self._dense_leg(query, cfg.top_dense)
            ms = (time.perf_counter() - t0) * 1000.0
            skip = None
            if not (cfg.use_dense and self.faiss_index is not None):
                skip = ("use_dense=False" if not cfg.use_dense
                        else "no faiss_index")
            trace.add(StageSnapshot(
                name="dense",
                count=len(dense_hits),
                top_items=snapshot_items(
                    dense_hits, cfg.debug_top_n,
                    score_key="dense_score",
                    extra_keys=("law_id", "ten_van_ban", "dieu_so"),
                ),
                elapsed_ms=ms,
                skip=skip,
            ))
        elif cfg.use_dense and self.faiss_index is not None:
            dense_hits = self._dense_leg(query, cfg.top_dense)

        # ---- 3. RRF fusion -------------------------------------------------
        rankings: List[List[Any]] = []
        if lexical_hits:
            rankings.append(lexical_hits)
        if dense_hits:
            rankings.append(dense_hits)
        if trace is not None:
            t0 = time.perf_counter()
            fused = rrf_fuse(
                rankings, k=cfg.rrf_k, fused_top=cfg.fused_top, seed=cfg.seed
            ) if rankings else []
            ms = (time.perf_counter() - t0) * 1000.0
            # Backfill rrf_score into each fused item for uniform printing.
            fused_items = [
                {**f, "score": f["rrf_score"],
                 **(f.get("payload") or {})}
                for f in fused
            ]
            trace.add(StageSnapshot(
                name="rrf",
                count=len(fused),
                top_items=snapshot_items(
                    fused_items, cfg.debug_top_n,
                    score_key="rrf_score",
                    extra_keys=("law_id", "ten_van_ban", "dieu_so"),
                ),
                elapsed_ms=ms,
                skip="no rankings to fuse" if not rankings else None,
                diagnostics={"rankings_in": len(rankings),
                             "k": cfg.rrf_k, "fused_top": cfg.fused_top},
            ))
        else:
            fused = rrf_fuse(
                rankings, k=cfg.rrf_k, fused_top=cfg.fused_top, seed=cfg.seed
            ) if rankings else []

        # ---- 4. graph expansion (optional) --------------------------------
        if trace is not None:
            t0 = time.perf_counter()
            if self.graph_expander is not None and fused:
                candidates = [(r["row_idx"], r["rrf_score"]) for r in fused]
                expanded = self.graph_expander.expand(
                    candidates, top_n=cfg.expanded_top
                )
                expanded_rows = {e["row_idx"]: e for e in expanded}
            else:
                expanded = []
                expanded_rows = {
                    r["row_idx"]: {
                        "row_idx": r["row_idx"],
                        "score": r["rrf_score"],
                        "source": "fused",
                    }
                    for r in fused
                }
            ms = (time.perf_counter() - t0) * 1000.0
            skip = None
            if self.graph_expander is None:
                skip = "no graph_expander"
            elif not fused:
                skip = "no fused seeds to expand"
            # Source distribution is the single most useful graph diagnostic.
            src_counts: Dict[str, int] = {}
            for e in expanded_rows.values():
                src_counts[str(e.get("source", "?"))] = \
                    src_counts.get(str(e.get("source", "?")), 0) + 1
            trace.add(StageSnapshot(
                name="graph",
                count=len(expanded_rows),
                top_items=snapshot_items(
                    list(expanded_rows.values()), cfg.debug_top_n,
                    score_key="score",
                    extra_keys=("source", "law_id", "ten_van_ban", "dieu_so"),
                ),
                elapsed_ms=ms,
                skip=skip,
                diagnostics={"expanded_top": cfg.expanded_top,
                             "source_counts": src_counts},
            ))
        elif self.graph_expander is not None and fused:
            candidates = [(r["row_idx"], r["rrf_score"]) for r in fused]
            expanded = self.graph_expander.expand(
                candidates, top_n=cfg.expanded_top
            )
            expanded_rows = {e["row_idx"]: e for e in expanded}
        else:
            expanded_rows = {
                r["row_idx"]: {
                    "row_idx": r["row_idx"],
                    "score": r["rrf_score"],
                    "source": "fused",
                }
                for r in fused
            }

        ordered = sorted(
            expanded_rows.values(),
            key=lambda e: (-e["score"], e["row_idx"]),
        )

        # ---- 5. fetch metadata + optional text ----------------------------
        row_idxs = [e["row_idx"] for e in ordered]
        if trace is not None:
            t0 = time.perf_counter()
            meta_by_row = self._fetch_meta(row_idxs)
            text_by_row: Dict[int, str] = {}
            if fetch_text:
                text_by_row = self._fetch_text(row_idxs)
            ms = (time.perf_counter() - t0) * 1000.0
            missing = [ri for ri in row_idxs if ri not in meta_by_row]
            ordered_with_meta = [
                {**e, **meta_by_row.get(e["row_idx"], {})}
                for e in ordered
            ]
            trace.add(StageSnapshot(
                name="fetch",
                count=len(meta_by_row),
                top_items=snapshot_items(
                    ordered_with_meta, cfg.debug_top_n,
                    score_key="score",
                    extra_keys=("source", "law_id", "ten_van_ban",
                                "dieu_so", "chunk_id", "doc_uid"),
                ),
                elapsed_ms=ms,
                skip="no candidates to fetch" if not row_idxs else None,
                diagnostics={
                    "requested": len(row_idxs),
                    "resolved": len(meta_by_row),
                    "missing_rows": missing,
                    "text_fetched": bool(fetch_text),
                },
            ))
        else:
            meta_by_row = self._fetch_meta(row_idxs)
            text_by_row: Dict[int, str] = {}
            if fetch_text:
                text_by_row = self._fetch_text(row_idxs)

        # ---- 6. rerank (optional) -----------------------------------------
        pre_rerank_count = len(ordered)
        if cfg.use_reranker and cfg.final_top_k > 0 and ordered:
            if trace is not None:
                t0 = time.perf_counter()
                was_unavailable = self._reranker_unavailable
                ordered = self._rerank(query, ordered, meta_by_row, text_by_row)
                ms = (time.perf_counter() - t0) * 1000.0
                skip = None
                if self._reranker_unavailable:
                    skip = ("reranker deps unavailable (torch/FlagEmbedding) "
                            "→ kept pre-rerank order")
                trace.add(StageSnapshot(
                    name="rerank",
                    count=len(ordered),
                    top_items=snapshot_items(
                        ordered, cfg.debug_top_n,
                        score_key="score",
                        extra_keys=("rerank_score", "source", "law_id",
                                    "ten_van_ban", "dieu_so"),
                    ),
                    elapsed_ms=ms,
                    skip=skip,
                    diagnostics={
                        "candidates_in": pre_rerank_count,
                        "model": cfg.rerank_model,
                        "deps_skipped_before": was_unavailable,
                    },
                ))
            else:
                ordered = self._rerank(query, ordered, meta_by_row, text_by_row)
        elif trace is not None:
            trace.add(StageSnapshot(
                name="rerank",
                count=pre_rerank_count,
                top_items=snapshot_items(
                    ordered, cfg.debug_top_n,
                    score_key="score",
                    extra_keys=("source", "law_id", "ten_van_ban", "dieu_so"),
                ),
                elapsed_ms=0.0,
                skip=("use_reranker=False" if not cfg.use_reranker
                      else "final_top_k<=0" if cfg.final_top_k <= 0
                      else "no candidates to rerank"),
            ))

        # ---- 7. build final Hit list --------------------------------------
        hits: List[Hit] = []
        for e in ordered[: cfg.final_top_k]:
            row_idx = e["row_idx"]
            meta = meta_by_row.get(row_idx, {})
            hits.append(
                Hit(
                    row_idx=row_idx,
                    score=float(e["score"]),
                    source=str(e.get("source", "fused")),
                    law_id=str(meta.get("law_id", "")),
                    ten_van_ban=str(meta.get("ten_van_ban", "")),
                    dieu_so=str(meta.get("dieu_so", "")),
                    chunk_id=str(meta.get("chunk_id", "")),
                    doc_uid=str(meta.get("doc_uid", "")),
                    chunk_text=text_by_row.get(row_idx, ""),
                )
            )

        if trace is not None:
            trace.add(StageSnapshot(
                name="final",
                count=len(hits),
                top_items=[
                    {
                        "row_idx": h.row_idx,
                        "score": h.score,
                        "source": h.source,
                        "law_id": h.law_id,
                        "ten_van_ban": h.ten_van_ban,
                        "dieu_so": h.dieu_so,
                        "chunk_id": h.chunk_id,
                    }
                    for h in hits[: cfg.debug_top_n]
                ],
                elapsed_ms=0.0,
                diagnostics={"final_top_k": cfg.final_top_k},
            ))
            # Post-step: the submission metadata collapse.
            docs, articles = make_relevant_lists(hits)
            trace.add(StageSnapshot(
                name="output",
                count=len(hits),
                elapsed_ms=0.0,
                diagnostics={
                    "relevant_docs": docs,
                    "relevant_articles": articles,
                },
            ))
            if t_start is not None:
                trace.total_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self.last_trace = trace
            if cfg.debug_print:
                print(_format_trace(trace, top_n=cfg.debug_top_n))

        return hits

    @property
    def last_trace_formatted(self) -> Optional[str]:
        """Render :attr:`last_trace` as a readable string, or ``None``."""
        if self.last_trace is None:
            return None
        return _format_trace(self.last_trace, top_n=self.config.debug_top_n)

    def debug_retrieve(
        self,
        query: str,
        fetch_text: bool = False,
        print_trace: bool = True,
    ) -> List[Hit]:
        """Run ``retrieve()`` with debug tracing forced on for this one call.

        Temporarily flips ``config.debug`` (and optionally ``debug_print``) on,
        runs the pipeline, and returns the hits — leaving a populated
        :attr:`last_trace`. The config flags are restored to their previous
        values afterwards, so a one-off debug call doesn't permanently change
        behaviour for subsequent normal ``retrieve()`` calls.

        Parameters
        ----------
        query, fetch_text:
            Forwarded to :meth:`retrieve`.
        print_trace:
            If True (default), print the formatted trace to stdout after the
            run — handy in a notebook. Set False to inspect
            ``self.last_trace`` / :attr:`last_trace_formatted` quietly.
        """
        prev_debug = self.config.debug
        prev_print = self.config.debug_print
        self.config.debug = True
        self.config.debug_print = bool(print_trace)
        try:
            return self.retrieve(query, fetch_text=fetch_text)
        finally:
            self.config.debug = prev_debug
            self.config.debug_print = prev_print

    # ------------------------------------------------------------------ #
    # Legs
    # ------------------------------------------------------------------ #
    def _lexical_leg(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """FTS5 lexical leg → list of metadata dicts (with ``row_idx``)."""
        if self.fts is None:
            return []
        return self.fts.search(query, top_k=top_k)

    def _dense_leg(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Dense leg via the pluggable backend (FAISS or Qdrant)."""
        if self.faiss_index is None:
            return []
        if self.query_encoder is None:
            raise RuntimeError(
                "HybridRetriever: use_dense=True but no query_encoder was "
                "provided. Pass a BGEQueryEncoder (or any callable "
                "query->vector)."
            )
        qvec = self.query_encoder.encode(query)
        return self.faiss_index.search(qvec, top_k=top_k)

    # ------------------------------------------------------------------ #
    # Metadata / text fetch
    # ------------------------------------------------------------------ #
    def _fetch_meta(self, row_idxs: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch chunk metadata for the given row indices from the lexical FTS."""
        if not row_idxs or self.fts is None:
            return {}
        rows = self.fts.fetch_chunks(list(row_idxs))
        out: Dict[int, Dict[str, Any]] = {}
        for r in rows:
            ri = int(r["row_idx"])
            out[ri] = {
                "row_idx": ri,
                "chunk_id": str(r.get("chunk_id", "")),
                "doc_uid": str(r.get("doc_uid", "")),
                "law_id": str(r.get("law_id", "")),
                "ten_van_ban": str(r.get("ten_van_ban", "")),
                "dieu_so": str(r.get("dieu_so", "")),
            }
        # If the dense backend carried metadata that FTS didn't return (e.g.
        # Qdrant payloads), backfill from the fused/expanded dicts is not
        # needed because fetch_chunks over the canonical row_idx always hits
        # the same chunks table.
        return out

    def _fetch_text(self, row_idxs: Sequence[int]) -> Dict[int, str]:
        if not row_idxs or self.fts is None:
            return {}
        rows = self.fts.fetch_chunks(list(row_idxs))
        return {int(r["row_idx"]): str(r.get("chunk_text", "")) for r in rows}

    # ------------------------------------------------------------------ #
    # Cross-encoder rerank (Stage 7.4)
    # ------------------------------------------------------------------ #
    def _rerank(
        self,
        query: str,
        ordered: List[Dict[str, Any]],
        meta_by_row: Dict[int, Dict[str, Any]],
        text_by_row: Dict[int, str],
    ) -> List[Dict[str, Any]]:
        """Cross-encoder rerank via ``BAAI/bge-reranker-v2-m3`` (fp16).

        Heavy deps are imported lazily. The original score is kept as a
        fallback tiebreaker behind the reranker score.
        """
        if not ordered:
            return ordered
        # If a previous call found the reranker deps unavailable, skip without
        # re-attempting the import on every query (avoids repeated try/except).
        if self._reranker_unavailable:
            return ordered
        try:
            import torch  # noqa: F401
            from FlagEmbedding import FlagReranker  # lazy
        except Exception:
            # No GPU / deps → skip rerank gracefully (CPU baseline path).
            self._reranker_unavailable = True
            return ordered

        cfg = self.config
        # Load the reranker once and cache it on the retriever. Re-instantiating
        # per query re-downloads/reloads the weights and spikes host RAM (OOM
        # risk on a 12 GB Colab instance running the 20-question dev batch).
        # Cached reuse keeps identical outputs (same model, fp16, compute_score).
        #
        # The load can raise OSError/EnvironmentError when the HF cache is
        # partial/corrupt (a local snapshot dir exists but is missing the weight
        # file — e.g. an interrupted download, or "Not enough free disk space"
        # mid-download) or the hub is unreachable. Self-heal: clear the partial
        # cache snapshot, re-download once via snapshot_download, then retry the
        # load. Only if the retry also fails do we degrade to pre-rerank order —
        # same contract as the import-unavailable path, but the reranker is NOT
        # silently dropped on a transient partial-cache.
        if self._reranker is None:
            try:
                self._reranker = _load_flag_reranker(
                    cfg.rerank_model, use_fp16=True, device=cfg.rerank_device)
            except Exception as e:
                print(f"[rerank] first load of {cfg.rerank_model} failed: {e!r}; "
                      f"attempting cache repair + retry")
                try:
                    import os
                    import shutil
                    from huggingface_hub import snapshot_download
                    # Wipe the partial snapshot so a clean re-download isn't
                    # blocked by a half-written blob/refs dir.
                    _snap = os.path.join(
                        os.environ.get("HUGGINGFACE_HUB_CACHE",
                                       os.path.join(os.path.expanduser("~"),
                                                    ".cache", "huggingface",
                                                    "hub")),
                        "models--" + cfg.rerank_model.replace("/", "--"))
                    if os.path.isdir(_snap):
                        shutil.rmtree(_snap, ignore_errors=True)
                        print(f"[rerank] cleared partial cache {_snap}")
                    _rp = snapshot_download(
                        cfg.rerank_model,
                        allow_patterns=["*.json", "*.txt", "*.safetensors",
                                        "pytorch_model.bin", "tokenizer*",
                                        "*.model"],
                    )
                    print(f"[rerank] re-downloaded snapshot -> {_rp}")
                    self._reranker = _load_flag_reranker(
                        cfg.rerank_model, use_fp16=True, device=cfg.rerank_device)
                    print(f"[rerank] retry load of {cfg.rerank_model} succeeded")
                except Exception as e2:
                    print(f"[rerank] retry also failed: {e2!r}; "
                          f"skipping rerank (pre-rerank order kept)")
                    self._reranker_unavailable = True
                    return ordered
        reranker = self._reranker

        pairs: List[Tuple[str, str]] = []
        for e in ordered:
            row_idx = e["row_idx"]
            text = text_by_row.get(row_idx, "")
            if not text:
                # Reranker needs text; skip entries without it.
                pairs.append((query, ""))
            else:
                text = text[: cfg.rerank_max_input_chars]
                pairs.append((query, text))

        scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        for e, sc in zip(ordered, scores):
            e["rerank_score"] = float(sc)

        # Sort by rerank_score desc, then original score desc, then row_idx asc.
        ordered.sort(
            key=lambda e: (
                -e.get("rerank_score", 0.0),
                -e.get("score", 0.0),
                e["row_idx"],
            )
        )
        # Promote rerank_score to the primary score field for downstream use.
        for e in ordered:
            e["score"] = e.get("rerank_score", e.get("score", 0.0))
            e["source"] = "rerank"
        return ordered
