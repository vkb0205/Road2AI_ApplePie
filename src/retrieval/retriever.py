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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retrieval.rrf import rrf_fuse
from retrieval.graph_expand import GraphExpander

__all__ = [
    "HybridRetriever",
    "RetrievalConfig",
    "Hit",
    "make_relevant_lists",
]


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
    rerank_max_input_chars: int = 2000
    seed: int = 42


@dataclass
class Hit:
    """A single final retrieval hit, carrying the metadata needed for output."""

    row_idx: int
    score: float
    source: str
    law_id: str = ""
    ten_van_ban: str = ""
    dieu_so: str = ""
    chunk_id: str = ""
    doc_uid: str = ""
    chunk_text: str = ""

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
        """
        cfg = self.config

        # ---- 1. lexical leg ------------------------------------------------
        lexical_hits = self._lexical_leg(query, cfg.top_bm25)

        # ---- 2. dense leg (optional) --------------------------------------
        dense_hits: List[Dict[str, Any]] = []
        if cfg.use_dense and self.faiss_index is not None:
            dense_hits = self._dense_leg(query, cfg.top_dense)

        # ---- 3. RRF fusion -------------------------------------------------
        rankings: List[List[Any]] = []
        if lexical_hits:
            rankings.append(lexical_hits)
        if dense_hits:
            rankings.append(dense_hits)
        fused = rrf_fuse(
            rankings, k=cfg.rrf_k, fused_top=cfg.fused_top, seed=cfg.seed
        ) if rankings else []

        # ---- 4. graph expansion (optional) --------------------------------
        if self.graph_expander is not None and fused:
            candidates = [(r["row_idx"], r["rrf_score"]) for r in fused]
            expanded = self.graph_expander.expand(
                candidates, top_n=cfg.expanded_top
            )
            expanded_rows = {
                e["row_idx"]: e for e in expanded
            }
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
        meta_by_row = self._fetch_meta(row_idxs)
        text_by_row: Dict[int, str] = {}
        if fetch_text:
            text_by_row = self._fetch_text(row_idxs)

        # ---- 6. rerank (optional) -----------------------------------------
        if cfg.use_reranker and cfg.final_top_k > 0 and ordered:
            ordered = self._rerank(query, ordered, meta_by_row, text_by_row)

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
        return hits

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
        if self._reranker is None:
            self._reranker = FlagReranker(cfg.rerank_model, use_fp16=True)
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
