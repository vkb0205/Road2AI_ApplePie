"""Kaggle-GPU integration for the recall-first, document-anchored pipeline.

This is the glue that wires the real Stage 6 bundle (FTS5 + FAISS + BGE-m3 +
cross-encoder reranker) into the pure-Python, unit-tested orchestration in
:mod:`retrieval.doc_anchor` and the F2 selector in
:mod:`retrieval.article_select`.

It is import-safe on a CPU-only box (heavy deps — torch / faiss / FlagEmbedding —
are imported lazily, only when the dense leg / reranker is actually built). On
Kaggle with a GPU it runs end-to-end:

    pipe = build_pipeline(STAGE6_DIR, use_dense=True, use_rerank=True)
    submission, pool = run_dev_set(pipe, "dev_set/ground_truth.json")

``submission`` is the grader-format dump (``relevant_docs`` / ``relevant_articles``)
and ``pool`` is the candidate-pool dump consumed by ``dev_set/tune_selector.py``
for offline policy tuning.

The whole pipeline is:

    query
      → lexical (FTS5)  ┐
      → dense  (FAISS)  ┘→ anchor_documents → harvest_articles
      → cross-encoder rerank (article passages)
      → select_articles (authority suppression + F2-optimal K)
      → relevant_docs / relevant_articles
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from retrieval.article_select import (
    ArticleCandidate,
    SelectConfig,
    select_articles,
)
from retrieval.doc_anchor import (
    DocAnchorConfig,
    DocAnchoredRetriever,
    pool_record,
)


# --------------------------------------------------------------------------- #
# Stage 6 bundle paths
# --------------------------------------------------------------------------- #
@dataclass
class BundlePaths:
    """Resolve the Stage 6 artefact paths under a single root directory."""

    root: str
    sqlite_name: str = "chunk_store.sqlite"
    faiss_name: str = "faiss_index__BAAI_bge-m3.index"
    meta_name: str = "chunk_meta_slim.parquet"
    model_meta_name: str = "embed_model_meta__BAAI_bge-m3.json"

    @property
    def sqlite(self) -> str:
        return str(Path(self.root) / self.sqlite_name)

    @property
    def faiss(self) -> str:
        return str(Path(self.root) / self.faiss_name)

    @property
    def meta(self) -> str:
        return str(Path(self.root) / self.meta_name)

    @property
    def model_meta(self) -> str:
        return str(Path(self.root) / self.model_meta_name)


# --------------------------------------------------------------------------- #
# Search-fn adapters around the real indices
# --------------------------------------------------------------------------- #
def make_lexical_search(fts) -> "callable":
    """Wrap an open :class:`retrieval.bm25_index.FTSIndex` as a SearchFn."""

    def _search(query: str, top_k: int) -> List[Dict[str, Any]]:
        return fts.search(query, top_k=top_k)

    return _search


def make_dense_search(faiss_index, encoder) -> "callable":
    """Wrap a loaded FAISSIndex + BGEQueryEncoder as a SearchFn."""

    def _search(query: str, top_k: int) -> List[Dict[str, Any]]:
        qvec = encoder.encode(query)
        return faiss_index.search(qvec, top_k=top_k)

    return _search


def make_fetch_text(fts) -> "callable":
    """Build a TextFn over the FTS index's chunk store.

    ``FTSIndex.fetch_chunks`` returns chunk dicts; we map row_idx → text,
    tolerating either a ``chunk_text`` or ``text`` field.
    """

    def _fetch(row_idxs: Sequence[int]) -> Dict[int, str]:
        rows = fts.fetch_chunks(list(row_idxs))
        out: Dict[int, str] = {}
        for r in rows:
            ridx = int(r.get("row_idx", -1))
            txt = r.get("chunk_text") or r.get("text") or ""
            out[ridx] = str(txt)
        return out

    return _fetch


def make_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: Optional[str] = "cuda:0",
    use_fp16: bool = True,
    max_input_chars: int = 2000,
) -> "callable":
    """Build a RerankFn backed by ``BAAI/bge-reranker-v2-m3`` (lazy, GPU).

    Reuses the project's reranker loader so device pinning / fp16 behaviour
    matches the rest of the pipeline.
    """
    from retrieval.retriever import _load_flag_reranker  # lazy: pulls torch

    reranker = _load_flag_reranker(model_name, use_fp16=use_fp16, device=device)

    def _rerank(query: str, passages: Sequence[str]) -> List[float]:
        pairs = [[query, (p or "")[:max_input_chars]] for p in passages]
        if not pairs:
            return []
        scores = reranker.compute_score(pairs, normalize=True)
        # compute_score returns a float for a single pair, else a list.
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(s) for s in scores]

    return _rerank


# --------------------------------------------------------------------------- #
# Pipeline assembly
# --------------------------------------------------------------------------- #
@dataclass
class Pipeline:
    retriever: DocAnchoredRetriever
    select_cfg: SelectConfig
    _fts: Any = None  # kept for lifecycle/close

    def answer(self, query: str) -> List[ArticleCandidate]:
        pool = self.retriever.retrieve_pool(query)
        return select_articles(pool, self.select_cfg)

    def close(self) -> None:
        if self._fts is not None:
            try:
                self._fts.close()
            except Exception:
                pass


def build_pipeline(
    stage6_root: str,
    use_dense: bool = True,
    use_rerank: bool = True,
    fts_mode: str = "bm25_ranked",
    gpu_id: int = 0,
    anchor_cfg: Optional[DocAnchorConfig] = None,
    select_cfg: Optional[SelectConfig] = None,
) -> Pipeline:
    """Construct the full pipeline from a Stage 6 bundle directory.

    On a CPU-only box call with ``use_dense=False, use_rerank=False`` to run the
    lexical-only anchoring path (no torch/faiss needed).
    """
    from retrieval.bm25_index import FTSIndex  # stdlib sqlite only

    paths = BundlePaths(stage6_root)
    fts = FTSIndex(paths.sqlite, mode=fts_mode).open()
    lexical = make_lexical_search(fts)

    dense = None
    if use_dense:
        from retrieval.faiss_index import FAISSIndex, BGEQueryEncoder  # lazy

        faiss_index = FAISSIndex(
            paths.faiss, paths.meta, paths.model_meta, gpu_id=gpu_id
        ).load_index()
        encoder = BGEQueryEncoder(device=f"cuda:{gpu_id}")
        dense = make_dense_search(faiss_index, encoder)

    fetch_text = make_fetch_text(fts) if use_rerank else None
    rerank = make_reranker(device=f"cuda:{gpu_id}") if use_rerank else None

    retriever = DocAnchoredRetriever(
        lexical_search=lexical,
        dense_search=dense,
        fetch_text=fetch_text,
        rerank=rerank,
        cfg=anchor_cfg or DocAnchorConfig(),
    )
    return Pipeline(
        retriever=retriever,
        select_cfg=select_cfg or SelectConfig(),
        _fts=fts,
    )


# --------------------------------------------------------------------------- #
# Dev-set driver
# --------------------------------------------------------------------------- #
def run_dev_set(
    pipe: Pipeline, ground_truth_path: str
) -> "tuple[List[Dict], List[Dict]]":
    """Run the pipeline over the dev set.

    Returns ``(submission, pool)`` where ``submission`` is the grader-format dump
    and ``pool`` is the candidate-pool dump for offline tuning. The pool is built
    from the *pre-selection* reranked candidates so policy knobs can be tuned
    without re-running the GPU legs.
    """
    gts = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    submission: List[Dict] = []
    pool: List[Dict] = []
    for gt in gts:
        qid = int(gt["id"])
        q = gt.get("question", "")
        candidates = pipe.retriever.retrieve_pool(q)
        pool.append(pool_record(qid, q, candidates))

        chosen = select_articles(candidates, pipe.select_cfg)
        rel_articles = [a.to_relevant_string() for a in chosen]
        # relevant_docs: distinct source documents of the chosen articles.
        seen = []
        for a in chosen:
            doc = f"{a.law_id}|{a.ten_van_ban}"
            if doc not in seen:
                seen.append(doc)
        submission.append(
            {
                "id": qid,
                "question": q,
                "answer": "",
                "relevant_docs": seen,
                "relevant_articles": rel_articles,
            }
        )
    return submission, pool


def dump_json(obj: Any, path: str) -> None:
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
