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
    max_length: int = 512,
    batch_size: int = 32,
    backend: str = "transformers",
) -> "callable":
    """Build a RerankFn for ``BAAI/bge-reranker-v2-m3`` (lazy, GPU).

    ``backend``:
      - ``"transformers"`` (default): score the cross-encoder directly with
        ``AutoModelForSequenceClassification`` + a **fast** tokenizer. This is
        version-robust and avoids FlagEmbedding's ``compute_score`` path, which
        breaks on recent ``transformers`` (slow ``XLMRobertaTokenizer`` lacks
        ``prepare_for_model`` → ``AttributeError``).
      - ``"flag"``: the legacy FlagEmbedding ``FlagReranker.compute_score``
        path (kept for parity where the version combo is known-good).

    Both return scores normalised to ``[0, 1]`` (sigmoid for the transformers
    backend, ``normalize=True`` for the flag backend).
    """
    if backend == "flag":
        from retrieval.retriever import _load_flag_reranker  # lazy: pulls torch

        reranker = _load_flag_reranker(model_name, use_fp16=use_fp16, device=device)

        def _rerank_flag(query: str, passages: Sequence[str]) -> List[float]:
            pairs = [[query, (p or "")[:max_input_chars]] for p in passages]
            if not pairs:
                return []
            scores = reranker.compute_score(pairs, normalize=True)
            if isinstance(scores, (int, float)):
                return [float(scores)]
            return [float(s) for s in scores]

        return _rerank_flag

    # --- transformers-native backend (default, version-robust) -------------
    import torch  # lazy
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    # Force the fast (Rust) tokenizer; the slow XLMRoberta tokenizer is what
    # triggers the prepare_for_model AttributeError under newer transformers.
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(dev)
    if use_fp16 and str(dev).startswith("cuda"):
        model = model.half()
    model.eval()

    def _rerank_hf(query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        scores: List[float] = []
        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            queries = [query] * len(batch)
            texts = [(p or "")[:max_input_chars] for p in batch]
            inputs = tokenizer(
                queries,
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(dev)
            with torch.no_grad():
                logits = model(**inputs, return_dict=True).logits.view(-1).float()
                probs = torch.sigmoid(logits)  # normalise to [0, 1]
            scores.extend(probs.detach().cpu().tolist())
        return scores

    return _rerank_hf


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


# --------------------------------------------------------------------------- #
# Unified pipeline factory (retrieve + optional decompose + select + generate)
# --------------------------------------------------------------------------- #
def make_fts_text_provider(fts) -> "callable":
    """Build a ``(law_id, ten_van_ban, dieu_so) -> str`` text provider over FTS.

    Looks up the best (longest) ``chunk_text`` for a given ``law_id`` +
    ``dieu_so`` from the Stage-6 ``chunks`` table, so the generator gets real
    article text as grounding context. Returns "" on any miss so generation
    degrades gracefully (the answer still cites via the allowed list).
    """

    def _provider(law_id: str, ten_van_ban: str, dieu_so: str) -> str:
        conn = getattr(fts, "_conn", None)
        if conn is None:
            return ""
        try:
            rows = conn.execute(
                "SELECT chunk_text FROM chunks WHERE law_id = ? AND dieu_so = ? "
                "ORDER BY length(chunk_text) DESC LIMIT 1",
                (str(law_id), str(dieu_so)),
            ).fetchall()
        except Exception:
            return ""
        if rows:
            r = rows[0]
            try:
                return str(r["chunk_text"] or "")
            except Exception:
                return str(r[0] or "")
        return ""

    return _provider


def build_unified_pipeline(
    stage6_root: str,
    use_dense: bool = True,
    use_rerank: bool = True,
    use_decomposition: bool = False,
    use_llm_decomposition: bool = False,
    use_generator: bool = True,
    fts_mode: str = "bm25_ranked",
    gpu_id: int = 0,
    anchor_cfg: Optional[DocAnchorConfig] = None,
    select_cfg: Optional[SelectConfig] = None,
    gen_model: str = "Qwen/Qwen2.5-7B-Instruct",
    gen_load_in_4bit: bool = True,
    max_sub_queries: int = 4,
    gen_context_topk: int = 6,
    use_multi_variant: bool = False,
    use_complexity_k: bool = False,
    coverage_quota: bool = True,
    max_variants: int = 5,
):
    """Construct the full unified pipeline from a Stage-6 bundle directory.

    Wires the document-anchored retriever (FTS + FAISS + cross-encoder), the
    optional decomposition router, the F2 selector, and the optional IRAC
    generator plus an FTS-backed text provider. No hardcoded law-id / domain
    tables anywhere.

    Decomposition has two modes. With ``use_decomposition=True`` and
    ``use_llm_decomposition=False`` the router uses the deterministic
    rule-based split. With ``use_llm_decomposition=True`` it additionally drives
    decomposition with the LLM (reusing the *same* 4-bit Qwen as the generator,
    so no second model load), and the rule-based split remains the fallback when
    the LLM call or JSON parse fails.

    Returns ``(pipe, fts)`` — ``pipe`` is a
    :class:`retrieval.unified_pipeline.UnifiedPipeline`; ``fts`` is kept so the
    caller can ``fts.close()`` / reopen for lifecycle management.
    """
    from retrieval.bm25_index import FTSIndex
    from retrieval.unified_pipeline import UnifiedConfig, UnifiedPipeline

    paths = BundlePaths(stage6_root)
    fts = FTSIndex(paths.sqlite, mode=fts_mode).open()
    lexical = make_lexical_search(fts)

    dense = None
    if use_dense:
        from retrieval.faiss_index import FAISSIndex, BGEQueryEncoder

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

    # The planner/router is needed whenever decomposition, multi-variant
    # retrieval, or complexity-K is active (all three consume the plan). When
    # any of these want the LLM planner, the shared Qwen is built once.
    needs_router = use_decomposition or use_multi_variant or use_complexity_k
    needs_llm_planner = needs_router and use_llm_decomposition

    # Build the (4-bit) Qwen llm_call exactly once and share it between the
    # generator and the optional LLM-driven planner. This avoids a second 7B
    # model load when both features are enabled.
    llm_call = None
    if use_generator or needs_llm_planner:
        from generation.generator import build_hf_llm_call

        llm_call = build_hf_llm_call(
            model_name=gen_model,
            device=f"cuda:{gpu_id}",
            load_in_4bit=gen_load_in_4bit,
        )

    router = None
    if needs_router:
        from retrieval.sub_query_router import SubQueryRouter, SubQueryRouterConfig

        router_llm = None
        if needs_llm_planner and llm_call is not None:
            # The router/planner expects ``(prompt: str) -> str``; the
            # generator's llm_call takes a chat ``messages`` list. Adapt by
            # wrapping the prompt as a single user turn. On any LLM/parse
            # failure the planner falls back to its deterministic rule-based
            # split + rule-based complexity internally.
            def router_llm(prompt: str) -> str:
                return llm_call([{"role": "user", "content": prompt}])

        router = SubQueryRouter(
            llm_call=router_llm,
            config=SubQueryRouterConfig(
                use_llm=needs_llm_planner and router_llm is not None,
                max_sub_queries=max_sub_queries,
            ),
        )

    generator = None
    if use_generator and llm_call is not None:
        from generation.generator import IRACGenerator

        generator = IRACGenerator(llm_call=llm_call)

    pipe = UnifiedPipeline(
        retriever=retriever,
        select_cfg=select_cfg or SelectConfig(),
        router=router,
        generator=generator,
        text_provider=make_fts_text_provider(fts),
        cfg=UnifiedConfig(
            use_decomposition=use_decomposition,
            max_sub_queries=max_sub_queries,
            gen_context_topk=gen_context_topk,
            use_multi_variant=use_multi_variant,
            max_variants=max_variants,
            use_complexity_k=use_complexity_k,
            coverage_quota=coverage_quota,
        ),
    )
    return pipe, fts
