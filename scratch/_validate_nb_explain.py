"""Validate the instrumented AdvancedHybridRetriever (from the notebook)
against the real local FTS bundle, lexical-only + multi-query + article-agg,
with explain=True so the per-stage trace is exercised end-to-end.

Executes the notebook's Section-4 code cells in a controlled namespace with
GPU-dependent flags forced off (no GPU locally). This is the closest local
proxy to running the notebook's explainability cell.
"""
import json, sys, time
from pathlib import Path

ROOT = Path("Road2AI_ApplePie")
sys.path.insert(0, str(ROOT / "src"))

nb = json.loads((ROOT / "notebooks/retrieval_colab_advanced.ipynb").read_text(encoding="utf-8"))

# Build the real base retriever (lexical-only) to feed the advanced layer.
from retrieval.bm25_index import FTSIndex
from retrieval.retriever import HybridRetriever, RetrievalConfig, make_relevant_lists

DB = ROOT / "data/stage6_data/chunk_store.sqlite"
fts = FTSIndex(str(DB), mode="bm25_ranked").open()

# Globals the notebook cells expect, forced CPU-only.
G = {
    "USE_DENSE": False, "USE_GRAPH": False, "USE_RERANK": False,
    "USE_MULTIQUERY": True, "USE_HYDE": False, "USE_ARTICLE_AGG": True,
    "RRF_K": 60, "BM25_TOPK": 40, "DENSE_TOPK": 100, "CANDIDATE_TOPK": 96,
    "RERANK_TOPK": 48, "ARTICLE_CTX_TOPK": 16, "ARTICLE_CTX_MAX": 24,
    "FINAL_TOP_K": 10, "MAX_QUERY_VARIANTS": 8, "HYDE_MAX_NEW_TOKENS": 192,
    "HYDE_MODEL": "Qwen/Qwen2.5-7B-Instruct",
    "FTS_MODE": "bm25_ranked",
    "fts": fts, "faiss_index": None, "query_encoder": None,
    "graph_expander": None, "make_relevant_lists": make_relevant_lists,
}

cfg = RetrievalConfig(use_dense=False, use_reranker=False,
                      top_bm25=G["BM25_TOPK"], top_dense=G["DENSE_TOPK"],
                      rrf_k=G["RRF_K"], fused_top=G["CANDIDATE_TOPK"],
                      expanded_top=G["CANDIDATE_TOPK"], final_top_k=G["FINAL_TOP_K"])
base_retriever = HybridRetriever(fts, faiss_index=None, graph_expander=None,
                                 query_encoder=None, config=cfg)
G["base_retriever"] = base_retriever

# Exec the Section-4 code cells (indices 14..18 in the patched notebook).
for i in range(14, 19):
    src = "".join(nb["cells"][i]["source"])
    # Skip the HyDE class instantiation only; it's guarded by USE_HYDE.
    exec(compile(src, f"<nb-cell-{i}>", "exec"), G)

advanced_retriever = G["advanced_retriever"]
print("=== advanced retriever built; multiquery=%s article_agg=%s ==="
      % (advanced_retriever.use_multiquery, advanced_retriever.use_article_agg))

QUERY = ("đối thủ sao chép trái phép phần mềm để cho thuê thu lợi; "
         "cần xác định hành vi xâm phạm quyền tác giả và chuẩn bị chứng cứ gì?")
t0 = time.time()
hits = advanced_retriever.retrieve(QUERY, fetch_text=True, explain=True, print_trace=True)
print("\n=== explained retrieve: %d hits in %.2fs ===" % (len(hits), time.time() - t0))

trace = advanced_retriever.last_trace
assert trace is not None, "last_trace should be populated with explain=True"
names = [s.name for s in trace.stages]
print("stages:", names)
expected = ["variants", "legs", "weighted_rrf", "graph", "fetch", "rerank", "article_agg", "final", "output"]
assert names == expected, f"stage order mismatch: {names}"
for s in trace.stages:
    assert s.elapsed_ms is None or s.elapsed_ms >= 0.0
# variants stage must carry the generated variants
v = trace.stage("variants")
assert v.count == len(G["build_query_variants"](QUERY)), "variant count mismatch"
# legs stage must carry per-variant counts
legs = trace.stage("legs")
assert "per_variant" in legs.diagnostics
# fetch stage must report resolved vs requested
fetch = trace.stage("fetch")
assert fetch.diagnostics["requested"] == fetch.diagnostics["resolved"] + len(fetch.diagnostics["missing_rows"])
# output stage must carry relevant_articles
out = trace.stage("output")
assert "relevant_articles" in out.diagnostics
print("\nALL EXPLAIN-TRACE ASSERTIONS PASSED")
fts.close()
