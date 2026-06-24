"""Stage 7.5 dev-set baseline runner.

Runs the G-LRAG retrieval pipeline over ``dev_set/questions.json`` using the
**lexical (FTS5) leg + graph expansion** (CPU-runnable, no faiss/torch) and
writes:

  - ``dev_set/results_baseline.json``  — submission-shaped records with
    ``relevant_docs`` / ``relevant_articles`` collapsed from the top-K hits.
  - ``dev_set/results_no_graph.json``  — same but with graph expansion
    disabled, for the 7.3 ablation (Δ recall).
  - prints F2 macro for both to stdout.

Usage::

    python scripts/run_devset_baseline.py
    python scripts/run_devset_baseline.py --no-graph   # ablation comparison only

This satisfies PLAN.md tasks 7.5 (baseline number committed) and contributes
the no-expansion vs. expansion recall delta for task 7.3.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval.bm25_index import FTSIndex
from retrieval.faiss_index import FAISSIndex
from retrieval.graph_expand import GraphExpander
from retrieval.neo4j_graph_expand import Neo4jGraphExpander
from retrieval.retriever import HybridRetriever, RetrievalConfig, make_relevant_lists

STAGE6 = ROOT / "data" / "stage6_data"
DB = STAGE6 / "chunk_store.sqlite"
META = STAGE6 / "chunk_meta_slim.parquet"
FAISS_IDX = STAGE6 / "faiss_index__BAAI_bge-m3.index"
MODEL_META = STAGE6 / "embed_model_meta__BAAI_bge-m3.json"
KG = ROOT / "data" / "kg.gpickle"
DEV = ROOT / "dev_set"


def load_questions():
    return json.loads((DEV / "questions.json").read_text(encoding="utf-8"))


def build_retriever(
    use_graph: bool,
    use_dense: bool = False,
    top_k: int = 5,
    fts_mode: str = "bm25_ranked",
    graph_backend: str = "pickle",
):
    # ``bm25_ranked`` reproduces the PLAN.md 7.5 committed baseline (F2 0.0670).
    # Use ``--fts-mode fts_fast`` for the faster unranked path (artifacts_guide.md
    # warns fts_fast does not sort by relevance, so it scores lower on F2).
    fts = FTSIndex(str(DB), mode=fts_mode).open()
    faiss = None
    if use_dense:
        try:
            faiss = FAISSIndex(str(FAISS_IDX), str(META), str(MODEL_META)).load_index()
            print("[baseline] FAISS dense leg enabled")
        except Exception as e:
            print(f"[baseline] FAISS unavailable ({e}); falling back to lexical-only")
            faiss = None
    expander = None
    if use_graph:
        t0 = time.time()
        try:
            if graph_backend == "neo4j":
                # .env is loaded inside from_env. Falls back to the pickle
                # expander if the neo4j driver is missing or the connection
                # fails (so a CPU-only checkout still runs).
                try:
                    expander = Neo4jGraphExpander.from_env(str(META))
                    print(f"[baseline] Neo4j graph expander connected in {time.time()-t0:.1f}s")
                except Exception as e:
                    print(f"[baseline] Neo4j unavailable ({e}); falling back to pickle")
                    expander = None
            # Fall through to pickle if neo4j failed OR pickle was requested.
            if expander is None and graph_backend != "neo4j" and KG.exists():
                expander = GraphExpander.from_graph_and_meta(str(KG), str(META))
                print(f"[baseline] pickle graph expander loaded in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[baseline] graph expander unavailable ({e}); running without expansion")
            expander = None
    cfg = RetrievalConfig(
        use_dense=faiss is not None,
        use_reranker=False,  # CPU baseline; reranker runs on Kaggle GPU
        top_bm25=50,
        top_dense=50,
        fused_top=30,
        expanded_top=50,
        final_top_k=top_k,
    )
    return HybridRetriever(fts, faiss_index=faiss, graph_expander=expander, config=cfg)


def run(
    use_graph: bool,
    top_k: int = 5,
    fts_mode: str = "bm25_ranked",
    graph_backend: str = "pickle",
):
    questions = load_questions()
    retriever = build_retriever(
        use_graph=use_graph,
        use_dense=False,
        top_k=top_k,
        fts_mode=fts_mode,
        graph_backend=graph_backend,
    )
    records = []
    t0 = time.time()
    for q in questions:
        hits = retriever.retrieve(q["question"], fetch_text=False)
        docs, articles = make_relevant_lists(hits)
        records.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answer": "",  # retrieval-only baseline; generation is Stage 8
                "relevant_docs": docs,
                "relevant_articles": articles,
            }
        )
    elapsed = time.time() - t0
    tag = "graph" if use_graph else "no_graph"
    print(f"[baseline:{tag}] {len(records)} questions in {elapsed:.1f}s")
    return records


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage 7.5 dev-set baseline runner")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--fts-mode",
        choices=["bm25_ranked", "fts_fast"],
        default="bm25_ranked",
        help="FTS5 query mode. 'bm25_ranked' (default) reproduces the PLAN "
        "baseline F2=0.0670; 'fts_fast' is unranked and scores lower.",
    )
    p.add_argument("--no-graph", action="store_true", help="also run no-graph ablation")
    p.add_argument(
        "--graph-backend",
        choices=["pickle", "neo4j"],
        default="pickle",
        help=(
            "Graph expansion backend. 'pickle' (default) loads kg.gpickle "
            "in memory; 'neo4j' queries a self-hosted Neo4j instance via "
            "NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD (.env). Falls back "
            "to pickle if Neo4j is unavailable."
        ),
    )
    args = p.parse_args(argv)

    # ---- with graph expansion (primary baseline) ----
    recs_graph = run(
        use_graph=True,
        top_k=args.top_k,
        fts_mode=args.fts_mode,
        graph_backend=args.graph_backend,
    )
    out_graph = DEV / "results_baseline.json"
    out_graph.write_text(json.dumps(recs_graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[baseline] wrote {out_graph}")

    # ---- without graph expansion (ablation) ----
    recs_no = run(use_graph=False, top_k=args.top_k, fts_mode=args.fts_mode)
    out_no = DEV / "results_no_graph.json"
    out_no.write_text(json.dumps(recs_no, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[baseline] wrote {out_no}")

    # ---- F2 scoring ----
    sys.path.insert(0, str(DEV))
    from eval import f2_macro  # noqa: E402

    gt = json.loads((DEV / "ground_truth.json").read_text(encoding="utf-8"))
    f2_graph = f2_macro(recs_graph, gt)
    f2_no = f2_macro(recs_no, gt)
    print(f"\n=== F2 macro (relevant_articles) ===")
    print(f"  lexical + graph expansion : {f2_graph:.4f}")
    print(f"  lexical only (no graph)   : {f2_no:.4f}")
    print(f"  delta (graph - no graph)  : {f2_graph - f2_no:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
