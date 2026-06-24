"""End-to-end single-query retrieval test — all stages, graph via Neo4j.

Runs ONE query through the full G-LRAG pipeline and prints what each stage
produces, so you can confirm the query is "found" (or not) at every stage:

  Stage 1  lexical leg   (FTS5 BM25)
  Stage 2  dense leg     (FAISS, optional — auto-skipped if faiss missing)
  Stage 3  RRF fusion
  Stage 4  graph expand  (Neo4j-backed, live instance via .env)
  Stage 5  fetch metadata + text
  Stage 6  rerank        (bge-reranker-v2-m3, optional — auto-skipped if torch missing)
  Stage 7  build final top-K Hit list
  Post    make_relevant_lists → (relevant_docs, relevant_articles)

The dense (Stage 2) and rerank (Stage 6) legs are enabled **only when their
heavy GPU deps are installed**; otherwise the retriever's lazy-import guards
skip them gracefully (the documented CPU-baseline path). The graph stage is
always Neo4j-backed here — that's the point of this script.

Usage::

    cd Road2AI_ApplePie && PYTHONPATH=src python scripts/test_single_query_neo4j.py
    cd Road2AI_ApplePie && PYTHONPATH=src python scripts/test_single_query_neo4j.py "thuế xuất nhập khẩu"
    cd Road2AI_ApplePie && PYTHONPATH=src python scripts/test_single_query_neo4j.py --top-k 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

STAGE6 = ROOT / "data" / "stage6_data"
DB = STAGE6 / "chunk_store.sqlite"
META = STAGE6 / "chunk_meta_slim.parquet"
FAISS_IDX = STAGE6 / "faiss_index__BAAI_bge-m3.index"
MODEL_META = STAGE6 / "embed_model_meta__BAAI_bge-m3.json"

DEFAULT_QUERY = "Thủ tục đăng ký doanh nghiệp lần đầu bao gồm những bước nào?"


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _fmt_meta(d: dict) -> str:
    law = d.get("law_id", "") or "-"
    ten = (d.get("ten_van_ban", "") or "-")
    if len(ten) > 48:
        ten = ten[:45] + "..."
    dieu = d.get("dieu_so", "") or "-"
    return f"law={law:<22} dieu={dieu:<10} ten={ten}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Single-query retrieval test (Neo4j graph)")
    p.add_argument("query", nargs="?", default=DEFAULT_QUERY,
                   help=f"Query string (default: {DEFAULT_QUERY!r})")
    p.add_argument("--top-k", type=int, default=5, help="final top-K")
    p.add_argument("--fts-mode", choices=["bm25_ranked", "fts_fast"],
                   default="bm25_ranked", help="FTS5 query mode")
    p.add_argument("--no-dense", action="store_true",
                   help="force-disable the dense leg even if faiss is installed")
    p.add_argument("--no-rerank", action="store_true",
                   help="force-disable the rerank step even if torch is installed")
    args = p.parse_args(argv)

    query: str = args.query
    top_k: int = args.top_k

    banner(f"QUERY: {query!r}")
    print(f"top_k={top_k}  fts_mode={args.fts_mode}  "
          f"dense={'forced-off' if args.no_dense else 'auto'}  "
          f"rerank={'forced-off' if args.no_rerank else 'auto'}")
    print(f"NEO4J_URI = {os.environ.get('NEO4J_URI', '(unset)')}")

    # ------------------------------------------------------------------ #
    # Imports (heavy GPU deps imported lazily inside the components)
    # ------------------------------------------------------------------ #
    from retrieval.bm25_index import FTSIndex
    from retrieval.rrf import rrf_fuse
    from retrieval.retriever import RetrievalConfig, Hit, make_relevant_lists

    # ------------------------------------------------------------------ #
    # Stage 1 — lexical leg (FTS5 BM25)
    # ------------------------------------------------------------------ #
    banner("Stage 1 — lexical leg (FTS5)")
    t0 = time.time()
    fts = FTSIndex(str(DB), mode=args.fts_mode).open()
    print(f"chunks table rows = {fts.n_rows:,}  lexical_backend={fts.lexical_backend!r}")
    lexical = fts.search(query, top_k=50)
    print(f"lexical hits = {len(lexical)}  ({time.time()-t0:.2f}s)")
    if not lexical:
        print("[warn] lexical leg returned nothing — the query may be too "
              "specific or the FTS index tokenises it away.")
    for r in lexical[:8]:
        print(f"  row_idx={r['row_idx']:<7} bm25={r.get('bm25_score','-')!s:<8} "
              f"{_fmt_meta(r)}")
    if len(lexical) > 8:
        print(f"  ... +{len(lexical)-8} more")

    # ------------------------------------------------------------------ #
    # Stage 2 — dense leg (optional)
    # ------------------------------------------------------------------ #
    banner("Stage 2 — dense leg (FAISS, optional)")
    dense: list = []
    faiss = None
    encoder = None
    want_dense = not args.no_dense
    if want_dense:
        try:
            import importlib.util as _u
            if _u.find_spec("faiss") is None or _u.find_spec("torch") is None \
                    or _u.find_spec("FlagEmbedding") is None:
                raise ImportError("faiss/torch/FlagEmbedding not installed")
            from retrieval.faiss_index import FAISSIndex, BGEQueryEncoder
            faiss = FAISSIndex(str(FAISS_IDX), str(META), str(MODEL_META)).load_index()
            encoder = BGEQueryEncoder(use_fp16=True)
            print(f"FAISS index loaded: ntotal={faiss.ntotal:,} dim={faiss.dim}")
        except Exception as e:
            print(f"[skip] dense leg disabled: {e}")
            faiss = None
            encoder = None
    if faiss is not None and encoder is not None:
        t0 = time.time()
        qvec = encoder.encode(query)
        dense = faiss.search(qvec, top_k=50)
        print(f"dense hits = {len(dense)}  ({time.time()-t0:.2f}s)")
        for r in dense[:8]:
            print(f"  row_idx={r['row_idx']:<7} dense={r['dense_score']:.4f} "
                  f"{_fmt_meta(r)}")
    else:
        print("[skip] no dense backend — Stage 3 fuses the lexical list alone "
              "(degenerate but valid RRF).")

    # ------------------------------------------------------------------ #
    # Stage 3 — RRF fusion
    # ------------------------------------------------------------------ #
    banner("Stage 3 — RRF fusion")
    rankings = []
    if lexical:
        rankings.append(lexical)
    if dense:
        rankings.append(dense)
    fused = rrf_fuse(rankings, k=60, fused_top=30, seed=42) if rankings else []
    print(f"fused candidates = {len(fused)}")
    for r in fused[:8]:
        print(f"  row_idx={r['row_idx']:<7} rrf={r['rrf_score']:.4f} "
              f"src={r['source']}")

    # ------------------------------------------------------------------ #
    # Stage 4 — graph expansion (Neo4j)
    # ------------------------------------------------------------------ #
    banner("Stage 4 — graph expansion (Neo4j)")
    from retrieval.neo4j_graph_expand import Neo4jGraphExpander
    t0 = time.time()
    try:
        expander = Neo4jGraphExpander.from_env(str(META))
    except Exception as e:
        print(f"[FAIL] could not connect to Neo4j: {e}")
        fts.close()
        return 2
    print(f"connected in {time.time()-t0:.2f}s  "
          f"row_to_uid entries={len(expander.row_to_uid):,}")

    # quick schema counts so we know the graph isn't empty
    with expander.driver.session() as s:
        for label in ("Document", "Article", "Chunk", "Concept"):
            n = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            print(f"  :{label:<10} {n:>10,}")

    candidates = [(r["row_idx"], r["rrf_score"]) for r in fused]
    t0 = time.time()
    expanded = expander.expand(candidates, top_n=50)
    print(f"expanded candidates = {len(expanded)}  ({time.time()-t0:.2f}s)")
    by_src: dict = {}
    for r in expanded:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"  by source: {by_src}")
    for r in expanded[:10]:
        print(f"  row_idx={r['row_idx']:<7} score={r['score']:.4f} src={r['source']}")
    if len(expanded) > 10:
        print(f"  ... +{len(expanded)-10} more")

    # ------------------------------------------------------------------ #
    # Stage 5 — fetch metadata + text
    # ------------------------------------------------------------------ #
    banner("Stage 5 — fetch metadata + text")
    ordered = sorted(expanded, key=lambda e: (-e["score"], e["row_idx"]))
    row_idxs = [e["row_idx"] for e in ordered]
    rows = fts.fetch_chunks(row_idxs)
    meta_by_row = {int(r["row_idx"]): r for r in rows}
    print(f"fetched metadata for {len(meta_by_row)}/{len(row_idxs)} rows")
    for e in ordered[:6]:
        ri = e["row_idx"]
        m = meta_by_row.get(ri)
        txt = (m["chunk_text"][:80] + "...") if m else "(missing)"
        print(f"  row_idx={ri:<7} src={e['source']:<16} "
              f"{_fmt_meta(m) if m else '(no meta)'}")
        print(f"          text: {txt}")

    # ------------------------------------------------------------------ #
    # Stage 6 — rerank (optional)
    # ------------------------------------------------------------------ #
    banner("Stage 6 — rerank (bge-reranker-v2-m3, optional)")
    text_by_row = {int(r["row_idx"]): str(r.get("chunk_text", "")) for r in rows}
    use_rerank = (not args.no_rerank) and bool(ordered)
    if use_rerank:
        try:
            import importlib.util as _u
            if _u.find_spec("torch") is None or _u.find_spec("FlagEmbedding") is None:
                raise ImportError("torch/FlagEmbedding not installed")
            from FlagEmbedding import FlagReranker
            reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
            pairs = [(query, text_by_row.get(e["row_idx"], "")[:2000]) for e in ordered]
            t0 = time.time()
            scores = reranker.compute_score(pairs, normalize=True)
            if isinstance(scores, float):
                scores = [scores]
            for e, sc in zip(ordered, scores):
                e["rerank_score"] = float(sc)
            ordered.sort(key=lambda e: (
                -e.get("rerank_score", 0.0), -e.get("score", 0.0), e["row_idx"]))
            for e in ordered:
                e["score"] = e.get("rerank_score", e.get("score", 0.0))
                e["source"] = "rerank"
            print(f"reranked {len(ordered)} pairs  ({time.time()-t0:.2f}s)")
            for e in ordered[:6]:
                print(f"  row_idx={e['row_idx']:<7} rerank={e['score']:.4f}")
        except Exception as e:
            print(f"[skip] rerank disabled: {e}")
    else:
        print("[skip] rerank disabled (no candidates or --no-rerank)")

    # ------------------------------------------------------------------ #
    # Stage 7 — build final top-K Hit list
    # ------------------------------------------------------------------ #
    banner(f"Stage 7 — final top-{top_k} Hits")
    hits: list[Hit] = []
    for e in ordered[:top_k]:
        ri = e["row_idx"]
        m = meta_by_row.get(ri, {})
        hits.append(Hit(
            row_idx=ri,
            score=float(e["score"]),
            source=str(e.get("source", "fused")),
            law_id=str(m.get("law_id", "")),
            ten_van_ban=str(m.get("ten_van_ban", "")),
            dieu_so=str(m.get("dieu_so", "")),
            chunk_id=str(m.get("chunk_id", "")),
            doc_uid=str(m.get("doc_uid", "")),
            chunk_text=text_by_row.get(ri, ""),
        ))
    for h in hits:
        print(f"  Hit row_idx={h.row_idx:<7} score={h.score:.4f} src={h.source:<16}")
        print(f"      {h.law_id} | {h.ten_van_ban} | {h.dieu_so}")
        print(f"      {h.chunk_text[:120]}")

    # ------------------------------------------------------------------ #
    # Post — make_relevant_lists
    # ------------------------------------------------------------------ #
    banner("Post — make_relevant_lists")
    docs, articles = make_relevant_lists(hits)
    print(f"relevant_docs     ({len(docs)}):")
    for d in docs:
        print(f"  - {d}")
    print(f"relevant_articles ({len(articles)}):")
    for a in articles:
        print(f"  - {a}")

    # ------------------------------------------------------------------ #
    # Verdict
    # ------------------------------------------------------------------ #
    banner("VERDICT")
    found_at: list[str] = []
    if lexical:
        found_at.append("1.lexical")
    if dense:
        found_at.append("2.dense")
    if fused:
        found_at.append("3.rrf")
    if expanded:
        found_at.append("4.graph(neo4j)")
    if meta_by_row:
        found_at.append("5.fetch")
    if any(h.source == "rerank" for h in hits):
        found_at.append("6.rerank")
    if hits:
        found_at.append("7.final-hit")
    print(f"query found at stages: {', '.join(found_at) if found_at else '(nowhere)'}")
    print(f"final hits = {len(hits)}  relevant_docs={len(docs)}  "
          f"relevant_articles={len(articles)}")
    if hits:
        print(f"best hit: row_idx={hits[0].row_idx} score={hits[0].score:.4f} "
              f"-> {hits[0].law_id} | {hits[0].dieu_so}")

    expander.close()
    fts.close()
    return 0 if hits else 1


if __name__ == "__main__":
    sys.exit(main())
