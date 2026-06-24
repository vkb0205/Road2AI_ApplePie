"""Smoke test for the Neo4j-backed graph expander.

Connects to the self-hosted Neo4j (creds from .env), introspects the schema,
then runs a single ``expand()`` call on a few seed chunks found in
``chunk_meta_slim.parquet`` to verify the Cypher traversal returns rows.

Run from the project root::

    cd Road2AI_ApplePie && PYTHONPATH=src python scripts/smoke_neo4j_expander.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load .env so NEO4J_* vars are visible.
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from retrieval.neo4j_graph_expand import Neo4jGraphExpander  # noqa: E402

META = ROOT / "data" / "stage6_data" / "chunk_meta_slim.parquet"


def main() -> int:
    if not META.exists():
        print(f"ERROR: {META} not found — run the Stage 6 bundle build first.")
        return 1

    print("=" * 60)
    print("Neo4jGraphExpander smoke test")
    print("=" * 60)
    print(f"NEO4J_URI      = {os.environ.get('NEO4J_URI', '(unset)')}")
    print(f"NEO4J_USERNAME = {os.environ.get('NEO4J_USERNAME', '(unset)')}")
    print(f"password       = {'(set)' if os.environ.get('NEO4J_PASSWORD') else '(unset)'}")

    # 1) connect
    t0 = time.time()
    try:
        expander = Neo4jGraphExpander.from_env(str(META))
    except Exception as e:
        print(f"\n[FAIL] could not build expander: {e}")
        return 2
    print(f"\n[ok] connected in {time.time() - t0:.2f}s")
    print(f"[ok] row_to_uid has {len(expander.row_to_uid)} entries")

    # 2) schema introspection
    print("\n--- schema counts ---")
    with expander.driver.session() as s:
        for label in ("GraphNode", "Document", "Article", "Chunk", "Concept"):
            n = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            print(f"  :{label:<10} {n:>10,}")
        for rel in ("HAS_ARTICLE", "HAS_CHUNK", "MENTIONS",
                    "DETAILS", "AMENDS", "REPLACES", "CITES_REF", "BASED_ON"):
            n = s.run(
                f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c"
            ).single()["c"]
            print(f"  -[:{rel:<12}]-> {n:>10,}")

    # 3) pick seed row_idx whose parent DOC has cross-doc relations.
    # The graph is DOC→ART only (no Chunk nodes), so seeds are picked by
    # joining the parquet row_to_uid (row_idx -> doc_uid) against Neo4j
    # Article nodes, then checking the parent Document's expansion relations.
    print("\n--- picking seed row_idx whose parent DOC has cross-doc rels ---")
    sample_uids = list(expander.row_to_uid.items())[:2000]
    sample_params = [
        {"row_idx": int(ri), "uid": str(uid)} for ri, uid in sample_uids
    ]
    with expander.driver.session() as s:
        seeds_res = s.run(
            """
            UNWIND $rows AS r
            MATCH (art:Article {id: 'ART:' + r.uid})
            MATCH (doc:Document)-[:HAS_ARTICLE]->(art)
            WHERE (doc)-[:DETAILS|AMENDS|REPLACES|CITES_REF|BASED_ON]->()
               OR ()-[:DETAILS|AMENDS|REPLACES|CITES_REF|BASED_ON]->(doc)
            RETURN r.row_idx AS row_idx LIMIT 3
            """,
            rows=sample_params,
        )
        seed_rows = [int(r["row_idx"]) for r in seeds_res]
    print(f"  seed row_idx = {seed_rows}")
    if not seed_rows:
        # Fallback: any row_idx whose doc_uid is a real Article node.
        with expander.driver.session() as s:
            seeds_res = s.run(
                """
                UNWIND $rows AS r
                MATCH (art:Article {id: 'ART:' + r.uid})
                RETURN r.row_idx AS row_idx LIMIT 3
                """,
                rows=sample_params,
            )
            seed_rows = [int(r["row_idx"]) for r in seeds_res]
        print(f"  (fallback) seed row_idx = {seed_rows}")
    if not seed_rows:
        print("\n[warn] no seed row_idx found whose doc_uid is in Neo4j — "
              "check that upload_graph_to_neo4j.py ran and ART ids match.")
        expander.close()
        return 3

    candidates = [(rid, 1.0) for rid in seed_rows]

    # 4) expand
    print("\n--- expand() ---")
    t0 = time.time()
    expanded = expander.expand(candidates, top_n=50)
    print(f"  {len(expanded)} rows in {time.time() - t0:.2f}s")
    by_src: dict = {}
    for r in expanded:
        by_src.setdefault(r["source"], 0)
        by_src[r["source"]] += 1
    print(f"  by source: {by_src}")
    for r in expanded[:6]:
        print(f"    row_idx={r['row_idx']:<8} score={r['score']:.4f} "
              f"source={r['source']}")

    # 5) build_graph_context sanity (needs a real DOC id with a cross-doc rel)
    print("\n--- build_graph_context() ---")
    with expander.driver.session() as s:
        doc_id = s.run(
            "MATCH (d:Document) WHERE (d)-[:DETAILS|AMENDS|REPLACES|CITES_REF|BASED_ON]->()"
            " RETURN d.id AS did LIMIT 1"
        ).single()
    if doc_id:
        ctx = expander.build_graph_context([doc_id["did"]])
        print(f"  DOC id = {doc_id['did']}")
        for ln in ctx.split("\n")[:5]:
            print(f"    {ln}")
    else:
        print("  (no Document with a cross-doc relation found — skipped)")

    expander.close()
    print("\n[done] smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
