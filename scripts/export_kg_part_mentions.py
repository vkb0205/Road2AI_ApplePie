#!/usr/bin/env python3
"""
Export MENTIONS edges from a kg_part_{n}.gpickle checkpoint to CSV,
formatted for the chunk_concept_mentions table.

Usage:
    python scripts/export_kg_part_mentions.py \
        --input data/kg_part_3.gpickle \
        --output data/chunk_concept_mentions.csv \
        --source llm
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError("networkx is required; install with `pip install networkx`") from exc


# ── helpers ──────────────────────────────────────────────────────────────


def load_graph(path: Path) -> nx.MultiDiGraph:
    """Load a gpickle checkpoint; raise if missing or corrupt."""
    if not path.exists():
        print(f"ERROR: input file not found: {path}", file=sys.stderr)
        sys.exit(1)
    import pickle

    try:
        with path.open("rb") as f:
            G = pickle.load(f)
    except Exception as exc:
        print(f"ERROR: failed to load gpickle: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(G, nx.MultiDiGraph):
        print(
            f"ERROR: expected nx.MultiDiGraph, got {type(G).__name__}",
            file=sys.stderr,
        )
        sys.exit(1)

    return G


def extract_mentions(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    """
    Iterate all MENTIONS edges, look up chunk & concept node attributes,
    and return rows suitable for chunk_concept_mentions.
    """
    rows: List[Dict[str, Any]] = []

    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get("relation") != "MENTIONS":
            continue

        # Expect: u is CHUNK node, v is CONCEPT node
        chunk_node = G.nodes[u]
        concept_node = G.nodes[v]

        if chunk_node.get("type") != "Chunk":
            # skip edges where source isn't a Chunk (e.g. Article→Concept)
            continue
        if concept_node.get("type") != "Concept":
            continue

        chunk_id = chunk_node.get("chunk_id") or _parse_chunk_id(u)
        doc_uid = chunk_node.get("doc_uid", "")
        doc_id = chunk_node.get("doc_id")
        concept_name = concept_node.get("name", "")

        if not chunk_id or not concept_name:
            continue

        # Normalise doc_id to int if possible
        if doc_id is not None:
            try:
                doc_id = int(doc_id)
            except (ValueError, TypeError):
                doc_id = None

        rows.append({
            "chunk_id": str(chunk_id),
            "doc_uid": str(doc_uid) if doc_uid else "",
            "doc_id": doc_id,
            "concept_name": concept_name,
        })

    return rows


def _parse_chunk_id(node_key: str) -> str | None:
    """Extract chunk_id from a node key like ``CHUNK:abc-123``."""
    if node_key.startswith("CHUNK:"):
        return node_key[len("CHUNK:"):]
    return None


# ── main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MENTIONS edges from kg_part_{n}.gpickle to CSV",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/kg_part_3.gpickle"),
        help="Path to the gpickle checkpoint (default: data/kg_part_3.gpickle)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/chunk_concept_mentions.csv"),
        help="Output CSV path (default: data/chunk_concept_mentions.csv)",
    )
    parser.add_argument(
        "--source",
        "-s",
        default="llm",
        choices=("llm", "substring"),
        help="Value for mentions_source column (default: llm)",
    )
    args = parser.parse_args()

    print(f"Loading graph from {args.input} …")
    G = load_graph(args.input)

    print(f"Graph has {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    rows = extract_mentions(G)
    print(f"Found {len(rows):,} MENTIONS edges")

    if not rows:
        print("No MENTIONS edges to export. Nothing written.", file=sys.stderr)
        sys.exit(0)

    # Write CSV
    fieldnames = ["chunk_id", "doc_uid", "doc_id", "concept_name", "mentions_source"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "chunk_id": row["chunk_id"],
                "doc_uid": row["doc_uid"],
                "doc_id": row["doc_id"],
                "concept_name": row["concept_name"],
                "mentions_source": args.source,
            })

    print(f"Wrote {len(rows):,} rows to {args.output}")


if __name__ == "__main__":
    main()
