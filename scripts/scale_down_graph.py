#!/usr/bin/env python3
"""
Scale down kg.gpickle for lightweight graphDB hosting.

Strategies implemented:
1. DOC_ONLY        — Keep only Document nodes + Doc↔Doc edges (drop ART/CHUNK layers)
2. DOC_NO_ISOLATES — DOC_ONLY minus isolated (degree-0) Document nodes
3. DOC_HUB         — Keep only Documents with degree >= threshold + their edges
4. DOC_LAW_FILTER  — Keep Documents for specific law_ids or loai_van_ban
5. LIGHTWEIGHT      — DOC_ONLY + strip all non-essential node attributes
6. DROP_HAS_CHUNK  — Drop only HAS_CHUNK edges + orphan Chunk nodes; keep DOC→ART and DOC↔DOC

Usage:
    python scripts/scale_down_graph.py --strategy DOC_NO_ISOLATES
    python scripts/scale_down_graph.py --strategy DOC_HUB --min-degree 5
    python scripts/scale_down_graph.py --strategy DOC_LAW_FILTER --law-ids "01/2024/QĐ-UBND,02/2024/QĐ-UBND"
    python scripts/scale_down_graph.py --strategy LIGHTWEIGHT
    python scripts/scale_down_graph.py --strategy DROP_HAS_CHUNK
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "data" / "kg.gpickle"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def load_graph(path: Path) -> nx.MultiDiGraph:
    print(f"Loading {path} ({path.stat().st_size / (1024**3):.2f} GB)...")
    with path.open("rb") as f:
        G = pickle.load(f)
    print(f"  Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def save_graph(G: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {path} ({size_mb:.2f} MB)")
    print(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")


def strategy_doc_only(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Keep only Document nodes and Document→Document edges."""
    doc_nodes = {n for n, d in G.nodes(data=True) if d.get("type") == "Document"}
    H = G.subgraph(doc_nodes).copy()
    return H


def strategy_doc_no_isolates(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """DOC_ONLY minus isolated (degree-0) nodes."""
    H = strategy_doc_only(G)
    isolated = [n for n in H.nodes() if H.degree(n) == 0]
    H.remove_nodes_from(isolated)
    print(f"  Removed {len(isolated):,} isolated Document nodes")
    return H


def strategy_doc_hub(G: nx.MultiDiGraph, min_degree: int) -> nx.MultiDiGraph:
    """Keep only Documents with total degree >= min_degree and their edges."""
    doc_nodes = {n for n, d in G.nodes(data=True) if d.get("type") == "Document"}
    hub_nodes = {n for n in doc_nodes if G.degree(n) >= min_degree}
    non_hubs = doc_nodes - hub_nodes
    print(f"  Hub nodes (degree >= {min_degree}): {len(hub_nodes):,}")
    print(f"  Dropped non-hub Documents: {len(non_hubs):,}")

    # Build subgraph from hub nodes only, but include edges between them
    H = G.subgraph(hub_nodes).copy()
    # Only keep Document→Document edges
    edges_to_remove = []
    for u, v, k in H.edges(keys=True):
        if H.nodes[u].get("type") != "Document" or H.nodes[v].get("type") != "Document":
            edges_to_remove.append((u, v, k))
    for u, v, k in edges_to_remove:
        H.remove_edge(u, v, k)
    return H


def strategy_doc_law_filter(
    G: nx.MultiDiGraph, law_ids: Optional[List[str]] = None, loai_list: Optional[List[str]] = None
) -> nx.MultiDiGraph:
    """Keep only Documents matching specific law_ids or loai_van_ban."""
    doc_nodes = set()
    for n, d in G.nodes(data=True):
        if d.get("type") != "Document":
            continue
        if law_ids and d.get("law_id") in law_ids:
            doc_nodes.add(n)
        elif loai_list and d.get("loai") in loai_list:
            doc_nodes.add(n)

    if not doc_nodes:
        print("  WARNING: No matching documents found!")
        return nx.MultiDiGraph()

    print(f"  Matching documents: {len(doc_nodes):,}")
    H = G.subgraph(doc_nodes).copy()
    # Remove non-Document nodes that may have been included
    non_doc = [n for n in H.nodes() if H.nodes[n].get("type") != "Document"]
    H.remove_nodes_from(non_doc)
    # Remove non-Document edges
    bad_edges = []
    for u, v, k in H.edges(keys=True):
        if H.nodes[u].get("type") != "Document" or H.nodes[v].get("type") != "Document":
            bad_edges.append((u, v, k))
    for u, v, k in bad_edges:
        H.remove_edge(u, v, k)
    return H


def strategy_lightweight(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """DOC_ONLY + strip all non-essential attributes, keeping only id and relation."""
    H = strategy_doc_only(G)

    # Keep only essential node attributes
    for n in H.nodes():
        attrs = dict(H.nodes[n])
        keep = {
            "type": attrs.get("type"),
            "doc_id": attrs.get("doc_id"),
            "law_id": attrs.get("law_id"),
            "ten": attrs.get("ten"),
            "loai": attrs.get("loai"),
            "nganh": attrs.get("nganh"),
            "ngay_ban_hanh": attrs.get("ngay_ban_hanh"),
            "tinh_trang_hieu_luc": attrs.get("tinh_trang_hieu_luc"),
        }
        keep = {k: v for k, v in keep.items() if v is not None}
        H.nodes[n].clear()
        H.nodes[n].update(keep)

    # Strip edge attributes to relation only
    for u, v, k in H.edges(keys=True):
        rel = H.edges[u, v, k].get("relation", "UNKNOWN")
        H.edges[u, v, k].clear()
        H.edges[u, v, k]["relation"] = rel

    return H


def strategy_drop_has_chunk(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Drop only HAS_CHUNK edges (ART→CHUNK) and orphaned Chunk nodes.

    Keeps the DOC→ART hierarchy, all DOC↔DOC cross-reference edges,
    and all node types except orphaned Chunks.
    """
    H = G.copy()

    # Remove all HAS_CHUNK edges
    has_chunk_edges = [
        (u, v, k) for u, v, k, d in H.edges(keys=True, data=True)
        if d.get("relation") == "HAS_CHUNK"
    ]
    H.remove_edges_from(has_chunk_edges)
    print(f"  Removed {len(has_chunk_edges):,} HAS_CHUNK edges")

    # Remove Chunk nodes (now orphaned)
    chunk_nodes = [n for n, d in H.nodes(data=True) if d.get("type") == "Chunk"]
    H.remove_nodes_from(chunk_nodes)
    print(f"  Removed {len(chunk_nodes):,} Chunk nodes")

    return H


def print_summary(G: nx.MultiDiGraph, label: str) -> None:
    """Print a summary of the graph."""
    from collections import Counter

    print(f"\n{'='*60}")
    print(f"  Strategy: {label}")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    type_counts = Counter(d.get("type", "?") for _, d in G.nodes(data=True))
    print(f"  Node types: {dict(type_counts)}")

    rel_counts = Counter(d.get("relation", "?") for _, _, _, d in G.edges(keys=True, data=True))
    print(f"  Edge types: {dict(rel_counts)}")

    # Connectivity
    if G.number_of_nodes() > 0:
        UG = nx.Graph(G)
        components = list(nx.connected_components(UG))
        comp_sizes = sorted([len(c) for c in components], reverse=True)
        isolated = sum(1 for n in G.nodes() if G.degree(n) == 0)
        print(f"  Components: {len(components):,}")
        print(f"  Largest component: {comp_sizes[0]:,}")
        print(f"  Isolated nodes: {isolated:,}")
        if isolated > 0:
            print(f"  Non-isolated nodes: {G.number_of_nodes() - isolated:,}")


def main():
    parser = argparse.ArgumentParser(description="Scale down kg.gpickle for graphDB hosting")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input gpickle path")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output gpickle path (auto-generated if not specified)")
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["DOC_ONLY", "DOC_NO_ISOLATES", "DOC_HUB", "DOC_LAW_FILTER", "LIGHTWEIGHT"],
                        help="Scale-down strategy")
    parser.add_argument("--min-degree", type=int, default=5,
                        help="Min degree for DOC_HUB strategy")
    parser.add_argument("--law-ids", type=str, default=None,
                        help="Comma-separated law_ids for DOC_LAW_FILTER")
    parser.add_argument("--loai", type=str, default=None,
                        help="Comma-separated loai_van_ban for DOC_LAW_FILTER")
    parser.add_argument("--print-only", action="store_true",
                        help="Print summary only, do not save")
    args = parser.parse_args()

    G = load_graph(args.input)
    print_summary(G, "ORIGINAL")

    if args.strategy == "DOC_ONLY":
        H = strategy_doc_only(G)
    elif args.strategy == "DOC_NO_ISOLATES":
        H = strategy_doc_no_isolates(G)
    elif args.strategy == "DOC_HUB":
        H = strategy_doc_hub(G, args.min_degree)
    elif args.strategy == "DOC_LAW_FILTER":
        law_ids = [x.strip() for x in args.law_ids.split(",")] if args.law_ids else None
        loai_list = [x.strip() for x in args.loai.split(",")] if args.loai else None
        H = strategy_doc_law_filter(G, law_ids=law_ids, loai_list=loai_list)
    elif args.strategy == "LIGHTWEIGHT":
        H = strategy_lightweight(G)
    else:
        print(f"Unknown strategy: {args.strategy}")
        return 1

    print_summary(H, args.strategy)

    if not args.print_only:
        if args.output:
            out_path = args.output
        else:
            out_path = DEFAULT_OUTPUT_DIR / f"kg_{args.strategy.lower()}.gpickle"
        save_graph(H, out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())