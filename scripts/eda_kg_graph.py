#!/usr/bin/env python3
"""
EDA script for kg.gpickle — analyze graph structure to identify scaling strategies.

Usage:
    python scripts/eda_kg_graph.py [--graph data/kg.gpickle]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "kg.gpickle"


def load_graph(path: Path) -> nx.MultiDiGraph:
    """Load a NetworkX MultiDiGraph from a gpickle file."""
    print(f"Loading graph from {path} ({path.stat().st_size / (1024**3):.2f} GB)...")
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError(f"Expected MultiDiGraph, got {type(graph).__name__}")
    print(f"  Loaded in memory.")
    return graph


def analyze_graph(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Run comprehensive EDA and return a report dict."""

    report: Dict[str, Any] = {}

    # --- 1. Basic counts ---
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"\n=== Basic Counts ===")
    print(f"  Nodes: {n_nodes:,}")
    print(f"  Edges: {n_edges:,}")

    # --- 2. Node type distribution ---
    type_counts: Dict[str, int] = Counter()
    type_to_ids: Dict[str, List[str]] = defaultdict(list)
    for node_id, attrs in G.nodes(data=True):
        ntype = attrs.get("type", "Unknown")
        type_counts[ntype] += 1
        type_to_ids[ntype].append(node_id)

    print(f"\n=== Node Type Distribution ===")
    for ntype in sorted(type_counts.keys()):
        pct = 100.0 * type_counts[ntype] / n_nodes
        print(f"  {ntype:<20} {type_counts[ntype]:>12,}  ({pct:5.1f}%)")

    report["node_type_counts"] = dict(type_counts)
    report["total_nodes"] = n_nodes
    report["total_edges"] = n_edges

    # --- 3. Edge relation type distribution ---
    relation_counts: Dict[str, int] = Counter()
    for u, v, k, d in G.edges(keys=True, data=True):
        rel = d.get("relation", "UNKNOWN")
        relation_counts[rel] += 1

    print(f"\n=== Edge Relation Types ===")
    for rel in sorted(relation_counts.keys()):
        pct = 100.0 * relation_counts[rel] / n_edges
        print(f"  {rel:<25} {relation_counts[rel]:>12,}  ({pct:5.1f}%)")

    report["edge_relation_counts"] = dict(relation_counts)

    # --- 4. Degree analysis per node type ---
    print(f"\n=== Degree Distribution by Node Type ===")
    degree_by_type: Dict[str, Dict[str, float]] = {}
    for ntype in sorted(type_counts.keys()):
        ids = type_to_ids[ntype]
        in_degrees = [G.in_degree(n) for n in ids]
        out_degrees = [G.out_degree(n) for n in ids]
        total_degrees = [in_degrees[i] + out_degrees[i] for i in range(len(ids))]

        stats = {
            "count": len(ids),
            "in_degree_min": min(in_degrees) if in_degrees else 0,
            "in_degree_max": max(in_degrees) if in_degrees else 0,
            "in_degree_avg": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
            "out_degree_min": min(out_degrees) if out_degrees else 0,
            "out_degree_max": max(out_degrees) if out_degrees else 0,
            "out_degree_avg": sum(out_degrees) / len(out_degrees) if out_degrees else 0,
            "total_degree_avg": sum(total_degrees) / len(total_degrees) if total_degrees else 0,
        }

        # Count isolated nodes (total degree == 0)
        isolated = sum(1 for d in total_degrees if d == 0)
        stats["isolated_count"] = isolated
        stats["isolated_pct"] = 100.0 * isolated / len(ids) if ids else 0

        # Degree distribution histogram (log-scale buckets)
        degree_hist = Counter()
        for d in total_degrees:
            if d == 0:
                bucket = "0"
            elif d <= 5:
                bucket = "1-5"
            elif d <= 20:
                bucket = "6-20"
            elif d <= 100:
                bucket = "21-100"
            elif d <= 500:
                bucket = "101-500"
            elif d <= 2000:
                bucket = "501-2000"
            else:
                bucket = "2000+"
            degree_hist[bucket] += 1
        stats["degree_histogram"] = dict(degree_hist)

        degree_by_type[ntype] = stats

        print(f"\n  --- {ntype} ({len(ids):,} nodes) ---")
        print(f"    In-degree:  min={stats['in_degree_min']}, max={stats['in_degree_max']}, avg={stats['in_degree_avg']:.1f}")
        print(f"    Out-degree: min={stats['out_degree_min']}, max={stats['out_degree_max']}, avg={stats['out_degree_avg']:.1f}")
        print(f"    Isolated:   {isolated:,} ({stats['isolated_pct']:.1f}%)")
        print(f"    Degree histogram:")
        for bucket in ["0", "1-5", "6-20", "21-100", "101-500", "501-2000", "2000+"]:
            if bucket in degree_hist:
                print(f"      {bucket:>10}: {degree_hist[bucket]:>10,}")

    report["degree_by_type"] = degree_by_type

    # --- 5. Node attribute coverage ---
    print(f"\n=== Node Attribute Coverage ===")
    for ntype in sorted(type_counts.keys()):
        ids = type_to_ids[ntype]
        if not ids:
            continue
        sample_attrs = G.nodes[ids[0]]
        attr_keys = sorted(k for k in sample_attrs.keys() if k != "type")

        if not attr_keys:
            print(f"  {ntype}: no extra attributes")
            continue

        print(f"\n  --- {ntype} ---")
        for key in attr_keys:
            present = sum(1 for nid in ids if G.nodes[nid].get(key) not in (None, "", []))
            pct = 100.0 * present / len(ids)
            print(f"    {key:<30} {present:>10,}/{len(ids):,} ({pct:5.1f}%)")

    # --- 6. Edge attribute coverage ---
    print(f"\n=== Edge Attribute Coverage ===")
    # Sample first edge of each relation type to see attributes
    edge_attrs_seen: Dict[str, Set[str]] = defaultdict(set)
    for u, v, k, d in G.edges(keys=True, data=True):
        rel = d.get("relation", "UNKNOWN")
        edge_attrs_seen[rel].update(d.keys())
    for rel in sorted(edge_attrs_seen.keys()):
        print(f"  {rel}: {sorted(edge_attrs_seen[rel])}")

    # --- 7. Largest connected component analysis (undirected) ---
    print(f"\n=== Connected Component Analysis (undirected view) ===")
    UG = nx.Graph(G)  # collapse multi-edges for connectivity
    components = list(nx.connected_components(UG))
    comp_sizes = [len(c) for c in components]
    comp_sizes.sort(reverse=True)

    print(f"  Total connected components: {len(components):,}")
    print(f"  Largest component size:     {comp_sizes[0]:,}")
    print(f"  Second largest:             {comp_sizes[1] if len(comp_sizes) > 1 else 0:,}")
    print(f"  Top 10 component sizes:     {comp_sizes[:10]}")
    print(f"  Singleton components:       {sum(1 for s in comp_sizes if s == 1):,}")

    # Percentage of nodes in largest component
    pct_in_largest = 100.0 * comp_sizes[0] / n_nodes
    print(f"  % nodes in largest:         {pct_in_largest:.1f}%")

    report["components"] = {
        "total_components": len(components),
        "largest_size": comp_sizes[0],
        "singletons": sum(1 for s in comp_sizes if s == 1),
        "pct_in_largest": pct_in_largest,
        "top10_sizes": comp_sizes[:10],
    }

    # --- 8. Memory estimate ---
    print(f"\n=== Memory Footprint ===")
    # Approximate: count total attribute values
    total_attr_chars = 0
    for _, attrs in G.nodes(data=True):
        for v in attrs.values():
            if isinstance(v, str):
                total_attr_chars += len(v)
            elif isinstance(v, (list, dict)):
                total_attr_chars += len(json.dumps(v, default=str))
    for _, _, _, attrs in G.edges(keys=True, data=True):
        for v in attrs.values():
            if isinstance(v, str):
                total_attr_chars += len(v)

    report["total_attr_chars"] = total_attr_chars
    report["attr_mb_approx"] = total_attr_chars / (1024 * 1024)
    print(f"  Approx attribute data: {total_attr_chars / (1024 * 1024):.1f} MB (string/JSON)")

    # --- 9. Specific subgraph analyses ---

    # 9a. DOC nodes: how many unique law_ids, loai_van_ban
    doc_ids = type_to_ids.get("Document", [])
    if doc_ids:
        law_ids = set()
        loais = set()
        nganhs = set()
        for nid in doc_ids:
            attrs = G.nodes[nid]
            if attrs.get("law_id"):
                law_ids.add(attrs["law_id"])
            if attrs.get("loai"):
                loais.add(attrs["loai"])
            if attrs.get("nganh"):
                nganhs.add(attrs["nganh"])
        print(f"\n=== DOC Node Diversity ===")
        print(f"  Unique law_ids:       {len(law_ids):,}")
        print(f"  Unique loai_van_ban:  {len(loais):,}")
        print(f"  Unique nganh:         {len(nganhs):,}")

        # law_id distribution (top 10)
        law_id_counts = Counter()
        for nid in doc_ids:
            law_id_counts[G.nodes[nid].get("law_id", "?")] += 1
        print(f"  Top law_ids by doc count:")
        for lid, cnt in law_id_counts.most_common(10):
            print(f"    {lid}: {cnt:,}")

        report["doc_diversity"] = {
            "unique_law_ids": len(law_ids),
            "unique_loai": len(loais),
            "unique_nganh": len(nganhs),
        }

    # 9b. CHUNK nodes: per-doc distribution
    chunk_ids = type_to_ids.get("Chunk", [])
    if chunk_ids:
        chunks_per_doc = Counter()
        for nid in chunk_ids:
            chunks_per_doc[G.nodes[nid].get("doc_id", "?")] += 1

        chunk_counts = list(chunks_per_doc.values())
        chunk_counts.sort(reverse=True)
        print(f"\n=== CHUNK per DOC Distribution ===")
        print(f"  Docs with chunks:     {len(chunks_per_doc):,}")
        print(f"  Min chunks/doc:       {min(chunk_counts):,}")
        print(f"  Max chunks/doc:       {max(chunk_counts):,}")
        print(f"  Avg chunks/doc:       {sum(chunk_counts) / len(chunk_counts):.1f}")
        print(f"  Median chunks/doc:    {chunk_counts[len(chunk_counts)//2]:,}")
        print(f"  Top 10 docs by chunk count:")
        for doc_id, cnt in chunks_per_doc.most_common(10):
            print(f"    {doc_id}: {cnt:,} chunks")

        report["chunk_per_doc"] = {
            "docs_with_chunks": len(chunks_per_doc),
            "min": min(chunk_counts),
            "max": max(chunk_counts),
            "avg": sum(chunk_counts) / len(chunk_counts),
            "median": chunk_counts[len(chunk_counts)//2],
        }

    # 9c. Concept node analysis
    concept_ids = type_to_ids.get("Concept", [])
    if concept_ids:
        print(f"\n=== CONCEPT Node Analysis ===")
        for nid in sorted(concept_ids):
            attrs = G.nodes[nid]
            in_deg = G.in_degree(nid)
            out_deg = G.out_degree(nid)
            print(f"  {attrs.get('name', nid):<40} in={in_deg}, out={out_deg}")

    # 9d. Article per doc distribution
    art_ids = type_to_ids.get("Article", [])
    if art_ids:
        arts_per_doc = Counter()
        for nid in art_ids:
            arts_per_doc[G.nodes[nid].get("doc_id", "?")] += 1
        art_counts = list(arts_per_doc.values())
        art_counts.sort(reverse=True)
        print(f"\n=== Article per DOC Distribution ===")
        print(f"  Docs with articles:   {len(arts_per_doc):,}")
        print(f"  Min articles/doc:     {min(art_counts):,}")
        print(f"  Max articles/doc:     {max(art_counts):,}")
        print(f"  Avg articles/doc:     {sum(art_counts) / len(art_counts):.1f}")
        print(f"  Top 10 docs by article count:")
        for doc_id, cnt in arts_per_doc.most_common(10):
            print(f"    {doc_id}: {cnt:,} articles")

    # --- 10. Edge direction analysis ---
    print(f"\n=== Edge Direction Analysis ===")
    for rel in sorted(relation_counts.keys()):
        # Sample source/target types
        src_types = Counter()
        tgt_types = Counter()
        sample_count = 0
        for u, v, k, d in G.edges(keys=True, data=True):
            if d.get("relation") == rel:
                src_types[G.nodes[u].get("type", "?")] += 1
                tgt_types[G.nodes[v].get("type", "?")] += 1
                sample_count += 1
                if sample_count >= 5000:
                    break

        src_str = ", ".join(f"{t}:{c}" for t, c in src_types.most_common(3))
        tgt_str = ", ".join(f"{t}:{c}" for t, c in tgt_types.most_common(3))
        print(f"  {rel:<25} src→tgt types: [{src_str}] → [{tgt_str}]")

    return report


def main():
    parser = argparse.ArgumentParser(description="EDA for kg.gpickle knowledge graph")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH,
                        help="Path to kg.gpickle")
    parser.add_argument("--json", type=Path, default=None,
                        help="Save report as JSON")
    args = parser.parse_args()

    G = load_graph(args.graph)
    report = analyze_graph(G)

    if args.json:
        # Convert non-serializable types
        def convert(o):
            if isinstance(o, (set,)):
                return list(o)
            if isinstance(o, Counter):
                return dict(o)
            return o

        with args.json.open("w") as f:
            json.dump(report, f, default=convert, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.json}")


if __name__ == "__main__":
    main()