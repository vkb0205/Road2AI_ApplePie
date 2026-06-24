#!/usr/bin/env python3
"""
Upload the NetworkX gpickle knowledge graph into Neo4j -- pure Cypher (no APOC).

The uploader expects the graph produced by src/data/stage5_build_graph.py.
It maps the graph node `type` attribute into Neo4j labels:

    Document -> :Document
    Article  -> :Article
    Chunk    -> :Chunk
    Concept  -> :Concept

The stable graph key, e.g. `DOC:123` or `CHUNK:...`, is stored as `id`.
Every node also gets the `:GraphNode` base label for unified MATCH.

Every edge `relation` value is sanitized and used as the Neo4j relationship
type, for example `HAS_ARTICLE`, `MENTIONS`, `AMENDS`, `REPLACES`.

Because this script targets Neo4j instances WITHOUT APOC installed, it
groups nodes by label and edges by relationship type, then issues separate
MERGE queries per group with the label/type hardcoded in the Cypher string.

Usage:
    export NEO4J_URI="bolt://localhost:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="your_password"
    python upload_graph_to_neo4j.py

Or:
    python upload_graph_to_neo4j.py --graph Road2AI_ApplePie/data/kg.gpickle \\
        --uri bolt://localhost:7687 --user neo4j --password 'your_password'
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError("networkx is required; install it with `pip install networkx`") from exc

try:
    from neo4j import GraphDatabase
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "neo4j driver is required; install it with `pip install neo4j`"
    ) from exc


def load_dotenv(path: Path) -> None:
    """Load environment variables from a .env file."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and os.environ.get(key) is None:
                os.environ[key] = value


DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent / "data" / "kg.gpickle"
BATCH_SIZE = 500
NEO4J_VALUE_LIMIT = 15_000



def load_graph(path: Path) -> nx.MultiDiGraph:
    """Load a NetworkX MultiDiGraph from a gpickle file."""
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    with path.open("rb") as handle:
        graph = pickle.load(handle)

    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError(
            f"Loaded graph at {path} is type {type(graph).__name__}, "
            "expected networkx.MultiDiGraph."
        )

    return graph


def sanitize_identifier(value: Any, fallback: str) -> str:
    """Return a Neo4j-safe identifier for labels, relationship types, or keys."""
    text = str(value or fallback).strip()
    text = re.sub(r"[^0-9A-Za-z_]", "_", text)
    if not text:
        text = fallback
    if not re.match(r"^[A-Za-z_]", text):
        text = f"_{text}"
    return text


def neo4j_value(value: Any) -> Any:
    """Convert graph attributes into Neo4j-compatible scalar/list values."""
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float, str)):
        return value

    if hasattr(value, "item"):
        try:
            return neo4j_value(value.item())
        except Exception:
            pass

    if isinstance(value, Mapping):
        return {str(k): neo4j_value(v) for k, v in value.items()}

    try:
        iterable = list(value)
    except TypeError:
        return str(value)

    return [neo4j_value(item) for item in iterable]


def compact_attrs(attrs: Mapping[str, Any]) -> Dict[str, Any]:
    """Trim Neo4j property values to Neo4j's 16 KiB property limit."""
    result: Dict[str, Any] = {}
    for key, value in attrs.items():
        converted = neo4j_value(value)
        if converted is None:
            continue
        if isinstance(converted, str) and len(converted.encode("utf-8")) > NEO4J_VALUE_LIMIT:
            converted = converted[:NEO4J_VALUE_LIMIT]
        result[str(key)] = converted
    return result


def label_for_node(attrs: Mapping[str, Any]) -> str:
    """Return the primary Neo4j label for a graph node."""
    node_type = attrs.get("type") or "GraphNode"
    return sanitize_identifier(node_type, "GraphNode")


def relationship_type_for_edge(attrs: Mapping[str, Any]) -> str:
    """Return the Neo4j relationship type for a graph edge."""
    relation = attrs.get("relation") or attrs.get("key") or "RELATED"
    return sanitize_identifier(relation, "RELATED")


def collect_node_records(graph: nx.MultiDiGraph) -> Dict[str, List[Dict[str, Any]]]:
    """Collect all node records grouped by their Neo4j label.

    Returns a dict mapping label -> list of {id, props} records.
    """
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node_id, attrs in graph.nodes(data=True):
        record = compact_attrs(attrs)
        record["id"] = str(node_id)
        label = label_for_node(record)
        by_label[label].append({"id": record["id"], "props": record})
    return dict(by_label)


def collect_edge_records(graph: nx.MultiDiGraph) -> Dict[str, List[Dict[str, Any]]]:
    """Collect all edge records grouped by their relationship type.

    Returns a dict mapping relation_type -> list of {source_id, target_id, key, props} records.
    """
    by_rel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        relation_type = relationship_type_for_edge(attrs)
        record = compact_attrs(attrs)
        record["source_id"] = str(source)
        record["target_id"] = str(target)
        record["key"] = str(key)
        by_rel[relation_type].append({
            "source_id": record["source_id"],
            "target_id": record["target_id"],
            "key": record["key"],
            "props": record,
        })
    return dict(by_rel)


def chunked(items: List[Dict[str, Any]], size: int) -> Iterator[List[Dict[str, Any]]]:
    """Yield fixed-size chunks from a list."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def create_constraints(session: Any) -> None:
    """Create the node uniqueness constraint used by MERGE."""
    session.run(
        """
        CREATE CONSTRAINT graph_node_id_unique IF NOT EXISTS
        FOR (n:GraphNode)
        REQUIRE n.id IS UNIQUE
        """
    )


def merge_nodes_for_label(session: Any, label: str, records: List[Dict[str, Any]]) -> int:
    """Merge nodes of a single label into Neo4j using pure Cypher.

    The label is injected directly into the Cypher string (it's sanitized and
    comes from a known set, so injection risk is minimal).
    """
    if not records:
        return 0

    # Build Cypher with the label hardcoded
    cypher = f"""
    UNWIND $rows AS row
    MERGE (n:GraphNode:{label} {{id: row.id}})
    SET n += row.props
    RETURN count(n) AS merged
    """

    total = 0
    for batch in chunked(records, BATCH_SIZE):
        result = session.run(cypher, rows=batch)
        total += int(result.single()["merged"])
    return total


def merge_edges_for_type(session: Any, relation_type: str, records: List[Dict[str, Any]]) -> int:
    """Merge edges of a single relationship type into Neo4j using pure Cypher.

    The relationship type is injected directly into the Cypher string.
    """
    if not records:
        return 0

    # Build Cypher with the relationship type hardcoded
    cypher = f"""
    UNWIND $rows AS row
    MATCH (source:GraphNode {{id: row.source_id}})
    MATCH (target:GraphNode {{id: row.target_id}})
    MERGE (source)-[rel:{relation_type} {{key: row.key}}]->(target)
    SET rel += row.props
    RETURN count(rel) AS merged
    """

    total = 0
    for batch in chunked(records, BATCH_SIZE):
        result = session.run(cypher, rows=batch)
        total += int(result.single()["merged"])
    return total


def verify_neo4j(session: Any, graph: nx.MultiDiGraph) -> None:
    """Print Neo4j counts and compare them with graph counts."""
    node_count = session.run("MATCH (n:GraphNode) RETURN count(n) AS count").single()["count"]
    edge_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]

    graph_node_counts: Dict[str, int] = defaultdict(int)
    graph_edge_counts: Dict[str, int] = defaultdict(int)
    for _, attrs in graph.nodes(data=True):
        graph_node_counts[label_for_node(attrs)] += 1
    for _, _, _, attrs in graph.edges(keys=True, data=True):
        graph_edge_counts[relationship_type_for_edge(attrs)] += 1

    print(f"  Neo4j nodes:  {node_count:,}")
    print(f"  Neo4j edges:  {edge_count:,}")
    print(f"  Graph nodes:  {graph.number_of_nodes():,}")
    print(f"  Graph edges:  {graph.number_of_edges():,}")
    print("  Graph labels:")
    for label, count in sorted(graph_node_counts.items()):
        print(f"    :{label:<15} {count:>12,}")
    print("  Graph relationship types:")
    for relation_type, count in sorted(graph_edge_counts.items()):
        print(f"    :{relation_type:<15} {count:>12,}")

    # Per-label verification
    for label in sorted(graph_node_counts.keys()):
        neo4j_label_count = session.run(
            f"MATCH (n:{label}) RETURN count(n) AS count"
        ).single()["count"]
        if neo4j_label_count != graph_node_counts[label]:
            print(f"  WARNING: :{label} count mismatch: Neo4j={neo4j_label_count:,} vs Graph={graph_node_counts[label]:,}")

    if node_count != graph.number_of_nodes():
        raise AssertionError(
            f"Neo4j node count {node_count} does not match graph node count "
            f"{graph.number_of_nodes()}."
        )
    if edge_count != graph.number_of_edges():
        raise AssertionError(
            f"Neo4j edge count {edge_count} does not match graph edge count "
            f"{graph.number_of_edges()}."
        )


def upload_graph(
    graph_path: Path,
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
    batch_size: int = BATCH_SIZE,
    verify: bool = True,
) -> None:
    """Upload a NetworkX gpickle graph into Neo4j (pure Cypher, no APOC)."""
    global BATCH_SIZE
    BATCH_SIZE = batch_size

    graph = load_graph(graph_path)
    print(f"Loaded graph: {graph_path}")
    print(f"  Nodes: {graph.number_of_nodes():,}")
    print(f"  Edges: {graph.number_of_edges():,}")

    # Pre-collect and group all records
    print("Collecting node records by label...")
    nodes_by_label = collect_node_records(graph)
    for label, recs in nodes_by_label.items():
        print(f"  :{label}: {len(recs):,} nodes")

    print("Collecting edge records by relationship type...")
    edges_by_type = collect_edge_records(graph)
    for rel_type, recs in edges_by_type.items():
        print(f"  [:{rel_type}]: {len(recs):,} edges")

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        print(f"Connected to Neo4j at {uri}")

        with driver.session(database=database) as session:
            create_constraints(session)
            print("Constraint ensured (GraphNode.id unique).")

            # Upload nodes by label
            node_total = 0
            for label, records in sorted(nodes_by_label.items()):
                print(f"Uploading :{label} nodes ({len(records):,})...")
                merged = merge_nodes_for_label(session, label, records)
                node_total += merged
                print(f"  Merged :{label}: {merged:,}")

            print(f"All nodes merged: {node_total:,}")

            # Upload edges by relationship type
            edge_total = 0
            for rel_type, records in sorted(edges_by_type.items()):
                print(f"Uploading [:{rel_type}] edges ({len(records):,})...")
                merged = merge_edges_for_type(session, rel_type, records)
                edge_total += merged
                print(f"  Merged [:{rel_type}]: {merged:,}")

            print(f"All edges merged: {edge_total:,}")

            if verify:
                verify_neo4j(session, graph)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a NetworkX gpickle graph into Neo4j (pure Cypher, no APOC)")
    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH_PATH,
        help="Path to kg.gpickle (default: Road2AI_ApplePie/data/kg.gpickle)",
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="Neo4j Bolt URI (default: NEO4J_URI env var or bolt://localhost:7687)",
    )
    parser.add_argument(
        "--user",
        default=None,
        help="Neo4j username (default: NEO4J_USER env var or neo4j)",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Neo4j password (default: NEO4J_PASSWORD env var)",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Neo4j database name (default: server default database)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of nodes/edges to merge per batch (default: 500)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip final count verification",
    )
    args = parser.parse_args()

    # Auto-load .env from the script directory or the graph's parent directory.
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir / ".env", args.graph.resolve().parent / ".env"):
        load_dotenv(candidate)

    uri = args.uri or os.environ.get("NEO4J_URI") or "bolt://localhost:7687"
    user = (
        args.user
        or os.environ.get("NEO4J_USERNAME")
        or os.environ.get("NEO4J_USER")
        or "neo4j"
    )
    password = args.password or os.environ.get("NEO4J_PASSWORD")
    database = args.database or os.environ.get("NEO4J_DATABASE") or None

    if not password:
        print("ERROR: Provide --password or set NEO4J_PASSWORD.")
        return 1

    try:
        upload_graph(
            graph_path=args.graph,
            uri=uri,
            user=user,
            password=password,
            database=database,
            batch_size=args.batch_size,
            verify=not args.no_verify,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    print("Upload complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
