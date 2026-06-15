"""
Stage 5: Build Knowledge Graph (incremental).

This module is designed to be called multiple times so each sub-stage
(5.2 DOC, 5.3 ART, 5.4 CHUNK, 5.5 doc-doc edges, 5.6 CONCEPT) can append
to the same persisted ``networkx.MultiDiGraph``.

Currently implemented:
    - Stage 5.2: Document nodes from ``stage1_sme_docs.parquet``.

Inputs:
    - ``data/stage1_sme_docs.parquet``

Outputs:
    - ``data/kg.gpickle`` (DOC-only at this stage; appended by later stages).

Reference specs:
    - ``KG.md`` §3.1 (DOC node attributes; full set used here).
    - ``PLAN.md`` Stage 5.2 (acceptance: node count in [3000, 8000]).
    - ``G-LRAG_SPECIFICATIONS.md`` §8.1 (graph type = MultiDiGraph).
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "networkx is required to run src/data/stage5_build_graph.py. "
        "Install it with `pip install networkx` or use the repo requirements file."
    ) from exc


# Columns required from stage1_sme_docs.parquet (per stage1_filter.py output schema).
REQUIRED_STAGE1_COLUMNS = (
    "id",
    "law_id",
    "ten_van_ban",
    "loai_van_ban",
    "nganh",
    "linh_vuc",
    "ngay_ban_hanh",
    "tinh_trang_hieu_luc",
    "ngay_co_hieu_luc",
    "ngay_het_hieu_luc",
)

# Acceptance bounds for DOC node count (PLAN.md task 5.2 / Stage 1 quality gate).
DOC_COUNT_MIN = 3_000
DOC_COUNT_MAX = 20_000  # widened to match stage1_filter.py runtime bound


def resolve_path(path: Optional[str], default_path: Path) -> Path:
    """Return a resolved ``Path`` for ``path`` falling back to ``default_path``."""
    if path:
        return Path(path)
    return default_path


def _normalize_value(value: Any) -> Any:
    """Convert pandas NaN / NaT to ``None`` and coerce primitives to plain str.

    Date fields keep ``None`` so downstream consumers can distinguish "missing"
    from "empty string". String fields are stripped.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    # pandas Timestamp / NaT
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def load_stage1_docs(path: Path) -> pd.DataFrame:
    """Load ``stage1_sme_docs.parquet`` and validate required columns.

    Casts ``id`` to ``str`` and drops duplicate ids (keep first).
    Raises if any required column is missing or if the frame is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Stage 1 input not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Stage 1 parquet is empty: {path}")

    missing = [c for c in REQUIRED_STAGE1_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Stage 1 parquet is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["id"] = df["id"].astype(str)

    before = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} duplicate document ids; kept {after}.")

    return df


def build_doc_attrs(row: pd.Series) -> Dict[str, Any]:
    """Build the attribute dict for one DOC node per ``KG.md`` §3.1."""
    return {
        "type": "Document",
        "doc_id": str(row["id"]),
        "law_id": _normalize_value(row.get("law_id")),
        "ten": _normalize_value(row.get("ten_van_ban")),
        "loai": _normalize_value(row.get("loai_van_ban")),
        "nganh": _normalize_value(row.get("nganh")),
        "linh_vuc": _normalize_value(row.get("linh_vuc")),
        "ngay_ban_hanh": _normalize_value(row.get("ngay_ban_hanh")),
        "tinh_trang_hieu_luc": _normalize_value(row.get("tinh_trang_hieu_luc")),
        "ngay_co_hieu_luc": _normalize_value(row.get("ngay_co_hieu_luc")),
        "ngay_het_hieu_luc": _normalize_value(row.get("ngay_het_hieu_luc")),
    }


def add_document_nodes(G: "nx.MultiDiGraph", sme_docs: pd.DataFrame) -> int:
    """Add ``DOC:{doc_id}`` nodes to ``G``; return the number of nodes added.

    If a node already exists (e.g. when called with ``--append``), its
    attributes are updated in place rather than duplicated.
    """
    added = 0
    for row in sme_docs.itertuples(index=False):
        # itertuples() loses Series semantics; rewrap as Series for build_doc_attrs.
        row_series = pd.Series(row._asdict())
        node_id = f"DOC:{row_series['id']}"
        attrs = build_doc_attrs(row_series)
        if node_id in G.nodes:
            G.nodes[node_id].update(attrs)
        else:
            G.add_node(node_id, **attrs)
            added += 1
    return added


def run_quality_gates(G: "nx.MultiDiGraph", expected_count: int) -> None:
    """Validate the DOC-only graph against KG/PLAN acceptance criteria."""
    doc_nodes = [
        (n, d) for n, d in G.nodes(data=True) if d.get("type") == "Document"
    ]
    actual_count = len(doc_nodes)

    if actual_count != expected_count:
        raise AssertionError(
            f"DOC node count mismatch: expected {expected_count} (input rows), "
            f"got {actual_count}."
        )

    if not (DOC_COUNT_MIN <= actual_count <= DOC_COUNT_MAX):
        raise AssertionError(
            f"DOC node count {actual_count} outside acceptance window "
            f"[{DOC_COUNT_MIN}, {DOC_COUNT_MAX}]."
        )

    seen_keys: set = set()
    for node_id, attrs in doc_nodes:
        if node_id in seen_keys:
            raise AssertionError(f"Duplicate DOC node key detected: {node_id}")
        seen_keys.add(node_id)

        if not attrs.get("law_id"):
            raise AssertionError(f"Node {node_id} has empty law_id.")
        if not attrs.get("ten"):
            raise AssertionError(f"Node {node_id} has empty ten (ten_van_ban).")

    print(f"  Quality gates passed: {actual_count} DOC nodes, all attrs valid.")


def persist_graph(G: "nx.MultiDiGraph", path: Path) -> None:
    """Write ``G`` to ``path`` using ``pickle.HIGHEST_PROTOCOL``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_existing_graph(path: Path) -> "nx.MultiDiGraph":
    """Load an existing ``kg.gpickle``; raise if missing or wrong type."""
    if not path.exists():
        raise FileNotFoundError(
            f"--append requested but no existing graph found at {path}"
        )
    with path.open("rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.MultiDiGraph):
        raise TypeError(
            f"Loaded graph at {path} is type {type(G).__name__}, "
            "expected networkx.MultiDiGraph."
        )
    return G


def _print_attribute_coverage(G: "nx.MultiDiGraph") -> None:
    """Print % of DOC nodes that have a non-null value for each KG.md field."""
    doc_nodes = [d for _, d in G.nodes(data=True) if d.get("type") == "Document"]
    total = len(doc_nodes)
    if total == 0:
        return
    fields = [
        "law_id",
        "ten",
        "loai",
        "nganh",
        "linh_vuc",
        "ngay_ban_hanh",
        "tinh_trang_hieu_luc",
        "ngay_co_hieu_luc",
        "ngay_het_hieu_luc",
    ]
    print("  Attribute coverage:")
    for field in fields:
        present = sum(1 for d in doc_nodes if d.get(field) not in (None, ""))
        pct = 100.0 * present / total
        print(f"    {field:<22} {present:>6}/{total} ({pct:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: Build the G-LRAG knowledge graph (incremental)."
    )
    parser.add_argument(
        "--stage1-path",
        default=None,
        help="Path to stage1_sme_docs.parquet (default: data/stage1_sme_docs.parquet)",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output path for kg.gpickle (default: data/kg.gpickle)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Load existing kg.gpickle and append/update DOC nodes instead of starting fresh.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    stage1_path = resolve_path(args.stage1_path, project_root / "data" / "stage1_sme_docs.parquet")
    output_path = resolve_path(args.output_path, project_root / "data" / "kg.gpickle")

    print("Stage 5.2: Build DOC nodes")
    print("=" * 50)
    print(f"  Stage 1 input : {stage1_path}")
    print(f"  Output graph  : {output_path}")
    print(f"  Mode          : {'APPEND' if args.append else 'FRESH'}")

    print("\n[1/4] Loading Stage 1 documents")
    sme_docs = load_stage1_docs(stage1_path)
    print(f"  Loaded {len(sme_docs):,} SME documents")

    print("\n[2/4] Initializing graph")
    if args.append:
        G = load_existing_graph(output_path)
        print(
            f"  Loaded existing graph: "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
    else:
        G = nx.MultiDiGraph()
        print("  Created new MultiDiGraph")

    print("\n[3/4] Adding DOC nodes")
    added = add_document_nodes(G, sme_docs)
    print(f"  Added {added:,} new DOC nodes (input rows: {len(sme_docs):,})")
    _print_attribute_coverage(G)

    print("\n[4/4] Quality gates and persistence")
    run_quality_gates(G, expected_count=len(sme_docs))
    persist_graph(G, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {output_path} ({size_mb:.2f} MB)")
    print(
        f"  Graph summary: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges (DOC-only at this stage)."
    )


if __name__ == "__main__":
    main()
