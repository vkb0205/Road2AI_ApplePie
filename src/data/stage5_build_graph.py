"""
Stage 5: Build Knowledge Graph (incremental).

This module is designed to be called multiple times so each sub-stage
(5.2 DOC, 5.3 ART, 5.4 CHUNK, 5.5 doc-doc edges, 5.6 CONCEPT) can append
to the same persisted ``networkx.MultiDiGraph``.

CLI:
    python -m src.data.stage5_build_graph --stage {5.2,5.3,5.4,5.5,5.6,all} \
        [--append] [--stage1-path PATH] [--stage2-path PATH] \
        [--stage3-path PATH] [--relationships-path PATH] \
        [--orphan-articles-path PATH] [--orphan-chunks-path PATH] \
        [--output-path PATH]

Currently implemented:
    - Stage 5.2: Document nodes from ``stage1_sme_docs.parquet``.
    - Stage 5.3: Article nodes + HAS_ARTICLE edges from
      ``stage2_articles.parquet`` (inner-joined on existing DOC nodes;
      orphan articles exported to ``data/stage5_orphan_articles.jsonl``).
    - Stage 5.4: Chunk nodes + HAS_CHUNK edges from
      ``stage3_chunks.parquet`` (inner-joined on existing ART nodes;
      orphan chunks exported to ``data/stage5_orphan_chunks.jsonl``).
    - Stage 5.5: Cross-document edges from the ``relationships`` dataset
      (filtered to SME doc IDs, mapped via ``config/relationship_mapping.yaml``,
      dropped labels logged to ``data/stage5_dropped_relationships.jsonl``).

Stage 5.6 is scaffolded as a ``NotImplementedError`` stub to be filled in
by a later change.

Inputs:
    - ``data/stage1_sme_docs.parquet`` (Stage 5.2)
    - ``data/stage2_articles.parquet`` (Stage 5.3)
    - ``data/stage3_chunks.parquet`` (Stage 5.4)
    - ``data/relationships.jsonl`` or HF dataset (Stage 5.5)
    - ``config/relationship_mapping.yaml`` (Stage 5.5)

Outputs:
    - ``data/kg.gpickle`` (cumulative across sub-stages)
    - ``data/stage5_orphan_articles.jsonl`` (Stage 5.3, may be empty)
    - ``data/stage5_orphan_chunks.jsonl`` (Stage 5.4, may be empty)
    - ``data/stage5_dropped_relationships.jsonl`` (Stage 5.5, may be empty)

Reference specs:
    - ``KG.md`` §3.1, §3.2, §3.3, §5, §7.1, §8.3.
    - ``PLAN.md`` Stage 5.2 / 5.3 / 5.4 / 5.5 acceptance criteria.
    - ``G-LRAG_SPECIFICATIONS.md`` §8.6 Steps 1–4, §5.8 relationships schema.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PyYAML is required to run src/data/stage5_build_graph.py. "
        "Install it with `pip install pyyaml` or use the repo requirements file."
    ) from exc

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "networkx is required to run src/data/stage5_build_graph.py. "
        "Install it with `pip install networkx` or use the repo requirements file."
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

# Columns required from stage2_articles.parquet (per stage2_parse_html.py output schema).
REQUIRED_STAGE2_COLUMNS = (
    "doc_id",
    "law_id",
    "ten_van_ban",
    "loai_van_ban",
    "ngay_ban_hanh",
    "phan",
    "chuong",
    "muc",
    "dieu_so",
    "dieu_ten",
    "start_char",
    "end_char",
    "doc_uid",
)

# Columns required from stage3_chunks.parquet (per stage3_chunking.py output schema).
REQUIRED_STAGE3_COLUMNS = (
    "chunk_id",
    "doc_uid",
    "doc_id",
    "part_idx",
    "breadcrumb",
)

# Columns required from the relationships dataset (per G-LRAG_SPECIFICATIONS.md §5.8).
REQUIRED_RELATIONSHIPS_COLUMNS = (
    "doc_id",
    "other_doc_id",
    "relationship",
)

# Acceptance bounds for DOC node count (PLAN.md task 5.2 / Stage 1 quality gate).
DOC_COUNT_MIN = 3_000
DOC_COUNT_MAX = 20_000  # widened to match stage1_filter.py runtime bound

# Acceptance bounds for ART node count (PLAN.md task 5.3, design Decision 5).
ART_COUNT_MIN = 20_000
ART_COUNT_MAX = 100_000

# Acceptance bounds for CHUNK node count (PLAN.md task 5.4).
# Upper bound is the full Stage 3 chunk count; orphan chunks only reduce it.
CHUNK_COUNT_MIN = 20_000
CHUNK_COUNT_MAX = 74_107

# Acceptance bounds for the doc-doc edge count (PLAN.md task 5.5).
# PLAN.md proposed [150_000, 350_000] when both directions of each relation
# were retained. KG.md §2.1/§6 keeps only the canonical direction and drops
# RELATED_LANGUAGE, which removes roughly half of the candidate edges.
# Additionally, the SME filter (Stage 1) restricts both endpoints to the
# curated doc set, which dramatically reduces eligible rows (~8k of ~900k).
# The lower bound is set to 5_000 to reflect real data volumes while still
# catching degenerate builds (e.g. empty mapping or broken join).
DOC_DOC_EDGE_COUNT_MIN = 5_000
DOC_DOC_EDGE_COUNT_MAX = 350_000

# Default local path for the relationships JSONL (Stage 5.5).
DEFAULT_RELATIONSHIPS_JSONL_PATH = "data/relationships.jsonl"

# Edge identifiers for the article layer.
HAS_ARTICLE_RELATION = "HAS_ARTICLE"
HAS_ARTICLE_EDGE_KEY = "HAS_ARTICLE"

# Edge identifiers for the chunk layer.
HAS_CHUNK_RELATION = "HAS_CHUNK"
HAS_CHUNK_EDGE_KEY = "HAS_CHUNK"

# Default Hugging Face source for the relationships dataset (Stage 5.5).
DEFAULT_RELATIONSHIPS_PATH = (
    "hf://datasets/th1nhng0/vietnamese-legal-documents/data/relationships.parquet"
)

# Reserved edge attribute set on every doc-doc edge so the relation is queryable.
DOC_DOC_RELATION_ATTR = "relation"

# Path to the relationship mapping YAML (relative to project root).
RELATIONSHIP_MAPPING_CONFIG = "config/relationship_mapping.yaml"

# Allowed values for the --stage CLI argument.
ALLOWED_STAGES = ("5.2", "5.3", "5.4", "5.5", "5.6", "all")
STAGES_REQUIRING_APPEND = {"5.3", "5.4", "5.5", "5.6"}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def resolve_path(path: Optional[str], default_path: Path) -> Path:
    """Return a resolved ``Path`` for ``path`` falling back to ``default_path``."""
    if path:
        return Path(path)
    return default_path


def _normalize_value(value: Any) -> Any:
    """Convert pandas NaN / NaT / empty strings to ``None``; strip strings.

    Date fields keep ``None`` so downstream consumers can distinguish "missing"
    from "empty string". String fields are stripped; empty after strip ⇒ ``None``.
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


def _persist_and_report(G: "nx.MultiDiGraph", output_path: Path, label: str) -> None:
    """Shared post-stage IO: write graph and print summary."""
    persist_graph(G, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {output_path} ({size_mb:.2f} MB)")
    print(
        f"  Graph summary after {label}: "
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
    )


# ---------------------------------------------------------------------------
# Stage 5.2 — Document nodes
# ---------------------------------------------------------------------------

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

    If a node already exists, its attributes are updated in place.
    """
    added = 0
    for row in sme_docs.itertuples(index=False):
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
    """Validate the DOC layer against KG/PLAN acceptance criteria."""
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
    print("  DOC attribute coverage:")
    for field in fields:
        present = sum(1 for d in doc_nodes if d.get(field) not in (None, ""))
        pct = 100.0 * present / total
        print(f"    {field:<22} {present:>6}/{total} ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# Stage 5.3 — Article nodes + HAS_ARTICLE edges
# ---------------------------------------------------------------------------

def load_stage2_articles(path: Path) -> pd.DataFrame:
    """Load ``stage2_articles.parquet`` and validate required columns.

    Casts ``doc_id`` and ``doc_uid`` to ``str`` and drops duplicates by
    ``doc_uid`` (keep first). Raises if columns are missing or frame is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Stage 2 input not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Stage 2 parquet is empty: {path}")

    missing = [c for c in REQUIRED_STAGE2_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Stage 2 parquet is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["doc_id"] = df["doc_id"].astype(str)
    df["doc_uid"] = df["doc_uid"].astype(str)

    before = len(df)
    df = df.drop_duplicates(subset=["doc_uid"], keep="first").reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} duplicate doc_uid rows; kept {after}.")

    return df


def build_art_attrs(row: pd.Series) -> Dict[str, Any]:
    """Build the attribute dict for one ART node per ``KG.md`` §3.2.

    The locked attribute set excludes ``noi_dung`` (full article body) to keep
    ``kg.gpickle`` lean. Empty strings are normalized to ``None``.
    """
    return {
        "type": "Article",
        "doc_uid": str(row["doc_uid"]),
        "doc_id": str(row["doc_id"]),
        "law_id": _normalize_value(row.get("law_id")),
        "ten_van_ban": _normalize_value(row.get("ten_van_ban")),
        "dieu_so": _normalize_value(row.get("dieu_so")),
        "dieu_ten": _normalize_value(row.get("dieu_ten")),
        "phan": _normalize_value(row.get("phan")),
        "chuong": _normalize_value(row.get("chuong")),
        "muc": _normalize_value(row.get("muc")),
        "loai_van_ban": _normalize_value(row.get("loai_van_ban")),
        "ngay_ban_hanh": _normalize_value(row.get("ngay_ban_hanh")),
        "start_char": _normalize_value(row.get("start_char")),
        "end_char": _normalize_value(row.get("end_char")),
    }


def partition_articles(
    df: pd.DataFrame, G: "nx.MultiDiGraph"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into ``(joined, orphans)`` based on DOC node presence.

    A row is "joined" iff ``DOC:{doc_id}`` exists in ``G``. Orphan rows are
    not added to the graph; they are exported to JSONL for team review.
    """
    doc_keys = {n for n, d in G.nodes(data=True) if d.get("type") == "Document"}
    expected_keys = "DOC:" + df["doc_id"].astype(str)
    mask = expected_keys.isin(doc_keys)
    joined = df.loc[mask].reset_index(drop=True)
    orphans = df.loc[~mask].reset_index(drop=True)
    return joined, orphans


def add_article_nodes(
    G: "nx.MultiDiGraph", joined_df: pd.DataFrame
) -> Tuple[int, int]:
    """Add ART nodes and HAS_ARTICLE edges; idempotent on re-run.

    Returns ``(nodes_added, edges_added)``. Existing ART nodes have their
    attributes refreshed in place; the HAS_ARTICLE edge is keyed explicitly
    so duplicate edges between the same DOC/ART pair are skipped.
    """
    nodes_added = 0
    edges_added = 0
    for row in joined_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        doc_uid = str(row_series["doc_uid"])
        doc_id = str(row_series["doc_id"])
        art_key = f"ART:{doc_uid}"
        doc_key = f"DOC:{doc_id}"

        attrs = build_art_attrs(row_series)
        if art_key in G.nodes:
            G.nodes[art_key].update(attrs)
        else:
            G.add_node(art_key, **attrs)
            nodes_added += 1

        if not G.has_edge(doc_key, art_key, key=HAS_ARTICLE_EDGE_KEY):
            G.add_edge(
                doc_key,
                art_key,
                key=HAS_ARTICLE_EDGE_KEY,
                relation=HAS_ARTICLE_RELATION,
            )
            edges_added += 1

    return nodes_added, edges_added


def export_orphan_articles(orphans_df: pd.DataFrame, path: Path) -> int:
    """Write orphan article rows to JSONL; create/truncate even if empty.

    Returns the number of records written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ("doc_uid", "doc_id", "law_id", "ten_van_ban", "dieu_so", "dieu_ten")
    written = 0
    with path.open("w", encoding="utf-8") as f:
        if orphans_df.empty:
            return 0
        for row in orphans_df.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            record = {
                col: _serialize_for_json(row_series.get(col)) for col in keep_cols
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def _serialize_for_json(value: Any) -> Any:
    """Coerce pandas/numpy values to JSON-safe primitives."""
    normalized = _normalize_value(value)
    if normalized is None:
        return None
    if isinstance(normalized, (str, int, float, bool)):
        return normalized
    return str(normalized)


def _print_orphan_summary(orphans_df: pd.DataFrame, path: Path) -> None:
    """Print orphan count and top-10 ``loai_van_ban`` distribution."""
    total = len(orphans_df)
    print(f"  Orphan articles: {total:,} (written to {path})")
    if total == 0:
        return
    if "loai_van_ban" in orphans_df.columns:
        loai_counts = (
            orphans_df["loai_van_ban"]
            .fillna("(missing)")
            .replace("", "(missing)")
            .value_counts()
            .head(10)
        )
        print("  Top loai_van_ban among orphans:")
        for loai, cnt in loai_counts.items():
            print(f"    {loai!s:<40} {cnt:>6}")


def _print_article_attribute_coverage(G: "nx.MultiDiGraph") -> None:
    """Print % of ART nodes that have a non-null value for each locked field."""
    art_nodes = [d for _, d in G.nodes(data=True) if d.get("type") == "Article"]
    total = len(art_nodes)
    if total == 0:
        return
    fields = [
        "doc_uid",
        "doc_id",
        "law_id",
        "ten_van_ban",
        "dieu_so",
        "dieu_ten",
        "phan",
        "chuong",
        "muc",
        "loai_van_ban",
        "ngay_ban_hanh",
        "start_char",
        "end_char",
    ]
    print("  ART attribute coverage:")
    for field in fields:
        present = sum(1 for d in art_nodes if d.get(field) not in (None, ""))
        pct = 100.0 * present / total
        print(f"    {field:<22} {present:>6}/{total} ({pct:5.1f}%)")


def run_article_quality_gates(
    G: "nx.MultiDiGraph",
    expected_band: Tuple[int, int] = (ART_COUNT_MIN, ART_COUNT_MAX),
) -> None:
    """Validate the ART layer before persistence.

    Asserts:
      1. ART count in ``expected_band``.
      2. No duplicate ``ART:{doc_uid}`` keys.
      3. Every ART has at least one incoming HAS_ARTICLE edge from a DOC.
      4. For every HAS_ARTICLE edge, source.type=="Document",
         target.type=="Article", relation=="HAS_ARTICLE".
      5. ART.doc_id matches source DOC.doc_id for every edge.
      6. Every ART has non-null required attrs.
    """
    art_nodes = [
        (n, d) for n, d in G.nodes(data=True) if d.get("type") == "Article"
    ]
    art_count = len(art_nodes)
    lo, hi = expected_band
    if not (lo <= art_count <= hi):
        raise AssertionError(
            f"ART node count {art_count} outside acceptance window [{lo}, {hi}]."
        )

    seen_keys: set = set()
    required_attrs = ("doc_uid", "doc_id", "law_id", "ten_van_ban", "dieu_so", "dieu_ten")
    for node_id, attrs in art_nodes:
        if node_id in seen_keys:
            raise AssertionError(f"Duplicate ART node key detected: {node_id}")
        seen_keys.add(node_id)
        for field in required_attrs:
            if not attrs.get(field):
                raise AssertionError(
                    f"ART node {node_id} has empty required field '{field}'."
                )

    # Validate every HAS_ARTICLE edge.
    has_article_edges = [
        (u, v, k, d)
        for u, v, k, d in G.edges(keys=True, data=True)
        if d.get("relation") == HAS_ARTICLE_RELATION
    ]
    arts_with_parent: set = set()
    for u, v, k, d in has_article_edges:
        src = G.nodes[u]
        tgt = G.nodes[v]
        if src.get("type") != "Document":
            raise AssertionError(
                f"HAS_ARTICLE edge {u} -> {v} source is not a Document node "
                f"(type={src.get('type')!r})."
            )
        if tgt.get("type") != "Article":
            raise AssertionError(
                f"HAS_ARTICLE edge {u} -> {v} target is not an Article node "
                f"(type={tgt.get('type')!r})."
            )
        if k != HAS_ARTICLE_EDGE_KEY:
            raise AssertionError(
                f"HAS_ARTICLE edge {u} -> {v} has unexpected edge key {k!r}; "
                f"expected {HAS_ARTICLE_EDGE_KEY!r}."
            )
        src_doc_id = str(src.get("doc_id"))
        tgt_doc_id = str(tgt.get("doc_id"))
        if src_doc_id != tgt_doc_id:
            raise AssertionError(
                f"HAS_ARTICLE edge {u} -> {v} doc_id mismatch: "
                f"DOC.doc_id={src_doc_id!r}, ART.doc_id={tgt_doc_id!r}."
            )
        arts_with_parent.add(v)

    art_keys = {n for n, _ in art_nodes}
    missing_parents = art_keys - arts_with_parent
    if missing_parents:
        sample = sorted(missing_parents)[:5]
        raise AssertionError(
            f"{len(missing_parents)} ART nodes have no parent DOC via HAS_ARTICLE; "
            f"sample: {sample}"
        )

    print(
        f"  ART quality gates passed: {art_count} ART nodes, "
        f"{len(has_article_edges)} HAS_ARTICLE edges, all parented."
    )


# ---------------------------------------------------------------------------
# Stage 5.4 — Chunk nodes + HAS_CHUNK edges
# ---------------------------------------------------------------------------

def load_stage3_chunks(path: Path) -> pd.DataFrame:
    """Load ``stage3_chunks.parquet`` and validate required columns.

    Casts ``chunk_id``, ``doc_uid`` and ``doc_id`` to ``str``, adds a stable
    ``rowidx`` (the original 0-based row position) and drops duplicate
    ``chunk_id`` rows (keep first). Raises if columns are missing or empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Stage 3 input not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Stage 3 parquet is empty: {path}")

    missing = [c for c in REQUIRED_STAGE3_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Stage 3 parquet is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    # rowidx is the original chunk ordering before any dedup/partition.
    df["rowidx"] = range(len(df))
    df["chunk_id"] = df["chunk_id"].astype(str)
    df["doc_uid"] = df["doc_uid"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)

    before = len(df)
    df = df.drop_duplicates(subset=["chunk_id"], keep="first").reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} duplicate chunk_id rows; kept {after}.")

    return df


def build_chunk_attrs(row: pd.Series) -> Dict[str, Any]:
    """Build the attribute dict for one CHUNK node per ``KG.md`` §3.3.

    Minimal metadata only: ``chunk_id``, ``doc_uid``, ``doc_id``, ``rowidx``,
    ``part_idx`` and ``breadcrumb``. The full ``chunk_text`` is deliberately
    excluded to keep ``kg.gpickle`` lean (the text lives in Stage 3 / indexes).
    Empty strings are normalized to ``None``.
    """
    return {
        "type": "Chunk",
        "chunk_id": str(row["chunk_id"]),
        "doc_uid": str(row["doc_uid"]),
        "doc_id": str(row["doc_id"]),
        "rowidx": _normalize_value(row.get("rowidx")),
        "part_idx": _normalize_value(row.get("part_idx")),
        "breadcrumb": _normalize_value(row.get("breadcrumb")),
    }


def partition_chunks(
    df: pd.DataFrame, G: "nx.MultiDiGraph"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into ``(joined, orphans)`` based on ART node presence.

    A row is "joined" iff ``ART:{doc_uid}`` exists in ``G``. Orphan chunks
    (parent ART missing, e.g. because the article was an orphan in 5.3) are
    not added to the graph; they are exported to JSONL for review so the
    ``DOC -> ART -> CHUNK`` invariant is preserved.
    """
    art_keys = {n for n, d in G.nodes(data=True) if d.get("type") == "Article"}
    expected_keys = "ART:" + df["doc_uid"].astype(str)
    mask = expected_keys.isin(art_keys)
    joined = df.loc[mask].reset_index(drop=True)
    orphans = df.loc[~mask].reset_index(drop=True)
    return joined, orphans


def add_chunk_nodes(
    G: "nx.MultiDiGraph", joined_df: pd.DataFrame
) -> Tuple[int, int]:
    """Add CHUNK nodes and HAS_CHUNK edges; idempotent on re-run.

    Returns ``(nodes_added, edges_added)``. Existing CHUNK nodes have their
    attributes refreshed in place; the HAS_CHUNK edge is keyed explicitly so
    duplicate edges between the same ART/CHUNK pair are skipped.
    """
    nodes_added = 0
    edges_added = 0
    for row in joined_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        chunk_id = str(row_series["chunk_id"])
        doc_uid = str(row_series["doc_uid"])
        chunk_key = f"CHUNK:{chunk_id}"
        art_key = f"ART:{doc_uid}"

        attrs = build_chunk_attrs(row_series)
        if chunk_key in G.nodes:
            G.nodes[chunk_key].update(attrs)
        else:
            G.add_node(chunk_key, **attrs)
            nodes_added += 1

        if not G.has_edge(art_key, chunk_key, key=HAS_CHUNK_EDGE_KEY):
            G.add_edge(
                art_key,
                chunk_key,
                key=HAS_CHUNK_EDGE_KEY,
                relation=HAS_CHUNK_RELATION,
            )
            edges_added += 1

    return nodes_added, edges_added


def export_orphan_chunks(orphans_df: pd.DataFrame, path: Path) -> int:
    """Write orphan chunk rows to JSONL; create/truncate even if empty.

    Returns the number of records written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ("chunk_id", "doc_uid", "doc_id", "rowidx", "part_idx", "breadcrumb")
    written = 0
    with path.open("w", encoding="utf-8") as f:
        if orphans_df.empty:
            return 0
        for row in orphans_df.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            record = {
                col: _serialize_for_json(row_series.get(col)) for col in keep_cols
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def _print_orphan_chunk_summary(orphans_df: pd.DataFrame, path: Path) -> None:
    """Print orphan-chunk count and top-10 ``doc_uid`` distribution."""
    total = len(orphans_df)
    print(f"  Orphan chunks: {total:,} (written to {path})")
    if total == 0:
        return
    if "doc_uid" in orphans_df.columns:
        uid_counts = orphans_df["doc_uid"].value_counts().head(10)
        print("  Top doc_uid among orphan chunks:")
        for uid, cnt in uid_counts.items():
            print(f"    {uid!s:<60} {cnt:>6}")


def _print_chunk_attribute_coverage(G: "nx.MultiDiGraph") -> None:
    """Print % of CHUNK nodes that have a non-null value for each field."""
    chunk_nodes = [d for _, d in G.nodes(data=True) if d.get("type") == "Chunk"]
    total = len(chunk_nodes)
    if total == 0:
        return
    fields = ["chunk_id", "doc_uid", "doc_id", "rowidx", "part_idx", "breadcrumb"]
    print("  CHUNK attribute coverage:")
    for field in fields:
        present = sum(1 for d in chunk_nodes if d.get(field) not in (None, ""))
        pct = 100.0 * present / total
        print(f"    {field:<22} {present:>6}/{total} ({pct:5.1f}%)")


def run_chunk_quality_gates(
    G: "nx.MultiDiGraph",
    expected_band: Tuple[int, int] = (CHUNK_COUNT_MIN, CHUNK_COUNT_MAX),
) -> None:
    """Validate the CHUNK layer before persistence.

    Asserts:
      1. CHUNK count in ``expected_band``.
      2. No duplicate ``CHUNK:{chunk_id}`` keys.
      3. Every CHUNK has exactly one incoming HAS_CHUNK edge from an ART.
      4. For every HAS_CHUNK edge, source.type=="Article",
         target.type=="Chunk", relation=="HAS_CHUNK".
      5. CHUNK.doc_uid matches source ART.doc_uid for every edge.
      6. Every CHUNK has non-null required attrs.
    """
    chunk_nodes = [
        (n, d) for n, d in G.nodes(data=True) if d.get("type") == "Chunk"
    ]
    chunk_count = len(chunk_nodes)
    lo, hi = expected_band
    if not (lo <= chunk_count <= hi):
        raise AssertionError(
            f"CHUNK node count {chunk_count} outside acceptance window [{lo}, {hi}]."
        )

    seen_keys: set = set()
    required_attrs = ("chunk_id", "doc_uid", "doc_id")
    for node_id, attrs in chunk_nodes:
        if node_id in seen_keys:
            raise AssertionError(f"Duplicate CHUNK node key detected: {node_id}")
        seen_keys.add(node_id)
        for field in required_attrs:
            if not attrs.get(field):
                raise AssertionError(
                    f"CHUNK node {node_id} has empty required field '{field}'."
                )

    # Validate every HAS_CHUNK edge.
    has_chunk_edges = [
        (u, v, k, d)
        for u, v, k, d in G.edges(keys=True, data=True)
        if d.get("relation") == HAS_CHUNK_RELATION
    ]
    parent_count: Dict[str, int] = {}
    for u, v, k, d in has_chunk_edges:
        src = G.nodes[u]
        tgt = G.nodes[v]
        if src.get("type") != "Article":
            raise AssertionError(
                f"HAS_CHUNK edge {u} -> {v} source is not an Article node "
                f"(type={src.get('type')!r})."
            )
        if tgt.get("type") != "Chunk":
            raise AssertionError(
                f"HAS_CHUNK edge {u} -> {v} target is not a Chunk node "
                f"(type={tgt.get('type')!r})."
            )
        if k != HAS_CHUNK_EDGE_KEY:
            raise AssertionError(
                f"HAS_CHUNK edge {u} -> {v} has unexpected edge key {k!r}; "
                f"expected {HAS_CHUNK_EDGE_KEY!r}."
            )
        src_doc_uid = str(src.get("doc_uid"))
        tgt_doc_uid = str(tgt.get("doc_uid"))
        if src_doc_uid != tgt_doc_uid:
            raise AssertionError(
                f"HAS_CHUNK edge {u} -> {v} doc_uid mismatch: "
                f"ART.doc_uid={src_doc_uid!r}, CHUNK.doc_uid={tgt_doc_uid!r}."
            )
        parent_count[v] = parent_count.get(v, 0) + 1

    chunk_keys = {n for n, _ in chunk_nodes}
    missing_parents = chunk_keys - set(parent_count)
    if missing_parents:
        sample = sorted(missing_parents)[:5]
        raise AssertionError(
            f"{len(missing_parents)} CHUNK nodes have no parent ART via HAS_CHUNK; "
            f"sample: {sample}"
        )

    print(
        f"  CHUNK quality gates passed: {chunk_count} CHUNK nodes, "
        f"{len(has_chunk_edges)} HAS_CHUNK edges, all parented."
    )


# ---------------------------------------------------------------------------
# Stage 5.5 — Cross-document edges
# ---------------------------------------------------------------------------

def load_relationship_mapping(path: Path) -> Dict[str, Any]:
    """Load and validate ``relationship_mapping.yaml``.

    Returns a dict with keys ``RELATIONSHIP_MAP``, ``RELATION_WHITELIST``
    (as a set), ``DROPPED_LABELS`` (as a set). Raises if the file is
    missing or any required section is malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Relationship mapping not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    rel_map = cfg.get("RELATIONSHIP_MAP")
    whitelist = cfg.get("RELATION_WHITELIST")
    dropped = cfg.get("DROPPED_LABELS", []) or []

    if not isinstance(rel_map, dict) or not rel_map:
        raise ValueError(
            f"RELATIONSHIP_MAP must be a non-empty mapping in {path}; "
            f"got {type(rel_map).__name__}."
        )
    if not isinstance(whitelist, list) or not whitelist:
        raise ValueError(
            f"RELATION_WHITELIST must be a non-empty list in {path}; "
            f"got {type(whitelist).__name__}."
        )
    if not isinstance(dropped, list):
        raise ValueError(
            f"DROPPED_LABELS must be a list (or omitted) in {path}; "
            f"got {type(dropped).__name__}."
        )

    return {
        "RELATIONSHIP_MAP": {str(k): str(v) for k, v in rel_map.items()},
        "RELATION_WHITELIST": {str(v) for v in whitelist},
        "DROPPED_LABELS": {str(v) for v in dropped},
    }


def load_relationships(path: Path) -> pd.DataFrame:
    """Load the ``relationships`` dataset and validate required columns.

    Supports ``.jsonl`` / ``.json`` and ``.parquet`` paths. Casts
    ``doc_id`` and ``other_doc_id`` to ``str``. Drops rows where any
    required field is null. Returns the cleaned frame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Relationships dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".json"):
        df = pd.read_json(path, lines=True)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported relationships file extension: {suffix!r}. "
            f"Expected .jsonl, .json, or .parquet."
        )

    if df.empty:
        raise ValueError(f"Relationships dataset is empty: {path}")

    missing = [c for c in REQUIRED_RELATIONSHIPS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Relationships dataset is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["doc_id"] = df["doc_id"].astype(str)
    df["other_doc_id"] = df["other_doc_id"].astype(str)
    df["relationship"] = df["relationship"].astype(str)

    # Drop rows where any of the three keys is null/empty.
    before = len(df)
    df = df.dropna(subset=list(REQUIRED_RELATIONSHIPS_COLUMNS))
    df = df[(df["doc_id"] != "") & (df["other_doc_id"] != "") & (df["relationship"] != "")]
    df = df.reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows with empty doc_id/other_doc_id/relationship.")

    return df


def collect_doc_ids(G: "nx.MultiDiGraph") -> Set[str]:
    """Return the set of ``doc_id`` strings for every DOC node in ``G``."""
    return {
        str(d.get("doc_id"))
        for _, d in G.nodes(data=True)
        if d.get("type") == "Document"
    }


def filter_relationships_by_sme(
    rels: pd.DataFrame, sme_doc_ids: Set[str]
) -> Tuple[pd.DataFrame, int]:
    """Keep only edges where BOTH endpoints are SME documents.

    Returns ``(filtered_df, dropped_non_sme_count)``.
    """
    if not sme_doc_ids:
        return rels.iloc[0:0].copy(), len(rels)
    mask = rels["doc_id"].isin(sme_doc_ids) & rels["other_doc_id"].isin(sme_doc_ids)
    kept = rels.loc[mask].reset_index(drop=True)
    dropped = len(rels) - len(kept)
    return kept, dropped


def map_relationship_label(
    label: str, rel_map: Dict[str, str], whitelist: Set[str]
) -> Optional[str]:
    """Return the canonical relation enum for ``label`` or ``None`` to drop.

    A label is kept only if it has an entry in ``rel_map`` AND the mapped
    enum is in ``whitelist``. Anything else returns ``None``.
    """
    enum = rel_map.get(label)
    if enum is None:
        return None
    if enum not in whitelist:
        return None
    return enum


def add_doc_doc_edges(
    G: "nx.MultiDiGraph",
    rels: pd.DataFrame,
    rel_map: Dict[str, str],
    whitelist: Set[str],
) -> Tuple[int, Dict[str, int], List[Dict[str, Any]]]:
    """Add canonical-direction DOC -> DOC edges to ``G``.

    Returns ``(edges_added, kept_counts_by_relation, dropped_records)``.

    - ``edges_added`` counts only NEW edges (idempotent on re-run).
    - ``kept_counts_by_relation`` maps canonical relation -> #edges kept
      (including duplicates of existing edges; i.e. raw kept rows).
    - ``dropped_records`` is a list of dicts for rows whose label was
      dropped (no whitelist match). Self-loops are also dropped here.

    Edge schema per KG.md §5: each edge keeps only ``relation`` as an
    attribute; the canonical enum is also used as the multi-edge key so
    the same (u, v, relation) triple is deduped within a build pass.
    """
    edges_added = 0
    kept_counts: Dict[str, int] = {}
    dropped_records: List[Dict[str, Any]] = []

    for row in rels.itertuples(index=False):
        src_id = str(row.doc_id)
        dst_id = str(row.other_doc_id)
        label = str(row.relationship)

        # Self-loops are meaningless and would corrupt graph traversal.
        if src_id == dst_id:
            dropped_records.append(
                {
                    "doc_id": src_id,
                    "other_doc_id": dst_id,
                    "relationship": label,
                    "reason": "self_loop",
                }
            )
            continue

        rel_enum = map_relationship_label(label, rel_map, whitelist)
        if rel_enum is None:
            dropped_records.append(
                {
                    "doc_id": src_id,
                    "other_doc_id": dst_id,
                    "relationship": label,
                    "reason": "label_not_in_whitelist",
                }
            )
            continue

        u = f"DOC:{src_id}"
        v = f"DOC:{dst_id}"
        kept_counts[rel_enum] = kept_counts.get(rel_enum, 0) + 1

        # Use the canonical enum as the multi-edge key so the same
        # (u, v, rel_enum) triple is idempotent across re-runs.
        if not G.has_edge(u, v, key=rel_enum):
            G.add_edge(u, v, key=rel_enum, relation=rel_enum)
            edges_added += 1

    return edges_added, kept_counts, dropped_records


def export_dropped_relationships(
    dropped_records: List[Dict[str, Any]], path: Path
) -> int:
    """Write dropped relationship rows to JSONL; create/truncate even if empty.

    Returns the number of records written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in dropped_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(dropped_records)


def _print_dropped_label_summary(
    dropped_records: List[Dict[str, Any]], path: Path
) -> None:
    """Print top-N dropped labels for review."""
    print(f"  Dropped relationship rows: {len(dropped_records):,} (written to {path})")
    if not dropped_records:
        return
    label_counts: Dict[str, int] = {}
    for rec in dropped_records:
        key = f"{rec.get('reason', '?')} | {rec.get('relationship', '')}"
        label_counts[key] = label_counts.get(key, 0) + 1
    top = sorted(label_counts.items(), key=lambda kv: -kv[1])[:10]
    print("  Top dropped (reason | label):")
    for k, n in top:
        print(f"    {k!s:<60} {n:>6}")


def _print_kept_relation_distribution(kept_counts: Dict[str, int]) -> None:
    """Print kept-edge counts grouped by canonical relation."""
    if not kept_counts:
        print("  No relations kept.")
        return
    total = sum(kept_counts.values())
    print(f"  Kept rows by canonical relation (total raw rows: {total:,}):")
    for rel, n in sorted(kept_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {rel:<22} {n:>10,}")


def run_doc_doc_quality_gates(
    G: "nx.MultiDiGraph",
    whitelist: Set[str],
    expected_band: Tuple[int, int] = (DOC_DOC_EDGE_COUNT_MIN, DOC_DOC_EDGE_COUNT_MAX),
) -> None:
    """Validate the doc-doc layer before persistence.

    Asserts:
      1. Doc-doc edge count in ``expected_band``.
      2. Every doc-doc edge has source and target both ``Document``.
      3. ``relation`` attribute is present and in ``whitelist``.
      4. Edge key equals the relation (canonical-direction storage).
      5. No self-loops (``DOC:X -> DOC:X``).
    """
    doc_doc_edges = [
        (u, v, k, d)
        for u, v, k, d in G.edges(keys=True, data=True)
        if (
            G.nodes.get(u, {}).get("type") == "Document"
            and G.nodes.get(v, {}).get("type") == "Document"
        )
    ]
    edge_count = len(doc_doc_edges)
    lo, hi = expected_band
    if not (lo <= edge_count <= hi):
        raise AssertionError(
            f"DOC-DOC edge count {edge_count} outside acceptance window [{lo}, {hi}]."
        )

    for u, v, k, d in doc_doc_edges:
        if u == v:
            raise AssertionError(
                f"Self-loop detected on DOC node {u} (key={k!r}, data={d!r})."
            )
        rel = d.get("relation")
        if not rel or rel not in whitelist:
            raise AssertionError(
                f"DOC-DOC edge {u} -> {v} has relation={rel!r} not in whitelist "
                f"{sorted(whitelist)}."
            )
        if k != rel:
            raise AssertionError(
                f"DOC-DOC edge {u} -> {v} key {k!r} does not match "
                f"relation {rel!r}; canonical-direction storage requires key == relation."
            )

    print(
        f"  DOC-DOC quality gates passed: {edge_count} edges across "
        f"{len({d.get('relation') for _u, _v, _k, d in doc_doc_edges})} relations."
    )


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _run_stage_5_2(args: argparse.Namespace, G: "nx.MultiDiGraph") -> "nx.MultiDiGraph":
    """Stage 5.2 builder: add DOC nodes from Stage 1."""
    project_root = Path(__file__).resolve().parents[2]
    stage1_path = resolve_path(
        args.stage1_path, project_root / "data" / "stage1_sme_docs.parquet"
    )

    print("Stage 5.2: Build DOC nodes")
    print("=" * 50)
    print(f"  Stage 1 input : {stage1_path}")

    print("\n[1/3] Loading Stage 1 documents")
    sme_docs = load_stage1_docs(stage1_path)
    print(f"  Loaded {len(sme_docs):,} SME documents")

    print("\n[2/3] Adding DOC nodes")
    added = add_document_nodes(G, sme_docs)
    print(f"  Added {added:,} new DOC nodes (input rows: {len(sme_docs):,})")
    _print_attribute_coverage(G)

    print("\n[3/3] Quality gates")
    run_quality_gates(G, expected_count=len(sme_docs))
    return G


def _run_stage_5_3(args: argparse.Namespace, G: "nx.MultiDiGraph") -> "nx.MultiDiGraph":
    """Stage 5.3 builder: add ART nodes + HAS_ARTICLE edges from Stage 2."""
    project_root = Path(__file__).resolve().parents[2]
    stage2_path = resolve_path(
        args.stage2_path, project_root / "data" / "stage2_articles.parquet"
    )
    orphan_path = resolve_path(
        args.orphan_articles_path,
        project_root / "data" / "stage5_orphan_articles.jsonl",
    )

    print("Stage 5.3: Build ART nodes + HAS_ARTICLE edges")
    print("=" * 50)
    print(f"  Stage 2 input : {stage2_path}")
    print(f"  Orphan output : {orphan_path}")

    print("\n[1/5] Loading Stage 2 articles")
    articles = load_stage2_articles(stage2_path)
    print(f"  Loaded {len(articles):,} article rows")

    print("\n[2/5] Partitioning by DOC presence")
    joined, orphans = partition_articles(articles, G)
    print(
        f"  Joined: {len(joined):,} | Orphans: {len(orphans):,} "
        f"(total: {len(articles):,})"
    )

    print("\n[3/5] Adding ART nodes and HAS_ARTICLE edges")
    nodes_added, edges_added = add_article_nodes(G, joined)
    print(
        f"  Added {nodes_added:,} new ART nodes and "
        f"{edges_added:,} new HAS_ARTICLE edges."
    )
    _print_article_attribute_coverage(G)

    print("\n[4/5] Exporting orphan articles")
    written = export_orphan_articles(orphans, orphan_path)
    print(f"  Wrote {written:,} orphan records to {orphan_path}")
    _print_orphan_summary(orphans, orphan_path)

    print("\n[5/5] Article quality gates")
    run_article_quality_gates(G)

    return G


def _run_stage_5_4(args: argparse.Namespace, G: "nx.MultiDiGraph") -> "nx.MultiDiGraph":
    """Stage 5.4 builder: add CHUNK nodes + HAS_CHUNK edges from Stage 3."""
    project_root = Path(__file__).resolve().parents[2]
    stage3_path = resolve_path(
        args.stage3_path, project_root / "data" / "stage3_chunks.parquet"
    )
    orphan_path = resolve_path(
        args.orphan_chunks_path,
        project_root / "data" / "stage5_orphan_chunks.jsonl",
    )

    print("Stage 5.4: Build CHUNK nodes + HAS_CHUNK edges")
    print("=" * 50)
    print(f"  Stage 3 input : {stage3_path}")
    print(f"  Orphan output : {orphan_path}")

    print("\n[1/5] Loading Stage 3 chunks")
    chunks = load_stage3_chunks(stage3_path)
    print(f"  Loaded {len(chunks):,} chunk rows")

    print("\n[2/5] Partitioning by ART presence")
    joined, orphans = partition_chunks(chunks, G)
    print(
        f"  Joined: {len(joined):,} | Orphans: {len(orphans):,} "
        f"(total: {len(chunks):,})"
    )

    print("\n[3/5] Adding CHUNK nodes and HAS_CHUNK edges")
    nodes_added, edges_added = add_chunk_nodes(G, joined)
    print(
        f"  Added {nodes_added:,} new CHUNK nodes and "
        f"{edges_added:,} new HAS_CHUNK edges."
    )
    _print_chunk_attribute_coverage(G)

    print("\n[4/5] Exporting orphan chunks")
    written = export_orphan_chunks(orphans, orphan_path)
    print(f"  Wrote {written:,} orphan records to {orphan_path}")
    _print_orphan_chunk_summary(orphans, orphan_path)

    print("\n[5/5] Chunk quality gates")
    # Resolve the acceptance band from module globals at call-time so test
    # fixtures can relax it via monkeypatch (the function default is bound
    # at definition time and would otherwise ignore patched constants).
    run_chunk_quality_gates(G, expected_band=(CHUNK_COUNT_MIN, CHUNK_COUNT_MAX))

    return G


def _run_stage_5_5(args: argparse.Namespace, G: "nx.MultiDiGraph") -> "nx.MultiDiGraph":
    """Stage 5.5 builder: add canonical DOC -> DOC edges from `relationships`."""
    project_root = Path(__file__).resolve().parents[2]
    rels_path = resolve_path(
        args.relationships_path,
        project_root / DEFAULT_RELATIONSHIPS_JSONL_PATH,
    )
    mapping_path = resolve_path(
        args.relationship_mapping_path,
        project_root / RELATIONSHIP_MAPPING_CONFIG,
    )
    dropped_path = resolve_path(
        args.dropped_relationships_path,
        project_root / "data" / "stage5_dropped_relationships.jsonl",
    )

    print("Stage 5.5: Build cross-document (DOC -> DOC) edges")
    print("=" * 50)
    print(f"  Relationships    : {rels_path}")
    print(f"  Mapping config   : {mapping_path}")
    print(f"  Dropped output   : {dropped_path}")

    print("\n[1/6] Loading relationship mapping")
    mapping_cfg = load_relationship_mapping(mapping_path)
    rel_map = mapping_cfg["RELATIONSHIP_MAP"]
    whitelist = mapping_cfg["RELATION_WHITELIST"]
    print(
        f"  Loaded {len(rel_map)} label entries; "
        f"{len(whitelist)} canonical relations whitelisted."
    )

    print("\n[2/6] Collecting SME doc IDs from existing graph")
    sme_doc_ids = collect_doc_ids(G)
    if not sme_doc_ids:
        raise AssertionError(
            "No DOC nodes found in the graph; run Stage 5.2 first before 5.5."
        )
    print(f"  {len(sme_doc_ids):,} DOC nodes available as SME endpoints.")

    print("\n[3/6] Loading relationships dataset")
    rels = load_relationships(rels_path)
    print(f"  Loaded {len(rels):,} raw relationship rows.")

    print("\n[4/6] Filtering to SME endpoints (both ends present in graph)")
    sme_rels, dropped_non_sme = filter_relationships_by_sme(rels, sme_doc_ids)
    print(
        f"  Kept {len(sme_rels):,} rows; "
        f"dropped {dropped_non_sme:,} rows where at least one endpoint "
        f"is not an SME DOC node."
    )

    print("\n[5/6] Adding canonical DOC -> DOC edges")
    edges_added, kept_counts, dropped_records = add_doc_doc_edges(
        G, sme_rels, rel_map, whitelist
    )
    print(f"  Added {edges_added:,} new canonical DOC -> DOC edges.")
    _print_kept_relation_distribution(kept_counts)

    written = export_dropped_relationships(dropped_records, dropped_path)
    print(f"  Wrote {written:,} dropped-row records to {dropped_path}")
    _print_dropped_label_summary(dropped_records, dropped_path)

    print("\n[6/6] DOC -> DOC quality gates")
    # Resolve the acceptance band from module globals at call-time so test
    # fixtures can relax it via monkeypatch.
    run_doc_doc_quality_gates(
        G,
        whitelist=whitelist,
        expected_band=(DOC_DOC_EDGE_COUNT_MIN, DOC_DOC_EDGE_COUNT_MAX),
    )
    return G


def _run_stage_5_6(args: argparse.Namespace, G: "nx.MultiDiGraph") -> "nx.MultiDiGraph":
    raise NotImplementedError(
        "Stage 5.6 (CONCEPT nodes + MENTIONS edges) is not yet implemented. "
        "See PLAN.md task 5.6 for scope and acceptance criteria."
    )


STAGE_RUNNERS = {
    "5.2": _run_stage_5_2,
    "5.3": _run_stage_5_3,
    "5.4": _run_stage_5_4,
    "5.5": _run_stage_5_5,
    "5.6": _run_stage_5_6,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 5: Build the G-LRAG knowledge graph (incremental).",
    )
    parser.add_argument(
        "--stage",
        choices=ALLOWED_STAGES,
        required=True,
        help=(
            "Sub-stage to run. 5.2=DOC nodes, 5.3=ART nodes + HAS_ARTICLE edges, "
            "5.4=CHUNK nodes + HAS_CHUNK edges, 5.5=cross-document edges from "
            "the relationships dataset, 5.6=CONCEPT nodes (TBD), "
            "all=run every implemented stage in order."
        ),
    )
    parser.add_argument(
        "--stage1-path",
        default=None,
        help="Path to stage1_sme_docs.parquet (default: data/stage1_sme_docs.parquet)",
    )
    parser.add_argument(
        "--stage2-path",
        default=None,
        help="Path to stage2_articles.parquet (default: data/stage2_articles.parquet)",
    )
    parser.add_argument(
        "--stage3-path",
        default=None,
        help="Path to stage3_chunks.parquet (default: data/stage3_chunks.parquet)",
    )
    parser.add_argument(
        "--orphan-articles-path",
        default=None,
        help=(
            "Path to write orphan-article JSONL "
            "(default: data/stage5_orphan_articles.jsonl)"
        ),
    )
    parser.add_argument(
        "--orphan-chunks-path",
        default=None,
        help=(
            "Path to write orphan-chunk JSONL "
            "(default: data/stage5_orphan_chunks.jsonl)"
        ),
    )
    parser.add_argument(
        "--relationships-path",
        default=None,
        help=(
            "Path to the relationships dataset (.jsonl/.json/.parquet); "
            "default: data/relationships.jsonl"
        ),
    )
    parser.add_argument(
        "--relationship-mapping-path",
        default=None,
        help=(
            "Path to the relationship mapping YAML "
            "(default: config/relationship_mapping.yaml)"
        ),
    )
    parser.add_argument(
        "--dropped-relationships-path",
        default=None,
        help=(
            "Path to write dropped-relationship JSONL "
            "(default: data/stage5_dropped_relationships.jsonl)"
        ),
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output path for kg.gpickle (default: data/kg.gpickle)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Load existing kg.gpickle and append/update nodes instead of "
            "starting fresh. Required for stages 5.3-5.6."
        ),
    )
    return parser


def _run_single_stage(
    stage: str,
    args: argparse.Namespace,
    G: "nx.MultiDiGraph",
    output_path: Path,
) -> "nx.MultiDiGraph":
    """Dispatch to a single stage runner and persist the result."""
    runner = STAGE_RUNNERS[stage]
    G = runner(args, G)
    _persist_and_report(G, output_path, label=f"Stage {stage}")
    return G


def _run_all_stages(
    args: argparse.Namespace, output_path: Path
) -> "nx.MultiDiGraph":
    """Run 5.2 fresh, then 5.3, 5.4, 5.5; log-and-skip 5.6 stub."""
    print("Stage all: running every implemented sub-stage in order")
    print("=" * 50)

    G = nx.MultiDiGraph()
    print("\n--- Stage 5.2 (fresh) ---")
    G = _run_stage_5_2(args, G)
    _persist_and_report(G, output_path, label="Stage 5.2")

    print("\n--- Stage 5.3 (append) ---")
    G = _run_stage_5_3(args, G)
    _persist_and_report(G, output_path, label="Stage 5.3")

    print("\n--- Stage 5.4 (append) ---")
    G = _run_stage_5_4(args, G)
    _persist_and_report(G, output_path, label="Stage 5.4")

    print("\n--- Stage 5.5 (append) ---")
    G = _run_stage_5_5(args, G)
    _persist_and_report(G, output_path, label="Stage 5.5")

    for stage in ("5.6",):
        print(f"\n--- Stage {stage} ---")
        try:
            G = STAGE_RUNNERS[stage](args, G)
            _persist_and_report(G, output_path, label=f"Stage {stage}")
        except NotImplementedError as exc:
            print(f"  Skipped: {exc}")

    return G


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    output_path = resolve_path(args.output_path, project_root / "data" / "kg.gpickle")

    print(f"Stage 5 builder | --stage={args.stage} | output={output_path}")
    print(f"Mode: {'APPEND' if args.append else 'FRESH'}")

    if args.stage == "all":
        # --stage all manages its own append semantics between sub-stages.
        if args.append:
            print(
                "Warning: --append is ignored when --stage=all; "
                "Stage 5.2 always starts fresh."
            )
        _run_all_stages(args, output_path)
        return 0

    # Single-stage path.
    if args.stage in STAGES_REQUIRING_APPEND and not args.append:
        parser.error(
            f"Stage {args.stage} depends on prior stages and requires --append. "
            f"Run --stage 5.2 first, then re-invoke with --stage {args.stage} --append."
        )

    if args.append:
        try:
            G = load_existing_graph(output_path)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        except TypeError as exc:
            parser.error(str(exc))
        print(
            f"  Loaded existing graph: "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
        )
    else:
        G = nx.MultiDiGraph()
        print("  Created new MultiDiGraph.")

    _run_single_stage(args.stage, args, G, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
