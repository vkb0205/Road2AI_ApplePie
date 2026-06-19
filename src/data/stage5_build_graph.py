"""
Stage 5: Build Knowledge Graph (incremental).

This module is designed to be called multiple times so each sub-stage
(5.2 DOC, 5.3 ART, 5.4 CHUNK, 5.5 doc-doc edges, 5.6 CONCEPT) can append
to the same persisted ``networkx.MultiDiGraph``.

CLI:
    python -m src.data.stage5_build_graph --stage {5.2,5.3,5.4,5.5,5.6,5.6-push,all} \
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

Stage 5.6 adds curated CONCEPT nodes and chunk-first ``MENTIONS`` edges from
``stage3_chunks.parquet``.

Stage 5.6-push extracts chunk->concept relations and pushes them directly to
Supabase (via ``chunk_concept_mentions`` table) without modifying the local graph.
Use this to collect relations incrementally; later rebuild the graph from Supabase.

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
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm.auto import tqdm

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

try:
    import psycopg2
except ImportError:  # pragma: no cover - optional when using Supabase API tracker
    psycopg2 = None


def load_dotenv(path: Path) -> None:
    """Load environment variables from .env file."""
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

# Stage 5.6 requires chunk text as the primary evidence source for concepts.
REQUIRED_STAGE3_CONCEPT_COLUMNS = REQUIRED_STAGE3_COLUMNS + ("chunk_text",)

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

# Edge identifiers for the concept layer.
MENTIONS_RELATION = "MENTIONS"
MENTIONS_EDGE_KEY = "MENTIONS"

# Concept layer acceptance bounds (PLAN.md task 5.6).
CONCEPT_COUNT_MIN = 50
CONCEPT_COUNT_MAX = 100
MAX_CONCEPTS_PER_CHUNK = 3

# Path to the legal concepts YAML (relative to project root).
LEGAL_CONCEPTS_CONFIG = "config/legal_concepts.yaml"

# Default Hugging Face source for the relationships dataset (Stage 5.5).
DEFAULT_RELATIONSHIPS_PATH = (
    "hf://datasets/th1nhng0/vietnamese-legal-documents/data/relationships.parquet"
)

# Reserved edge attribute set on every doc-doc edge so the relation is queryable.
DOC_DOC_RELATION_ATTR = "relation"

# Path to the relationship mapping YAML (relative to project root).
RELATIONSHIP_MAPPING_CONFIG = "config/relationship_mapping.yaml"

# Allowed values for the --stage CLI argument.
ALLOWED_STAGES = (
    "5.2", "5.3", "5.4", "5.5", "5.6", "5.6-push", "5.7", "5.8",
    "all", "merge-partitions",
)
STAGES_REQUIRING_APPEND = {"5.3", "5.4", "5.5", "5.6", "5.7", "5.8"}

# Default connection string for the Supabase chunk_processing tracker.
# Resolved from env var DATABASE_URL or --db-connection-string CLI arg.
DEFAULT_DB_CONNECTION_STRING = ""


# ---------------------------------------------------------------------------
# DB-backed chunk processing tracker (Stage 5.6)
# ---------------------------------------------------------------------------

class ChunkProcessingTracker:
    """Records which chunks have had concept extraction applied.

    Writes to a Supabase/Postgres ``chunk_processing`` table so the pipeline
    can monitor progress, resume after interruption, and skip already-processed
    chunks on re-run.
    """

    INSERT_SQL = """
        INSERT INTO chunk_processing
            (chunk_id, doc_uid, doc_id, mentions_source, concepts_found, concepts_count)
        VALUES
            (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (chunk_id) DO UPDATE SET
            processed_at    = now(),
            mentions_source = EXCLUDED.mentions_source,
            concepts_found  = EXCLUDED.concepts_found,
            concepts_count  = EXCLUDED.concepts_count
    """

    MENTIONS_INSERT_SQL = """
        INSERT INTO chunk_concept_mentions
            (chunk_id, doc_uid, doc_id, concept_name, mentions_source)
        VALUES
            (%s, %s, %s, %s, %s)
        ON CONFLICT (chunk_id, concept_name) DO NOTHING
    """

    SELECT_PROCESSED_SQL = """
        SELECT chunk_id FROM chunk_processing
    """

    SELECT_MENTIONS_CHUNK_IDS_SQL = """
        SELECT DISTINCT chunk_id FROM chunk_concept_mentions
    """

    def __init__(self, connection_string: str):
        if not connection_string:
            raise ValueError(
                "DATABASE_URL is not set and no --db-connection-string was provided."
            )
        self._conn = psycopg2.connect(connection_string)
        self._conn.autocommit = True

    def close(self) -> None:
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    def already_processed(self) -> Set[str]:
        """Return the set of chunk_ids that have processing records."""
        with self._conn.cursor() as cur:
            cur.execute(self.SELECT_PROCESSED_SQL)
            return {row[0] for row in cur.fetchall()}

    def get_processed_chunk_ids_from_mentions(self) -> Set[str]:
        """Return the set of chunk_ids that already have concept mentions in chunk_concept_mentions."""
        with self._conn.cursor() as cur:
            cur.execute(self.SELECT_MENTIONS_CHUNK_IDS_SQL)
            return {row[0] for row in cur.fetchall()}

    def record_chunk(
        self,
        chunk_id: str,
        doc_uid: str,
        doc_id: str,
        mentions_source: str,
        concepts_found: List[str],
    ) -> None:
        """Insert or update a processing record for one chunk."""
        with self._conn.cursor() as cur:
            cur.execute(
                self.INSERT_SQL,
                (
                    chunk_id,
                    doc_uid,
                    doc_id,
                    mentions_source,
                    concepts_found,
                    len(concepts_found),
                ),
            )

    def record_mention(
        self,
        chunk_id: str,
        doc_uid: str,
        doc_id: str,
        concept_name: str,
        mentions_source: str,
    ) -> None:
        """Insert one chunk → concept mention row (idempotent)."""
        with self._conn.cursor() as cur:
            cur.execute(
                self.MENTIONS_INSERT_SQL,
                (chunk_id, doc_uid, doc_id, concept_name, mentions_source),
            )

    def record_chunk_mentions(
        self,
        chunk_id: str,
        doc_uid: str,
        doc_id: str,
        mentions_source: str,
        concept_names: List[str],
    ) -> None:
        """Insert all chunk → concept mentions for one chunk in a single transaction."""
        if not concept_names:
            return
        with self._conn.cursor() as cur:
            for concept_name in concept_names:
                cur.execute(
                    self.MENTIONS_INSERT_SQL,
                    (chunk_id, doc_uid, doc_id, concept_name, mentions_source),
                )


# ---------------------------------------------------------------------------
# Supabase Data API Tracker (Stage 5.6)
# ---------------------------------------------------------------------------

class SupabaseApiTracker:
    """HTTP-based tracker using Supabase Data API.

    Writes chunk processing records and mentions to the Supabase tables
    via the RESTful Data API. This works in environments where direct
    PostgreSQL connections are blocked (e.g. Kaggle sandboxes) but HTTPS
    outbound is allowed.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        table_chunk_processing: str = "chunk_processing",
        table_mentions: str = "chunk_concept_mentions",
    ):
        """Initialize the tracker with Supabase credentials.

        Args:
            supabase_url: Supabase project URL, e.g. https://xyz.supabase.co
            supabase_key: Anon/public API key (service_role key may be needed for writes)
            table_chunk_processing: Name of the chunk_processing table
            table_mentions: Name of the chunk_concept_mentions table
        """
        if not supabase_url:
            raise ValueError("supabase_url is required")
        if not supabase_key:
            raise ValueError("supabase_key is required")

        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self.table_chunk_processing = table_chunk_processing
        self.table_mentions = table_mentions
        self._session = self._create_session()

    def _create_session(self):
        """Create a requests.Session with auth headers."""
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "requests is required for SupabaseApiTracker. "
                "Install it with `pip install requests`."
            ) from exc

        session = requests.Session()
        session.headers.update({
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
        })
        return session

    def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            self._session.close()

    def _table_endpoint(self, table_name: str) -> str:
        """Build the REST endpoint URL for a table."""
        return f"{self.supabase_url}/rest/v1/{table_name}"

    def already_processed(self) -> Set[str]:
        """Return the set of chunk_ids that have processing records.

        Fetches all chunk_id values from chunk_processing table.
        Note: This could be large; for production consider a filtered query.
        """
        url = self._table_endpoint(self.table_chunk_processing)
        params = {"select": "chunk_id"}
        resp = self._session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return {row["chunk_id"] for row in data if row.get("chunk_id")}

    def get_processed_chunk_ids_from_mentions(self) -> Set[str]:
        """Return the set of chunk_ids that already have concept mentions in chunk_concept_mentions.

        Fetches distinct chunk_id values from chunk_concept_mentions table.
        Note: This could be large; for production consider a filtered query.
        """
        url = self._table_endpoint(self.table_mentions)
        params = {"select": "chunk_id"}
        resp = self._session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return {row["chunk_id"] for row in data if row.get("chunk_id")}

    def record_chunk(
        self,
        chunk_id: str,
        doc_uid: str,
        doc_id: str,
        mentions_source: str,
        concepts_found: List[str],
    ) -> None:
        """Insert or update a processing record for one chunk via upsert.

        Uses the `on_conflict` query parameter to handle duplicates.
        """
        url = self._table_endpoint(self.table_chunk_processing)
        payload = {
            "chunk_id": chunk_id,
            "doc_uid": doc_uid,
            "doc_id": doc_id,
            "mentions_source": mentions_source,
            "concepts_found": concepts_found,
            "concepts_count": len(concepts_found),
        }
        # Supabase upsert: insert or update if conflict on chunk_id
        params = {
            "on_conflict": "chunk_id",
        }
        resp = self._session.post(url, json=payload, params=params)
        if resp.status_code not in (200, 201, 204):
            raise RuntimeError(
                f"Failed to upsert chunk_processing record: {resp.status_code} {resp.text}"
            )

    def record_mention(
        self,
        chunk_id: str,
        doc_uid: str,
        doc_id: str,
        concept_name: str,
        mentions_source: str,
    ) -> None:
        """Insert one chunk → concept mention row (idempotent via unique constraint)."""
        url = self._table_endpoint(self.table_mentions)
        payload = {
            "chunk_id": chunk_id,
            "doc_uid": doc_uid,
            "doc_id": doc_id,
            "concept_name": concept_name,
            "mentions_source": mentions_source,
        }
        # Don't fail on duplicate due to (chunk_id, concept_name) unique constraint
        resp = self._session.post(url, json=payload)
        if resp.status_code not in (200, 201, 204):
            # Ignore duplicate key errors (23505)
            if resp.status_code == 409:
                return
            raise RuntimeError(
                f"Failed to insert chunk_concept_mentions record: {resp.status_code} {resp.text}"
            )

    def record_chunk_mentions(
        self,
        chunk_id: str,
        doc_uid: str,
        doc_id: str,
        mentions_source: str,
        concept_names: List[str],
    ) -> None:
        """Insert all chunk → concept mentions for one chunk in a single transaction."""
        if not concept_names:
            return
        for concept_name in concept_names:
            self.record_mention(
                chunk_id=chunk_id,
                doc_uid=doc_uid,
                doc_id=doc_id,
                concept_name=concept_name,
                mentions_source=mentions_source,
            )


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


def _partition_output_path(base_path: Path, partition_idx: int) -> Path:
    """Derive a partition-specific save path from a base path.

    Example: ``data/kg.gpickle`` + idx=3 → ``data/kg_part_3.gpickle``.
    This naming convention is used by both the partition run (Stage 5.6
    with ``--partition-idx``) and the merge step (``--stage merge-partitions``).
    """
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    return parent / f"{stem}_part_{partition_idx}{suffix}"


def _partition_df(df: pd.DataFrame, num_partitions: int, partition_idx: int) -> pd.DataFrame:
    """Select a non-overlapping row subset for one partition using modular arithmetic.

    Each partition processes ``~1/num_partitions`` of the rows so that multiple
    Stage 5.6 invocations can run in parallel on disjoint chunk subsets.
    """
    if num_partitions <= 0:
        raise ValueError(f"num_partitions must be >= 1; got {num_partitions}")
    if not (0 <= partition_idx < num_partitions):
        raise ValueError(
            f"partition_idx {partition_idx} out of range [0, {num_partitions})."
        )
    return df.iloc[partition_idx::num_partitions].reset_index(drop=True)


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
    ``part_idx`` and ``breadcrumb``. A ``process_concept`` boolean (default
    ``False``) tracks whether Stage 5.6 concept extraction has been applied
    to this chunk. The full ``chunk_text`` is deliberately excluded to keep
    ``kg.gpickle`` lean (the text lives in Stage 3 / indexes).
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
        "process_concept": False,
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

    multiple_parents = {node: count for node, count in parent_count.items() if count != 1}
    if multiple_parents:
        sample = sorted(multiple_parents.items())[:5]
        raise AssertionError(
            f"{len(multiple_parents)} CHUNK nodes do not have exactly one parent ART "
            f"via HAS_CHUNK; sample: {sample}"
        )

    print(
        f"  CHUNK quality gates passed: {chunk_count} CHUNK nodes, "
        f"{len(has_chunk_edges)} HAS_CHUNK edges, all parented exactly once."
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
# Stage 5.6 — Concept nodes + MENTIONS edges
# ---------------------------------------------------------------------------

def load_legal_concepts(path: Path) -> List[str]:
    """Load the curated legal concept vocabulary from YAML.

    Expected schema: ``LEGAL_CONCEPTS`` is a list of 50-100 non-empty strings.
    Concepts are de-duplicated case-insensitively while preserving order.
    """
    if not path.exists():
        raise FileNotFoundError(f"Legal concepts config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    raw = cfg.get("LEGAL_CONCEPTS")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"LEGAL_CONCEPTS must be a non-empty list in {path}; "
            f"got {type(raw).__name__}."
        )

    concepts: List[str] = []
    seen: Set[str] = set()
    for item in raw:
        concept = str(item).strip()
        if not concept:
            continue
        key = concept.lower()
        if key not in seen:
            seen.add(key)
            concepts.append(concept)

    count = len(concepts)
    if not (CONCEPT_COUNT_MIN <= count <= CONCEPT_COUNT_MAX):
        raise ValueError(
            f"LEGAL_CONCEPTS count {count} outside acceptance window "
            f"[{CONCEPT_COUNT_MIN}, {CONCEPT_COUNT_MAX}]."
        )
    return concepts


def load_stage3_chunks_for_concepts(path: Path) -> pd.DataFrame:
    """Load Stage 3 chunks and require ``chunk_text`` for concept extraction."""
    df = load_stage3_chunks(path)
    missing = [c for c in REQUIRED_STAGE3_CONCEPT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Stage 3 parquet is missing columns required for Stage 5.6: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def build_concept_attrs(concept: str) -> Dict[str, Any]:
    """Build the locked attribute dict for one CONCEPT node."""
    name = concept.strip()
    return {"type": "Concept", "name": name, "name_lower": name.lower()}


def add_concept_nodes(G: "nx.MultiDiGraph", concepts: List[str]) -> int:
    """Add curated CONCEPT nodes; return newly-created node count."""
    added = 0
    for concept in concepts:
        attrs = build_concept_attrs(concept)
        node_id = f"CONCEPT:{attrs['name_lower']}"
        if node_id in G.nodes:
            G.nodes[node_id].update(attrs)
        else:
            G.add_node(node_id, **attrs)
            added += 1
    return added


def match_concepts_in_text(text: Any, concepts: List[str]) -> List[str]:
    """Return up to ``MAX_CONCEPTS_PER_CHUNK`` concepts found in chunk text."""
    if text is None:
        return []
    text_norm = str(text).lower()
    if not text_norm.strip():
        return []
    matches: List[str] = []
    for concept in concepts:
        if len(matches) >= MAX_CONCEPTS_PER_CHUNK:
            break
        if concept.lower() in text_norm:
            matches.append(concept)
    return matches


def _add_mentions_for_matches(
    G: "nx.MultiDiGraph", chunk_key: str, matches: List[str]
) -> Tuple[int, List[str]]:
    """Add MENTIONS edges for already-normalized concept matches.

    Returns ``(edges_added, actually_matched_concepts)`` where
    ``actually_matched_concepts`` is the subset of ``matches`` that
    correspond to existing CONCEPT nodes and were actually linked.
    """
    edges_added = 0
    matched: List[str] = []
    for concept in matches[:MAX_CONCEPTS_PER_CHUNK]:
        concept_key = f"CONCEPT:{concept.lower()}"
        if concept_key not in G.nodes:
            continue
        if not G.has_edge(chunk_key, concept_key, key=MENTIONS_EDGE_KEY):
            G.add_edge(
                chunk_key,
                concept_key,
                key=MENTIONS_EDGE_KEY,
                relation=MENTIONS_RELATION,
            )
            edges_added += 1
        matched.append(concept)
    return edges_added, matched


def add_mentions_edges(
    G: "nx.MultiDiGraph",
    chunks: pd.DataFrame,
    concepts: List[str],
    tracker: Optional[Any] = None,
    mentions_source: str = "substring",
    checkpoint_path: Optional[Path] = None,
) -> Tuple[int, int]:
    """Add CHUNK -> CONCEPT MENTIONS edges from Stage 3 chunk text.

    Only chunks already present in the graph are eligible. Returns
    ``(edges_added, chunks_with_mentions)``.

    When ``tracker`` is provided, every mention is also written to the
    Supabase ``chunk_concept_mentions`` table for durable persistence.

    When ``checkpoint_path`` is provided, the graph is persisted after every
    processed chunk so partition outputs (``kg_part_x.gpickle``) survive
    interruption.
    """
    edges_added = 0
    chunks_with_mentions = 0
    for row in tqdm(
        chunks.itertuples(index=False),
        total=len(chunks),
        desc="  Matching concepts in chunks",
        unit=" chunk",
        leave=False,
    ):
        chunk_id = str(row.chunk_id)
        doc_uid = str(row.doc_uid)
        doc_id = str(row.doc_id)
        chunk_key = f"CHUNK:{chunk_id}"
        if chunk_key not in G.nodes or G.nodes[chunk_key].get("type") != "Chunk":
            continue

        matches = match_concepts_in_text(getattr(row, "chunk_text", None), concepts)
        if matches:
            chunks_with_mentions += 1
        added, matched = _add_mentions_for_matches(G, chunk_key, matches)
        edges_added += added

        # Persist to Supabase if tracker is available
        if tracker is not None and matched:
            tracker.record_chunk_mentions(
                chunk_id=chunk_id,
                doc_uid=doc_uid,
                doc_id=doc_id,
                mentions_source=mentions_source,
                concept_names=matched,
            )

        if checkpoint_path is not None:
            persist_graph(G, checkpoint_path)
            print(f"  Checkpointed {checkpoint_path} after chunk {chunk_id}.")
    return edges_added, chunks_with_mentions


def _canonicalize_llm_concepts(raw_concepts: Any, concepts_by_lower: Dict[str, str]) -> List[str]:
    """Map LLM-returned concept names to curated concept strings."""
    if not isinstance(raw_concepts, list):
        return []
    matches: List[str] = []
    seen: Set[str] = set()
    for item in raw_concepts:
        key = str(item).strip().lower()
        concept = concepts_by_lower.get(key)
        if concept and key not in seen:
            seen.add(key)
            matches.append(concept)
        if len(matches) >= MAX_CONCEPTS_PER_CHUNK:
            break
    return matches


def add_mentions_edges_llm(
    G: "nx.MultiDiGraph",
    chunks: pd.DataFrame,
    concepts: List[str],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
    batch_size: int = 10,
    provider: str = "gemini",
    tracker: Optional[Any] = None,
    mentions_source: str = "llm",
    checkpoint_path: Optional[Path] = None,
) -> Tuple[int, int]:
    """Add CHUNK -> CONCEPT MENTIONS edges using the script LLM extractor.

    The LLM is only allowed to select concepts that already exist in the curated
    Stage 5.6 vocabulary, preserving the fixed CONCEPT node acceptance window.
    """
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from extract_concepts import (
            gemini_extract_mentions,
            openai_compatible_extract_mentions,
            tokenize_for_prompt,
        )
    except ImportError as exc:
        raise ImportError(
            "LLM-generated MENTIONS require scripts/extract_concepts.py helpers "
            "to be importable."
        ) from exc

    provider = provider.lower().replace("_", "-")
    if provider not in {"gemini", "openai-compatible", "deepseek"}:
        raise ValueError(
            "Unsupported LLM provider "
            f"{provider!r}; expected 'gemini', 'openai-compatible', or 'deepseek'."
        )

    # Adjust model name default if it wasn't customized
    if model_name == "gemini-2.0-flash":
        if provider == "deepseek":
            model_name = "deepseek-v4-flash"
        elif provider == "openai-compatible":
            model_name = "gpt-4o-mini"

    llm = None
    openai_client = None
    # Resolve base_url from explicit arg, then provider-specific or default env var.
    if provider == "gemini":
        resolved_base_url = base_url or os.getenv("BASE_URL")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "Gemini MENTIONS extraction requires google-generativeai."
            ) from exc
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Gemini API key is required for --mentions-source llm. Provide "
                "--llm-api-key or set GEMINI_API_KEY or API_KEY."
            )
        client_options: Dict[str, str] = {}
        if resolved_base_url:
            client_options["api_endpoint"] = resolved_base_url
        genai.configure(api_key=resolved_api_key, client_options=client_options)
        llm = genai.GenerativeModel(model_name)
    else:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                f"{provider.upper()} MENTIONS extraction requires the openai package. "
                "Install it with `pip install openai`."
            ) from exc

        if provider == "deepseek":
            resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
            resolved_base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("BASE_URL") or "https://api.deepseek.com"
        else:
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            resolved_base_url = base_url or os.getenv("BASE_URL")

        if not resolved_api_key:
            raise ValueError(
                f"{provider.upper()} API key is required for --mentions-source llm. Provide "
                f"--llm-api-key or set appropriate environment variables."
            )
        client_kwargs: Dict[str, str] = {"api_key": resolved_api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        openai_client = OpenAI(**client_kwargs)

    concepts_by_lower = {concept.lower(): concept for concept in concepts}

    # Build eligible items with synthetic short IDs so the LLM never sees
    # the long, special-character-heavy real chunk_id values. The real
    # chunk_id is recovered from the positional index after the call returns.
    eligible_items: List[Dict[str, str]] = []
    for row in chunks.itertuples(index=False):
        chunk_id = str(row.chunk_id)
        chunk_key = f"CHUNK:{chunk_id}"
        if chunk_key not in G.nodes or G.nodes[chunk_key].get("type") != "Chunk":
            continue
        eligible_items.append(
            {
                "source_id": chunk_id,
                "doc_uid": str(row.doc_uid),
                "doc_id": str(row.doc_id),
                "text": tokenize_for_prompt(getattr(row, "chunk_text", None)),
            }
        )

    # Map: position in eligible_items -> real chunk_id
    idx_to_chunk_id: Dict[int, str] = {
        i: item["source_id"] for i, item in enumerate(eligible_items)
    }

    edges_added = 0
    chunks_with_mentions = 0
    if batch_size <= 0:
        raise ValueError(f"LLM batch size must be positive; got {batch_size}.")

    def _run_batch(batch_items):
        """Run one batch through the LLM. Returns parsed result dict or raises."""
        if provider == "gemini":
            return gemini_extract_mentions(llm, batch_items, concepts)
        return openai_compatible_extract_mentions(
            openai_client, model_name, batch_items, concepts
        )

    def _process_result(result, batch_items):
        """Apply LLM result to the graph. Returns (edges_added, chunks_with_mentions)."""
        ea = 0
        cwm = 0
        by_source = {
            str(item.get("source_id")): item for item in result.get("items", [])
        }
        for item in batch_items:
            source_id = item["source_id"]
            doc_uid = item.get("doc_uid", "")
            doc_id = item.get("doc_id", "")
            chunk_key = f"CHUNK:{source_id}"
            matches = _canonicalize_llm_concepts(
                by_source.get(source_id, {}).get("concepts", []), concepts_by_lower
            )
            if matches:
                cwm += 1
            added, matched = _add_mentions_for_matches(G, chunk_key, matches)
            ea += added

            # Persist to Supabase if tracker is available
            if tracker is not None and matched:
                tracker.record_chunk_mentions(
                    chunk_id=source_id,
                    doc_uid=doc_uid,
                    doc_id=doc_id,
                    mentions_source=mentions_source,
                    concept_names=matched,
                )

            if checkpoint_path is not None:
                persist_graph(G, checkpoint_path)
                print(f"  Checkpointed {checkpoint_path} after chunk {source_id}.")
        return ea, cwm

    total_batches = math.ceil(len(eligible_items) / batch_size)
    for batch_start in tqdm(
        range(0, len(eligible_items), batch_size),
        total=total_batches,
        desc="  LLM concept extraction",
        unit=" batch",
        leave=False,
    ):
        batch = eligible_items[batch_start : batch_start + batch_size]
        # Swap in synthetic short IDs for the LLM call
        synth_batch = [
            {"source_id": f"c{i}", "text": item["text"]}
            for i, item in enumerate(batch, start=batch_start)
        ]
        try:
            result = _run_batch(synth_batch)
        except (ValueError, json.JSONDecodeError) as exc:
            batch_real_ids = [
                idx_to_chunk_id.get(batch_start + i, "?")
                for i in range(len(batch))
            ]
            print(
                f"\n  [WARN] LLM extraction failed for batch starting at idx "
                f"{batch_start}: {exc}\n"
                f"  Chunk IDs in failed batch: {batch_real_ids}\n"
                f"  Retrying each chunk individually...\n",
                file=sys.stderr,
            )
            # Per-item fallback: try each chunk solo before giving up entirely
            for j, item in enumerate(batch):
                synth_single = [
                    {"source_id": f"c{batch_start + j}", "text": item["text"]}
                ]
                try:
                    single_result = _run_batch(synth_single)
                except (ValueError, json.JSONDecodeError) as exc2:
                    print(
                        f"\n  [WARN] Individual chunk also failed "
                        f"(idx={batch_start + j}, "
                        f"id={idx_to_chunk_id.get(batch_start + j, '?')}): "
                        f"{exc2}\n",
                        file=sys.stderr,
                    )
                    continue
                real_id = idx_to_chunk_id[batch_start + j]
                single_result["items"] = [
                    {**it, "source_id": real_id}
                    for it in single_result.get("items", [])
                ]
                ea, cwm = _process_result(
                    single_result,
                    [{"source_id": real_id, "doc_uid": item["doc_uid"], "doc_id": item["doc_id"], "text": item["text"]}],
                )
                edges_added += ea
                chunks_with_mentions += cwm
            continue
        # Remap synthetic IDs back to real chunk_ids in the result
        real_items = []
        for it in result.get("items", []):
            synth_id = str(it.get("source_id", ""))
            # synth_id is like "c42" — extract the index
            if synth_id.startswith("c") and synth_id[1:].isdigit():
                idx = int(synth_id[1:])
                real_id = idx_to_chunk_id.get(idx)
                if real_id is not None:
                    real_items.append({**it, "source_id": real_id})
        result["items"] = real_items
        # Build a real-id batch view for _process_result
        real_batch = [
            {
                "source_id": idx_to_chunk_id[batch_start + i],
                "doc_uid": item["doc_uid"],
                "doc_id": item["doc_id"],
                "text": item["text"],
            }
            for i, item in enumerate(batch)
        ]
        ea, cwm = _process_result(result, real_batch)
        edges_added += ea
        chunks_with_mentions += cwm
    if edges_added == 0 and total_batches > 0:
        print(
            "\n  [WARN] No MENTIONS edges were added after processing all batches. "
            "Check LLM provider / model compatibility.\n",
            file=sys.stderr,
        )
    return edges_added, chunks_with_mentions


def run_concept_quality_gates(
    G: "nx.MultiDiGraph",
    expected_band: Tuple[int, int] = (CONCEPT_COUNT_MIN, CONCEPT_COUNT_MAX),
) -> None:
    """Validate concept nodes and chunk-first MENTIONS edge invariants."""
    concept_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "Concept"]
    concept_count = len(concept_nodes)
    lo, hi = expected_band
    if not (lo <= concept_count <= hi):
        raise AssertionError(
            f"CONCEPT node count {concept_count} outside acceptance window [{lo}, {hi}]."
        )

    for node_id, attrs in concept_nodes:
        if not attrs.get("name") or not attrs.get("name_lower"):
            raise AssertionError(f"CONCEPT node {node_id} has empty name/name_lower.")
        if node_id != f"CONCEPT:{attrs.get('name_lower')}":
            raise AssertionError(
                f"CONCEPT node key {node_id!r} does not match name_lower "
                f"{attrs.get('name_lower')!r}."
            )

    mentions_edges = [
        (u, v, k, d)
        for u, v, k, d in G.edges(keys=True, data=True)
        if d.get("relation") == MENTIONS_RELATION
    ]
    for u, v, k, d in mentions_edges:
        if G.nodes[u].get("type") != "Chunk":
            raise AssertionError(
                f"MENTIONS edge {u} -> {v} source is not a Chunk node "
                f"(type={G.nodes[u].get('type')!r})."
            )
        if G.nodes[v].get("type") != "Concept":
            raise AssertionError(
                f"MENTIONS edge {u} -> {v} target is not a Concept node "
                f"(type={G.nodes[v].get('type')!r})."
            )
        if k != MENTIONS_EDGE_KEY:
            raise AssertionError(
                f"MENTIONS edge {u} -> {v} has unexpected edge key {k!r}; "
                f"expected {MENTIONS_EDGE_KEY!r}."
            )

    art_to_concept = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if G.nodes[u].get("type") == "Article" and G.nodes[v].get("type") == "Concept"
    ]
    if art_to_concept:
        raise AssertionError(
            f"Found {len(art_to_concept)} ART -> CONCEPT edges; Stage 5.6 requires "
            "MENTIONS edges to be CHUNK -> CONCEPT only."
        )

    print(
        f"  CONCEPT quality gates passed: {concept_count} CONCEPT nodes, "
        f"{len(mentions_edges)} CHUNK -> CONCEPT MENTIONS edges."
    )


# ---------------------------------------------------------------------------
# Stage 5.7 / 5.8 — Final persistence and validation
# ---------------------------------------------------------------------------

def _assert_graph_readable(path: Path) -> "nx.MultiDiGraph":
    """Round-trip-load ``path`` and verify it contains a NetworkX MultiDiGraph."""
    G = load_existing_graph(path)
    print(
        f"  Readable graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges."
    )
    return G


def _print_graph_info(G: "nx.MultiDiGraph") -> None:
    """Print a stable replacement for deprecated ``nx.info(G)`` output."""
    print(
        "  nx.info equivalent: "
        f"MultiDiGraph with {G.number_of_nodes()} nodes and "
        f"{G.number_of_edges()} edges"
    )
    type_counts: Dict[str, int] = {}
    for _node, attrs in G.nodes(data=True):
        node_type = str(attrs.get("type", "(missing)"))
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    print("  Node counts by type:")
    for node_type, count in sorted(type_counts.items()):
        print(f"    {node_type:<10} {count:>8,}")


def run_full_graph_quality_gates(G: "nx.MultiDiGraph", whitelist: Set[str]) -> None:
    """Validate final Stage 5 graph invariants for the chunk-first KG."""
    doc_count = sum(1 for _n, d in G.nodes(data=True) if d.get("type") == "Document")
    if not (DOC_COUNT_MIN <= doc_count <= DOC_COUNT_MAX):
        raise AssertionError(
            f"DOC node count {doc_count} outside acceptance window "
            f"[{DOC_COUNT_MIN}, {DOC_COUNT_MAX}]."
        )

    run_article_quality_gates(G, expected_band=(ART_COUNT_MIN, ART_COUNT_MAX))
    run_chunk_quality_gates(G, expected_band=(CHUNK_COUNT_MIN, CHUNK_COUNT_MAX))
    run_concept_quality_gates(G, expected_band=(CONCEPT_COUNT_MIN, CONCEPT_COUNT_MAX))

    doc_doc_edges = [
        (u, v, k, d)
        for u, v, k, d in G.edges(keys=True, data=True)
        if G.nodes.get(u, {}).get("type") == "Document"
        and G.nodes.get(v, {}).get("type") == "Document"
    ]
    if doc_doc_edges:
        run_doc_doc_quality_gates(
            G,
            whitelist=whitelist,
            expected_band=(DOC_DOC_EDGE_COUNT_MIN, DOC_DOC_EDGE_COUNT_MAX),
        )

    invalid_type_edges: List[Tuple[str, str, str, str]] = []
    allowed = {
        ("Document", "Document"),
        ("Document", "Article"),
        ("Article", "Chunk"),
        ("Chunk", "Concept"),
    }
    for u, v, _k, data in G.edges(keys=True, data=True):
        src_type = G.nodes[u].get("type")
        dst_type = G.nodes[v].get("type")
        if (src_type, dst_type) not in allowed:
            invalid_type_edges.append((u, v, str(src_type), str(dst_type)))

    if invalid_type_edges:
        raise AssertionError(
            f"Found {len(invalid_type_edges)} edges outside DOC->DOC, DOC->ART, "
            f"ART->CHUNK, CHUNK->CONCEPT schema; sample: {invalid_type_edges[:5]}"
        )

    print("  Full graph quality gates passed: DOC -> ART -> CHUNK -> CONCEPT enforced.")


def _run_stage_5_7(args: argparse.Namespace, G: "nx.MultiDiGraph") -> "nx.MultiDiGraph":
    """Stage 5.7: re-persist graph with ``pickle.HIGHEST_PROTOCOL`` and verify readable."""
    output_path = resolve_path(
        args.output_path,
        Path(__file__).resolve().parents[2] / "data" / "kg.gpickle",
    )
    print("Stage 5.7: Persist graph with pickle.HIGHEST_PROTOCOL")
    print("=" * 50)
    persist_graph(G, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Re-wrote {output_path} ({size_mb:.2f} MB) with protocol {pickle.HIGHEST_PROTOCOL}.")
    reloaded = _assert_graph_readable(output_path)
    _print_graph_info(reloaded)
    return reloaded


def _run_stage_5_8(args: argparse.Namespace, G: "nx.MultiDiGraph") -> "nx.MultiDiGraph":
    """Stage 5.8: final graph invariant validation for chunk-first pipeline."""
    project_root = Path(__file__).resolve().parents[2]
    mapping_path = resolve_path(
        args.relationship_mapping_path,
        project_root / RELATIONSHIP_MAPPING_CONFIG,
    )
    print("Stage 5.8: Validate final graph quality gates")
    print("=" * 50)
    print(f"  Mapping config: {mapping_path}")
    whitelist = load_relationship_mapping(mapping_path)["RELATION_WHITELIST"]
    run_full_graph_quality_gates(G, whitelist=whitelist)
    _print_graph_info(G)
    return G


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
    """Stage 5.6 builder: add CONCEPT nodes + chunk-first MENTIONS edges.

    When ``--num-partitions`` is provided, only a row-wise slice of the chunks
    is processed so that multiple processes can run concurrently on disjoint
    subsets.

    When ``--db-connection-string`` is provided, every chunk→concept mention
    is also written to the PostgreSQL ``chunk_concept_mentions`` table for
    durable persistence. Alternatively, use ``--supabase-url`` and
    ``--supabase-key`` to write via the Supabase Data API (HTTP-based,
    works in restricted network environments like Kaggle).
    """
    project_root = Path(__file__).resolve().parents[2]
    stage3_path = resolve_path(
        args.stage3_path, project_root / "data" / "stage3_chunks.parquet"
    )
    concepts_path = resolve_path(
        args.legal_concepts_path,
        project_root / LEGAL_CONCEPTS_CONFIG,
    )

    print("Stage 5.6: Build CONCEPT nodes + CHUNK -> CONCEPT MENTIONS edges")
    print("=" * 50)
    print(f"  Stage 3 input  : {stage3_path}")
    print(f"  Concepts config: {concepts_path}")

    # --- Tracker setup -------------------------------------------------------
    # Tracker can be either PostgreSQL-based (ChunkProcessingTracker) or
    # Supabase Data API-based (SupabaseApiTracker). Prefer PostgreSQL if both
    # are configured, otherwise use whichever is available.
    tracker: Optional[Any] = None
    db_conn_str = getattr(args, "db_connection_string", "") or os.getenv("DATABASE_URL", "")
    supabase_url = getattr(args, "supabase_url", "") or os.getenv("SUPABASE_URL", "")
    supabase_key = getattr(args, "supabase_key", "") or os.getenv("SUPABASE_KEY", "")

    if db_conn_str:
        if psycopg2 is None:
            print(
                "  [WARN] psycopg2 not available but --db-connection-string provided. "
                "Install psycopg2-binary or use --supabase-url/--supabase-key instead."
            )
        else:
            tracker = ChunkProcessingTracker(db_conn_str)
            print(f"  DB tracker enabled (PostgreSQL): writing mentions to chunk_concept_mentions")
    elif supabase_url and supabase_key:
        tracker = SupabaseApiTracker(supabase_url, supabase_key)
        print(f"  DB tracker enabled (Supabase API): writing mentions to chunk_concept_mentions")
    else:
        print("  DB tracker disabled (provide --db-connection-string or --supabase-url+--supabase-key)")

    print("\n[1/5] Loading legal concept vocabulary")
    concepts = load_legal_concepts(concepts_path)
    print(f"  Loaded {len(concepts):,} curated concepts.")

    print("\n[2/5] Loading Stage 3 chunks with chunk_text")
    chunks = load_stage3_chunks_for_concepts(stage3_path)

    # --- Partition slicing ---------------------------------------------------
    num_partitions = getattr(args, "num_partitions", 0)
    partition_idx = getattr(args, "partition_idx", 0)
    if num_partitions and num_partitions > 1:
        chunks = _partition_df(chunks, num_partitions, partition_idx)
        print(
            f"  Running partition {partition_idx} of {num_partitions}: "
            f"{len(chunks):,} chunk rows (of {len(chunks) * num_partitions:,} total)."
        )
    else:
        print(f"  Loaded {len(chunks):,} chunk rows for concept extraction.")

    # --- Skip already-processed chunks ---------------------------------------
    # If a tracker is configured, query the chunk_concept_mentions table
    # to find chunks that already have concept extraction results. Those
    # chunks are skipped to make Stage 5.6 resumable and idempotent.
    if tracker is not None:
        print("\n[2.5/5] Checking for already-processed chunks in chunk_concept_mentions")
        try:
            processed_chunk_ids = tracker.get_processed_chunk_ids_from_mentions()
            before_count = len(chunks)
            chunks = chunks.loc[~chunks["chunk_id"].isin(processed_chunk_ids)].reset_index(drop=True)
            skipped = before_count - len(chunks)
            if skipped:
                print(f"  Skipped {skipped:,} chunks already present in chunk_concept_mentions.")
            else:
                print("  No previously processed chunks found; all chunks will be processed.")
        except Exception as exc:
            print(
                f"  [WARN] Failed to fetch processed chunk IDs from tracker: {exc}. "
                "Proceeding with all chunks.",
                file=sys.stderr,
            )

    print("\n[3/5] Adding CONCEPT nodes")
    nodes_added = add_concept_nodes(G, concepts)
    print(f"  Added {nodes_added:,} new CONCEPT nodes.")

    print("\n[4/5] Adding CHUNK -> CONCEPT MENTIONS edges")
    mentions_source = getattr(args, "mentions_source", "substring")
    output_path = resolve_path(args.output_path, project_root / "data" / "kg.gpickle")
    checkpoint_path = None
    if num_partitions and num_partitions > 1:
        checkpoint_path = _partition_output_path(output_path, partition_idx)
        print(f"  Per-chunk graph checkpoint enabled: {checkpoint_path}")
    if mentions_source == "llm":
        print(
            "  Using LLM-generated MENTIONS via scripts/extract_concepts.py "
            f"with provider {args.llm_provider!r} and model {args.llm_model_name!r}."
        )
        edges_added, chunks_with_mentions = add_mentions_edges_llm(
            G,
            chunks,
            concepts,
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            model_name=args.llm_model_name,
            batch_size=args.llm_batch_size,
            provider=args.llm_provider,
            tracker=tracker,
            mentions_source=mentions_source,
            checkpoint_path=checkpoint_path,
        )
    else:
        print("  Using deterministic substring matching.")
        edges_added, chunks_with_mentions = add_mentions_edges(
            G, chunks, concepts,
            tracker=tracker,
            mentions_source=mentions_source,
            checkpoint_path=checkpoint_path,
        )
    print(
        f"  Added {edges_added:,} new MENTIONS edges; "
        f"{chunks_with_mentions:,} chunks matched at least one concept."
    )
    if tracker is not None:
        print(f"  Pushed {edges_added:,} chunk→concept relations to Supabase.")

    # --- Clean up DB connection ----------------------------------------------
    if tracker is not None:
        tracker.close()
        print("  DB tracker connection closed.")

    print("\n[5/5] Concept quality gates")
    run_concept_quality_gates(
        G,
        expected_band=(CONCEPT_COUNT_MIN, CONCEPT_COUNT_MAX),
    )
    return G


def _run_merge_partitions(args: argparse.Namespace, G: "nx.MultiDiGraph") -> "nx.MultiDiGraph":
    """Merge partition output files back into the main graph.

    Walks ``output_path``-derived partition files (e.g. ``kg_part_0.gpickle``
    through ``kg_part_{num_partitions - 1}.gpickle``) and merges their
    CONCEPT nodes and MENTIONS edges into the base graph read from
    ``output_path``.

    Usage::

        python -m src.data.stage5_build_graph --stage merge-partitions --append
            --num-partitions 10 --output-path data/kg.gpickle
    """
    project_root = Path(__file__).resolve().parents[2]
    output_path = resolve_path(args.output_path, project_root / "data" / "kg.gpickle")
    num_partitions = getattr(args, "num_partitions", 0)

    print("Stage merge-partitions: Merge partition graphs into main graph")
    print("=" * 50)
    print(f"  Base graph     : {output_path}")
    print(f"  Number of parts : {num_partitions}")

    if num_partitions < 2:
        raise ValueError(
            f"--num-partitions must be >= 2 for merge; got {num_partitions}."
        )

    G_loaded = False
    for idx in range(num_partitions):
        part_path = _partition_output_path(output_path, idx)
        if not part_path.exists():
            print(
                f"  [WARN] Partition file {part_path} not found, skipping.",
                file=sys.stderr,
            )
            continue
        with part_path.open("rb") as f:
            part_G: nx.MultiDiGraph = pickle.load(f)
        if not G_loaded:
            # On the first valid partition, seed from the main graph (or create fresh).
            if output_path.exists():
                G = load_existing_graph(output_path)
            G_loaded = True

        # Merge Concept nodes
        concept_nodes = [
            (n, d) for n, d in part_G.nodes(data=True) if d.get("type") == "Concept"
        ]
        for node_id, attrs in concept_nodes:
            if node_id not in G.nodes:
                G.add_node(node_id, **attrs)
            else:
                G.nodes[node_id].update(attrs)

        # Merge MENTIONS edges
        mentions_added = 0
        for u, v, k, d in part_G.edges(keys=True, data=True):
            if d.get("relation") != MENTIONS_RELATION:
                continue
            if not G.has_edge(u, v, key=k):
                G.add_edge(u, v, key=k, **d)
                mentions_added += 1

        print(
            f"  Part {idx}: merged {len(concept_nodes)} Concept nodes, "
            f"{mentions_added} new MENTIONS edges."
        )

        # Clean up partition file to avoid double-merge in a re-run.
        part_path.unlink(missing_ok=True)
        print(f"  Removed {part_path}.")

    if not G_loaded:
        print("  [WARN] No partition files found; graph is unchanged.")
    else:
        print(
            f"\n  Merge complete: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges."
        )

    return G


def _run_stage_5_6_push(args: argparse.Namespace, G: "nx.MultiDiGraph") -> "nx.MultiDiGraph":
    """Stage 5.6-push: extract chunk->concept relations and push to Supabase only.

    This mode does NOT modify the local graph. It only:
    - Loads legal concepts vocabulary
    - Loads Stage 3 chunks (with chunk_text)
    - Extracts concept mentions (substring or LLM)
    - Pushes each mention to Supabase via the tracker

    Use this to collect relations incrementally. Later, you can rebuild the
    graph from the Supabase chunk_concept_mentions table using a future stage.

    Requires either --db-connection-string or --supabase-url+--supabase-key.
    """
    project_root = Path(__file__).resolve().parents[2]
    stage3_path = resolve_path(
        args.stage3_path, project_root / "data" / "stage3_chunks.parquet"
    )
    concepts_path = resolve_path(
        args.legal_concepts_path,
        project_root / LEGAL_CONCEPTS_CONFIG,
    )

    print("Stage 5.6-push: Extract concepts and push to Supabase only")
    print("=" * 50)
    print(f"  Stage 3 input  : {stage3_path}")
    print(f"  Concepts config: {concepts_path}")
    print("  NOTE: Graph is NOT modified. Only Supabase is updated.")

    # --- Tracker setup (required for this mode) ---------------------------------
    tracker: Optional[Any] = None
    db_conn_str = getattr(args, "db_connection_string", "") or os.getenv("DATABASE_URL", "")
    supabase_url = getattr(args, "supabase_url", "") or os.getenv("SUPABASE_URL", "")
    supabase_key = getattr(args, "supabase_key", "") or os.getenv("SUPABASE_KEY", "")

    if db_conn_str:
        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2 is not available but --db-connection-string provided. "
                "Install psycopg2-binary or use --supabase-url/--supabase-key."
            )
        tracker = ChunkProcessingTracker(db_conn_str)
        print(f"  DB tracker enabled (PostgreSQL): writing to chunk_concept_mentions")
    elif supabase_url and supabase_key:
        tracker = SupabaseApiTracker(supabase_url, supabase_key)
        print(f"  DB tracker enabled (Supabase API): writing to chunk_concept_mentions")
    else:
        raise ValueError(
            "Stage 5.6-push requires either --db-connection-string or "
            "--supabase-url+--supabase-key to write to Supabase."
        )

    print("\n[1/4] Loading legal concept vocabulary")
    concepts = load_legal_concepts(concepts_path)
    print(f"  Loaded {len(concepts):,} curated concepts.")

    print("\n[2/4] Loading Stage 3 chunks with chunk_text")
    chunks = load_stage3_chunks_for_concepts(stage3_path)
    print(f"  Loaded {len(chunks):,} chunk rows for concept extraction.")

    # --- Skip already-processed chunks -----------------------------------------
    print("\n[2.5/4] Checking for already-processed chunks in chunk_concept_mentions")
    try:
        processed_chunk_ids = tracker.get_processed_chunk_ids_from_mentions()
        before_count = len(chunks)
        chunks = chunks.loc[~chunks["chunk_id"].isin(processed_chunk_ids)].reset_index(drop=True)
        skipped = before_count - len(chunks)
        if skipped:
            print(f"  Skipped {skipped:,} chunks already present in chunk_concept_mentions.")
        else:
            print("  No previously processed chunks found; all chunks will be processed.")
    except Exception as exc:
        print(
            f"  [WARN] Failed to fetch processed chunk IDs from tracker: {exc}. "
            "Proceeding with all chunks.",
            file=sys.stderr,
        )

    print("\n[3/4] Extracting concepts and pushing to Supabase")
    mentions_source = getattr(args, "mentions_source", "substring")
    if mentions_source == "llm":
        print(
            f"  Using LLM-generated MENTIONS with provider {args.llm_provider!r} "
            f"and model {args.llm_model_name!r}."
        )
        # We still need a graph to pass to add_mentions_edges_llm, but we'll
        # create an empty one and ignore the result. The tracker will be populated.
        empty_G = nx.MultiDiGraph()
        edges_added, chunks_with_mentions = add_mentions_edges_llm(
            empty_G,
            chunks,
            concepts,
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            model_name=args.llm_model_name,
            batch_size=args.llm_batch_size,
            provider=args.llm_provider,
            tracker=tracker,
            mentions_source=mentions_source,
            checkpoint_path=None,  # No checkpointing in push-only mode
        )
    else:
        print("  Using deterministic substring matching.")
        empty_G = nx.MultiDiGraph()
        edges_added, chunks_with_mentions = add_mentions_edges(
            empty_G, chunks, concepts,
            tracker=tracker,
            mentions_source=mentions_source,
            checkpoint_path=None,  # No checkpointing in push-only mode
        )
    print(
        f"  Pushed {edges_added:,} MENTIONS edges to Supabase; "
        f"{chunks_with_mentions:,} chunks matched at least one concept."
    )

    # --- Clean up ----------------------------------------------------------------
    print("\n[4/4] Closing tracker connection")
    tracker.close()
    print("  Tracker connection closed.")

    print("\nStage 5.6-push complete. Relations are stored in Supabase.")
    print("To rebuild the graph from Supabase, implement a future stage that")
    print("reads chunk_concept_mentions and constructs CONCEPT nodes + MENTIONS edges.")
    return G


STAGE_RUNNERS = {
    "5.2": _run_stage_5_2,
    "5.3": _run_stage_5_3,
    "5.4": _run_stage_5_4,
    "5.5": _run_stage_5_5,
    "5.6": _run_stage_5_6,
    "5.6-push": _run_stage_5_6_push,
    "5.7": _run_stage_5_7,
    "5.8": _run_stage_5_8,
    "merge-partitions": _run_merge_partitions,
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
            "the relationships dataset, 5.6=CONCEPT nodes + MENTIONS edges, "
            "5.6-push=extract concepts and push to Supabase only (no graph changes), "
            "5.7=final pickle persistence/readability check, 5.8=final graph "
            "validation, all=run every implemented stage in order."
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
        "--legal-concepts-path",
        default=None,
        help=(
            "Path to legal concepts YAML "
            "(default: config/legal_concepts.yaml)"
        ),
    )
    parser.add_argument(
        "--mentions-source",
        choices=("substring", "llm"),
        default="substring",
        help=(
            "How Stage 5.6 generates CHUNK -> CONCEPT MENTIONS edges: "
            "deterministic substring matching or LLM extraction via "
            "scripts/extract_concepts.py."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        choices=("gemini", "openai-compatible", "deepseek"),
        default="gemini",
        help="LLM provider for --mentions-source llm.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help=(
            "API key for --mentions-source llm. Overrides GEMINI_API_KEY for "
            "Gemini, OPENAI_API_KEY for OpenAI-compatible, or DEEPSEEK_API_KEY for Deepseek."
        ),
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help=(
            "Custom API endpoint for --mentions-source llm: Gemini api_endpoint, "
            "OpenAI-compatible base_url, or Deepseek base_url."
        ),
    )
    parser.add_argument(
        "--llm-model-name",
        default="gemini-2.0-flash",
        help=(
            "Model name for --mentions-source llm. Defaults to 'gemini-2.0-flash' "
            "for Gemini, 'deepseek-v4-flash' for Deepseek, or 'gpt-4o-mini' for OpenAI-compatible."
        ),
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=10,
        help="Number of chunks per LLM request for --mentions-source llm.",
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
            "starting fresh. Required for stages 5.3-5.8."
        ),
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=0,
        help=(
            "Split Stage 5.6 chunk processing into this many disjoint "
            "partitions for parallel execution. Pass --stage 5.6 --append "
            "--partition-idx N for each worker, then run "
            "--stage merge-partitions --append --num-partitions N to merge."
        ),
    )
    parser.add_argument(
        "--partition-idx",
        type=int,
        default=0,
        help=(
            "Zero-based partition index for this worker "
            "(requires --num-partitions)."
        ),
    )
    parser.add_argument(
        "--db-connection-string",
        type=str,
        default="",
        help=(
            "PostgreSQL connection string for the Supabase chunk_concept_mentions "
            "table. Falls back to DATABASE_URL env var. When set, every "
            "chunk→concept mention is written to the DB for durable tracking."
        ),
    )
    parser.add_argument(
        "--supabase-url",
        type=str,
        default="",
        help=(
            "Supabase project URL (e.g. https://xyz.supabase.co). "
            "Falls back to SUPABASE_URL env var. Used with --supabase-key "
            "for HTTP-based tracking when direct PostgreSQL is unavailable."
        ),
    )
    parser.add_argument(
        "--supabase-key",
        type=str,
        default="",
        help=(
            "Supabase anon/public API key. Falls back to SUPABASE_KEY env var. "
            "Required with --supabase-url for HTTP-based tracking."
        ),
    )
    return parser


def _run_single_stage(
    stage: str,
    args: argparse.Namespace,
    G: "nx.MultiDiGraph",
    output_path: Path,
) -> "nx.MultiDiGraph":
    """Dispatch to a single stage runner and persist the result (unless stage doesn't modify graph)."""
    runner = STAGE_RUNNERS[stage]
    G = runner(args, G)
    # Skip persistence for stages that do not modify the graph
    if stage not in ("5.6-push",):
        _persist_and_report(G, output_path, label=f"Stage {stage}")
    else:
        print(f"  Skipping graph persistence for stage {stage} (no graph modifications).")
    return G


def _run_all_stages(
    args: argparse.Namespace, output_path: Path
) -> "nx.MultiDiGraph":
    """Run 5.2 fresh, then 5.3, 5.4, 5.5, 5.6, 5.7 and 5.8."""
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

    print("\n--- Stage 5.6 (append) ---")
    G = _run_stage_5_6(args, G)
    _persist_and_report(G, output_path, label="Stage 5.6")

    print("\n--- Stage 5.7 (append) ---")
    G = _run_stage_5_7(args, G)
    _persist_and_report(G, output_path, label="Stage 5.7")

    print("\n--- Stage 5.8 (append) ---")
    G = _run_stage_5_8(args, G)
    _persist_and_report(G, output_path, label="Stage 5.8")

    return G


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    # Load .env configuration before any API calls.
    load_dotenv(project_root / ".env")
    output_path = resolve_path(args.output_path, project_root / "data" / "kg.gpickle")

    print(f"Stage 5 builder | --stage={args.stage} | output=    {output_path}")
    print(f"Mode: {'APPEND' if args.append else 'FRESH'}")

    num_partitions = getattr(args, "num_partitions", 0)
    partition_idx = getattr(args, "partition_idx", 0)
    if num_partitions > 1:
        print(
            f"  Partition mode: {num_partitions} partitions, "
            f"this worker = idx {partition_idx}."
        )

    # --- merge-partitions special case --------------------------------------
    # This stage reads partition files on its own and does not require --append.
    if args.stage == "merge-partitions":
        _run_single_stage(args.stage, args, nx.MultiDiGraph(), output_path)
        return 0

    # --- 5.6-push special case -----------------------------------------------
    # This stage only pushes to Supabase and does not modify or use the graph file.
    # It does not require --append.
    if args.stage == "5.6-push":
        _run_single_stage(args.stage, args, nx.MultiDiGraph(), output_path)
        return 0

    # --- "all" stage --------------------------------------------------------
    if args.stage == "all":
        # --stage all manages its own append semantics between sub-stages.
        if args.append:
            print(
                "Warning: --append is ignored when --stage=all; "
                "Stage 5.2 always starts fresh."
            )
        _run_all_stages(args, output_path)
        return 0

    # --- Single-stage path ---------------------------------------------------
    # Stages 5.3-5.8 require --append (they load a graph built by prior stages).
    if args.stage in STAGES_REQUIRING_APPEND and not args.append:
        parser.error(
            f"Stage {args.stage} depends on prior stages and requires --append. "
            f"Run --stage 5.2 first, then re-invoke with --stage {args.stage} --append."
        )

    # Resolve the actual file to load/save: in partition mode each worker
    # writes to its own dedicated file so they never collide.
    if args.stage == "5.6" and num_partitions > 1 and partition_idx is not None:
        stage_output_path = _partition_output_path(output_path, partition_idx)
    else:
        stage_output_path = output_path

    if args.append:
        try:
            path_to_load = stage_output_path
            # For partition workers, if the partition file does not exist, fall back to base graph
            if args.stage == "5.6" and num_partitions > 1 and partition_idx is not None and not stage_output_path.exists():
                path_to_load = output_path
            G = load_existing_graph(path_to_load)
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

    _run_single_stage(args.stage, args, G, stage_output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
