"""Unit tests for Stage 5.4 — Chunk nodes + HAS_CHUNK edges.

Covers the public surface added for PLAN.md task 5.4 (KG.md §3.3, §7.3 and
G-LRAG_SPECIFICATIONS.md §8.6 Step 4): chunk loading, attribute schema,
partition by ART presence, node/edge creation, idempotency, orphan JSONL
export, and the chunk-layer quality gates.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import pandas as pd
import pytest

from data.stage5_build_graph import (
    CHUNK_COUNT_MAX,
    CHUNK_COUNT_MIN,
    HAS_CHUNK_EDGE_KEY,
    HAS_CHUNK_RELATION,
    add_article_nodes,
    add_chunk_nodes,
    add_document_nodes,
    build_chunk_attrs,
    export_orphan_chunks,
    main,
    partition_articles,
    partition_chunks,
    run_chunk_quality_gates,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "src" / "data" / "stage5_build_graph.py"


def _doc_row(doc_id: str, law_id: str, ten: str) -> Dict[str, Any]:
    return {
        "id": doc_id,
        "law_id": law_id,
        "ten_van_ban": ten,
        "loai_van_ban": "Luật",
        "nganh": "Doanh nghiệp",
        "linh_vuc": "thuế",
        "ngay_ban_hanh": "01/01/2024",
        "tinh_trang_hieu_luc": "Còn hiệu lực",
        "ngay_co_hieu_luc": "01/02/2024",
        "ngay_het_hieu_luc": "",
    }


def _article_row(
    doc_id: str,
    law_id: str,
    ten: str,
    dieu_so: str,
    dieu_ten: str,
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "law_id": law_id,
        "ten_van_ban": ten,
        "loai_van_ban": "Luật",
        "ngay_ban_hanh": "01/01/2024",
        "phan": "",
        "chuong": "Chương I",
        "muc": "",
        "dieu_so": dieu_so,
        "dieu_ten": dieu_ten,
        "noi_dung": f"Nội dung của {dieu_so} thuộc {ten}",
        "start_char": 0,
        "end_char": 100,
        "doc_uid": f"{law_id}|{ten}|{dieu_so}",
    }


def _chunk_row(
    doc_id: str,
    doc_uid: str,
    part_idx: int,
    *,
    breadcrumb: str = "Luật > Chương I",
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "law_id": "01/2024/QH15",
        "ten_van_ban": "Luật Doanh nghiệp",
        "loai_van_ban": "Luật",
        "ngay_ban_hanh": "01/01/2024",
        "nganh": "Doanh nghiệp",
        "linh_vuc": "thuế",
        "phan": "",
        "chuong": "Chương I",
        "muc": "",
        "dieu_so": "Điều 1",
        "dieu_ten": "Phạm vi",
        "noi_dung": "Nội dung",
        "start_char": 0,
        "end_char": 100,
        "doc_uid": doc_uid,
        "breadcrumb": breadcrumb,
        "chunk_id": f"{doc_uid}#{part_idx}",
        "part_idx": part_idx,
        "chunk_text": f"{breadcrumb}\nNội dung chunk {part_idx}",
    }


@pytest.fixture
def art_graph() -> nx.MultiDiGraph:
    """A graph with DOC + ART nodes, mirroring a post-Stage 5.3 state."""
    G = nx.MultiDiGraph()
    docs_df = pd.DataFrame(
        [
            _doc_row("100", "01/2024/QH15", "Luật Doanh nghiệp"),
            _doc_row("200", "02/2024/QH15", "Luật Thuế GTGT"),
        ]
    )
    add_document_nodes(G, docs_df)

    arts_df = pd.DataFrame(
        [
            _article_row("100", "01/2024/QH15", "Luật Doanh nghiệp", "Điều 1", "Phạm vi"),
            _article_row("200", "02/2024/QH15", "Luật Thuế GTGT", "Điều 1", "Phạm vi"),
        ]
    )
    joined, _orphans = partition_articles(arts_df, G)
    add_article_nodes(G, joined)
    return G


@pytest.fixture
def chunks_df() -> pd.DataFrame:
    """Synthetic chunk frame: 3 chunks for ART:100-art, 1 for ART:200-art, 2 orphans."""
    uid_100 = "01/2024/QH15|Luật Doanh nghiệp|Điều 1"
    uid_200 = "02/2024/QH15|Luật Thuế GTGT|Điều 1"
    uid_orphan = "99/2024/QH15|Luật Khác|Điều 1"
    rows = [
        _chunk_row("100", uid_100, 0),
        _chunk_row("100", uid_100, 1),
        _chunk_row("100", uid_100, 2),
        _chunk_row("200", uid_200, 0),
        _chunk_row("999", uid_orphan, 0),
        _chunk_row("999", uid_orphan, 1),
    ]
    df = pd.DataFrame(rows)
    # Mirror loader behaviour: add a stable rowidx column.
    df["rowidx"] = range(len(df))
    return df


# ---------------------------------------------------------------------------
# partition_chunks
# ---------------------------------------------------------------------------

def test_partition_chunks_split(art_graph, chunks_df):
    joined, orphans = partition_chunks(chunks_df, art_graph)
    assert len(joined) == 4
    assert len(orphans) == 2
    assert set(orphans["doc_id"].astype(str)) == {"999"}


# ---------------------------------------------------------------------------
# build_chunk_attrs
# ---------------------------------------------------------------------------

def test_build_chunk_attrs_locked_schema(chunks_df):
    row = chunks_df.iloc[0]
    attrs = build_chunk_attrs(row)

    expected_keys = {
        "type",
        "chunk_id",
        "doc_uid",
        "doc_id",
        "rowidx",
        "part_idx",
        "breadcrumb",
    }
    assert set(attrs.keys()) == expected_keys
    assert attrs["type"] == "Chunk"
    # chunk_text / noi_dung must not leak into the node.
    assert "chunk_text" not in attrs
    assert "noi_dung" not in attrs
    assert attrs["part_idx"] == 0
    assert attrs["breadcrumb"] == "Luật > Chương I"


def test_build_chunk_attrs_empty_breadcrumb_normalizes_to_none():
    row = pd.Series(
        {
            "chunk_id": "uid#0",
            "doc_uid": "uid",
            "doc_id": "100",
            "rowidx": 0,
            "part_idx": 0,
            "breadcrumb": "   ",
        }
    )
    attrs = build_chunk_attrs(row)
    assert attrs["breadcrumb"] is None


# ---------------------------------------------------------------------------
# add_chunk_nodes — keys, edge attrs
# ---------------------------------------------------------------------------

def test_add_chunk_nodes_keys_and_edges(art_graph, chunks_df):
    joined, _orphans = partition_chunks(chunks_df, art_graph)
    nodes_added, edges_added = add_chunk_nodes(art_graph, joined)

    assert nodes_added == 4
    assert edges_added == 4

    for row in joined.itertuples(index=False):
        chunk_key = f"CHUNK:{row.chunk_id}"
        art_key = f"ART:{row.doc_uid}"
        assert chunk_key in art_graph.nodes
        assert art_graph.nodes[chunk_key]["type"] == "Chunk"
        assert art_graph.nodes[chunk_key]["doc_uid"] == str(row.doc_uid)

        assert art_graph.has_edge(art_key, chunk_key, key=HAS_CHUNK_EDGE_KEY)
        edge_data = art_graph.edges[art_key, chunk_key, HAS_CHUNK_EDGE_KEY]
        assert edge_data == {"relation": HAS_CHUNK_RELATION}


# ---------------------------------------------------------------------------
# add_chunk_nodes — idempotency
# ---------------------------------------------------------------------------

def test_add_chunk_nodes_idempotent(art_graph, chunks_df):
    joined, _orphans = partition_chunks(chunks_df, art_graph)
    add_chunk_nodes(art_graph, joined)

    nodes_before = art_graph.number_of_nodes()
    edges_before = art_graph.number_of_edges()

    nodes_added, edges_added = add_chunk_nodes(art_graph, joined)
    assert nodes_added == 0
    assert edges_added == 0
    assert art_graph.number_of_nodes() == nodes_before
    assert art_graph.number_of_edges() == edges_before


# ---------------------------------------------------------------------------
# export_orphan_chunks
# ---------------------------------------------------------------------------

def test_export_orphan_chunks_jsonl_shape(tmp_path, art_graph, chunks_df):
    _joined, orphans = partition_chunks(chunks_df, art_graph)
    out_path = tmp_path / "orphan_chunks.jsonl"
    written = export_orphan_chunks(orphans, out_path)

    assert written == 2
    assert out_path.exists()

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    required_keys = {"chunk_id", "doc_uid", "doc_id", "rowidx", "part_idx", "breadcrumb"}
    for line in lines:
        record = json.loads(line)
        assert set(record.keys()) == required_keys


def test_export_orphan_chunks_empty_file(tmp_path):
    out_path = tmp_path / "orphan_chunks.jsonl"
    empty_df = pd.DataFrame(
        columns=["chunk_id", "doc_uid", "doc_id", "rowidx", "part_idx", "breadcrumb"]
    )
    written = export_orphan_chunks(empty_df, out_path)

    assert written == 0
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# run_chunk_quality_gates
# ---------------------------------------------------------------------------

def test_quality_gates_pass_with_relaxed_band(art_graph, chunks_df):
    joined, _orphans = partition_chunks(chunks_df, art_graph)
    add_chunk_nodes(art_graph, joined)
    run_chunk_quality_gates(art_graph, expected_band=(1, 100))


def test_quality_gates_reject_count_outside_band(art_graph, chunks_df):
    joined, _orphans = partition_chunks(chunks_df, art_graph)
    add_chunk_nodes(art_graph, joined)
    with pytest.raises(AssertionError, match="outside acceptance window"):
        run_chunk_quality_gates(
            art_graph, expected_band=(CHUNK_COUNT_MIN, CHUNK_COUNT_MAX)
        )


def test_quality_gates_reject_orphan_chunk(art_graph, chunks_df):
    joined, _orphans = partition_chunks(chunks_df, art_graph)
    add_chunk_nodes(art_graph, joined)

    # Inject an orphan CHUNK (no parent ART edge).
    art_graph.add_node(
        "CHUNK:orphan#0",
        type="Chunk",
        chunk_id="orphan#0",
        doc_uid="orphan",
        doc_id="100",
        rowidx=99,
        part_idx=0,
        breadcrumb=None,
    )
    with pytest.raises(AssertionError, match="have no parent ART via HAS_CHUNK"):
        run_chunk_quality_gates(art_graph, expected_band=(1, 100))


def test_quality_gates_reject_doc_uid_mismatch(art_graph, chunks_df):
    joined, _orphans = partition_chunks(chunks_df, art_graph)
    add_chunk_nodes(art_graph, joined)

    # Tamper with one CHUNK's doc_uid so it no longer matches its ART parent.
    chunk_key = next(
        n for n, d in art_graph.nodes(data=True) if d.get("type") == "Chunk"
    )
    art_graph.nodes[chunk_key]["doc_uid"] = "tampered-uid"
    with pytest.raises(AssertionError, match="doc_uid mismatch"):
        run_chunk_quality_gates(art_graph, expected_band=(1, 100))


# ---------------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------------

def test_cli_stage_5_4_without_append_exits_nonzero():
    """Running --stage 5.4 without --append must fail fast with a clear msg."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--stage", "5.4"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    combined = proc.stderr + proc.stdout
    assert "--append" in combined


def test_cli_stage_5_4_end_to_end(tmp_path, monkeypatch, art_graph):
    """Run the 5.4 CLI path against a pre-built DOC+ART graph.

    The DOC/ART layers are built directly via the helper API and persisted
    so this test exercises only the Stage 5.4 ``main`` surface (loading the
    existing graph, adding CHUNK nodes, persisting, and gates).
    """
    output_path = tmp_path / "kg.gpickle"
    with output_path.open("wb") as f:
        pickle.dump(art_graph, f)

    # Stage 3 chunks: 2 chunks for ART:100, 2 for ART:200.
    uid_100 = "01/2024/QH15|Luật Doanh nghiệp|Điều 1"
    uid_200 = "02/2024/QH15|Luật Thuế GTGT|Điều 1"
    chunk_rows = [
        _chunk_row("100", uid_100, 0),
        _chunk_row("100", uid_100, 1),
        _chunk_row("200", uid_200, 0),
        _chunk_row("200", uid_200, 1),
    ]
    chunks = pd.DataFrame(chunk_rows)
    stage3_path = tmp_path / "stage3.parquet"
    chunks.to_parquet(stage3_path)

    orphan_chunk = tmp_path / "orphan_chunk.jsonl"

    # Loosen the CHUNK band for the small fixture (runner reads it at call-time).
    monkeypatch.setattr("data.stage5_build_graph.CHUNK_COUNT_MIN", 1)
    monkeypatch.setattr("data.stage5_build_graph.CHUNK_COUNT_MAX", 1000)

    assert main(
        ["--stage", "5.4", "--append", "--stage3-path", str(stage3_path),
         "--orphan-chunks-path", str(orphan_chunk),
         "--output-path", str(output_path)]
    ) == 0

    with output_path.open("rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.MultiDiGraph)

    chunk_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Chunk"]
    has_chunk_edges = [
        1 for _u, _v, d in G.edges(data=True)
        if d.get("relation") == HAS_CHUNK_RELATION
    ]
    assert len(chunk_nodes) == 4
    assert len(has_chunk_edges) == 4

    # Every chunk has exactly one parent ART.
    for chunk_key in chunk_nodes:
        parents = [
            u for u, _v, d in G.in_edges(chunk_key, data=True)
            if d.get("relation") == HAS_CHUNK_RELATION
        ]
        assert len(parents) == 1
        assert G.nodes[parents[0]]["type"] == "Article"

    # Orphan chunk file written and empty (no orphans in this fixture).
    assert orphan_chunk.exists()
    assert orphan_chunk.read_text(encoding="utf-8") == ""
