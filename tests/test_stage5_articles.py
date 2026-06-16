"""Unit tests for Stage 5.3 — Article nodes + HAS_ARTICLE edges.

Covers the public surface defined by the openspec change
``stage-5-3-article-nodes`` (specs ``kg-article-layer`` and ``kg-build-cli``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import pandas as pd
import pytest

from data.stage5_build_graph import (
    ART_COUNT_MAX,
    ART_COUNT_MIN,
    HAS_ARTICLE_EDGE_KEY,
    HAS_ARTICLE_RELATION,
    add_article_nodes,
    add_document_nodes,
    build_art_attrs,
    export_orphan_articles,
    main,
    partition_articles,
    run_article_quality_gates,
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
    *,
    phan: str = "",
    chuong: str = "Chương I",
    muc: str = "",
    loai: str = "Luật",
    ngay: str = "01/01/2024",
    start_char: int = 0,
    end_char: int = 100,
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "law_id": law_id,
        "ten_van_ban": ten,
        "loai_van_ban": loai,
        "ngay_ban_hanh": ngay,
        "phan": phan,
        "chuong": chuong,
        "muc": muc,
        "dieu_so": dieu_so,
        "dieu_ten": dieu_ten,
        "noi_dung": f"Nội dung của {dieu_so} thuộc {ten}",
        "start_char": start_char,
        "end_char": end_char,
        "doc_uid": f"{law_id}|{ten}|{dieu_so}",
    }


@pytest.fixture
def doc_graph() -> nx.MultiDiGraph:
    """A graph with two DOC nodes, mirroring a post-Stage 5.2 state."""
    G = nx.MultiDiGraph()
    docs_df = pd.DataFrame(
        [
            _doc_row("100", "01/2024/QH15", "Luật Doanh nghiệp"),
            _doc_row("200", "02/2024/QH15", "Luật Thuế GTGT"),
        ]
    )
    add_document_nodes(G, docs_df)
    return G


@pytest.fixture
def articles_df() -> pd.DataFrame:
    """Synthetic article frame: 3 rows for DOC:100, 2 for DOC:200, 2 orphans."""
    rows = [
        _article_row("100", "01/2024/QH15", "Luật Doanh nghiệp", "Điều 1", "Phạm vi"),
        _article_row("100", "01/2024/QH15", "Luật Doanh nghiệp", "Điều 2", "Giải thích"),
        _article_row("100", "01/2024/QH15", "Luật Doanh nghiệp", "Điều 3", "Áp dụng",
                     phan="", chuong="", muc=""),
        _article_row("200", "02/2024/QH15", "Luật Thuế GTGT", "Điều 1", "Phạm vi"),
        _article_row("200", "02/2024/QH15", "Luật Thuế GTGT", "Điều 5", "Đối tượng"),
        _article_row("999", "99/2024/QH15", "Luật Khác", "Điều 1", "Orphan"),
        _article_row("888", "88/2024/QH15", "Văn bản khác", "Điều 1", "Orphan 2",
                     loai="Thông tư"),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4.2  partition_articles
# ---------------------------------------------------------------------------

def test_partition_articles_split(doc_graph, articles_df):
    joined, orphans = partition_articles(articles_df, doc_graph)
    assert len(joined) == 5
    assert len(orphans) == 2
    assert set(joined["doc_id"].astype(str)) == {"100", "200"}
    assert set(orphans["doc_id"].astype(str)) == {"999", "888"}


# ---------------------------------------------------------------------------
# 4.3  build_art_attrs
# ---------------------------------------------------------------------------

def test_build_art_attrs_locked_schema(articles_df):
    row = articles_df.iloc[2]  # DOC:100 / Điều 3 with empty phan/chuong/muc
    attrs = build_art_attrs(row)

    expected_keys = {
        "type",
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
    }
    assert set(attrs.keys()) == expected_keys
    assert attrs["type"] == "Article"
    assert "noi_dung" not in attrs

    # Empty strings should normalize to None.
    assert attrs["phan"] is None
    assert attrs["chuong"] is None
    assert attrs["muc"] is None

    # Non-empty strings are preserved verbatim.
    assert attrs["dieu_so"] == "Điều 3"
    assert attrs["dieu_ten"] == "Áp dụng"
    assert attrs["law_id"] == "01/2024/QH15"


def test_build_art_attrs_preserves_chuong_when_present(articles_df):
    row = articles_df.iloc[0]  # chuong="Chương I"
    attrs = build_art_attrs(row)
    assert attrs["chuong"] == "Chương I"
    assert attrs["phan"] is None
    assert attrs["muc"] is None


# ---------------------------------------------------------------------------
# 4.4  add_article_nodes — keys, edge attrs, idempotency
# ---------------------------------------------------------------------------

def test_add_article_nodes_keys_and_edges(doc_graph, articles_df):
    joined, _orphans = partition_articles(articles_df, doc_graph)
    nodes_added, edges_added = add_article_nodes(doc_graph, joined)

    assert nodes_added == 5
    assert edges_added == 5

    # Every joined row produced an ART:{doc_uid} node.
    for row in joined.itertuples(index=False):
        art_key = f"ART:{row.doc_uid}"
        assert art_key in doc_graph.nodes
        assert doc_graph.nodes[art_key]["type"] == "Article"
        assert doc_graph.nodes[art_key]["doc_id"] == str(row.doc_id)

        # Edge from DOC to ART with explicit key and minimal data.
        doc_key = f"DOC:{row.doc_id}"
        assert doc_graph.has_edge(doc_key, art_key, key=HAS_ARTICLE_EDGE_KEY)
        edge_data = doc_graph.edges[doc_key, art_key, HAS_ARTICLE_EDGE_KEY]
        assert edge_data == {"relation": HAS_ARTICLE_RELATION}


# ---------------------------------------------------------------------------
# 4.5  add_article_nodes — idempotency on re-run
# ---------------------------------------------------------------------------

def test_add_article_nodes_idempotent(doc_graph, articles_df):
    joined, _orphans = partition_articles(articles_df, doc_graph)
    add_article_nodes(doc_graph, joined)

    nodes_before = doc_graph.number_of_nodes()
    edges_before = doc_graph.number_of_edges()

    nodes_added, edges_added = add_article_nodes(doc_graph, joined)
    assert nodes_added == 0
    assert edges_added == 0
    assert doc_graph.number_of_nodes() == nodes_before
    assert doc_graph.number_of_edges() == edges_before


# ---------------------------------------------------------------------------
# 4.6 / 4.7  export_orphan_articles
# ---------------------------------------------------------------------------

def test_export_orphan_articles_jsonl_shape(tmp_path, doc_graph, articles_df):
    _joined, orphans = partition_articles(articles_df, doc_graph)
    out_path = tmp_path / "orphans.jsonl"
    written = export_orphan_articles(orphans, out_path)

    assert written == 2
    assert out_path.exists()

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    required_keys = {"doc_uid", "doc_id", "law_id", "ten_van_ban", "dieu_so", "dieu_ten"}
    for line in lines:
        record = json.loads(line)
        assert set(record.keys()) == required_keys


def test_export_orphan_articles_empty_file(tmp_path):
    out_path = tmp_path / "orphans.jsonl"
    empty_df = pd.DataFrame(
        columns=["doc_uid", "doc_id", "law_id", "ten_van_ban", "dieu_so", "dieu_ten"]
    )
    written = export_orphan_articles(empty_df, out_path)

    assert written == 0
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# 4.8  run_article_quality_gates
# ---------------------------------------------------------------------------

def test_quality_gates_pass_with_relaxed_band(doc_graph, articles_df):
    joined, _orphans = partition_articles(articles_df, doc_graph)
    add_article_nodes(doc_graph, joined)
    # Relax band so the 5-article fixture is acceptable.
    run_article_quality_gates(doc_graph, expected_band=(1, 100))


def test_quality_gates_reject_count_outside_band(doc_graph, articles_df):
    joined, _orphans = partition_articles(articles_df, doc_graph)
    add_article_nodes(doc_graph, joined)
    with pytest.raises(AssertionError, match="outside acceptance window"):
        run_article_quality_gates(doc_graph, expected_band=(ART_COUNT_MIN, ART_COUNT_MAX))


def test_quality_gates_reject_orphan_art(doc_graph, articles_df):
    joined, _orphans = partition_articles(articles_df, doc_graph)
    add_article_nodes(doc_graph, joined)

    # Inject an orphan ART (not connected to any DOC).
    orphan_attrs = {
        "type": "Article",
        "doc_uid": "X|Y|Z",
        "doc_id": "100",
        "law_id": "01/2024/QH15",
        "ten_van_ban": "Luật Doanh nghiệp",
        "dieu_so": "Điều X",
        "dieu_ten": "Orphan injected",
        "phan": None,
        "chuong": None,
        "muc": None,
        "loai_van_ban": "Luật",
        "ngay_ban_hanh": "01/01/2024",
        "start_char": 0,
        "end_char": 1,
    }
    doc_graph.add_node("ART:X|Y|Z", **orphan_attrs)
    with pytest.raises(AssertionError, match="have no parent DOC via HAS_ARTICLE"):
        run_article_quality_gates(doc_graph, expected_band=(1, 100))


def test_quality_gates_reject_doc_id_mismatch(doc_graph, articles_df):
    joined, _orphans = partition_articles(articles_df, doc_graph)
    add_article_nodes(doc_graph, joined)

    # Tamper with one ART's doc_id so it no longer matches its DOC parent.
    art_key = next(n for n, d in doc_graph.nodes(data=True) if d.get("type") == "Article")
    doc_graph.nodes[art_key]["doc_id"] = "999"
    with pytest.raises(AssertionError, match="doc_id mismatch"):
        run_article_quality_gates(doc_graph, expected_band=(1, 100))


# ---------------------------------------------------------------------------
# 4.9  CLI without --append
# ---------------------------------------------------------------------------

def test_cli_stage_5_3_without_append_exits_nonzero():
    """Running --stage 5.3 without --append must fail fast with a clear msg."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--stage", "5.3"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    combined = proc.stderr + proc.stdout
    assert "--append" in combined


def test_cli_no_stage_argument_fails():
    """Missing --stage is a usage error per the kg-build-cli spec."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "--stage" in proc.stderr


# ---------------------------------------------------------------------------
# 4.10  Stage 5.2 round-trip via the new CLI surface
# ---------------------------------------------------------------------------

def test_cli_stage_5_2_dispatches_doc_builder(tmp_path, monkeypatch):
    """--stage 5.2 still produces a DOC-only graph with the expected schema."""
    docs_df = pd.DataFrame(
        [_doc_row(f"{i}", f"{i}/2024/QH15", f"Luật {i}") for i in range(3000, 3010)]
    )
    stage1_path = tmp_path / "stage1.parquet"
    docs_df.to_parquet(stage1_path)
    output_path = tmp_path / "kg.gpickle"

    # Loosen the DOC-count gate for the small test fixture.
    monkeypatch.setattr("data.stage5_build_graph.DOC_COUNT_MIN", 1)
    monkeypatch.setattr("data.stage5_build_graph.DOC_COUNT_MAX", 1000)

    rc = main(
        [
            "--stage",
            "5.2",
            "--stage1-path",
            str(stage1_path),
            "--output-path",
            str(output_path),
        ]
    )
    assert rc == 0
    assert output_path.exists()

    import pickle

    with output_path.open("rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.MultiDiGraph)
    doc_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Document"]
    assert len(doc_nodes) == 10
    sample = G.nodes["DOC:3000"]
    assert sample["law_id"] == "3000/2024/QH15"
    assert sample["ten"] == "Luật 3000"
