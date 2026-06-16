"""Unit tests for Stage 5.5 — Cross-document (DOC -> DOC) edges.

Covers the public surface added for PLAN.md task 5.5 (KG.md §5, §7.1, §8.3
and G-LRAG_SPECIFICATIONS.md §8.6 Step 4): relationship-mapping loading,
relationships-dataset loading, SME-endpoint filter, label mapping, edge
creation (canonical direction only, idempotent, no self-loops), dropped-row
JSONL export, and the doc-doc quality gates.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import pandas as pd
import pytest

from data.stage5_build_graph import (
    DOC_DOC_EDGE_COUNT_MAX,
    DOC_DOC_EDGE_COUNT_MIN,
    add_doc_doc_edges,
    add_document_nodes,
    collect_doc_ids,
    export_dropped_relationships,
    filter_relationships_by_sme,
    load_relationship_mapping,
    load_relationships,
    main,
    map_relationship_label,
    run_doc_doc_quality_gates,
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


@pytest.fixture
def doc_graph() -> nx.MultiDiGraph:
    """A graph with three DOC nodes — mirroring a post-Stage 5.2 state."""
    G = nx.MultiDiGraph()
    docs_df = pd.DataFrame(
        [
            _doc_row("100", "01/2024/QH15", "Luật Doanh nghiệp"),
            _doc_row("200", "02/2024/QH15", "Luật Thuế GTGT"),
            _doc_row("300", "03/2024/QH15", "Luật Hỗ trợ DNNVV"),
        ]
    )
    add_document_nodes(G, docs_df)
    return G


@pytest.fixture
def mapping_yaml(tmp_path: Path) -> Path:
    """Minimal mapping YAML with canonical labels + one dropped reverse label."""
    content = textwrap.dedent(
        """
        RELATIONSHIP_MAP:
          "Văn bản sửa đổi": "AMENDS"
          "Văn bản dẫn chiếu": "CITES_REF"
          "Văn bản căn cứ": "BASED_ON"
          "Văn bản HD, QĐ chi tiết": "DETAILS"
          "Văn bản hết hiệu lực": "REPLACES"
          # Mapped but NOT whitelisted -> must be dropped.
          "Some experimental label": "EXPERIMENTAL"

        RELATION_WHITELIST:
          - "AMENDS"
          - "CITES_REF"
          - "BASED_ON"
          - "DETAILS"
          - "REPLACES"

        DROPPED_LABELS:
          - "Văn bản được sửa đổi"
        """
    ).strip() + "\n"
    p = tmp_path / "mapping.yaml"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def rels_jsonl(tmp_path: Path) -> Path:
    """Synthetic relationships JSONL covering every behaviour we care about."""
    rows = [
        # Canonical: 100 amends 200 — KEEP as AMENDS.
        {"doc_id": 100, "other_doc_id": "200", "relationship": "Văn bản sửa đổi"},
        # Canonical: 200 cites 300 — KEEP as CITES_REF.
        {"doc_id": 200, "other_doc_id": "300", "relationship": "Văn bản dẫn chiếu"},
        # Canonical: 100 based on 300 — KEEP as BASED_ON.
        {"doc_id": 100, "other_doc_id": "300", "relationship": "Văn bản căn cứ"},
        # Reverse-only label — DROP (not in mapping).
        {"doc_id": 200, "other_doc_id": "100", "relationship": "Văn bản được sửa đổi"},
        # Mapped but not whitelisted — DROP.
        {"doc_id": 100, "other_doc_id": "200", "relationship": "Some experimental label"},
        # Self-loop — DROP.
        {"doc_id": 100, "other_doc_id": "100", "relationship": "Văn bản dẫn chiếu"},
        # Endpoint not in SME graph (999) — DROP at the SME filter step.
        {"doc_id": 100, "other_doc_id": "999", "relationship": "Văn bản sửa đổi"},
        # Duplicate of the first row — should NOT add a new edge.
        {"doc_id": 100, "other_doc_id": "200", "relationship": "Văn bản sửa đổi"},
    ]
    p = tmp_path / "relationships.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


# ---------------------------------------------------------------------------
# load_relationship_mapping
# ---------------------------------------------------------------------------

def test_load_relationship_mapping_shapes(mapping_yaml):
    cfg = load_relationship_mapping(mapping_yaml)
    assert isinstance(cfg["RELATIONSHIP_MAP"], dict)
    assert isinstance(cfg["RELATION_WHITELIST"], set)
    assert isinstance(cfg["DROPPED_LABELS"], set)
    assert cfg["RELATIONSHIP_MAP"]["Văn bản sửa đổi"] == "AMENDS"
    assert "AMENDS" in cfg["RELATION_WHITELIST"]
    assert "Văn bản được sửa đổi" in cfg["DROPPED_LABELS"]


def test_load_relationship_mapping_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_relationship_mapping(tmp_path / "does-not-exist.yaml")


def test_load_relationship_mapping_rejects_empty_map(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("RELATIONSHIP_MAP: {}\nRELATION_WHITELIST: [\"X\"]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RELATIONSHIP_MAP"):
        load_relationship_mapping(p)


def test_load_relationship_mapping_rejects_empty_whitelist(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        'RELATIONSHIP_MAP:\n  "x": "Y"\nRELATION_WHITELIST: []\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="RELATION_WHITELIST"):
        load_relationship_mapping(p)


# ---------------------------------------------------------------------------
# load_relationships
# ---------------------------------------------------------------------------

def test_load_relationships_jsonl(rels_jsonl):
    df = load_relationships(rels_jsonl)
    assert {"doc_id", "other_doc_id", "relationship"}.issubset(df.columns)
    assert len(df) == 8
    # Stringified IDs across the board.
    assert df["doc_id"].dtype == object
    assert df["other_doc_id"].dtype == object
    assert df["doc_id"].iloc[0] == "100"
    assert df["other_doc_id"].iloc[0] == "200"


def test_load_relationships_drops_empty_rows(tmp_path):
    p = tmp_path / "rels.jsonl"
    rows = [
        {"doc_id": 100, "other_doc_id": "200", "relationship": "Văn bản sửa đổi"},
        # Empty relationship label — must be dropped by the loader.
        {"doc_id": 100, "other_doc_id": "200", "relationship": ""},
    ]
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    df = load_relationships(p)
    assert len(df) == 1


def test_load_relationships_unsupported_extension(tmp_path):
    p = tmp_path / "rels.txt"
    p.write_text("anything", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported relationships file extension"):
        load_relationships(p)


def test_load_relationships_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_relationships(tmp_path / "missing.jsonl")


# ---------------------------------------------------------------------------
# collect_doc_ids
# ---------------------------------------------------------------------------

def test_collect_doc_ids(doc_graph):
    ids = collect_doc_ids(doc_graph)
    assert ids == {"100", "200", "300"}


def test_collect_doc_ids_ignores_non_doc_nodes(doc_graph):
    doc_graph.add_node("ART:foo|bar|Điều 1", type="Article", doc_uid="foo|bar|Điều 1")
    ids = collect_doc_ids(doc_graph)
    assert ids == {"100", "200", "300"}


# ---------------------------------------------------------------------------
# filter_relationships_by_sme
# ---------------------------------------------------------------------------

def test_filter_relationships_by_sme_split(rels_jsonl):
    rels = load_relationships(rels_jsonl)
    sme_ids = {"100", "200", "300"}
    kept, dropped_count = filter_relationships_by_sme(rels, sme_ids)
    # Only the row pointing at "999" should be dropped here.
    assert len(kept) == len(rels) - 1
    assert dropped_count == 1
    assert not (kept["other_doc_id"] == "999").any()


def test_filter_relationships_by_sme_empty_set(rels_jsonl):
    rels = load_relationships(rels_jsonl)
    kept, dropped = filter_relationships_by_sme(rels, set())
    assert len(kept) == 0
    assert dropped == len(rels)


# ---------------------------------------------------------------------------
# map_relationship_label
# ---------------------------------------------------------------------------

def test_map_relationship_label_keeps_whitelisted():
    rel_map = {"x": "AMENDS"}
    whitelist = {"AMENDS"}
    assert map_relationship_label("x", rel_map, whitelist) == "AMENDS"


def test_map_relationship_label_drops_unmapped():
    assert map_relationship_label("nope", {"x": "AMENDS"}, {"AMENDS"}) is None


def test_map_relationship_label_drops_non_whitelisted():
    rel_map = {"x": "EXPERIMENTAL"}
    whitelist = {"AMENDS"}
    assert map_relationship_label("x", rel_map, whitelist) is None


# ---------------------------------------------------------------------------
# add_doc_doc_edges
# ---------------------------------------------------------------------------

def test_add_doc_doc_edges_keeps_canonical_drops_others(doc_graph, mapping_yaml, rels_jsonl):
    cfg = load_relationship_mapping(mapping_yaml)
    rels = load_relationships(rels_jsonl)
    sme_ids = collect_doc_ids(doc_graph)
    sme_rels, _ = filter_relationships_by_sme(rels, sme_ids)

    edges_added, kept_counts, dropped_records = add_doc_doc_edges(
        doc_graph, sme_rels, cfg["RELATIONSHIP_MAP"], cfg["RELATION_WHITELIST"]
    )

    # 3 unique canonical edges: AMENDS(100,200), CITES_REF(200,300),
    # BASED_ON(100,300). The duplicate "Văn bản sửa đổi" row counts in
    # kept_counts but does NOT increment edges_added.
    assert edges_added == 3
    assert kept_counts == {"AMENDS": 2, "CITES_REF": 1, "BASED_ON": 1}

    # Each canonical edge must be present with key == relation.
    assert doc_graph.has_edge("DOC:100", "DOC:200", key="AMENDS")
    assert doc_graph.has_edge("DOC:200", "DOC:300", key="CITES_REF")
    assert doc_graph.has_edge("DOC:100", "DOC:300", key="BASED_ON")
    assert doc_graph.edges["DOC:100", "DOC:200", "AMENDS"] == {"relation": "AMENDS"}

    # Reverse direction must NOT be stored.
    assert not doc_graph.has_edge("DOC:200", "DOC:100", key="AMENDS")

    # Dropped records: reverse label, non-whitelisted, self-loop.
    reasons = {r["reason"] for r in dropped_records}
    assert "self_loop" in reasons
    assert "label_not_in_whitelist" in reasons


def test_add_doc_doc_edges_idempotent(doc_graph, mapping_yaml, rels_jsonl):
    cfg = load_relationship_mapping(mapping_yaml)
    rels = load_relationships(rels_jsonl)
    sme_ids = collect_doc_ids(doc_graph)
    sme_rels, _ = filter_relationships_by_sme(rels, sme_ids)

    add_doc_doc_edges(
        doc_graph, sme_rels, cfg["RELATIONSHIP_MAP"], cfg["RELATION_WHITELIST"]
    )
    nodes_before = doc_graph.number_of_nodes()
    edges_before = doc_graph.number_of_edges()

    edges_added, _, _ = add_doc_doc_edges(
        doc_graph, sme_rels, cfg["RELATIONSHIP_MAP"], cfg["RELATION_WHITELIST"]
    )
    assert edges_added == 0
    assert doc_graph.number_of_nodes() == nodes_before
    assert doc_graph.number_of_edges() == edges_before


# ---------------------------------------------------------------------------
# export_dropped_relationships
# ---------------------------------------------------------------------------

def test_export_dropped_relationships_writes_lines(tmp_path):
    out = tmp_path / "dropped.jsonl"
    records = [
        {"doc_id": "1", "other_doc_id": "2", "relationship": "x", "reason": "self_loop"},
        {"doc_id": "1", "other_doc_id": "3", "relationship": "y", "reason": "label_not_in_whitelist"},
    ]
    n = export_dropped_relationships(records, out)
    assert n == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["reason"] == "self_loop"
    assert parsed[1]["reason"] == "label_not_in_whitelist"


def test_export_dropped_relationships_empty(tmp_path):
    out = tmp_path / "dropped.jsonl"
    n = export_dropped_relationships([], out)
    assert n == 0
    assert out.exists()
    assert out.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# run_doc_doc_quality_gates
# ---------------------------------------------------------------------------

def test_quality_gates_pass_with_relaxed_band(doc_graph, mapping_yaml, rels_jsonl):
    cfg = load_relationship_mapping(mapping_yaml)
    rels = load_relationships(rels_jsonl)
    sme_rels, _ = filter_relationships_by_sme(rels, collect_doc_ids(doc_graph))
    add_doc_doc_edges(doc_graph, sme_rels, cfg["RELATIONSHIP_MAP"], cfg["RELATION_WHITELIST"])

    run_doc_doc_quality_gates(
        doc_graph, whitelist=cfg["RELATION_WHITELIST"], expected_band=(1, 100)
    )


def test_quality_gates_reject_count_outside_band(doc_graph, mapping_yaml, rels_jsonl):
    cfg = load_relationship_mapping(mapping_yaml)
    rels = load_relationships(rels_jsonl)
    sme_rels, _ = filter_relationships_by_sme(rels, collect_doc_ids(doc_graph))
    add_doc_doc_edges(doc_graph, sme_rels, cfg["RELATIONSHIP_MAP"], cfg["RELATION_WHITELIST"])

    with pytest.raises(AssertionError, match="outside acceptance window"):
        run_doc_doc_quality_gates(
            doc_graph,
            whitelist=cfg["RELATION_WHITELIST"],
            expected_band=(DOC_DOC_EDGE_COUNT_MIN, DOC_DOC_EDGE_COUNT_MAX),
        )


def test_quality_gates_reject_self_loop(doc_graph):
    doc_graph.add_edge("DOC:100", "DOC:100", key="AMENDS", relation="AMENDS")
    with pytest.raises(AssertionError, match="Self-loop"):
        run_doc_doc_quality_gates(
            doc_graph, whitelist={"AMENDS"}, expected_band=(1, 100)
        )


def test_quality_gates_reject_relation_not_in_whitelist(doc_graph):
    doc_graph.add_edge("DOC:100", "DOC:200", key="MYSTERY", relation="MYSTERY")
    with pytest.raises(AssertionError, match="not in whitelist"):
        run_doc_doc_quality_gates(
            doc_graph, whitelist={"AMENDS"}, expected_band=(1, 100)
        )


def test_quality_gates_reject_key_relation_mismatch(doc_graph):
    # Edge stored with key != relation — violates canonical-direction storage.
    doc_graph.add_edge("DOC:100", "DOC:200", key="WRONG_KEY", relation="AMENDS")
    with pytest.raises(AssertionError, match="does not match"):
        run_doc_doc_quality_gates(
            doc_graph, whitelist={"AMENDS"}, expected_band=(1, 100)
        )


# ---------------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------------

def test_cli_stage_5_5_without_append_exits_nonzero():
    """Running --stage 5.5 without --append must fail fast with a clear msg."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--stage", "5.5"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    combined = proc.stderr + proc.stdout
    assert "--append" in combined


def test_cli_stage_5_5_end_to_end(
    tmp_path, monkeypatch, doc_graph, mapping_yaml, rels_jsonl
):
    """Run the 5.5 CLI path against a pre-built DOC-only graph.

    The DOC layer is built directly via the helper API and persisted so this
    test exercises only the Stage 5.5 ``main`` surface (loading the existing
    graph, filtering relationships, adding edges, persisting, and gates).
    """
    output_path = tmp_path / "kg.gpickle"
    with output_path.open("wb") as f:
        pickle.dump(doc_graph, f)

    dropped_path = tmp_path / "dropped.jsonl"

    # Loosen the doc-doc band for the small fixture.
    monkeypatch.setattr("data.stage5_build_graph.DOC_DOC_EDGE_COUNT_MIN", 1)
    monkeypatch.setattr("data.stage5_build_graph.DOC_DOC_EDGE_COUNT_MAX", 1000)

    rc = main(
        [
            "--stage",
            "5.5",
            "--append",
            "--relationships-path",
            str(rels_jsonl),
            "--relationship-mapping-path",
            str(mapping_yaml),
            "--dropped-relationships-path",
            str(dropped_path),
            "--output-path",
            str(output_path),
        ]
    )
    assert rc == 0

    with output_path.open("rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.MultiDiGraph)

    doc_doc_edges = [
        (u, v, k, d)
        for u, v, k, d in G.edges(keys=True, data=True)
        if (
            G.nodes.get(u, {}).get("type") == "Document"
            and G.nodes.get(v, {}).get("type") == "Document"
        )
    ]
    assert len(doc_doc_edges) == 3

    # Canonical edges are present.
    assert G.has_edge("DOC:100", "DOC:200", key="AMENDS")
    assert G.has_edge("DOC:200", "DOC:300", key="CITES_REF")
    assert G.has_edge("DOC:100", "DOC:300", key="BASED_ON")

    # Dropped JSONL was written and includes a self-loop and a non-whitelisted row.
    assert dropped_path.exists()
    dropped = [json.loads(line) for line in dropped_path.read_text(encoding="utf-8").splitlines()]
    reasons = {rec["reason"] for rec in dropped}
    assert "self_loop" in reasons
    assert "label_not_in_whitelist" in reasons
