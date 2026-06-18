"""Unit tests for Stage 5.7 / 5.8 final graph persistence and validation."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import pandas as pd
import pytest

from data.stage5_build_graph import (
    MENTIONS_RELATION,
    add_article_nodes,
    add_chunk_nodes,
    add_concept_nodes,
    add_document_nodes,
    add_mentions_edges,
    main,
    partition_articles,
    persist_graph,
    run_full_graph_quality_gates,
)


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


def _article_row(doc_id: str, law_id: str, ten: str, dieu_so: str) -> Dict[str, Any]:
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
        "dieu_ten": "Phạm vi",
        "noi_dung": f"Nội dung của {dieu_so} thuộc {ten}",
        "start_char": 0,
        "end_char": 100,
        "doc_uid": f"{law_id}|{ten}|{dieu_so}",
    }


def _chunk_row(doc_id: str, doc_uid: str, part_idx: int, text: str) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "doc_uid": doc_uid,
        "breadcrumb": "Luật > Chương I",
        "chunk_id": f"{doc_uid}#{part_idx}",
        "part_idx": part_idx,
        "rowidx": part_idx,
        "chunk_text": text,
    }


@pytest.fixture
def complete_graph(monkeypatch) -> nx.MultiDiGraph:
    monkeypatch.setattr("data.stage5_build_graph.DOC_COUNT_MIN", 1)
    monkeypatch.setattr("data.stage5_build_graph.DOC_COUNT_MAX", 100)
    monkeypatch.setattr("data.stage5_build_graph.ART_COUNT_MIN", 1)
    monkeypatch.setattr("data.stage5_build_graph.ART_COUNT_MAX", 100)
    monkeypatch.setattr("data.stage5_build_graph.CHUNK_COUNT_MIN", 1)
    monkeypatch.setattr("data.stage5_build_graph.CHUNK_COUNT_MAX", 100)
    monkeypatch.setattr("data.stage5_build_graph.CONCEPT_COUNT_MIN", 1)
    monkeypatch.setattr("data.stage5_build_graph.CONCEPT_COUNT_MAX", 100)
    monkeypatch.setattr("data.stage5_build_graph.DOC_DOC_EDGE_COUNT_MIN", 1)
    monkeypatch.setattr("data.stage5_build_graph.DOC_DOC_EDGE_COUNT_MAX", 100)

    G = nx.MultiDiGraph()
    docs = pd.DataFrame(
        [
            _doc_row("100", "01/2024/QH15", "Luật Doanh nghiệp"),
            _doc_row("200", "02/2024/QH15", "Luật Thuế GTGT"),
        ]
    )
    add_document_nodes(G, docs)

    articles = pd.DataFrame(
        [
            _article_row("100", "01/2024/QH15", "Luật Doanh nghiệp", "Điều 1"),
            _article_row("200", "02/2024/QH15", "Luật Thuế GTGT", "Điều 1"),
        ]
    )
    joined, _orphans = partition_articles(articles, G)
    add_article_nodes(G, joined)

    chunks = []
    for row in joined.itertuples(index=False):
        chunks.append(
            _chunk_row(
                row.doc_id,
                row.doc_uid,
                0,
                "Doanh nghiệp có vốn điều lệ và sử dụng hóa đơn điện tử.",
            )
        )
    chunks_df = pd.DataFrame(chunks)
    add_chunk_nodes(G, chunks_df)

    concepts = ["vốn điều lệ", "hóa đơn điện tử"]
    add_concept_nodes(G, concepts)
    add_mentions_edges(G, chunks_df, concepts)
    G.add_edge("DOC:100", "DOC:200", key="AMENDS", relation="AMENDS")
    return G


def test_stage_5_7_repersists_with_highest_protocol(tmp_path, complete_graph):
    output_path = tmp_path / "kg.gpickle"
    persist_graph(complete_graph, output_path)

    assert main(["--stage", "5.7", "--append", "--output-path", str(output_path)]) == 0

    first_two = output_path.read_bytes()[:2]
    assert first_two == bytes([0x80, pickle.HIGHEST_PROTOCOL])
    with output_path.open("rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, nx.MultiDiGraph)
    assert loaded.number_of_nodes() == complete_graph.number_of_nodes()
    assert loaded.number_of_edges() == complete_graph.number_of_edges()


def test_stage_5_8_full_graph_quality_gates_pass(complete_graph):
    run_full_graph_quality_gates(complete_graph, whitelist={"AMENDS"})


def test_stage_5_8_rejects_duplicate_chunk_parent(complete_graph):
    chunk_key = next(n for n, d in complete_graph.nodes(data=True) if d.get("type") == "Chunk")
    other_art = next(
        n
        for n, d in complete_graph.nodes(data=True)
        if d.get("type") == "Article" and n != next(complete_graph.predecessors(chunk_key))
    )
    complete_graph.add_edge(other_art, chunk_key, key="HAS_CHUNK_DUP", relation="HAS_CHUNK")

    with pytest.raises(
        AssertionError,
        match="unexpected edge key|doc_uid mismatch|exactly one parent ART",
    ):
        run_full_graph_quality_gates(complete_graph, whitelist={"AMENDS"})


def test_stage_5_8_rejects_art_to_concept_edge(complete_graph):
    art_key = next(n for n, d in complete_graph.nodes(data=True) if d.get("type") == "Article")
    concept_key = next(n for n, d in complete_graph.nodes(data=True) if d.get("type") == "Concept")
    complete_graph.add_edge(art_key, concept_key, key="BAD", relation=MENTIONS_RELATION)

    with pytest.raises(AssertionError, match="source is not a Chunk node"):
        run_full_graph_quality_gates(complete_graph, whitelist={"AMENDS"})


def test_stage_5_8_cli_validates_existing_graph(tmp_path, complete_graph):
    output_path = tmp_path / "kg.gpickle"
    mapping_path = tmp_path / "relationship_mapping.yaml"
    mapping_path.write_text(
        "RELATIONSHIP_MAP:\n  Văn bản sửa đổi: AMENDS\n"
        "RELATION_WHITELIST:\n  - AMENDS\n"
        "DROPPED_LABELS: []\n",
        encoding="utf-8",
    )
    persist_graph(complete_graph, output_path)

    assert main(
        [
            "--stage",
            "5.8",
            "--append",
            "--output-path",
            str(output_path),
            "--relationship-mapping-path",
            str(mapping_path),
        ]
    ) == 0
