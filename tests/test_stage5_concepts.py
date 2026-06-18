"""Unit tests for Stage 5.6 — CONCEPT nodes + chunk-first MENTIONS edges."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import pandas as pd
import pytest

from data.stage5_build_graph import (
    MENTIONS_EDGE_KEY,
    MENTIONS_RELATION,
    _canonicalize_llm_concepts,
    add_article_nodes,
    add_chunk_nodes,
    add_concept_nodes,
    add_document_nodes,
    add_mentions_edges,
    build_concept_attrs,
    load_legal_concepts,
    main,
    match_concepts_in_text,
    partition_articles,
    run_concept_quality_gates,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
        "breadcrumb": "Luật > Chương I",
        "chunk_id": f"{doc_uid}#{part_idx}",
        "part_idx": part_idx,
        "chunk_text": text,
    }


@pytest.fixture
def concepts() -> List[str]:
    return [
        "vốn điều lệ",
        "hóa đơn điện tử",
        "thuế giá trị gia tăng",
        "người lao động",
    ]


@pytest.fixture
def chunk_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    docs_df = pd.DataFrame([_doc_row("100", "01/2024/QH15", "Luật Doanh nghiệp")])
    add_document_nodes(G, docs_df)
    articles_df = pd.DataFrame([
        _article_row("100", "01/2024/QH15", "Luật Doanh nghiệp", "Điều 1")
    ])
    joined, _orphans = partition_articles(articles_df, G)
    add_article_nodes(G, joined)
    doc_uid = "01/2024/QH15|Luật Doanh nghiệp|Điều 1"
    chunks_df = pd.DataFrame(
        [
            _chunk_row(
                "100",
                doc_uid,
                0,
                "Doanh nghiệp có vốn điều lệ và sử dụng hóa đơn điện tử.",
            ),
            _chunk_row(
                "100",
                doc_uid,
                1,
                "Thuế giá trị gia tăng áp dụng với hàng hóa, dịch vụ.",
            ),
        ]
    )
    chunks_df["rowidx"] = range(len(chunks_df))
    add_chunk_nodes(G, chunks_df)
    return G


@pytest.fixture
def chunks_df() -> pd.DataFrame:
    doc_uid = "01/2024/QH15|Luật Doanh nghiệp|Điều 1"
    rows = [
        _chunk_row(
            "100",
            doc_uid,
            0,
            "Doanh nghiệp có vốn điều lệ và sử dụng hóa đơn điện tử.",
        ),
        _chunk_row(
            "100",
            doc_uid,
            1,
            "Thuế giá trị gia tăng áp dụng với hàng hóa, dịch vụ.",
        ),
    ]
    df = pd.DataFrame(rows)
    df["rowidx"] = range(len(df))
    return df


def test_build_concept_attrs_locked_schema():
    attrs = build_concept_attrs("Hóa đơn điện tử")
    assert attrs == {
        "type": "Concept",
        "name": "Hóa đơn điện tử",
        "name_lower": "hóa đơn điện tử",
    }


def test_match_concepts_in_text_is_case_insensitive_and_limited(concepts, monkeypatch):
    monkeypatch.setattr("data.stage5_build_graph.MAX_CONCEPTS_PER_CHUNK", 2)
    text = "VỐN ĐIỀU LỆ, hóa đơn điện tử và thuế giá trị gia tăng đều xuất hiện."
    assert match_concepts_in_text(text, concepts) == ["vốn điều lệ", "hóa đơn điện tử"]


def test_add_concept_nodes_idempotent(chunk_graph, concepts):
    added = add_concept_nodes(chunk_graph, concepts)
    assert added == 4
    assert chunk_graph.nodes["CONCEPT:vốn điều lệ"]["type"] == "Concept"

    nodes_before = chunk_graph.number_of_nodes()
    assert add_concept_nodes(chunk_graph, concepts) == 0
    assert chunk_graph.number_of_nodes() == nodes_before


def test_add_mentions_edges_chunk_first_only(chunk_graph, chunks_df, concepts):
    add_concept_nodes(chunk_graph, concepts)
    edges_added, chunks_with_mentions = add_mentions_edges(chunk_graph, chunks_df, concepts)

    assert edges_added == 3
    assert chunks_with_mentions == 2
    for _u, _v, key, data in chunk_graph.edges(keys=True, data=True):
        if data.get("relation") == MENTIONS_RELATION:
            assert key == MENTIONS_EDGE_KEY
            assert chunk_graph.nodes[_u]["type"] == "Chunk"
            assert chunk_graph.nodes[_v]["type"] == "Concept"

    run_concept_quality_gates(chunk_graph, expected_band=(1, 10))


def test_add_mentions_edges_idempotent(chunk_graph, chunks_df, concepts):
    add_concept_nodes(chunk_graph, concepts)
    add_mentions_edges(chunk_graph, chunks_df, concepts)
    nodes_before = chunk_graph.number_of_nodes()
    edges_before = chunk_graph.number_of_edges()

    edges_added, chunks_with_mentions = add_mentions_edges(chunk_graph, chunks_df, concepts)
    assert edges_added == 0
    assert chunks_with_mentions == 2
    assert chunk_graph.number_of_nodes() == nodes_before
    assert chunk_graph.number_of_edges() == edges_before


def test_canonicalize_llm_concepts_filters_to_curated_names(concepts):
    concepts_by_lower = {concept.lower(): concept for concept in concepts}
    assert _canonicalize_llm_concepts(
        ["Vốn Điều Lệ", "không có trong registry", "hóa đơn điện tử", "vốn điều lệ"],
        concepts_by_lower,
    ) == ["vốn điều lệ", "hóa đơn điện tử"]


def test_quality_gates_reject_art_to_concept(chunk_graph, concepts):
    add_concept_nodes(chunk_graph, concepts)
    art_key = next(n for n, d in chunk_graph.nodes(data=True) if d.get("type") == "Article")
    chunk_graph.add_edge(art_key, "CONCEPT:vốn điều lệ", relation=MENTIONS_RELATION)

    with pytest.raises(AssertionError, match="source is not a Chunk node"):
        run_concept_quality_gates(chunk_graph, expected_band=(1, 10))


def test_load_legal_concepts_validates_count(tmp_path, monkeypatch):
    path = tmp_path / "concepts.yaml"
    path.write_text(
        "LEGAL_CONCEPTS:\n  - vốn điều lệ\n  - hóa đơn điện tử\n  - vốn điều lệ\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("data.stage5_build_graph.CONCEPT_COUNT_MIN", 1)
    monkeypatch.setattr("data.stage5_build_graph.CONCEPT_COUNT_MAX", 10)
    assert load_legal_concepts(path) == ["vốn điều lệ", "hóa đơn điện tử"]


def test_cli_stage_5_6_end_to_end(tmp_path, monkeypatch, chunk_graph, chunks_df):
    output_path = tmp_path / "kg.gpickle"
    with output_path.open("wb") as f:
        pickle.dump(chunk_graph, f)

    stage3_path = tmp_path / "stage3.parquet"
    chunks_df.to_parquet(stage3_path)
    concepts_path = tmp_path / "concepts.yaml"
    concepts_path.write_text(
        "LEGAL_CONCEPTS:\n  - vốn điều lệ\n  - hóa đơn điện tử\n  - thuế giá trị gia tăng\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("data.stage5_build_graph.CONCEPT_COUNT_MIN", 1)
    monkeypatch.setattr("data.stage5_build_graph.CONCEPT_COUNT_MAX", 10)

    assert main(
        [
            "--stage",
            "5.6",
            "--append",
            "--stage3-path",
            str(stage3_path),
            "--legal-concepts-path",
            str(concepts_path),
            "--output-path",
            str(output_path),
        ]
    ) == 0

    with output_path.open("rb") as f:
        G = pickle.load(f)
    concept_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Concept"]
    mentions_edges = [
        1 for _u, _v, data in G.edges(data=True) if data.get("relation") == MENTIONS_RELATION
    ]
    assert len(concept_nodes) == 3
    assert len(mentions_edges) == 3
