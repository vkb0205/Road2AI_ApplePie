from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vlr.preprocessing.sme_filter import (
    SMEFilterConfig,
    classify_row,
    filter_metadata,
    filter_relationships,
    has_title_keyword,
    join_sme_content,
    normalize_text,
    parse_labels,
    validation_report,
    verify_key_consistency,
    write_filter_outputs,
)


def sample_config(tmp_path: Path | None = None) -> SMEFilterConfig:
    output_paths = {}
    if tmp_path is not None:
        output_paths = {
            "filtered_metadata": "data/interim/sme_filtered_metadata.jsonl",
            "joined_content": "data/interim/sme_joined_content.jsonl",
            "filtered_relationships": "data/interim/sme_filtered_relationships.jsonl",
            "removal_log": "artifacts/reports/sme_filter_removals.jsonl",
            "validation_report": "artifacts/reports/sme_filter_validation.json",
        }
    return SMEFilterConfig(
        include_labels=frozenset({"doanh nghiệp", "thuế"}),
        conditional_labels=frozenset({"hành chính", "dân sự"}),
        exclude_labels=frozenset({"quốc phòng"}),
        title_keywords=("doanh nghiệp", "hợp đồng", "xử phạt"),
        accepted_document_types=frozenset({"luật", "nghị định", "thông tư", "quyết định"}),
        accepted_validity_statuses=frozenset({"còn hiệu lực", "chưa có hiệu lực"}),
        allow_unknown_status=True,
        output_paths=output_paths,
    )


def test_normalize_parse_labels_and_keyword_matching() -> None:
    assert normalize_text("  Thuế, DOANH nghiệp!!! ") == "thuế doanh nghiệp"
    assert parse_labels(" Thuế ; Doanh nghiệp / Hành chính ") == ["doanh nghiệp", "hành chính", "thuế"]
    assert parse_labels(["Thuế", "thuế", " Dân sự "]) == ["dân sự", "thuế"]
    assert has_title_keyword(normalize_text("Nghị định xử phạt doanh nghiệp"), ("xử phạt",))


def test_classify_include_conditional_exclude_type_and_status() -> None:
    config = sample_config()
    include = classify_row(
        {"id": "1", "title": "Luật doanh nghiệp", "linh_vuc": "Doanh nghiệp", "loai_van_ban": "Luật", "status": "Còn hiệu lực"},
        config,
    )
    assert include["sme_keep"] is True
    assert include["sme_reason"] == "keep_include_label"

    rescued = classify_row(
        {"id": "2", "title": "Nghị định xử phạt vi phạm của doanh nghiệp", "linh_vuc": "Hành chính", "loai_van_ban": "Nghị định", "status": "Còn hiệu lực"},
        config,
    )
    assert rescued["sme_keep"] is True
    assert rescued["sme_reason"] == "keep_conditional_keyword"

    conditional_removed = classify_row(
        {"id": "3", "title": "Quy định chung", "linh_vuc": "Dân sự", "loai_van_ban": "Luật", "status": "Còn hiệu lực"},
        config,
    )
    assert conditional_removed["sme_keep"] is False
    assert conditional_removed["sme_reason"] == "remove_conditional_without_keyword"

    excluded = classify_row(
        {"id": "4", "title": "Quy định quốc phòng", "linh_vuc": "Quốc phòng", "loai_van_ban": "Luật", "status": "Còn hiệu lực"},
        config,
    )
    assert excluded["sme_keep"] is False
    assert excluded["sme_reason"] == "remove_excluded_label"

    bad_type = classify_row(
        {"id": "5", "title": "Luật doanh nghiệp", "linh_vuc": "Doanh nghiệp", "loai_van_ban": "Công văn", "status": "Còn hiệu lực"},
        config,
    )
    assert bad_type["sme_reason"] == "remove_unaccepted_type"

    bad_status = classify_row(
        {"id": "6", "title": "Luật doanh nghiệp", "linh_vuc": "Doanh nghiệp", "loai_van_ban": "Luật", "status": "Hết hiệu lực"},
        config,
    )
    assert bad_status["sme_reason"] == "remove_unaccepted_status"


def test_checkpoints_reports_content_and_relationship_scope(tmp_path: Path) -> None:
    config = sample_config(tmp_path)
    metadata = pd.DataFrame(
        [
            {"id": "1", "title": "Luật doanh nghiệp", "linh_vuc": "Doanh nghiệp", "loai_van_ban": "Luật", "status": "Còn hiệu lực"},
            {"id": "2", "title": "Nghị định xử phạt doanh nghiệp", "linh_vuc": "Hành chính", "loai_van_ban": "Nghị định", "status": "Còn hiệu lực"},
            {"id": "3", "title": "Quy định quốc phòng", "linh_vuc": "Quốc phòng", "loai_van_ban": "Luật", "status": "Còn hiệu lực"},
        ]
    )
    normalized, filtered = filter_metadata(metadata, config)
    assert list(filtered["lawid"]) == ["id:1", "id:2"]

    content = pd.DataFrame(
        [
            {"id": "1", "html": "<p>usable</p>"},
            {"id": "2", "html": ""},
            {"id": "3", "html": "<p>out</p>"},
        ]
    )
    joined, content_removals = join_sme_content(filtered, content)
    assert list(joined["lawid"]) == ["id:1"]
    assert list(content_removals["sme_reason"]) == ["remove_missing_usable_html"]

    relationships = pd.DataFrame(
        [
            {"source_id": "1", "target_id": "1", "relation": "cites"},
            {"source_id": "1", "target_id": "3", "relation": "cites"},
        ]
    )
    scoped, removed_edges = filter_relationships(relationships, joined["lawid"], config)
    assert len(scoped) == 1
    assert len(removed_edges) == 1
    verify_key_consistency(filtered, joined, scoped)

    paths = write_filter_outputs(
        normalized_metadata=normalized,
        filtered_metadata=filtered,
        config=config,
        project_root=tmp_path,
        joined_content=joined,
        filtered_relationships=scoped,
        extra_removals=content_removals,
    )
    for path in paths.values():
        assert path.exists()

    filtered_rows = (tmp_path / "data/interim/sme_filtered_metadata.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(filtered_rows) == 2
    report = json.loads((tmp_path / "artifacts/reports/sme_filter_validation.json").read_text(encoding="utf-8"))
    assert report["metadata_rows"] == 3
    assert report["kept_metadata_rows"] == 2
    assert report["joined_content_rows"] == 1
    assert report["by_reason"]["keep_include_label"] == 1


def test_validation_report_contains_required_counts() -> None:
    config = sample_config()
    metadata = pd.DataFrame(
        [
            {"id": "1", "title": "Luật doanh nghiệp", "linh_vuc": "Doanh nghiệp", "loai_van_ban": "Luật", "status": "Còn hiệu lực"},
            {"id": "2", "title": "Quy định chung", "linh_vuc": "Dân sự", "loai_van_ban": "Luật", "status": "Còn hiệu lực"},
        ]
    )
    normalized, _filtered = filter_metadata(metadata, config)
    report = validation_report(normalized)
    assert report["by_keep_decision"][False] == 1
    assert report["by_label"]["doanh nghiệp"] == 1
    assert report["by_document_type"]["luật"] == 2
