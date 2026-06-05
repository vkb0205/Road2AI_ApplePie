"""SME legal corpus filtering utilities.

This module implements deterministic metadata filtering for the Vietnamese legal
RAG preprocessing pipeline. It keeps one stable ``lawid`` across filtered
metadata, joined content, and scoped relationships, while writing restartable
JSONL checkpoints and validation reports.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Optional at import time for lightweight unit tests.
    import pandas as pd
except Exception:  # pragma: no cover - exercised only when pandas unavailable.
    pd = None  # type: ignore[assignment]


LABEL_SPLIT_RE = re.compile(r"\s*(?:[,;|/]|\n|\r|、|，|;|；)\s*")
SPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^\w\s-]", flags=re.UNICODE)

DEFAULT_KEY_COLUMNS = (
    "lawid",
    "id",
    "doc_id",
    "document_id",
    "van_ban_id",
    "so_hieu",
    "title",
    "ten_van_ban",
)
TITLE_COLUMNS = ("title", "ten_van_ban", "name", "trich_yeu")
LABEL_COLUMNS = ("linh_vuc", "sector", "field", "fields")
TYPE_COLUMNS = ("doc_type", "document_type", "loai_van_ban", "loai")
STATUS_COLUMNS = ("validity_status", "status", "tinh_trang_hieu_luc", "hieu_luc")
CONTENT_KEY_COLUMNS = DEFAULT_KEY_COLUMNS
HTML_COLUMNS = ("html", "content", "noi_dung", "text", "body")
REL_SOURCE_COLUMNS = ("source_lawid", "source_id", "from_lawid", "from_id", "src", "source")
REL_TARGET_COLUMNS = ("target_lawid", "target_id", "to_lawid", "to_id", "dst", "target")


def normalize_text(value: Any) -> str:
    """Return comparable Unicode text while preserving Vietnamese accents."""

    if value is None:
        return ""
    if isinstance(value, float) and pd is not None and pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value)).strip().lower()
    text = NON_WORD_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def parse_labels(value: Any) -> list[str]:
    """Parse a potentially multi-label ``linh_vuc`` cell deterministically."""

    if value is None:
        return []
    if isinstance(value, float) and pd is not None and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_parts: Iterable[Any] = value
    else:
        raw_parts = LABEL_SPLIT_RE.split(str(value))
    labels = {normalize_text(part) for part in raw_parts if normalize_text(part)}
    return sorted(labels)


def first_present(row: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    """Return the first non-empty row value for candidate column names."""

    for column in candidates:
        if column in row:
            value = row[column]
            if value is not None and str(value).strip() and str(value).lower() != "nan":
                return value
    return ""


def stable_lawid(row: Mapping[str, Any], key_columns: Sequence[str] = DEFAULT_KEY_COLUMNS) -> str:
    """Create a stable document key from source identifiers or metadata."""

    for column in key_columns:
        if column in row:
            normalized = normalize_text(row[column])
            if normalized:
                prefix = normalize_text(column).replace(" ", "_")
                return f"{prefix}:{normalized}"
    fingerprint = json.dumps({k: str(v) for k, v in sorted(row.items())}, ensure_ascii=False)
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]
    return f"generated:{digest}"


@dataclass(frozen=True)
class SMEFilterConfig:
    """Configuration for deterministic SME metadata filtering."""

    include_labels: frozenset[str]
    conditional_labels: frozenset[str]
    exclude_labels: frozenset[str]
    title_keywords: tuple[str, ...]
    accepted_document_types: frozenset[str]
    accepted_validity_statuses: frozenset[str]
    allow_unknown_status: bool = True
    allow_external_references: bool = False
    output_paths: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SMEFilterConfig":
        """Build config from a mapping matching ``configs/smecorpus.yaml``."""

        root = raw.get("sme_filtering", raw)
        labels = root.get("labels", {})
        doc_types = root.get("document_types", {})
        statuses = root.get("validity_status", {})
        relationships = root.get("relationships", {})
        return cls(
            include_labels=frozenset(parse_labels(labels.get("include", []))),
            conditional_labels=frozenset(parse_labels(labels.get("conditional", []))),
            exclude_labels=frozenset(parse_labels(labels.get("exclude", []))),
            title_keywords=tuple(parse_labels(root.get("title_keywords", []))),
            accepted_document_types=frozenset(parse_labels(doc_types.get("accepted", []))),
            accepted_validity_statuses=frozenset(parse_labels(statuses.get("accepted", []))),
            allow_unknown_status=bool(statuses.get("allow_unknown", True)),
            allow_external_references=bool(relationships.get("allow_external_references", False)),
            output_paths=dict(root.get("output_paths", {})),
        )


def has_title_keyword(normalized_title: str, keywords: Sequence[str]) -> bool:
    """Return whether a normalized title contains any configured SME keyword."""

    return any(keyword and keyword in normalized_title for keyword in keywords)


def classify_row(row: Mapping[str, Any], config: SMEFilterConfig) -> dict[str, Any]:
    """Normalize and classify a metadata row with an auditable reason code."""

    original_labels = first_present(row, LABEL_COLUMNS)
    labels = parse_labels(original_labels)
    title = first_present(row, TITLE_COLUMNS)
    doc_type = normalize_text(first_present(row, TYPE_COLUMNS))
    status = normalize_text(first_present(row, STATUS_COLUMNS))
    title_norm = normalize_text(title)
    lawid = stable_lawid(row)

    type_ok = not config.accepted_document_types or doc_type in config.accepted_document_types
    status_unknown = not status
    status_ok = (status in config.accepted_validity_statuses) or (
        status_unknown and config.allow_unknown_status
    )
    keyword_hit = has_title_keyword(title_norm, config.title_keywords)

    include_hit = sorted(set(labels) & config.include_labels)
    conditional_hit = sorted(set(labels) & config.conditional_labels)
    exclude_hit = sorted(set(labels) & config.exclude_labels)
    unmatched = sorted(set(labels) - config.include_labels - config.conditional_labels - config.exclude_labels)

    if not type_ok:
        keep = False
        reason = "remove_unaccepted_type"
    elif not status_ok:
        keep = False
        reason = "remove_unaccepted_status"
    elif include_hit:
        keep = True
        reason = "keep_include_label"
    elif conditional_hit and keyword_hit:
        keep = True
        reason = "keep_conditional_keyword"
    elif conditional_hit:
        keep = False
        reason = "remove_conditional_without_keyword"
    elif exclude_hit and not keyword_hit:
        keep = False
        reason = "remove_excluded_label"
    elif unmatched and keyword_hit:
        keep = True
        reason = "keep_unmatched_keyword_rescue"
    else:
        keep = False
        reason = "remove_unmatched_label"

    return {
        **dict(row),
        "lawid": lawid,
        "linh_vuc_original": original_labels,
        "linh_vuc_normalized": labels,
        "title_normalized": title_norm,
        "document_type_normalized": doc_type,
        "validity_status_normalized": status,
        "sme_title_keyword_hit": keyword_hit,
        "sme_include_labels": include_hit,
        "sme_conditional_labels": conditional_hit,
        "sme_exclude_labels": exclude_hit,
        "sme_unmatched_labels": unmatched,
        "sme_keep": keep,
        "sme_reason": reason,
    }


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for dataframe filtering")
    return pd


def normalize_metadata(metadata: Any, config: SMEFilterConfig) -> Any:
    """Return metadata with normalized SME filtering columns and decisions."""

    pandas = _require_pandas()
    records = [classify_row(row, config) for row in metadata.to_dict(orient="records")]
    frame = pandas.DataFrame.from_records(records)
    if not frame["lawid"].is_unique:
        duplicates = sorted(frame.loc[frame["lawid"].duplicated(), "lawid"].unique())
        raise ValueError(f"duplicate lawid values after normalization: {duplicates[:5]}")
    return frame.sort_values("lawid").reset_index(drop=True)


def filter_metadata(metadata: Any, config: SMEFilterConfig) -> tuple[Any, Any]:
    """Return normalized metadata and surviving SME metadata."""

    normalized = normalize_metadata(metadata, config)
    filtered = normalized.loc[normalized["sme_keep"]].copy().reset_index(drop=True)
    return normalized, filtered


def _row_key(row: Mapping[str, Any]) -> str:
    return stable_lawid(row, CONTENT_KEY_COLUMNS)


def join_sme_content(filtered_metadata: Any, content: Any) -> tuple[Any, Any]:
    """Join surviving SME metadata to content and audit missing/empty HTML rows."""

    pandas = _require_pandas()
    content_records = []
    for row in content.to_dict(orient="records"):
        record = dict(row)
        record["lawid"] = _row_key(record)
        html_value = first_present(record, HTML_COLUMNS)
        record["html_content"] = html_value
        record["has_usable_html"] = bool(str(html_value).strip())
        content_records.append(record)
    content_frame = pandas.DataFrame.from_records(content_records)
    joined = filtered_metadata.merge(content_frame, on="lawid", how="left", suffixes=("", "_content"))
    missing = joined.loc[~joined["has_usable_html"].fillna(False)].copy()
    if not missing.empty:
        missing["sme_reason"] = "remove_missing_usable_html"
    usable = joined.loc[joined["has_usable_html"].fillna(False)].copy().reset_index(drop=True)
    removal_columns = [
        "lawid",
        "linh_vuc_original",
        "linh_vuc_normalized",
        "document_type_normalized",
        "validity_status_normalized",
        "sme_reason",
    ]
    available = [column for column in removal_columns if column in missing.columns]
    return usable.sort_values("lawid").reset_index(drop=True), missing[available].reset_index(drop=True)


def filter_relationships(relationships: Any, surviving_lawids: Iterable[str], config: SMEFilterConfig) -> tuple[Any, Any]:
    """Scope relationship edges to surviving SME documents."""

    pandas = _require_pandas()
    surviving = set(surviving_lawids)
    records = []
    for row in relationships.to_dict(orient="records"):
        record = dict(row)
        source = stable_lawid({"id": first_present(record, REL_SOURCE_COLUMNS)})
        target = stable_lawid({"id": first_present(record, REL_TARGET_COLUMNS)})
        record["source_lawid"] = source
        record["target_lawid"] = target
        source_ok = source in surviving
        target_ok = target in surviving or config.allow_external_references
        record["sme_keep"] = source_ok and target_ok
        record["sme_reason"] = "keep_scoped_relationship" if record["sme_keep"] else "remove_out_of_scope_relationship"
        records.append(record)
    frame = pandas.DataFrame.from_records(records)
    kept = frame.loc[frame["sme_keep"]].copy().reset_index(drop=True)
    removed = frame.loc[~frame["sme_keep"]].copy().reset_index(drop=True)
    return kept.sort_values(["source_lawid", "target_lawid"]).reset_index(drop=True), removed


def validation_report(normalized_metadata: Any, joined_content: Any | None = None, relationships: Any | None = None) -> dict[str, Any]:
    """Build deterministic validation counts for SME filtering outputs."""

    report: dict[str, Any] = {
        "metadata_rows": int(len(normalized_metadata)),
        "kept_metadata_rows": int(normalized_metadata["sme_keep"].sum()),
        "removed_metadata_rows": int((~normalized_metadata["sme_keep"]).sum()),
        "by_reason": normalized_metadata["sme_reason"].value_counts().sort_index().to_dict(),
        "by_document_type": normalized_metadata["document_type_normalized"].value_counts().sort_index().to_dict(),
        "by_validity_status": normalized_metadata["validity_status_normalized"].value_counts().sort_index().to_dict(),
        "by_keep_decision": normalized_metadata["sme_keep"].value_counts().sort_index().to_dict(),
    }
    label_counts: dict[str, int] = {}
    for labels in normalized_metadata["linh_vuc_normalized"]:
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
    report["by_label"] = dict(sorted(label_counts.items()))
    if joined_content is not None:
        report["joined_content_rows"] = int(len(joined_content))
    if relationships is not None:
        report["filtered_relationship_rows"] = int(len(relationships))
    return report


def _write_jsonl(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(path, orient="records", lines=True, force_ascii=False)


def _write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_filter_outputs(
    normalized_metadata: Any,
    filtered_metadata: Any,
    config: SMEFilterConfig,
    project_root: str | Path,
    joined_content: Any | None = None,
    filtered_relationships: Any | None = None,
    extra_removals: Any | None = None,
) -> dict[str, Path]:
    """Write deterministic JSONL checkpoints and validation report."""

    root = Path(project_root)
    paths = {name: root / relative for name, relative in config.output_paths.items()}
    if "filtered_metadata" in paths:
        _write_jsonl(filtered_metadata.sort_values("lawid"), paths["filtered_metadata"])
    removals = normalized_metadata.loc[~normalized_metadata["sme_keep"]].copy()
    if extra_removals is not None and len(extra_removals):
        removals = pd.concat([removals, extra_removals], ignore_index=True, sort=False)  # type: ignore[union-attr]
    if "removal_log" in paths:
        _write_jsonl(removals.sort_values("lawid"), paths["removal_log"])
    if joined_content is not None and "joined_content" in paths:
        _write_jsonl(joined_content.sort_values("lawid"), paths["joined_content"])
    if filtered_relationships is not None and "filtered_relationships" in paths:
        _write_jsonl(filtered_relationships.sort_values(["source_lawid", "target_lawid"]), paths["filtered_relationships"])
    if "validation_report" in paths:
        report = validation_report(normalized_metadata, joined_content, filtered_relationships)
        _write_json(report, paths["validation_report"])
    return paths


def verify_key_consistency(filtered_metadata: Any, joined_content: Any | None = None, filtered_relationships: Any | None = None) -> None:
    """Validate that all downstream rows use the surviving metadata ``lawid`` scope."""

    surviving = set(filtered_metadata["lawid"])
    if joined_content is not None:
        missing = set(joined_content["lawid"]) - surviving
        if missing:
            raise ValueError(f"joined content contains out-of-scope lawid values: {sorted(missing)[:5]}")
    if filtered_relationships is not None and len(filtered_relationships):
        rel_ids = set(filtered_relationships["source_lawid"]) | set(filtered_relationships["target_lawid"])
        missing = rel_ids - surviving
        if missing:
            raise ValueError(f"relationships contain out-of-scope lawid values: {sorted(missing)[:5]}")
