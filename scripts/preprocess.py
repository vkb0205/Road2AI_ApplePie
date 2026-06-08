"""CLI entry point for SME metadata filtering preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from vlr.preprocessing.sme_filter import (
    SMEFilterConfig,
    filter_metadata,
    filter_relationships,
    join_sme_content,
    verify_key_consistency,
    write_filter_outputs,
)

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def load_config(path: Path) -> SMEFilterConfig:
    """Load SME filter configuration from YAML or JSON."""

    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read YAML configuration files")
        raw: dict[str, Any] = yaml.safe_load(raw_text) or {}
    else:
        raw = json.loads(raw_text)
    return SMEFilterConfig.from_mapping(raw)


def read_table(path: Path) -> pd.DataFrame:
    """Read a deterministic local table supported by the preprocessing CLI."""

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, orient="records", lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Vietnamese legal data to an SME-relevant scope.",
        epilog=(
            "Examples: "
            "python scripts/preprocess.py --metadata data/metadata.jsonl --content data/content.jsonl "
            "--relationships data/relationships.jsonl; or "
            "python scripts/preprocess.py data/metadata.jsonl data/content.jsonl data/relationships.jsonl"
        ),
    )
    parser.add_argument("metadata_pos", nargs="?", type=Path, help="Path to metadata parquet/jsonl/json/csv table.")
    parser.add_argument("content_pos", nargs="?", type=Path, help="Optional content parquet/jsonl/json/csv table.")
    parser.add_argument("relationships_pos", nargs="?", type=Path, help="Optional relationships parquet/jsonl/json/csv table.")
    parser.add_argument("--metadata", type=Path, help="Path to metadata parquet/jsonl/json/csv table.")
    parser.add_argument("--content", type=Path, help="Optional content parquet/jsonl/json/csv table.")
    parser.add_argument("--relationships", type=Path, help="Optional relationships parquet/jsonl/json/csv table.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "smecorpus.yaml",
        help="SME filter config path.",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root for configured outputs.")
    args = parser.parse_args()
    args.metadata = args.metadata or args.metadata_pos
    args.content = args.content or args.content_pos
    args.relationships = args.relationships or args.relationships_pos
    if args.metadata is None:
        parser.error("metadata is required; pass --metadata data/metadata.jsonl or use the first positional argument")
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metadata = read_table(args.metadata)
    normalized, filtered = filter_metadata(metadata, config)

    joined = None
    content_removals = None
    if args.content:
        joined, content_removals = join_sme_content(filtered, read_table(args.content))

    scoped_relationships = None
    if args.relationships:
        scope_frame = joined if joined is not None else filtered
        scoped_relationships, _relationship_removals = filter_relationships(
            read_table(args.relationships), scope_frame["lawid"], config
        )

    verify_key_consistency(filtered, joined, scoped_relationships)
    paths = write_filter_outputs(
        normalized_metadata=normalized,
        filtered_metadata=filtered,
        config=config,
        project_root=args.project_root,
        joined_content=joined,
        filtered_relationships=scoped_relationships,
        extra_removals=content_removals,
    )
    print(json.dumps({name: str(path.as_posix()) for name, path in sorted(paths.items())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
