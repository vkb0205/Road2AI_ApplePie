# AGENTS.md

This file provides guidance to agents when working with code in this repository.

- Treat `Road2AI_ApplePie/vietnamese-legal-rag/` as the active project root; the workspace has an extra wrapper directory.
- The checked-in Python/config files are mostly empty scaffolding; only `docs/preprocessing.md` currently contains substantive project workflow.
- Dataset target is Hugging Face `th1nhng0/vietnamese-legal-documents`; expected tables are `metadata`, `content`, and `relationships`.
- Preprocessing should keep an SME legal scope, favor active documents, and prioritize laws/decrees/circulars/decisions.
- Stable join key convention is a `lawid`/document key shared across metadata, content, relationships, document nodes, and article rows.
- Main retrieval unit is one row per Vietnamese legal article (`Điều`); long articles should split by clause (`Khoản`) only after exceeding token limits.
- Preserve Vietnamese legal hierarchy markers (`Phần`, `Chương`, `Mục`, `Điều`) and add breadcrumb context to chunks.
- Outputs should be restartable checkpoints under `data/` or `artifacts/`, preferably deterministic parquet/jsonl with removal logs and validation reports.
- Planned retrieval architecture uses enriched chunk text, optional cached summaries, BM25 plus dense indexes, and sidecar metadata with row-to-index alignment checks.
- Planned graph architecture creates document/article nodes, `HAS_ARTICLE` edges, and cleaned legal relation edges such as amends/cites/replaces/details.
- On Windows, use forward slashes or raw strings for paths containing `vietnamese-legal-rag`; normal strings with `\v` become `\x0b`.
- No project-specific build/lint/test commands are defined yet; use direct module/script or `python -m pytest tests/test_parser.py` style commands from `Road2AI_ApplePie/vietnamese-legal-rag/` once implementations exist.
