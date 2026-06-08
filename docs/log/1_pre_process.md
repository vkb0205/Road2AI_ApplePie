[`preprocess.py`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py) is a CLI wrapper around the SME legal-corpus filtering utilities in [`sme_filter.py`](Road2AI_ApplePie/vietnamese-legal-rag/src/vlr/preprocessing/sme_filter.py). Its job is to read local metadata/content/relationship tables, apply SME-scoping rules, validate key consistency, write deterministic outputs, and print the generated output paths as JSON.

Main flow:

1. Bootstraps imports
   - It computes [`PROJECT_ROOT`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:11) as the parent of [`scripts/`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py).
   - It adds [`src`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:12) to [`sys.path`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:13), so local package imports like [`vlr.preprocessing.sme_filter`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:18) work even when running the script directly.
   - It imports [`pandas`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:16) and the SME filtering functions/classes from [`sme_filter.py`](Road2AI_ApplePie/vietnamese-legal-rag/src/vlr/preprocessing/sme_filter.py).

2. Loads config with [`load_config()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:33)
   - Reads the config file as UTF-8 text.
   - If the suffix is `.yaml` or `.yml`, it uses [`yaml.safe_load()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:40).
   - Otherwise, it parses JSON with [`json.loads()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:42).
   - The raw mapping is converted into an [`SMEFilterConfig`](Road2AI_ApplePie/vietnamese-legal-rag/src/vlr/preprocessing/sme_filter.py:101) via [`SMEFilterConfig.from_mapping()`](Road2AI_ApplePie/vietnamese-legal-rag/src/vlr/preprocessing/sme_filter.py:116). This config contains include/conditional/exclude labels, title keywords, accepted document types/statuses, relationship policy, and output paths.

3. Reads input tables with [`read_table()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:46)
   - Supports deterministic local table formats:
     - `.parquet` via [`pd.read_parquet()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:51)
     - `.jsonl` via [`pd.read_json(..., orient="records", lines=True)`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:53)
     - `.json` via [`pd.read_json()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:55)
     - `.csv` via [`pd.read_csv()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:57)
   - Any other suffix raises [`ValueError`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:58).

4. Parses CLI arguments with [`parse_args()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:61)
   - Required input: metadata table.
   - Optional inputs: content table and relationships table.
   - Inputs can be supplied either positionally or by flags:
     - [`metadata_pos`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:71) or [`--metadata`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:74)
     - [`content_pos`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:72) or [`--content`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:75)
     - [`relationships_pos`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:73) or [`--relationships`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:76)
   - Default config is [`configs/smecorpus.yaml`](Road2AI_ApplePie/vietnamese-legal-rag/configs/smecorpus.yaml), wired at [`--config`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:77).
   - Default output root is [`PROJECT_ROOT`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:83).
   - If metadata is missing, [`parser.error()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:89) stops execution.

5. Executes preprocessing in [`main()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:93)
   - Parses args with [`parse_args()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:94).
   - Loads SME filter config with [`load_config()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:95).
   - Reads metadata with [`read_table()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:96).
   - Calls [`filter_metadata()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:97), which returns:
     - `normalized`: all metadata rows with normalized/audit columns.
     - `filtered`: only rows where SME filtering decided to keep the document.

6. Optional content join
   - If [`args.content`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:101) is provided, it reads the content table and calls [`join_sme_content()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:102).
   - [`join_sme_content()`](Road2AI_ApplePie/vietnamese-legal-rag/src/vlr/preprocessing/sme_filter.py:239) creates/normalizes a stable [`lawid`](Road2AI_ApplePie/vietnamese-legal-rag/src/vlr/preprocessing/sme_filter.py:246), extracts usable HTML/text into [`html_content`](Road2AI_ApplePie/vietnamese-legal-rag/src/vlr/preprocessing/sme_filter.py:248), removes rows missing usable content, and returns:
     - `joined`: SME metadata joined to usable content.
     - `content_removals`: audit rows for missing/empty content.

7. Optional relationship filtering
   - If [`args.relationships`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:105) is provided, it chooses the surviving scope:
     - joined content rows if content was provided and joined successfully.
     - otherwise filtered metadata rows.
   - Then it calls [`filter_relationships()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:107).
   - [`filter_relationships()`](Road2AI_ApplePie/vietnamese-legal-rag/src/vlr/preprocessing/sme_filter.py:269) normalizes source/target IDs to [`source_lawid`](Road2AI_ApplePie/vietnamese-legal-rag/src/vlr/preprocessing/sme_filter.py:279) and [`target_lawid`](Road2AI_ApplePie/vietnamese-legal-rag/src/vlr/preprocessing/sme_filter.py:280), then keeps only relationships whose source is in the SME scope and whose target is either in scope or allowed as an external reference by config.

8. Validates key consistency
   - [`verify_key_consistency()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:111) checks that downstream outputs do not contain out-of-scope [`lawid`](Road2AI_ApplePie/vietnamese-legal-rag/src/vlr/preprocessing/sme_filter.py:356) values.
   - For joined content, every [`lawid`](Road2AI_ApplePie/vietnamese-legal-rag/src/vlr/preprocessing/sme_filter.py:361) must exist in filtered metadata.
   - For relationships, all source/target law IDs must exist in filtered metadata according to the current implementation.

9. Writes outputs
   - [`write_filter_outputs()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:112) writes configured JSONL/JSON checkpoints under the configured [`project_root`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:116).
   - Depending on [`output_paths`](Road2AI_ApplePie/vietnamese-legal-rag/src/vlr/preprocessing/sme_filter.py:338), it can write:
     - filtered metadata
     - removal log
     - joined content
     - filtered relationships
     - validation report
   - It also appends content-removal audit rows to the removal log via [`extra_removals`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:119).

10. Prints output paths
   - Finally, it prints a sorted JSON object mapping output names to POSIX-style paths using [`path.as_posix()`](Road2AI_ApplePie/vietnamese-legal-rag/scripts/preprocess.py:121).

Conceptually, the script is not doing low-level filtering itself. It orchestrates this pipeline:

`config + metadata table -> normalize/classify metadata -> keep SME-relevant documents -> optionally attach usable content -> optionally scope legal relationships -> validate stable lawid keys -> write restartable outputs + report`

A typical invocation is:

`python scripts/preprocess.py --metadata data/metadata.jsonl --content data/content.jsonl --relationships data/relationships.jsonl`

from [`Road2AI_ApplePie/vietnamese-legal-rag`](Road2AI_ApplePie/vietnamese-legal-rag).