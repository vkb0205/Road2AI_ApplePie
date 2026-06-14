# G-LRAG Project Progress

## Preprocessing Pipeline

### Checkpoint 09/06/2026 - VKB
- Create `src/data/stage1_filter.py`
- Populate `config/default.yaml` with filtering constants (`SME_TITLE_KEYWORDS`, `SME_LINH_VUC`, `SME_NGANH`, `VALID_DOCUMENT_TYPES`)
- Implement filtering logic:
  - Match SME-related keywords, `linh_vuc`, or `nganh`
  - Restrict to `tinh_trang_hieu_luc` containing "Còn hiệu lực"
  - Restrict to allowed document types
- Exclude documents missing from the `content` dataset
- Parameterize inputs to allow downloading from Hugging Face or reading local files
- Support processing both `.parquet` and `.jsonl` input files natively
- Drop duplicate `(law_id, ten_van_ban)` pairs and enforce data quality gates
- Export to `data/stage1_sme_docs.parquet`

### Checkpoint 10/06/2026 - KL
- Implement Stage 2 HTML parsing in `src/data/stage2_parse_html.py`:
  - Clean raw HTML with `BeautifulSoup(..., "html.parser")`, remove `script/style`, normalize whitespace, collapse repeated newlines, and deduplicate consecutive identical lines.
  - Detect `Điều` boundaries with a flexible regex supporting optional suffixes (`a`, `b`, `đ`) and separators `.` `:` `-` `–` `—` `)`.
  - Assign document hierarchy context for `Phần`, `Chương`, and `Mục` based on the most recent header before each article.
  - Drop article rows whose `noi_dung` is shorter than 30 characters (minimum content threshold).
  - Preserve non-standard documents as a single fallback article when no `Điều` markers are found, unless the cleaned text is too short.
  - Log parse failures to `data/stage2_parse_failures.jsonl` using failure reasons: `text_too_short_after_cleaning`, `zero_dieu_law_like_doc`, `zero_parsable_dieu_law_like_doc`, `single_dieu_law_like_doc`.
  - Deduplicate `stage2_articles.parquet` by `(doc_id, dieu_so)` while keeping the longest article text.

- **Final keep/drop policy** (Decided 2026-06-10):
  - **Keep in output**: fallback placeholder articles (`dieu_so = "Điều VB"`) when no `Điều` markers found (flagged as `zero_dieu_law_like_doc`); single-article law-like documents (flagged as `single_dieu_law_like_doc`).
  - **Drop from output** (logged in failures only): `text_too_short_after_cleaning` (cleaned text < 30 chars) and `zero_parsable_dieu_law_like_doc` (law-like title but all parsed articles dropped due to short content).

- **Stage 2 run results** (executed 2026-06-10):
  - Command: `python src/data/stage2_parse_html.py` (auto-discovered local `content` file)
  - Output: `data/stage2_articles.parquet` — **56,269 article rows** from **12,633 unique documents**
  - Logs: `data/stage2_parse_failures.jsonl` — 2,406 failure records; `data/stage2_parse_failures_summary.json` — aggregated counts
  - Deduplication: removed 2,417 duplicate article rows by `(doc_id, dieu_so)`, kept longest `noi_dung`
  - Failure breakdown by reason:
    - `text_too_short_after_cleaning`: **2,027** (cleaned text < 30 chars → dropped)
    - `zero_dieu_law_like_doc`: **275** (no Điều, law-like title → kept as `Điều VB`)
    - `single_dieu_law_like_doc`: **70** (law-like with 1 article → kept, flagged for review)
    - `zero_parsable_dieu_law_like_doc`: **34** (law-like, had Điều but all dropped for short content → dropped)
  - **Interpretation**: Majority failures are empty/short documents. Fallback `Điều VB` preserved ~275 regulatory documents despite lacking formal article structure. For triage, inspect failure samples via `src/data/debug_stage2_failures.py`.

### Checkpoint 10/06/2026 - KL
- Implement Stage 3 chunking in `src/data/stage3_chunking.py`:
  - Input: `data/stage2_articles.parquet`, optionally merged with `data/stage2_manual_fixes.json`/`.jsonl`.
  - Output: `data/stage3_chunks.parquet`.
  - Parameters: `MAX_TOKENS = 1024`, tokenizer `google/gemma-3-12b-it`.
  - Support gated repo access via `HUGGINGFACE_HUB_TOKEN`, `HF_TOKEN`, or `--hf-token`.
  - Load the tokenizer with explicit auth: `AutoTokenizer.from_pretrained(..., token=hf_token, trust_remote_code=True)`.
  - Preserve Stage 2 metadata and build `breadcrumb` from `loai_van_ban`, `ten_van_ban`, `phan`, `chuong`, `muc`, `dieu_so`, `dieu_ten`.
  - Split `noi_dung` into Khoản-level pieces using `re.split(r"(?=^\s*\d+\.\s)", flags=re.M)`.
  - Greedily pack Khoản pieces into chunks while keeping cumulative token count ≤ `MAX_TOKENS`.
  - At chunk boundary, prepend the last Khoản of the previous chunk into the next chunk when it fits.
  - If a single Khoản exceeds `MAX_TOKENS`, split it by sentence boundaries and then by token blocks.
  - Emit chunk records with `chunk_id`, `part_idx`, `breadcrumb`, and `chunk_text`.
  - `chunk_text` begins with breadcrumb followed by the joined chunk body.
  - Runtime validation: produced `74,107` chunks from `56,269` Stage 2 rows, covering `12,633` unique documents.
  - Warnings observed during this run are non-blocking: PyTorch disabled due to version `2.2.1`, Windows symlink cache warning, and BPE tokenizer cleanup warning.

### Checkpoint 11/06/2026 - DB
- Stage 4 summary injection was evaluated and has been deprioritized as a core pipeline step.
- The main KG and index build now use Stage 2 article metadata and Stage 3 chunk text.
- Keep `src/data/stage4_summarize.py` and `notebooks/kaggle_02_summarization.ipynb` as experimental artifacts, but do not rely on `stage4_enriched.parquet` for the production retrieval pipeline.
- Concept extraction should rely on a curated legal vocabulary and rule-based matching, not on LLM-generated summary text.

**Status**: Stage 4 summary injection removed from the main pipeline.

---

## Knowledge Graph & Concept Extraction — Chunk-First Approach

### Checkpoint 14/06/2026 - System Design Update

**Scope**: Updated all markdown specifications (KG.md, PLAN.md, G-LRAG_SPECIFICATIONS.md, PROGRESS.md) to implement **chunk-centric concept extraction** while preserving DOC and ART as standard layers.

**Key Changes**:

1. **Concept extraction source**: Moved from article-level to **chunk-level extraction**.
   - **Primary source**: `chunk_text` from `stage3_chunks.parquet` (mandatory).
   - **Secondary source**: Optional `enriched_text` from Stage 4 if available, but chunk text remains primary.
   - **Article aggregation**: Article text used only for reverse aggregation / sanity checking, not as primary extraction source.

2. **New edge in KG core**: Added `ART -> CHUNK` edge type.
   - **Previous core edges**: `DOC -> DOC`, `DOC -> ART`, `ART -> CONCEPT`.
   - **New core edges**: `DOC -> DOC`, `DOC -> ART`, `ART -> CHUNK`, `CHUNK -> CONCEPT`.

3. **CHUNK as intermediate node**:
   - Chunk nodes (`CHUNK:{chunk_id}`) now included in KG as intermediate layer.
   - Chunks are retrieval units (indexed by BM25/FAISS).
   - Chunks carry tightly-scoped evidence for concept matching.
   - Metadata on chunk nodes: `chunk_id`, `doc_uid`, `doc_id`, `rowidx`, `part_idx`, `breadcrumb`.

4. **Concept node quality gates**:
   - **No orphan chunks**: Every `CHUNK` must have parent `ART` via `HAS_CHUNK` edge.
   - **No `ART -> CONCEPT` in chunk-first pipeline**: All concept mentions now flow through chunks.
   - **`CHUNK -> CONCEPT` as primary**: Rule-based matching on chunk text (controlled vocabulary, no LLM).

5. **Motivation for chunk-first**:
   - With 74,107 chunks vs. 56,269 articles, chunk-level extraction provides:
     - More granular evidence scoping (chunk is smaller, tighter unit).
     - Better alignment with retrieval (chunks are what BM25/FAISS indexes).
     - Reduced false-positive concept assignments (smaller text span = lower ambiguity).

6. **Files updated**:
   - `KG.md`: Sections 2 (design principles), 3.3 (CHUNK node spec), 4 (concept design), 5.2 (edge types), 7.3–7.5 (relation sources), 10 (chunk role), 11 (quality gates).
   - `PLAN.md`: Stage 5 tasks 5.6 and 5.8 to reflect chunk-first extraction.
   - `G-LRAG_SPECIFICATIONS.md`: Section 8.2 (node types with CHUNK), Section 8.5 and 8.8 (concept extraction procedure and stats).
   - `PROGRESS.md`: This checkpoint documenting the system-wide update.

**Next steps**: 
- Stage 5 (graph build) implementation must enforce chunk-first concept extraction.
- Update Stage 6 (indexing) if needed to reflect chunk as retrieval unit.
- Verify quality gates during graph build to ensure no orphan chunks and no `ART -> CONCEPT` edges.