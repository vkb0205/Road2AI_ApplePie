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
- Implement Stage 4 summarization in `src/data/stage4_summarize.py`:
  - Use `google/gemma-3-12b-it` with 4-bit NF4 quantization for summarization
  - Load tokenizer and model with explicit HF token auth for gated repo access
  - Configure `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)`
  - Use `attn_implementation="eager"` for compatibility with quantized models
  - Generate two-field JSON summaries: `short` (≤30 words) and `key` (3-5 bullet points)
  - Apply chat template with system and user prompts in Vietnamese
  - Extract JSON from model response using regex patterns with fallback logic
  - Build `enriched_text` format: `[TÓM TẮT]` + `[Ý CHÍNH]` + `[NỘI DUNG ĐẦY ĐỦ]`
  - Implement resumable JSONL cache (`summary_cache.jsonl`) - append after each chunk
  - Cache loading skips already-processed chunks on restart
  - Store `key` column as JSON string for Parquet compatibility
  - Generate statistics on non-empty summaries and key points
  - Parameters: `max_input_chars=3000`, `max_new_tokens=256`, `temperature=0.1`, `repetition_penalty=1.05`, `batch_size=2`

- Add Stage 4 configuration to `config/default.yaml`:
  - Model: `google/gemma-3-12b-it`
  - Quantization: 4-bit NF4 with float16 compute
  - Generation parameters: temperature 0.1, repetition penalty 1.05
  - Batch size: 2 (conservative for 16GB VRAM)

- Implement `notebooks/kaggle_02_summarization.ipynb` for Kaggle GPU T4 [kaggle.com/code/boinhbo/kaggle-02-summarization]:
  - Complete notebook with 10 sections: setup, config, prompts, helpers, data loading, model loading, generation, enriched parquet building, output saving, upload instructions
  - Install dependencies: `transformers==4.46.0`, `torch==2.4.1`, `accelerate==1.0.1`, `bitsandbytes==0.44.1`
  - Load HF token from Kaggle Secrets or environment variables
  - Mount `stage3_chunks.parquet` from Kaggle Dataset
  - Display GPU memory usage and estimated runtime (~17 hours for full corpus)
  - Progress bar with tqdm for chunk processing
  - Error handling with fallback to empty summaries
  - Resume capability from existing cache
  - Output validation and sample enriched chunk display
  - Upload instructions for creating `glrag-stage4` Kaggle Dataset

**Status**: Stage 4 implementation complete. Ready for execution on Kaggle GPU T4.
**Next steps**: Upload `stage3_chunks.parquet` to Kaggle Dataset, run Notebook 02 to generate summaries, then upload outputs for Stage 5.
