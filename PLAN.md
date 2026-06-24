# G-LRAG Project Plan

**Project:** Graph-enhanced Legal Retrieval Augmented Generation (G-LRAG)  
**Competition deadline:** 30 June 2026, 23:59 ICT  
**DemoDay:** 11 July 2026  
**Plan version:** 1.0 — drafted 11 June 2026  

---

## Overview

This document maps every remaining task to a phase, assignee, environment, and
acceptance criterion.  Tasks already completed (Stages 1–3 confirmed in
PROGRESS.md) are marked ✅ and kept for traceability.  All open tasks are
marked 🔲, and tasks that are blocked pending an upstream artifact are marked
⏳.

The plan is organised into five phases that mirror the pipeline:

| Phase | Focus | Target complete |
|-------|-------|-----------------|
| 0 | Repo & environment hygiene | 11 Jun 2026 |
| 1 | Offline data pipeline (Stages 1–4) | 14 Jun 2026 |
| 2 | Offline graph & indexing (Stages 5–6) | 16 Jun 2026 |
| 3 | Retrieval & generation (online inference) | 22 Jun 2026 |
| 4 | Evaluation, tuning & submission | 29 Jun 2026 |

---

## Phase 0 — Repo & Environment Hygiene

> **Goal:** Everyone can run every stage from a clean checkout with one command.

| # | Task | Owner | Env | Acceptance |
|---|------|-------|-----|------------|
| 0.1 | 🔲 Finalise `requirements.txt` / `pyproject.toml` with pinned versions (torch 2.4.1, transformers 4.46.0, etc.) | All | Local | `pip install -e .` succeeds; `python -c "import torch, transformers, faiss, networkx"` passes |
| 0.2 | 🔲 Add `.env.example` documenting `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `ARTIFACTS_DIR` | All | Local | File committed; README updated |
| 0.3 | 🔲 Add `scripts/buildall.sh` that runs Stages 1–6 end-to-end | All | Local/Kaggle | Script exits 0 on a complete fresh run |
| 0.4 | 🔲 Set up `devset/` folder: draft 50 SME questions with gold `relevant_articles` for offline F2 scoring | All | Local | `devset/questions.json` and `devset/groundtruth.json` committed; `eval.py` reports a baseline F2 |
| 0.5 | 🔲 Confirm Kaggle Dataset upload workflow: Notebook 01 → upload artifacts → Notebook 02 mounts them | All | Kaggle | Notebook 02 reads `stage3_chunks.parquet` from mounted dataset without error |

---

## Phase 1 — Offline Data Pipeline

### Stage 1 — Scope Filter ✅

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 1.1 | Create `src/data/stage1_filter.py` with filtering logic (SME keywords, `linh_vuc`, `nganh`, validity, document type, content join) | VKB | ✅ Done | `stage1_sme_docs.parquet` produced; row count in [3 000, 8 000] |
| 1.2 | Populate `config/default.yaml` with `SME_TITLE_KEYWORDS`, `SME_LINHVUC`, `SME_NGANH`, `VALID_DOCUMENT_TYPES` | VKB | ✅ Done | All constants present in YAML; import test passes |
| 1.3 | Quality gates: assert no duplicate `id`, no duplicate `(lawid, tenvanban)`, every retained `id` has content | VKB | ✅ Done | Assertions pass on known corpus |

### Stage 2 — HTML Parsing ✅

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 2.1 | Implement HTML → article-level parser in `src/data/stage2_parse_html.py` (BeautifulSoup + plain-text regex) | KL | ✅ Done | Current 2026-06-21 rerun produced 524 070 article rows after Stage 1 expansion |
| 2.2 | Emit `stage2_parse_failures.jsonl` with typed reasons (`text_too_short_after_cleaning`, `zero_dieu_law_like_doc`, `zero_dieu_no_fallback`, etc.) | KL | ✅ Done | Failures logged for no-`Điều`, too-short, zero-parsable, and single-law-like review cases |
| 2.3 | Apply keep/drop policy: drop documents with no `Điều` markers instead of emitting fallback `Điều VB`; keep `single_dieu_law_like_doc` rows for review; drop short/zero-parsable classes | KL | ✅ Updated 2026-06-21 | Policy documented in PROGRESS.md |
| 2.4 | 🔲 Stage 2.5 manual review: inspect `single_dieu_law_like_doc` records (70) for the 5 key laws; produce `stage2_manual_fixes.json` if needed | KL | 🔲 Open | Manual review log committed; fixes file present (may be empty) |

### Stage 3 — Chunking ✅

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 3.1 | Implement chunking in `src/data/stage3_chunking.py` (Khoản-aware greedy packing, 1 024 token limit, 128-token overlap) | KL | ✅ Done | Historical run: 74 107 chunks from 56 269 articles; current 524k-row rerun pending with batch writer |
| 3.2 | Breadcrumb prefix injected into `chunk_text`; `chunk_id` format `{doc_uid}#{part_idx}` | KL | ✅ Done | Sample inspection passes |
| 3.3 | Add batch parquet writer for large Stage 2 artifacts (`--batch-size`, temp parquet, incremental `ParquetWriter`) | DB | ✅ Done 2026-06-21 | Unit test passes; avoids final all-chunks `DataFrame` OOM on 524k Stage 2 rows |
| 3.4 | 🔲 Upload `stage3_chunks.parquet` as Kaggle Dataset so Notebook 02 can mount it | All | 🔲 Open | Dataset visible in Kaggle; Notebook 02 mounts without error |

### Stage 4 — Summary Injection ⏳

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 4.1 | 🔲 Evaluate `src/data/stage4_summarize.py` and `notebooks/kaggle_02_summarization.ipynb` as experimental artifacts | All | ✅ Deferred | Kept for analysis, not required for core KG/index build |
| 4.2 | 🔲 Do not depend on `stage4_enriched.parquet` for production retrieval artifacts | All | ✅ Done | KG/index build uses Stage 2/3 artifacts instead |
| 4.3 | 🔲 Preserve optional support for Stage 4 metadata if available, but make it non-blocking | All | ✅ Done | `short`, `key`, `enriched_text` are optional metadata only |

> **Runtime note:** Summary injection is not a hard dependency for the pipeline. Prioritize Stage 2 and Stage 3 artifacts for graph/index construction.

---

## Phase 2 — Graph & Indexing

### Stage 5 — Knowledge Graph ✅

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 5.1 | 🔲 Implement `src/data/stage5_build_graph.py`; build `networkx.MultiDiGraph` with Document, Article, Chunk, Concept nodes | All | ⏳ Blocked on Stage 2/3 artifacts | Script completes without error |
| 5.2 | 🔲 Add Document nodes from `stage1_sme_docs.parquet` (attributes: `law_id`, `ten`, `loai`, `nganh`, `ngay_ban_hanh`) | VKB | ✅ Done 2026-06-16 | Node count in [3 000, 8 000] |
| 5.3 | ✅ Add Article nodes + `HAS_ARTICLE` edges from `stage2_articles.parquet`; inner-join on existing DOC nodes (orphan ART rows whose `doc_id` is not a DOC node are exported to `stage5_orphan_articles.jsonl` for team review, not added to the graph). ART node attrs: `type`, `doc_uid`, `doc_id`, `law_id`, `ten_van_ban`, `dieu_so`, `dieu_ten`, `phan`, `chuong`, `muc`, `loai_van_ban`, `ngay_ban_hanh`, `start_char`, `end_char` (empty strings normalized to `None`; `noi_dung` deliberately excluded). `HAS_ARTICLE` edge keeps `relation` only; article metadata lives on the ART node. CLI uses sub-command style: `--stage {5.2,5.3,5.4,5.5,5.6,all}`. | VKB | ✅ Done 2026-06-16 | **56 269** ART nodes / **56 269** HAS_ARTICLE edges (in [20 000, 100 000]); every ART has exactly one parent DOC; `stage5_orphan_articles.jsonl` written with **0** records; `kg.gpickle` = **48.50 MB** |
| 5.4 | ✅ Add Chunk nodes + `HAS_CHUNK` edges from `stage3_chunks.parquet`; keep `chunk_id`, `doc_uid`, `doc_id`, `rowidx`, `part_idx`, `breadcrumb` metadata (full `chunk_text` deliberately excluded to keep `kg.gpickle` lean). `rowidx` is the original 0-based Stage 3 row position. Inner-join on existing ART nodes; chunks whose parent ART is missing (because the ART was an orphan in 5.3) are skipped and logged to `stage5_orphan_chunks.jsonl` to preserve the `DOC -> ART -> CHUNK` invariant. `HAS_CHUNK` edge keyed as `HAS_CHUNK` and keeps `relation` only; chunk metadata lives on the CHUNK node. CLI adds `--stage3-path` / `--orphan-chunks-path`; `--stage 5.4 --append` required. | All | ✅ Done 2026-06-16 | **74 107** CHUNK nodes / **74 107** HAS_CHUNK edges (≤ 74 107); every CHUNK has exactly one parent `ART`; `stage5_orphan_chunks.jsonl` written with **0** records; `kg.gpickle` = **141.60 MB** (145 070 nodes / 130 376 edges total) |
| 5.5 | 🔲 Add cross-document edges from `relationships` config; filter to SME doc IDs; map 14 Vietnamese labels via `RELATIONSHIP_MAP` | All | ⏳ | Edge count in [150 000, 350 000] |
| 5.6 | ✅ Add Concept nodes and `MENTIONS` edges via **chunk text extraction** (chunk-first pipeline); curated vocabulary in `config/legal_concepts.yaml`; `chunk_text` from `stage3_chunks.parquet` is the only matching source; article text is not used for edge creation. CLI adds `--legal-concepts-path`; `--stage 5.6 --append` required. | All | ✅ Done 2026-06-18 | **80** CONCEPT nodes / **72 916** `MENTIONS` edges; **41 696** chunks matched at least one concept; `MENTIONS` edges are `CHUNK -> CONCEPT` only; no `ART -> CONCEPT`; `kg.gpickle` = **155.11 MB** (**145 150** nodes / **210 670** edges total) |
| 5.7 | ✅ Persist graph to `kg.gpickle` using `pickle.HIGHEST_PROTOCOL`; CLI adds `--stage 5.7 --append` to re-persist and round-trip-read the final graph, then print an `nx.info`-equivalent summary plus node counts by type. | All | ✅ Done 2026-06-18 | `kg.gpickle` readable after re-write; **145 150** nodes / **210 670** edges; file size **155.10 MB** |
| 5.8 | ✅ Validate: enforce `DOC -> ART -> CHUNK -> CONCEPT`; check every chunk has parent article; no orphan chunks; no `ART -> CONCEPT` edges. CLI adds `--stage 5.8 --append` to run final graph-wide quality gates using runtime acceptance bands and relationship whitelist config. | All | ✅ Done 2026-06-18 | Zero crash; quality gates passed; chunk-first pipeline confirmed with **56 269** ART, **74 107** CHUNK, **80** CONCEPT, **72 916** `MENTIONS`, and **7 378** DOC→DOC edges |

### Stage 6 — Indexing ✅ (VKB, completed 2026-06-23)

> **Artifact bundle redesign (vs. original PLAN):** Stage 6 produced a **SQLite FTS5** lexical backend (`chunk_store.sqlite`, 636,585 chunks) + a **single FAISS `IndexFlatIP`** (`faiss_index__BAAI_bge-m3.index`, dim 1024, L2-normalised = cosine) + `chunk_meta_slim.parquet` sidecar — **not** the original `bm25.pkl` + `faiss_summary` + `faiss_full` + `chunk_meta.npy` design. Bundle contract: **FAISS index position i == chunk_meta_slim.row_idx i == chunks.row_idx i == chunks_fts.rowid i** (must not reorder). Retrieval modules were adapted to this actual bundle.

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 6.1 | ✅ Lexical index as **SQLite FTS5** (`chunks_fts`, tokenizer `unicode61`) with `bm25(chunks_fts)` scoring; `artifact_meta` table records `lexical_backend=fts5`, `rows=636585`, `embed_model=BAAI/bge-m3` | VKB | ✅ Done 2026-06-23 | `chunk_store.sqlite` written; query smoke test returns ranked, query-relevant hits |
| 6.2 | — *Superseded*: no separate FAISS summary index; the single full-chunk FAISS index (6.3) serves both summary and full retrieval legs | — | ➖ N/A | — |
| 6.3 | ✅ Build FAISS full index: embed chunk text with `BAAI/bge-m3` (dim 1024) → single `IndexFlatIP` over L2-normalised vectors (cosine via inner product) | VKB | ✅ Done 2026-06-23 | `faiss_index__BAAI_bge-m3.index` written; **636,585** rows; dim 1024 |
| 6.4 | ✅ Build `chunk_meta_slim.parquet` sidecar (`row_idx`, `chunk_id`, `doc_uid`, `law_id`, `ten_van_ban`, `dieu_so`, `doc_id`); verified 1-to-1 alignment: `index.ntotal == meta.row_idx contiguous 0..N-1 == chunks.row_idx` | VKB | ✅ Done 2026-06-23 | Asserted in [`FAISSIndex.load_index()`](Road2AI_ApplePie/src/retrieval/faiss_index.py:85) |
| 6.5 | ✅ Artifacts placed in `data/stage6_data/` (local) + `embed_model_meta__BAAI_bge-m3.json` model metadata | VKB | ✅ Done 2026-06-23 | Retrieval modules load without error |

> **Runtime note:** Stage 6 embedding is estimated at ~4 GPU-hours on Notebook 03.

---

## Phase 3 — Retrieval & Generation (Online Inference)

### Retrieval Module (Stage 7) ✅ — Zoo 2026-06-23

> Adapted to the actual Stage 6 bundle (SQLite FTS5 + single FAISS `IndexFlatIP`), not the original `bm25.pkl` + two-FAISS design. All heavy GPU deps (`faiss`, `FlagEmbedding`, `torch`) are imported lazily so modules import & unit-test on CPU; the full hybrid + rerank path runs on Kaggle GPU.

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 7.1 | ✅ Implement [`src/retrieval/retriever.py`](Road2AI_ApplePie/src/retrieval/retriever.py): parallel lexical (FTS5) + dense (FAISS) legs (top-50 each); lazy heavy imports; `use_dense`/`use_reranker` flags enable a CPU lexical-only baseline | Zoo | ✅ Done 2026-06-23 | Returns ≥ 1 hit for all 20 devset questions (`test_retrieve_smoke_all_devset_questions`) |
| 7.2 | ✅ Implement [`src/retrieval/rrf.py`](Road2AI_ApplePie/src/retrieval/rrf.py): RRF fusion k=60, fused_top=30, deterministic sort (score desc, row_idx asc, seeded RNG tiebreak, seed 42) | Zoo | ✅ Done 2026-06-23 | `test_determinism` — output identical across re-runs with seed 42 |
| 7.3 | ✅ Implement [`src/retrieval/graph_expand.py`](Road2AI_ApplePie/src/retrieval/graph_expand.py): 1-hop DOC→ART expansion via `{DETAILS,AMENDS,REPLACES,CITES_REF,BASED_ON}` (canonical + reverse), discount 0.6; 1.5-hop concept co-mention (discount 0.3) with graceful degrade when CHUNK/CONCEPT nodes absent | Zoo | ✅ Done 2026-06-23 (code) / ⏳ ablation pending GPU | **CPU baseline Δ=0.0000 (expected)** — expander adds 20 new candidates per query but they are discounted below the 30 lexical candidates, so none reach top-K=5 without a cross-encoder reranker. ≥5 pp recall lift target requires the full GPU pipeline (dense leg + reranker) and will be re-measured on Kaggle. |
| 7.4 | ✅ Implement cross-encoder rerank via `BAAI/bge-reranker-v2-m3` (fp16) in [`Retriever._rerank()`](Road2AI_ApplePie/src/retrieval/retriever.py:348); final top-K=5 | Zoo | ✅ Code done 2026-06-23 / ⏳ F2≥0.55 pending GPU run | Code complete & unit-tested; F2≥0.55 acceptance deferred to the Kaggle GPU run (laptop has no GPU) |
| 7.5 | ✅ Run end-to-end retrieval on devset; record F2 macro as baseline | Zoo | ✅ Done 2026-06-23 | **F2 macro = 0.0670** (lexical-only CPU baseline) committed to [`dev_set/results_baseline.json`](Road2AI_ApplePie/dev_set/results_baseline.json); no-graph ablation in [`results_no_graph.json`](Road2AI_ApplePie/dev_set/results_no_graph.json) |

### Generation Module ⏳

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 8.1 | 🔲 Implement `src/generation/prompt.py`: build prompt from query + top-K hit contexts (1 500 chars/hit) + graph neighbours | All | ⏳ Blocked on Phase 2 | Prompt fits within Gemma-3-12B context; no truncation warnings |
| 8.2 | 🔲 Implement `src/generation/generator.py`: load `google/gemma-3-12b-it` (4-bit QAT NF4/bfloat16, eager attn); `max_new_tokens=900`, `temperature=0.1`, `repetition_penalty=1.05` | All | ⏳ | Loads within 16 GB VRAM; generates for a smoke-test query |
| 8.3 | 🔲 Implement `src/generation/guardrails.py`: assert ≥ 1 `Điều X` citation; reject fabricated `lawid`; min answer 200 chars; append disclaimer if missing; retry up to 2× on fail | All | ⏳ | Guardrail rejects a deliberately bad test response; passes a good one |
| 8.4 | 🔲 Implement post-generation citation augmentation: append `Căn cứ bổ sung Điều X của {tenvanban}` for every top-K hit whose `dieuso` is not already cited verbatim | All | ⏳ | Scoring regex `Điều X` matches all `relevant_articles` in smoke test |
| 8.5 | 🔲 Prompt iteration: run 20 devset questions; review answer quality; adjust context construction and few-shot exemplars | All | ⏳ | QA grounding rate (auto) ≥ 0.80 on devset |

### Pipeline Orchestration ⏳

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 9.1 | 🔲 Implement `pipeline.py`: batch-encode all queries once with `bge-m3`, unload, then load Gemma for generation; persist `results.json` every 50 records | All | ⏳ Blocked on 7.x and 8.x | Runs to completion on Kaggle T4 within 1.5 h per 500 questions |
| 9.2 | 🔲 Implement `scripts/infer.py` CLI wrapper: `python infer.py --test test_set.json --out results.json` | All | ⏳ | Exits 0; output file validates against submission schema |
| 9.3 | 🔲 Implement `scripts/validate_submission.py`: assert all five schema rules (JSON list, five fields, unique IDs match test set, `relevant_docs` format, `relevant_articles` format, answer contains ≥ 1 cited article) | All | ⏳ | Validator catches all intentionally broken sample inputs; passes a valid sample |

---

## Phase 4 — Evaluation, Tuning & Submission

### Offline Evaluation & Ablation

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 10.1 | 🔲 Run `eval.py` on devset; record F2 macro and QA grounding rate | All | 🔲 Open | Numbers logged in `devset/eval_log.md` |
| 10.2 | 🔲 Ablation 1 — retrieval: compare BM25-only, FAISS-only, hybrid, hybrid+graph on devset F2 | All | 🔲 Open | Table committed; best config identified |
| 10.3 | 🔲 Ablation 2 — top-K: test K ∈ {3, 5, 7, 10}; measure F2 and latency | All | 🔲 Open | Optimal K chosen; latency ≤ 15 s on T4 |
| 10.4 | 🔲 Ablation 3 — prompt: test with/without summary prefix, with/without graph context block | All | 🔲 Open | Best prompt variant recorded |
| 10.5 | 🔲 Review 10 failure cases from devset; document error taxonomy (wrong retrieval / hallucinated article / incomplete answer) | All | 🔲 Open | Failure analysis note committed |

### First Public Leaderboard Submission

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 11.1 | 🔲 Run full inference on competition test set | All | ⏳ Blocked on Phase 3 | `results.json` produced for all test questions |
| 11.2 | 🔲 Run `validate_submission.py`; fix any schema errors | All | ⏳ | Validator exits 0 |
| 11.3 | 🔲 Pack `submission.zip` (flat, no subdirectory): `zip submission.zip results.json` | All | ⏳ | `unzip -l submission.zip` shows `results.json` at root |
| 11.4 | 🔲 Upload to leaderboard; record public F2 score | All | ⏳ | Score posted; delta vs. devset F2 < 0.05 |
| 11.5 | 🔲 Promote submission to QA evaluation queue on leaderboard | All | ⏳ | Submission shown as "promoted" on dashboard |

### Iterative Improvement

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 12.1 | 🔲 Based on leaderboard delta, re-tune retrieval or generation as needed (≤ 10 submissions/day) | All | 🔲 Open | Public F2 ≥ 0.55 |
| 12.2 | 🔲 Select best Public Phase submission before private phase opens | All | 🔲 Open | Best submission identified; no more than 5 Private Phase submissions used |

### Private Phase & DemoDay Prep

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 13.1 | 🔲 Final private-phase submissions (max 5 total); choose carefully | All | 🔲 Open | Submitted before 30 Jun 2026 23:59 ICT |
| 13.2 | 🔲 Prepare working-notes paper describing methods, models, data pipeline, and results (required for official results) | All | 🔲 Open | Paper submitted on time |
| 13.3 | 🔲 Prepare DemoDay demo: live or recorded walkthrough of the system answering 3–5 sample questions | All | 🔲 Open | Demo ready by 10 Jul 2026 |

---

## Dependency Graph (summary)

```
Stage1 ✅ → Stage2 ✅ → Stage3 ✅ → Stage5 (DOC→ART→CHUNK→CONCEPT) → Stage6
                                          ↓               ↓
                                     Retrieval ←──────────┘
                                          ↓
                                     Generation
                                          ↓
                                     Submission
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Stage 4 summarisation takes > 17 GPU-hours due to Kaggle quota | Medium | High | Start early; use JSONL cache to spread across sessions; reduce batch to 2 if OOM |
| Stage 2 parse failures for important laws not covered by manual fixes | Medium | High | Prioritise manual-fix review for the 5 key laws before Stage 4 |
| Gemma-3-12B OOM on T4 (16 GB VRAM) | Low | High | Use 4-bit QAT + eager attention; if still OOM, fall back to `google/gemma-3-4b-it` |
| Low public leaderboard F2 (< 0.40) after first submission | Medium | Medium | Pre-validate devset F2 ≥ 0.50 before uploading; do not submit untested builds |
| SME filter too narrow — misses articles the test set expects | Low | High | Widen `SME_TITLE_KEYWORDS`; re-run Stage 1 and downstream stages before final submission |
| Grader regex misses citations in answer | Medium | Medium | Always run `validate_submission.py`; check `Điều X` pattern matches after augmentation |

---

## Key Dates

| Date | Milestone |
|------|-----------|
| 11 Jun 2026 | Phase 0 complete; Stage 3 artifacts on Kaggle |
| 12 Jun 2026 | Stage 2.5 manual review complete |
| 14 Jun 2026 | Stage 4 summarisation considered optional; pipeline validated on Stage 2/3 artifacts |
| 16 Jun 2026 | Stage 5–6 complete; all indexes on Kaggle |
| 18 Jun 2026 | Retrieval module validated on devset (F2 ≥ 0.55) |
| 22 Jun 2026 | Generation + guardrails complete; first full pipeline run |
| 24 Jun 2026 | First public leaderboard submission |
| 28 Jun 2026 | Final leaderboard tuning done |
| **30 Jun 2026** | **Submission deadline 23:59 ICT** |
| 5 Jul 2026 | Top-10 announcement |
| 10 Jul 2026 | DemoDay demo ready |
| **11 Jul 2026** | **DemoDay** |
