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
| 2.1 | Implement HTML → article-level parser in `src/data/stage2_parse_html.py` (BeautifulSoup + plain-text regex) | KL | ✅ Done | 56 269 article rows from 12 633 documents |
| 2.2 | Emit `stage2_parse_failures.jsonl` with typed reasons (`text_too_short_after_cleaning`, `zero_dieu_law_like_doc`, etc.) | KL | ✅ Done | 2 406 failure records; summary JSON present |
| 2.3 | Apply keep/drop policy: keep fallback `Điều VB` rows and `singledieulawlikedoc`; drop the two drop-classes | KL | ✅ Done | Policy documented in PROGRESS.md |
| 2.4 | 🔲 Stage 2.5 manual review: inspect `single_dieu_law_like_doc` records (70) for the 5 key laws; produce `stage2_manual_fixes.json` if needed | KL | 🔲 Open | Manual review log committed; fixes file present (may be empty) |

### Stage 3 — Chunking ✅

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 3.1 | Implement chunking in `src/data/stage3_chunking.py` (Khoản-aware greedy packing, 1 024 token limit, 128-token overlap) | KL | ✅ Done | 74 107 chunks from 56 269 articles |
| 3.2 | Breadcrumb prefix injected into `chunk_text`; `chunk_id` format `{doc_uid}#{part_idx}` | KL | ✅ Done | Sample inspection passes |
| 3.3 | 🔲 Upload `stage3_chunks.parquet` as Kaggle Dataset so Notebook 02 can mount it | All | 🔲 Open | Dataset visible in Kaggle; Notebook 02 mounts without error |

### Stage 4 — Summary Injection ⏳

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 4.1 | 🔲 Implement `src/data/stage4_summarize.py` using `Qwen/Qwen2.5-3B-Instruct` (4-bit NF4) | All | 🔲 Open | Runs on Kaggle GPU T4; outputs valid `summary_cache.jsonl` |
| 4.2 | 🔲 Prompt engineering: two-field JSON response (`short` ≤ 30 words, `key` 3–5 bullets); fallback on invalid JSON keeps full `chunktext` | All | 🔲 Open | Fewer than 5 % of chunks fall back to empty summary on a 1 000-row smoke test |
| 4.3 | 🔲 JSONL cache resumability: restart reads existing `chunkid` entries and skips them | All | 🔲 Open | Kill and restart mid-corpus; output is identical to a single uninterrupted run |
| 4.4 | 🔲 Produce `stage4_enriched.parquet` with `enriched_text` column (summary prefix + full chunk body) | All | 🔲 Open | `enriched_text` is non-null for every row; spot-check 20 random rows against source articles |
| 4.5 | 🔲 Upload `stage4_enriched.parquet` + `summary_cache.jsonl` as Kaggle Dataset | All | ⏳ Blocked on 4.4 | Dataset visible; Notebook 03 mounts without error |

> **Runtime note:** Stage 4 is estimated at ~17 GPU-hours across multiple sessions.  
> Start Notebook 02 as early as possible to leave buffer for re-runs.

---

## Phase 2 — Graph & Indexing

### Stage 5 — Knowledge Graph ⏳

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 5.1 | 🔲 Implement `src/data/stage5buildgraph.py`; build `networkx.MultiDiGraph` with Document, Article, Concept nodes | All | ⏳ Blocked on 4.4 | Script completes without error |
| 5.2 | 🔲 Add Document nodes from `stage1_sme_docs.parquet` (attributes: `law_id`, `ten`, `loai`, `nganh`, `ngay_ban_hanh`) | All | ⏳ | Node count in [3 000, 8 000] |
| 5.3 | 🔲 Add Article nodes + `HAS_ARTICLE` edges from `stage4_enriched.parquet` (deduplicated by `doc_uid`) | All | ⏳ | Article node count ≈ 50 000 |
| 5.4 | 🔲 Add cross-document edges from `relationships` config; filter to SME doc IDs; map 14 Vietnamese labels via `RELATIONSHIP_MAP` | All | ⏳ | Edge count in [150 000, 350 000] |
| 5.5 | 🔲 Add Concept nodes and `MENTIONS` edges via string-match on `enriched_text` against `config/legal_concepts.yaml` | All | ⏳ | 50–100 concept nodes; `MENTIONS` edges ≈ 30 000–50 000 |
| 5.6 | 🔲 Persist graph to `kg.gpickle` using `pickle.HIGHEST_PROTOCOL` | All | ⏳ | File readable; `nx.info(G)` shows expected node/edge counts |
| 5.7 | 🔲 Validate: log warnings for unmapped relationship labels (stored verbatim under key `rel`) | All | ⏳ | Zero crash; warnings visible in log |

### Stage 6 — Indexing ⏳

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 6.1 | 🔲 Implement BM25 index in `src/data/stage6index.py`; tokenise with `pyvi.ViTokenizer`; parameters k₁=1.5, b=0.75 | All | ⏳ Blocked on 4.4 | `bm25.pkl` written; query smoke test returns non-empty ranking |
| 6.2 | 🔲 Build FAISS summary index: embed `short + " ".join(key)` via `BAAI/bge-m3` (fp16, batch 32, max 256 tokens); `IndexFlatIP` | All | ⏳ | `faiss_summary.index` written; dimension 1 024 |
| 6.3 | 🔲 Build FAISS full index: embed `enriched_text` (batch 8, max 1 024 tokens) | All | ⏳ | `faiss_full.index` written; row count equals `stage4_enriched.parquet` |
| 6.4 | 🔲 Build `chunk_meta.npy` structured array with columns `chunk_id`, `doc_uid`, `law_id`, `ten_van_ban`, `dieu_so`, `doc_id`, `row_idx`; verify 1-to-1 alignment with all three indexes | All | ⏳ | Assert `len(bm25.doc_ids) == len(faiss_summary) == len(faiss_full) == len(chunk_meta)` |
| 6.5 | 🔲 Upload all four index artifacts as Kaggle Dataset for Notebook 04 | All | ⏳ Blocked on 6.1–6.4 | Notebook 04 mounts and loads without error |

> **Runtime note:** Stage 6 embedding is estimated at ~4 GPU-hours on Notebook 03.

---

## Phase 3 — Retrieval & Generation (Online Inference)

### Retrieval Module ⏳

| # | Task | Owner | Status | Acceptance |
|---|------|-------|--------|------------|
| 7.1 | 🔲 Implement `src/retrieval/retriever.py`: parallel BM25 + FAISS-summary + FAISS-full (top-50 each) | All | ⏳ Blocked on Phase 2 | Returns ≥ 1 hit for all 50 devset questions |
| 7.2 | 🔲 Implement `src/retrieval/rrf.py`: RRF fusion with k=60; keep top-30 | All | ⏳ | Output is deterministic across re-runs with seed 42 |
| 7.3 | 🔲 Implement `src/retrieval/graphexpand.py`: 1-hop DOC → ART expansion via `DETAILS/AMENDS/REPLACES/CITESREF/BASISOF`; concept co-mention siblings with discount 0.3; output top-50 | All | ⏳ | Graph expansion increases recall on devset by ≥ 5 pp vs. no expansion |
| 7.4 | 🔲 Implement cross-encoder rerank via `BAAI/bge-reranker-v2-m3` (fp16); keep final top-K = 5 | All | ⏳ | F2 macro on devset ≥ 0.55 |
| 7.5 | 🔲 Run end-to-end retrieval on devset; record F2 macro as baseline for ablation | All | ⏳ | Baseline number committed to `devset/results_baseline.json` |

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
Stage1 ✅ → Stage2 ✅ → Stage3 ✅ → Stage4 → Stage5 → Stage6
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
| 14 Jun 2026 | Stage 4 summarisation complete (`stage4_enriched.parquet` ready) |
| 16 Jun 2026 | Stage 5–6 complete; all indexes on Kaggle |
| 18 Jun 2026 | Retrieval module validated on devset (F2 ≥ 0.55) |
| 22 Jun 2026 | Generation + guardrails complete; first full pipeline run |
| 24 Jun 2026 | First public leaderboard submission |
| 28 Jun 2026 | Final leaderboard tuning done |
| **30 Jun 2026** | **Submission deadline 23:59 ICT** |
| 5 Jul 2026 | Top-10 announcement |
| 10 Jul 2026 | DemoDay demo ready |
| **11 Jul 2026** | **DemoDay** |
