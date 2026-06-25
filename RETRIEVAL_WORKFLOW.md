# Retrieval Workflow — `HybridRetriever.retrieve()`

> Implementation-accurate walkthrough of the G-LRAG online retrieval pipeline
> defined in [`src/retrieval/retriever.py`](src/retrieval/retriever.py:1).
>
> This document complements:
> - the high-level spec — [`G-LRAG_SPECIFICATIONS.md`](G-LRAG_SPECIFICATIONS.md:1) §10 *Retrieval Specification*
> - the build plan — [`PLAN.md`](PLAN.md:1) Stage 7 (tasks 7.1–7.5)
> - the artifact bundle contract — [`data/stage6_data/artifacts_guide.md`](data/stage6_data/artifacts_guide.md:1)
> - the build log — [`PROGRESS.md`](PROGRESS.md:1) *Stage 7 retrieval checkpoint*

---

## 1. Pipeline at a glance

`HybridRetriever.retrieve(query, fetch_text=False)` runs a 7-stage hybrid
retrieval pipeline per query. All stages are joined on a single key —
`row_idx` — so RRF, graph expansion, metadata fetch and output assembly never
need to translate ids.

```
query
  │
  ├── 1. lexical leg   (FTS5 BM25,  top_bm25  = 50)  ─┐
  ├── 2. dense leg     (FAISS|Qdrant, top_dense = 50) ├→ 3. RRF fuse  (fused_top    = 30)
  │   [optional, use_dense]                             │
  └─────────────────────────────────────────────────────┴→ 4. graph expand (expanded_top = 50)
       [optional, graph_expander]                          │
                                                     5. fetch metadata + text  [fetch_text]
                                                            │
                                                     6. cross-encoder rerank    [optional, use_reranker]
                                                            │
                                                     7. build final top-K Hit list (final_top_k = 5)
                                                            │
                                                     → make_relevant_lists(hits) → (relevant_docs, relevant_articles)
```

The entry point is [`HybridRetriever.retrieve()`](src/retrieval/retriever.py:180);
the orchestrating class is [`HybridRetriever`](src/retrieval/retriever.py:140).

---

## 2. The 7 stages in detail

### Stage 1 — Lexical leg

[`retrieve()`](src/retrieval/retriever.py:194) calls
[`_lexical_leg()`](src/retrieval/retriever.py:269), which delegates to
[`FTSIndex.search()`](src/retrieval/bm25_index.py:171).

- Builds an FTS5 `MATCH` expression: the whole query as a quoted phrase first
  (highest phrase-match weight) followed by each individual token OR'd in.
- Orders by `bm25(chunks_fts)` in `bm25_ranked` mode (the committed-baseline
  mode; `fts_fast` is available for smoke tests but returns unranked row order).
- Returns up to `top_bm25` (default **50**) dicts, each carrying
  `row_idx`, `chunk_id`, `doc_uid`, `law_id`, `ten_van_ban`, `dieu_so`,
  `bm25_score`.

This leg **always runs** — it is the backbone of the CPU-only baseline and the
single source of metadata/text in Stage 5.

### Stage 2 — Dense leg (optional)

Guarded by `cfg.use_dense`. [`_dense_leg()`](src/retrieval/retriever.py:275):

1. Requires a `query_encoder` (raises `RuntimeError` if `use_dense=True` but no
   encoder was passed) — typically a `BGEQueryEncoder` producing a
   `(1, dim)` float32, L2-normalised vector.
2. Calls the pluggable dense backend's `search(query_vec, top_k)`:
   - [`FAISSIndex.search()`](src/retrieval/faiss_index.py:132) — local
     `IndexFlatIP` (cosine via inner product over normalised vectors), or
   - [`QdrantIndex.search()`](src/retrieval/qdrant_index.py:142) — remote
     Qdrant collection sharing the identical contract.
3. Returns up to `top_dense` (default **50**) dicts keyed by `row_idx`.

Skipped entirely when `use_dense=False` → pure lexical CPU baseline. The dense
backend is selected via `config/default.yaml` `retrieval.dense_backend:
faiss|qdrant`; no other call-site change is required.

### Stage 3 — RRF fusion

[`rrf_fuse()`](src/retrieval/rrf.py:42) fuses the lexical + dense ranked lists:

- Score for a candidate `d` appearing at 1-based ranks `r_1 … r_m`:
  `rrf_score(d) = Σ 1/(k + r_i)`, with `k = 60` (RRF smoothing constant).
- **Deterministic ordering**: `(-rrf_score, row_idx, seeded_rng)` — score desc,
  then `row_idx` asc, then a seeded RNG tiebreak (`seed = 42`) so identical-score
  collisions break identically across runs (satisfies `test_determinism`).
- Truncated to `fused_top` (default **30**) dicts of
  `{row_idx, rrf_score, source:"fused", payload}`.
- Pure Python — no heavy deps — so it runs on any machine.

When only the lexical leg produced hits, fusion still runs over a single ranking
(no-op degenerate case) to keep the contract uniform.

### Stage 4 — Graph expansion (optional)

If a [`GraphExpander`](src/retrieval/graph_expand.py:71) is wired in,
[`GraphExpander.expand()`](src/retrieval/graph_expand.py:219) takes the fused
seeds `[(row_idx, rrf_score), …]` and adds graph-derived neighbours, then
truncates to `expanded_top` (default **50**):

- **1-hop DOC→ART** — [`_expand_doc_hop()`](src/retrieval/graph_expand.py:252),
  discount `DISCOUNT_DOC = 0.6`:
  seed chunk → parent `DOC:{doc_id}` → cross-document relations
  `EXPANSION_DOC_RELS = {DETAILS, AMENDS, REPLACES, CITES_REF, BASED_ON}`
  (canonical out-edges **and** reverse in-edges) → neighbour DOCs' article
  chunks.
- **1.5-hop concept co-mention** —
  [`_expand_concept_comention()`](src/retrieval/graph_expand.py:301),
  discount `DISCOUNT_CONCEPT = 0.3` (only when both CHUNK and CONCEPT nodes
  are present): seed chunk → `MENTIONS` → CONCEPT nodes → sibling chunks that
  also `MENTIONS` those concepts.

Invariants:
- **Seeds are always retained** at full score (`source:"candidate"`); expanded
  entries are discounted so they only survive if the cross-encoder reranker
  (Stage 6) judges them relevant.
- **Graceful degradation**: when CHUNK/CONCEPT nodes are absent (e.g. a
  DOC→ART-only graph), concept co-mention is silently skipped — verified by the
  CPU baseline ablation.
- Output sorted by `(-score, row_idx)` and truncated.

If no expander is configured, the fused dicts are re-wrapped with
`source:"fused"` so downstream code is unchanged.

### Stage 5 — Fetch metadata + text

[`_fetch_meta()`](src/retrieval/retriever.py:291) and
[`_fetch_text()`](src/retrieval/retriever.py:313) both call
[`FTSIndex.fetch_chunks()`](src/retrieval/bm25_index.py:227) on the ordered
`row_idx` list.

- The canonical `chunks` table is the **single source of truth** for metadata
  and text: because `row_idx` is the bundle join key (Stage 6 invariant),
  `fetch_chunks` over the canonical `row_idx` always hits the same rows,
  regardless of which dense backend produced the candidate. No backfill from
  Qdrant payloads is ever needed.
- `_fetch_text` runs only when `fetch_text=True` (required for rerank + the
  generation context).

### Stage 6 — Cross-encoder rerank (optional)

Guarded by `cfg.use_reranker and cfg.final_top_k > 0`. [`_rerank()`](src/retrieval/retriever.py:322):

- **Lazy imports** `torch` + `FlagEmbedding.FlagReranker` inside the method, so
  the module imports and unit-tests on CPU-only machines with none of those
  installed.
- Loads `BAAI/bge-reranker-v2-m3` in fp16.
- Builds `(query, chunk_text)` pairs, truncating text to
  `rerank_max_input_chars = 2000`.
- `compute_score(pairs, normalize=True)` → re-sorts by
  `(-rerank_score, -original_score, row_idx)`. The original score is kept as a
  fallback tiebreaker.
- Promotes `rerank_score` into the `score` field and sets `source:"rerank"`.
- **Graceful fallback**: if `torch`/`FlagEmbedding`/GPU are unavailable, it
  silently returns the pre-rerank ordering (CPU baseline path).

### Stage 7 — Build final top-K Hit list

The ordered list is truncated to `cfg.final_top_k` (default **5**) and each entry
is materialised into a [`Hit`](src/retrieval/retriever.py:71) dataclass carrying
`row_idx`, `score`, `source`, and the legal metadata (`law_id`, `ten_van_ban`,
`dieu_so`, `chunk_id`, `doc_uid`, `chunk_text`). This is the object downstream
generation (Stage 8.1) consumes.

---

## 3. Downstream — submission metadata

[`make_relevant_lists()`](src/retrieval/retriever.py:103) collapses the final
top-K [`Hit`](src/retrieval/retriever.py:71) list into the two grader-expected
outputs (mirrors `artifacts_guide.md` *Tạo submission metadata* and the grader's
`Điều X` normalisation in `dev_set/eval.py`):

| Output | Format | Filter | Dedupe |
|---|---|---|---|
| `relevant_docs` | `"{law_id}\|{ten_van_ban}"` | non-empty `law_id` **and** `ten_van_ban` | yes, order-preserving |
| `relevant_articles` | `"{law_id}\|{ten_van_ban}\|{dieu_so}"` | above **and** `dieu_so.startswith("Điều")` | yes, order-preserving |

The `Điều` prefix filter keeps non-article hits out of `relevant_articles` so
the F2 article score stays clean.

---

## 4. Configuration knobs

[`RetrievalConfig`](src/retrieval/retriever.py:54) — defaults mirror
`config/default.yaml`:

| Knob | Default | Role |
|---|---|---|
| `use_dense` | `False` | enable the FAISS/Qdrant dense leg |
| `use_reranker` | `False` | enable the cross-encoder rerank step |
| `top_bm25` | `50` | lexical leg truncation |
| `top_dense` | `50` | dense leg truncation |
| `rrf_k` | `60` | RRF smoothing constant |
| `fused_top` | `30` | post-RRF truncation |
| `expanded_top` | `50` | post-graph-expansion truncation |
| `final_top_k` | `5` | final Hit list size |
| `rerank_model` | `BAAI/bge-reranker-v2-m3` | cross-encoder model id |
| `rerank_max_input_chars` | `2000` | per-pair text truncation |
| `seed` | `42` | deterministic RNG tiebreak in RRF |
| `debug` | `False` | record a per-stage `RetrievalTrace` on `last_trace` |
| `debug_top_n` | `8` | items kept per stage in the trace's `top_items` |
| `debug_print` | `False` | also `print()` the formatted trace at the end of `retrieve()` |

---

## 5. Design invariants

| Invariant | Enforced by |
|---|---|
| `row_idx` is the single join key across the whole pipeline | [`rrf_fuse()`](src/retrieval/rrf.py:42), [`GraphExpander.expand()`](src/retrieval/graph_expand.py:219), [`_fetch_meta()`](src/retrieval/retriever.py:291) |
| Heavy GPU deps (`faiss`, `torch`, `FlagEmbedding`, `qdrant_client`) imported lazily inside methods | [`_dense_leg()`](src/retrieval/retriever.py:275), [`_rerank()`](src/retrieval/retriever.py:322) |
| CPU-only baseline via `use_dense=False, use_reranker=False` (lexical + graph, no GPU deps) | [`retrieve()`](src/retrieval/retriever.py:180) guards on both flags |
| Dense backend is pluggable (FAISS **or** Qdrant) via shared `search(query_vec, top_k)` contract | [`_dense_leg()`](src/retrieval/retriever.py:275) |
| Deterministic, reproducible fusion | seeded RNG tiebreak in [`rrf_fuse()`](src/retrieval/rrf.py:107) |
| Graph expansion gracefully degrades when CHUNK/CONCEPT absent | [`GraphExpander`](src/retrieval/graph_expand.py:71) `_has_chunks` / `_has_concepts` flags |
| Canonical chunks table is the sole metadata/text source | [`FTSIndex.fetch_chunks()`](src/retrieval/bm25_index.py:227) used by both `_fetch_meta` and `_fetch_text` |

---

## 6. The CPU baseline path (PLAN 7.5 acceptance)

With `use_dense=False, use_reranker=False` the pipeline collapses to:

```
FTS5 BM25 (top 50) → RRF (single-list, no-op fuse) → graph expand (top 50)
   → fetch metadata → top-5 Hit → make_relevant_lists
```

- No `faiss` / `torch` / `FlagEmbedding` / `qdrant_client` import is triggered.
- Committed baseline: **F2 macro = 0.0670** — see
  [`dev_set/results_baseline.json`](dev_set/results_baseline.json) and the
  no-graph ablation in [`dev_set/results_no_graph.json`](dev_set/results_no_graph.json).
- Graph vs no-graph ablation Δ = +0.0000 on the CPU baseline (expected: the
  discounted graph candidates only lift F2 once the dense leg + reranker are
  enabled on GPU — PLAN 7.3/7.4).

---

## 7. Component contracts (quick reference)

| Component | Contract | File |
|---|---|---|
| `FTSIndex.search(query, top_k)` | `→ list[{row_idx, chunk_id, doc_uid, law_id, ten_van_ban, dieu_so, bm25_score}]` | [`bm25_index.py`](src/retrieval/bm25_index.py:171) |
| `FTSIndex.fetch_chunks(row_idxs)` | `→ list[{row_idx, chunk_id, doc_uid, law_id, ten_van_ban, dieu_so, chunk_text}]` | [`bm25_index.py`](src/retrieval/bm25_index.py:227) |
| `FAISSIndex.search(query_vec, top_k)` | `→ list[{row_idx, ...}]` | [`faiss_index.py`](src/retrieval/faiss_index.py:132) |
| `QdrantIndex.search(query_vec, top_k)` | `→ list[{row_idx, ...}]` (same shape as FAISS) | [`qdrant_index.py`](src/retrieval/qdrant_index.py:142) |
| `rrf_fuse(rankings, k, fused_top, seed)` | `→ list[{row_idx, rrf_score, source, payload}]` | [`rrf.py`](src/retrieval/rrf.py:42) |
| `GraphExpander.expand(candidates, top_n)` | `→ list[{row_idx, score, source}]` | [`graph_expand.py`](src/retrieval/graph_expand.py:219) |
| `query_encoder.encode(query)` | `→ (1, dim) float32` | (pluggable, e.g. `BGEQueryEncoder`) |
| `Hit` dataclass | `{row_idx, score, source, law_id, ten_van_ban, dieu_so, chunk_id, doc_uid, chunk_text}` | [`retriever.py`](src/retrieval/retriever.py:71) |
| `make_relevant_lists(hits)` | `→ (relevant_docs, relevant_articles)` | [`retriever.py`](src/retrieval/retriever.py:103) |

---

## 8. Debug / explainability — `RetrievalTrace`

When retrieval quality is not what you expect, you want to see *what each
stage produced*, not just the final hits. The retriever records an opt-in
per-stage trace via [`src/retrieval/debug.py`](src/retrieval/debug.py).

### How to turn it on

Two equivalent ways — either is zero-overhead when off:

```python
# (a) flag on the config — every retrieve() call is traced
cfg = RetrievalConfig(use_dense=False, use_reranker=False, debug=True, debug_print=True)
r = HybridRetriever(fts, config=cfg)
r.retrieve("đăng ký doanh nghiệp", fetch_text=True)

# (b) one-off — force debug on for a single call, then restore the flags
r.debug_retrieve("đăng ký doanh nghiệp", fetch_text=True, print_trace=True)
```

The trace lands on `r.last_trace` (a [`RetrievalTrace`](src/retrieval/debug.py)),
and `r.last_trace_formatted` renders it as a readable string.

### What each stage records

[`retrieve()`](src/retrieval/retriever.py:211) appends one
[`StageSnapshot`](src/retrieval/debug.py) per stage, in pipeline order:

| Stage `name` | `count` | `top_items` (top-N with legal metadata) | key `diagnostics` |
|---|---|---|---|
| `lexical` | FTS hits | `bm25_score` + `law_id\|ten\|dieu` | — |
| `dense` | dense hits | `dense_score` + meta | — |
| `rrf` | fused candidates | `rrf_score` + meta | `rankings_in`, `k`, `fused_top` |
| `graph` | expanded candidates | `score` + `source` + meta | `source_counts` (candidate/doc_expand/concept_expand) |
| `fetch` | resolved metadata rows | meta + `chunk_id`/`doc_uid` | `requested`, `resolved`, **`missing_rows`**, `text_fetched` |
| `rerank` | post-rerank order | `rerank_score` + meta | `candidates_in`, `model`, `reranker_unavailable` |
| `final` | top-K hits | final `Hit` fields | `final_top_k` |
| `output` | top-K hits | — | `relevant_docs`, `relevant_articles` |

Every stage also carries:
- **`elapsed_ms`** — wall-clock time (or `None` when skipped before work).
- **`skip`** — a human-readable reason when a stage is off or degrades
  (e.g. `"use_dense=False"`, `"no graph_expander"`,
  `"reranker deps unavailable → kept pre-rerank order"`). This distinguishes
  *a stage was disabled* from *a stage ran but found nothing* (count=0, skip=None).

The single most useful silent-failure signal is the fetch stage's
`missing_rows`: any `row_idx` that survives RRF/graph but cannot be resolved
from the canonical `chunks` table is dropped silently — and that is exactly
the kind of leak that tanks the F2 article score without any other symptom.

### Design invariants

| Invariant | Enforced by |
|---|---|
| **Zero overhead when `debug=False`** — the trace is never constructed, no snapshot dicts allocated on the hot path (PLAN 7.5 acceptance path untouched) | [`retrieve()`](src/retrieval/retriever.py:211) gates all trace code behind `if trace is not None` |
| Debug-on and debug-off produce **identical hits** | covered by `test_debug_off_and_on_produce_same_hits` |
| `debug_retrieve()` restores the config flags afterward | covered by `test_debug_retrieve_restores_config_and_prints` |
| No heavy deps in the debug module | [`debug.py`](src/retrieval/debug.py) imports only `time` + `dataclasses` |
| Reusable by the advanced notebook's `AdvancedHybridRetriever` | same `RetrievalTrace`/`StageSnapshot`/`format_trace` helpers ([§5b](notebooks/retrieval_colab_advanced.ipynb)) |

### Advanced retriever (notebook) trace

The Colab notebook's multi-query `AdvancedHybridRetriever` layers its own
stages on top and supports `explain=True`, recording a trace with stages
`variants → legs → weighted_rrf → graph → fetch → rerank → article_agg →
final → output` on `advanced_retriever.last_trace`. See the **5b.
Explainability** section of [`retrieval_colab_advanced.ipynb`](notebooks/retrieval_colab_advanced.ipynb).

---

## 9. Cross-references

- **Spec**: [`G-LRAG_SPECIFICATIONS.md`](G-LRAG_SPECIFICATIONS.md:1) §10
  *Retrieval Specification* (RRF formula §10.3, graph expansion edges §10.4).
- **Plan**: [`PLAN.md`](PLAN.md:1) Stage 7 — tasks 7.1 (retriever), 7.2 (RRF),
  7.3 (graph expand), 7.4 (rerank), 7.5 (baseline).
- **KG design**: [`KG.md`](KG.md:1) — node-id conventions
  (`DOC:{doc_id}`, `ART:{doc_uid}`, `CHUNK:{chunk_id}`, `CONCEPT:{name_lower}`)
  and edge types used by expansion.
- **Artifact bundle**: [`data/stage6_data/artifacts_guide.md`](data/stage6_data/artifacts_guide.md:1)
  — the `row_idx` invariant and the standard retrieval flow.
- **Build log**: [`PROGRESS.md`](PROGRESS.md:1) *Stage 7 retrieval checkpoint*
  (file inventory, test results, baseline numbers).
- **Tests**: `tests/test_retrieval.py` — 33 passing, including
  `test_retrieve_smoke_all_devset_questions` (Stage 7.1 acceptance) and
  `test_determinism` (Stage 7.2).

---

## 9. Input & output of each stage

The pipeline is a chain: each stage consumes the previous stage's output and
emits a typed structure. The universal join key — **`row_idx`** (`int`) — flows
through every stage unchanged, so no id translation is ever needed.

### Entry point

[`retrieve(query, fetch_text=False)`](src/retrieval/retriever.py:180)

| Param | Type | Notes |
|---|---|---|
| `query` | `str` | the user question |
| `fetch_text` | `bool` | `False` → no `chunk_text` fetched; `True` → needed for rerank + generation |

Final return: `List[Hit]` (built in Stage 7).

### Stage 1 — Lexical leg ([`_lexical_leg()`](src/retrieval/retriever.py:269))

**Input**: `query: str`, `top_k: int` (= `cfg.top_bm25`, default 50)
**Mechanism**: [`FTSIndex.search()`](src/retrieval/bm25_index.py:171) — FTS5 `MATCH` (whole phrase + token ORs) ordered by `bm25(chunks_fts)`.
**Output**: `List[Dict[str, Any]]` (best→worst, truncated to `top_k`)

```python
[{"row_idx": int, "chunk_id": str, "doc_uid": str,
  "law_id": str, "ten_van_ban": str, "dieu_so": str,
  "bm25_score": float}, ...]   # bm25_score only in "bm25_ranked" mode
```

`[]` if `self.fts is None` or the query tokenizes to nothing.

### Stage 2 — Dense leg ([`_dense_leg()`](src/retrieval/retriever.py:275), optional)

**Input**: `query: str` (encoded by `query_encoder.encode(query)` → `(1, dim) float32`), `top_k: int` (= `cfg.top_dense`, default 50)
**Mechanism**: encodes query → pluggable backend `search(query_vec, top_k)` — [`FAISSIndex.search()`](src/retrieval/faiss_index.py:132) or [`QdrantIndex.search()`](src/retrieval/qdrant_index.py:142) (identical contract). Raises `RuntimeError` if `use_dense=True` but no `query_encoder`.
**Output**: `List[Dict[str, Any]]` (ranked by similarity, truncated to `top_k`)

```python
[{"row_idx": int, "dense_rank": int, "dense_score": float,
  "chunk_id": str, "doc_uid": str, "law_id": str,
  "ten_van_ban": str, "dieu_so": str}, ...]
```

Skipped (`[]`) when `use_dense=False` → CPU baseline.

### Stage 3 — RRF fusion ([`rrf_fuse()`](src/retrieval/rrf.py:42))

**Input**: `rankings: List[List[Dict | int]]` (= `[lexical_hits, dense_hits]` — only legs that produced hits), `k: int = 60`, `fused_top: int = 30`, `seed: int = 42`
**Mechanism**: per candidate at 1-based rank `r` in list `i`, adds `1/(k + r)`. Sort key `(-rrf_score, row_idx, seeded_rng.random())`.
**Output**: `List[Dict[str, Any]]` (sorted, truncated to `fused_top`)

```python
[{"row_idx": int, "rrf_score": float,
  "source": "fused", "payload": Dict | None}, ...]
```

`[]` if both legs were empty.

### Stage 4 — Graph expansion ([`GraphExpander.expand()`](src/retrieval/graph_expand.py:219), optional)

**Input**: `candidates: List[Tuple[int, float]]` (= `[(r["row_idx"], r["rrf_score"]) for r in fused]`), `top_n: int = 50`
**Mechanism**: retains seeds at full score; adds 1-hop DOC→ART neighbours at `score × 0.6` (`source="doc_expand"`) and 1.5-hop concept co-mention siblings at `score × 0.3` (`source="concept_expand"`).
**Output**: `List[Dict[str, Any]]` (sorted by `(-score, row_idx)`, truncated to `top_n`)

```python
[{"row_idx": int, "score": float,
  "source": "candidate" | "doc_expand" | "concept_expand"}, ...]
```

When no expander is configured, the retriever re-wraps the fused dicts ([retriever.py:221](src/retrieval/retriever.py:221)) as `{"row_idx", "score"=rrf_score, "source"="fused"}`. The retriever then keys these by `row_idx` into `expanded_rows` and sorts by `(-score, row_idx)` → `ordered` ([retriever.py:230](src/retrieval/retriever.py:230)).

### Stage 5 — Fetch metadata + text ([`_fetch_meta()`](src/retrieval/retriever.py:291) / [`_fetch_text()`](src/retrieval/retriever.py:313))

**Input**: `row_idxs: Sequence[int]` (= `[e["row_idx"] for e in ordered]`)
**Mechanism**: both call [`FTSIndex.fetch_chunks()`](src/retrieval/bm25_index.py:227) on the canonical `chunks` table (single source of truth).
**Output** — two mappings:

`_fetch_meta()` → `Dict[int, Dict[str, Any]]` (keyed by `row_idx`)

```python
{row_idx: {"row_idx": int, "chunk_id": str, "doc_uid": str,
           "law_id": str, "ten_van_ban": str, "dieu_so": str}, ...}
```

`_fetch_text()` → `Dict[int, str]` (only built when `fetch_text=True`)

```python
{row_idx: "full chunk_text string", ...}
```

### Stage 6 — Cross-encoder rerank ([`_rerank()`](src/retrieval/retriever.py:322), optional)

**Input**: `query: str`, `ordered: List[Dict]` (post-expansion), `meta_by_row: Dict[int, Dict]`, `text_by_row: Dict[int, str]` (required for scoring)
**Mechanism**: lazily imports `FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)`, builds `(query, text[:2000])` pairs, `compute_score(pairs, normalize=True)`. New sort `(-rerank_score, -original_score, row_idx)`.
**Output**: the **same `ordered` list, mutated in place**

```python
[{"row_idx": int, "score": float,        # OVERWRITTEN with rerank_score
  "source": "rerank",                    # OVERWRITTEN
  "rerank_score": float}, ...]           # ADDED
```

**Graceful fallback**: if `torch`/`FlagEmbedding`/GPU unavailable, returns `ordered` unchanged (CPU baseline).

### Stage 7 — Build final Hit list ([retriever.py:247](src/retrieval/retriever.py:247))

**Input**: `ordered: List[Dict]` (post-rerank or post-expansion), `meta_by_row`, `text_by_row`, `cfg.final_top_k = 5`
**Output**: `List[Hit]` (truncated to `final_top_k`)

```python
[Hit(row_idx=int, score=float, source=str,
     law_id=str, ten_van_ban=str, dieu_so=str,
     chunk_id=str, doc_uid=str, chunk_text=str), ...]
```

`source` is one of `"rerank" | "candidate" | "doc_expand" | "concept_expand" | "fused"`. `chunk_text` is `""` unless `fetch_text=True`. [`Hit.to_dict()`](src/retrieval/retriever.py:85) yields the JSON-serialisable subset (omits `chunk_text`).

### Post — [`make_relevant_lists()`](src/retrieval/retriever.py:103)

**Input**: `Sequence[Hit]`
**Output**: `Tuple[List[str], List[str]]`

```python
(["{law_id}|{ten_van_ban}", ...],                          # relevant_docs
 ["{law_id}|{ten_van_ban}|{dieu_so}", ...])                # relevant_articles (dieu_so startswith "Điều" only)
```

### Data-flow summary

| Stage | Input | Output type | Field carried forward |
|---|---|---|---|
| 1 Lexical | `query, top_k` | `List[Dict]` + `bm25_score` | `row_idx` |
| 2 Dense | `query_vec, top_k` | `List[Dict]` + `dense_score` | `row_idx` |
| 3 RRF | `[lexical, dense]` | `List[Dict]` + `rrf_score` | `row_idx` |
| 4 Graph expand | `[(row_idx, rrf_score)]` | `List[Dict]` + `source` | `row_idx` |
| 5 Fetch | `[row_idx, ...]` | `Dict[int, Dict]` + `Dict[int, str]` | `row_idx` |
| 6 Rerank | `ordered, text_by_row` | mutated `List[Dict]` + `rerank_score` | `row_idx` |
| 7 Build Hit | `ordered, meta, text` | `List[Hit]` (top-K) | `row_idx` |
| Post | `List[Hit]` | `(relevant_docs, relevant_articles)` | — |
