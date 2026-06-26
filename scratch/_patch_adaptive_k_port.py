"""
Idempotent patcher that ports the full "adaptive-k" mechanism from
kaggle-hybridrag-decomp-anchor-v2.ipynb into retrieval_colab_decomp_submit.ipynb.

Strategy
--------
1. Load both notebooks as JSON.
2. Extract exact function bodies from the kaggle notebook by *cell-local* line
   ranges (guarantees fidelity for the Vietnamese-laden rule templates).
3. Insert new helper blocks into the colab planner cell (cell index 22) and the
   generation/batch cell (cell index 23), guarded by sentinel marker comments
   so re-running the patcher is a no-op.
4. In-place upgrade the colab's `validate_query_plan`, `LLMQueryPlanner.build_prompt`
   and the batch-loop selection section.
5. Add a CFG dict + missing config knobs to the colab config cell.

Run:  python Road2AI_ApplePie/scratch/_patch_adaptive_k_port.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_NB = ROOT / "notebooks" / "kaggle-hybridrag-decomp-anchor-v2.ipynb"
COLAB_NB = ROOT / "notebooks" / "retrieval_colab_decomp_submit.ipynb"
MARKER = "# __ADAPTIVE_K_PORT__"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def cell_lines(nb, idx):
    s = nb["cells"][idx]["source"]
    return list(s) if isinstance(s, list) else s.split("\n")


def set_cell_source(cell, lines):
    """Store source as a list of strings; nbformat tolerates either but the
    canonical form is a list where each element ends with \\n except the last."""
    # Keep existing trailing-newline convention: ensure every line but the last
    # carries a trailing newline, the last carries none.
    norm = []
    for i, ln in enumerate(lines):
        if i < len(lines) - 1:
            if not ln.endswith("\n"):
                ln = ln + "\n"
        else:
            if ln.endswith("\n"):
                ln = ln[:-1]
        norm.append(ln)
    cell["source"] = norm


def kaggle_slice(K, cell_idx, lo, hi):
    """Return kaggle source lines [lo, hi) (cell-local 0-based), de-blanked of
    the blank-line doubling that some editors inject. We keep them verbatim."""
    L = cell_lines(K, cell_idx)
    return L[lo:hi]


def find_line(lines, needle, start=0):
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i
    return -1


def has_marker(lines, marker=MARKER):
    return any(marker in ln for ln in lines)


# ---------------------------------------------------------------------------
# load notebooks
# ---------------------------------------------------------------------------
K = json.loads(KAGGLE_NB.read_text(encoding="utf-8"))
C = json.loads(COLAB_NB.read_text(encoding="utf-8"))

# Kaggle cell indices (from _map_defs output):
#   cell[3] -> planner helpers + LLMQueryPlanner + variant builders
#   cell[4] -> scoring + selection
# Colab cell indices (from _inspect_nb_cells output):
#   cell[20] -> config (GEN_MODEL ... SUBMISSION_ZIP_PATH)
#   cell[22] -> text helpers + planner + LLMQueryPlanner + article_bounds
#   cell[23] -> build_gen_contexts + batch loop

# Locate the colab config cell dynamically (the one defining GEN_MODEL).
CFG_CELL = None
for i, c in enumerate(C["cells"]):
    if c.get("cell_type") != "code":
        continue
    src = c.get("source", [])
    body = "".join(src) if isinstance(src, list) else src
    if "GEN_MODEL" in body and "SUBMIT_ARTICLE_MAX" in body and "RESULTS_PATH" in body:
        CFG_CELL = i
        break
assert CFG_CELL is not None, "could not locate colab config cell"

# Locate the colab planner cell (defines validate_query_plan + LLMQueryPlanner).
PLANNER_CELL = None
for i, c in enumerate(C["cells"]):
    if c.get("cell_type") != "code":
        continue
    src = c.get("source", [])
    body = "".join(src) if isinstance(src, list) else src
    if "def validate_query_plan(" in body and "class LLMQueryPlanner" in body:
        PLANNER_CELL = i
        break
assert PLANNER_CELL is not None, "could not locate colab planner cell"

# Locate the colab batch cell (defines build_gen_contexts(hits, ...)).
BATCH_CELL = None
for i, c in enumerate(C["cells"]):
    if c.get("cell_type") != "code":
        continue
    src = c.get("source", [])
    body = "".join(src) if isinstance(src, list) else src
    if "def build_gen_contexts(hits" in body and "decomp_retriever.retrieve" in body:
        BATCH_CELL = i
        break
assert BATCH_CELL is not None, "could not locate colab batch cell"

print(f"[patch] colab config cell  = {CFG_CELL}")
print(f"[patch] colab planner cell = {PLANNER_CELL}")
print(f"[patch] colab batch cell   = {BATCH_CELL}")


# ---------------------------------------------------------------------------
# Module 0 — CFG dict + missing config knobs in the config cell
# ---------------------------------------------------------------------------
cfg_lines = cell_lines(C, CFG_CELL)
if not has_marker(cfg_lines, MARKER + " CFG"):
    # Insert right before the "# --- Submission output paths" block.
    ins = find_line(cfg_lines, "# --- Submission output paths")
    assert ins >= 0, "could not find submission-paths anchor in config cell"
    cfg_block = [
        "# " + MARKER + " CFG  (adaptive-k port — added config knobs) ----------",
        "# A single CFG dict mirrors the kaggle notebook so the ported helpers can",
        "# read knobs without colliding with the existing flat GEN_*/PLANNER_* names.",
        "CANDIDATE_TOPK            = int(os.environ.get('CANDIDATE_TOPK', '60'))",
        "RERANK_TOPK               = int(os.environ.get('RERANK_TOPK', '48'))",
        "DIAG_TOPK                 = int(os.environ.get('DIAG_TOPK', '12'))",
        "USE_MUST_TERMS_VARIANT    = os.environ.get('USE_MUST_TERMS_VARIANT', '1') == '1'",
        "CANDIDATE_ARTICLE_DEBUG_TOPK = int(os.environ.get('CANDIDATE_ARTICLE_DEBUG_TOPK', '40'))",
        "CFG = {",
        "    'rrf_k': 60,",
        "    'bm25_topk': 60,",
        "    'dense_topk': 60,",
        "    'candidate_topk': CANDIDATE_TOPK,",
        "    'rerank_topk': RERANK_TOPK,",
        "    'article_context_topk': ARTICLE_CONTEXT_TOPK,",
        "    'gen_context_topk': GEN_CONTEXT_TOPK,",
        "    'gen_chunk_char_limit': GEN_CHUNK_CHAR_LIMIT,",
        "    'use_llm_planner': USE_LLM_PLANNER,",
        "    'planner_max_atomic': PLANNER_MAX_ATOMIC,",
        "    'planner_min_overlap': PLANNER_MIN_OVERLAP,",
        "    'planner_max_new_tokens': PLANNER_MAX_NEW_TOKENS,",
        "    'planner_load_in_4bit': PLANNER_LOAD_IN_4BIT,",
        "    'use_must_terms_variant': USE_MUST_TERMS_VARIANT,",
        "    'submit_article_max': SUBMIT_ARTICLE_MAX,",
        "    'submit_article_max_simple': SUBMIT_ARTICLE_MAX_SIMPLE,",
        "    'submit_article_max_medium': SUBMIT_ARTICLE_MAX_MEDIUM,",
        "    'submit_article_max_complex': SUBMIT_ARTICLE_MAX_COMPLEX,",
        "    'candidate_article_debug_topk': CANDIDATE_ARTICLE_DEBUG_TOPK,",
        "    'diag_topk': DIAG_TOPK,",
        "}",
        "print({'candidate_topk': CANDIDATE_TOPK, 'rerank_topk': RERANK_TOPK,",
        "       'use_must_terms_variant': USE_MUST_TERMS_VARIANT,",
        "       'candidate_article_debug_topk': CANDIDATE_ARTICLE_DEBUG_TOPK})",
        "# end " + MARKER + " CFG ------------------------------------------------",
        "",
    ]
    cfg_lines[ins:ins] = cfg_block
    set_cell_source(C["cells"][CFG_CELL], cfg_lines)
    print("[patch] Module 0: CFG dict + missing knobs inserted into config cell")
else:
    print("[patch] Module 0: CFG block already present (skip)")


# ---------------------------------------------------------------------------
# Build the big helper block to insert into the planner cell.
# Extracted verbatim from kaggle cell[3] by cell-local line ranges.
# ---------------------------------------------------------------------------
# cell[3] layout (from _map_defs):
#   L10   anchor_plain
#   L16   _append_unique
#   L21   _plain_has_any
#   L25   build_rule_domain_profile
#   L126  is_generic_atomic_question
#   L137  repair_generic_atomic_question
#   L151  lexical_jaccard
#   L158  atomic_questions_too_similar
#   L167  dedupe_atomic_questions
#   L180  facet_profile
#   L203  sanitize_facet_profiles
#   L234  build_rule_facet_profiles
#   L372  build_rule_legal_facets
#   L378  merge_facet_profiles
#   L390  enrich_query_plan
#   L434  fallback_must_terms_inline
#   L486  split_question_clauses_rule   <-- STOP here (already in colab)
# So the new helper block is cell[3] L10 .. L485 (inclusive).

k_planner_helpers = kaggle_slice(K, 3, 10, 486)

# Variant builders: cell[3] L809 .. L867 (add_variant, must_terms_query,
# build_query_variants). Stop at L867 (load_or_build_query_plan_cache) which is
# kaggle-only.
k_variant_builders = kaggle_slice(K, 3, 809, 867)

# ---------------------------------------------------------------------------
# Build the scoring + selection block for the batch cell.
# Extracted verbatim from kaggle cell[4] by cell-local line ranges.
# ---------------------------------------------------------------------------
# cell[4] layout (from _map_defs):
#   L211  weighted_rrf_fuse          (skip — colab retriever already fuses)
#   L230  article_key
#   L238  source_kind
#   L244  canonical_article_key
#   L252  score_candidate_domain
#   L293  is_generic_variant_text
#   L301  candidate_facet_haystack
#   L311  score_one_facet
#   L356  score_candidate_facets
#   L379  aggregate_article_candidates_from_variants
#   L487  article_bounds_for_complexity  (already in colab — skip)
#   L494  article_law_family
#   L497  add_selected_article
#   ...   selection helpers ...
#   L676  select_article_contexts
#   L795  build_gen_contexts (dict-based)
#   L805  make_relevant_lists (dict-based)
#   L824  compact_hit (kaggle-only — skip)
k_scoring = kaggle_slice(K, 4, 230, 487)        # article_key .. aggregate (excl. weighted_rrf_fuse)
k_selection = kaggle_slice(K, 4, 494, 794)       # article_law_family .. select_article_contexts (excl. article_bounds)
k_dict_helpers = kaggle_slice(K, 4, 795, 823)    # build_gen_contexts + make_relevant_lists (dict-based)


# ---------------------------------------------------------------------------
# Module 1-5 — insert planner helpers + variant builders into planner cell.
# Insert BEFORE `def validate_query_plan(` so they are defined first.
# ---------------------------------------------------------------------------
planner_lines = cell_lines(C, PLANNER_CELL)
if not has_marker(planner_lines, MARKER + " PLANNER-HELPERS"):
    vq_anchor = find_line(planner_lines, "def validate_query_plan(")
    assert vq_anchor >= 0, "could not find validate_query_plan in planner cell"
    header = [
        "# " + MARKER + " PLANNER-HELPERS  (adaptive-k port: anchor/domain/facet/variant) -----",
        "import math",
        "from collections import OrderedDict",
        "",
    ]
    block = header + k_planner_helpers + [""] + k_variant_builders + [
        "",
        "# end " + MARKER + " PLANNER-HELPERS ------------------------------------------------",
        "",
    ]
    planner_lines[vq_anchor:vq_anchor] = block
    set_cell_source(C["cells"][PLANNER_CELL], planner_lines)
    print("[patch] Modules 1-5: planner helpers + variant builders inserted")
else:
    print("[patch] Modules 1-5: planner helpers already present (skip)")


# ---------------------------------------------------------------------------
# Module 4 — in-place upgrade of validate_query_plan (add 4 plan fields +
# call enrich_query_plan instead of returning refine_query_plan_heuristics).
# ---------------------------------------------------------------------------
planner_lines = cell_lines(C, PLANNER_CELL)
if not has_marker(planner_lines, MARKER + " VQP-UPGRADE"):
    # Replace the plan dict + final return of the colab validate_query_plan.
    # Colab original (cell-local after the helper insertion may have shifted,
    # so we search by content, not line number):
    #   plan = {
    #       'complexity': complexity,
    #       'atomic_questions': clean_atomic,
    #       'must_have_terms': terms,
    #       'question_type': qtype,
    #       'rationale_short': normalize_text(raw_plan.get('rationale_short', ''))[:260],
    #       'planner_fallback': False,
    #       'planner_error': '',
    #       'raw_plan_text': raw_text,
    #   }
    #   return refine_query_plan_heuristics(question, plan)
    # Anchor on the validate_query_plan def line FIRST, then search for its
    # plan dict AFTER it. This avoids matching fallback_query_plan's `plan = {`
    # (which appears earlier in the cell and has a different shape).
    vqp_def = find_line(planner_lines, "def validate_query_plan(")
    assert vqp_def >= 0, "could not find def validate_query_plan in planner cell"
    plan_start = find_line(planner_lines, "    plan = {", vqp_def)
    # Make sure we found the *colab* validate_query_plan plan dict (the one
    # that contains 'rationale_short' within a few lines and is followed by
    # `return refine_query_plan_heuristics`).
    assert plan_start > vqp_def, "could not find plan = { inside validate_query_plan"
    # find the closing brace line
    plan_end = -1
    for i in range(plan_start, plan_start + 20):
        if planner_lines[i].strip() == "}":
            plan_end = i
            break
    assert plan_end > plan_start, "could not find plan dict close brace"
    ret_line = find_line(planner_lines, "return refine_query_plan_heuristics(question, plan)", plan_end)
    assert ret_line >= 0, "could not find return refine_query_plan_heuristics"

    new_plan = [
        "    # " + MARKER + " VQP-UPGRADE  (emit anchor_terms/legal_facets/facet_profiles/domain_profile)",
        "    plan = {",
        "        'complexity': complexity,",
        "        'atomic_questions': clean_atomic,",
        "        'must_have_terms': terms,",
        "        'anchor_terms': sanitize_list_of_strings(raw_plan.get('anchor_terms', []), max_items=12, max_chars=80),",
        "        'legal_facets': sanitize_list_of_strings(raw_plan.get('legal_facets', []), max_items=max(6, CFG['planner_max_atomic']), max_chars=220),",
        "        'facet_profiles': sanitize_facet_profiles(raw_plan.get('facet_profiles', []), max_items=max(6, CFG['planner_max_atomic'])),",
        "        'domain_profile': raw_plan.get('domain_profile', {}) if isinstance(raw_plan.get('domain_profile', {}), dict) else {},",
        "        'question_type': qtype,",
        "        'rationale_short': normalize_text(raw_plan.get('rationale_short', ''))[:260],",
        "        'planner_fallback': False,",
        "        'planner_error': '',",
        "        'raw_plan_text': raw_text,",
        "    }",
        "    return enrich_query_plan(question, refine_query_plan_heuristics(question, plan))",
    ]
    # Replace [plan_start .. ret_line] inclusive.
    planner_lines[plan_start:ret_line + 1] = new_plan
    set_cell_source(C["cells"][PLANNER_CELL], planner_lines)
    print("[patch] Module 4: validate_query_plan upgraded (anchor/facet/domain fields + enrich_query_plan)")
else:
    print("[patch] Module 4: validate_query_plan already upgraded (skip)")


# ---------------------------------------------------------------------------
# Module 4 — in-place upgrade of LLMQueryPlanner.build_prompt JSON schema.
# Replace the colab schema block with the kaggle schema block (adds the 4 new
# fields + the domain-anchor rules).
# ---------------------------------------------------------------------------
planner_lines = cell_lines(C, PLANNER_CELL)
if not has_marker(planner_lines, MARKER + " PROMPT-UPGRADE"):
    # The colab prompt body is a triple-quoted f-string. We replace the segment
    # between 'Hãy trả về JSON theo schema:' and the closing triple-quote line
    # of the user string, then the rules block, with the kaggle version.
    schema_anchor = find_line(planner_lines, "Hãy trả về JSON theo schema:")
    assert schema_anchor >= 0, "could not find JSON schema anchor in build_prompt"
    # Find the closing triple-quote of the user f-string (a line that is "  '''")
    # within ~40 lines after the anchor.
    close_idx = -1
    for i in range(schema_anchor, schema_anchor + 60):
        if planner_lines[i].strip().endswith("'''"):
            close_idx = i
            break
    assert close_idx > schema_anchor, "could not find user f-string close"
    new_schema = [
        "Hãy trả về JSON theo schema:",
        "{{",
        "  \"complexity\": \"simple|medium|complex\",",
        "  \"atomic_questions\": [\"mệnh đề truy hồi độc lập, tối đa 4\"],",
        "  \"must_have_terms\": [\"thuật ngữ bắt buộc lấy từ hoặc bám rất sát câu hỏi\"],",
        "  \"anchor_terms\": [\"domain anchors that every atomic question must preserve\"],",
        "  \"legal_facets\": [\"separate legal retrieval facets, not paraphrases\"],",
        "  \"facet_profiles\": [{{\"facet_id\": \"stable_id\", \"label\": \"legal facet\", \"anchor_terms\": [], \"preferred_law_ids\": [], \"preferred_title_terms\": [], \"target_terms\": [], \"negative_terms\": [], \"priority\": 1.0}}],",
        "  \"domain_profile\": {{\"labels\": [\"domain labels if clear\"]}},",
        "  \"question_type\": \"procedure|condition|rights_obligations|sanction|support_incentive|scenario|deadline|comparison|definition_listing|other\",",
        "  \"rationale_short\": \"lý do ngắn, không quá 1 câu\"",
        "}}",
        "",
        "Quy tắc:",
        "- Every atomic question must preserve specific domain anchors from the original question, such as copyright, software, customs, assessment, vulnerable consumers, small and medium enterprises, or procurement.",
        "- Do not shorten an atomic question into a generic form like documents/evidence, contract contents, dispute handling, or request processing when the original question contains a specific domain.",
        "- legal_facets should split legal aspects, not paraphrase the same question multiple times.",
        "- facet_profiles should name the legal facets that need separate slot coverage; keep law/title preferences only when directly implied by the question.",
        "- simple: 1 vấn đề, thường 1 văn bản/1 điều.",
        "- medium: 2 ý hoặc cross-doc nhẹ.",
        "- complex: nhiều mệnh đề, multi-hop, nhiều lĩnh vực/văn bản.",
        "- atomic_questions phải giữ đúng ý gốc, không thêm vấn đề mới.",
        "- Không nêu \"Điều X\", mã luật, tên văn bản cụ thể nếu câu hỏi không nêu.",
        "- Không dùng markdown, không giải thích ngoài JSON.",
        "'''",
        "        # " + MARKER + " PROMPT-UPGRADE",
    ]
    planner_lines[schema_anchor:close_idx + 1] = new_schema
    set_cell_source(C["cells"][PLANNER_CELL], planner_lines)
    print("[patch] Module 4: build_prompt JSON schema upgraded (anchor/facet/domain fields + rules)")
else:
    print("[patch] Module 4: build_prompt already upgraded (skip)")


# ---------------------------------------------------------------------------
# Module 6-7 — insert scoring + selection + adapter + dict-based helpers into
# the batch cell, BEFORE `def build_gen_contexts(hits`.
# ---------------------------------------------------------------------------
batch_lines = cell_lines(C, BATCH_CELL)
if not has_marker(batch_lines, MARKER + " SCORING-SELECTION"):
    bgc_anchor = find_line(batch_lines, "def build_gen_contexts(hits")
    assert bgc_anchor >= 0, "could not find build_gen_contexts(hits) in batch cell"

    # The adapter bridges the colab flat List[Hit] to the kaggle article_candidates
    # dict schema expected by select_article_contexts / aggregate_*.
    adapter = [
        "# " + MARKER + " ADAPTER  (colab List[Hit] -> kaggle article_candidates dict) ----------",
        "def _hit_to_candidate(h, rank, variant_kind, rrf_score=0.0):",
        "    \"\"\"Convert a retrieval Hit (or its dict form) into a kaggle-style",
        "    candidate dict carrying the keys consumed by score_candidate_domain,",
        "    score_candidate_facets, article_key, canonical_article_key, and the",
        "    generation context builder.\"\"\"",
        "    if isinstance(h, dict):",
        "        d = h",
        "    else:",
        "        d = h.to_dict() if hasattr(h, 'to_dict') else {",
        "            'row_idx': getattr(h, 'row_idx', 0),",
        "            'score': getattr(h, 'score', 0.0),",
        "            'source': getattr(h, 'source', ''),",
        "            'law_id': getattr(h, 'law_id', '') or '',",
        "            'ten_van_ban': getattr(h, 'ten_van_ban', '') or '',",
        "            'dieu_so': getattr(h, 'dieu_so', '') or '',",
        "            'chunk_id': getattr(h, 'chunk_id', '') or '',",
        "            'doc_uid': getattr(h, 'doc_uid', '') or '',",
        "            'chunk_text': getattr(h, 'chunk_text', '') or '',",
        "        }",
        "    c = {",
        "        'law_id': str(d.get('law_id', '') or '').strip(),",
        "        'ten_van_ban': str(d.get('ten_van_ban', '') or '').strip(),",
        "        'dieu_so': str(d.get('dieu_so', '') or '').strip(),",
        "        'chunk_id': str(d.get('chunk_id', '') or '').strip(),",
        "        'doc_uid': str(d.get('doc_uid', '') or '').strip(),",
        "        'chunk_text': str(d.get('chunk_text', '') or ''),",
        "        'row_idx': int(d.get('row_idx', 0) or 0),",
        "        'source': str(d.get('source', '') or ''),",
        "        'rerank_score': float(d.get('score', 0.0) or 0.0),",
        "        'rrf_score': float(rrf_score or 0.0),",
        "        'source_hits': [{'source': str(d.get('source', '') or '')}],",
        "        'variant_kind': str(variant_kind or ''),",
        "    }",
        "    return c",
        "",
        "",
        "def build_article_candidates_from_hits(hits, query_plan, variants=None, sub_query_traces=None):",
        "    \"\"\"Bridge the colab DecomposingHybridRetriever flat Hit list to the",
        "    kaggle article_candidates schema consumed by select_article_contexts.",
        "",
        "    The colab retriever already fuses per-sub-query Hit lists by row_idx",
        "    (DecomposingHybridRetriever._fuse_hits), so we treat the single fused",
        "    `hits` list as the 'original' variant ranking. When sub_query_traces",
        "    are available (decomp_retriever.last_sub_query_traces), each trace's",
        "    final_hits becomes an 'atomic_N' variant ranking, giving the aggregator",
        "    genuine per-variant support signals without re-running retrieval.",
        "    \"\"\"",
        "    variants = variants or build_query_variants(str(query_plan.get('question', '')), query_plan)",
        "    sub_query_traces = sub_query_traces if sub_query_traces is not None else []",
        "    topk = int(CFG.get('candidate_topk', 60))",
        "    rerank_topk = int(CFG.get('rerank_topk', 48))",
        "",
        "    variant_results = []",
        "",
        "    # 'original' variant = the fused hit list.",
        "    orig_variant = {'kind': 'original', 'weight': 1.0, 'text': str(query_plan.get('question', ''))}",
        "    orig_reranked = []",
        "    for rank, h in enumerate(hits[:rerank_topk], start=1):",
        "        rrf = 1.0 / (CFG.get('rrf_k', 60) + rank)",
        "        c = _hit_to_candidate(h, rank, 'original', rrf_score=rrf)",
        "        orig_reranked.append(c)",
        "    variant_results.append({'variant': orig_variant, 'reranked': orig_reranked})",
        "",
        "    # 'atomic_N' variants = per-sub-query traces (the decomposed leg hits).",
        "    for idx, tr in enumerate(sub_query_traces, start=1):",
        "        kind = f'atomic_{idx}'",
        "        text = getattr(tr, 'sub_query_text', '') or (tr.get('sub_query_text') if isinstance(tr, dict) else '')",
        "        final_hits = getattr(tr, 'final_hits', None)",
        "        if final_hits is None and isinstance(tr, dict):",
        "            final_hits = tr.get('final_hits', [])",
        "        final_hits = list(final_hits or [])",
        "        if not final_hits:",
        "            continue",
        "        reranked = []",
        "        for rank, h in enumerate(final_hits[:rerank_topk], start=1):",
        "            rrf = 1.0 / (CFG.get('rrf_k', 60) + rank)",
        "            reranked.append(_hit_to_candidate(h, rank, kind, rrf_score=rrf))",
        "        variant_results.append({'variant': {'kind': kind, 'weight': 0.85, 'text': text}, 'reranked': reranked})",
        "",
        "    # 'must_terms' variant = the must-terms query string (re-uses the same",
        "    # fused hits but tags the top rerank_topk so the aggregator can apply its",
        "    # weak_penalty / support logic.",
        "    if CFG.get('use_must_terms_variant', True):",
        "        mq = must_terms_query(query_plan)",
        "        if mq:",
        "            mt_reranked = []",
        "            for rank, h in enumerate(hits[:rerank_topk], start=1):",
        "                rrf = 1.0 / (CFG.get('rrf_k', 60) + rank)",
        "                mt_reranked.append(_hit_to_candidate(h, rank, 'must_terms', rrf_score=rrf))",
        "            variant_results.append({'variant': {'kind': 'must_terms', 'weight': 0.6, 'text': mq}, 'reranked': mt_reranked})",
        "",
        "    candidates = aggregate_article_candidates_from_variants(variant_results, query_plan=query_plan)",
        "    return candidates[:topk]",
        "",
        "",
        "# kaggle dict-based helpers renamed to avoid clobbering the src import.",
        "def build_gen_contexts_from_articles(article_contexts):",
        "    gen_contexts = []",
        "    for c in article_contexts:",
        "        item = dict(c)",
        "        item['chunk_text'] = str(item.get('chunk_text', ''))[:CFG['gen_chunk_char_limit']]",
        "        gen_contexts.append(item)",
        "        if len(gen_contexts) >= CFG['gen_context_topk']:",
        "            break",
        "    return gen_contexts",
        "",
        "",
        "def make_relevant_lists_from_articles(article_contexts):",
        "    docs, articles = [], []",
        "    seen_docs, seen_articles = set(), set()",
        "    for c in article_contexts:",
        "        law_id = str(c.get('law_id', '')).strip()",
        "        ten = str(c.get('ten_van_ban', '')).strip()",
        "        dieu = str(c.get('dieu_so', '')).strip()",
        "        if law_id and ten:",
        "            d = f'{law_id}|{ten}'",
        "            if d not in seen_docs:",
        "                seen_docs.add(d)",
        "                docs.append(d)",
        "        if law_id and ten and dieu:",
        "            a = f'{law_id}|{ten}|{dieu}'",
        "            if a not in seen_articles:",
        "                seen_articles.add(a)",
        "                articles.append(a)",
        "    return docs, articles",
        "",
        "# end " + MARKER + " ADAPTER ---------------------------------------------------",
        "",
    ]

    block = (
        ["# " + MARKER + " SCORING-SELECTION  (adaptive-k port: scoring + selection) ----------", ""]
        + k_scoring
        + ["", ""]
        + k_selection
        + ["", ""]
        + adapter
        + ["# end " + MARKER + " SCORING-SELECTION ------------------------------------------------", ""]
    )
    batch_lines[bgc_anchor:bgc_anchor] = block
    set_cell_source(C["cells"][BATCH_CELL], batch_lines)
    print("[patch] Modules 6-7: scoring + selection + adapter + dict helpers inserted into batch cell")
else:
    print("[patch] Modules 6-7: scoring/selection already present (skip)")


# ---------------------------------------------------------------------------
# Module 8 — rewire the batch loop selection section.
# Replace the block:
#     docs, articles = make_relevant_lists(hits)
#     # Apply the per-complexity-tier article quota.
#     bounds = article_bounds_for_complexity(complexity)
#     _max_articles = bounds['max_k']
#     articles = articles[:_max_articles]
#     _keep_doc_keys = {a.rsplit('|', 1)[0] for a in articles if '|' in a}
#     docs = [d for d in docs if d in _keep_doc_keys]
#     contexts = build_gen_contexts(hits)
# with the new adaptive-k flow.
# ---------------------------------------------------------------------------
batch_lines = cell_lines(C, BATCH_CELL)
if not has_marker(batch_lines, MARKER + " LOOP-REWIRE"):
    loop_start = find_line(batch_lines, "docs, articles = make_relevant_lists(hits)")
    assert loop_start >= 0, "could not find make_relevant_lists(hits) in batch loop"
    # The block ends at the `contexts = build_gen_contexts(hits)` line.
    ctx_line = find_line(batch_lines, "contexts = build_gen_contexts(hits)", loop_start)
    assert ctx_line > loop_start, "could not find contexts = build_gen_contexts(hits)"
    new_loop = [
        "    # " + MARKER + " LOOP-REWIRE  (adaptive-k selection flow) ---------------------------",
        "    # Build per-variant article candidates from the fused Hit list + the",
        "    # decomposed sub-query traces, then run the facet-coverage-aware selector.",
        "    _variants = build_query_variants(str(query), query_plan)",
        "    _traces = getattr(decomp_retriever, 'last_sub_query_traces', None) or []",
        "    _candidates = build_article_candidates_from_hits(hits, query_plan, variants=_variants, sub_query_traces=_traces)",
        "    _article_contexts, _sel_debug = select_article_contexts(_candidates, query_plan, variants=_variants)",
        "    docs, articles = make_relevant_lists_from_articles(_article_contexts)",
        "    contexts = build_gen_contexts_from_articles(_article_contexts)",
        "    # Trim docs to those whose law_id+ten_van_ban match a kept article.",
        "    _keep_doc_keys = {a.rsplit('|', 1)[0] for a in articles if '|' in a}",
        "    docs = [d for d in docs if d in _keep_doc_keys]",
    ]
    # Replace [loop_start .. ctx_line] inclusive.
    batch_lines[loop_start:ctx_line + 1] = new_loop
    set_cell_source(C["cells"][BATCH_CELL], batch_lines)
    print("[patch] Module 8: batch loop rewired (adaptive-k selection flow)")
else:
    print("[patch] Module 8: batch loop already rewired (skip)")


# ---------------------------------------------------------------------------
# Module 9 — optional: stash selection_debug + compact candidate slice on the
# per-record output dict, behind a DIAG_TOPK / debug flag.
# ---------------------------------------------------------------------------
batch_lines = cell_lines(C, BATCH_CELL)
if not has_marker(batch_lines, MARKER + " RECORD-DEBUG"):
    # Insert after the `out[\"relevant_articles\"] = articles` line.
    rec_anchor = find_line(batch_lines, 'out["relevant_articles"] = articles')
    assert rec_anchor >= 0, "could not find out['relevant_articles'] = articles"
    debug_block = [
        "    # " + MARKER + " RECORD-DEBUG  (optional selection diagnostics) ---------------------",
        "    if DIAG_TOPK > 0:",
        "        out[\"selection_debug\"] = {",
        "            \"complexity\": _sel_debug.get(\"complexity\", complexity),",
        "            \"selected_k\": int(_sel_debug.get(\"selected_k\", len(articles))),",
        "            \"reason\": _sel_debug.get(\"reason\", \"\"),",
        "            \"covered_facets\": list(_sel_debug.get(\"covered_facets\", [])),",
        "            \"candidate_topk\": [",
        "                {k: c.get(k) for k in (\"article_key\", \"canonical_article_key\",",
        "                 \"article_score\", \"domain_score\", \"facet_score\",",
        "                 \"matched_facets\", \"best_variant_kind\", \"support_variants\")}",
        "                for c in _candidates[:DIAG_TOPK]",
        "            ],",
        "        }",
    ]
    batch_lines[rec_anchor + 1:rec_anchor + 1] = debug_block
    set_cell_source(C["cells"][BATCH_CELL], batch_lines)
    print("[patch] Module 9: selection_debug + candidate_topk slice added to record output")
else:
    print("[patch] Module 9: record debug already present (skip)")


# ---------------------------------------------------------------------------
# write back
# ---------------------------------------------------------------------------
with open(COLAB_NB, "w", encoding="utf-8") as f:
    json.dump(C, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(f"\n[patch] wrote {COLAB_NB}")
