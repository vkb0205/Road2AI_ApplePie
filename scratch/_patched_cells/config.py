import os, re, gc, json as _json, zipfile, time
from pathlib import Path

# --- Generation config (mirrors hybridrag-decomp.ipynb CFG) ---------------
# The decomposition/retrieval knobs live in §1; these govern the answer LLM.
GEN_MODEL            = os.environ.get("GEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
USE_GENERATOR        = True                # False -> skip LLM, leave answer empty
GEN_LOAD_IN_4BIT     = os.environ.get("GEN_LOAD_IN_4BIT", "0") == "1"
GEN_MAX_INPUT_TOKENS = int(os.environ.get("GEN_MAX_INPUT_TOKENS", "3072"))
GEN_MAX_NEW_TOKENS   = int(os.environ.get("GEN_MAX_NEW_TOKENS", "448"))
GEN_TEMPERATURE      = float(os.environ.get("GEN_TEMPERATURE", "0.1"))
GEN_TOP_P            = float(os.environ.get("GEN_TOP_P", "0.9"))
GEN_CHUNK_CHAR_LIMIT = int(os.environ.get("GEN_CHUNK_CHAR_LIMIT", "900"))
GEN_CONTEXT_TOPK     = int(os.environ.get("GEN_CONTEXT_TOPK", "6"))
GEN_USE_CACHE        = os.environ.get("GEN_USE_CACHE", "1") == "1"
ARTICLE_CONTEXT_TOPK = int(os.environ.get("ARTICLE_CONTEXT_TOPK", "16"))

# --- Query-planning config (ported from hybridrag-decomp.ipynb §2) -----------
# Drives SIMPLE/MEDIUM/COMPLEX complexity assessment and the per-tier article
# quota via article_bounds_for_complexity(). Set USE_LLM_PLANNER=0 to use only
# the rule-based fallback (no extra GPU model loaded).
USE_LLM_PLANNER          = os.environ.get("USE_LLM_PLANNER", "1") == "1"
PLANNER_MODEL            = os.environ.get("PLANNER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
PLANNER_LOAD_IN_4BIT     = os.environ.get("PLANNER_LOAD_IN_4BIT", "1") == "1"
PLANNER_ALLOW_CPU        = os.environ.get("PLANNER_ALLOW_CPU", "0") == "1"
PLANNER_MAX_NEW_TOKENS   = int(os.environ.get("PLANNER_MAX_NEW_TOKENS", "512"))
PLANNER_MAX_ATOMIC       = int(os.environ.get("PLANNER_MAX_ATOMIC", "4"))
PLANNER_MIN_OVERLAP      = float(os.environ.get("PLANNER_MIN_OVERLAP", "0.18"))
# Per-tier article quotas (max_k returned by article_bounds_for_complexity).
SUBMIT_ARTICLE_MAX       = int(os.environ.get("SUBMIT_ARTICLE_MAX", "16"))
SUBMIT_ARTICLE_MAX_SIMPLE  = int(os.environ.get("SUBMIT_ARTICLE_MAX_SIMPLE", "2"))
SUBMIT_ARTICLE_MAX_MEDIUM  = int(os.environ.get("SUBMIT_ARTICLE_MAX_MEDIUM", "4"))
SUBMIT_ARTICLE_MAX_COMPLEX = int(os.environ.get("SUBMIT_ARTICLE_MAX_COMPLEX", "12"))

# # __ADAPTIVE_K_PORT__ CFG  (adaptive-k port — added config knobs) ----------
# A single CFG dict mirrors the kaggle notebook so the ported helpers can
# read knobs without colliding with the existing flat GEN_*/PLANNER_* names.
CANDIDATE_TOPK            = int(os.environ.get('CANDIDATE_TOPK', '60'))
RERANK_TOPK               = int(os.environ.get('RERANK_TOPK', '48'))
DIAG_TOPK                 = int(os.environ.get('DIAG_TOPK', '12'))
USE_MUST_TERMS_VARIANT    = os.environ.get('USE_MUST_TERMS_VARIANT', '1') == '1'
CANDIDATE_ARTICLE_DEBUG_TOPK = int(os.environ.get('CANDIDATE_ARTICLE_DEBUG_TOPK', '40'))
CFG = {
    'rrf_k': 60,
    'bm25_topk': 60,
    'dense_topk': 60,
    'candidate_topk': CANDIDATE_TOPK,
    'rerank_topk': RERANK_TOPK,
    'article_context_topk': ARTICLE_CONTEXT_TOPK,
    'gen_context_topk': GEN_CONTEXT_TOPK,
    'gen_chunk_char_limit': GEN_CHUNK_CHAR_LIMIT,
    'use_llm_planner': USE_LLM_PLANNER,
    'planner_max_atomic': PLANNER_MAX_ATOMIC,
    'planner_min_overlap': PLANNER_MIN_OVERLAP,
    'planner_max_new_tokens': PLANNER_MAX_NEW_TOKENS,
    'planner_load_in_4bit': PLANNER_LOAD_IN_4BIT,
    'use_must_terms_variant': USE_MUST_TERMS_VARIANT,
    'submit_article_max': SUBMIT_ARTICLE_MAX,
    'submit_article_max_simple': SUBMIT_ARTICLE_MAX_SIMPLE,
    'submit_article_max_medium': SUBMIT_ARTICLE_MAX_MEDIUM,
    'submit_article_max_complex': SUBMIT_ARTICLE_MAX_COMPLEX,
    'candidate_article_debug_topk': CANDIDATE_ARTICLE_DEBUG_TOPK,
    'diag_topk': DIAG_TOPK,
}
print({'candidate_topk': CANDIDATE_TOPK, 'rerank_topk': RERANK_TOPK,
       'use_must_terms_variant': USE_MUST_TERMS_VARIANT,
       'candidate_article_debug_topk': CANDIDATE_ARTICLE_DEBUG_TOPK})
# end # __ADAPTIVE_K_PORT__ CFG ------------------------------------------------

# --- Submission output paths (grader contract: results.json inside the zip) -
RESULTS_PATH        = "/content/results.json"
SUBMISSION_ZIP_PATH = "/content/submission.zip"

print({"gen_model": GEN_MODEL, "use_generator": USE_GENERATOR,
       "load_in_4bit": GEN_LOAD_IN_4BIT, "gen_context_topk": GEN_CONTEXT_TOPK,
       "results_path": RESULTS_PATH, "submission_zip_path": SUBMISSION_ZIP_PATH})