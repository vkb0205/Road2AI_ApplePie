"""Extract the patched colab cells to .py files and compile-check them."""
import json, py_compile, tempfile, os, sys

C = json.load(open("Road2AI_ApplePie/notebooks/retrieval_colab_decomp_submit.ipynb", encoding="utf-8"))

def cell_src(c):
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s

# The three cells we patched.
targets = {}
for i, c in enumerate(C["cells"]):
    if c.get("cell_type") != "code":
        continue
    body = cell_src(c)
    if "GEN_MODEL" in body and "__ADAPTIVE_K_PORT__ CFG" in body:
        targets["config"] = (i, body)
    if "def validate_query_plan(" in body and "class LLMQueryPlanner" in body:
        targets["planner"] = (i, body)
    if "def build_gen_contexts(hits" in body and "decomp_retriever.retrieve" in body:
        targets["batch"] = (i, body)

print("found cells:", {k: v[0] for k, v in targets.items()})

os.makedirs("Road2AI_ApplePie/scratch/_patched_cells", exist_ok=True)
ok = True
for name, (idx, body) in targets.items():
    path = f"Road2AI_ApplePie/scratch/_patched_cells/{name}.py"
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    try:
        py_compile.compile(path, doraise=True)
        print(f"  [ok] {name} cell[{idx}] compiles ({len(body.splitlines())} lines)")
    except py_compile.PyCompileError as e:
        ok = False
        print(f"  [FAIL] {name} cell[{idx}]: {e}")

# Also check the planner cell references: confirm CFG, OrderedDict, math, and
# the new functions are present and referenced consistently.
import re
planner = targets["planner"][1]
batch = targets["batch"][1]
config = targets["config"][1]

checks = {
    "CFG defined in config": "CFG = {" in config,
    "math imported in planner": "import math" in planner,
    "OrderedDict imported in planner": "from collections import OrderedDict" in planner,
    "anchor_plain in planner": "def anchor_plain(" in planner,
    "build_rule_domain_profile in planner": "def build_rule_domain_profile(" in planner,
    "build_rule_facet_profiles in planner": "def build_rule_facet_profiles(" in planner,
    "enrich_query_plan in planner": "def enrich_query_plan(" in planner,
    "validate_query_plan calls enrich_query_plan": "return enrich_query_plan(question, refine_query_plan_heuristics(question, plan))" in planner,
    "build_prompt has anchor_terms schema": '"anchor_terms"' in planner,
    "build_prompt has facet_profiles schema": '"facet_profiles"' in planner,
    "build_query_variants in planner": "def build_query_variants(" in planner,
    "article_key in batch": "def article_key(" in batch,
    "score_candidate_domain in batch": "def score_candidate_domain(" in batch,
    "aggregate_article_candidates_from_variants in batch": "def aggregate_article_candidates_from_variants(" in batch,
    "select_article_contexts in batch": "def select_article_contexts(" in batch,
    "build_article_candidates_from_hits adapter in batch": "def build_article_candidates_from_hits(" in batch,
    "build_gen_contexts_from_articles in batch": "def build_gen_contexts_from_articles(" in batch,
    "make_relevant_lists_from_articles in batch": "def make_relevant_lists_from_articles(" in batch,
    "batch loop rewired": "build_article_candidates_from_hits(hits, query_plan" in batch,
    "selection_debug in batch": '"selection_debug"' in batch,
}
print("\n--- reference checks ---")
all_ok = True
for k, v in checks.items():
    flag = "ok" if v else "MISSING"
    if not v:
        all_ok = False
    print(f"  [{flag}] {k}")

sys.exit(0 if (ok and all_ok) else 1)
