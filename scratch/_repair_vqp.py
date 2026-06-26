"""Repair the validate_query_plan / fallback_query_plan corruption caused by the
first patcher run's content-based `plan = {` search matching the wrong function.

This script:
  1. Restores fallback_query_plan to its correct original body.
  2. Upgrades validate_query_plan (the REAL one, located by its `def` line) to
     emit anchor_terms/legal_facets/facet_profiles/domain_profile and call
     enrich_query_plan.
It is idempotent and marker-guarded.
"""
import json
from pathlib import Path

NB = Path("Road2AI_ApplePie/notebooks/retrieval_colab_decomp_submit.ipynb")
C = json.loads(NB.read_text(encoding="utf-8"))
MARKER = "# __ADAPTIVE_K_PORT__"

def cell_lines(cell):
    s = cell["source"]
    return list(s) if isinstance(s, list) else s.split("\n")

def set_cell_source(cell, lines):
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

def find_line(lines, needle, start=0):
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i
    return -1

# Locate the planner cell.
PLANNER_CELL = None
for i, c in enumerate(C["cells"]):
    if c.get("cell_type") != "code":
        continue
    body = "".join(c.get("source", [])) if isinstance(c.get("source"), list) else c.get("source", "")
    if "def validate_query_plan(" in body and "class LLMQueryPlanner" in body:
        PLANNER_CELL = i
        break
assert PLANNER_CELL is not None
print(f"[repair] planner cell = {PLANNER_CELL}")

lines = cell_lines(C["cells"][PLANNER_CELL])

# --- 1. Restore fallback_query_plan -------------------------------------------
# The corrupted body is:
#   def fallback_query_plan(question, reason='rule_fallback'):
#       clauses = split_question_clauses_rule(question)
#       complexity = fallback_complexity(question, clauses)
#       atomic = clauses if complexity != 'simple' else []
#       if not atomic and complexity in {'medium', 'complex'}:
#           atomic = [normalize_text(question)]
#       # # __ADAPTIVE_K_PORT__ VQP-UPGRADE  ...   <-- corruption starts here
#       plan = { ...clean_atomic... }
#       return enrich_query_plan(...)
# We replace from the marker comment line through the `return enrich_query_plan`
# line with the correct fallback body tail.
fb_def = find_line(lines, "def fallback_query_plan(")
assert fb_def >= 0
# Find the corruption marker within fallback_query_plan (before extract_json_object).
marker_line = find_line(lines, MARKER + " VQP-UPGRADE", fb_def)
assert marker_line >= 0, "could not find corrupted VQP-UPGRADE marker in fallback_query_plan"
# The corruption ends at the `return enrich_query_plan` line right after.
ret_enrich = find_line(lines, "return enrich_query_plan(question, refine_query_plan_heuristics(question, plan))", marker_line)
assert ret_enrich > marker_line, "could not find corrupted return enrich_query_plan"

correct_fb_tail = [
    "    plan = {",
    "        'complexity': complexity,",
    "        'atomic_questions': atomic[:CFG['planner_max_atomic']],",
    "        'must_have_terms': fallback_must_terms(question),",
    "        'question_type': detect_question_type(question),",
    "        'rationale_short': reason,",
    "        'planner_fallback': True,",
    "        'planner_error': reason,",
    "        'raw_plan_text': '',",
    "    }",
    "    return enrich_query_plan(question, refine_query_plan_heuristics(question, plan))",
]
lines[marker_line:ret_enrich + 1] = correct_fb_tail
print("[repair] fallback_query_plan restored to correct body")

# --- 2. Upgrade the REAL validate_query_plan ---------------------------------
# Re-read line numbers after the edit above.
vqp_def = find_line(lines, "def validate_query_plan(")
assert vqp_def >= 0
# Find `    plan = {` AFTER the validate_query_plan def line (function-scoped).
plan_start = find_line(lines, "    plan = {", vqp_def)
assert plan_start > vqp_def, "could not find plan = { inside validate_query_plan"
# Find the closing brace.
plan_end = -1
for i in range(plan_start, plan_start + 20):
    if lines[i].strip() == "}":
        plan_end = i
        break
assert plan_end > plan_start, "could not find plan dict close in validate_query_plan"
ret_line = find_line(lines, "return refine_query_plan_heuristics(question, plan)", plan_end)
assert ret_line > plan_end, "could not find return refine_query_plan_heuristics in validate_query_plan"

# Check idempotency: if already upgraded (has the marker + enrich_query_plan), skip.
already = MARKER + " VQP-UPGRADE" in "\n".join(lines[plan_start:ret_line + 1])
if already:
    print("[repair] validate_query_plan already upgraded (skip)")
else:
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
    lines[plan_start:ret_line + 1] = new_plan
    print("[repair] validate_query_plan upgraded (anchor/facet/domain fields + enrich_query_plan)")

set_cell_source(C["cells"][PLANNER_CELL], lines)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(C, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(f"[repair] wrote {NB}")
