"""Dump kaggle cell[3] local lines 584-614 to verify fallback_query_plan body.

Cell-local line numbers are used (NOT file offsets). Per _map_defs:
  L584 fallback_must_terms
  L592 fallback_query_plan
  L645 validate_query_plan
"""
import json, pathlib

NB = pathlib.Path("Road2AI_ApplePie/notebooks/kaggle-hybridrag-decomp-anchor-v2.ipynb")
K = json.loads(NB.read_text())
cell3 = K["cells"][3]["source"]
lines = cell3 if isinstance(cell3, list) else cell3.splitlines(keepends=True)
# strip trailing newline chars for display
print(f"cell[3] total lines: {len(lines)}")
print("=" * 70)
for i in range(584, 615):  # local line index 1-based -> 0-based = i-1
    src = lines[i - 1].rstrip("\n")
    print(f"{i:4d}| {src}")
