"""Dump specific kaggle function bodies by (cell_idx, start_local, end_local)."""
import json

with open("Road2AI_ApplePie/notebooks/kaggle-hybridrag-decomp-anchor-v2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

def dump(cell_idx, lo, hi, label):
    cell = nb["cells"][cell_idx]
    src = cell["source"]
    lines = src if isinstance(src, list) else src.split("\n")
    print(f"\n{'='*70}\n# {label}  (cell[{cell_idx}] L{lo}-L{hi})\n{'='*70}")
    for i in range(lo, min(hi, len(lines))):
        print(f"{i:4d}| {lines[i]}")

# cell[4] local lines from _map_defs output:
# L379 aggregate_article_candidates_from_variants .. L487 article_bounds_for_complexity
dump(4, 379, 487, "aggregate_article_candidates_from_variants")
# L676 select_article_contexts .. L795 build_gen_contexts
dump(4, 676, 795, "select_article_contexts")
# L795 build_gen_contexts .. L824 compact_hit
dump(4, 795, 824, "build_gen_contexts + make_relevant_lists")
