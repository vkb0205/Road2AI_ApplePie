"""Dump all kaggle + colab function bodies needed for the adaptive-k port."""
import json

K = json.load(open("Road2AI_ApplePie/notebooks/kaggle-hybridrag-decomp-anchor-v2.ipynb", encoding="utf-8"))
C = json.load(open("Road2AI_ApplePie/notebooks/retrieval_colab_decomp_submit.ipynb", encoding="utf-8"))

def lines(nb, idx):
    s = nb["cells"][idx]["source"]
    return s if isinstance(s, list) else s.split("\n")

def dump(nb, idx, lo, hi, label):
    L = lines(nb, idx)
    print(f"\n{'#'*78}\n# {label}\n{'#'*78}")
    for i in range(lo, min(hi, len(L))):
        print(f"{i:4d}| {L[i]}")

# ---- KAGGLE cell[3]: anchor + domain + facet + enrich + variants ----
dump(K, 3, 0, 10, "K cell[3] header/imports")
dump(K, 3, 10, 25, "K anchor_plain/_append_unique/_plain_has_any")
dump(K, 3, 25, 180, "K build_rule_domain_profile + is_generic_atomic + repair + jaccard + dedupe")
dump(K, 3, 180, 390, "K facet_profile + sanitize + build_rule_facet_profiles + build_rule_legal_facets + merge")
dump(K, 3, 390, 486, "K enrich_query_plan + fallback_must_terms_inline")
dump(K, 3, 645, 696, "K validate_query_plan")
dump(K, 3, 696, 810, "K LLMQueryPlanner")
dump(K, 3, 809, 867, "K add_variant + must_terms_query + build_query_variants")

# ---- KAGGLE cell[4]: scoring + selection (article_key .. make_relevant_lists) ----
dump(K, 4, 211, 379, "K article_key..score_candidate_facets")
dump(K, 4, 487, 676, "K article_bounds..select_facet_coverage")
dump(K, 4, 795, 824, "K build_gen_contexts + make_relevant_lists (dict-based)")

# ---- COLAB cell[22]: planner (validate_query_plan + LLMQueryPlanner) ----
dump(C, 22, 0, 36, "C cell[22] header/imports")
dump(C, 22, 206, 360, "C validate_query_plan + LLMQueryPlanner")
dump(C, 22, 360, 378, "C article_bounds_for_complexity")
dump(C, 22, 378, 464, "C cell[22] tail (QwenIRACGenerator head + rest)")

# ---- COLAB cell[23]: gen contexts + batch loop ----
dump(C, 23, 0, 157, "C cell[23] build_gen_contexts + batch loop")
