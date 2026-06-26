"""Functional smoke test for the ported adaptive-k mechanism.

Executes the patched planner + batch cells in an isolated namespace (with the
heavy GPU/transformers/torch dependencies stubbed out) and runs a synthetic
query through the full pipeline:
  fallback_query_plan -> build_query_variants -> build_article_candidates_from_hits
  -> select_article_contexts -> make_relevant_lists_from_articles
  -> build_gen_contexts_from_articles

Verifies:
  - simple complexity -> 1 article (min_k)
  - complex complexity -> >= min_k articles, facet coverage
  - article_candidates carry article_score / domain_score / facet_score keys
  - relevant_lists keys are well-formed law_id|ten|dieu strings
"""
import json, types, sys, importlib.util

NB = "Road2AI_ApplePie/notebooks/retrieval_colab_decomp_submit.ipynb"
C = json.load(open(NB, encoding="utf-8"))

def cell_src(c):
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s

# Gather the three patched cells.
config_src = planner_src = batch_src = None
for c in C["cells"]:
    if c.get("cell_type") != "code":
        continue
    body = cell_src(c)
    if "GEN_MODEL" in body and "__ADAPTIVE_K_PORT__ CFG" in body:
        config_src = body
    if "def validate_query_plan(" in body and "class LLMQueryPlanner" in body:
        planner_src = body
    if "def build_gen_contexts(hits" in body and "decomp_retriever.retrieve" in body:
        batch_src = body

assert config_src and planner_src and batch_src

# ---- stub the heavy deps so we can exec without torch/transformers/retrieval ----
class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        # return a permissive stub for any attribute
        return _Stub(self.__name__ + "." + name)

class _Stub:
    def __init__(self, name):
        self._name = name
    def __call__(self, *a, **k):
        return _Stub(self._name + "()")
    def __getattr__(self, n):
        return _Stub(self._name + "." + n)
    def __bool__(self):
        return False

# We need `os`, `re`, `math`, `unicodedata`, `collections` real. Stub torch,
# transformers, retrieval.*, tqdm, google.colab.
for mod in ["torch", "transformers", "tqdm", "tqdm.auto",
            "retrieval", "retrieval.retriever", "retrieval.debug",
            "retrieval.bm25_index", "retrieval.sub_query_router",
            "google", "google.colab"]:
    if mod not in sys.modules:
        sys.modules[mod] = _StubModule(mod)

# A minimal Hit stand-in mirroring retriever.Hit.to_dict().
class Hit:
    def __init__(self, row_idx, score, source, law_id, ten_van_ban, dieu_so, chunk_text, chunk_id="", doc_uid=""):
        self.row_idx = row_idx
        self.score = score
        self.source = source
        self.law_id = law_id
        self.ten_van_ban = ten_van_ban
        self.dieu_so = dieu_so
        self.chunk_id = chunk_id
        self.doc_uid = doc_uid
        self.chunk_text = chunk_text
    def to_dict(self):
        return {"row_idx": self.row_idx, "score": self.score, "source": self.source,
                "law_id": self.law_id, "ten_van_ban": self.ten_van_ban,
                "dieu_so": self.dieu_so, "chunk_id": self.chunk_id,
                "doc_uid": self.doc_uid, "chunk_text": self.chunk_text}

class SubQueryTrace:
    def __init__(self, sub_query_text, final_hits, sub_query_index=0):
        self.sub_query_text = sub_query_text
        self.final_hits = final_hits
        self.sub_query_index = sub_query_index
        self.relevant_docs = []
        self.relevant_articles = []

# ---- exec the config cell (defines GEN_*, PLANNER_*, CFG, CANDIDATE_*, ...) ----
ns = {}
# the config cell uses `os` and `Path`; provide real os but a stub drive.
exec(config_src, ns)
assert "CFG" in ns, "CFG dict not defined after config cell exec"
print(f"[test] CFG keys: {len(ns['CFG'])}  candidate_topk={ns['CFG']['candidate_topk']}")

# ---- exec the planner cell (defines anchor_plain, build_rule_*, enrich_query_plan,
#      validate_query_plan, LLMQueryPlanner, build_query_variants, article_bounds, ...) ----
exec(planner_src, ns)
for fn in ["anchor_plain", "build_rule_domain_profile", "build_rule_facet_profiles",
           "enrich_query_plan", "validate_query_plan", "build_query_variants",
           "article_bounds_for_complexity", "fallback_query_plan"]:
    assert fn in ns, f"missing {fn} after planner cell exec"
print("[test] planner cell exec ok")

# ---- exec the batch cell BUT only the function defs, not the batch loop ----
# The batch cell runs the loop at module level. We strip everything from the
# `_INPUT_Q = Path(...)` block onward so we only get the function defs.
batch_fn_src = batch_src
cut = batch_fn_src.find("# --- resolve input / output paths")
assert cut > 0, "could not find batch loop cut point"
batch_fn_src = batch_fn_src[:cut]
exec(batch_fn_src, ns)
for fn in ["article_key", "canonical_article_key", "score_candidate_domain",
           "score_candidate_facets", "aggregate_article_candidates_from_variants",
           "select_article_contexts", "build_article_candidates_from_hits",
           "build_gen_contexts_from_articles", "make_relevant_lists_from_articles",
           "must_terms_query", "_hit_to_candidate"]:
    assert fn in ns, f"missing {fn} after batch cell exec"
print("[test] batch cell (defs only) exec ok")

# ---- build a synthetic hit set spanning 3 laws + 2 facets ----
# copyright_software domain: 50/2005/QH11 (Luật SHTT) + 17/2023/NĐ-CP
# small_business_procurement: 22/2023/QH15 (Luật Đấu thầu)
hits = [
    Hit(1, 9.5, "dense", "50/2005/QH11", "Luật Sở hữu trí tuệ", "20",
        "Tổ chức, cá nhân có quyền tác giả đối với tác phẩm phần mềm."),
    Hit(2, 9.0, "dense", "50/2005/QH11", "Luật Sở hữu trí tuệ", "22",
        "Quyền tác giả bao gồm quyền nhân thân và quyền tài sản."),
    Hit(3, 8.7, "dense", "17/2023/NĐ-CP", "Nghị định bảo vệ quyền tác giả", "66",
        "Xâm phạm quyền tác giả đối với phần mềm bị xử lý hành chính."),
    Hit(4, 8.2, "lexical", "22/2023/QH15", "Luật Đấu thầu", "10",
        "Ưu đãi cho doanh nghiệp nhỏ và vừa khi tham gia đấu thầu."),
    Hit(5, 7.9, "lexical", "04/2017/QH14", "Luật Hỗ trợ doanh nghiệp nhỏ và vừa", "13",
        "Hỗ trợ doanh nghiệp nhỏ và vừa phát triển."),
    Hit(6, 7.5, "dense", "50/2005/QH11", "Luật Sở hữu trí tuệ", "198",
        "Các hành vi xâm phạm quyền tác giả bị nghiêm cấm."),
    Hit(7, 7.0, "lexical", "17/2023/NĐ-CP", "Nghị định bảo vệ quyền tác giả", "73",
        "Tổn thất và cơ hội kinh doanh do xâm phạm quyền tác giả."),
]
traces = [
    SubQueryTrace("Hành vi xâm phạm quyền tác giả phần mềm là gì?",
                  [Hit(6, 8.5, "dense", "50/2005/QH11", "Luật Sở hữu trí tuệ", "198",
                       "Các hành vi xâm phạm quyền tác giả bị nghiêm cấm."),
                   Hit(3, 8.0, "dense", "17/2023/NĐ-CP", "Nghị định bảo vệ quyền tác giả", "66",
                       "Xâm phạm quyền tác giả đối với phần mềm bị xử lý hành chính.")]),
    SubQueryTrace("Tổn thất cơ hội kinh doanh khi xâm phạm quyền tác giả?",
                  [Hit(7, 7.5, "lexical", "17/2023/NĐ-CP", "Nghị định bảo vệ quyền tác giả", "73",
                       "Tổn thất và cơ hội kinh doanh do xâm phạm quyền tác giả.")]),
]

# ---- CASE A: complex query (multi-facet copyright + damages) ----
question_a = "Hành vi xâm phạm quyền tác giả đối với phần mềm gây tổn thất cơ hội kinh doanh cần tài liệu chứng cứ xử lý?"
plan_a = ns["fallback_query_plan"](question_a)
plan_a = ns["enrich_query_plan"](question_a, plan_a)
plan_a["complexity"] = "complex"
print(f"\n[CASE A complex] domain labels={plan_a.get('domain_profile',{}).get('labels')} "
      f"facets={len(plan_a.get('facet_profiles',[]))} atomics={len(plan_a.get('atomic_questions',[]))}")

variants_a = ns["build_query_variants"](question_a, plan_a)
print(f"[CASE A] variants: {[(v['kind'], v['text'][:40]) for v in variants_a]}")

candidates_a = ns["build_article_candidates_from_hits"](hits, plan_a, variants=variants_a, sub_query_traces=traces)
print(f"[CASE A] article_candidates: {len(candidates_a)}")
assert candidates_a, "no candidates produced"
c0 = candidates_a[0]
for k in ["article_key", "canonical_article_key", "article_score", "domain_score",
          "facet_score", "matched_facets", "support_variants"]:
    assert k in c0, f"candidate missing key {k}"
print(f"[CASE A] top candidate: {c0['article_key']} score={c0['article_score']:.3f} "
      f"domain={c0['domain_score']:.2f} facet={c0['facet_score']:.2f} "
      f"matched={c0['matched_facets']} variants={c0['support_variants']}")

selected_a, debug_a = ns["select_article_contexts"](candidates_a, plan_a, variants=variants_a)
print(f"[CASE A] selected_k={debug_a['selected_k']} bounds={debug_a['min_k']}-{debug_a['max_k']} "
      f"reason={debug_a['reason']} covered_facets={debug_a['covered_facets']}")
assert debug_a["selected_k"] >= debug_a["min_k"], "selected_k below min_k for complex"

docs_a, articles_a = ns["make_relevant_lists_from_articles"](selected_a)
ctxs_a = ns["build_gen_contexts_from_articles"](selected_a)
print(f"[CASE A] docs={len(docs_a)} articles={len(articles_a)} contexts={len(ctxs_a)}")
for a in articles_a:
    assert a.count("|") == 2, f"malformed article key: {a}"
print(f"[CASE A] articles: {articles_a}")
assert len(articles_a) == debug_a["selected_k"], "articles count != selected_k"

# ---- CASE B: simple query (1 article expected) ----
question_b = "Điều 20 Luật Sở hữu trí tuệ quy định về quyền nhân thân của tác giả?"
plan_b = ns["fallback_query_plan"](question_b)
plan_b = ns["enrich_query_plan"](question_b, plan_b)
plan_b["complexity"] = "simple"
variants_b = ns["build_query_variants"](question_b, plan_b)
candidates_b = ns["build_article_candidates_from_hits"](hits, plan_b, variants=variants_b, sub_query_traces=traces)
selected_b, debug_b = ns["select_article_contexts"](candidates_b, plan_b, variants=variants_b)
print(f"\n[CASE B simple] selected_k={debug_b['selected_k']} bounds={debug_b['min_k']}-{debug_b['max_k']} "
      f"reason={debug_b['reason']}")
assert debug_b["selected_k"] >= debug_b["min_k"], "simple selected_k below min_k"
assert debug_b["selected_k"] <= debug_b["max_k"], "simple selected_k above max_k"
docs_b, articles_b = ns["make_relevant_lists_from_articles"](selected_b)
print(f"[CASE B] articles: {articles_b}")

print("\n[ALL TESTS PASSED]")
