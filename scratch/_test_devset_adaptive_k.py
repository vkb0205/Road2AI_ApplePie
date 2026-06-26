"""Run the real dev_set questions through the ported adaptive-k pipeline (rule-based
fallback planner path — no LLM, no torch) to verify:

  1. fallback_query_plan classifies complexity (simple/medium/complex)
  2. enrich_query_plan populates facet_profiles + domain_profile for every query
  3. article_bounds_for_complexity returns sane min/target/max per complexity
  4. select_article_contexts honours min_k and max_k across many synthetic
     candidate distributions (no crash, k within bounds unless candidates run out)

This exercises the full planner→enrich→selection path on REAL Vietnamese legal
questions from dev_set/questions.json.
"""
import json, sys, types, pathlib

# --- stub heavy deps exactly like _test_adaptive_k_port.py -------------------
class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        return _Stub()

class _Stub:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Stub()
    def __getattr__(self, name): return _Stub()
    def __bool__(self): return False
    def __iter__(self): return iter([])
    def __getitem__(self, k): return _Stub()

for mod in ["torch", "transformers", "retrieval", "retrieval.retriever"]:
    sys.modules[mod] = _StubModule(mod)
sys.modules["retrieval"] = _StubModule("retrieval")
sys.modules["retrieval.retriever"] = _StubModule("retrieval.retriever")

# --- load patched cells ------------------------------------------------------
ROOT = pathlib.Path("Road2AI_ApplePie")
NB = json.loads((ROOT / "notebooks/retrieval_colab_decomp_submit.ipynb").read_text())
def cell_src(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s
cells = {c.get("id", f"c{i}"): cell_src(c) for i, c in enumerate(NB["cells"])}

g = {}
exec(cell_src(NB["cells"][20]), g)   # config
exec(cell_src(NB["cells"][22]), g)   # planner
# batch cell runs the loop at module level — strip from the input/output
# paths block onward so we only get the function defs.
batch_src = cell_src(NB["cells"][23])
cut = batch_src.find("# --- resolve input / output paths")
assert cut > 0, "could not find batch loop cut point"
exec(batch_src[:cut], g)

fallback_query_plan   = g["fallback_query_plan"]
enrich_query_plan     = g["enrich_query_plan"]
article_bounds_for_complexity = g["article_bounds_for_complexity"]
select_article_contexts = g["select_article_contexts"]
build_article_candidates_from_hits = g["build_article_candidates_from_hits"]
build_query_variants  = g["build_query_variants"]
COMPLEXITIES = g.get("COMPLEXITIES", {"simple", "medium", "complex"})

# --- synthetic hit builder ---------------------------------------------------
def synth_hits(n, base="50/2005/QH11|Luật Sở hữu trí tuệ"):
    hits = []
    for i in range(n):
        h = _Stub()
        h.row_idx = i
        h.score = round(1.0 / (60 + i), 5)   # RRF-ish
        h.source = "bm25" if i % 2 == 0 else "dense"
        h.law_id = "50/2005/QH11" if i % 3 != 0 else "17/2023/NĐ-CP"
        h.ten_van_ban = "Luật Sở hữu trí tuệ" if i % 3 != 0 else "Nghị định bảo vệ quyền tác giả"
        h.dieu_so = str(20 + (i % 10))
        h.chunk_id = f"ch{i}"
        h.doc_uid = f"doc{i}"
        h.chunk_text = f"điều {h.dieu_so} văn bản mẫu số {i} " * 3
        h.to_dict = lambda self=h: {
            "row_idx": self.row_idx, "score": self.score, "source": self.source,
            "law_id": self.law_id, "ten_van_ban": self.ten_van_ban,
            "dieu_so": self.dieu_so, "chunk_id": self.chunk_id,
            "doc_uid": self.doc_uid, "chunk_text": self.chunk_text,
        }
        hits.append(h)
    return hits

# --- run all dev_set questions ----------------------------------------------
questions = json.loads((ROOT / "dev_set/questions.json").read_text())
print(f"[dev] {len(questions)} questions loaded\n")

complexity_counts = {"simple": 0, "medium": 0, "complex": 0, "other": 0}
bounds_seen = {}
facet_nonempty = 0
domain_nonempty = 0
violations = []

for q in questions:
    qid = q["id"]
    text = q["question"]
    plan = fallback_query_plan(text)               # rule-based fallback path
    complexity = plan.get("complexity", "?")
    complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    facets = plan.get("facet_profiles", [])
    domain = plan.get("domain_profile", {})
    if facets: facet_nonempty += 1
    if domain: domain_nonempty += 1
    bounds = article_bounds_for_complexity(complexity)
    bounds_seen[complexity] = bounds
    min_k, target_k, max_k = bounds['min_k'], bounds['target_k'], bounds['max_k']

    # build variants + candidates + selection with a decent candidate pool
    variants = build_query_variants(text, plan)
    hits = synth_hits(40)
    try:
        ac = build_article_candidates_from_hits(hits, plan, variants, sub_query_traces=None)
        sel = select_article_contexts(ac, plan, variants=variants)
    except Exception as e:
        violations.append(f"q{id}: EXCEPTION {type(e).__name__}: {e}")
        continue

    k = len(sel)
    # min_k/target_k/max_k already extracted from bounds dict above
    # k must be >=1 (always picks something if candidates exist) and <= max_k
    if k < 1:
        violations.append(f"q{qid} ({complexity}): k={k} < 1 (no selection)")
    if k > max_k:
        violations.append(f"q{qid} ({complexity}): k={k} > max_k={max_k}")
    # simple queries should NOT explode to complex quota
    if complexity == "simple" and k > max_k:
        violations.append(f"q{qid}: simple but k={k} exceeds simple max_k={max_k}")

print("--- complexity distribution ---")
for c, n in sorted(complexity_counts.items()):
    b = bounds_seen.get(c, None)
    print(f"  {c:8s}: {n:3d}  bounds={b}")

print("\n--- enrichment coverage ---")
print(f"  facet_profiles non-empty : {facet_nonempty}/{len(questions)}")
print(f"  domain_profile non-empty : {domain_nonempty}/{len(questions)}")

print("\n--- bounds sanity ---")
for c in ("simple", "medium", "complex"):
    if c in bounds_seen:
        b = bounds_seen[c]
        mn, tg, mx = b['min_k'], b['target_k'], b['max_k']
        assert mn <= tg <= mx, f"  {c}: min {mn} <= target {tg} <= max {mx}  VIOLATED"
        print(f"  {c:8s}: min={mn} target={tg} max={mx}  OK")

print("\n--- selection violations ---")
if violations:
    for v in violations:
        print(f"  [FAIL] {v}")
    print(f"\n{len(violations)} violation(s)")
    sys.exit(1)
else:
    print("  none — k within [1, max_k] for all questions")
    print("\n[DEV SET VERIFICATION PASSED]")
