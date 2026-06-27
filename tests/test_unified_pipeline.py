"""Unit tests for the generation modules + the unified pipeline glue.

All CPU-safe: no torch / faiss / transformers. The retriever and generator are
exercised through fakes so the orchestration, fusion, selection wiring and the
submission validator are covered without a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generation.guardrails import (  # noqa: E402
    GuardrailConfig,
    answer_cites_allowed,
    apply_guardrails,
    extract_dieu_citations,
    strip_prompt_echo,
)
from generation.prompt import PromptConfig, build_messages, build_irac_prompt  # noqa: E402
from generation.generator import GenerationConfig, IRACGenerator  # noqa: E402

from retrieval.article_select import ArticleCandidate, SelectConfig  # noqa: E402
from retrieval.doc_anchor import DocAnchoredRetriever, DocAnchorConfig  # noqa: E402
from retrieval.sub_query_router import SubQueryRouter, SubQueryRouterConfig  # noqa: E402
from retrieval.unified_pipeline import (  # noqa: E402
    summarize_timings,
    UnifiedConfig,
    UnifiedPipeline,
    build_query_variants,
    build_record,
    fuse_candidate_pools,
    validate_submission,
)
from retrieval.article_select import (  # noqa: E402
    COMPLEXITY_K_BOUNDS,
    select_config_for_complexity,
)
from retrieval.sub_query_router import infer_complexity  # noqa: E402


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
def test_prompt_includes_question_context_and_allowed_citations():
    ctx = [{"law_id": "01/2021/NĐ-CP", "ten_van_ban": "Nghị định 01", "dieu_so": "Điều 12", "chunk_text": "Hồ sơ ..."}]
    arts = ["01/2021/NĐ-CP|Nghị định 01|Điều 12"]
    msgs = build_messages("Câu hỏi test?", ctx, arts, PromptConfig())
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "Câu hỏi test?" in user
    assert "Điều 12" in user
    assert "IRAC" in user


def test_prompt_caps_context_passages():
    ctx = [{"law_id": f"{i}", "ten_van_ban": "t", "dieu_so": f"Điều {i}", "chunk_text": "x"} for i in range(20)]
    text = build_irac_prompt("q", ctx, [], PromptConfig(max_context_passages=3))
    # Only 3 numbered blocks [1] [2] [3] should appear.
    assert "[3]" in text and "[4]" not in text


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
def test_strip_prompt_echo_removes_assistant_prefix():
    assert strip_prompt_echo("assistant: Trả lời nội dung") == "Trả lời nội dung"


def test_guardrail_backfills_missing_citation_and_disclaimer():
    arts = ["01/2021/NĐ-CP|Nghị định 01|Điều 12"]
    out, report = apply_guardrails("Doanh nghiệp cần nộp hồ sơ.", arts, GuardrailConfig())
    assert "Điều 12" in out
    assert report["backfilled_citation"] is True
    assert "tham khảo" in out.lower()
    assert answer_cites_allowed(out, arts)


def test_guardrail_keeps_existing_citation_no_backfill():
    arts = ["01/2021/NĐ-CP|Nghị định 01|Điều 12"]
    out, report = apply_guardrails(
        "Theo Điều 12, doanh nghiệp cần nộp hồ sơ. Thông tin chỉ mang tính tham khảo, không thay thế tư vấn.",
        arts,
        GuardrailConfig(),
    )
    assert report["backfilled_citation"] is False
    assert report["appended_disclaimer"] is False


def test_guardrail_flags_out_of_list_citation():
    arts = ["01/2021/NĐ-CP|Nghị định 01|Điều 12"]
    out, report = apply_guardrails("Theo Điều 99 thì ...", arts, GuardrailConfig())
    assert report["flagged_out_of_list"] is True


def test_extract_dieu_citations_canonical():
    cites = extract_dieu_citations("Theo Điều 12 và  Điều   12 thì ...")
    assert cites == {"điều 12"}


# --------------------------------------------------------------------------- #
# Generator (fake LLM)
# --------------------------------------------------------------------------- #
def test_generator_retrieval_only_mode_returns_empty():
    gen = IRACGenerator(llm_call=None)
    assert gen.generate("q", [], []) == ""


def test_generator_applies_guardrails_to_llm_output():
    arts = ["38/2015/TT-BTC|Thông tư 38|Điều 1"]
    gen = IRACGenerator(llm_call=lambda messages: "Nội dung trả lời không trích dẫn.")
    out = gen.generate("q", [{"law_id": "38/2015/TT-BTC", "ten_van_ban": "Thông tư 38", "dieu_so": "Điều 1", "chunk_text": "..."}], arts)
    assert "Điều 1" in out
    assert answer_cites_allowed(out, arts)


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def test_fuse_candidate_pools_takes_max_score_per_article():
    pool_a = [ArticleCandidate("L1", "T1", "Điều 5", 0.40)]
    pool_b = [ArticleCandidate("L1", "T1", "Điều 5", 0.90),
              ArticleCandidate("L2", "T2", "Điều 7", 0.30)]
    fused = fuse_candidate_pools([pool_a, pool_b])
    by_dieu = {c.dieu_so: c.score for c in fused}
    assert by_dieu["Điều 5"] == 0.90  # max, not sum
    assert by_dieu["Điều 7"] == 0.30
    assert fused[0].dieu_so == "Điều 5"  # sorted desc


# --------------------------------------------------------------------------- #
# Unified pipeline (fake retriever + fake generator)
# --------------------------------------------------------------------------- #
def _hit(row_idx, law_id, dieu, ten="", **scores):
    h = {"row_idx": row_idx, "law_id": law_id, "ten_van_ban": ten or f"VB {law_id}",
         "dieu_so": dieu, "doc_uid": law_id}
    h.update(scores)
    return h


def _make_retriever(hits):
    def lex(q, k):
        return hits[:k]
    return DocAnchoredRetriever(lexical_search=lex, dense_search=None,
                                fetch_text=None, rerank=None, cfg=DocAnchorConfig())


def test_pipeline_single_query_record_schema():
    hits = [_hit(1, "01/2021/NĐ-CP", "Điều 12", bm25_score=5.0),
            _hit(2, "01/2021/NĐ-CP", "Điều 1", bm25_score=4.0)]
    pipe = UnifiedPipeline(_make_retriever(hits), select_cfg=SelectConfig(max_k=2))
    rec = pipe.answer_record(7, "Hồ sơ đăng ký doanh nghiệp gồm gì?")
    assert rec["id"] == 7
    assert rec["answer"] == ""  # no generator
    assert all("|" in d for d in rec["relevant_docs"])
    assert all(s.count("|") == 2 for s in rec["relevant_articles"])
    validate_submission([rec])  # schema must pass


def test_pipeline_with_generator_populates_grounded_answer():
    hits = [_hit(1, "01/2021/NĐ-CP", "Điều 12", bm25_score=5.0)]
    gen = IRACGenerator(llm_call=lambda m: "Trả lời chung không trích dẫn.")
    pipe = UnifiedPipeline(_make_retriever(hits), select_cfg=SelectConfig(max_k=1), generator=gen)
    rec = pipe.answer_record(1, "câu hỏi")
    assert rec["answer"]  # non-empty
    assert answer_cites_allowed(rec["answer"], rec["relevant_articles"])
    validate_submission([rec], require_answer_citation=True)


def test_pipeline_decomposition_switch_fuses_sub_queries():
    hits = [_hit(1, "L1", "Điều 5", bm25_score=5.0),
            _hit(2, "L2", "Điều 7", bm25_score=4.0)]
    router = SubQueryRouter(llm_call=None, config=SubQueryRouterConfig(use_llm=False))
    pipe = UnifiedPipeline(_make_retriever(hits), select_cfg=SelectConfig(max_k=3),
                           router=router, cfg=UnifiedConfig(use_decomposition=True))
    # A multi-clause legal question the rule-based router will split.
    q = ("Trách nhiệm bồi thường thiệt hại được quy định thế nào; "
         "và hồ sơ đăng ký doanh nghiệp gồm những giấy tờ gì?")
    cands = pipe.retrieve_candidates(q)
    assert cands  # fusion produced a non-empty pool
    rec = pipe.answer_record(1, q)
    validate_submission([rec])


def test_pipeline_llm_decomposition_via_prompt_adapter():
    """The factory wires an LLM router as ``(prompt: str) -> str`` (a single chat
    turn over the shared Qwen). Drive that exact shape with a fake LLM and assert
    decomposition splits + fuses, with the rule-based path as the safety net."""
    hits = [_hit(1, "L1", "Điều 5", bm25_score=5.0),
            _hit(2, "L2", "Điều 7", bm25_score=4.0)]

    def fake_llm_call(prompt: str) -> str:
        # Mirrors build_unified_pipeline's router_llm: prompt-in, JSON-out.
        assert isinstance(prompt, str)
        return ('{"should_decompose": true, "sub_queries": '
                '["xác định trách nhiệm bồi thường thiệt hại", '
                '"xác định quyền sở hữu và nghĩa vụ của các bên"]}')

    router = SubQueryRouter(llm_call=fake_llm_call,
                            config=SubQueryRouterConfig(use_llm=True, max_sub_queries=4))
    pipe = UnifiedPipeline(_make_retriever(hits), select_cfg=SelectConfig(max_k=3),
                           router=router, cfg=UnifiedConfig(use_decomposition=True))
    subs = pipe._sub_queries("một câu hỏi pháp lý phức hợp")
    assert len(subs) == 2
    cands = pipe.retrieve_candidates("một câu hỏi pháp lý phức hợp")
    assert cands
    validate_submission([pipe.answer_record(1, "một câu hỏi pháp lý phức hợp")])


def test_answer_record_timed_reports_stages_and_matches_record():
    hits = [_hit(1, "01/2021/NĐ-CP", "Điều 12", bm25_score=5.0)]
    gen = IRACGenerator(llm_call=lambda m: "Theo Điều 12, hồ sơ gồm...")
    pipe = UnifiedPipeline(_make_retriever(hits), select_cfg=SelectConfig(max_k=1),
                           generator=gen)
    rec, t = pipe.answer_record_timed(7, "câu hỏi")
    # Record is schema-identical to answer_record's output.
    assert rec["id"] == 7 and rec["answer"]
    validate_submission([rec])
    # Timing has all stages; total is their sum; no decomposition here.
    for k in ("decompose", "retrieve", "select", "generate", "total", "n_sub", "decomposed"):
        assert k in t
    assert t["n_sub"] == 1 and t["decomposed"] is False
    assert t["generate"] > 0.0  # the fake LLM call was timed
    assert abs(t["total"] - (t["decompose"] + t["retrieve"] + t["select"] + t["generate"])) < 1e-6


def test_answer_record_timed_counts_sub_queries_when_decomposed():
    hits = [_hit(1, "L1", "Điều 5", bm25_score=5.0)]
    router = SubQueryRouter(llm_call=None, config=SubQueryRouterConfig(use_llm=False))
    pipe = UnifiedPipeline(_make_retriever(hits), select_cfg=SelectConfig(max_k=2),
                           router=router, cfg=UnifiedConfig(use_decomposition=True))
    q = ("xác định trách nhiệm bồi thường theo Điều 15; "
         "xác định thiệt hại theo Điều 20")
    _rec, t = pipe.answer_record_timed(1, q)
    assert t["decomposed"] is True
    assert t["n_sub"] >= 2


def test_summarize_timings_aggregates():
    timings = [
        {"decompose": 1.0, "retrieve": 2.0, "select": 0.5, "generate": 4.0,
         "total": 7.5, "n_sub": 1, "decomposed": False},
        {"decompose": 1.0, "retrieve": 6.0, "select": 0.5, "generate": 4.0,
         "total": 11.5, "n_sub": 3, "decomposed": True},
    ]
    s = summarize_timings(timings)
    assert s["n"] == 2
    assert s["retrieve_mean"] == 4.0
    assert s["retrieve_total"] == 8.0
    assert s["decomposed_count"] == 1
    assert s["decomposed_rate"] == 0.5
    assert s["mean_n_sub"] == 2.0


def test_summarize_timings_empty():
    assert summarize_timings([]) == {"n": 0}


def test_pipeline_llm_decomposition_falls_back_on_bad_json():
    """A garbage LLM response must not break routing: the router falls back to
    the deterministic rule-based split internally."""
    hits = [_hit(1, "L1", "Điều 5", bm25_score=5.0)]

    def bad_llm_call(prompt: str) -> str:
        return "not json at all"

    router = SubQueryRouter(llm_call=bad_llm_call,
                            config=SubQueryRouterConfig(use_llm=True))
    pipe = UnifiedPipeline(_make_retriever(hits), select_cfg=SelectConfig(max_k=3),
                           router=router, cfg=UnifiedConfig(use_decomposition=True))
    # Must still return a usable (>=1) sub-query list and a valid record.
    subs = pipe._sub_queries("xác định trách nhiệm bồi thường theo Điều 15")
    assert len(subs) >= 1
    validate_submission([pipe.answer_record(1, "xác định trách nhiệm bồi thường")])


def test_text_provider_feeds_generation_context():
    hits = [_hit(1, "01/2021/NĐ-CP", "Điều 12", bm25_score=5.0)]
    seen = {}

    def llm(messages):
        # capture the user prompt so we can assert the chunk text reached it
        seen["user"] = messages[1]["content"]
        return "Theo Điều 12 ..."

    def text_provider(law_id, ten, dieu):
        return "NỘI DUNG ĐIỀU 12 ĐẦY ĐỦ"

    pipe = UnifiedPipeline(_make_retriever(hits), select_cfg=SelectConfig(max_k=1),
                           generator=IRACGenerator(llm_call=llm), text_provider=text_provider)
    pipe.answer_record(1, "q")
    assert "NỘI DUNG ĐIỀU 12" in seen["user"]


# --------------------------------------------------------------------------- #
# Submission validator
# --------------------------------------------------------------------------- #
def _good_record(qid=1):
    return {
        "id": qid, "question": "q", "answer": "Theo Điều 12 ...",
        "relevant_docs": ["01/2021/NĐ-CP|Nghị định 01"],
        "relevant_articles": ["01/2021/NĐ-CP|Nghị định 01|Điều 12"],
    }


def test_validate_submission_accepts_good_record():
    assert validate_submission([_good_record()])["schema_ok"]


def test_validate_submission_rejects_wrong_keys():
    bad = _good_record()
    del bad["answer"]
    with pytest.raises(AssertionError):
        validate_submission([bad])


def test_validate_submission_rejects_bad_article_format():
    bad = _good_record()
    bad["relevant_articles"] = ["01/2021/NĐ-CP|Nghị định 01"]  # only one |
    with pytest.raises(AssertionError):
        validate_submission([bad])


def test_validate_submission_checks_id_coverage():
    with pytest.raises(AssertionError):
        validate_submission([_good_record(1)], expected_ids=[1, 2])


def test_validate_submission_requires_answer_citation_when_asked():
    bad = _good_record()
    bad["answer"] = "Không có trích dẫn điều luật nào."
    with pytest.raises(AssertionError):
        validate_submission([bad], require_answer_citation=True)


def test_build_record_helper_shapes_output():
    from retrieval.article_select import select_articles
    cands = [ArticleCandidate("01/2021/NĐ-CP", "Nghị định 01", "Điều 12", 0.9)]
    chosen = select_articles(cands, SelectConfig(max_k=1))
    rec = build_record(3, "q", chosen, answer="Theo Điều 12 ...")
    assert rec["id"] == 3
    assert rec["relevant_articles"] == ["01/2021/NĐ-CP|Nghị định 01|Điều 12"]
    validate_submission([rec])


# --------------------------------------------------------------------------- #
# Combined recall layer: variants, complexity-K, multi-variant, coverage quota
# --------------------------------------------------------------------------- #
def test_build_query_variants_dedupes_and_caps():
    v = build_query_variants(
        "câu gốc",
        sub_queries=["câu gốc", "Sub HAI", "sub ba"],  # 'câu gốc' dup, 'Sub HAI' case dup of none
        facets=["facet một", "sub ba", "facet bốn"],   # 'sub ba' dup of sub-query
        max_variants=4,
    )
    assert v[0] == "câu gốc"               # original always first
    lowered = [s.lower() for s in v]
    assert len(lowered) == len(set(lowered))  # no case-insensitive duplicates
    assert len(v) <= 4                     # hard cap honoured


def test_build_query_variants_handles_no_subs_or_facets():
    assert build_query_variants("chỉ một câu") == ["chỉ một câu"]


def test_infer_complexity_levels():
    assert infer_complexity("Mức phạt vi phạm là bao nhiêu?") == "simple"
    assert infer_complexity("q", ["a", "b", "c"]) == "complex"
    long_cross = (
        "Xác định trách nhiệm bồi thường thiệt hại đồng thời làm rõ nghĩa vụ "
        "của các bên liên quan trong hợp đồng dân sự hiện hành"
    )
    assert infer_complexity(long_cross) == "complex"


def test_select_config_for_complexity_sets_k_bounds():
    base = SelectConfig(min_k=1, max_k=3, rel_margin=0.2)
    simple = select_config_for_complexity("simple", base)
    complex_ = select_config_for_complexity("complex", base)
    assert (simple.min_k, simple.max_k) == COMPLEXITY_K_BOUNDS["simple"]
    assert (complex_.min_k, complex_.max_k) == COMPLEXITY_K_BOUNDS["complex"]
    # other knobs inherited unchanged
    assert simple.rel_margin == base.rel_margin == complex_.rel_margin
    # unknown label falls back to the medium band
    assert (
        select_config_for_complexity("???", base).max_k
        == COMPLEXITY_K_BOUNDS["medium"][1]
    )


class _CountingRetriever:
    """Fake retriever exposing both APIs; counts rerank passes."""

    def __init__(self, multi_cands, single_cands=None):
        self.multi_cands = multi_cands
        self.single_cands = single_cands or multi_cands[:1]
        self.rerank_calls = 0
        self.last_variants = None

    def retrieve_pool(self, q):
        self.rerank_calls += 1
        return list(self.single_cands)

    def retrieve_pool_multi(self, variants, rerank_query=None):
        self.rerank_calls += 1  # ONE rerank per question regardless of N variants
        self.last_variants = list(variants)
        return list(self.multi_cands)


def _planner_router(complexity, atomics, facets):
    import json

    def fake_llm(_prompt):
        return json.dumps(
            {"complexity": complexity, "atomic_questions": atomics, "facets": facets},
            ensure_ascii=False,
        )

    return SubQueryRouter(llm_call=fake_llm, config=SubQueryRouterConfig(use_llm=True))


def test_multi_variant_reranks_once_and_widens_variants():
    cands = [
        ArticleCandidate("L1", "Luật A", "Điều 1", 0.95),
        ArticleCandidate("L2", "Luật B", "Điều 5", 0.80),
        ArticleCandidate("L3", "Luật C", "Điều 9", 0.55),
    ]
    ret = _CountingRetriever(cands)
    router = _planner_router(
        "complex",
        ["xác định trách nhiệm bồi thường thiệt hại",
         "xác định quyền và nghĩa vụ của các bên trong hợp đồng"],
        ["mức bồi thường", "căn cứ xác định thiệt hại"],
    )
    pipe = UnifiedPipeline(
        ret, select_cfg=SelectConfig(max_k=10), router=router,
        cfg=UnifiedConfig(
            use_decomposition=True, use_multi_variant=True,
            use_complexity_k=True, coverage_quota=True, max_variants=5,
        ),
    )
    chosen = pipe.select("một câu hỏi pháp lý phức tạp")
    assert ret.rerank_calls == 1                      # the single-rerank invariant
    assert ret.last_variants[0] == "một câu hỏi pháp lý phức tạp"  # original first, verbatim
    assert len(ret.last_variants) > 1                 # widened by atomic + facet
    assert len(chosen) >= 2                            # complexity band allowed >1


def test_coverage_quota_floors_k_at_subquery_count():
    cands = [
        ArticleCandidate("L1", "Luật A", "Điều 1", 0.95),
        ArticleCandidate("L2", "Luật B", "Điều 5", 0.30),  # far below top → margin would drop it
    ]
    ret = _CountingRetriever(cands)
    router = _planner_router(
        "complex",
        ["xác định trách nhiệm bồi thường thiệt hại",
         "xác định quyền và nghĩa vụ của các bên trong hợp đồng"],
        [],
    )
    # With coverage_quota the 2 sub-queries floor min_k=2, so the low-score
    # second article is still admitted despite the margin.
    pipe = UnifiedPipeline(
        ret, select_cfg=SelectConfig(max_k=10, rel_margin=0.1), router=router,
        cfg=UnifiedConfig(
            use_decomposition=True, use_multi_variant=True,
            use_complexity_k=True, coverage_quota=True, max_variants=5,
        ),
    )
    chosen = pipe.select("câu hỏi phức tạp")
    assert len(chosen) >= 2

    # Without the quota the margin alone would keep only the top article.
    ret2 = _CountingRetriever(cands)
    router2 = _planner_router(
        "complex",
        ["xác định trách nhiệm bồi thường thiệt hại",
         "xác định quyền và nghĩa vụ của các bên trong hợp đồng"],
        [],
    )
    pipe2 = UnifiedPipeline(
        ret2, select_cfg=SelectConfig(max_k=10, min_k=1, rel_margin=0.1),
        router=router2,
        cfg=UnifiedConfig(
            use_decomposition=True, use_multi_variant=True,
            use_complexity_k=False, coverage_quota=False, max_variants=5,
        ),
    )
    assert len(pipe2.select("câu hỏi phức tạp")) == 1


def test_complexity_k_caps_simple_query_articles():
    # Three strong, near-tied candidates; a simple query should still cap small.
    cands = [
        ArticleCandidate("L1", "Luật A", "Điều 1", 0.95),
        ArticleCandidate("L2", "Luật B", "Điều 5", 0.93),
        ArticleCandidate("L3", "Luật C", "Điều 9", 0.92),
    ]
    ret = _CountingRetriever(cands)
    router = _planner_router("simple", [], [])  # no decomposition
    pipe = UnifiedPipeline(
        ret, select_cfg=SelectConfig(max_k=10, rel_margin=0.5), router=router,
        cfg=UnifiedConfig(
            use_decomposition=True, use_multi_variant=True,
            use_complexity_k=True, coverage_quota=True, max_variants=5,
        ),
    )
    chosen = pipe.select("Mức phạt là bao nhiêu?")
    assert len(chosen) <= COMPLEXITY_K_BOUNDS["simple"][1]  # simple band caps K


def test_multi_variant_disabled_preserves_single_rerank_path():
    cands = [ArticleCandidate("L1", "Luật A", "Điều 1", 0.9)]
    ret = _CountingRetriever(cands)
    pipe = UnifiedPipeline(
        ret, select_cfg=SelectConfig(max_k=3),
        cfg=UnifiedConfig(use_multi_variant=False),  # legacy path
    )
    pipe.select("câu hỏi đơn")
    assert ret.rerank_calls == 1
    assert ret.last_variants is None  # multi path never touched


def test_timing_reports_variants_and_multi_flag():
    cands = [
        ArticleCandidate("L1", "Luật A", "Điều 1", 0.95),
        ArticleCandidate("L2", "Luật B", "Điều 5", 0.80),
    ]
    ret = _CountingRetriever(cands)
    router = _planner_router(
        "complex",
        ["xác định trách nhiệm bồi thường thiệt hại",
         "xác định quyền và nghĩa vụ của các bên trong hợp đồng"],
        ["mức bồi thường"],
    )
    pipe = UnifiedPipeline(
        ret, select_cfg=SelectConfig(max_k=10), router=router,
        cfg=UnifiedConfig(
            use_decomposition=True, use_multi_variant=True,
            use_complexity_k=True, coverage_quota=True, max_variants=5,
        ),
    )
    _rec, t = pipe.answer_record_timed(1, "câu hỏi phức tạp")
    assert t["multi_variant"] is True
    assert t["complexity"] == "complex"
    assert t["n_variants"] >= t["n_sub"] >= 2
    summ = summarize_timings([t])
    assert summ["multi_variant_rate"] == 1.0
    assert summ["mean_n_variants"] >= 2
