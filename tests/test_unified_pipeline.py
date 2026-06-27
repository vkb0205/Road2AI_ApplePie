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
    UnifiedConfig,
    UnifiedPipeline,
    build_record,
    fuse_candidate_pools,
    validate_submission,
)


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
