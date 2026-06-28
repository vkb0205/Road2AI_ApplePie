"""Unit tests for the query decomposer (pre-retrieval splitting).

Covers:
- clean_sub_query / normalize_sub_queries (salvaged pure functions).
- rule_based_decompose splitting on semicolons + conjunctions.
- Decomposer with and without a mock LLM.
- The "should_decompose=False when <2 valid subs" guard.
- Role inference + SubQuestion / DecompositionResult dataclasses.

No heavy deps — pure Python. The LLM path is tested with a mock callable.
"""

from __future__ import annotations

import json

from retrieval.decomposer import (
    Decomposer,
    DecomposerConfig,
    DecompositionResult,
    SubQuestion,
    clean_sub_query,
    normalize_sub_queries,
    rule_based_decompose,
)


# ============================ clean_sub_query ============================ #


class TestCleanSubQuery:
    def test_basic_clean_adds_terminal_punct(self):
        assert clean_sub_query("  xác định trách nhiệm bồi thường  ") == \
            "xác định trách nhiệm bồi thường."

    def test_collapse_whitespace(self):
        assert clean_sub_query("xác định\n\n\nthiệt hại") == "xác định thiệt hại."

    def test_empty_variants(self):
        assert clean_sub_query("") == ""
        assert clean_sub_query("   ") == ""
        assert clean_sub_query(".") == ""

    def test_preserves_article_citation(self):
        result = clean_sub_query("Điều 15 Bộ luật Dân sự quy định về bồi thường")
        assert "Điều 15" in result


# ============================ normalize_sub_queries ============================ #


class TestNormalizeSubQueries:
    def test_dedup_near_duplicates(self):
        raw = [
            "xác định trách nhiệm bồi thường thiệt hại",
            "xác định trách nhiệm bồi thường thiệt hại",
        ]
        result = normalize_sub_queries(raw, min_chars=5, dedup_threshold=0.85)
        assert len(result) == 1

    def test_discard_short(self):
        raw = ["ngắn", "xác định trách nhiệm bồi thường theo hợp đồng lao động"]
        result = normalize_sub_queries(raw, min_chars=20)
        assert len(result) == 1

    def test_discard_no_legal_terms(self):
        raw = ["hôm nay trời đẹp quá đi thôi", "cần mua sữa cho em bé nhỏ"]
        result = normalize_sub_queries(raw, min_chars=5)
        assert result == []

    def test_truncate_to_max(self):
        raw = [
            "điều kiện hưởng ưu đãi thuế thu nhập doanh nghiệp là gì",
            "thủ tục đăng ký hưởng ưu đãi thuế gồm những bước nào",
            "hồ sơ xin ưu đãi thuế bao gồm những giấy tờ gì",
            "mức ưu đãi thuế thu nhập doanh nghiệp là bao nhiêu phần trăm",
        ]
        result = normalize_sub_queries(raw, max_count=2, min_chars=5)
        assert len(result) == 2


# ============================ rule_based_decompose ============================ #


class TestRuleBasedDecompose:
    def test_semicolon_split(self):
        q = "điều kiện hưởng ưu đãi thuế là gì; thủ tục đăng ký gồm bước nào"
        parts = rule_based_decompose(q)
        assert len(parts) == 2

    def test_conjunction_split(self):
        q = (
            "điều kiện hưởng ưu đãi thuế thu nhập doanh nghiệp đồng thời "
            "thủ tục đăng ký hồ sơ ưu đãi gồm những bước nào"
        )
        parts = rule_based_decompose(q)
        assert len(parts) >= 2

    def test_no_split_single_clause(self):
        q = "điều kiện hưởng ưu đãi thuế là gì"
        parts = rule_based_decompose(q)
        assert len(parts) == 1


# ============================ Decomposer (rules) ============================ #


class TestDecomposerRules:
    def setup_method(self):
        self.decomposer = Decomposer(config=DecomposerConfig(use_llm=False))

    def test_single_intent_no_decompose(self):
        result = self.decomposer.decompose("điều kiện hưởng ưu đãi thuế là gì")
        assert result.should_decompose is False
        assert result.source == "rules"

    def test_multi_intent_decomposes(self):
        q = "điều kiện hưởng ưu đãi thuế là gì; thủ tục đăng ký hồ sơ gồm bước nào"
        result = self.decomposer.decompose(q)
        assert result.should_decompose is True
        assert len(result.sub_questions) == 2

    def test_empty_query(self):
        result = self.decomposer.decompose("")
        assert result.should_decompose is False
        assert result.confidence == 0.0

    def test_roles_inferred(self):
        q = "điều kiện hưởng ưu đãi là gì; thủ tục đăng ký gồm bước nào"
        result = self.decomposer.decompose(q)
        roles = {sq.role for sq in result.sub_questions}
        assert "condition" in roles or "procedure" in roles


# ============================ Decomposer (LLM) ============================ #


class TestDecomposerLLM:
    def test_llm_decomposition_used(self):
        def fake_llm(prompt: str) -> str:
            return json.dumps({
                "should_decompose": True,
                "sub_questions": [
                    {"id": "q1", "text": "Điều kiện hưởng ưu đãi thuế là gì?", "role": "condition"},
                    {"id": "q2", "text": "Thủ tục đăng ký ưu đãi thuế gồm bước nào?", "role": "procedure"},
                ],
            })

        d = Decomposer(llm_call=fake_llm, config=DecomposerConfig(use_llm=True))
        result = d.decompose("Câu hỏi phức tạp về ưu đãi thuế doanh nghiệp.")
        assert result.should_decompose is True
        assert result.source == "llm"
        assert len(result.sub_questions) == 2

    def test_llm_single_sub_demotes_to_no_decompose(self):
        # LLM says decompose but only one valid sub survives → no-decompose.
        def fake_llm(prompt: str) -> str:
            return json.dumps({
                "should_decompose": True,
                "sub_questions": [
                    {"id": "q1", "text": "Điều kiện hưởng ưu đãi thuế là gì?", "role": "condition"},
                ],
            })

        d = Decomposer(llm_call=fake_llm, config=DecomposerConfig(use_llm=True))
        result = d.decompose("Câu hỏi.")
        assert result.should_decompose is False

    def test_llm_failure_falls_back_to_rules(self):
        def broken_llm(prompt: str) -> str:
            raise RuntimeError("down")

        d = Decomposer(llm_call=broken_llm, config=DecomposerConfig(use_llm=True))
        q = "điều kiện hưởng ưu đãi là gì; thủ tục đăng ký gồm bước nào"
        result = d.decompose(q)
        assert result.source == "rules"


# ============================ Dataclasses ============================ #


class TestDataclasses:
    def test_subquestion_invalid_role_coerced(self):
        sq = SubQuestion(id="q1", text="x", role="bogus")
        assert sq.role == "other"

    def test_result_dict_subquestions(self):
        r = DecompositionResult(
            should_decompose=True,
            original_query="q",
            sub_questions=[{"id": "q1", "text": "abc", "role": "condition"}],
        )
        # dict sub-questions are coerced to SubQuestion in __post_init__.
        assert isinstance(r.sub_questions[0], SubQuestion)
        assert r.to_dict()["sub_questions"][0]["role"] == "condition"
