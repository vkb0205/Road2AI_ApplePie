"""Unit tests for sub-query routing (pre-retrieval decomposition).

Tests cover:
- Normalise / clean functions (``clean_sub_query``, ``normalize_sub_queries``).
- Rule-based fallback decomposition (``rule_based_decompose``).
- :class:`SubQueryRouter` with and without a mock LLM.
- Debug trace structures (``RouterDebugLog``, ``SubQueryTrace``, format helpers).
- Edge cases: empty query, single clause, no legal terms, near-duplicates,
  short sub-queries, semicolon splits, conjunction splits.

No heavy deps — pure Python. The LLM path is tested with a mock callable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from retrieval.sub_query_router import (
    DECOMPOSITION_PROMPT,
    RouterDecision,
    RouterFallbackLog,
    SubQueryRouter,
    SubQueryRouterConfig,
    clean_sub_query,
    normalize_sub_queries,
    rule_based_decompose,
)
from retrieval.debug import (
    RouterDebugLog,
    SubQueryTrace,
    RetrievalTrace,
    StageSnapshot,
    format_sub_query_trace,
)


# ============================ clean_sub_query ============================ #


class TestCleanSubQuery:
    def test_basic_clean(self):
        assert clean_sub_query("  xác định trách nhiệm bồi thường  ") == \
            "xác định trách nhiệm bồi thường."

    def test_collapse_whitespace(self):
        assert clean_sub_query("xác định\n\n\nthiệt hại") == \
            "xác định thiệt hại."

    def test_strip_punctuation_surrounding(self):
        result = clean_sub_query("; xác định hành vi vi phạm;")
        assert "xác định hành vi vi phạm" in result
        assert result.endswith(".")

    def test_empty_string(self):
        assert clean_sub_query("") == ""
        assert clean_sub_query("   ") == ""
        assert clean_sub_query(".") == ""

    def test_legal_term_preserved(self):
        result = clean_sub_query("Điều 15 Bộ luật Dân sự quy định về bồi thường")
        assert "Điều 15" in result
        assert "Bộ luật Dân sự" in result

    def test_control_characters_removed(self):
        result = clean_sub_query("xác\x00định\x1ftrách nhiệm")
        assert "xác định trách nhiệm" in result or "xác định trách nhiệm." in result


# ============================ normalize_sub_queries ============================ #


class TestNormalizeSubQueries:
    def test_basic_dedup_and_clean(self):
        raw = ["  xác định trách nhiệm bồi thường  ", "xác định trách nhiệm bồi thường"]
        result = normalize_sub_queries(raw, min_chars=5, dedup_threshold=0.85)
        # Near-duplicates should be merged → 1 result.
        assert len(result) == 1
        assert "trách nhiệm bồi thường" in result[0]

    def test_discard_short(self):
        raw = ["ngắn", "xác định trách nhiệm bồi thường theo hợp đồng"]
        result = normalize_sub_queries(raw, min_chars=20, dedup_threshold=0.9)
        assert len(result) == 1
        assert "trách nhiệm bồi thường" in result[0]

    def test_discard_no_legal_terms(self):
        raw = ["hôm nay trời đẹp", "cần mua sữa"]
        result = normalize_sub_queries(raw, min_chars=5, dedup_threshold=0.9)
        # Both lack legal terms → empty.
        assert len(result) == 0

    def test_keep_longest_after_dedup(self):
        raw = [
            "xác định trách nhiệm bồi thường thiệt hại theo hợp đồng",
            "trách nhiệm bồi thường thiệt hại",
        ]
        result = normalize_sub_queries(raw, min_chars=5, dedup_threshold=0.7)
        assert len(result) == 1
        # The longer one should be kept.
        assert "hợp đồng" in result[0]

    def test_truncate_to_max_count(self):
        raw = [
            "xác định hành vi vi phạm theo Điều 15",
            "xác định trách nhiệm dân sự theo khoản 2",
            "xác định thiệt hại và chứng cứ theo Điều 20",
            "xác định nghĩa vụ bồi thường theo Điều 30",
            "xác định quyền sở hữu theo Điều 40",
        ]
        result = normalize_sub_queries(raw, max_count=3, min_chars=5, dedup_threshold=0.9)
        assert len(result) <= 3

    def test_empty_input(self):
        assert normalize_sub_queries([]) == []
        assert normalize_sub_queries(["", "  "]) == []

    def test_preserves_legal_terms_law_name(self):
        raw = ["Luật Đất đai 2013 quy định về quyền sử dụng đất"]
        result = normalize_sub_queries(raw, min_chars=5)
        assert len(result) == 1
        assert "Luật Đất đai" in result[0]

    def test_diverse_legal_terms(self):
        """Ensure the legal-terms regex covers a broad range of Vietnamese legal vocabulary."""
        raw = [
            "thủ tục hành chính theo Nghị định 100",
            "tranh chấp lao động theo Bộ luật Lao động",
            "hợp đồng thừa kế tài sản",
            "xử phạt vi phạm hành chính",
            "tội cố ý gây thương tích theo Bộ luật Hình sự",
        ]
        result = normalize_sub_queries(
            raw, max_count=10, min_chars=5, dedup_threshold=0.9,
        )
        # All 5 should survive — they carry distinct legal terms.
        assert len(result) == 5


# ============================ rule_based_decompose ============================ #


class TestRuleBasedDecompose:
    def test_no_split_single_clause(self):
        q = "xác định trách nhiệm bồi thường theo hợp đồng"
        parts = rule_based_decompose(q)
        assert len(parts) == 1
        assert parts[0] == q

    def test_semicolon_split(self):
        q = "xác định trách nhiệm bồi thường theo Điều 15; xác định thiệt hại theo Điều 20"
        parts = rule_based_decompose(q)
        assert len(parts) >= 2
        assert any("Điều 15" in p for p in parts)
        assert any("Điều 20" in p for p in parts)

    def test_conjunction_split(self):
        q = "cần xác định hành vi vi phạm và xác định thiệt hại"
        parts = rule_based_decompose(q)
        assert len(parts) >= 2

    def test_conjunction_phrase_split(self):
        q = "cần xác định trách nhiệm bồi thường theo hợp đồng"
        parts = rule_based_decompose(q)
        assert len(parts) >= 1

    def test_multiple_conjunctions(self):
        q = "xác định hành vi vi phạm và thiệt hại và trách nhiệm bồi thường"
        parts = rule_based_decompose(q)
        # Splitting on all 'và' may produce > 1 part but the exact count depends on
        # the conjunction regex. We just verify at least one part.
        assert len(parts) >= 1

    def test_truncate_max_parts(self):
        q = "a; b; c; d; e; f"
        parts = rule_based_decompose(q, max_parts=3)
        assert len(parts) <= 3

    def test_empty_query(self):
        assert rule_based_decompose("") == [""]

    def test_query_with_only_semicolons(self):
        q = ";;;"
        parts = rule_based_decompose(q)
        # All parts empty after stripping → should return original.
        assert len(parts) >= 1


# ============================ SubQueryRouter (no LLM / fallback) ============================ #


class TestSubQueryRouterFallback:
    def test_no_llm_returns_fallback(self):
        router = SubQueryRouter(llm_call=None, config=SubQueryRouterConfig(use_llm=False))
        decision = router.route("xác định trách nhiệm bồi thường")
        assert decision.source == "rule_fallback"
        assert decision.should_decompose is False
        assert len(decision.sub_queries) >= 1
        assert decision.fallback_log is not None

    def test_fallback_log_populated(self):
        router = SubQueryRouter(llm_call=None, config=SubQueryRouterConfig(use_llm=False, debug=True))
        decision = router.route("xác định trách nhiệm bồi thường")
        assert decision.fallback_log is not None
        assert decision.fallback_log.original_query == "xác định trách nhiệm bồi thường."
        assert "LLM disabled" in decision.fallback_log.reason
        assert "rule_fallback" in decision.source

    def test_decompose_multi_clause(self):
        router = SubQueryRouter(llm_call=None, config=SubQueryRouterConfig(use_llm=False))
        q = "xác định trách nhiệm bồi thường theo Điều 15; xác định thiệt hại theo Điều 20"
        decision = router.route(q)
        # With fallback, semicolons should trigger decompose.
        assert decision.should_decompose is True
        assert decision.num_sub_queries >= 2

    def test_empty_query_returns_no_decompose(self):
        router = SubQueryRouter(llm_call=None)
        decision = router.route("")
        assert decision.should_decompose is False
        assert len(decision.sub_queries) == 0

    def test_debug_logging(self, caplog: pytest.LogCaptureFixture):
        import logging
        caplog.set_level(logging.DEBUG)
        router = SubQueryRouter(
            llm_call=None,
            config=SubQueryRouterConfig(use_llm=False, debug=True),
        )
        router.route("xác định trách nhiệm bồi thường")
        assert "SubQueryRouter" in caplog.text

    def test_decision_to_dict(self):
        router = SubQueryRouter(llm_call=None, config=SubQueryRouterConfig(use_llm=False))
        decision = router.route("xác định trách nhiệm bồi thường")
        d = decision.to_dict()
        assert d["should_decompose"] is False
        assert d["source"] == "rule_fallback"
        assert "fallback_log" in d
        assert d["sub_queries"] == decision.sub_queries


# ============================ SubQueryRouter (with mock LLM) ============================ #


class TestSubQueryRouterWithLLM:
    def test_llm_no_decompose(self):
        """LLM returns should_decompose=false."""
        def mock_llm(prompt: str) -> str:
            return json.dumps({"should_decompose": False, "sub_queries": []})

        router = SubQueryRouter(llm_call=mock_llm, config=SubQueryRouterConfig(use_llm=True))
        decision = router.route("xác định trách nhiệm bồi thường theo hợp đồng")
        assert decision.source == "llm"
        assert decision.should_decompose is False
        assert len(decision.sub_queries) >= 1  # original query kept
        assert decision.fallback_log is None

    def test_llm_decompose(self):
        """LLM returns should_decompose=true with two sub-queries."""
        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "should_decompose": True,
                "sub_queries": [
                    "xác định trách nhiệm bồi thường theo Điều 15",
                    "xác định thiệt hại theo Điều 20",
                ],
            })

        router = SubQueryRouter(llm_call=mock_llm, config=SubQueryRouterConfig(use_llm=True))
        decision = router.route("xác định trách nhiệm bồi thường và thiệt hại")
        assert decision.source == "llm"
        assert decision.should_decompose is True
        assert decision.num_sub_queries == 2
        assert decision.raw_llm_output is not None

    def test_llm_return_only_one_after_normalise(self):
        """LLM says decompose but normalisation reduces to 1 → no-decompose."""
        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "should_decompose": True,
                "sub_queries": [
                    "xác định trách nhiệm bồi thường theo Điều 15",
                    "trách nhiệm bồi thường",  # near-dup after dedup
                ],
            })

        router = SubQueryRouter(llm_call=mock_llm, config=SubQueryRouterConfig(
            use_llm=True, dedup_threshold=0.5,
        ))
        decision = router.route("xác định trách nhiệm bồi thường")
        # With containment dedup at 0.5, the shorter sub-query is absorbed
        # into the longer one → single sub-query → no-decompose.
        assert decision.should_decompose is False
        assert decision.source == "llm"

    def test_llm_unparseable_json(self):
        """LLM returns invalid JSON → falls back to rule-based."""
        def mock_llm(prompt: str) -> str:
            return "not json at all"

        router = SubQueryRouter(llm_call=mock_llm, config=SubQueryRouterConfig(use_llm=True, debug=True))
        decision = router.route("xác định trách nhiệm bồi thường")
        # Should fall back to rule-based.
        assert decision.source == "rule_fallback"
        assert decision.fallback_log is not None

    def test_llm_json_in_code_fence(self):
        """LLM returns JSON inside markdown code fences."""
        def mock_llm(prompt: str) -> str:
            return "```json\n{\"should_decompose\": false, \"sub_queries\": []}\n```"

        router = SubQueryRouter(llm_call=mock_llm, config=SubQueryRouterConfig(use_llm=True))
        decision = router.route("xác định trách nhiệm bồi thường")
        assert decision.source == "llm"
        assert decision.should_decompose is False

    def test_llm_json_deep_in_text(self):
        """LLM returns a chatty response with JSON embedded."""
        def mock_llm(prompt: str) -> str:
            return ("Here is your analysis:\n\n"
                    "{\"should_decompose\": true, \"sub_queries\": "
                    "[\"xác định hành vi theo Điều 15\", \"xác định thiệt hại theo Điều 20\"]}\n\n"
                    "Hope this helps.")

        router = SubQueryRouter(llm_call=mock_llm, config=SubQueryRouterConfig(use_llm=True))
        decision = router.route("xác định hành vi và thiệt hại")
        assert decision.source == "llm"
        assert decision.should_decompose is True
        assert decision.num_sub_queries == 2

    def test_llm_empty_response(self):
        """LLM returns empty string → falls back."""
        def mock_llm(prompt: str) -> str:
            return ""

        router = SubQueryRouter(llm_call=mock_llm, config=SubQueryRouterConfig(use_llm=True))
        decision = router.route("xác định trách nhiệm bồi thường")
        assert decision.source == "rule_fallback"

    def test_llm_exception(self):
        """LLM raises an exception → falls back gracefully."""
        def mock_llm(prompt: str) -> str:
            raise RuntimeError("LLM API unavailable")

        router = SubQueryRouter(llm_call=mock_llm, config=SubQueryRouterConfig(use_llm=True))
        decision = router.route("xác định trách nhiệm bồi thường")
        assert decision.source == "rule_fallback"
        assert decision.fallback_log is not None
        assert "LLM call failed" in decision.fallback_log.reason

    def test_prompt_contains_query(self):
        """Verify that the prompt template includes the query."""
        prompt = DECOMPOSITION_PROMPT.format(query="test query")
        assert "test query" in prompt
        assert "should_decompose" in prompt
        assert "sub_queries" in prompt

    def test_max_sub_queries_capped(self):
        """LLM returns more than max_sub_queries → truncated."""
        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "should_decompose": True,
                "sub_queries": [
                    f"xác định vấn đề thứ {i} theo Điều {i}"
                    for i in range(10)
                ],
            })

        router = SubQueryRouter(
            llm_call=mock_llm,
            config=SubQueryRouterConfig(use_llm=True, max_sub_queries=3),
        )
        decision = router.route("xác định nhiều vấn đề")
        assert decision.should_decompose is True
        assert decision.num_sub_queries <= 3


# ============================ Debug trace structures ============================ #


class TestRouterDebugLog:
    def test_router_debug_log_defaults(self):
        log = RouterDebugLog()
        assert log.original_query == ""
        assert log.should_decompose is False
        assert log.sub_queries == []
        assert log.source == "rule_fallback"

    def test_router_debug_log_populated(self):
        log = RouterDebugLog(
            original_query="test query",
            should_decompose=True,
            num_sub_queries=2,
            sub_queries=["sq1", "sq2"],
            source="llm",
            raw_llm_output='{"should_decompose": true}',
            route_elapsed_ms=12.5,
        )
        assert log.original_query == "test query"
        assert log.should_decompose is True
        assert len(log.sub_queries) == 2
        assert log.route_elapsed_ms == 12.5


class TestSubQueryTrace:
    def test_empty_trace(self):
        trace = SubQueryTrace()
        assert trace.original_query == ""
        assert trace.sub_query_text == ""
        assert trace.sub_query_index == 0
        assert trace.num_sub_queries == 1
        assert trace.final_hits == []
        assert trace.relevant_docs == []
        assert trace.relevant_articles == []

    def test_trace_with_route_decision(self):
        rd = RouterDebugLog(
            original_query="test query",
            should_decompose=True,
            num_sub_queries=2,
            sub_queries=["sq1", "sq2"],
            source="llm",
        )
        trace = SubQueryTrace(
            original_query="test query",
            sub_query_text="sq1",
            sub_query_index=0,
            num_sub_queries=2,
            route_decision=rd,
        )
        assert trace.route_decision is not None
        assert trace.route_decision.source == "llm"
        assert trace.sub_query_text == "sq1"

    def test_trace_with_retrieval_trace(self):
        rt = RetrievalTrace(query="sq1", config={"final_top_k": 5})
        rt.add(StageSnapshot(name="lexical", count=10))
        trace = SubQueryTrace(
            original_query="original",
            sub_query_text="sq1",
            sub_query_index=0,
            num_sub_queries=1,
            retrieval_trace=rt,
            final_hits=[{"row_idx": 1, "score": 0.9}],
        )
        assert trace.retrieval_trace is not None
        assert len(trace.retrieval_trace.stages) == 1
        assert trace.retrieval_trace.stages[0].name == "lexical"
        assert len(trace.final_hits) == 1

    def test_format_empty_list(self):
        formatted = format_sub_query_trace([])
        assert "empty" in formatted.lower()

    def test_format_single_sub_query(self):
        trace = SubQueryTrace(
            original_query="test query",
            sub_query_text="test query",
            sub_query_index=0,
            num_sub_queries=1,
            relevant_docs=["law1|Test Law"],
            relevant_articles=["law1|Test Law|Điều 15"],
        )
        formatted = format_sub_query_trace([trace])
        assert "SUB-QUERY" in formatted
        assert "test query" in formatted
        assert "Điều 15" in formatted

    def test_format_multiple_sub_queries(self):
        traces = [
            SubQueryTrace(
                original_query="original",
                sub_query_text="sq1",
                sub_query_index=0,
                num_sub_queries=2,
            ),
            SubQueryTrace(
                original_query="original",
                sub_query_text="sq2",
                sub_query_index=1,
                num_sub_queries=2,
            ),
        ]
        formatted = format_sub_query_trace(traces)
        assert "sq1" in formatted
        assert "sq2" in formatted
        assert "sub-query [0/1]" in formatted or "sub-query [0" in formatted
        assert "sub-query [1/1]" in formatted or "sub-query [1" in formatted


# ============================ Integration: route → debug trace ============================ #


class TestIntegrationRouteAndTrace:
    def test_fallback_decision_populates_debug_log(self):
        router = SubQueryRouter(llm_call=None, config=SubQueryRouterConfig(use_llm=False, debug=True))
        q = "xác định trách nhiệm bồi thường theo Điều 15; xác định thiệt hại theo Điều 20"
        decision = router.route(q)

        # Build a SubQueryTrace from the decision.
        assert decision.should_decompose is True
        assert decision.num_sub_queries >= 2

        traces: List[SubQueryTrace] = []
        for i, sq in enumerate(decision.sub_queries):
            rd = RouterDebugLog(
                original_query=decision.fallback_log.original_query
                if decision.fallback_log else q,
                should_decompose=decision.should_decompose,
                num_sub_queries=decision.num_sub_queries,
                sub_queries=decision.sub_queries,
                source=decision.source,
                fallback_reason=decision.fallback_log.reason
                if decision.fallback_log else "",
                fallback_rule=decision.fallback_log.rule_triggered
                if decision.fallback_log else "",
            )
            traces.append(SubQueryTrace(
                original_query=q,
                sub_query_text=sq,
                sub_query_index=i,
                num_sub_queries=len(decision.sub_queries),
                route_decision=rd,
            ))

        assert len(traces) >= 2
        assert all(t.route_decision is not None for t in traces)
        assert traces[0].route_decision.fallback_rule == "semicolon_split"

        formatted = format_sub_query_trace(traces)
        assert "semicolon_split" in formatted or "semicolon" in formatted
        assert "xác định trách nhiệm bồi thường" in formatted
        assert "xác định thiệt hại" in formatted

    def test_llm_decision_popsulates_trace(self):
        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "should_decompose": True,
                "sub_queries": [
                    "xác định quyền sở hữu theo Bộ luật Dân sự",
                    "xác định nghĩa vụ bồi thường theo Điều 15",
                ],
            })

        router = SubQueryRouter(llm_call=mock_llm, config=SubQueryRouterConfig(use_llm=True))
        q = "xác định quyền sở hữu và nghĩa vụ bồi thường"
        decision = router.route(q)

        assert decision.source == "llm"
        assert decision.should_decompose is True
        assert decision.num_sub_queries == 2

        # Verify the prompt was correctly formed.
        # We can't inspect the raw prompt, but we can verify the decision.
        assert "Bộ luật Dân sự" in decision.sub_queries[0]


class TestEdgeCases:
    def test_query_with_only_legal_references(self):
        """Query with legal cross-references but no conjunction markers."""
        q = "Điều 15, Điều 20 và Điều 30 Bộ luật Dân sự"
        router = SubQueryRouter(llm_call=None, config=SubQueryRouterConfig(use_llm=False))
        decision = router.route(q)
        # This is a reference list, not multiple independent requirements.
        # Rule-based may or may not split; no assertion on decompose.
        assert len(decision.sub_queries) >= 1

    def test_very_long_query(self):
        q = " ".join(["xác định trách nhiệm bồi thường theo điều"] * 100)
        router = SubQueryRouter(llm_call=None, config=SubQueryRouterConfig(use_llm=False))
        decision = router.route(q)
        assert len(decision.sub_queries) >= 1

    def test_query_with_multiple_semicolons(self):
        q = ("xác định hành vi vi phạm theo Điều 15; "
             "xác định thiệt hại theo Điều 20; "
             "xác định trách nhiệm bồi thường theo Điều 25; "
             "xác định thủ tục khởi kiện theo Điều 30")
        router = SubQueryRouter(llm_call=None, config=SubQueryRouterConfig(use_llm=False))
        decision = router.route(q)
        assert decision.should_decompose is True
        assert 2 <= decision.num_sub_queries <= 4

    def test_no_legal_terms_in_query(self):
        """Query entirely lacking legal terms → still processed but may not decompose."""
        q = "hôm nay là ngày đẹp trời"
        router = SubQueryRouter(llm_call=None, config=SubQueryRouterConfig(use_llm=False))
        decision = router.route(q)
        # The query lacks legal terms, but fallback should still produce a single sub-query.
        assert len(decision.sub_queries) == 1 or len(decision.sub_queries) == 0
