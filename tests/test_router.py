"""Unit tests for the query router (lane classification).

Covers:
- Lane classification for each of the 6 lanes via surface cues.
- Strategy switch derivation (need_graph_expand, should_decompose, depth, strength).
- LLM path with a mock callable + graceful fallback to rules.
- Edge cases: empty query, no-cue short/long queries, JSON extraction.

No heavy deps — pure Python. The LLM path is tested with a mock callable.
"""

from __future__ import annotations

from retrieval.router import (
    LANES,
    Router,
    RouterConfig,
    RoutingDecision,
    _extract_json,
)


# ============================ RoutingDecision ============================ #


class TestRoutingDecision:
    def test_invalid_lane_coerced(self):
        d = RoutingDecision(lane="nonsense")
        assert d.lane == "direct_lookup"

    def test_invalid_depth_strength_coerced(self):
        d = RoutingDecision(retrieval_depth="huge", rerank_strength="max")
        assert d.retrieval_depth == "normal"
        assert d.rerank_strength == "normal"

    def test_confidence_clamped(self):
        assert RoutingDecision(confidence=2.0).confidence == 1.0
        assert RoutingDecision(confidence=-1.0).confidence == 0.0

    def test_to_dict_round_trips_fields(self):
        d = RoutingDecision(lane="cross_doc", need_graph_expand=True)
        out = d.to_dict()
        assert out["lane"] == "cross_doc"
        assert out["need_graph_expand"] is True


# ============================ Rule-based lanes ============================ #


class TestRuleClassification:
    def setup_method(self):
        self.router = Router()  # no LLM → pure rules

    def test_procedure_lane(self):
        d = self.router.route("Thủ tục đăng ký doanh nghiệp gồm những bước nào và hồ sơ ra sao?")
        assert d.lane == "procedure_detail"
        assert d.source == "rules"

    def test_condition_lane(self):
        d = self.router.route("Điều kiện để doanh nghiệp được hưởng ưu đãi thuế là gì?")
        assert d.lane == "condition_requirement"

    def test_sanction_lane(self):
        d = self.router.route("Mức xử phạt vi phạm hành chính về thuế là bao nhiêu?")
        assert d.lane == "sanction"

    def test_scenario_lane(self):
        d = self.router.route(
            "Giả sử một doanh nghiệp nhỏ muốn chuyển đổi loại hình thì cần làm gì?"
        )
        assert d.lane == "scenario"

    def test_cross_doc_lane(self):
        d = self.router.route(
            "Văn bản nào quy định chi tiết về hỗ trợ tư vấn pháp luật cho doanh nghiệp?"
        )
        assert d.lane == "cross_doc"

    def test_direct_lookup_lane(self):
        d = self.router.route("Nội dung Điều 17 quy định gì?")
        assert d.lane == "direct_lookup"

    def test_empty_query(self):
        d = self.router.route("")
        assert d.lane == "direct_lookup"
        assert d.confidence == 0.0

    def test_no_cue_short_query_is_direct(self):
        d = self.router.route("Vốn điều lệ tối thiểu?")
        assert d.lane == "direct_lookup"

    def test_no_cue_long_query_is_cross_doc(self):
        # 30+ words with no strong lane cue → assume multi-hop.
        long_q = " ".join(["abcxyz"] * 35)
        d = self.router.route(long_q)
        assert d.lane == "cross_doc"


# ============================ Strategy switches ============================ #


class TestStrategySwitches:
    def setup_method(self):
        self.router = Router()

    def test_cross_doc_triggers_graph_and_decompose(self):
        d = self.router.route(
            "Văn bản nào quy định chi tiết và hướng dẫn thi hành về hỗ trợ doanh nghiệp?"
        )
        assert d.need_graph_expand is True
        assert d.should_decompose is True
        assert d.retrieval_depth == "wide"

    def test_direct_lookup_narrow_strong(self):
        d = self.router.route("Điều 5 nói gì?")
        assert d.lane == "direct_lookup"
        assert d.retrieval_depth == "narrow"
        assert d.rerank_strength == "strong"
        assert d.need_graph_expand is False

    def test_long_multi_intent_triggers_decompose(self):
        q = (
            "Doanh nghiệp nhỏ và vừa muốn nhận hỗ trợ tư vấn pháp luật thì cần "
            "điều kiện gì và đồng thời hồ sơ thủ tục bao gồm những giấy tờ nào "
            "để được cơ quan có thẩm quyền chấp thuận theo quy định hiện hành?"
        )
        d = self.router.route(q)
        assert d.should_decompose is True


# ============================ LLM path ============================ #


class TestLLMPath:
    def test_llm_lane_used_when_valid(self):
        def fake_llm(prompt: str) -> str:
            return '{"lane": "sanction", "reason": "phạt"}'

        router = Router(llm_call=fake_llm, config=RouterConfig(use_llm=True))
        d = router.route("Câu hỏi về một vấn đề pháp lý nào đó.")
        assert d.lane == "sanction"
        assert d.source == "llm"

    def test_llm_failure_falls_back_to_rules(self):
        def broken_llm(prompt: str) -> str:
            raise RuntimeError("LLM down")

        router = Router(llm_call=broken_llm, config=RouterConfig(use_llm=True))
        d = router.route("Thủ tục đăng ký kinh doanh gồm các bước nào?")
        assert d.source == "rules"
        assert d.lane == "procedure_detail"

    def test_llm_invalid_lane_falls_back(self):
        def bad_lane_llm(prompt: str) -> str:
            return '{"lane": "made_up", "reason": "x"}'

        router = Router(llm_call=bad_lane_llm, config=RouterConfig(use_llm=True))
        d = router.route("Điều kiện hưởng ưu đãi là gì?")
        assert d.source == "rules"

    def test_use_llm_false_ignores_llm(self):
        def fake_llm(prompt: str) -> str:
            return '{"lane": "scenario", "reason": "x"}'

        # use_llm defaults False → LLM never consulted.
        router = Router(llm_call=fake_llm)
        d = router.route("Mức phạt vi phạm là bao nhiêu?")
        assert d.source == "rules"


# ============================ _extract_json ============================ #


class TestExtractJson:
    def test_bare_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_trailing_prose(self):
        assert _extract_json('{"a": 1} and some explanation') == {"a": 1}

    def test_garbage_returns_none(self):
        assert _extract_json("no json here") is None

    def test_empty_returns_none(self):
        assert _extract_json("") is None
