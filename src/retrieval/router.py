"""Query router for SME Legal QA retrieval (lane classification).

The router is the first stage of the simplified 7-stage pipeline
(``router.md``). It does **not** retrieve or answer — it only classifies the
query into one retrieval *lane* and emits the strategy switches the downstream
stages consume:

    query → Router.route(query) → RoutingDecision
          → (decompose? retrieve → rerank → graph-expand → select → answer)

Design goals
------------
* **Lane-first, single purpose.** The router picks one of six lanes and sets
  three coarse strategy switches (``need_graph_expand``, ``should_decompose``,
  ``retrieval_depth``, ``rerank_strength``). It owns no retrieval logic and no
  decomposition logic (that lives in :mod:`retrieval.decomposer`).
* **Rule-based with optional LLM.** A deterministic, dependency-free rule
  classifier always works (CPU baseline). An injected ``llm_call`` callable can
  override the lane when available; any failure degrades silently to rules.
* **Pure Python, zero heavy deps.** Only stdlib (``json``, ``re``,
  ``logging``, ``dataclasses``). The LLM client is injected by the caller.

Lanes (``router.md`` §4)
------------------------
``direct_lookup`` · ``procedure_detail`` · ``condition_requirement`` ·
``sanction`` · ``cross_doc`` · ``scenario``
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "LANES",
    "RETRIEVAL_DEPTHS",
    "RERANK_STRENGTHS",
    "RoutingDecision",
    "RouterConfig",
    "Router",
    "ROUTER_PROMPT",
]

# Valid lane labels (router.md §4). Order is significant only for docs.
LANES = (
    "direct_lookup",
    "procedure_detail",
    "condition_requirement",
    "sanction",
    "cross_doc",
    "scenario",
)
RETRIEVAL_DEPTHS = ("narrow", "normal", "wide")
RERANK_STRENGTHS = ("light", "normal", "strong")

# --------------------------------------------------------------------------- #
# Lane-detection patterns (Vietnamese legal surface cues)
# --------------------------------------------------------------------------- #
# Each lane has a keyword family. The classifier scores every lane by counting
# matched cues, then applies tie-breaks. These are *surface* cues only — no
# law-id / domain tables (the generalisation risk flagged in the legacy code).

_PROCEDURE_RE = re.compile(
    r"(thủ\s*tục|hồ\s*sơ|trình\s*tự|đăng\s*ký|cấp\s*(?:giấy|phép)|"
    r"các\s*bước|quy\s*trình|nộp\s*(?:đơn|hồ\s*sơ)|thành\s*phần\s*hồ\s*sơ|"
    r"gồm\s*những\s*(?:gì|bước)|cần\s*chuẩn\s*bị)",
    re.IGNORECASE,
)
_CONDITION_RE = re.compile(
    r"(điều\s*kiện|yêu\s*cầu|tiêu\s*chí|tiêu\s*chuẩn|được\s*phép|"
    r"có\s*được|đủ\s*điều\s*kiện|trường\s*hợp\s*nào|khi\s*nào\s*(?:được|phải)|"
    r"đối\s*tượng\s*(?:nào|áp\s*dụng))",
    re.IGNORECASE,
)
_SANCTION_RE = re.compile(
    r"(xử\s*phạt|vi\s*phạm|chế\s*tài|mức\s*phạt|phạt\s*tiền|"
    r"xử\s*lý\s*vi\s*phạm|hình\s*thức\s*xử\s*phạt|tịch\s*thu|"
    r"đình\s*chỉ|thu\s*hồi\s*giấy)",
    re.IGNORECASE,
)
_SCENARIO_RE = re.compile(
    r"(giả\s*sử|trong\s*trường\s*hợp|nếu\s+(?:một|công\s*ty|doanh\s*nghiệp)|"
    r"một\s*doanh\s*nghiệp\s+\S+\s+(?:muốn|cần|bị|đã)|"
    r"giả\s*định|tình\s*huống|ví\s*dụ\s*(?:một|công\s*ty))",
    re.IGNORECASE,
)
# Direct single-article lookup: explicit citation of an article / clause.
_DIRECT_RE = re.compile(
    r"(điều\s*\d+|khoản\s*\d+|theo\s*điều|quy\s*định\s*tại\s*điều|"
    r"nội\s*dung\s*điều\s*\d+)",
    re.IGNORECASE,
)
# Cross-document / multi-hop cues: references to "another / detailing /
# replacing" document, or explicit multi-intent connectors.
_CROSS_DOC_RE = re.compile(
    r"(văn\s*bản\s*nào|nghị\s*định\s*nào|thông\s*tư\s*nào|luật\s*nào|"
    r"quy\s*định\s*chi\s*tiết|hướng\s*dẫn\s*thi\s*hành|sửa\s*đổi|"
    r"thay\s*thế|bổ\s*sung|liên\s*quan\s*đến|so\s*với|khác\s*nhau\s*(?:gì|như))",
    re.IGNORECASE,
)
# Multi-intent connectors (signal mixed-intent / decomposable queries).
_MULTI_INTENT_RE = re.compile(
    r"([;；]|\bvà\b|đồng\s*thời|ngoài\s*ra|bên\s*cạnh\s*đó|"
    r"cũng\s*như|kèm\s*theo|vừa\s+\S+\s+vừa)",
    re.IGNORECASE,
)

# LLM prompt — single call returns the lane + a short reason. Defensive parse.
ROUTER_PROMPT = """\
Bạn là bộ định tuyến truy vấn pháp lý. Phân loại câu hỏi vào ĐÚNG MỘT lane.

Các lane:
- direct_lookup: hỏi nội dung một điều/khoản cụ thể.
- procedure_detail: hỏi thủ tục, hồ sơ, trình tự, các bước.
- condition_requirement: hỏi điều kiện, yêu cầu, tiêu chí.
- sanction: hỏi xử phạt, vi phạm, chế tài.
- cross_doc: cần nhiều văn bản, đối chiếu, quy định chi tiết/hướng dẫn.
- scenario: tình huống giả định, ví dụ cụ thể.

Trả về JSON hợp lệ, không giải thích thêm:
{{
  "lane": "direct_lookup|procedure_detail|condition_requirement|sanction|cross_doc|scenario",
  "reason": "ngắn gọn"
}}

Câu hỏi: {query}"""


# --------------------------------------------------------------------------- #
# Output / config dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class RoutingDecision:
    """Structured router output (``router.md`` §4).

    Attributes
    ----------
    lane:
        One of :data:`LANES`.
    need_graph_expand:
        Whether the pipeline should run graph expansion for this query.
        ``True`` for cross-document / scenario lanes (multi-hop coverage).
    should_decompose:
        Whether the pipeline should call the decomposer before retrieval.
        ``True`` for long / multi-intent / cross-doc / scenario queries.
    retrieval_depth:
        Candidate-pool width hint: ``narrow`` | ``normal`` | ``wide``.
    rerank_strength:
        Reranker effort hint: ``light`` | ``normal`` | ``strong``.
    confidence:
        ``0.0–1.0`` confidence in the lane assignment.
    reason:
        Short human-readable explanation for debugging.
    source:
        ``"llm"`` or ``"rules"``.
    """

    lane: str = "direct_lookup"
    need_graph_expand: bool = False
    should_decompose: bool = False
    retrieval_depth: str = "normal"
    rerank_strength: str = "normal"
    confidence: float = 0.5
    reason: str = ""
    source: str = "rules"

    def __post_init__(self) -> None:
        if self.lane not in LANES:
            self.lane = "direct_lookup"
        if self.retrieval_depth not in RETRIEVAL_DEPTHS:
            self.retrieval_depth = "normal"
        if self.rerank_strength not in RERANK_STRENGTHS:
            self.rerank_strength = "normal"
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane": self.lane,
            "need_graph_expand": self.need_graph_expand,
            "should_decompose": self.should_decompose,
            "retrieval_depth": self.retrieval_depth,
            "rerank_strength": self.rerank_strength,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "source": self.source,
        }


@dataclass
class RouterConfig:
    """Knobs for :class:`Router`.

    Attributes
    ----------
    use_llm:
        When ``True`` and an ``llm_call`` is injected, the LLM decides the lane
        (rules still set the strategy switches). When ``False``, pure rules.
    long_query_words:
        Word count at/above which a query is treated as *potentially* multi-hop
        or mixed-intent (``Agents_task.md``: ~57% of questions ≥ 30 words).
    decompose_lanes:
        Lanes that always trigger decomposition regardless of length.
    graph_expand_lanes:
        Lanes that trigger graph expansion.
    """

    use_llm: bool = False
    long_query_words: int = 30
    decompose_lanes: tuple = ("cross_doc", "scenario")
    graph_expand_lanes: tuple = ("cross_doc", "scenario", "procedure_detail")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
class Router:
    """Classify a legal query into one retrieval lane + strategy switches.

    Parameters
    ----------
    llm_call:
        Optional ``(prompt: str) -> str`` returning the LLM's raw text. The
        router expects JSON matching :data:`ROUTER_PROMPT`. ``None`` (default)
        disables the LLM path — pure rule-based classification.
    config:
        :class:`RouterConfig`.
    """

    def __init__(
        self,
        llm_call: Optional[Callable[[str], str]] = None,
        config: Optional[RouterConfig] = None,
    ) -> None:
        self._llm_call = llm_call
        self.config = config or RouterConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def route(self, query: str) -> RoutingDecision:
        """Return the :class:`RoutingDecision` for ``query``.

        The lane is decided by the LLM when enabled and available, else by
        rules. The strategy switches (``need_graph_expand``,
        ``should_decompose``, ``retrieval_depth``, ``rerank_strength``) are
        always derived deterministically from the lane + query shape, so they
        stay consistent regardless of which path picked the lane.
        """
        q = (query or "").strip()
        if not q:
            return RoutingDecision(
                lane="direct_lookup", confidence=0.0,
                reason="empty query", source="rules",
            )

        lane, confidence, reason, source = self._classify_lane(q)
        switches = self._derive_switches(q, lane)
        return RoutingDecision(
            lane=lane,
            confidence=confidence,
            reason=reason,
            source=source,
            **switches,
        )

    # ------------------------------------------------------------------ #
    # Lane classification
    # ------------------------------------------------------------------ #
    def _classify_lane(self, query: str):
        """Return ``(lane, confidence, reason, source)``."""
        if self.config.use_llm and self._llm_call is not None:
            llm_lane = self._classify_lane_llm(query)
            if llm_lane is not None:
                lane, reason = llm_lane
                return lane, 0.9, reason, "llm"
            logger.debug("Router LLM failed/empty; falling back to rules.")
        lane, confidence, reason = self._classify_lane_rules(query)
        return lane, confidence, reason, "rules"

    def _classify_lane_llm(self, query: str):
        """Call the injected LLM. Return ``(lane, reason)`` or ``None``."""
        prompt = ROUTER_PROMPT.format(query=query)
        try:
            raw = self._llm_call(prompt)
        except Exception as exc:  # noqa: BLE001 — degrade to rules on any error
            logger.warning("Router LLM call raised %s; using rules.", exc)
            return None
        if not raw:
            return None
        data = _extract_json(raw)
        if not isinstance(data, dict):
            return None
        lane = str(data.get("lane", "")).strip()
        if lane not in LANES:
            return None
        reason = str(data.get("reason", "")).strip() or "LLM lane classification"
        return lane, reason

    def _classify_lane_rules(self, query: str):
        """Deterministic rule-based lane scoring. Returns ``(lane, conf, reason)``.

        Each lane scores by counting matched cue groups. Priority on ties
        favours the more *specific* / higher-value lanes (cross_doc, scenario,
        sanction) over the generic ones, matching the spec's emphasis on
        catching multi-hop and scenario questions rather than defaulting them
        to a single-article lookup.
        """
        scores: Dict[str, int] = {lane: 0 for lane in LANES}
        scores["procedure_detail"] += len(_PROCEDURE_RE.findall(query))
        scores["condition_requirement"] += len(_CONDITION_RE.findall(query))
        scores["sanction"] += len(_SANCTION_RE.findall(query))
        scores["scenario"] += len(_SCENARIO_RE.findall(query))
        scores["cross_doc"] += len(_CROSS_DOC_RE.findall(query))
        scores["direct_lookup"] += len(_DIRECT_RE.findall(query))

        # A multi-intent connector nudges toward cross_doc (likely multi-hop).
        n_words = len(query.split())
        if _MULTI_INTENT_RE.search(query) and n_words >= self.config.long_query_words:
            scores["cross_doc"] += 1

        max_score = max(scores.values())
        if max_score == 0:
            # No cue matched. Long questions are treated as potentially
            # multi-hop (Agents_task.md §Router); short ones as direct lookup.
            if n_words >= self.config.long_query_words:
                return "cross_doc", 0.35, "no cue; long query → assume multi-hop"
            return "direct_lookup", 0.4, "no cue; short query → direct lookup"

        # Tie-break priority (most specific / highest-value first).
        priority = (
            "scenario",
            "cross_doc",
            "sanction",
            "procedure_detail",
            "condition_requirement",
            "direct_lookup",
        )
        best = max(priority, key=lambda lane: (scores[lane], -priority.index(lane)))
        # Confidence scales with lead over the runner-up.
        ordered = sorted(scores.values(), reverse=True)
        lead = ordered[0] - (ordered[1] if len(ordered) > 1 else 0)
        confidence = min(0.95, 0.55 + 0.15 * lead)
        reason = f"rule cues: {best} (score={scores[best]}, lead={lead})"
        return best, confidence, reason

    # ------------------------------------------------------------------ #
    # Strategy switches (always rule-derived from lane + query shape)
    # ------------------------------------------------------------------ #
    def _derive_switches(self, query: str, lane: str) -> Dict[str, Any]:
        """Map ``(lane, query shape)`` → the coarse strategy switches."""
        n_words = len(query.split())
        is_long = n_words >= self.config.long_query_words
        has_multi_intent = bool(_MULTI_INTENT_RE.search(query))

        need_graph_expand = lane in self.config.graph_expand_lanes
        # A semicolon is an explicit clause boundary → decompose regardless of
        # length. Otherwise require a long query AND a softer multi-intent cue,
        # plus the always-decompose lanes (cross_doc / scenario).
        has_semicolon = bool(re.search(r"[;；]", query))
        should_decompose = (
            lane in self.config.decompose_lanes
            or has_semicolon
            or (is_long and has_multi_intent)
        )

        # Retrieval depth: wider pools for complex / multi-hop lanes.
        if lane in ("cross_doc", "scenario"):
            retrieval_depth = "wide"
        elif lane == "direct_lookup" and not is_long:
            retrieval_depth = "narrow"
        else:
            retrieval_depth = "normal"

        # Rerank strength: stronger for direct lookups (precision matters most
        # when one article is the answer); lighter when we lean on graph/select.
        if lane == "direct_lookup":
            rerank_strength = "strong"
        elif lane in ("cross_doc", "scenario"):
            rerank_strength = "normal"
        else:
            rerank_strength = "normal"

        return {
            "need_graph_expand": need_graph_expand,
            "should_decompose": should_decompose,
            "retrieval_depth": retrieval_depth,
            "rerank_strength": rerank_strength,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _extract_json(raw: str) -> Optional[Any]:
    """Best-effort JSON extraction from an LLM response.

    Handles bare JSON, fenced ```json blocks, and trailing prose by locating
    the first balanced ``{...}`` object. Returns ``None`` on failure.
    """
    if not raw:
        return None
    text = raw.strip()
    # Strip code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Try a direct parse first.
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    # Fall back to the first balanced object.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except (ValueError, TypeError):
                    return None
    return None
