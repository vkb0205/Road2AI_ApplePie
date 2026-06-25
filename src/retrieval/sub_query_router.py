"""Sub-query routing for G-LRAG retrieval (pre-retrieval decomposition).

This module provides a :class:`SubQueryRouter` that optionally decomposes a
complex legal query into 2–4 independent sub-queries, each preserving legal
context and the primary subject, so the downstream :class:`HybridRetriever` can
retrieve more targeted chunks per sub-query.

Design goals
------------
* **LLM-driven with rule-based fallback.** The router can call an injected LLM
  client to decide whether to decompose and how many sub-queries to produce.
  When the LLM is unavailable or fails, a lightweight rule-based fallback
  (keyword/semicolon splitter) kicks in with full logging.
* **Pure Python, zero heavy deps.** The module itself imports only stdlib
  (``json``, ``re``, ``logging``, ``dataclasses``). The LLM client is
  injected at construction time — the caller decides which library to use
  (OpenAI, Anthropic, local vLLM, etc.).
* **Deterministic normalize/clean.** Post-processing normalises whitespace,
  removes near-duplicates, discards sub-queries that are too short or lack a
  legal subject, and preserves important legal terms (law names, article
  numbers, legal acts).
* **Debug-first.** Every decomposition (LLM or fallback) produces a
  :class:`RouterDecision` carrying the full decision trace, and each
  sub-query gets an independent :class:`SubQueryTrace` when the retriever
  is run in debug mode.
* **Drop-in pre-processing.** ``router.route(query) → RouterDecision`` is
  designed to be called *before* ``HybridRetriever.retrieve()``. The
  downstream pipeline can loop over ``decision.sub_queries`` unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = [
    "RouterDecision",
    "RouterFallbackLog",
    "SubQueryRouter",
    "SubQueryRouterConfig",
    "normalize_sub_queries",
    "clean_sub_query",
    "rule_based_decompose",
    "DECOMPOSITION_PROMPT",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template (Vietnamese legal domain)
# ---------------------------------------------------------------------------

DECOMPOSITION_PROMPT = """\
Bạn là bộ định tuyến truy vấn pháp lý.

Nhiệm vụ:
1. Quyết định query này có cần tách thành các sub-query hay không.
2. Nếu cần, tách thành các sub-query độc lập, mỗi câu phải giữ nguyên bối cảnh pháp lý và chủ thể chính.
3. Không tách theo từ nối bề mặt nếu làm mất nghĩa pháp lý.
4. Mỗi sub-query phải đủ rõ để có thể truy hồi tài liệu một cách độc lập.
5. Trả về JSON hợp lệ, không giải thích thêm.

Quy tắc:
- Nếu query chỉ có một ý rõ ràng, trả về should_decompose=false.
- Nếu query có nhiều yêu cầu pháp lý độc lập, trả về should_decompose=true.
- Số sub-query tối đa: 4.
- Không tạo sub-query quá ngắn, mơ hồ, hoặc trùng ý.

Output schema:
{{
  "should_decompose": true/false,
  "sub_queries": [
    "...",
    "..."
  ]
}}

Query: {query}"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RouterFallbackLog:
    """Record of a rule-based fallback decomposition.

    Stored on :class:`RouterDecision` when the LLM call fails or is
    unavailable, so the operator can debug why a query was decomposed
    a certain way.
    """

    original_query: str = ""
    reason: str = ""
    rule_triggered: str = ""
    sub_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "reason": self.reason,
            "rule_triggered": self.rule_triggered,
            "sub_queries": self.sub_queries,
        }


@dataclass
class RouterDecision:
    """The output of :meth:`SubQueryRouter.route`.

    Attributes
    ----------
    should_decompose:
        Whether the router decided to split the query.
    num_sub_queries:
        ``len(sub_queries)`` — convenience for downstream branching.
    sub_queries:
        List of independent sub-queries (length ≥ 1). When
        ``should_decompose=False``, this is a single-element list
        containing the original query.
    source:
        How the decision was made — ``"llm"`` or ``"rule_fallback"``.
    raw_llm_output:
        The raw string returned by the LLM (``None`` for fallback).
    fallback_log:
        Populated only when the rule-based fallback was used.
    """

    should_decompose: bool = False
    num_sub_queries: int = 1
    sub_queries: List[str] = field(default_factory=list)
    source: str = "rule_fallback"
    raw_llm_output: Optional[str] = None
    fallback_log: Optional[RouterFallbackLog] = None

    def __post_init__(self) -> None:
        if not self.sub_queries:
            self.sub_queries = []
        self.num_sub_queries = len(self.sub_queries)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "should_decompose": self.should_decompose,
            "num_sub_queries": self.num_sub_queries,
            "sub_queries": self.sub_queries,
            "source": self.source,
        }
        if self.raw_llm_output is not None:
            d["raw_llm_output"] = self.raw_llm_output
        if self.fallback_log is not None:
            d["fallback_log"] = self.fallback_log.to_dict()
        return d


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SubQueryRouterConfig:
    """Knobs for :class:`SubQueryRouter`.

    Attributes
    ----------
    use_llm:
        When ``True``, the router calls the injected LLM client. When
        ``False``, it skips straight to rule-based fallback (useful for
        offline / CPU-only baselines).
    llm_timeout:
        Seconds to wait for the LLM response before falling back.
    max_sub_queries:
        Hard cap on the number of sub-queries (2–4 recommended).
    min_sub_query_chars:
        Sub-queries shorter than this are discarded during normalisation.
    dedup_threshold:
        Jaccard-like token overlap above which sub-queries are considered
        near-duplicates and merged.
    debug:
        When ``True``, emit debug logs for every routing decision.
    """

    use_llm: bool = True
    llm_timeout: float = 10.0
    max_sub_queries: int = 4
    min_sub_query_chars: int = 10
    dedup_threshold: float = 0.85
    debug: bool = False


# ---------------------------------------------------------------------------
# Normalise / clean (pure functions)
# ---------------------------------------------------------------------------

# Vietnamese legal terms that signal a sub-query still has legal subject matter.
_LEGAL_TERMS_RE = re.compile(
    r"(điều\s+\d+|khoản\s+\d+|luật\s+\S+|nghị\s+định|thông\s+tư|"
    r"bộ\s+luật|hiến\s+pháp|pháp\s+lệnh|quyết\s+định|"
    r"hành\s+vi|chứng\s+cứ|thiệt\s+hại|trách\s+nhiệm|"
    r"quyền\s+\S+|nghĩa\s+vụ|hợp\s+đồng|bồi\s+thường|"
    r"xử\s+phạt|vi\s+phạm|tội\s+\S+|hình\s+sự|dân\s+sự|"
    r"hành\s+chính|lao\s+động|sở\s+hữu|thừa\s+kế|hợp\s+đồng|thừa\s+kế)",
    re.IGNORECASE,
)


def clean_sub_query(text: str) -> str:
    """Clean a single sub-query string.

    - Trim leading/trailing whitespace.
    - Collapse multiple spaces/newlines into a single space.
    - Normalise Vietnamese diacritic punctuation (remove stray combining marks).
    - Remove non-printable / control characters.
    - Ensure the text ends with appropriate terminal punctuation if missing.

    Returns the cleaned string, or ``""`` if the result is empty.
    """
    if not text:
        return ""
    # Collapse whitespace.
    text = " ".join(text.split())
    # Replace control characters with spaces (avoid fusing words).
    text = "".join(ch if ch.isprintable() or ch in "\n\t" else " " for ch in text)
    # Collapse whitespace again after control-char replacement.
    text = " ".join(text.split())
    text = text.strip()
    # Remove leading/trailing punctuation clutter that doesn't belong.
    text = text.strip(".,;:-–— \t")
    if not text:
        return ""
    # Ensure terminal punctuation if the text looks like a sentence.
    if text[-1] not in ".?!。？":
        text += "."
    return text


def normalize_sub_queries(
    sub_queries: List[str],
    *,
    max_count: int = 4,
    min_chars: int = 10,
    dedup_threshold: float = 0.85,
) -> List[str]:
    """Normalise and deduplicate a list of sub-queries.

    1. Clean each sub-query with :func:`clean_sub_query`.
    2. Discard sub-queries that are too short (``< min_chars``) or empty.
    3. Discard sub-queries that lack any legal subject term.
    4. Merge near-duplicates (token-set overlap ≥ ``dedup_threshold``).
    5. Truncate to ``max_count``, keeping the longest sub-queries first
       (they tend to carry more legal context).

    Parameters
    ----------
    sub_queries:
        Raw sub-query strings from LLM or rule-based splitter.
    max_count:
        Maximum number of sub-queries to return.
    min_chars:
        Minimum character length for a sub-query to be kept.
    dedup_threshold:
        Jaccard-like token overlap threshold for deduplication (0.0–1.0).

    Returns
    -------
    List[str]
        Cleaned, deduplicated, truncated list of sub-queries.
    """
    # Step 1: Clean.
    cleaned: List[str] = []
    for sq in sub_queries:
        c = clean_sub_query(sq)
        if c:
            cleaned.append(c)

    # Step 2: Filter by length.
    cleaned = [c for c in cleaned if len(c) >= min_chars]

    # Step 3: Filter by legal subject presence.
    cleaned = [c for c in cleaned if _LEGAL_TERMS_RE.search(c)]

    if not cleaned:
        return []

    # Step 4: Deduplicate near-duplicates (keep the longer one).
    deduped: List[str] = []
    for c in cleaned:
        c_tokens = set(c.lower().split())
        is_dup = False
        for i, existing in enumerate(deduped):
            e_tokens = set(existing.lower().split())
            if not c_tokens or not e_tokens:
                continue
            intersection = c_tokens & e_tokens
            union = c_tokens | e_tokens
            jaccard = len(intersection) / len(union) if union else 0.0
            containment = len(intersection) / min(len(c_tokens), len(e_tokens))
            # Treat near-identical token sets and subset-style shorter rewrites
            # as duplicates. The containment check catches cases like
            # "trách nhiệm bồi thường" vs "xác định trách nhiệm bồi thường
            # theo Điều 15", which are not high-Jaccard but are not useful as
            # independent sub-queries.
            if jaccard >= dedup_threshold or containment >= dedup_threshold:
                # Keep the longer one.
                if len(c) > len(existing):
                    deduped[i] = c
                is_dup = True
                break
        if not is_dup:
            deduped.append(c)

    # Step 5: Sort by length desc (longer = more context), then truncate.
    deduped.sort(key=len, reverse=True)
    return deduped[:max_count]


# ---------------------------------------------------------------------------
# Rule-based fallback decomposition
# ---------------------------------------------------------------------------

# Patterns that suggest multiple independent legal requirements.
_MULTI_CLAUSE_RE = re.compile(r"[;；]")
_CONJUNCTION_RE = re.compile(
    r"\s+(?:và|cần\s+xác\s+định|cần\s+làm\s+rõ|phải\s+làm\s+rõ|"
    r"cần\s+chuẩn\s+bị|phải\s+chuẩn\s+bị|đồng\s+thời|"
    r"ngoài\s+ra|bên\s+cạnh\s+đó)\s+",
    re.IGNORECASE,
)


def rule_based_decompose(query: str, max_parts: int = 4) -> List[str]:
    """Rule-based decomposition fallback.

    Splits on:
    1. Semicolons that precede a capital letter (clause boundaries).
    2. Conjunction keywords (``và``, ``cần xác định``, ``phải làm rõ``, etc.).

    Each resulting part is cleaned and returned. If no split points are
    found, the original query is returned as a single-element list.

    This is intentionally simple and deterministic — it's a *fallback*,
    not the primary decomposition strategy. The LLM-driven path should
    handle nuanced legal clause boundaries.
    """
    parts: List[str] = []

    # Pass 1: semicolon split.
    semicolon_splits = _MULTI_CLAUSE_RE.split(query)
    for part in semicolon_splits:
        part = part.strip()
        if not part:
            continue
        # Pass 2: conjunction split within each semicolon segment.
        conj_splits = _CONJUNCTION_RE.split(part)
        for cp in conj_splits:
            cp = cp.strip()
            if cp and len(cp) >= 5:
                parts.append(cp)

    if not parts:
        return [query.strip()]

    return parts[:max_parts]


# ---------------------------------------------------------------------------
# Sub-query router
# ---------------------------------------------------------------------------


class SubQueryRouter:
    """Pre-retrieval query decomposition router.

    Decides whether a legal query should be decomposed into independent
    sub-queries, using an injected LLM client with rule-based fallback.

    Parameters
    ----------
    llm_call:
        A callable ``(prompt: str) -> str`` that returns the LLM's raw
        text response. The router expects JSON output matching the
        decomposition prompt schema. ``None`` disables the LLM path
        (always uses rule-based fallback).
    config:
        :class:`SubQueryRouterConfig` with knobs for the router behaviour.
    """

    def __init__(
        self,
        llm_call: Optional[Callable[[str], str]] = None,
        config: Optional[SubQueryRouterConfig] = None,
    ) -> None:
        self._llm_call = llm_call
        self.config = config or SubQueryRouterConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def route(self, query: str) -> RouterDecision:
        """Route a query and return the decomposition decision.

        The returned :class:`RouterDecision` always contains at least one
        sub-query. When decomposition is not needed,
        ``decision.sub_queries == [query]`` and
        ``decision.should_decompose == False``.

        Parameters
        ----------
        query:
            The raw user query string.

        Returns
        -------
        RouterDecision
        """
        cfg = self.config

        # Normalise the input query first.
        query = clean_sub_query(query)
        if not query:
            return RouterDecision(
                should_decompose=False,
                sub_queries=[],
                source="rule_fallback",
                fallback_log=RouterFallbackLog(
                    original_query="",
                    reason="empty query after cleaning",
                    rule_triggered="empty_input",
                ),
            )

        # ---- LLM path -------------------------------------------------
        if cfg.use_llm and self._llm_call is not None:
            try:
                decision = self._llm_decompose(query)
                if decision is not None:
                    if cfg.debug:
                        logger.debug(
                            "SubQueryRouter: LLM decomposition succeeded "
                            "for query=%r → should_decompose=%s, "
                            "sub_queries=%s",
                            query,
                            decision.should_decompose,
                            decision.sub_queries,
                        )
                    return decision
            except Exception as exc:
                logger.warning(
                    "SubQueryRouter: LLM call failed (%s), falling back to "
                    "rule-based decomposition for query=%r",
                    exc,
                    query,
                )

        # ---- Rule-based fallback --------------------------------------
        return self._rule_fallback(query)

    # ------------------------------------------------------------------ #
    # LLM decomposition
    # ------------------------------------------------------------------ #
    def _llm_decompose(self, query: str) -> Optional[RouterDecision]:
        """Call the LLM and parse its JSON response.

        Returns ``None`` if the LLM response cannot be parsed, so the
        caller can fall back to rule-based decomposition.
        """
        cfg = self.config
        prompt = DECOMPOSITION_PROMPT.format(query=query)
        raw = self._llm_call(prompt)  # type: ignore[misc]

        if not raw or not raw.strip():
            return None

        # Strip markdown code fences if present.
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            # Remove opening fence.
            raw_clean = re.sub(r"^```(?:json)?\s*", "", raw_clean)
            # Remove closing fence.
            raw_clean = re.sub(r"\s*```\s*$", "", raw_clean)

        try:
            parsed = json.loads(raw_clean)
        except json.JSONDecodeError:
            # Try to extract JSON from the middle of the response.
            match = re.search(r"\{[^{}]*\}", raw_clean, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
            else:
                return None

        should_decompose = bool(parsed.get("should_decompose", False))
        raw_sub_queries: List[str] = parsed.get("sub_queries", [])

        if not isinstance(raw_sub_queries, list):
            raw_sub_queries = []

        # Normalise the sub-queries.
        sub_queries = normalize_sub_queries(
            raw_sub_queries,
            max_count=cfg.max_sub_queries,
            min_chars=cfg.min_sub_query_chars,
            dedup_threshold=cfg.dedup_threshold,
        )

        # If normalisation wiped out all sub-queries, treat as no-decompose.
        if not sub_queries:
            return RouterDecision(
                should_decompose=False,
                sub_queries=[query],
                source="llm",
                raw_llm_output=raw,
            )

        # If LLM said decompose but we only have 1 sub-query after cleaning,
        # or if there's only 1 anyway, treat as no-decompose.
        if len(sub_queries) <= 1:
            return RouterDecision(
                should_decompose=False,
                sub_queries=sub_queries if sub_queries else [query],
                source="llm",
                raw_llm_output=raw,
            )

        return RouterDecision(
            should_decompose=should_decompose and len(sub_queries) > 1,
            sub_queries=sub_queries,
            source="llm",
            raw_llm_output=raw,
        )

    # ------------------------------------------------------------------ #
    # Rule-based fallback
    # ------------------------------------------------------------------ #
    def _rule_fallback(self, query: str) -> RouterDecision:
        """Deterministic rule-based decomposition with full logging."""
        cfg = self.config

        raw_parts = rule_based_decompose(query, max_parts=cfg.max_sub_queries)
        sub_queries = normalize_sub_queries(
            raw_parts,
            max_count=cfg.max_sub_queries,
            min_chars=cfg.min_sub_query_chars,
            dedup_threshold=cfg.dedup_threshold,
        )

        # Build fallback reason.
        if not cfg.use_llm:
            reason = "LLM disabled (use_llm=False)"
        elif self._llm_call is None:
            reason = "no LLM client injected"
        else:
            reason = "LLM call failed or returned unparseable output"

        should_decompose = len(sub_queries) > 1

        fallback_log = RouterFallbackLog(
            original_query=query,
            reason=reason,
            rule_triggered=(
                "semicolon_split" if ";" in query
                else "conjunction_split" if _CONJUNCTION_RE.search(query)
                else "no_split_points"
            ),
            sub_queries=sub_queries,
        )

        if cfg.debug:
            logger.debug(
                "SubQueryRouter: rule-based fallback for query=%r → "
                "should_decompose=%s, sub_queries=%s, rule=%s",
                query,
                should_decompose,
                sub_queries,
                fallback_log.rule_triggered,
            )

        return RouterDecision(
            should_decompose=should_decompose,
            sub_queries=sub_queries if sub_queries else [query],
            source="rule_fallback",
            fallback_log=fallback_log,
        )