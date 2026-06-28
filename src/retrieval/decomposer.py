"""Query decomposition for SME Legal QA retrieval (pre-retrieval splitting).

The decomposer is Stage 2 of the 7-stage pipeline (``query_decomposition.md``).
It runs **only when the router sets ``should_decompose=True``** — typically for
long / multi-intent / cross-document / scenario queries where a single
retrieval pass misses facets.

The decomposition produces 2–5 **focused sub-questions**, each representing one
legal facet (one condition, one procedure, one sanction, etc.). The critical
rule (§6):

    Retrieve with sub-questions, rerank with the original query.

This preserves user intent for relevance scoring while broadening recall.

Design goals
------------
* **LLM-driven with rule-based fallback.** A deterministic, dependency-free
  rule splitter always works. An injected ``llm_call`` overrides when
  available; failures degrade silently to rules.
* **Facet-focused output.** Each sub-question has a ``role`` label
  (``"condition"`` | ``"procedure"`` | ``"support_level"`` | ``"cross_doc"`` |
  ...) so downstream logic can treat sub-questions differently if needed.
* **Validation + dedup.** Sub-questions too short, lacking legal subject, or
  near-duplicates are discarded.
* **Pure Python, zero heavy deps.** Only stdlib. LLM client is injected.

Salvaged from ``sub_query_router.py`` (~810 lines → ~250 lines) by removing:
  - PLANNER_PROMPT (complexity + facets in one call — too heavy).
  - ``infer_complexity()`` function.
  - Multi-variant query building (that's in the legacy unified_pipeline, not
    needed in the new clean 7-stage flow).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "SUB_QUESTION_ROLES",
    "SubQuestion",
    "DecompositionResult",
    "DecomposerConfig",
    "Decomposer",
    "clean_sub_query",
    "normalize_sub_queries",
    "rule_based_decompose",
    "DECOMPOSITION_PROMPT",
]

# Valid role labels for sub-questions (query_decomposition.md §4).
SUB_QUESTION_ROLES = (
    "condition",
    "procedure",
    "support_level",
    "sanction",
    "cross_doc",
    "direct_lookup",
    "other",
)

# LLM prompt — returns 2–5 independent sub-questions with role labels.
DECOMPOSITION_PROMPT = """\
Bạn là bộ phân tích truy vấn pháp lý.

Nhiệm vụ:
1. Quyết định query này có cần tách thành các sub-query hay không.
2. Nếu cần, tách thành các sub-query độc lập, mỗi câu phải giữ nguyên bối cảnh pháp lý và chủ thể chính.
3. Không tách theo từ nối bề mặt nếu làm mất nghĩa pháp lý.
4. Mỗi sub-query phải đủ rõ để có thể truy hồi tài liệu một cách độc lập.
5. Gán role cho mỗi sub-query: condition, procedure, support_level, sanction, cross_doc, direct_lookup, other.
6. Trả về JSON hợp lệ, không giải thích thêm.

Quy tắc:
- Nếu query chỉ có một ý rõ ràng, trả về should_decompose=false.
- Nếu query có nhiều yêu cầu pháp lý độc lập, trả về should_decompose=true.
- Số sub-query tối đa: 5.
- Không tạo sub-query quá ngắn, mơ hồ, hoặc trùng ý.

Output schema:
{{
  "should_decompose": true/false,
  "sub_questions": [
    {{"id": "q1", "text": "...", "role": "condition"}},
    {{"id": "q2", "text": "...", "role": "procedure"}}
  ]
}}

Query: {query}"""

# Vietnamese legal terms that signal a sub-query still has legal subject matter.
# Salvaged from sub_query_router.py (line 250-258) and BROADENED for the SME
# legal domain: the legacy regex was criminal/civil-focused (bồi thường, trách
# nhiệm) and missed the tax / business / procedural vocabulary that dominates
# this dataset (thuế, ưu đãi, thủ tục, điều kiện, đăng ký, doanh nghiệp ...),
# which silently dropped every valid SME sub-query.
_LEGAL_TERMS_RE = re.compile(
    r"(điều\s+\d+|khoản\s+\d+|luật\s+\S+|nghị\s+định|thông\s+tư|"
    r"bộ\s+luật|hiến\s+pháp|pháp\s+lệnh|quyết\s+định|"
    r"hành\s+vi|chứng\s+cứ|thiệt\s+hại|trách\s+nhiệm|"
    r"quyền\s+\S+|nghĩa\s+vụ|hợp\s+đồng|bồi\s+thường|"
    r"xử\s+phạt|vi\s+phạm|tội\s+\S+|hình\s+sự|dân\s+sự|"
    r"hành\s+chính|lao\s+động|sở\s+hữu|thừa\s+kế|"
    # SME / business / tax / procedural domain.
    r"điều\s+kiện|yêu\s+cầu|tiêu\s+chí|thủ\s+tục|hồ\s+sơ|trình\s+tự|"
    r"đăng\s+ký|kinh\s+doanh|doanh\s+nghiệp|thuế|ưu\s+đãi|hỗ\s+trợ|"
    r"giấy\s+phép|vốn|ngân\s+sách|chế\s+tài|mức\s+phạt|văn\s+bản)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Output dataclasses (query_decomposition.md §7)
# --------------------------------------------------------------------------- #
@dataclass
class SubQuestion:
    """One focused sub-question representing a single legal facet.

    Attributes
    ----------
    id:
        Unique identifier (``"q1"``, ``"q2"``, ...).
    text:
        The sub-question text (cleaned, terminal-punctuated).
    role:
        One of :data:`SUB_QUESTION_ROLES`.
    """

    id: str
    text: str
    role: str = "other"

    def __post_init__(self) -> None:
        if self.role not in SUB_QUESTION_ROLES:
            self.role = "other"

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text, "role": self.role}


@dataclass
class DecompositionResult:
    """The output of :meth:`Decomposer.decompose`.

    Attributes
    ----------
    should_decompose:
        Whether the decomposer decided to split the query.
    original_query:
        The user's query verbatim (for traceability).
    sub_questions:
        List of 2–5 :class:`SubQuestion`. Empty if ``should_decompose=False``.
    confidence:
        ``0.0–1.0`` confidence in the decomposition decision.
    reason:
        Short explanation.
    source:
        How the decision was made — ``"llm"`` or ``"rules"``.
    """

    should_decompose: bool = False
    original_query: str = ""
    sub_questions: List[SubQuestion] = field(default_factory=list)
    confidence: float = 0.5
    reason: str = ""
    source: str = "rules"

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        # Validate sub_questions.
        validated = []
        for sq in self.sub_questions:
            if isinstance(sq, SubQuestion):
                validated.append(sq)
            elif isinstance(sq, dict):
                validated.append(SubQuestion(**sq))
        self.sub_questions = validated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_decompose": self.should_decompose,
            "original_query": self.original_query,
            "sub_questions": [sq.to_dict() for sq in self.sub_questions],
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "source": self.source,
        }


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DecomposerConfig:
    """Knobs for :class:`Decomposer`.

    Attributes
    ----------
    use_llm:
        When ``True`` and an ``llm_call`` is injected, the LLM decides how to
        decompose. When ``False``, rule-based fallback only.
    max_sub_questions:
        Hard cap on the number of sub-questions (2–5 recommended).
    min_sub_query_chars:
        Sub-questions shorter than this are discarded.
    dedup_threshold:
        Jaccard-like token overlap above which sub-queries are merged.
    """

    use_llm: bool = True
    max_sub_questions: int = 5
    min_sub_query_chars: int = 10
    dedup_threshold: float = 0.85


# --------------------------------------------------------------------------- #
# Pure functions: clean, normalize, rule-based split
# Salvaged from sub_query_router.py (lines 261-450).
# --------------------------------------------------------------------------- #
def clean_sub_query(text: str) -> str:
    """Clean a single sub-query string.

    - Trim leading/trailing whitespace.
    - Collapse multiple spaces/newlines into a single space.
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
    max_count: int = 5,
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
            # Treat near-identical token sets and subset/superset pairs as
            # duplicates. Keep the longer one (more context).
            if jaccard >= dedup_threshold or containment >= 0.9:
                is_dup = True
                if len(c) > len(existing):
                    deduped[i] = c
                break
        if not is_dup:
            deduped.append(c)

    # Step 5: Sort by length desc, truncate.
    deduped.sort(key=len, reverse=True)
    return deduped[:max_count]


def rule_based_decompose(query: str, max_sub_queries: int = 5) -> List[str]:
    """Rule-based fallback decomposition (lightweight, deterministic).

    Split on semicolons and strong conjunctions (``"và đồng thời"``, etc.).
    Each clause becomes one sub-query if it carries legal subject matter.

    Parameters
    ----------
    query:
        The user's raw query.
    max_sub_queries:
        Hard cap on the number of splits.

    Returns
    -------
    List[str]
        Up to ``max_sub_queries`` sub-query strings (not yet cleaned or
        deduplicated — call :func:`normalize_sub_queries` on the output).
    """
    # Split on semicolons.
    clauses = [c.strip() for c in re.split(r"[;；]", query) if c.strip()]

    # Further split on strong conjunctions if a clause is still long.
    # "đồng thời" / "ngoài ra" / "bên cạnh đó" / "cũng như" mark independent
    # legal facets (with or without a leading "và").
    _conj = r"(?:và\s+)?đồng\s+thời|ngoài\s+ra|bên\s+cạnh\s+đó|cũng\s+như"
    split_clauses: List[str] = []
    for clause in clauses:
        parts = re.split(rf"\b(?:{_conj})\b", clause, flags=re.IGNORECASE)
        for p in parts:
            p = p.strip()
            if p:
                split_clauses.append(p)

    # Only keep clauses with legal terms.
    valid = [c for c in split_clauses if _LEGAL_TERMS_RE.search(c)]
    return valid[:max_sub_queries]


# --------------------------------------------------------------------------- #
# Decomposer
# --------------------------------------------------------------------------- #
class Decomposer:
    """Query decomposition for complex legal questions.

    Parameters
    ----------
    llm_call:
        Optional ``(prompt: str) -> str`` returning the LLM's raw text. The
        decomposer expects JSON matching :data:`DECOMPOSITION_PROMPT`.
        ``None`` disables the LLM path — pure rule-based splitting.
    config:
        :class:`DecomposerConfig`.
    """

    def __init__(
        self,
        llm_call: Optional[Callable[[str], str]] = None,
        config: Optional[DecomposerConfig] = None,
    ) -> None:
        self._llm_call = llm_call
        self.config = config or DecomposerConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def decompose(self, query: str) -> DecompositionResult:
        """Return the :class:`DecompositionResult` for ``query``.

        When ``should_decompose=True``, the result carries 2–5 sub-questions.
        When ``False``, ``sub_questions`` is empty and the caller should
        retrieve with the original query only.
        """
        q = (query or "").strip()
        if not q:
            return DecompositionResult(
                should_decompose=False,
                original_query=q,
                confidence=0.0,
                reason="empty query",
                source="rules",
            )

        # Try LLM first if enabled.
        if self.config.use_llm and self._llm_call is not None:
            llm_result = self._decompose_llm(q)
            if llm_result is not None:
                return llm_result
            logger.debug("Decomposer LLM failed/empty; falling back to rules.")

        # Fall back to rule-based split.
        return self._decompose_rules(q)

    # ------------------------------------------------------------------ #
    # LLM decomposition
    # ------------------------------------------------------------------ #
    def _decompose_llm(self, query: str) -> Optional[DecompositionResult]:
        """Call the injected LLM. Return :class:`DecompositionResult` or ``None``."""
        prompt = DECOMPOSITION_PROMPT.format(query=query)
        try:
            raw = self._llm_call(prompt)
        except Exception as exc:  # noqa: BLE001 — degrade to rules on any error
            logger.warning("Decomposer LLM call raised %s; using rules.", exc)
            return None
        if not raw:
            return None
        data = _extract_json(raw)
        if not isinstance(data, dict):
            return None

        should_decompose = bool(data.get("should_decompose", False))
        sub_q_raw = data.get("sub_questions", [])
        if not isinstance(sub_q_raw, list):
            return None

        # Parse and normalize sub-questions.
        texts = []
        roles = []
        for item in sub_q_raw:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                role = str(item.get("role", "other")).strip()
            elif isinstance(item, str):
                text = item.strip()
                role = "other"
            else:
                continue
            if text:
                texts.append(text)
                roles.append(role if role in SUB_QUESTION_ROLES else "other")

        normalized = normalize_sub_queries(
            texts,
            max_count=self.config.max_sub_questions,
            min_chars=self.config.min_sub_query_chars,
            dedup_threshold=self.config.dedup_threshold,
        )

        # Build SubQuestion objects (match role if text survived normalization).
        sub_questions = []
        for i, text in enumerate(normalized):
            # Find the original role for this text (best-effort match).
            role = "other"
            for j, orig in enumerate(texts):
                if orig.lower().strip() in text.lower() or text.lower() in orig.lower().strip():
                    role = roles[j]
                    break
            sub_questions.append(SubQuestion(id=f"q{i+1}", text=text, role=role))

        # If LLM said should_decompose but normalization left <2 valid subs,
        # treat as no-decompose (single-intent query misclassified).
        if should_decompose and len(sub_questions) < 2:
            should_decompose = False

        return DecompositionResult(
            should_decompose=should_decompose,
            original_query=query,
            sub_questions=sub_questions,
            confidence=0.85,
            reason="LLM decomposition",
            source="llm",
        )

    # ------------------------------------------------------------------ #
    # Rule-based decomposition
    # ------------------------------------------------------------------ #
    def _decompose_rules(self, query: str) -> DecompositionResult:
        """Deterministic rule-based split (fallback)."""
        raw_splits = rule_based_decompose(query, self.config.max_sub_questions)
        normalized = normalize_sub_queries(
            raw_splits,
            max_count=self.config.max_sub_questions,
            min_chars=self.config.min_sub_query_chars,
            dedup_threshold=self.config.dedup_threshold,
        )

        # Only decompose if we got 2+ valid sub-queries.
        should_decompose = len(normalized) >= 2

        # Assign roles (best-effort heuristics based on keywords).
        sub_questions = []
        for i, text in enumerate(normalized):
            role = _infer_role(text)
            sub_questions.append(SubQuestion(id=f"q{i+1}", text=text, role=role))

        return DecompositionResult(
            should_decompose=should_decompose,
            original_query=query,
            sub_questions=sub_questions,
            confidence=0.6 if should_decompose else 0.4,
            reason=f"rule split: {len(sub_questions)} sub-queries" if should_decompose else "single-intent query",
            source="rules",
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _extract_json(raw: str) -> Optional[Any]:
    """Best-effort JSON extraction from an LLM response (same as router.py)."""
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


def _infer_role(text: str) -> str:
    """Heuristic role assignment for a sub-query (rule-based fallback)."""
    text_lower = text.lower()
    if re.search(r"(điều kiện|yêu cầu|tiêu chí|được phép)", text_lower):
        return "condition"
    if re.search(r"(thủ tục|hồ sơ|trình tự|đăng ký|nộp đơn)", text_lower):
        return "procedure"
    if re.search(r"(mức|tỷ lệ|phần trăm|ngân sách hỗ trợ)", text_lower):
        return "support_level"
    if re.search(r"(xử phạt|vi phạm|chế tài|mức phạt)", text_lower):
        return "sanction"
    if re.search(r"(văn bản nào|nghị định nào|quy định chi tiết|hướng dẫn)", text_lower):
        return "cross_doc"
    if re.search(r"(điều \d+|khoản \d+|nội dung điều)", text_lower):
        return "direct_lookup"
    return "other"
