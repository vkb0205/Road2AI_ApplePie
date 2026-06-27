"""Post-generation guardrails for G-LRAG answers (pure, stdlib-only).

Enforces the submission/QA contract on a generated answer without any model
dependency, so it imports and unit-tests on a CPU-only box:

* **FR-02** — the ``answer`` must cite at least one ``Điều X`` that exists in
  the retrieved ``relevant_articles`` (grounded citation). If the model cited
  none, we backfill the top allowed citations.
* **Grounding** — citations the model invented that are NOT in the allowed list
  trigger a hedge note rather than silently passing fabricated law to the grader.
* **FR-05** — the answer must carry a reference-only disclaimer; we append a
  standard one when missing.
* **Hygiene** — strip chat-template / prompt-echo scaffolding the model may emit.

No hardcoded law ids or domain tables — everything is derived from the answer
text and the allowed-citation list passed in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

__all__ = [
    "GuardrailConfig",
    "normalize_text",
    "strip_prompt_echo",
    "extract_dieu_citations",
    "allowed_dieu_numbers",
    "answer_cites_allowed",
    "apply_guardrails",
]

# Match a "Điều <number>" citation, tolerant of an optional letter suffix
# (e.g. "Điều 12a") so canonicalisation matches the grader's number-only view.
_DIEU_CITE_RE = re.compile(r"Điều\s+\d+[A-Za-zÀ-ỹ]*", re.IGNORECASE | re.UNICODE)

_DEFAULT_DISCLAIMER = (
    "Lưu ý: Thông tin trên chỉ mang tính tham khảo, dựa trên các căn cứ pháp lý "
    "đã truy xuất và không thay thế cho tư vấn pháp lý chính thức."
)

_HEDGE_NOTE = (
    "Lưu ý: Một số điều luật được nhắc đến nằm ngoài danh sách căn cứ đã truy "
    "xuất; cần kiểm chứng thêm trước khi áp dụng."
)


@dataclass
class GuardrailConfig:
    """Knobs for the post-generation guardrails."""

    ensure_citation: bool = True       # backfill a citation if the answer has none
    max_backfill_citations: int = 3    # how many allowed Điều to append if missing
    flag_out_of_list: bool = True      # add a hedge note for invented citations
    ensure_disclaimer: bool = True     # append the reference-only disclaimer
    disclaimer: str = _DEFAULT_DISCLAIMER
    min_answer_chars: int = 0          # 0 = no minimum (the grader has none)


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip zero-width / BOM characters."""
    text = str(text).replace("\u200b", " ").replace("\ufeff", " ")
    return re.sub(r"\s+", " ", text).strip()


def strip_prompt_echo(text: str) -> str:
    """Remove chat-template / prompt-echo scaffolding from the model output."""
    text = normalize_text(text)
    for pat in (
        r"(?is)^system\s*.*?user\s*",   # echoed system+user turns
        r"(?is)^assistant\s*[:\-]?\s*",  # leading 'assistant:'
        r"(?is)^trả lời\s*[:\-]\s*",     # leading 'Trả lời:' (colon/dash required
                                          # so legitimate 'Trả lời ...' prose is kept)
    ):
        text = re.sub(pat, "", text).strip()
    return re.sub(r"^\s*[-–—]\s*", "", text).strip()


def _canon(cite: str) -> str:
    """Canonicalise a 'Điều X' citation to lowercase single-spaced form."""
    return normalize_text(cite).lower()


def extract_dieu_citations(text: str) -> Set[str]:
    """Return the set of canonical ``Điều X`` citations mentioned in *text*."""
    return {_canon(m.group(0)) for m in _DIEU_CITE_RE.finditer(str(text))}


def allowed_dieu_numbers(relevant_articles: Sequence[str]) -> Set[str]:
    """Set of canonical ``Điều X`` citations present in the allowed list.

    Each allowed item is a ``law_id|ten_van_ban|Điều X`` string; we pull every
    ``Điều X`` token out of it (robust to the third-segment format).
    """
    allowed: Set[str] = set()
    for a in relevant_articles:
        for m in _DIEU_CITE_RE.finditer(str(a)):
            allowed.add(_canon(m.group(0)))
    return allowed


def answer_cites_allowed(answer: str, relevant_articles: Sequence[str]) -> bool:
    """True iff the answer cites at least one allowed ``Điều X`` (FR-02)."""
    return bool(extract_dieu_citations(answer) & allowed_dieu_numbers(relevant_articles))


def _allowed_dieu_strings(relevant_articles: Sequence[str], limit: int) -> List[str]:
    """Ordered, de-duplicated ``Điều X`` display strings from the allowed list."""
    out: List[str] = []
    seen: Set[str] = set()
    for a in relevant_articles:
        m = _DIEU_CITE_RE.search(str(a))
        if not m:
            continue
        disp = normalize_text(m.group(0))
        key = disp.lower()
        if key not in seen:
            seen.add(key)
            out.append(disp)
        if len(out) >= limit:
            break
    return out


def apply_guardrails(
    answer: str,
    relevant_articles: Sequence[str],
    cfg: GuardrailConfig | None = None,
) -> Tuple[str, dict]:
    """Apply the post-generation guardrails to a raw answer.

    Returns ``(clean_answer, report)`` where ``report`` records what was done
    (useful for diagnostics / tests). The function is idempotent enough to be
    safe to run once per answer.
    """
    cfg = cfg or GuardrailConfig()
    report = {
        "stripped_echo": False,
        "had_citation": False,
        "backfilled_citation": False,
        "flagged_out_of_list": False,
        "appended_disclaimer": False,
    }

    cleaned = strip_prompt_echo(answer)
    report["stripped_echo"] = cleaned != normalize_text(answer)

    allowed = allowed_dieu_numbers(relevant_articles)
    mentioned = extract_dieu_citations(cleaned)
    report["had_citation"] = bool(mentioned & allowed)

    # FR-02: ensure at least one in-list citation.
    if cfg.ensure_citation and allowed and not (mentioned & allowed):
        backfill = _allowed_dieu_strings(relevant_articles, cfg.max_backfill_citations)
        if backfill:
            cleaned = (
                cleaned + "\n\nCăn cứ tham khảo: " + ", ".join(backfill) + "."
            ).strip()
            report["backfilled_citation"] = True

    # Grounding: flag invented citations outside the allowed list.
    if cfg.flag_out_of_list and allowed:
        invented = extract_dieu_citations(cleaned) - allowed
        if invented:
            cleaned = cleaned + "\n\n" + _HEDGE_NOTE
            report["flagged_out_of_list"] = True

    # FR-05: ensure a reference-only disclaimer is present.
    if cfg.ensure_disclaimer:
        low = cleaned.lower()
        if "tham khảo" not in low or "tư vấn" not in low:
            cleaned = cleaned + "\n\n" + cfg.disclaimer
            report["appended_disclaimer"] = True

    return cleaned.strip(), report
