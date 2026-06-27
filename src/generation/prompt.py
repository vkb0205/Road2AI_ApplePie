"""IRAC answer-prompt construction for G-LRAG generation (pure, stdlib-only).

The official grader extracts ``Điều X`` citations from the ``answer`` field and
the weekly QA criterion scores answer quality + grounding. This module builds a
Vietnamese IRAC-structured prompt (Issue / Rule / Application / Conclusion) from
the retrieved article passages and the allowed-citation list, so the generator
answers *only* from grounded context and always has the in-list ``Điều X`` to
cite.

It is deliberately dependency-free (no torch / transformers) so it imports and
unit-tests on a CPU-only box. The actual model call lives in
:mod:`generation.generator`; this module only assembles strings.

Design notes
------------
* **No hardcoded law ids / domains.** Unlike the legacy decomposition notebook,
  nothing here pins specific ``law_id`` values or topic tables — the prompt is
  built purely from the retrieved context, so it generalises to any question.
* **Context is capped** by character budget per passage and passage count, to
  keep the prompt within the model's input window.
* **Chat-template aware.** :func:`build_messages` returns a role/content list
  suitable for ``tokenizer.apply_chat_template``; :func:`render_plain` flattens
  it for tokenizers without a chat template.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

__all__ = [
    "PromptConfig",
    "format_context_block",
    "format_allowed_citations",
    "build_messages",
    "render_plain",
    "build_irac_prompt",
]


# Default system instruction: grounded, no fabrication, IRAC, always cite.
_SYSTEM_VI = (
    "Bạn là trợ lý pháp lý AI cho doanh nghiệp nhỏ và vừa tại Việt Nam. "
    "Chỉ trả lời dựa trên NGỮ CẢNH và DANH SÁCH ĐIỀU LUẬT ĐƯỢC PHÉP VIỆN DẪN "
    "được cung cấp. Tuyệt đối không bịa văn bản, số điều, khoản hoặc nguồn. "
    "Nếu ngữ cảnh chưa đủ căn cứ, hãy nói rõ là chưa đủ căn cứ."
)


@dataclass
class PromptConfig:
    """Knobs for IRAC prompt construction."""

    max_context_passages: int = 6     # how many retrieved passages to include
    max_chars_per_passage: int = 900  # truncate each passage to this many chars
    max_allowed_citations: int = 16   # cap the allowed-citation list length
    system_prompt: str = _SYSTEM_VI
    include_disclaimer_hint: bool = True  # ask the model to add the disclaimer


def format_context_block(
    contexts: Sequence[Dict[str, Any]], cfg: PromptConfig
) -> str:
    """Render retrieved passages into a numbered context block.

    Each context dict may carry ``law_id`` / ``ten_van_ban`` / ``dieu_so`` /
    ``chunk_text`` (the fields :class:`retrieval.retriever.Hit` and the
    doc-anchor candidates expose). Missing fields degrade to empty strings.
    """
    blocks: List[str] = []
    for i, c in enumerate(contexts[: cfg.max_context_passages], start=1):
        meta = " | ".join(
            str(c.get(k, "") or "") for k in ("law_id", "ten_van_ban", "dieu_so")
        )
        text = str(c.get("chunk_text", "") or "")[: cfg.max_chars_per_passage]
        blocks.append(f"[{i}] {meta}\n{text}")
    return "\n\n".join(blocks)


def format_allowed_citations(
    relevant_articles: Sequence[str], cfg: PromptConfig
) -> str:
    """Render the allowed-citation list (``law|ten|Điều`` strings) as bullets."""
    items = [str(a) for a in relevant_articles[: cfg.max_allowed_citations] if str(a).strip()]
    return "\n".join("- " + a for a in items)


def _user_prompt(
    question: str,
    contexts: Sequence[Dict[str, Any]],
    relevant_articles: Sequence[str],
    cfg: PromptConfig,
) -> str:
    allowed = format_allowed_citations(relevant_articles, cfg)
    ctx = format_context_block(contexts, cfg)
    lines = [
        f"Câu hỏi: {question}",
        "",
        "Danh sách điều luật được phép viện dẫn:",
        allowed if allowed else "(không có — hãy nêu rõ là chưa đủ căn cứ)",
        "",
        "Ngữ cảnh pháp lý:",
        ctx if ctx else "(không có ngữ cảnh)",
        "",
        "Yêu cầu trả lời:",
        "- Trả lời bằng tiếng Việt, ngắn gọn nhưng đủ ý.",
        "- Dùng cấu trúc IRAC ngắn: Vấn đề, Quy định, Áp dụng, Kết luận.",
        "- Luôn nêu \"Điều X\" khi có căn cứ trong danh sách được phép viện dẫn.",
        "- Không trích dẫn điều luật ngoài danh sách, không nhắc lại toàn bộ ngữ cảnh.",
    ]
    if cfg.include_disclaimer_hint:
        lines.append(
            "- Kết thúc bằng một câu lưu ý rằng thông tin chỉ mang tính tham khảo "
            "và không thay thế tư vấn pháp lý chính thức."
        )
    return "\n".join(lines)


def build_messages(
    question: str,
    contexts: Sequence[Dict[str, Any]],
    relevant_articles: Sequence[str],
    cfg: PromptConfig | None = None,
) -> List[Dict[str, str]]:
    """Build a chat ``[{role, content}, ...]`` message list for the generator."""
    cfg = cfg or PromptConfig()
    return [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": _user_prompt(question, contexts, relevant_articles, cfg)},
    ]


def render_plain(messages: Sequence[Dict[str, str]]) -> str:
    """Flatten a chat message list for tokenizers without a chat template."""
    parts: List[str] = []
    for m in messages:
        parts.append(f"{m.get('role', '')}:\n{m.get('content', '')}")
    parts.append("assistant:")
    return "\n\n".join(parts)


def build_irac_prompt(
    question: str,
    contexts: Sequence[Dict[str, Any]],
    relevant_articles: Sequence[str],
    cfg: PromptConfig | None = None,
) -> str:
    """Convenience: build the plain-rendered IRAC prompt in one call."""
    return render_plain(build_messages(question, contexts, relevant_articles, cfg))
