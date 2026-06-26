"""Idempotent patcher for retrieval_colab_decomposition.ipynb.

Adds three notebook-local improvements WITHOUT touching router/retrieval core:
  A. Tighten DECOMPOSITION_PROMPT via module-attribute override (fuller,
     self-contained, legal-anchored sub-queries + examples).
  B. Lightweight pre-retrieval cleanup helpers (clean_subquery / anchor
     backfill / normalize_subqueries) defined in the retriever cell.
  C. Apply that cleanup right after router.route() and before the
     base.retrieve(sub_query) loop.

Re-running is safe: each insertion is guarded by a marker check.
"""
import json

NB = "Road2AI_ApplePie/notebooks/retrieval_colab_decomposition.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


def _src(cell):
    s = cell.get("source", [])
    if isinstance(s, str):
        s = s.splitlines(keepends=True)
        cell["source"] = s
    return s


# ---------------------------------------------------------------------------
# Block A — tighten the decomposition prompt (notebook-local override)
# ---------------------------------------------------------------------------
prompt_override = [
    "import retrieval.sub_query_router as _sqr_mod\n",
    "_EXTRA_DECOMPOSITION_RULES = \"\"\"\\\n",
    "\n",
    "Quy tắc tách sub-query (bổ sung):\n",
    "- Mỗi sub-query phải là một truy vấn hoàn chỉnh, không phải mảnh câu.\n",
    "- Mỗi sub-query phải giữ nguyên ngữ cảnh pháp lý và subject chính của ý đó.\n",
    "- Nếu sub-query bị quá ngắn hoặc quá chung chung, hãy mở rộng bằng cách thêm đối tượng, hành vi, hoặc bối cảnh pháp lý liên quan.\n",
    "- Không được tạo sub-query chỉ là cụm danh từ hay câu cụt.\n",
    "- Ưu tiên dạng: [hành vi / vấn đề pháp lý] + [đối tượng / bối cảnh] + [mục tiêu truy vấn].\n",
    "\n",
    "Ví dụ sub-query tốt (đầy đủ, độc lập, có legal anchor):\n",
    "- Xác định hành vi xâm phạm quyền tác giả khi đối thủ sao chép trái phép phần mềm để cho thuê thu lợi.\n",
    "- Cần chuẩn bị tài liệu và chứng cứ gì khi gửi đơn yêu cầu xử lý hành vi xâm phạm quyền tác giả đối với phần mềm.\n",
    "- Cách tính thiệt hại và mất cơ hội kinh doanh do hành vi xâm phạm quyền tác giả đối với phần mềm.\"\"\"\n",
    "_sqr_mod.DECOMPOSITION_PROMPT = _sqr_mod.DECOMPOSITION_PROMPT.replace(\n",
    "    \"Quy tắc:\", _EXTRA_DECOMPOSITION_RULES + \"\\n\\nQuy tắc:\", 1)\n",
    "# Keep the notebook-local import name in sync too.\n",
    "DECOMPOSITION_PROMPT = _sqr_mod.DECOMPOSITION_PROMPT\n",
    "print(\"[router] DECOMPOSITION_PROMPT tightened (notebook-local override)\")\n",
]

# ---------------------------------------------------------------------------
# Block B — pre-retrieval cleanup helpers
# ---------------------------------------------------------------------------
helpers = [
    "import re as _re\n",
    "\n",
    "\n",
    "# --- Pre-retrieval sub-query cleanup (lightweight, no split-logic change) --\n",
    "# Applied right after router.route() and before base.retrieve(sub_query) to\n",
    "# stabilise the text the retriever sees: trim/collapse whitespace, drop\n",
    "# fragments & near-duplicates, and backfill a minimal legal anchor from the\n",
    "# original query when a sub-query is too generic. This protects the pipeline\n",
    "# from 'câu cụt' / noun-phrase sub-queries regardless of router source.\n",
    "_LEGAL_ANCHOR_TERMS = (\n",
    "    \"quyền tác giả\", \"phần mềm\", \"chứng cứ\", \"thiệt hại\",\n",
    "    \"hành vi\", \"xâm phạm\", \"bồi thường\", \"trách nhiệm\",\n",
    "    \"hợp đồng\", \"sở hữu trí tuệ\", \"tổn thất\",\n",
    ")\n",
    "_ANCHOR_RE = _re.compile(\"|\".join(_re.escape(t) for t in _LEGAL_ANCHOR_TERMS),\n",
    "                        _re.IGNORECASE)\n",
    "\n",
    "\n",
    "def clean_subquery(q: str) -> str:\n",
    "    \"\"\"Trim + collapse whitespace + strip stray leading/trailing punctuation.\"\"\"\n",
    "    q = \" \".join(q.strip().split())\n",
    "    q = q.strip(\" .;,:-–—\")\n",
    "    return q\n",
    "\n",
    "\n",
    "def _ensure_legal_anchor(q: str, context: str) -> str:\n",
    "    \"\"\"If a sub-query lacks any core legal phrase, graft the first anchor term\n",
    "    found in the original query context onto the sub-query so the retriever\n",
    "    has a legal hook. No-op if the sub-query already carries an anchor or the\n",
    "    context has none.\"\"\"\n",
    "    if not q or _ANCHOR_RE.search(q):\n",
    "        return q\n",
    "    m = _ANCHOR_RE.search(context or \"\")\n",
    "    if not m:\n",
    "        return q\n",
    "    anchor = m.group(0).lower()\n",
    "    q = q.rstrip(\" .\")\n",
    "    return f\"{q} ({anchor})\"\n",
    "\n",
    "\n",
    "def normalize_subqueries(subqueries, context=\"\", min_chars=10):\n",
    "    \"\"\"Lightweight normalisation applied right before retrieval:\n",
    "      - clean whitespace / stray punctuation\n",
    "      - drop sub-queries shorter than min_chars (fragments / câu cụt)\n",
    "      - backfill a minimal legal anchor from context when too generic\n",
    "      - dedup near-identical sub-queries (case-insensitive)\n",
    "    Returns a list of stable, self-contained sub-query strings.\"\"\"\n",
    "    out, seen = [], set()\n",
    "    for q in subqueries:\n",
    "        q = clean_subquery(q)\n",
    "        if len(q) < min_chars:\n",
    "            continue\n",
    "        q = _ensure_legal_anchor(q, context)\n",
    "        key = \" \".join(q.lower().split())\n",
    "        if key in seen:\n",
    "            continue\n",
    "        seen.add(key)\n",
    "        out.append(q)\n",
    "    return out\n",
]

# ---------------------------------------------------------------------------
# Block C — apply cleanup inside DecomposingHybridRetriever.retrieve()
# ---------------------------------------------------------------------------
cleanup_apply = [
    "            # Pre-retrieval cleanup: normalise each sub-query, drop fragments /\n",
    "            # duplicates, backfill a minimal legal anchor from the original query\n",
    "            # context. Does NOT change the router's split logic — only stabilises\n",
    "            # the text the retriever actually sees.\n",
    "            cleaned = normalize_subqueries(decision.sub_queries, context=query,\n",
    "                                           min_chars=MIN_SUB_QUERY_CHARS)\n",
    "            decision.sub_queries = cleaned\n",
    "            decision.num_sub_queries = len(cleaned)\n",
]

# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
for cell in nb["cells"]:
    src = _src(cell)
    if not src:
        continue

    # Block A: router-build cell.
    if any("from retrieval.sub_query_router import" in s for s in src) and \
       any("def _build_llm_call" in s for s in src):
        for i, s in enumerate(src):
            if s.strip() == "DECOMPOSITION_PROMPT,":
                assert src[i + 1].strip() == ")", src[i + 1]
                if not any("_EXTRA_DECOMPOSITION_RULES" in s2 for s2 in src):
                    src[i + 2:i + 2] = prompt_override
                break

    # Blocks B & C: retriever cell.
    if any("class DecomposingHybridRetriever" in s for s in src):
        # B: helpers after the defaultdict import.
        for i, s in enumerate(src):
            if s.strip() == "from collections import defaultdict":
                if not any("def clean_subquery" in s2 for s2 in src):
                    src[i + 1:i + 1] = helpers
                break
        # C: cleanup right after `self.last_decision = decision`.
        for i, s in enumerate(src):
            if s.strip() == "self.last_decision = decision":
                if not any("cleaned = normalize_subqueries" in s2 for s2 in src):
                    src[i + 1:i + 1] = cleanup_apply
                break

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print("patched OK")
