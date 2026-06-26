"""Idempotent patcher: add batch I/O (outside JSON -> output JSON) to
``retrieval_colab_decomposition.ipynb``.

Two edits, both guarded so re-running is safe:

  1. Section 1 config cell — add ``INPUT_QUERIES_PATH`` / ``OUTPUT_RESULTS_PATH``
     knobs so the batch input/output files are configurable from the top of the
     notebook.

  2. Section 8 batch cell — replace its source with a batch runner that:
       * reads queries from an OUTSIDE JSON file (``INPUT_QUERIES_PATH``); each
         record needs at least ``id`` + ``question`` (extra fields preserved);
       * runs the decomposition retriever for every question;
       * assembles a grounded ``answer`` from the retrieved passages
         (concatenated ``chunk_text`` + inline ``[N]`` citations);
       * fills ``relevant_docs`` / ``relevant_articles``;
       * writes the full records to a SEPARATE output file
         (``OUTPUT_RESULTS_PATH``);
       * optionally scores F2 when a ``ground_truth.json`` is available.

  3. Section 8 markdown header — updated to describe the new outside-JSON I/O.

Re-running is safe: every insertion is guarded by a marker check.
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
# Block 1 — batch I/O config knobs (inserted into the Section 1 config cell)
# ---------------------------------------------------------------------------
BATCH_CONFIG = [
    "\n",
    "# --- Batch I/O (Section 8) --------------------------------------------\n",
    "# Path to an OUTSIDE JSON file of queries to run in batch. Each record needs\n",
    "# at least \"id\" and \"question\"; any extra fields are preserved in the output.\n",
    "# Leave '' to fall back to dev_set/questions.json (runnable out of the box).\n",
    "INPUT_QUERIES_PATH  = \"\"                         # e.g. \"/content/drive/.../my_queries.json\"\n",
    "# Path to the SEPARATE output file the notebook writes the results to.\n",
    "OUTPUT_RESULTS_PATH = \"/content/results_batch_decomposition.json\"\n",
]

# ---------------------------------------------------------------------------
# Block 2 — new Section 8 batch cell source
# ---------------------------------------------------------------------------
BATCH_CELL_SRC = '''# ======================================================================
# 8. Batch run — OUTSIDE JSON input  ->  SEPARATE output JSON file
# ======================================================================
# Reads queries from `INPUT_QUERIES_PATH` (an outside JSON file with the same
# shape as dev_set/questions.json: a list of {"id", "question", ...} records),
# runs the decomposition retriever for every question, assembles a grounded
# "answer" from the retrieved passages (concatenated chunk_text + inline [N]
# citations), fills relevant_docs / relevant_articles, and writes the full
# records to `OUTPUT_RESULTS_PATH`.
#
# Any extra fields already present on an input record (e.g. a pre-filled
# "answer") are preserved — only id + question are required. The retrieval-
# derived fields (answer, relevant_docs, relevant_articles) are always (re)built
# from the current retriever run.
import json as _json
from pathlib import Path

# --- resolve input / output paths -------------------------------------------
# Fall back to the dev_set questions.json when no explicit input file is given,
# so this cell is always runnable out of the box.
_INPUT_Q  = Path(INPUT_QUERIES_PATH) if str(INPUT_QUERIES_PATH).strip() else None
_OUTPUT_R = Path(OUTPUT_RESULTS_PATH)

if _INPUT_Q is None or not _INPUT_Q.exists():
    _DEV_LOCAL = Path(REPO_DIR) / "dev_set"
    _cand = (DEV / "questions.json") if (DEV / "questions.json").exists() \\
            else (_DEV_LOCAL / "questions.json")
    _INPUT_Q = _cand
    print(f"[batch] INPUT_QUERIES_PATH not set / missing -> using {_INPUT_Q}")
else:
    print(f"[batch] input  : {_INPUT_Q}")
print(f"[batch] output : {_OUTPUT_R}")

assert _INPUT_Q.exists(), f"input query file not found: {_INPUT_Q}"
_OUTPUT_R.parent.mkdir(parents=True, exist_ok=True)


# --- answer builder: concatenate retrieved passages + inline citations -------
def _build_answer(hits, max_chars_per_hit=2000):
    """Retrieval-only answer (no generation LLM).

    Each hit's ``chunk_text`` is followed by a
    ``[N] {law_id} | {ten_van_ban} | {dieu_so}`` citation marker so every
    sentence can be traced back to its source. Matches the grader-expected
    record shape (id / question / answer / relevant_docs / relevant_articles).
    """
    parts = []
    for i, h in enumerate(hits, 1):
        text = (getattr(h, "chunk_text", "") or "").strip()
        if not text:
            continue
        if max_chars_per_hit and len(text) > max_chars_per_hit:
            text = text[:max_chars_per_hit] + " \\u2026[truncated]"
        cite = f"[{i}] {getattr(h, 'law_id', '') or ''}"
        ten = str(getattr(h, "ten_van_ban", "") or "").strip()
        if ten:
            cite += f" | {ten}"
        dieu = str(getattr(h, "dieu_so", "") or "").strip()
        if dieu:
            cite += f" | {dieu}"
        parts.append(f"{text} {cite}")
    return " ".join(parts).strip()


# --- run the batch -----------------------------------------------------------
records_in = _json.loads(_INPUT_Q.read_text(encoding="utf-8"))
print(f"[batch] loaded {len(records_in)} queries")

records_out, t0 = [], time.time()
for rec in records_in:
    qid   = rec.get("id")
    query = rec.get("question") or rec.get("query") or ""
    if not query:
        print(f"  [skip] id={qid} has no question text")
        continue
    hits = decomp_retriever.retrieve(query, fetch_text=True)
    docs, articles = make_relevant_lists(hits)
    out = dict(rec)                       # preserve any extra fields from the input
    out["id"]                = qid
    out["question"]          = query
    out["answer"]            = _build_answer(hits)
    out["relevant_docs"]     = docs
    out["relevant_articles"] = articles
    records_out.append(out)
    print(f"  [done] id={qid}  hits={len(hits)}  docs={len(docs)}  articles={len(articles)}")

_OUTPUT_R.write_text(_json.dumps(records_out, ensure_ascii=False, indent=2),
                     encoding="utf-8")
print(f"\\n[batch] wrote {len(records_out)} records to {_OUTPUT_R} "
      f"in {time.time()-t0:.1f}s")

# --- (optional) F2 score if a ground-truth file is available -----------------
try:
    _DEV_LOCAL = Path(REPO_DIR) / "dev_set"
    _GTPATH = ((DEV / "ground_truth.json") if (DEV / "ground_truth.json").exists()
               else (_DEV_LOCAL / "ground_truth.json"))
    if _GTPATH.exists():
        import sys as _sys
        if str(_DEV_LOCAL.parent) not in _sys.path:
            _sys.path.insert(0, str(_DEV_LOCAL.parent))
        from dev_set.eval import f2_macro
        _gt = _json.loads(_GTPATH.read_text(encoding="utf-8"))
        print(f"\\nF2 macro (decomposition, batch) = "
              f"{f2_macro(records_out, _gt):.4f}  [gt={_GTPATH}]")
    else:
        print("\\n(ground_truth.json not found - skipping F2 score)")
except Exception as _e:
    print(f"\\n(F2 scoring skipped: {_e})")
'''
BATCH_CELL = BATCH_CELL_SRC.splitlines(keepends=True)


# ---------------------------------------------------------------------------
# Block 3 — updated Section 8 markdown header
# ---------------------------------------------------------------------------
SECTION8_MD = [
    "## 8. Batch run — outside JSON input → separate output file\n",
    "\n",
    "Reads a list of queries from an **outside JSON file**\n",
    "(``INPUT_QUERIES_PATH``, set in [§1](#)) — each record needs at least ``id``\n",
    "and ``question`` (extra fields are preserved — falls back to\n",
    "`dev_set/questions.json` when left blank). For every question it runs the\n",
    "decomposition retrieval, assembles a grounded ``answer`` from the retrieved\n",
    "passages (concatenated ``chunk_text`` + inline ``[N]`` citations), fills\n",
    "``relevant_docs`` / ``relevant_articles``, and writes the full records to a\n",
    "**separate output file** (``OUTPUT_RESULTS_PATH``).\n",
    "\n",
    "The output record shape matches the grader contract:\n",
    "`id` / `question` / `answer` / `relevant_docs` / `relevant_articles`.\n",
    "\n",
    "> When a `ground_truth.json` is available, an F2 macro score is printed at the\n",
    "> end. Compare it against the baseline / advanced notebooks to measure the\n",
    "> gain from the pre-retrieval decomposition layer.\n",
]


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
for cell in nb["cells"]:
    src = _src(cell)
    if not src:
        continue
    joined = "".join(src)

    # 1) Section 1 config cell (identified by GITHUB_REPO): insert batch I/O knobs.
    if "GITHUB_REPO = " in joined and "INPUT_QUERIES_PATH" not in joined:
        for i, s in enumerate(src):
            if s.startswith("HF_CACHE_DIR = "):
                if not any("INPUT_QUERIES_PATH" in s2 for s2 in src):
                    src[i + 1:i + 1] = BATCH_CONFIG
                break

    # 2) Section 8 code cell (identified by the old output path): replace source.
    if 'results_colab_decomposition.json' in joined and "_build_answer" not in joined:
        cell["source"] = BATCH_CELL
        cell["outputs"] = []
        cell["execution_count"] = None

    # 3) Section 8 markdown header (identified by the old heading text).
    if "Batch-run the whole dev set" in joined and "outside JSON" not in joined:
        cell["source"] = SECTION8_MD

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print("patched OK")
