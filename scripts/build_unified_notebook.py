"""Generate notebooks/retrieval_kaggle_unified.ipynb from clean cell sources.

Run once (CPU, no heavy deps) to (re)build the unified Kaggle notebook:

    python scripts/build_unified_notebook.py

Keeping the notebook authored as Python strings here avoids hand-editing
fragile .ipynb JSON and lets us syntax-check every code cell before writing.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

CELLS = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(text: str) -> None:
    CELLS.append(("code", text))


# --------------------------------------------------------------------------- #
md(
    "# 🧩 G-LRAG — Unified Retrieval + IRAC Generation (Kaggle)\n"
    "\n"
    "A single, maintainable pipeline that supersedes the ~7700-line decomposition\n"
    "notebook by composing the **unit-tested** `src/` building blocks:\n"
    "\n"
    "| Stage | Module | What it does |\n"
    "|---|---|---|\n"
    "| Retrieve | `retrieval.doc_anchor` | FTS + FAISS → RRF doc-anchor → harvest → cross-encoder rerank |\n"
    "| Decompose (opt) | `retrieval.sub_query_router` | rule-based split of multi-clause questions |\n"
    "| Select | `retrieval.article_select` | authority prior by document **type** + adaptive K |\n"
    "| Generate | `generation.generator` | grounded IRAC answer, always cites an in-list `Điều X` |\n"
    "| Submit | `retrieval.unified_pipeline` | grader record + schema validation + `submission.zip` |\n"
    "\n"
    "**What changed vs the decomposition notebook**\n"
    "- Drops all hardcoded `preferred_law_ids` / domain / facet tables (the main\n"
    "  generalisation risk — those were overfit to specific dev questions).\n"
    "- One 4-bit Qwen load for generation (no router+planner+generator triple load).\n"
    "- Reuses the validated document-anchored retrieval that puts the gold article\n"
    "  at rerank rank 0 on clean questions.\n"
)

md("## 1. Configuration")

code(
    '# ===== Configuration (edit these) =====================================\n'
    'GITHUB_REPO = "https://github.com/vkb0205/Road2AI_ApplePie.git"\n'
    'REPO_BRANCH = "new_retrieval"\n'
    'REPO_DIR    = "/kaggle/working/Road2AI_ApplePie"\n'
    '\n'
    'KAGGLE_DATASET = "vkb0205/stage6-data"   # <-- your dataset slug\n'
    'DATA_DIR = f"/kaggle/input/{KAGGLE_DATASET.split(\'/\')[-1]}"\n'
    'DEV_DIR  = f"{REPO_DIR}/dev_set"\n'
    '\n'
    '# --- Pipeline switches -------------------------------------------------\n'
    'USE_DENSE         = True    # FAISS + BGE-m3 (GPU). Required for recall.\n'
    'USE_RERANK        = True    # cross-encoder BAAI/bge-reranker-v2-m3 (GPU).\n'
    'USE_DECOMPOSITION = False   # rule-based sub-query split (no LLM router)\n'
    'USE_GENERATOR     = True    # IRAC answer generation (one 4-bit Qwen)\n'
    'FTS_MODE          = "bm25_ranked"\n'
    'GPU_ID            = 0\n'
    '\n'
    '# --- Retrieval knobs (DocAnchorConfig) --------------------------------\n'
    'TOP_BM25           = 300\n'
    'TOP_DENSE          = 150\n'
    'RRF_K              = 60\n'
    'TOP_DOCS           = 40\n'
    'PER_DOC_ARTICLES   = 12\n'
    'RERANK_POOL        = 120\n'
    'CHUNKS_PER_ARTICLE = 4\n'
    '\n'
    '# --- Selection knobs (SelectConfig) -----------------------------------\n'
    'DROP_PROVINCIAL = True\n'
    'MAX_K           = 3\n'
    'MIN_K           = 1\n'
    'REL_MARGIN      = 0.18\n'
    'ABS_MARGIN      = 0.12\n'
    '\n'
    '# --- Generation knobs --------------------------------------------------\n'
    'GEN_MODEL        = "Qwen/Qwen2.5-7B-Instruct"\n'
    'GEN_LOAD_IN_4BIT = True\n'
    'GEN_CONTEXT_TOPK = 6\n'
    'MAX_SUB_QUERIES  = 4\n'
    '\n'
    '# --- Input / output ----------------------------------------------------\n'
    '# Point INPUT_QUERIES at the competition test set ({id, question} list).\n'
    '# Defaults to the dev questions so the notebook runs out of the box.\n'
    'INPUT_QUERIES       = ""   # e.g. "/kaggle/input/<dataset>/test.json"\n'
    'RESULTS_PATH        = "/kaggle/working/results.json"\n'
    'SUBMISSION_ZIP_PATH = "/kaggle/working/submission.zip"\n'
    'GROUND_TRUTH        = f"{DEV_DIR}/ground_truth.json"  # optional local F2\n'
    '# ======================================================================\n'
    '\n'
    'import subprocess, sys, os\n'
    'from pathlib import Path\n'
    '\n'
    'if USE_DENSE or USE_RERANK or USE_GENERATOR:\n'
    '    gpu_ok = subprocess.run("nvidia-smi", shell=True,\n'
    '                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0\n'
    '    if not gpu_ok:\n'
    '        raise SystemExit("\\u274c No GPU detected. Settings \\u2192 Accelerator \\u2192 GPU, or disable USE_DENSE/USE_RERANK/USE_GENERATOR.")\n'
    '    print("[gpu] confirmed")\n'
)

md("## 2. Clone repo + install deps")

code(
    'if not Path(REPO_DIR).exists():\n'
    '    print(f"[clone] {GITHUB_REPO} ({REPO_BRANCH})")\n'
    '    get_ipython().system("git clone --depth 1 -b {REPO_BRANCH} {GITHUB_REPO} {REPO_DIR}")\n'
    'else:\n'
    '    print(f"[clone] {REPO_DIR} present; fetching {REPO_BRANCH}")\n'
    '    get_ipython().system("cd {REPO_DIR} && git fetch --depth 1 origin {REPO_BRANCH} && git reset --hard origin/{REPO_BRANCH}")\n'
    '\n'
    'SRC = str(Path(REPO_DIR) / "src")\n'
    'if SRC not in sys.path:\n'
    '    sys.path.insert(0, SRC)\n'
    'if REPO_DIR not in sys.path:\n'
    '    sys.path.insert(0, REPO_DIR)\n'
    'print("[src on path]", SRC)\n'
    '\n'
    'get_ipython().system("pip -q install hf_transfer pandas pyarrow networkx pyyaml psutil")\n'
    'os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"\n'
    'if USE_DENSE or USE_RERANK:\n'
    '    get_ipython().system("pip -q install faiss-gpu-cu12 \'FlagEmbedding>=1.2.10,<1.3\' \'transformers>=4.41,<4.46\' || pip -q install faiss-cpu")\n'
    'if USE_GENERATOR and GEN_LOAD_IN_4BIT:\n'
    '    get_ipython().system("pip -q install bitsandbytes accelerate")\n'
    'print("[deps] ready")\n'
)

md(
    "## 3. Build the unified pipeline\n"
    "\n"
    "`build_unified_pipeline` wires the document-anchored retriever, optional\n"
    "rule-based decomposition, the F2 selector, and (when enabled) the IRAC\n"
    "generator with an FTS-backed text provider — all from the Stage-6 bundle."
)

code(
    'from retrieval.kaggle_pipeline import build_unified_pipeline, dump_json\n'
    'from retrieval.unified_pipeline import run_dev_set, validate_submission\n'
    'from retrieval.doc_anchor import DocAnchorConfig\n'
    'from retrieval.article_select import SelectConfig\n'
    '\n'
    'anchor_cfg = DocAnchorConfig(\n'
    '    top_bm25=TOP_BM25, top_dense=(TOP_DENSE if USE_DENSE else 0),\n'
    '    rrf_k=RRF_K, top_docs=TOP_DOCS,\n'
    '    per_doc_articles=PER_DOC_ARTICLES, rerank_pool=RERANK_POOL,\n'
    '    chunks_per_article=CHUNKS_PER_ARTICLE,\n'
    ')\n'
    'select_cfg = SelectConfig(\n'
    '    drop_provincial=DROP_PROVINCIAL, max_k=MAX_K, min_k=MIN_K,\n'
    '    rel_margin=REL_MARGIN, abs_margin=ABS_MARGIN,\n'
    ')\n'
    '\n'
    'pipe, fts = build_unified_pipeline(\n'
    '    DATA_DIR,\n'
    '    use_dense=USE_DENSE, use_rerank=USE_RERANK,\n'
    '    use_decomposition=USE_DECOMPOSITION, use_generator=USE_GENERATOR,\n'
    '    fts_mode=FTS_MODE, gpu_id=GPU_ID,\n'
    '    anchor_cfg=anchor_cfg, select_cfg=select_cfg,\n'
    '    gen_model=GEN_MODEL, gen_load_in_4bit=GEN_LOAD_IN_4BIT,\n'
    '    max_sub_queries=MAX_SUB_QUERIES, gen_context_topk=GEN_CONTEXT_TOPK,\n'
    ')\n'
    'print("[pipeline] built | dense=%s rerank=%s decompose=%s generator=%s"\n'
    '      % (USE_DENSE, USE_RERANK, USE_DECOMPOSITION, USE_GENERATOR))\n'
)

md("## 4. Smoke test — one question")

code(
    '_demo = pipe.answer_record(0, "Hồ sơ đăng ký doanh nghiệp gồm những giấy tờ gì cho công ty TNHH một thành viên?")\n'
    'print("relevant_articles:", _demo["relevant_articles"])\n'
    'print("relevant_docs    :", _demo["relevant_docs"])\n'
    'print("\\nanswer:\\n", _demo["answer"][:1200])\n'
)

md(
    "## 5. Batch run → results.json\n"
    "\n"
    "Runs the pipeline over the input questions (the competition test set when\n"
    "`INPUT_QUERIES` is set, else the dev questions). Writes `results.json`\n"
    "incrementally so a crashed kernel can be inspected."
)

code(
    'import json, time\n'
    '\n'
    '_inp = Path(INPUT_QUERIES) if str(INPUT_QUERIES).strip() else Path(f"{DEV_DIR}/questions.json")\n'
    'assert _inp.exists(), f"input questions not found: {_inp}"\n'
    'print("[batch] input:", _inp)\n'
    '\n'
    '_t0 = time.time()\n'
    'records = run_dev_set(pipe, str(_inp))\n'
    'dump_json(records, RESULTS_PATH)\n'
    'print(f"[batch] {len(records)} records in {time.time()-_t0:.1f}s -> {RESULTS_PATH}")\n'
    '\n'
    '# Free GPU after the batch.\n'
    'try:\n'
    '    import torch, gc\n'
    '    gc.collect()\n'
    '    if torch.cuda.is_available():\n'
    '        torch.cuda.empty_cache()\n'
    'except Exception:\n'
    '    pass\n'
)

md(
    "## 6. Validate + package submission.zip\n"
    "\n"
    "Validates `results.json` against the grader contract (5-field schema,\n"
    "`relevant_docs`/`relevant_articles` formats, id coverage, and — when a\n"
    "generator ran — that each answer cites an in-list `Điều X`), then zips it\n"
    "with arcname `results.json`."
)

code(
    'import json, zipfile\n'
    '\n'
    '_data = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))\n'
    '_expected = [r.get("id") for r in _data]\n'
    'summary = validate_submission(\n'
    '    _data, expected_ids=_expected, require_answer_citation=USE_GENERATOR\n'
    ')\n'
    'print("[validate]", summary)\n'
    '\n'
    'with zipfile.ZipFile(SUBMISSION_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:\n'
    '    zf.write(RESULTS_PATH, arcname="results.json")\n'
    'print("results   :", RESULTS_PATH)\n'
    'print("submission:", SUBMISSION_ZIP_PATH)\n'
)

md(
    "## 7. (Optional) Local F2 — diagnostic only\n"
    "\n"
    "> The dev-set ground truth is known to be unreliable (contradictory labels,\n"
    "> generic `Điều 1` golds), so treat this number as a smoke test, **not** a\n"
    "> target. The trustworthy signal is the public leaderboard."
)

code(
    'import json\n'
    'from pathlib import Path\n'
    '_gt = Path(GROUND_TRUTH)\n'
    'if _gt.exists():\n'
    '    get_ipython().system("cd {REPO_DIR} && python dev_set/f2_diagnostics.py --ground-truth dev_set/ground_truth.json --predictions {RESULTS_PATH}")\n'
    'else:\n'
    '    print("(ground_truth.json not found — skipping local F2)")\n'
)

# --------------------------------------------------------------------------- #
# Validate every code cell parses, then write the notebook.
# --------------------------------------------------------------------------- #
for kind, src in CELLS:
    if kind == "code":
        # Replace IPython-only get_ipython() shell calls with `pass` (keeping
        # indentation) so the ast check sees valid blocks, not empty if/else.
        lines = []
        for l in src.splitlines():
            if "get_ipython()" in l:
                indent = l[: len(l) - len(l.lstrip())]
                lines.append(indent + "pass")
            else:
                lines.append(l)
        check = "\n".join(lines)
        try:
            ast.parse(check)
        except SyntaxError as e:
            raise SystemExit(f"Syntax error in a code cell:\n{e}\n---\n{src}")

nb = {
    "cells": [
        {
            "cell_type": kind,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        }
        for kind, src in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "retrieval_kaggle_unified.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out}  ({len(CELLS)} cells)")
