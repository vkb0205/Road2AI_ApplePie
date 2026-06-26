"""Build the combined decomposition + generation + submission notebook.

Reads  notebooks/retrieval_colab_decomposition.ipynb  (the decomposition +
hybrid-retrieval + explainability vehicle, §1-§7) and appends the missing
generation + submission modules ported from notebooks/hybridrag-decomp.ipynb:

  §8  Generation + batch run  -> QwenIRACGenerator (IRAC prompt) +
      citation postprocessing + batch loop over INPUT_QUERIES_PATH +
      optional F2 score.
  §9  Validate + zip submission -> schema/id validation + submission.zip
      containing results.json (grader contract).
  §10 Cleanup -> extended teardown (also releases the generator).

Non-destructive: writes a NEW file
  notebooks/retrieval_colab_decomp_submit.ipynb
and leaves both source notebooks untouched.

Run:  python Road2AI_ApplePie/scratch/_build_decomp_submit_nb.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NB = ROOT / "notebooks" / "retrieval_colab_decomposition.ipynb"
OUT_NB = ROOT / "notebooks" / "retrieval_colab_decomp_submit.ipynb"

nb = json.loads(SRC_NB.read_text(encoding="utf-8"))


def _src(cell):
    s = cell.get("source", [])
    if isinstance(s, str):
        s = s.splitlines(keepends=True)
    return s


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


# ---------------------------------------------------------------------------
# 1. Find the section boundaries in the source notebook.
#    Keep everything up to (but not including) the "## 8." markdown header,
#    then drop the old retrieval-only §8 batch + §9 cleanup and replace them.
# ---------------------------------------------------------------------------
cut = None
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "markdown":
        continue
    head = "".join(_src(cell))
    if head.lstrip().startswith("## 8."):
        cut = i
        break

assert cut is not None, "Could not find '## 8.' markdown header in source notebook"
kept = nb["cells"][:cut]
print(f"[build] keeping {len(kept)} source cells (§1-§7 + title)")

# Tack a "combined edition" banner onto the first (title) markdown cell so the
# new file is self-documenting, without rewriting the whole title.
if kept and kept[0].get("cell_type") == "markdown":
    title_src = _src(kept[0])
    banner = (
        "\n\n> **Combined edition** — this notebook merges "
        "[`retrieval_colab_decomposition.ipynb`](notebooks/retrieval_colab_decomposition.ipynb)\n"
        "(query decomposition + hybrid retrieval + explainability, §1-§7) with the generation +\n"
        "submission modules from [`hybridrag-decomp.ipynb`](notebooks/hybridrag-decomp.ipynb)\n"
        "(IRAC answer generator, citation postprocessing, batch processing, `submission.zip`, §8-§10).\n"
        "The decomposition + retrieval strategy is inherited unchanged from the decomposition notebook.\n"
    )
    title_src = list(title_src) + banner.splitlines(keepends=True)
    kept[0]["source"] = title_src

# ---------------------------------------------------------------------------
# 2. §8 — Generation + batch run
# ---------------------------------------------------------------------------
sec8_md = md(
    "## 8. Generation + batch run → `results.json`\n"
    "\n"
    "This section adds the **missing generation module** ported from\n"
    "[`hybridrag-decomp.ipynb`](notebooks/hybridrag-decomp.ipynb): an IRAC-structured\n"
    "[`QwenIRACGenerator`](notebooks/hybridrag-decomp.ipynb) that answers each question\n"
    "from the retrieved passages, plus citation postprocessing that keeps the answer grounded\n"
    "in the `relevant_articles` produced by [`make_relevant_lists`](src/retrieval/retriever.py).\n"
    "\n"
    "The decomposition + retrieval strategy is **inherited unchanged** from §4-§5: for every\n"
    "query we call `decomp_retriever.retrieve(query, fetch_text=True)`, then feed the returned\n"
    "[`Hit`](src/retrieval/retriever.py) list (which carries `law_id` / `ten_van_ban` / `dieu_so` /\n"
    "`chunk_text`) into the generator. The output record matches the grader contract:\n"
    "`id` / `question` / `answer` / `relevant_docs` / `relevant_articles`.\n"
    "\n"
    "> Set `USE_GENERATOR = False` to skip the LLM and leave `answer` empty (retrieval-only,\n"
    "> identical to the original decomposition notebook's §8).\n"
)

sec8_cfg = code(
    'import os, re, gc, json as _json, zipfile, time\n'
    'from pathlib import Path\n'
    '\n'
    '# --- Generation config (mirrors hybridrag-decomp.ipynb CFG) ---------------\n'
    '# The decomposition/retrieval knobs live in §1; these govern the answer LLM.\n'
    'GEN_MODEL            = os.environ.get("GEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")\n'
    'USE_GENERATOR        = True                # False -> skip LLM, leave answer empty\n'
    'GEN_LOAD_IN_4BIT     = os.environ.get("GEN_LOAD_IN_4BIT", "0") == "1"\n'
    'GEN_MAX_INPUT_TOKENS = int(os.environ.get("GEN_MAX_INPUT_TOKENS", "3072"))\n'
    'GEN_MAX_NEW_TOKENS   = int(os.environ.get("GEN_MAX_NEW_TOKENS", "448"))\n'
    'GEN_TEMPERATURE      = float(os.environ.get("GEN_TEMPERATURE", "0.1"))\n'
    'GEN_TOP_P            = float(os.environ.get("GEN_TOP_P", "0.9"))\n'
    'GEN_CHUNK_CHAR_LIMIT = int(os.environ.get("GEN_CHUNK_CHAR_LIMIT", "900"))\n'
    'GEN_CONTEXT_TOPK     = int(os.environ.get("GEN_CONTEXT_TOPK", "6"))\n'
    'GEN_USE_CACHE        = os.environ.get("GEN_USE_CACHE", "1") == "1"\n'
    'ARTICLE_CONTEXT_TOPK = int(os.environ.get("ARTICLE_CONTEXT_TOPK", "16"))\n'
    '\n'
    '# --- Submission output paths (grader contract: results.json inside the zip) -\n'
    'RESULTS_PATH        = "/content/results.json"\n'
    'SUBMISSION_ZIP_PATH = "/content/submission.zip"\n'
    '\n'
    'print({"gen_model": GEN_MODEL, "use_generator": USE_GENERATOR,\n'
    '       "load_in_4bit": GEN_LOAD_IN_4BIT, "gen_context_topk": GEN_CONTEXT_TOPK,\n'
    '       "results_path": RESULTS_PATH, "submission_zip_path": SUBMISSION_ZIP_PATH})\n'
)

sec8_citations = code(
    '# --- citation postprocessing (ported from hybridrag-decomp.ipynb) ----------\n'
    '# Keeps the generated answer grounded in the relevant_articles produced by\n'
    '# make_relevant_lists(): flags out-of-list article citations and backfills\n'
    '# top citations when the model cites none.\n'
    '\n'
    'def _normalize_text(text):\n'
    '    text = str(text).replace("\\u200b", " ").replace("\\ufeff", " ")\n'
    '    return re.sub(r"\\s+", " ", text).strip()\n'
    '\n'
    '\n'
    'def strip_leading_prompt_echo(text):\n'
    '    """Remove chat-template scaffolding the model may echo at the start."""\n'
    '    text = _normalize_text(text)\n'
    '    for pat in (r"(?is)^system\\s*.*?user\\s*", r"(?is)^assistant\\s*",\n'
    '                r"(?is)^trả lời\\s*:?"):\n'
    '        text = re.sub(pat, "", text).strip()\n'
    '    return re.sub(r"^\\s*[-–—]\\s*", "", text).strip()\n'
    '\n'
    '\n'
    'def allowed_article_numbers(relevant_articles):\n'
    '    """Set of normalised `Điều X` strings actually present in the citation list."""\n'
    '    allowed = set()\n'
    '    for a in relevant_articles:\n'
    '        for m in re.finditer(r"Điều\\s+\\d+[A-Za-zÀ-ỹ]*", str(a), flags=re.IGNORECASE):\n'
    '            allowed.add(_normalize_text(m.group(0)).lower())\n'
    '    return allowed\n'
    '\n'
    '\n'
    'def postprocess_answer_citations(answer, relevant_articles):\n'
    '    answer = strip_leading_prompt_echo(answer)\n'
    '    allowed = allowed_article_numbers(relevant_articles)\n'
    '    mentioned = {_normalize_text(m.group(0)).lower()\n'
    '                 for m in re.finditer(r"Điều\\s+\\d+[A-Za-zÀ-ỹ]*", answer, flags=re.IGNORECASE)}\n'
    '    missing = sorted(mentioned - allowed)\n'
    '    if missing:\n'
    '        answer += ("\\n\\nLưu ý: Các nhận định trên chỉ dựa trên các căn cứ đã truy xuất; "\n'
    '                   "chưa đủ căn cứ để khẳng định các điều luật ngoài danh sách liên quan.")\n'
    '    if allowed and not re.search(r"Điều\\s+\\d+", answer, flags=re.IGNORECASE):\n'
    '        top_cites = []\n'
    '        for a in relevant_articles[:5]:\n'
    '            parts = str(a).split("|")\n'
    '            if len(parts) >= 3:\n'
    '                top_cites.append(parts[-1])\n'
    '        if top_cites:\n'
    '            answer += "\\n\\nCăn cứ tham khảo: " + ", ".join(dict.fromkeys(top_cites)) + "."\n'
    '    return answer\n'
    '\n'
    'print("[citations] postprocess_answer_citations ready")\n'
)

sec8_generator = code(
    '# --- IRAC answer generator (ported from hybridrag-decomp.ipynb) -----------\n'
    '# Builds an IRAC-structured prompt from the retrieved Hit contexts + the\n'
    '# allowed citation list, generates with the Qwen causal LM, and strips any\n'
    '# prompt echo. Honours GEN_LOAD_IN_4BIT for memory-constrained GPUs.\n'
    '\n'
    'class QwenIRACGenerator:\n'
    '    def __init__(self, model_name=GEN_MODEL):\n'
    '        import torch\n'
    '        from transformers import AutoTokenizer, AutoModelForCausalLM\n'
    '        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)\n'
    '        device = "cuda" if torch.cuda.is_available() else "cpu"\n'
    '        load_kwargs = {"trust_remote_code": True}\n'
    '        if device == "cuda" and GEN_LOAD_IN_4BIT:\n'
    '            from transformers import BitsAndBytesConfig\n'
    '            load_kwargs.update({\n'
    '                "device_map": "auto",\n'
    '                "torch_dtype": torch.float16,\n'
    '                "quantization_config": BitsAndBytesConfig(\n'
    '                    load_in_4bit=True,\n'
    '                    bnb_4bit_compute_dtype=torch.float16,\n'
    '                    bnb_4bit_quant_type="nf4",\n'
    '                    bnb_4bit_use_double_quant=True,\n'
    '                ),\n'
    '            })\n'
    '        elif device == "cuda":\n'
    '            load_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})\n'
    '        else:\n'
    '            load_kwargs.update({"torch_dtype": torch.float32})\n'
    '        print({"generator_device": device,\n'
    '               "load_in_4bit": bool(device == "cuda" and GEN_LOAD_IN_4BIT),\n'
    '               "gen_context_topk": GEN_CONTEXT_TOPK,\n'
    '               "gen_chunk_char_limit": GEN_CHUNK_CHAR_LIMIT,\n'
    '               "gen_max_input_tokens": GEN_MAX_INPUT_TOKENS})\n'
    '        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)\n'
    '        self.model.eval()\n'
    '        self.device = device\n'
    '\n'
    '    def build_prompt(self, question, contexts, relevant_articles):\n'
    '        ctx_blocks = []\n'
    '        for i, c in enumerate(contexts, start=1):\n'
    '            meta = " | ".join(str(c.get(k, "")) for k in ["law_id", "ten_van_ban", "dieu_so"])\n'
    '            text = str(c.get("chunk_text", ""))[:GEN_CHUNK_CHAR_LIMIT]\n'
    '            ctx_blocks.append(f"[{i}] {meta}\\n{text}")\n'
    '        allowed = "\\n".join("- " + str(a) for a in relevant_articles[:ARTICLE_CONTEXT_TOPK])\n'
    '        system = (\n'
    '            "Bạn là trợ lý pháp lý AI cho doanh nghiệp SME tại Việt Nam. "\n'
    '            "Chỉ trả lời dựa trên ngữ cảnh và danh sách căn cứ được cung cấp. "\n'
    '            "Không bịa văn bản, điều luật, khoản hoặc nguồn tham chiếu. "\n'
    '            "Nếu thiếu căn cứ, nói rõ là chưa đủ căn cứ."\n'
    '        )\n'
    '        user = (\n'
    '            f"Câu hỏi: {question}\\n\\n"\n'
    '            "Danh sách điều luật được phép viện dẫn:\\n"\n'
    '            f"{allowed}\\n\\n"\n'
    '            "Ngữ cảnh pháp lý:\\n"\n'
    '            f"{chr(10).join(ctx_blocks)}\\n\\n"\n'
    '            "Yêu cầu trả lời:\\n"\n'
    '            "- Trả lời bằng tiếng Việt, ngắn gọn nhưng đủ ý.\\n"\n'
    '            "- Dùng cấu trúc IRAC ngắn: Vấn đề, Quy định, Áp dụng, Kết luận.\\n"\n'
    '            "- Luôn nêu Điều X khi có căn cứ trong danh sách được phép viện dẫn.\\n"\n'
    '            "- Không nhắc lại toàn bộ ngữ cảnh, không trích dẫn điều ngoài danh sách.\\n"\n'
    '        )\n'
    '        if hasattr(self.tokenizer, "apply_chat_template"):\n'
    '            return self.tokenizer.apply_chat_template([\n'
    '                {"role": "system", "content": system},\n'
    '                {"role": "user", "content": user},\n'
    '            ], tokenize=False, add_generation_prompt=True)\n'
    '        return system + "\\n\\n" + user + "\\n\\nTrả lời:"\n'
    '\n'
    '    def generate(self, question, contexts, relevant_articles):\n'
    '        import torch\n'
    '        prompt = self.build_prompt(question, contexts, relevant_articles)\n'
    '        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,\n'
    '                                max_length=GEN_MAX_INPUT_TOKENS).to(self.model.device)\n'
    '        with torch.no_grad():\n'
    '            out = self.model.generate(\n'
    '                **inputs,\n'
    '                max_new_tokens=GEN_MAX_NEW_TOKENS,\n'
    '                temperature=GEN_TEMPERATURE,\n'
    '                top_p=GEN_TOP_P,\n'
    '                do_sample=GEN_TEMPERATURE > 0,\n'
    '                repetition_penalty=1.02,\n'
    '                use_cache=GEN_USE_CACHE,\n'
    '                pad_token_id=self.tokenizer.eos_token_id,\n'
    '                eos_token_id=self.tokenizer.eos_token_id,\n'
    '            )\n'
    '        gen_ids = out[0][inputs["input_ids"].shape[1]:]\n'
    '        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)\n'
    '        return strip_leading_prompt_echo(raw_text)\n'
    '\n'
    'print("[generator] QwenIRACGenerator class ready (instantiated lazily below)")\n'
)

sec8_batch = code(
    '# --- build generation contexts from the retrieved Hit list ---------------\n'
    '# Hit carries exactly the fields the generator consumes.\n'
    'def build_gen_contexts(hits, top_k=None):\n'
    '    top_k = top_k or GEN_CONTEXT_TOPK\n'
    '    out = []\n'
    '    for h in hits[:top_k]:\n'
    '        out.append({\n'
    '            "law_id": getattr(h, "law_id", "") or "",\n'
    '            "ten_van_ban": getattr(h, "ten_van_ban", "") or "",\n'
    '            "dieu_so": getattr(h, "dieu_so", "") or "",\n'
    '            "chunk_text": getattr(h, "chunk_text", "") or "",\n'
    '        })\n'
    '    return out\n'
    '\n'
    '\n'
    '# --- resolve input / output paths ----------------------------------------\n'
    '# Falls back to dev_set/questions.json when INPUT_QUERIES_PATH is blank, so\n'
    '# this cell is runnable out of the box (same behaviour as the original §8).\n'
    '_INPUT_Q = Path(INPUT_QUERIES_PATH) if str(INPUT_QUERIES_PATH).strip() else None\n'
    'if _INPUT_Q is None or not _INPUT_Q.exists():\n'
    '    _DEV_LOCAL = Path(REPO_DIR) / "dev_set"\n'
    '    _cand = ((DEV / "questions.json") if (DEV / "questions.json").exists()\n'
    '             else (_DEV_LOCAL / "questions.json"))\n'
    '    _INPUT_Q = _cand\n'
    '    print(f"[batch] INPUT_QUERIES_PATH not set / missing -> using {_INPUT_Q}")\n'
    'else:\n'
    '    print(f"[batch] input  : {_INPUT_Q}")\n'
    'print(f"[batch] results: {RESULTS_PATH}")\n'
    'print(f"[batch] zip    : {SUBMISSION_ZIP_PATH}")\n'
    '\n'
    'assert _INPUT_Q.exists(), f"input query file not found: {_INPUT_Q}"\n'
    'Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)\n'
    '\n'
    '\n'
    '# --- instantiate the generator (only when enabled) -----------------------\n'
    'import torch\n'
    'generator = None\n'
    'if USE_GENERATOR:\n'
    '    gc.collect()\n'
    '    if torch.cuda.is_available():\n'
    '        torch.cuda.empty_cache()\n'
    '    generator = QwenIRACGenerator()\n'
    'else:\n'
    '    print("[batch] USE_GENERATOR=False -> answer field left empty (retrieval-only).")\n'
    '\n'
    '\n'
    '# --- batch run: decompose -> retrieve -> generate -> postprocess ---------\n'
    'records_in = _json.loads(_INPUT_Q.read_text(encoding="utf-8"))\n'
    'print(f"[batch] loaded {len(records_in)} queries")\n'
    '\n'
    'records_out, t0 = [], time.time()\n'
    'for rec in records_in:\n'
    '    qid   = rec.get("id")\n'
    '    query = rec.get("question") or rec.get("query") or ""\n'
    '    if not query:\n'
    '        print(f"  [skip] id={qid} has no question text")\n'
    '        continue\n'
    '\n'
    '    hits = decomp_retriever.retrieve(query, fetch_text=True)\n'
    '    docs, articles = make_relevant_lists(hits)\n'
    '    contexts = build_gen_contexts(hits)\n'
    '\n'
    '    if generator is not None:\n'
    '        try:\n'
    '            answer = generator.generate(str(query), contexts, articles)\n'
    '        except torch.cuda.OutOfMemoryError:\n'
    '            print(f"  [oom] id={qid}; retrying with shorter contexts")\n'
    '            if torch.cuda.is_available():\n'
    '                torch.cuda.empty_cache()\n'
    '            short_ctx = []\n'
    '            for c in contexts[:2]:\n'
    '                c2 = dict(c)\n'
    '                c2["chunk_text"] = str(c2["chunk_text"])[:600]\n'
    '                short_ctx.append(c2)\n'
    '            answer = generator.generate(str(query), short_ctx, articles[:8])\n'
    '        answer = postprocess_answer_citations(answer, articles)\n'
    '    else:\n'
    '        answer = ""\n'
    '\n'
    '    out = dict(rec)                       # preserve any extra input fields\n'
    '    out["id"]                = qid\n'
    '    out["question"]          = query\n'
    '    out["answer"]            = answer\n'
    '    out["relevant_docs"]     = docs\n'
    '    out["relevant_articles"] = articles\n'
    '    records_out.append(out)\n'
    '    print(f"  [done] id={qid}  hits={len(hits)}  docs={len(docs)}  "\n'
    '          f"articles={len(articles)}  ans_len={len(answer)}")\n'
    '\n'
    'Path(RESULTS_PATH).write_text(_json.dumps(records_out, ensure_ascii=False, indent=2),\n'
    '                              encoding="utf-8")\n'
    'print(f"\\n[batch] wrote {len(records_out)} records to {RESULTS_PATH} in {time.time()-t0:.1f}s")\n'
    '\n'
    '\n'
    '# --- (optional) F2 score if a ground-truth file is available -------------\n'
    'try:\n'
    '    _DEV_LOCAL = Path(REPO_DIR) / "dev_set"\n'
    '    _GTPATH = ((DEV / "ground_truth.json") if (DEV / "ground_truth.json").exists()\n'
    '               else (_DEV_LOCAL / "ground_truth.json"))\n'
    '    if _GTPATH.exists():\n'
    '        import sys as _sys\n'
    '        if str(_DEV_LOCAL.parent) not in _sys.path:\n'
    '            _sys.path.insert(0, str(_DEV_LOCAL.parent))\n'
    '        from dev_set.eval import f2_macro\n'
    '        _gt = _json.loads(_GTPATH.read_text(encoding="utf-8"))\n'
    '        print(f"\\nF2 macro (decomposition + IRAC generation, batch) = "\n'
    '              f"{f2_macro(records_out, _gt):.4f}  [gt={_GTPATH}]")\n'
    '    else:\n'
    '        print("\\n(ground_truth.json not found - skipping F2 score)")\n'
    'except Exception as _e:\n'
    '    print(f"\\n(F2 scoring skipped: {_e})")\n'
    '\n'
    '\n'
    '# release the generator before §9 validation / §10 cleanup\n'
    'del generator\n'
    'gc.collect()\n'
    'if torch.cuda.is_available():\n'
    '    torch.cuda.empty_cache()\n'
)

# ---------------------------------------------------------------------------
# 3. §9 — Validate + zip submission
# ---------------------------------------------------------------------------
sec9_md = md(
    "## 9. Validate + package `submission.zip`\n"
    "\n"
    "Validates the `results.json` written in §8 against the grader contract\n"
    "(record schema + id coverage) and packages it into `submission.zip` with\n"
    "the arcname `results.json` — ported from [`hybridrag-decomp.ipynb`](notebooks/hybridrag-decomp.ipynb).\n"
)

sec9_code = code(
    '# --- validate + package submission (ported from hybridrag-decomp.ipynb) ---\n'
    'def validate_submission(path=None):\n'
    '    path = Path(path or RESULTS_PATH)\n'
    '    assert path.exists(), f"Missing results file: {path}"\n'
    '    data = _json.loads(path.read_text(encoding="utf-8"))\n'
    '    assert isinstance(data, list), "results.json must be a JSON list"\n'
    '    # id coverage against the records we just produced in §8\n'
    '    expected_ids = {int(r.get("id")) for r in records_out if r.get("id") is not None}\n'
    '    got_ids = {int(r.get("id")) for r in data}\n'
    '    assert expected_ids == got_ids, {\n'
    '        "missing": sorted(expected_ids - got_ids)[:20],\n'
    '        "extra": sorted(got_ids - expected_ids)[:20]}\n'
    '    required = {"id", "question", "answer", "relevant_docs", "relevant_articles"}\n'
    '    bad = [r.get("id") for r in data if set(r.keys()) != required]\n'
    '    assert not bad, f"Records with wrong schema: {bad[:10]}"\n'
    '    for r in data[:5]:\n'
    '        assert isinstance(r["relevant_docs"], list)\n'
    '        assert isinstance(r["relevant_articles"], list)\n'
    '    print({"records": len(data), "schema_ok": True, "ids_ok": True})\n'
    '\n'
    '\n'
    'validate_submission(RESULTS_PATH)\n'
    '\n'
    'with zipfile.ZipFile(SUBMISSION_ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:\n'
    '    zf.write(RESULTS_PATH, arcname="results.json")\n'
    '\n'
    'print(f"results written   : {RESULTS_PATH}")\n'
    'print(f"submission written: {SUBMISSION_ZIP_PATH}")\n'
)

# ---------------------------------------------------------------------------
# 4. §10 — Cleanup (extended from the decomposition notebook's §9 to also
#    release the generator and validate the GPU is clear).
# ---------------------------------------------------------------------------
sec10_md = md(
    "## 10. Cleanup\n"
    "\n"
    "Full memory teardown — release GPU + host resources before a fresh run.\n"
    "Extended from the decomposition notebook's §9 to also drop the generator.\n"
    "Run this BEFORE re-running §1-§9 from scratch (or just do Kernel → Restart).\n"
)

sec10_code = code(
    '# ── Full memory teardown — release GPU + host resources before a fresh run ──\n'
    'import gc, subprocess\n'
    '\n'
    '\n'
    'def _safe_close(obj, attr, label):\n'
    '    try:\n'
    '        fn = getattr(obj, attr, None)\n'
    '        if callable(fn):\n'
    '            fn()\n'
    '            print(f"[cleanup] {label} closed")\n'
    '    except Exception as e:\n'
    '        print(f"[cleanup] {label} close skipped ({e})")\n'
    '\n'
    '\n'
    '# 1. Close the FTS SQLite connection (host RAM).\n'
    "if 'fts' in globals():\n"
    "    _safe_close(fts, 'close', 'fts')\n"
    '\n'
    '\n'
    '# 2. Drop heavy GPU/host objects so their refcounts hit zero.\n'
    '#    faiss_index._index (GPU), query_encoder (BGE-m3), base_retriever._reranker\n'
    '#    (bge-reranker-v2-m3), the LLM router model, and the QwenIRACGenerator all\n'
    '#    live behind these names.\n'
    "for _name in ('decomp_retriever', 'base_retriever', 'router',\n"
    "              'query_encoder', 'faiss_index', 'graph_expander', 'fts',\n"
    "              'generator'):\n"
    "    if _name in globals():\n"
    "        globals()[_name] = None\n"
    "        print(f'[cleanup] {_name} released')\n"
    'del _name\n'
    '\n'
    'gc.collect()\n'
    '\n'
    '\n'
    '# 3. Free CUDA: sync, empty the caching allocator, reset peak stats.\n'
    'try:\n'
    '    import torch\n'
    '    if torch.cuda.is_available():\n'
    '        torch.cuda.synchronize()\n'
    '        torch.cuda.empty_cache()\n'
    '        torch.cuda.reset_peak_memory_stats()\n'
    '        print(f"[cleanup] CUDA cache emptied | "\n'
    '              f"allocated={torch.cuda.memory_allocated()/1e9:.2f} GB "\n'
    '              f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")\n'
    'except Exception as e:\n'
    '    print(f"[cleanup] torch CUDA cleanup skipped ({e})")\n'
    '\n'
    '\n'
    '# 4. Verify the GPU is actually clear from the driver\'s perspective.\n'
    'try:\n'
    '    r = subprocess.run(\n'
    '        "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits",\n'
    '        shell=True, capture_output=True, text=True)\n'
    '    used, total = (float(x) for x in r.stdout.strip().split(","))\n'
    '    print(f"[cleanup] nvidia-smi: {used:.0f} / {total:.0f} MiB used "\n'
    '          f"({100*used/total:.1f}% of VRAM)")\n'
    'except Exception:\n'
    '    pass\n'
    '\n'
    '\n'
    'print("\\n✅ In-kernel memory cleared. For a guaranteed-fresh start "\n'
    '      "(no stale Python state / cached models), also do: Kernel → Restart.")\n'
)

# ---------------------------------------------------------------------------
# 5. Assemble + write.
# ---------------------------------------------------------------------------
nb["cells"] = kept + [
    sec8_md, sec8_cfg, sec8_citations, sec8_generator, sec8_batch,
    sec9_md, sec9_code,
    sec10_md, sec10_code,
]

OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"[build] wrote {OUT_NB}  ({len(nb['cells'])} cells)")
