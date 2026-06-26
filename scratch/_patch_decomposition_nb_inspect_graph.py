"""Idempotent patcher for retrieval_colab_decomposition.ipynb.

Adds a **read-only inspection cell** (Section 6c) after the explainability cell
(§6b). It does NOT change any retrieval/router/rerank code and does NOT alter
results — it only *inspects* the per-sub-query traces that §6b already captured
(`decomp_retriever.last_sub_query_traces`) to answer:

  Do broad / unwanted legal neighbours already enter the fused candidate set at
  the LEXICAL + DENSE -> RRF fusion stage (i.e. BEFORE graph expansion)?

For each sub-query it:
  1. Dumps the raw `lexical` stage top items (bm25_score + legal metadata).
  2. Dumps the raw `dense` stage top items (dense_score + legal metadata).
  3. Dumps the `rrf` fused top items (rrf_score + legal metadata).
  4. Classifies every recorded rrf item by leg attribution:
       - lexical_only : in lexical recorded slice, NOT in dense recorded slice
       - dense_only   : in dense recorded slice, NOT in lexical recorded slice
       - both_legs    : in both recorded slices
       - beyond_slice : in neither recorded slice (it sits deeper than the
                        per-stage top-N recording, but still within the leg's
                        full `count` — RRF can only draw from lexical+dense).
     This reveals whether broad neighbours arrive via the dense (semantic) leg,
     the lexical leg, or both, BEFORE graph expansion is even considered.

Re-running this patcher is safe: it first removes any previously inserted §6c
cells (matched by the marker) and then re-inserts the current version, so edits
to the inspection cell propagate cleanly without leaving duplicates.
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


MARKER = "## 6c. 🔬 Lexical vs Dense vs RRF — pre-expansion leg inspection"
CODE_MARKER = "# ── 6c. Read-only inspection"

# Remove ANY previously inserted §6c cells (markdown header + code), regardless
# of which version, so the patcher is re-runnable and edits propagate without
# duplicates. Matches any markdown line starting with "## 6c." and any code
# line starting with "# ── 6c. Read-only inspection".
kept = []
removed = 0
for c in nb["cells"]:
    s = _src(c)
    if c.get("cell_type") == "markdown" and any(line.lstrip().startswith("## 6c.") for line in s):
        removed += 1
        continue
    if c.get("cell_type") == "code" and any(line.lstrip().startswith(CODE_MARKER) for line in s):
        removed += 1
        continue
    kept.append(c)
nb["cells"] = kept
if removed:
    print(f"removed {removed} previously inserted §6c cell(s)")

md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6c. 🔬 Lexical vs Dense vs RRF — pre-expansion leg inspection\n",
        "\n",
        "Read-only diagnosis of the traces captured in §6b. The question here is\n",
        "whether broad / unwanted legal neighbours already enter the fused candidate\n",
        "set at the **lexical + dense → RRF** stage — i.e. *before* graph expansion is\n",
        "even considered.\n",
        "\n",
        "For each sub-query it dumps the raw top items of the [`lexical`](src/retrieval/retriever.py)\n",
        "leg (`bm25_score`), the [`dense`](src/retrieval/retriever.py) leg (`dense_score`) and the\n",
        "[`rrf`](src/retrieval/rrf.py) fused set (`rrf_score`), then classifies every recorded\n",
        "fused item by which leg(s) it came from:\n",
        "\n",
        "- **lexical_only** — in the lexical recorded slice, not the dense one.\n",
        "- **dense_only** — in the dense recorded slice, not the lexical one.\n",
        "- **both_legs** — present in both legs (RRF boosts these most).\n",
        "- **beyond_slice** — in neither recorded slice: it ranks deeper than the\n",
        "  per-stage top-N the trace keeps, but still within a leg's full `count`\n",
        "  (RRF can only draw from lexical + dense, so it must originate in one of\n",
        "  them).\n",
        "\n",
        "> Caveat: each trace stage only *records* the top `debug_top_n` items\n",
        "> (default 8), while `count` holds the true full leg size. So `beyond_slice`\n",
        "is not\n",
        "> \"graph noise\" — it is a leg member that was too deep to be recorded.\n",
        "> Graph expansion is **not** involved in this cell at all.\n",
        "\n",
        "> This cell never mutates `decomp_retriever`, `base_retriever`, the router\n",
        "> or any config — it only reads `last_sub_query_traces`.\n",
    ],
}

code_cell = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# ── 6c. Read-only inspection: lexical vs dense vs rrf legs ───────────────\n",
        "# Uses the traces already captured by §6b. If they are missing (cell run\n",
        "# out of order) we lazily re-run ONE explained retrieve so this cell is\n",
        "# still runnable standalone. No retrieval/rerank/config code is modified.\n",
        "from collections import OrderedDict\n",
        "\n",
        "traces = getattr(decomp_retriever, \"last_sub_query_traces\", None)\n",
        "if not traces:\n",
        "    print(\"[inspect] no traces yet — running one explained retrieve …\")\n",
        "    decomp_retriever.retrieve(QUERY, fetch_text=True, explain=True, print_trace=False)\n",
        "    traces = decomp_retriever.last_sub_query_traces\n",
        "\n",
        "assert traces, \"[inspect] no SubQueryTrace available — run §6b first.\"\n",
        "\n",
        "\n",
        "def _stage(snapshot):\n",
        "    \"\"\"Return (rowidx_list_in_order, {row_idx: (score, meta_str)}) from a StageSnapshot.\"\"\"\n",
        "    if snapshot is None:\n",
        "        return [], {}\n",
        "    ids, score_map, meta_map = [], {}, {}\n",
        "    for it in snapshot.top_items:\n",
        "        ri = it.get(\"row_idx\")\n",
        "        if ri is None:\n",
        "            continue\n",
        "        ri = int(ri)\n",
        "        ids.append(ri)\n",
        "        sc = it.get(\"score\")\n",
        "        score_map[ri] = (\"%.4f\" % sc) if isinstance(sc, (int, float)) else \"—\"\n",
        "        law = it.get(\"law_id\", \"\") or \"\"\n",
        "        ten = it.get(\"ten_van_ban\", \"\") or \"\"\n",
        "        dieu = it.get(\"dieu_so\", \"\") or \"\"\n",
        "        meta_map[ri] = f\"[{law}|{ten}|{dieu}]\"\n",
        "    return ids, {ri: (score_map[ri], meta_map[ri]) for ri in ids}\n",
        "\n",
        "\n",
        "def _dump(label, ids, info):\n",
        "    print(f\"  [{label}]\")\n",
        "    if not ids:\n",
        "        print(\"     (no recorded items)\")\n",
        "        return\n",
        "    for rank, ri in enumerate(ids):\n",
        "        sc, meta = info[ri]\n",
        "        print(f\"     #{rank:<2} row_idx={ri:<6} score={sc:<8} {meta}\")\n",
        "\n",
        "\n",
        "print(\"=\" * 92)\n",
        "print(\"INSPECT: lexical  vs  dense  vs  rrf (fused)  — BEFORE graph expansion\")\n",
        "print(\"=\" * 92)\n",
        "\n",
        "# Aggregate fused-item leg-attribution across sub-queries.\n",
        "global_attr = OrderedDict()  # row_idx -> list of (sub_query_index, bucket)\n",
        "\n",
        "for t in traces:\n",
        "    rt = t.retrieval_trace\n",
        "    if rt is None:\n",
        "        print(f\"\\n[sub {t.sub_query_index}] NO retrieval_trace — re-run §6b with explain=True\")\n",
        "        continue\n",
        "    lex_snap = rt.stage(\"lexical\")\n",
        "    dense_snap = rt.stage(\"dense\")\n",
        "    rrf_snap = rt.stage(\"rrf\")\n",
        "\n",
        "    lex_ids, lex_info = _stage(lex_snap)\n",
        "    dense_ids, dense_info = _stage(dense_snap)\n",
        "    rrf_ids, rrf_info = _stage(rrf_snap)\n",
        "    lex_set, dense_set, rrf_set = set(lex_ids), set(dense_ids), set(rrf_ids)\n",
        "\n",
        "    print(f\"\\n— sub-query [{t.sub_query_index}/{t.num_sub_queries - 1}]  text={t.sub_query_text!r}\")\n",
        "    if lex_snap is not None:\n",
        "        print(f\"  lexical full count={lex_snap.count}  recorded={len(lex_ids)}  \"\n",
        "              f\"skip={lex_snap.skip!r}\")\n",
        "    if dense_snap is not None:\n",
        "        print(f\"  dense   full count={dense_snap.count}  recorded={len(dense_ids)}  \"\n",
        "              f\"skip={dense_snap.skip!r}\")\n",
        "    if rrf_snap is not None:\n",
        "        diag = rrf_snap.diagnostics\n",
        "        print(f\"  rrf     full count={rrf_snap.count}  recorded={len(rrf_ids)}  \"\n",
        "              f\"skip={rrf_snap.skip!r}  rankings_in={diag.get('rankings_in','?')} \"\n",
        "              f\"fused_top={diag.get('fused_top','?')} k={diag.get('k','?')}\")\n",
        "\n",
        "    print(\"\")\n",
        "    _dump(\"lexical top items (bm25_score)\", lex_ids, lex_info)\n",
        "    print(\"\")\n",
        "    _dump(\"dense top items (dense_score)\", dense_ids, dense_info)\n",
        "    print(\"\")\n",
        "    _dump(\"rrf fused top items (rrf_score)\", rrf_ids, rrf_info)\n",
        "\n",
        "    # --- classify every recorded fused item by leg attribution -----------\n",
        "    buckets = {\"both_legs\": [], \"lexical_only\": [], \"dense_only\": [], \"beyond_slice\": []}\n",
        "    for ri in rrf_ids:\n",
        "        in_lex = ri in lex_set\n",
        "        in_dense = ri in dense_set\n",
        "        if in_lex and in_dense:\n",
        "            bucket = \"both_legs\"\n",
        "        elif in_lex:\n",
        "            bucket = \"lexical_only\"\n",
        "        elif in_dense:\n",
        "            bucket = \"dense_only\"\n",
        "        else:\n",
        "            bucket = \"beyond_slice\"\n",
        "        buckets[bucket].append(ri)\n",
        "        global_attr.setdefault(ri, []).append((t.sub_query_index, bucket))\n",
        "\n",
        "    print(\"\")\n",
        "    print(\"  rrf item leg-attribution (recorded fused items):\")\n",
        "    for b, items in buckets.items():\n",
        "        tag = \"  <-- broad semantic neighbours likely enter here\" if b == \"dense_only\" else \"\"\n",
        "        print(f\"     {b:<13} n={len(items):<3} {items}{tag}\")\n",
        "\n",
        "    # Which docs are in a leg but DROPPED by fusion (truncated to fused_top)?\n",
        "    lex_dropped = lex_set - rrf_set\n",
        "    dense_dropped = dense_set - rrf_set\n",
        "    if lex_dropped:\n",
        "        print(f\"  lexical recorded items NOT in rrf (truncated by fused_top): {sorted(lex_dropped)}\")\n",
        "    if dense_dropped:\n",
        "        print(f\"  dense   recorded items NOT in rrf (truncated by fused_top): {sorted(dense_dropped)}\")\n",
        "\n",
        "    # --- last check: final relevant_docs / relevant_articles --------------\n",
        "    print(f\"  [final] hits={len(t.final_hits)}  relevant_docs={len(t.relevant_docs)}  \"\n",
        "          f\"relevant_articles={len(t.relevant_articles)}\")\n",
        "    if t.relevant_articles:\n",
        "        print(f\"     relevant_articles: {t.relevant_articles}\")\n",
        "    if t.relevant_docs:\n",
        "        print(f\"     relevant_docs     : {t.relevant_docs}\")\n",
        "\n",
        "# --- Cross-sub-query summary: how do fused items get attributed? ----------\n",
        "print(\"\\n\" + \"=\" * 92)\n",
        "print(\"SUMMARY — rrf fused-item leg attribution aggregated across sub-queries\")\n",
        "print(\"=\" * 92)\n",
        "if not global_attr:\n",
        "    print(\"No rrf items recorded.\")\n",
        "else:\n",
        "    bucket_count = {\"both_legs\": 0, \"lexical_only\": 0, \"dense_only\": 0, \"beyond_slice\": 0}\n",
        "    for ri, subs in global_attr.items():\n",
        "        for _, b in subs:\n",
        "            bucket_count[b] = bucket_count.get(b, 0) + 1\n",
        "    total = sum(bucket_count.values())\n",
        "    print(f\"recorded rrf item-appearance across all sub-queries: {total}\")\n",
        "    for b, n in bucket_count.items():\n",
        "        pct = (100.0 * n / total) if total else 0.0\n",
        "        tag = \"  <-- broad neighbours entering via the dense (semantic) leg\" if b == \"dense_only\" else \"\"\n",
        "        print(f\"  {b:<13} n={n:<3} ({pct:4.1f}%){tag}\")\n",
        "    # Unique dense_only row_idx (the prime suspects for unwanted neighbours)\n",
        "    dense_only_unique = sorted({ri for ri, subs in global_attr.items()\n",
        "                               if any(b == \"dense_only\" for _, b in subs)})\n",
        "    print(f\"\\nunique dense_only row_idx (suspect broad semantic neighbours): {dense_only_unique}\")\n",
        "\n",
        "print(\"\\n[inspect] done — no retrieval/config code was modified.\")\n",
    ],
}

# Find the index of the explainability code cell (the one starting with
# `expl_hits = decomp_retriever.retrieve`) and insert the markdown + code
# cells right after it.
insert_at = None
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code":
        continue
    if any("expl_hits = decomp_retriever.retrieve" in s for s in _src(c)):
        insert_at = i + 1
        break

assert insert_at is not None, "could not locate the §6b explainability cell to insert after"
nb["cells"][insert_at:insert_at] = [md_cell, code_cell]

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print(f"inserted inspection cell after index {insert_at - 1}")
