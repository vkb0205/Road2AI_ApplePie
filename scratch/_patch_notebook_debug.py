"""One-off patcher: add per-stage debug/explainability to the advanced notebook.

- Instruments the in-notebook AdvancedHybridRetriever.retrieve() with an
  `explain=` mode that records a RetrievalTrace across its own stages
  (variants, legs, weighted_rrf, graph, fetch, rerank, article_agg, final,
  output), reusing src/retrieval/debug.py helpers.
- Inserts a new "5b. Explainability" markdown + code cell after the
  "Run a query" section so the user can print the per-stage trace for a query.

Idempotent-ish: it operates on string anchors; re-running on an already
patched notebook is a no-op for the import/anchor lines (replace is a no-op
when the anchor text is absent) and will skip inserting the explainability
cells if their marker is already present.
"""
import json
from pathlib import Path

NB = Path("notebooks/retrieval_colab_advanced.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


# ---------------------------------------------------------------------------
# 1) Instrument the AdvancedHybridRetriever cell (the one defining the class)
# ---------------------------------------------------------------------------
adv_idx = None
for i, c in enumerate(cells):
    s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    if "class AdvancedHybridRetriever" in s:
        adv_idx = i
        break
assert adv_idx is not None, "AdvancedHybridRetriever cell not found"

src = "".join(cells[adv_idx]["source"]) if isinstance(cells[adv_idx]["source"], list) else cells[adv_idx]["source"]

# 1a. import debug helpers next to `from retrieval.retriever import Hit`
old_imp = "from retrieval.retriever import Hit"
new_imp = (
    "from retrieval.retriever import Hit\n"
    "from retrieval.debug import (\n"
    "    RetrievalTrace, StageSnapshot, snapshot_items,\n"
    "    format_trace as _fmt_trace,\n"
    ")"
)
if "_fmt_trace" not in src:
    src = src.replace(old_imp, new_imp, 1)

# 1b. add last_trace attribute in __init__
old_init = "        self._hyde = None  # lazily built"
new_init = (
    "        self._hyde = None  # lazily built\n"
    "        self.last_trace = None  # per-stage debug trace (set when explain=True)"
)
if "self.last_trace = None  # per-stage debug trace" not in src:
    src = src.replace(old_init, new_init, 1)

# 1c. add a last_trace_formatted property right before `def retrieve`
prop_block = (
    "    @property\n"
    "    def last_trace_formatted(self):\n"
    "        return _fmt_trace(self.last_trace) if self.last_trace is not None else None\n"
    "\n"
)
if "def last_trace_formatted" not in src:
    src = src.replace("    # -- main entry ---------------------------------------------------- #\n",
                      prop_block + "    # -- main entry ---------------------------------------------------- #\n", 1)

# 1d. Replace the whole retrieve() method with an instrumented version.
start_marker = "    def retrieve(self, query, fetch_text=False):"
end_marker = "\n\nadvanced_retriever = AdvancedHybridRetriever("
s = src.find(start_marker)
e = src.find(end_marker)
assert s != -1 and e != -1 and e > s, "retrieve() method span not found"

new_method = '''    def retrieve(self, query, fetch_text=False, explain=False, print_trace=False):
        cfg = self.base.config

        # ---- debug trace setup (zero overhead when explain=False) ----------
        trace = None
        if explain:
            import time as _time
            _t0 = _time.perf_counter()
            trace = RetrievalTrace(query=query, config={
                'use_dense': cfg.use_dense, 'use_reranker': cfg.use_reranker,
                'graph': self.base.graph_expander is not None,
                'use_multiquery': self.use_multiquery,
                'use_article_agg': self.use_article_agg,
                'top_bm25': cfg.top_bm25, 'top_dense': cfg.top_dense,
                'fused_top': cfg.fused_top, 'final_top_k': cfg.final_top_k,
                'rrf_k': cfg.rrf_k,
            })

        # ---- 0. single-query fast path (collapses to base behaviour) ----
        if not self.use_multiquery:
            hits = self.base.retrieve(query, fetch_text=fetch_text)
            if trace is not None:
                trace.add(StageSnapshot(
                    name='base_retrieve', count=len(hits),
                    top_items=[{'row_idx': h.row_idx, 'score': h.score,
                                'source': h.source, 'law_id': h.law_id,
                                'ten_van_ban': h.ten_van_ban, 'dieu_so': h.dieu_so}
                               for h in hits[:8]],
                    skip='use_multiquery=False -> base.retrieve()',
                ))
                trace.total_elapsed_ms = (_time.perf_counter() - _t0) * 1000.0
                self.last_trace = trace
                if print_trace:
                    print(_fmt_trace(trace))
            return hits

        # ---- 1. query variants + HyDE --------------------------------
        if trace is not None:
            t0 = _time.perf_counter()
        hyde_text = self._hyde_text(query)
        variants = build_query_variants(query, hyde_text)
        print(f"[advanced] {len(variants)} variants: " +
              ', '.join(f"{v['kind']}({v['weight']})" for v in variants))
        if trace is not None:
            trace.add(StageSnapshot(
                name='variants', count=len(variants),
                elapsed_ms=(_time.perf_counter() - t0) * 1000.0,
                top_items=[{'row_idx': i, 'score': v['weight'], 'source': v['kind'],
                            'text': (v['text'][:90] + ('...' if len(v['text']) > 90 else ''))}
                           for i, v in enumerate(variants[:8])],
                diagnostics={'hyde': bool(hyde_text),
                             'dense_only_kinds': [v['kind'] for v in variants if v.get('dense_only')]},
            ))

        # ---- 2. per-variant legs -> weighted rankings ----------------
        if trace is not None:
            t0 = _time.perf_counter()
        weighted_rankings = []
        per_variant_counts = []
        for v in variants:
            lex, dense = self._variant_legs(v)
            per_variant_counts.append({'kind': v['kind'], 'lexical': len(lex), 'dense': len(dense)})
            if lex:
                weighted_rankings.append((lex, v['weight'], f"lexical:{v['kind']}"))
            if dense:
                weighted_rankings.append((dense, v['weight'], f"dense:{v['kind']}"))
        if trace is not None:
            trace.add(StageSnapshot(
                name='legs', count=len(weighted_rankings),
                elapsed_ms=(_time.perf_counter() - t0) * 1000.0,
                diagnostics={'per_variant': per_variant_counts,
                             'lexical_rankings': sum(1 for r in weighted_rankings if r[2].startswith('lexical')),
                             'dense_rankings': sum(1 for r in weighted_rankings if r[2].startswith('dense'))},
            ))

        # ---- 3. weighted RRF fuse ------------------------------------
        if trace is not None:
            t0 = _time.perf_counter()
        fused = weighted_rrf_fuse(weighted_rankings, k=cfg.rrf_k, topk=cfg.fused_top)
        if trace is not None:
            trace.add(StageSnapshot(
                name='weighted_rrf', count=len(fused),
                elapsed_ms=(_time.perf_counter() - t0) * 1000.0,
                top_items=snapshot_items(
                    [{**f, 'score': f.get('rrf_score', 0.0)} for f in fused], 8,
                    score_key='rrf_score',
                    extra_keys=('law_id', 'ten_van_ban', 'dieu_so')),
                diagnostics={'k': cfg.rrf_k, 'topk': cfg.fused_top,
                             'rankings_in': len(weighted_rankings)},
            ))
        if not fused:
            if trace is not None:
                trace.total_elapsed_ms = (_time.perf_counter() - _t0) * 1000.0
                self.last_trace = trace
            return []

        # ---- 4. graph expansion (optional, reuse base) ---------------
        if trace is not None:
            t0 = _time.perf_counter()
        if self.base.graph_expander is not None:
            candidates = [(r['row_idx'], r['rrf_score']) for r in fused]
            expanded = self.base.graph_expander.expand(candidates, top_n=cfg.expanded_top)
            ordered = [{**e, 'score': e.get('score', 0.0)} for e in expanded]
        else:
            ordered = [{'row_idx': r['row_idx'], 'score': r['rrf_score'],
                        'source': 'fused', 'rrf_score': r['rrf_score']} for r in fused]
        ordered.sort(key=lambda e: (-e['score'], e['row_idx']))
        if trace is not None:
            src_counts = {}
            for e in ordered:
                sk = str(e.get('source', '?'))
                src_counts[sk] = src_counts.get(sk, 0) + 1
            trace.add(StageSnapshot(
                name='graph', count=len(ordered),
                elapsed_ms=(_time.perf_counter() - t0) * 1000.0,
                top_items=snapshot_items(ordered, 8, score_key='score',
                    extra_keys=('source', 'law_id', 'ten_van_ban', 'dieu_so')),
                skip='no graph_expander' if self.base.graph_expander is None else None,
                diagnostics={'expanded_top': cfg.expanded_top, 'source_counts': src_counts},
            ))

        # ---- 5. fetch metadata + optional text -----------------------
        if trace is not None:
            t0 = _time.perf_counter()
        row_idxs = [e['row_idx'] for e in ordered]
        meta_by_row = self.base._fetch_meta(row_idxs)
        text_by_row = self.base._fetch_text(row_idxs) if fetch_text else {}
        for e in ordered:
            e['rrf_score'] = e.get('rrf_score', e.get('score', 0.0))
        if trace is not None:
            missing = [ri for ri in row_idxs if ri not in meta_by_row]
            ordered_meta = [{**e, **meta_by_row.get(e['row_idx'], {})} for e in ordered]
            trace.add(StageSnapshot(
                name='fetch', count=len(meta_by_row),
                elapsed_ms=(_time.perf_counter() - t0) * 1000.0,
                top_items=snapshot_items(ordered_meta, 8, score_key='score',
                    extra_keys=('source', 'law_id', 'ten_van_ban', 'dieu_so', 'chunk_id', 'doc_uid')),
                diagnostics={'requested': len(row_idxs), 'resolved': len(meta_by_row),
                             'missing_rows': missing, 'text_fetched': bool(fetch_text)},
            ))

        # ---- 6. rerank (optional, reuse base) ------------------------
        if trace is not None:
            t0 = _time.perf_counter()
        pre = len(ordered)
        if cfg.use_reranker and ordered:
            ordered = self.base._rerank(query, ordered, meta_by_row, text_by_row)
        ordered = ordered[:RERANK_TOPK]
        if trace is not None:
            skip = None if (cfg.use_reranker and pre) else (
                'use_reranker=False' if not cfg.use_reranker else 'no candidates to rerank')
            trace.add(StageSnapshot(
                name='rerank', count=len(ordered),
                elapsed_ms=(_time.perf_counter() - t0) * 1000.0,
                top_items=snapshot_items(ordered, 8, score_key='score',
                    extra_keys=('rerank_score', 'source', 'law_id', 'ten_van_ban', 'dieu_so')),
                skip=skip,
                diagnostics={'candidates_in': pre, 'rerank_topk': RERANK_TOPK,
                             'reranker_unavailable': self.base._reranker_unavailable},
            ))

        # ---- 7. article aggregation (optional) -----------------------
        if trace is not None:
            t0 = _time.perf_counter()
        if self.use_article_agg:
            for e in ordered:
                m = meta_by_row.get(int(e['row_idx']), {})
                e.update({k: m.get(k, e.get(k, '')) for k in
                          ('law_id', 'ten_van_ban', 'dieu_so', 'chunk_id', 'doc_uid')})
                if fetch_text:
                    e['chunk_text'] = text_by_row.get(int(e['row_idx']), e.get('chunk_text', ''))
            agg = aggregate_article_contexts(ordered, query)
            ordered = agg
        if trace is not None:
            items = []
            for e in ordered[:8]:
                items.append({'row_idx': int(e['row_idx']),
                              'score': float(e.get('article_score', e.get('score', 0.0))),
                              'source': str(e.get('source', '')),
                              'law_id': e.get('law_id', ''),
                              'ten_van_ban': e.get('ten_van_ban', ''),
                              'dieu_so': e.get('dieu_so', ''),
                              'support_count': e.get('support_count')})
            trace.add(StageSnapshot(
                name='article_agg', count=len(ordered),
                elapsed_ms=(_time.perf_counter() - t0) * 1000.0,
                top_items=items,
                skip='use_article_agg=False' if not self.use_article_agg else None,
                diagnostics={'adaptive_topk': adaptive_article_topk(query)},
            ))

        # ---- 8. build final List[Hit] --------------------------------
        hits = []
        for e in ordered[:cfg.final_top_k]:
            row_idx = int(e['row_idx'])
            m = meta_by_row.get(row_idx, {})
            hits.append(Hit(
                row_idx=row_idx,
                score=float(e.get('score', e.get('article_score', 0.0))),
                source=str(e.get('source', 'fused')),
                law_id=str(e.get('law_id', m.get('law_id', ''))),
                ten_van_ban=str(e.get('ten_van_ban', m.get('ten_van_ban', ''))),
                dieu_so=str(e.get('dieu_so', m.get('dieu_so', ''))),
                chunk_id=str(e.get('chunk_id', m.get('chunk_id', ''))),
                doc_uid=str(e.get('doc_uid', m.get('doc_uid', ''))),
                chunk_text=text_by_row.get(row_idx, '') if fetch_text else '',
            ))

        if trace is not None:
            trace.add(StageSnapshot(
                name='final', count=len(hits),
                top_items=[{'row_idx': h.row_idx, 'score': h.score, 'source': h.source,
                            'law_id': h.law_id, 'ten_van_ban': h.ten_van_ban,
                            'dieu_so': h.dieu_so, 'chunk_id': h.chunk_id}
                           for h in hits[:8]],
                diagnostics={'final_top_k': cfg.final_top_k},
            ))
            docs, articles = make_relevant_lists(hits)
            trace.add(StageSnapshot(
                name='output', count=len(hits),
                diagnostics={'relevant_docs': docs, 'relevant_articles': articles},
            ))
            trace.total_elapsed_ms = (_time.perf_counter() - _t0) * 1000.0
            self.last_trace = trace
            if print_trace:
                print(_fmt_trace(trace))

        return hits'''

src = src[:s] + new_method + src[e:]
set_source(cells[adv_idx], src)

# ---------------------------------------------------------------------------
# 2) Insert "5b. Explainability" markdown + code cells after the Run-a-query
#    code cell (the one containing "EDIT YOUR QUERY HERE").
# ---------------------------------------------------------------------------
EXPLAIN_MARKER = "## 5b. Explainability"
if not any(EXPLAIN_MARKER in ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
           for c in cells):
    run_idx = None
    for i, c in enumerate(cells):
        s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if "EDIT YOUR QUERY HERE" in s:
            run_idx = i
            break
    assert run_idx is not None, "Run-a-query cell not found"

    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": (
            "## 5b. 🔬 Explainability — per-stage debug trace\n"
            "\n"
            "When retrieval quality is off, you want to see *what each stage produced*, not just the final hits.\n"
            "The advanced retriever now supports `explain=True`, which records a `RetrievalTrace` with one\n"
            "`StageSnapshot` per stage of the multi-query pipeline and stores it on `advanced_retriever.last_trace`.\n"
            "\n"
            "Stages traced (in order):\n"
            "\n"
            "| Stage | What it records |\n"
            "|---|---|\n"
            "| `variants` | generated query variants (kind, weight, text preview) + HyDE on/off |\n"
            "| `legs` | per-variant lexical/dense hit counts + how many rankings feed RRF |\n"
            "| `weighted_rrf` | fused candidate pool (top-N with rrf_score + legal metadata) + k/topk |\n"
            "| `graph` | post-expansion candidates + `source_counts` (candidate/doc_expand/concept_expand) |\n"
            "| `fetch` | requested vs resolved `row_idx`, `missing_rows`, text_fetched |\n"
            "| `rerank` | post cross-encoder order (rerank_score) + unavailable/skip reason |\n"
            "| `article_agg` | one best chunk per article (article_score, support_count) + adaptive top-k |\n"
            "| `final` / `output` | final `List[Hit]` + collapsed `relevant_docs` / `relevant_articles` |\n"
            "\n"
            "Each stage also carries wall-clock `ms` and a `skip` reason when a stage is turned off or degrades,\n"
            "so you can tell the difference between *a stage was disabled* and *a stage ran but found nothing*.\n"
            "\n"
            "> Zero overhead when `explain=False` (the normal `retrieve()` path is unchanged). The helpers live in\n"
            "> [`src/retrieval/debug.py`](src/retrieval/debug.py) and are reused by the base `HybridRetriever` too.\n"
        ).splitlines(keepends=True),
    }

    code_cell = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": (
            "# Run the SAME query as Section 5, but with the per-stage trace printed.\n"
            "# explain=True records advanced_retriever.last_trace; print_trace=True also prints it.\n"
            "t0 = time.time()\n"
            "expl_hits = advanced_retriever.retrieve(QUERY, fetch_text=True, explain=True, print_trace=True)\n"
            "print(f\"\\nExplained retrieve: {len(expl_hits)} hits in {time.time()-t0:.2f}s\")\n"
            "\n"
            "# The structured trace is also available for programmatic inspection / diffing:\n"
            "trace = advanced_retriever.last_trace\n"
            "if trace is not None:\n"
            "    print(\"\\nstage -> count  elapsed_ms  skip\")\n"
            "    for s in trace.stages:\n"
            "        print(f\"  {s.name:12s} {s.count:5d}  {str(s.elapsed_ms):>8s}  {s.skip or ''}\")\n"
            "    # e.g. inspect which articles survived article aggregation:\n"
            "    out = trace.stage('output')\n"
            "    if out is not None:\n"
            "        print(\"\\nrelevant_articles:\", out.diagnostics.get('relevant_articles'))\n"
        ).splitlines(keepends=True),
    }

    cells.insert(run_idx + 1, md_cell)
    cells.insert(run_idx + 2, code_cell)

# ---------------------------------------------------------------------------
# 3) Write back
# ---------------------------------------------------------------------------
nb["cells"] = cells
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("patched notebook OK | total cells:", len(cells))
