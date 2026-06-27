"""Live GPU diagnostic: WHY does the reranker bury the gold article?

``diagnose_pool.py`` works on the dumped pool and showed: gold is IN the pool
for 16/20 questions but the cross-encoder ranks it below rank 5 for 13 of them.
That has three possible causes, which need different fixes:

  1. Gold's best chunk was never harvested (lost a per-doc / rerank_pool cut)
     -> harvesting / recall bug (widen per_doc_articles / rerank_pool / top_docs).
  2. Gold's harvested passage is degraded or wrong text (fetch_text miss -> the
     bare 'Điều X' label is reranked, or a row_idx misalignment feeds the WRONG
     chunk text to the reranker) -> plumbing bug.
  3. Gold's passage is real, on-topic text but genuinely scores low -> a true
     reranker limitation (gold is often the generic scope article 'Điều 1');
     fix is query/passage construction, not plumbing.

This script rebuilds the pipeline and, for a few question ids, prints:

  - lexical / dense hit counts            (is the dense leg ALIVE?)
  - harvested chunk count vs rerank_pool  (is truncation evicting candidates?)
  - whether each gold article number was harvested at all, and if so with what
    pre-rerank base_score, what rerank score, and the first 160 chars of the
    EXACT passage text the reranker saw (so we can see if it degraded to a bare
    label or got the wrong chunk)
  - the reranked top-10 with passage snippets, gold rows marked >>>.

Usage (on Kaggle, after the Stage-6 dataset is attached)::

    !cd {REPO_DIR} && python dev_set/diagnose_live.py \
        --data {DATA_DIR} --dev dev_set/ground_truth.json --ids 17,8,10

Defaults mirror the notebook's DocAnchorConfig so the probe matches the run that
produced the 0/20 pool.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from retrieval.doc_anchor import (  # noqa: E402
    attach_rank_scores,
    anchor_documents,
    harvest_articles,
)

_DIEU_RE = re.compile(r"Điều\s*(\d+[a-zA-Z]*)", re.UNICODE)


def num(s: str) -> str:
    m = _DIEU_RE.search(str(s))
    return ("Điều " + m.group(1)) if m else str(s).strip()


def gold_numbers(rec: Dict) -> set:
    out = set()
    for a in rec.get("relevant_articles", []):
        parts = str(a).split("|")
        if len(parts) >= 3 and parts[2].strip():
            out.add(num(parts[2]))
    return out


def snippet(t: Optional[str], n: int = 160) -> str:
    if not t:
        return "<EMPTY>"
    return repr(t[:n])


def diagnose(pipe, dev_path: str, ids: Sequence[int]) -> None:
    """Diagnose the bury-gold root cause REUSING an already-built pipeline.

    Call this from the notebook kernel with the ``pipe`` you already built so no
    second copy of the GPU models is loaded (building a second pipeline is what
    OOMs Kaggle — FAISS + BGE-m3 + reranker is ~7 GB and the kernel already
    holds one). Example::

        from dev_set.diagnose_live import diagnose
        diagnose(pipe, f"{DEV_DIR}/ground_truth.json", [17, 8, 10])
    """
    r = pipe.retriever
    cfg = r.cfg
    want = {int(x) for x in ids}
    gts = json.loads(Path(dev_path).read_text(encoding="utf-8"))
    gt_by_id = {int(rec["id"]): rec for rec in gts}

    for qid in sorted(want):
        rec = gt_by_id.get(qid)
        if rec is None:
            print("\n=== id %d NOT in ground truth ===" % qid)
            continue
        q = rec.get("question", "")
        gold = gold_numbers(rec)
        print("\n" + "=" * 78)
        print("id %d  gold=%s" % (qid, ",".join(sorted(gold))))
        print("Q:", q)

        lex = r.lexical_search(q, cfg.top_bm25) or []
        den = r.dense_search(q, cfg.top_dense) if r.dense_search else []
        den = den or []
        print("[legs] lexical=%d  dense=%d   %s"
              % (len(lex), len(den),
                 "<<< DENSE LEG IS DEAD" if len(den) == 0 else ""))

        attach_rank_scores(lex, cfg.rrf_k)
        attach_rank_scores(den, cfg.rrf_k)
        anchored = anchor_documents(lex, den, cfg)
        harv = harvest_articles(lex, den, anchored, cfg)
        print("[harvest] %d chunks (rerank_pool=%d, truncation %s)"
              % (len(harv), cfg.rerank_pool,
                 "BINDING" if len(harv) > cfg.rerank_pool else "not binding"))

        # Did gold get harvested at all, and from which doc?
        harv_gold = [h for h in harv if num(h.dieu_so) in gold]
        if not harv_gold:
            print("[gold] NOT harvested into the pre-rerank pool at all "
                  "(retrieval/harvest miss).")
        else:
            print("[gold] harvested as %d chunk(s):" % len(harv_gold))
            for h in harv_gold[:6]:
                print("   %s  law_id=%s  row_idx=%d  base_score=%.5f"
                      % (num(h.dieu_so), h.law_id, h.row_idx, h.base_score))

        # Apply the same truncation the live pipeline uses, then fetch the EXACT
        # passages the reranker scores and rerank them.
        harv_sorted = sorted(harv, key=lambda h: h.base_score, reverse=True)
        harv_trunc = harv_sorted[: cfg.rerank_pool]
        row_idxs = [h.row_idx for h in harv_trunc if h.row_idx >= 0]
        texts = r.fetch_text(row_idxs) if r.fetch_text else {}
        passages = [texts.get(h.row_idx, h.dieu_so) for h in harv_trunc]
        scores = r.rerank(q, passages) if r.rerank else [h.base_score for h in harv_trunc]

        degraded = sum(
            1 for h, pa in zip(harv_trunc, passages)
            if (pa is None) or (pa == "") or (pa == h.dieu_so)
        )
        print("[passages] %d/%d degraded to bare label (fetch_text miss)"
              % (degraded, len(harv_trunc)))

        # Was gold truncated out before reranking?
        if harv_gold and not any(num(h.dieu_so) in gold for h in harv_trunc):
            print("[gold] harvested but TRUNCATED OUT by rerank_pool before "
                  "reranking (raise rerank_pool / per_doc_articles).")

        scored = list(zip(harv_trunc, passages, scores))
        scored.sort(key=lambda t: t[2], reverse=True)

        print("[rerank] top-10 (>>> marks gold):")
        for rank, (h, pa, sc) in enumerate(scored[:10]):
            mark = ">>>" if num(h.dieu_so) in gold else "   "
            print("  %s #%2d  %-9s score=%.5f  %s"
                  % (mark, rank, num(h.dieu_so), float(sc), snippet(pa)))

        # Where did gold actually land, with its passage?
        for rank, (h, pa, sc) in enumerate(scored):
            if num(h.dieu_so) in gold:
                print("  >>> GOLD %s at rerank rank %d  score=%.5f"
                      % (num(h.dieu_so), rank, float(sc)))
                print("      passage: %s" % snippet(pa, 300))
                break

    # NOTE: do NOT close the pipeline here — the caller owns its lifecycle and
    # typically keeps using it after the diagnostic.


if __name__ == "__main__":
    print(
        "diagnose_live is an in-kernel helper to avoid a second GPU model load.\n"
        "Run it from the notebook that already built `pipe`:\n\n"
        "    from dev_set.diagnose_live import diagnose\n"
        "    diagnose(pipe, f'{DEV_DIR}/ground_truth.json', [17, 8, 10])\n"
    )
    raise SystemExit(0)
