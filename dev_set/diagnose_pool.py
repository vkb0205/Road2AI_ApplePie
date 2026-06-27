"""Diagnose WHY the F2 selector scores 0/20 despite a 0.80 recall ceiling.

Pure stdlib, no GPU / no live pipeline. Reads the dumped candidate pool and the
ground truth, then for every question reports:

  - whether the gold article NUMBER is present anywhere in the pool;
  - if present, at what RANK (rerank-score order, 0 = top) and with what SCORE;
  - the top-5 article numbers the reranker put first;
  - the score GAP between the pool top and the gold candidate.

This isolates the real bottleneck:

  * gold NOT_IN_POOL for the failing questions  -> retrieval recall problem
    (widen TOP_DENSE / PER_DOC_ARTICLES / TOP_DOCS, or the dense leg is dead).
  * gold IN pool but at rank >= MAX_K with a score far below the top
    -> the cross-encoder cannot lift gold above its siblings (rerank-quality /
    passage-quality problem; raising MAX_K only partly helps).
  * gold IN pool at rank < MAX_K -> a selection/threshold problem.

Usage::

    python dev_set/diagnose_pool.py \
        --pool /kaggle/working/pool.json \
        --ground-truth dev_set/ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_DIEU_RE = re.compile(r"Điều\s*(\d+[a-zA-Z]*)", re.UNICODE)


def num(s: str) -> str:
    """Reduce any 'Điều X ...' string to canonical 'Điều <number>'."""
    m = _DIEU_RE.search(str(s))
    return ("Điều " + m.group(1)) if m else str(s).strip()


def gold_numbers(gt_record: Dict) -> set:
    out = set()
    for a in gt_record.get("relevant_articles", []):
        parts = str(a).split("|")
        if len(parts) >= 3 and parts[2].strip():
            out.add(num(parts[2]))
    return out


def load(path: str) -> List[Dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def diagnose(
    pool: Sequence[Dict], ground_truth: Sequence[Dict]
) -> Tuple[List[Dict], Dict[str, float]]:
    gt_by_id = {int(r["id"]): r for r in ground_truth}
    rows: List[Dict] = []
    in_pool = 0
    in_top1 = 0
    in_top2 = 0
    in_top5 = 0
    for rec in pool:
        pid = int(rec["id"])
        gold = gold_numbers(gt_by_id[pid])
        cand = rec.get("candidates", [])
        # First (best) rank + score per article number (pool is score-desc).
        rank_of: Dict[str, Tuple[int, float]] = {}
        for i, c in enumerate(cand):
            key = num(c.get("dieu_so", ""))
            if key not in rank_of:
                rank_of[key] = (i, float(c.get("score", 0.0)))
        top_score = float(cand[0]["score"]) if cand else 0.0
        # Best (lowest-rank) gold hit.
        best_rank: Optional[int] = None
        best_score: Optional[float] = None
        for g in gold:
            if g in rank_of:
                r_i, r_s = rank_of[g]
                if best_rank is None or r_i < best_rank:
                    best_rank, best_score = r_i, r_s
        present = best_rank is not None
        in_pool += int(present)
        if present:
            in_top1 += int(best_rank < 1)
            in_top2 += int(best_rank < 2)
            in_top5 += int(best_rank < 5)
        rows.append(
            {
                "id": pid,
                "pool_size": len(cand),
                "gold": sorted(gold),
                "top5": [num(c.get("dieu_so", "")) for c in cand[:5]],
                "gold_rank": best_rank,
                "gold_score": best_score,
                "top_score": round(top_score, 4),
                "gap_from_top": (
                    round(top_score - best_score, 4) if best_score is not None else None
                ),
            }
        )
    n = len(pool) or 1
    summary = {
        "n": len(pool),
        "gold_in_pool": in_pool,
        "gold_at_rank0": in_top1,
        "gold_in_top2": in_top2,
        "gold_in_top5": in_top5,
        "recall_ceiling": round(in_pool / n, 4),
    }
    return rows, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Pool / rerank-rank diagnostic")
    p.add_argument("--pool", required=True)
    p.add_argument("--ground-truth", required=True)
    args = p.parse_args(argv)

    pool = load(args.pool)
    gts = load(args.ground_truth)
    rows, summary = diagnose(pool, gts)

    print(
        "SUMMARY  n=%d  gold_in_pool=%d  gold@rank0=%d  gold_in_top2=%d"
        "  gold_in_top5=%d  recall_ceiling=%.3f"
        % (
            summary["n"],
            summary["gold_in_pool"],
            summary["gold_at_rank0"],
            summary["gold_in_top2"],
            summary["gold_in_top5"],
            summary["recall_ceiling"],
        )
    )
    print(
        "\n%3s %5s %9s %9s %9s %9s  %s"
        % ("id", "pool", "goldRank", "goldScr", "topScr", "gap", "gold / top5")
    )
    for x in rows:
        gr = "MISS" if x["gold_rank"] is None else str(x["gold_rank"])
        gs = "-" if x["gold_score"] is None else "%.4f" % x["gold_score"]
        gp = "-" if x["gap_from_top"] is None else "%.4f" % x["gap_from_top"]
        print(
            "%3d %5d %9s %9s %9.4f %9s  gold=%s top5=%s"
            % (
                x["id"],
                x["pool_size"],
                gr,
                gs,
                x["top_score"],
                gp,
                ",".join(x["gold"]),
                ",".join(x["top5"]),
            )
        )

    print("\nINTERPRETATION")
    miss = summary["gold_in_pool"]
    n = summary["n"]
    if miss < n:
        print(
            "  - %d/%d questions have gold MISSING from the pool entirely -> "
            "retrieval recall problem (dense leg dead, or widen "
            "TOP_DENSE / PER_DOC_ARTICLES / TOP_DOCS)." % (n - miss, n)
        )
    if summary["gold_in_pool"] and summary["gold_in_top5"] < summary["gold_in_pool"]:
        print(
            "  - gold is IN the pool but below rerank rank 5 for "
            "%d question(s) -> cross-encoder cannot lift gold above siblings "
            "(rerank / passage-quality problem; raising MAX_K helps only partly)."
            % (summary["gold_in_pool"] - summary["gold_in_top5"])
        )
    if summary["gold_in_top2"] and summary["gold_in_top2"] >= summary["gold_in_pool"]:
        print(
            "  - gold sits within rerank top-2 wherever present -> a selection/"
            "threshold problem, not retrieval (raise MAX_K / loosen margins)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
