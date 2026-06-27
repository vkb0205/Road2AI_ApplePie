"""Candidate-pool tuning + scoring harness for the F2 article selector.

The redesigned Kaggle pipeline dumps, per dev question, the *reranked candidate
pool* (more than the final K) so selection policy can be tuned offline without a
GPU. This harness:

  1. Loads such a pool dump + ground truth.
  2. Runs :func:`retrieval.article_select.select_articles` per question.
  3. Scores P / R / F2 with the grader-faithful (number-only) normalisation.
  4. Optionally grid-searches the :class:`SelectConfig` knobs and prints the
     best configuration by macro F2.

Pool dump schema (JSON array)::

    [
      {
        "id": 1,
        "question": "...",
        "candidates": [
          {"law_id": "01/2021/NĐ-CP", "ten_van_ban": "Nghị định ...",
           "dieu_so": "Điều 12", "score": 0.83},
          ...
        ]
      },
      ...
    ]

It also reports the **recall ceiling**: the fraction of questions whose gold
article number appears *anywhere* in the candidate pool. The selector can never
beat this ceiling, so it isolates retrieval-side vs selection-side headroom.

Pure stdlib; runs in the local numpy-only env and on Kaggle.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from retrieval.article_select import (  # noqa: E402
    ArticleCandidate,
    AuthorityConfig,
    SelectConfig,
    canonical_dieu,
    is_provincial,
    select_articles,
)


# --------------------------------------------------------------------------- #
# Grader-faithful scoring (number-only)
# --------------------------------------------------------------------------- #
def _gold_dieu_set(gt_record: Dict) -> set:
    out = set()
    for a in gt_record.get("relevant_articles", []):
        parts = str(a).split("|")
        if len(parts) >= 3 and parts[2].strip():
            out.add(canonical_dieu(parts[2]))
    return out


def _prf2(pred: set, gold: set) -> Tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f2 = 0.0 if (p + r) == 0 else 5 * p * r / (4 * p + r)
    return p, r, f2


def _load(path: str) -> List[Dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _candidates(pool_record: Dict) -> List[ArticleCandidate]:
    out: List[ArticleCandidate] = []
    for c in pool_record.get("candidates", []):
        out.append(
            ArticleCandidate(
                law_id=str(c.get("law_id", "")),
                ten_van_ban=str(c.get("ten_van_ban", "")),
                dieu_so=str(c.get("dieu_so", "")),
                score=float(c.get("score", 0.0)),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    pool: Sequence[Dict], ground_truth: Sequence[Dict], cfg: SelectConfig
) -> Dict[str, float]:
    """Run the selector over a pool and return macro metrics + ceilings."""
    gt_by_id = {int(r["id"]): r for r in ground_truth}
    n = 0
    sum_p = sum_r = sum_f2 = 0.0
    correct_q = 0
    sum_k = 0
    ceiling_hits = 0  # gold present anywhere in (non-suppressed) pool
    raw_ceiling_hits = 0  # gold present anywhere in raw pool (no suppression)

    for rec in pool:
        pid = int(rec["id"])
        gt = gt_by_id.get(pid)
        if gt is None:
            raise KeyError(f"pool id={pid} not in ground truth")
        gold = _gold_dieu_set(gt)
        cands = _candidates(rec)

        # Recall ceilings.
        raw_pool_dieus = {canonical_dieu(c.dieu_so) for c in cands}
        if gold & raw_pool_dieus:
            raw_ceiling_hits += 1
        kept_dieus = {
            canonical_dieu(c.dieu_so)
            for c in cands
            if not (cfg.drop_provincial and is_provincial(c.law_id, cfg.authority))
        }
        if gold & kept_dieus:
            ceiling_hits += 1

        chosen = select_articles(cands, cfg)
        pred = {a.dieu for a in chosen}
        p, r, f2 = _prf2(pred, gold)
        sum_p += p
        sum_r += r
        sum_f2 += f2
        sum_k += len(pred)
        if pred & gold:
            correct_q += 1
        n += 1

    n = n or 1
    return {
        "f2": sum_f2 / n,
        "precision": sum_p / n,
        "recall": sum_r / n,
        "avg_k": sum_k / n,
        "correct_q": correct_q,
        "n": n - 0 if n else 0,
        "recall_ceiling": ceiling_hits / n,
        "raw_recall_ceiling": raw_ceiling_hits / n,
    }


def per_question(
    pool: Sequence[Dict], ground_truth: Sequence[Dict], cfg: SelectConfig
) -> List[Dict]:
    gt_by_id = {int(r["id"]): r for r in ground_truth}
    rows: List[Dict] = []
    for rec in pool:
        pid = int(rec["id"])
        gold = _gold_dieu_set(gt_by_id[pid])
        cands = _candidates(rec)
        chosen = select_articles(cands, cfg)
        pred = {a.dieu for a in chosen}
        p, r, f2 = _prf2(pred, gold)
        rows.append(
            {
                "id": pid,
                "precision": p,
                "recall": r,
                "f2": f2,
                "pred": sorted(pred),
                "gold": sorted(gold),
                "missing": sorted(gold - pred),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Grid search
# --------------------------------------------------------------------------- #
def _grid_configs() -> List[SelectConfig]:
    cfgs: List[SelectConfig] = []
    for drop_prov in (True, False):
        for max_k in (2, 3):
            for rel in (0.10, 0.15, 0.20, 0.25):
                for ab in (0.08, 0.12, 0.16):
                    cfgs.append(
                        SelectConfig(
                            authority=AuthorityConfig(),
                            drop_provincial=drop_prov,
                            max_k=max_k,
                            min_k=1,
                            rel_margin=rel,
                            abs_margin=ab,
                        )
                    )
    return cfgs


def grid_search(
    pool: Sequence[Dict], ground_truth: Sequence[Dict]
) -> Tuple[SelectConfig, Dict[str, float], List[Tuple[SelectConfig, Dict]]]:
    """Return (best_cfg, best_metrics, all_results sorted by F2 desc)."""
    results: List[Tuple[SelectConfig, Dict]] = []
    for cfg in _grid_configs():
        m = evaluate(pool, ground_truth, cfg)
        results.append((cfg, m))
    results.sort(key=lambda t: (t[1]["f2"], t[1]["recall"]), reverse=True)
    best_cfg, best_m = results[0]
    return best_cfg, best_m, results


def _fmt_cfg(cfg: SelectConfig) -> str:
    return (
        f"drop_prov={cfg.drop_provincial} max_k={cfg.max_k}"
        f" rel={cfg.rel_margin} abs={cfg.abs_margin}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="F2 selector tuning/scoring harness")
    p.add_argument("--pool", required=True, help="candidate-pool dump JSON")
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--grid", action="store_true", help="grid-search SelectConfig")
    p.add_argument("--top", type=int, default=5, help="grid: show top-N configs")
    p.add_argument("--per-question", action="store_true")
    args = p.parse_args(argv)

    pool = _load(args.pool)
    gts = _load(args.ground_truth)

    if args.grid:
        best_cfg, best_m, results = grid_search(pool, gts)
        print("Top configurations by macro F2:")
        for cfg, m in results[: args.top]:
            print(
                f"  F2={m['f2']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}"
                f" avg_K={m['avg_k']:.2f} correct={m['correct_q']}/{m['n']}"
                f"  [{_fmt_cfg(cfg)}]"
            )
        print(
            f"\nRecall ceiling (gold in pool): "
            f"raw={results[0][1]['raw_recall_ceiling']:.4f}"
            f"  after-suppression={results[0][1]['recall_ceiling']:.4f}"
        )
        cfg = best_cfg
    else:
        cfg = SelectConfig()
        m = evaluate(pool, gts, cfg)
        print(
            f"F2={m['f2']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}"
            f" avg_K={m['avg_k']:.2f} correct={m['correct_q']}/{m['n']}"
        )
        print(
            f"Recall ceiling (gold in pool): raw={m['raw_recall_ceiling']:.4f}"
            f"  after-suppression={m['recall_ceiling']:.4f}"
        )

    if args.per_question:
        print(f"\n{'id':>3} {'P':>5} {'R':>5} {'F2':>5}  pred -> missing")
        for x in per_question(pool, gts, cfg):
            print(
                f"{x['id']:>3} {x['precision']:>5.2f} {x['recall']:>5.2f}"
                f" {x['f2']:>5.2f}  {','.join(x['pred']) or '-'}"
                f"  ->  {','.join(x['missing']) or '-'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
