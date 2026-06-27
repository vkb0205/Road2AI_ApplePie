"""Per-question F2 / precision / recall diagnostics for the G-LRAG dev set.

This complements ``dev_set/eval.py`` (which prints only the macro F2 scalar) by
breaking the score down per question and per metric, using *exactly* the
official grader's normalisation: every ``law_id|ten_van_ban|Điều X`` string is
reduced to its bare ``Điều X`` number (the third segment) before matching, so a
prediction is correct iff it cites the right article *number* regardless of the
document it came from.

It is intentionally dependency-free (stdlib only) so it runs in the local
numpy-only environment and on Kaggle alike.

Usage::

    python dev_set/f2_diagnostics.py \
        --predictions dev_set/results_baseline.json \
        --ground-truth dev_set/ground_truth.json

    # compare several dumps side by side
    python dev_set/f2_diagnostics.py \
        --ground-truth dev_set/ground_truth.json \
        --predictions dev_set/results_baseline.json dev_set/results_no_graph.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Mirror eval.py exactly so diagnostics never drift from the grader.
_DIEU_RE = re.compile(r"Điều\s*(\d+[a-zA-Z]*(?:[.\-]\d+)?)", re.UNICODE)


def canonical_dieu(s: str) -> str:
    """Canonicalise a ``Điều X`` string to ``"Điều <number>"`` (single space)."""
    s = str(s).strip()
    m = re.match(r"^(Điều)\s*(.+)$", s, re.UNICODE)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    return s


def normalise_article_set(articles: Sequence[str]) -> set:
    """Reduce ``law|ten|dieu`` strings to a set of bare ``Điều X`` numbers."""
    out = set()
    for a in articles:
        a = str(a).strip()
        if not a:
            continue
        parts = a.split("|")
        if len(parts) >= 3:
            dieu = parts[2].strip()
            if dieu:
                out.add(canonical_dieu(dieu))
        else:
            out.add(a)
    return out


def prf2_single(pred_set: set, gt_set: set) -> Tuple[float, float, float]:
    """Return (precision, recall, F2) for one question (SPEC §14.3 edges)."""
    if not pred_set and not gt_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gt_set:
        return 0.0, 0.0, 0.0
    tp = len(pred_set & gt_set)
    p = tp / len(pred_set)
    r = tp / len(gt_set)
    f2 = 0.0 if (p + r) == 0 else 5 * p * r / (4 * p + r)
    return p, r, f2


def per_question(
    predictions: Sequence[Dict], ground_truth: Sequence[Dict]
) -> List[Dict]:
    """Build a per-question diagnostic record list."""
    gt_by_id = {int(r["id"]): r for r in ground_truth}
    rows: List[Dict] = []
    for pred in predictions:
        pid = int(pred["id"])
        gt = gt_by_id.get(pid)
        if gt is None:
            raise KeyError(f"prediction id={pid} not in ground truth")
        pred_set = normalise_article_set(pred.get("relevant_articles", []))
        gt_set = normalise_article_set(gt.get("relevant_articles", []))
        p, r, f2 = prf2_single(pred_set, gt_set)
        rows.append(
            {
                "id": pid,
                "precision": p,
                "recall": r,
                "f2": f2,
                "n_pred": len(pred_set),
                "n_gold": len(gt_set),
                "tp": len(pred_set & gt_set),
                "gold": sorted(gt_set),
                "pred": sorted(pred_set),
                "missing": sorted(gt_set - pred_set),
            }
        )
    return rows


def macro(rows: Sequence[Dict]) -> Dict[str, float]:
    n = len(rows) or 1
    return {
        "precision": sum(x["precision"] for x in rows) / n,
        "recall": sum(x["recall"] for x in rows) / n,
        "f2": sum(x["f2"] for x in rows) / n,
        "n_correct_questions": sum(1 for x in rows if x["tp"] > 0),
        "avg_k": sum(x["n_pred"] for x in rows) / n,
        "n": len(rows),
    }


def _fmt_report(name: str, rows: Sequence[Dict], m: Dict[str, float]) -> str:
    lines = [
        f"== {name} ==",
        f"F2_macro={m['f2']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}"
        f"  correct_q={m['n_correct_questions']}/{m['n']}  avg_K={m['avg_k']:.2f}",
        "",
        f"{'id':>3} {'P':>5} {'R':>5} {'F2':>5} {'tp':>3} {'K':>3} {'gold':>4}  missing",
    ]
    for x in rows:
        lines.append(
            f"{x['id']:>3} {x['precision']:>5.2f} {x['recall']:>5.2f}"
            f" {x['f2']:>5.2f} {x['tp']:>3} {x['n_pred']:>3} {x['n_gold']:>4}"
            f"  {','.join(x['missing']) if x['missing'] else '-'}"
        )
    return "\n".join(lines)


def _load_json(path: str) -> List[Dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="G-LRAG per-question F2 diagnostics")
    p.add_argument("--predictions", nargs="+", required=True,
                   help="one or more results JSON dumps")
    p.add_argument("--ground-truth", required=True)
    args = p.parse_args(argv)

    gts = _load_json(args.ground_truth)
    for pred_path in args.predictions:
        preds = _load_json(pred_path)
        rows = per_question(preds, gts)
        m = macro(rows)
        print(_fmt_report(Path(pred_path).name, rows, m))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
