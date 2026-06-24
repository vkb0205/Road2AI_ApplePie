"""Offline F2 evaluation for the G-LRAG dev set.

Implements the F2 macro metric from G-LRAG_SPECIFICATIONS.md §14.3 and the
local evaluation procedure for PLAN.md tasks 7.5 / 10.1.

Usage::

    python dev_set/eval.py \
        --predictions dev_set/results_baseline.json \
        --ground-truth dev_set/ground_truth.json \
        [--report dev_set/eval_log.md]

The predictions file is a JSON array of records each containing at least
``id`` and ``relevant_articles`` (a list of ``"{law_id}|{ten_van_ban}|{dieu_so}"``
strings). The ground-truth file has the same shape. F2 is macro-averaged
across questions, with the same edge cases as the spec (both-empty → 1.0,
one-empty → 0.0).

The official grader extracts predicted articles from the *answer* field using
the ``Điều X`` regex; here we also provide ``f2_from_answers`` which extracts
citations from free-text answers so retrieval-only runs (no generation) can
still be scored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Match "Điều 12", "Điều 12a", "Điều 12.3" inside an answer.
_DIEU_RE = re.compile(r"Điều\s*(\d+[a-zA-Z]*(?:[.\-]\d+)?)", re.UNICODE)


def _canonical_dieu(s: str) -> str:
    """Canonicalise a ``Điều X`` string to ``"Điều <number>"`` (single space).

    Collapses internal whitespace so ``"Điều  5"`` and ``"Điều5"`` both become
    ``"Điều 5"``. This makes grader-extracted citations comparable to gold
    ``relevant_articles`` third segments regardless of spacing.
    """
    s = str(s).strip()
    # Normalise: ensure exactly one space between "Điều" and the number.
    m = re.match(r"^(Điều)\s*(.+)$", s, re.UNICODE)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    return s


def _normalise_article_set(articles: Sequence[str]) -> set:
    """Normalise a list of ``law|ten|dieu`` strings to a set of ``Điều X`` values.

    The official grader normalises to ``Điều X`` (the third segment). We mirror
    that here so a prediction can be compared against ground truth even when the
    full ``law|ten|dieu`` strings differ in the first two segments (e.g. slight
    title wording differences). When the third segment is missing we keep the
    full string for an exact-match fallback.
    """
    out = set()
    for a in articles:
        a = str(a).strip()
        if not a:
            continue
        parts = a.split("|")
        if len(parts) >= 3:
            dieu = parts[2].strip()
            if dieu:
                out.add(_canonical_dieu(dieu))
        else:
            out.add(a)
    return out


def f2_single(pred_set: set, gt_set: set) -> float:
    """F2 for a single question (SPEC §14.3 edge cases)."""
    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0
    tp = len(pred_set & gt_set)
    p = tp / len(pred_set)
    r = tp / len(gt_set)
    if p + r == 0:
        return 0.0
    return 5 * p * r / (4 * p + r)


def f2_macro(predictions: Sequence[Dict], ground_truth: Sequence[Dict]) -> float:
    """Macro-averaged F2 across questions, keyed by ``id`` (SPEC §14.3)."""
    gt_by_id = {int(r["id"]): r for r in ground_truth}
    scores: List[float] = []
    for pred in predictions:
        pid = int(pred["id"])
        gt = gt_by_id.get(pid)
        if gt is None:
            raise KeyError(f"f2_macro: prediction id={pid} not in ground truth")
        pred_set = _normalise_article_set(pred.get("relevant_articles", []))
        gt_set = _normalise_article_set(gt.get("relevant_articles", []))
        scores.append(f2_single(pred_set, gt_set))
    return sum(scores) / len(scores) if scores else 0.0


def extract_articles_from_answer(answer: str) -> set:
    """Extract ``Điều X`` citations from a free-text answer (grader regex).

    Each match is canonicalised to ``"Điều <number>"`` (single space) so it is
    directly comparable to the normalised gold ``relevant_articles`` set.
    """
    return {
        _canonical_dieu(m.group(0))
        for m in _DIEU_RE.finditer(answer or "")
    }


def f2_from_answers(predictions: Sequence[Dict], ground_truth: Sequence[Dict]) -> float:
    """F2 macro where predicted articles are extracted from the ``answer`` field.

    Mirrors the official grader: it pulls ``Điều X`` out of the answer text and
    matches against the gold ``relevant_articles`` normalised to ``Điều X``.
    """
    gt_by_id = {int(r["id"]): r for r in ground_truth}
    scores: List[float] = []
    for pred in predictions:
        pid = int(pred["id"])
        gt = gt_by_id.get(pid)
        if gt is None:
            raise KeyError(f"f2_from_answers: prediction id={pid} not in ground truth")
        pred_set = extract_articles_from_answer(pred.get("answer", ""))
        gt_set = _normalise_article_set(gt.get("relevant_articles", []))
        scores.append(f2_single(pred_set, gt_set))
    return sum(scores) / len(scores) if scores else 0.0


def grounding_rate(predictions: Sequence[Dict], ground_truth: Sequence[Dict]) -> float:
    """Proportion of questions with ≥1 correctly cited article (SPEC §14.1)."""
    gt_by_id = {int(r["id"]): r for r in ground_truth}
    hits = 0
    for pred in predictions:
        gt = gt_by_id.get(int(pred["id"]))
        if gt is None:
            continue
        pred_set = extract_articles_from_answer(pred.get("answer", ""))
        gt_set = _normalise_article_set(gt.get("relevant_articles", []))
        if pred_set & gt_set:
            hits += 1
    return hits / len(predictions) if predictions else 0.0


def _load_json(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="G-LRAG offline F2 evaluator")
    p.add_argument("--predictions", required=True, help="path to results JSON (list of records)")
    p.add_argument("--ground-truth", required=True, help="path to ground_truth JSON")
    p.add_argument("--report", default=None, help="optional path to write a markdown report")
    p.add_argument("--from-answers", action="store_true",
                   help="extract predicted articles from the 'answer' field (grader mode)")
    args = p.parse_args(argv)

    preds = _load_json(Path(args.predictions))
    gts = _load_json(Path(args.ground_truth))

    if args.from_answers:
        score = f2_from_answers(preds, gts)
        mode = "answer-citation"
    else:
        score = f2_macro(preds, gts)
        mode = "relevant_articles"

    gr = grounding_rate(preds, gts)
    line = f"F2 macro ({mode}): {score:.4f} | grounding_rate: {gr:.4f} | n={len(preds)}"
    print(line)

    if args.report:
        Path(args.report).write_text(
            f"# G-LRAG dev-set evaluation\n\n"
            f"- Predictions: `{args.predictions}`\n"
            f"- Ground truth: `{args.ground_truth}`\n"
            f"- Mode: {mode}\n"
            f"- Questions: {len(preds)}\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| F2 macro | {score:.4f} |\n"
            f"| Grounding rate | {gr:.4f} |\n\n"
            f"_Run: {line}_\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
