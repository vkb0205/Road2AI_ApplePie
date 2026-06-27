"""Build a *demonstration* candidate-pool fixture for offline selector testing.

This does NOT run retrieval (the dense leg needs a GPU). Instead it reconstructs,
from the real ``ground_truth.json``, the kind of post-rerank candidate pool the
redesigned Kaggle pipeline is expected to emit:

  - the gold article(s) appear in the pool (because doc-anchored dense retrieval
    + article rerank recovers them) with strong-but-not-perfect rerank scores;
  - realistic distractor noise is injected: provincial decisions (``QĐ-UBND``),
    a superseded same-topic decree, and an off-topic central article.

It is a *modelling* tool: it lets us validate that the selector + harness turn a
recall-recovered pool into a high macro F2, and lets the user tune
``SelectConfig`` knobs before spending GPU time. Real pools dumped on Kaggle
replace this fixture verbatim (same schema).

Usage::

    python dev_set/make_demo_pool.py \
        --ground-truth dev_set/ground_truth.json \
        --out dev_set/demo_pool.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


# A small bank of realistic distractor documents drawn from the observed
# baseline dumps (provincial decisions + superseded registration decrees).
_PROVINCIAL_NOISE = [
    ("1934/2007/QĐ-UBND", "Quyết định 1934/2007/QĐ-UBND ... đăng ký kinh doanh"),
    ("15/2009/QĐ-UBND", "Quyết định 15/2009/QĐ-UBND ... một cửa liên thông"),
    ("72/2002/QĐ-UB", "Quyết định 72/2002/QĐ-UB ... đăng ký đất đai"),
]
_SUPERSEDED_NOISE = [
    ("02/2000/NĐ-CP", "Nghị định 02/2000/NĐ-CP Về đăng ký kinh doanh"),
    ("109/2004/NĐ-CP", "Nghị định 109/2004/NĐ-CP Về đăng ký kinh doanh"),
]
_OFFTOPIC_NOISE = [
    ("98/2021/NĐ-CP", "Nghị định 98/2021/NĐ-CP Về quản lý trang thiết bị y tế"),
    ("2396/QĐ-TCHQ", "Quyết định 2396/QĐ-TCHQ ... thủ tục hải quan điện tử"),
]


def _parse_gold(gt_record: Dict) -> List[Dict]:
    """Return [{law_id, ten_van_ban, dieu}] for the gold articles."""
    out = []
    for a in gt_record.get("relevant_articles", []):
        parts = str(a).split("|")
        if len(parts) >= 3:
            out.append(
                {"law_id": parts[0], "ten_van_ban": parts[1], "dieu": parts[2]}
            )
    return out


def build_pool(ground_truth: List[Dict], seed: int = 13) -> List[Dict]:
    rng = random.Random(seed)
    pool: List[Dict] = []
    for gt in ground_truth:
        gold = _parse_gold(gt)
        cands: List[Dict] = []

        # Gold articles: strong rerank scores, slightly decreasing per extra
        # gold article so the K policy must admit a close second when n_gold==2.
        for i, g in enumerate(gold):
            cands.append(
                {
                    "law_id": g["law_id"],
                    "ten_van_ban": g["ten_van_ban"],
                    "dieu_so": g["dieu"],
                    "score": round(0.80 - 0.03 * i + rng.uniform(-0.01, 0.01), 4),
                }
            )

        # Provincial noise: high raw lexical pull but the reranker leaves them
        # mid/high; suppression must remove them.
        for law_id, ten in rng.sample(_PROVINCIAL_NOISE, k=2):
            cands.append(
                {
                    "law_id": law_id,
                    "ten_van_ban": ten,
                    "dieu_so": f"Điều {rng.randint(2, 9)}",
                    "score": round(rng.uniform(0.55, 0.75), 4),
                }
            )

        # Superseded same-topic decree: on-topic, mid score; authority prior
        # (recency) should rank it below the current decree.
        law_id, ten = rng.choice(_SUPERSEDED_NOISE)
        cands.append(
            {
                "law_id": law_id,
                "ten_van_ban": ten,
                "dieu_so": f"Điều {rng.randint(4, 12)}",
                "score": round(rng.uniform(0.50, 0.66), 4),
            }
        )

        # Off-topic central article: low score.
        law_id, ten = rng.choice(_OFFTOPIC_NOISE)
        cands.append(
            {
                "law_id": law_id,
                "ten_van_ban": ten,
                "dieu_so": f"Điều {rng.randint(20, 40)}",
                "score": round(rng.uniform(0.30, 0.45), 4),
            }
        )

        rng.shuffle(cands)
        pool.append(
            {"id": int(gt["id"]), "question": gt.get("question", ""), "candidates": cands}
        )
    return pool


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build demo candidate pool")
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args(argv)

    gts = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    pool = build_pool(gts, seed=args.seed)
    Path(args.out).write_text(
        json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(pool)} pool records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
