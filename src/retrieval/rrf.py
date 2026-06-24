"""Reciprocal Rank Fusion (RRF) for the G-LRAG retrieval pipeline.

Implements PLAN.md task 7.2: RRF fusion with ``k=60``, ``fused_top=30``,
deterministic ordering (score desc, row_idx asc, seeded RNG tiebreak,
seed 42).

The fusion is backend-agnostic: each input ranking is a list of either
plain ``row_idx`` integers or dicts that *must* contain a ``"row_idx"``
key (optionally with arbitrary payload fields). Fused output preserves the
carried payload and adds an ``rrf_score`` field.

Heavy GPU deps are never imported here — RRF is pure Python.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Union

__all__ = ["rrf_fuse", "rrf_score_only", "DEFAULT_K"]

DEFAULT_K = 60
DEFAULT_SEED = 42

# A ranking element is either an int (row_idx) or a mapping with a "row_idx" key.
RankElem = Union[int, Dict[str, Any]]


def _extract_row_idx(elem: RankElem) -> int:
    """Return the integer ``row_idx`` from a ranking element."""
    if isinstance(elem, dict):
        if "row_idx" not in elem:
            raise KeyError(
                "rrf_fuse: ranking element dict is missing the required "
                "'row_idx' key"
            )
        return int(elem["row_idx"])
    return int(elem)


def rrf_fuse(
    rankings: Sequence[Sequence[RankElem]],
    k: int = DEFAULT_K,
    fused_top: int = 30,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion.

    Parameters
    ----------
    rankings:
        A sequence of ranking lists. Each inner list is ordered best→worst.
        Elements are ``row_idx`` ints or dicts carrying ``"row_idx"`` plus an
        optional payload (the payload of the first list a row appears in is
        kept).
    k:
        RRF smoothing constant. Must be ``> 0``. The score contribution of a
        hit at 1-based rank ``r`` is ``1 / (k + r)``.
    fused_top:
        Truncate the fused output to this many results.
    seed:
        Seed for the deterministic RNG tiebreak used when scores tie.

    Returns
    -------
    list of dict
        Each dict is ``{row_idx, rrf_score, source, payload}`` where
        ``source`` is ``"fused"`` and ``payload`` is the carried element (an
        int → ``None``, a dict → the original dict). Output is sorted by
        ``(rrf_score desc, row_idx asc)`` with a seeded RNG tiebreak so that
        identical-score collisions break deterministically across runs.

    Raises
    ------
    ValueError
        If ``k <= 0``.
    KeyError
        If a dict element lacks ``"row_idx"``.
    """
    if k <= 0:
        raise ValueError(f"rrf_fuse: k must be > 0, got {k}")

    scores: Dict[int, float] = defaultdict(float)
    payloads: Dict[int, Any] = {}

    for ranking in rankings:
        for rank, elem in enumerate(ranking, start=1):
            row_idx = _extract_row_idx(elem)
            scores[row_idx] += 1.0 / (k + rank)
            if row_idx not in payloads:
                payloads[row_idx] = elem if isinstance(elem, dict) else None

    # Deterministic ordering: score desc, then row_idx asc, then seeded RNG
    # tiebreak for any remaining exact ties (so re-runs with the same seed are
    # bit-identical, satisfying test_determinism).
    rng = random.Random(seed)
    fused = [
        {
            "row_idx": row_idx,
            "rrf_score": float(score),
            "source": "fused",
            "payload": payloads[row_idx],
        }
        for row_idx, score in scores.items()
    ]
    fused.sort(
        key=lambda r: (
            -r["rrf_score"],
            r["row_idx"],
            rng.random(),
        )
    )
    return fused[:fused_top]


def rrf_score_only(
    rankings: Sequence[Sequence[RankElem]],
    k: int = DEFAULT_K,
) -> Dict[int, float]:
    """Return a raw ``{row_idx: rrf_score}`` mapping (no truncation, no payload).

    Useful for tests and ablations where only the aggregate scores matter.
    """
    if k <= 0:
        raise ValueError(f"rrf_score_only: k must be > 0, got {k}")
    scores: Dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, elem in enumerate(ranking, start=1):
            row_idx = _extract_row_idx(elem)
            scores[row_idx] += 1.0 / (k + rank)
    return dict(scores)
