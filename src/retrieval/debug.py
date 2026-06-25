"""Per-stage debug tracing for the G-LRAG retrieval pipeline.

This module gives a lightweight, opt-in observability layer over
:class:`retrieval.retriever.HybridRetriever.retrieve()`. It captures a
:class:`StageSnapshot` for each of the 7 pipeline stages (lexical, dense,
RRF fusion, graph expansion, metadata+text fetch, rerank, final Hit build)
plus the post-step ``make_relevant_lists``, so you can see *exactly* what
flows from one stage to the next when retrieval quality is not what you
expect.

Design goals
------------
* **No heavy deps.** Pure Python (``time``, ``dataclasses``). Imports &
  unit-tests on a CPU-only machine with no ``faiss``/``torch`` installed.
* **Off by default, zero overhead when off.** When ``RetrievalConfig.debug``
  is ``False`` the retriever never constructs a trace and never calls into
  this module — there is no timing overhead and no extra dict allocation on
  the hot path (PLAN.md 7.5 acceptance path is untouched).
* **Structured + human-readable.** Snapshots are dataclasses (so they can be
  serialised / diffed in a notebook), and :func:`format_trace` renders a
  compact, readable block for ``print()`` or logging.
* **Metadata-rich.** Each stage records ``count`` plus a ``top_items`` slice
  carrying ``row_idx``, the stage-native score, and the legal metadata
  (``law_id`` / ``ten_van_ban`` / ``dieu_so``) when available — so you can
  spot at a glance whether a stage is dropping or promoting the wrong
  chunks/articles.
* **Skips & fallbacks explained.** Optional stages (dense, graph, rerank)
  record a ``skip`` reason when they are turned off or degrade, and stage 5
  records which ``row_idx`` could not be resolved from the canonical chunks
  table (a common silent failure mode for the F2 article score).

A :class:`RetrievalTrace` is attached to the retriever as ``last_trace`` at
the end of every ``retrieve()`` call when debug is enabled, and
``HybridRetriever.last_trace_formatted`` renders it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "StageSnapshot",
    "RetrievalTrace",
    "snapshot_items",
    "format_trace",
]

# How many items to keep per stage by default — enough to spot ordering
# problems without flooding a notebook cell.
DEFAULT_TOP_N = 8


# ---------------------------------------------------------------------------
# Snapshot data structures
# ---------------------------------------------------------------------------


@dataclass
class StageSnapshot:
    """A single stage's recorded state inside a :class:`RetrievalTrace`.

    Attributes
    ----------
    name:
        Short stage id, e.g. ``"lexical"``, ``"dense"``, ``"rrf"``,
        ``"graph"``, ``"fetch"``, ``"rerank"``, ``"final"``, ``"output"``.
    count:
        Number of items produced by the stage (before any final truncation
        that happens in a later stage). For ``fetch`` this is the number of
        metadata rows actually resolved from the chunks table.
    top_items:
        A compact list of dicts describing the top-N items. Each dict has at
        least ``row_idx`` and ``score``; legal metadata is included when the
        stage has it. ``source`` is included when the stage tags items.
    elapsed_ms:
        Wall-clock time the stage took, in milliseconds. ``None`` when the
        stage was skipped before any work was done.
    skip:
        If the stage was skipped (flag off, deps missing, no input), a short
        human-readable reason; otherwise ``None``.
    diagnostics:
        Extra stage-specific notes — e.g. ``{"missing_rows": [...]}`` for the
        fetch stage when some ``row_idx`` were not found in the chunks table,
        or ``{"rankings_in": 2}`` for RRF.
    """

    name: str
    count: int = 0
    top_items: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_ms: Optional[float] = None
    skip: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    """A full per-stage trace for one ``retrieve()`` call.

    The retriever appends one :class:`StageSnapshot` per stage, in pipeline
    order. ``query`` and ``config`` are recorded so a trace is self-contained
    and can be diffed against another query's trace.
    """

    query: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    stages: List[StageSnapshot] = field(default_factory=list)
    total_elapsed_ms: Optional[float] = None

    def add(self, snapshot: StageSnapshot) -> None:
        self.stages.append(snapshot)

    def stage(self, name: str) -> Optional[StageSnapshot]:
        """Return the snapshot for ``name`` (last one if duplicated)."""
        for s in reversed(self.stages):
            if s.name == name:
                return s
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_score(v: Any) -> Optional[float]:
    """Best-effort float coercion that tolerates ``None`` / non-numeric."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def snapshot_items(
    items: Sequence[Dict[str, Any]],
    top_n: int = DEFAULT_TOP_N,
    *,
    score_key: str = "score",
    extra_keys: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Reduce a stage's raw item list to a compact ``top_items`` slice.

    Each retained item is a small dict with ``row_idx``, ``score`` (coerced
    from ``score_key``), ``source`` (when present on the item) and any keys
    listed in ``extra_keys`` that exist on the item. Items lacking a
    ``row_idx`` are skipped.

    This normalises the heterogeneous per-stage dicts (``bm25_score``,
    ``dense_score``, ``rrf_score``, ``score`` ...) into a single ``score``
    field so :func:`format_trace` can print them uniformly.
    """
    out: List[Dict[str, Any]] = []
    # Prefer the stage-native score key; fall back to a generic "score".
    for it in items[:top_n]:
        if not isinstance(it, dict):
            continue
        row_idx = it.get("row_idx")
        if row_idx is None:
            continue
        sc = _coerce_score(it.get(score_key))
        if sc is None and score_key != "score":
            sc = _coerce_score(it.get("score"))
        d: Dict[str, Any] = {"row_idx": int(row_idx), "score": sc}
        if "source" in it:
            d["source"] = str(it["source"])
        for k in extra_keys:
            if k in it and k not in d:
                d[k] = it[k]
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt_meta(d: Dict[str, Any]) -> str:
    """Format the legal metadata fragment ``[law_id|ten|dieu]`` if present."""
    law = d.get("law_id")
    ten = d.get("ten_van_ban")
    dieu = d.get("dieu_so")
    if law or ten or dieu:
        return f"[{law or ''}|{ten or ''}|{dieu or ''}]"
    return ""


def _fmt_items(items: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for it in items:
        ri = it.get("row_idx")
        sc = it.get("score")
        sc_s = f"{sc:.4f}" if isinstance(sc, (int, float)) else "—"
        src = it.get("source")
        src_s = f" <{src}>" if src else ""
        meta_s = _fmt_meta(it)
        lines.append(f"    #{ri:>6}  score={sc_s}{src_s} {meta_s}".rstrip())
    return lines


def format_trace(trace: RetrievalTrace, *, top_n: int = DEFAULT_TOP_N) -> str:
    """Render a :class:`RetrievalTrace` as a readable, indented string.

    ``top_n`` caps how many items per stage are printed (the trace may carry
    more in ``top_items``; only the first ``top_n`` are shown).
    """
    if not trace.stages:
        return "(empty retrieval trace)"

    lines: List[str] = []
    q = trace.query or ""
    lines.append("=" * 78)
    lines.append(f"RETRIEVAL TRACE  query={q!r}")
    cfg = trace.config or {}
    if cfg:
        flags = {
            "use_dense": cfg.get("use_dense"),
            "use_reranker": cfg.get("use_reranker"),
            "graph": cfg.get("graph_expander"),
            "top_bm25": cfg.get("top_bm25"),
            "top_dense": cfg.get("top_dense"),
            "fused_top": cfg.get("fused_top"),
            "expanded_top": cfg.get("expanded_top"),
            "final_top_k": cfg.get("final_top_k"),
        }
        lines.append("config: " + "  ".join(f"{k}={v}" for k, v in flags.items()))
    lines.append("=" * 78)

    for snap in trace.stages:
        lines.append("")
        title = f"[{snap.name}] count={snap.count}"
        if snap.elapsed_ms is not None:
            title += f"  {snap.elapsed_ms:.2f}ms"
        lines.append(title)
        if snap.skip:
            lines.append(f"  skipped: {snap.skip}")
        for k, v in snap.diagnostics.items():
            if isinstance(v, (list, tuple)) and len(v) > 8:
                v = list(v[:8]) + [f"... (+{len(v) - 8} more)"]
            lines.append(f"  {k}: {v}")
        shown = snap.top_items[:top_n]
        if shown:
            lines.extend(_fmt_items(shown))
            if len(snap.top_items) > top_n:
                lines.append(
                    f"    ... (+{len(snap.top_items) - top_n} more recorded, "
                    f"of {snap.count})"
                )
        elif snap.count == 0 and not snap.skip:
            lines.append("  (no items)")

    lines.append("")
    lines.append("-" * 78)
    if trace.total_elapsed_ms is not None:
        lines.append(f"total: {trace.total_elapsed_ms:.2f}ms  ({len(trace.stages)} stages)")
    else:
        lines.append(f"({len(trace.stages)} stages)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tiny timing context manager (kept here so the retriever needs no extra import)
# ---------------------------------------------------------------------------


class _StageTimer:
    """Measures wall-clock ms around a ``with`` block.

    Used by the instrumented retriever. The ``elapsed_ms`` attribute is
    populated on exit; if the body is skipped (no ``with`` entered), it stays
    ``None`` — matching :class:`StageSnapshot.elapsed_ms` semantics.
    """

    __slots__ = ("elapsed_ms", "_start")

    def __init__(self) -> None:
        self.elapsed_ms: Optional[float] = None
        self._start: Optional[float] = None

    def __enter__(self) -> "_StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start is not None:
            self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
