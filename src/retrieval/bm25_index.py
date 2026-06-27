"""Lexical retrieval backends for G-LRAG.

Two backends are provided:

1. :class:`FTSIndex` — wraps the Stage 6 ``chunk_store.sqlite`` FTS5 virtual
   table (``chunks_fts``). Two query modes:

   * ``fts_fast`` (default) — ``MATCH`` with a quoted-phrase ``OR`` query,
     fast, no global BM25 sort. Used for the CPU baseline.
   * ``bm25_ranked`` — ``ORDER BY bm25(chunks_fts)`` for debug / small top-k.

   The FTS5 invariant ``chunks_fts.rowid == chunks.row_idx`` (see
   ``data/stage6_data/artifacts_guide.md``) means every returned row keeps
   its canonical ``row_idx`` so it aligns with FAISS / metadata sidecar.

2. :class:`PureBM25` — a minimal pure-Python Okapi BM25 over an in-memory
   corpus. Used by unit tests and as a fallback when FTS5 is unavailable.

Query-string builders (:func:`tokenize_query`, :func:`build_fts_match_expr`)
are pure functions so they can be unit-tested without any DB.

No heavy GPU deps are imported.
"""

from __future__ import annotations

import math
import re
import sqlite3
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "FTSIndex",
    "PureBM25",
    "tokenize_query",
    "filter_query_terms",
    "build_fts_match_expr",
]

# Columns returned by FTSIndex.search / fetch_chunks (chunk metadata fields).
_META_COLS = ("row_idx", "chunk_id", "doc_uid", "law_id", "ten_van_ban", "dieu_so")


# ---------------------------------------------------------------------------
# Query-string builders (pure functions)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# High-frequency Vietnamese function words. OR-ing these into an FTS5 MATCH is
# what makes ``bm25_ranked`` pathological: each appears in a large fraction of
# the 636k-chunk corpus, so the match set explodes and ``ORDER BY bm25()`` ends
# up scoring + sorting almost the whole table before LIMIT throws it away
# (measured: a single dev query spent ~319 s in this one stage). Dropping them
# from the OR list keeps only discriminative content terms and brings the
# lexical leg back to milliseconds, with no loss of ranking quality (the whole
# query is still kept as a quoted phrase, which carries the precision signal).
_VI_STOPWORDS = frozenset(
    {
        "và", "là", "của", "có", "được", "cho", "các", "những", "này", "đó",
        "khi", "nào", "gì", "bao", "gồm", "theo", "với", "để", "trong", "ra",
        "vào", "lên", "xuống", "từ", "đến", "tại", "về", "bị", "bởi", "thì",
        "mà", "hay", "hoặc", "nếu", "nên", "sẽ", "đã", "đang", "như", "thế",
        "ai", "đâu", "sao", "bằng", "cũng", "rất", "quá", "lại", "còn", "chỉ",
        "một", "hai", "ba", "không", "phải", "đây", "kia", "ấy", "sự", "việc",
    }
)


def tokenize_query(query: str) -> List[str]:
    """Split a Vietnamese query into whitespace/punctuation-delimited tokens.

    Uses a ``\\w+`` regex (Unicode-aware) so accented Vietnamese terms survive.
    Returns an empty list for empty / whitespace-only input.
    """
    if not query:
        return []
    return _WORD_RE.findall(query)


def filter_query_terms(tokens: Sequence[str]) -> List[str]:
    """Drop high-frequency stopwords (and 1-char tokens) from an OR token list.

    Pure + unit-testable. These terms add enormous OR fanout to an FTS5 MATCH
    but almost no discriminative signal. Falls back to the original token list
    if filtering would empty it, so a query made entirely of stopwords still
    returns something rather than degrading to an empty MATCH.
    """
    kept = [t for t in tokens if len(t) > 1 and t.lower() not in _VI_STOPWORDS]
    return kept if kept else list(tokens)


def build_fts_match_expr(phrases: Sequence[str]) -> str:
    """Build an FTS5 ``MATCH`` expression from a list of phrase strings.

    Each phrase is double-quoted (so multi-word phrases match as a phrase)
    and joined with `` OR ``. An empty input yields an empty string (caller
    should short-circuit rather than run an empty MATCH).
    """
    parts: List[str] = []
    for p in phrases:
        p = (p or "").strip()
        if not p:
            continue
        # Escape any embedded double-quotes per FTS5 string quoting rules.
        p_escaped = p.replace('"', '""')
        parts.append(f'"{p_escaped}"')
    return " OR ".join(parts)


# ---------------------------------------------------------------------------
# FTSIndex — SQLite FTS5 wrapper for the Stage 6 bundle
# ---------------------------------------------------------------------------


class FTSIndex:
    """Context-managed wrapper around the Stage 6 ``chunk_store.sqlite`` FTS5.

    Parameters
    ----------
    db_path:
        Path to ``chunk_store.sqlite``.
    mode:
        ``"fts_fast"`` (default) — fast ``MATCH`` + ``LIMIT`` (no global sort).
        ``"bm25_ranked"`` — ranked by ``bm25(chunks_fts)`` (slower on wide
        queries; good for debug / small top-k).

    Examples
    --------
    >>> with FTSIndex("chunk_store.sqlite", mode="fts_fast") as fts:
    ...     hits = fts.search("đăng ký doanh nghiệp", top_k=50)

    The object is also usable without ``with`` via explicit
    :meth:`open` / :meth:`close`.
    """

    def __init__(self, db_path: str, mode: str = "fts_fast") -> None:
        if mode not in ("fts_fast", "bm25_ranked", "default"):
            raise ValueError(f"FTSIndex: unknown mode {mode!r}")
        self.db_path = db_path
        # Normalise legacy "default" → "fts_fast".
        self.mode = "fts_fast" if mode in ("fts_fast", "default") else "bm25_ranked"
        self._conn: Optional[sqlite3.Connection] = None
        self._n_rows: int = 0
        self._lexical_backend: str = ""

    # -- context manager -------------------------------------------------- #
    def __enter__(self) -> "FTSIndex":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- lifecycle -------------------------------------------------------- #
    def open(self) -> "FTSIndex":
        """Open the SQLite connection in read-only mode and read metadata."""
        if self._conn is not None:
            return self
        # uri=True + mode=ro makes the connection read-only.
        uri = f"file:{self.db_path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True)
        self._conn.row_factory = sqlite3.Row
        # Apply a couple of pragmatic read-only tuning options.
        try:
            self._conn.execute("PRAGMA query_only = ON")
            self._conn.execute("PRAGMA journal_mode = MEMORY")
        except sqlite3.DatabaseError:
            # Some environments reject these pragmas on read-only DBs; ignore.
            pass
        self._n_rows = int(
            self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        )
        try:
            am = dict(
                self._conn.execute(
                    "SELECT key, value FROM artifact_meta"
                ).fetchall()
            )
            self._lexical_backend = str(am.get("lexical_backend", ""))
        except sqlite3.DatabaseError:
            self._lexical_backend = ""
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- introspection ---------------------------------------------------- #
    @property
    def n_rows(self) -> int:
        """Number of rows in the ``chunks`` table."""
        if self._conn is None:
            raise RuntimeError("FTSIndex not opened; call .open() or use 'with'")
        return self._n_rows

    @property
    def lexical_backend(self) -> str:
        """The ``artifact_meta.lexical_backend`` value (e.g. ``"fts5"``)."""
        return self._lexical_backend

    # -- search ----------------------------------------------------------- #
    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """Lexical search over ``chunks_fts``; returns metadata dicts.

        Each hit dict contains ``row_idx``, ``chunk_id``, ``doc_uid``,
        ``law_id``, ``ten_van_ban``, ``dieu_so`` and (in ``bm25_ranked``
        mode) ``bm25_score``. The returned list is already truncated to
        ``top_k`` and preserves the FTS5 ranking order.
        """
        if self._conn is None:
            raise RuntimeError("FTSIndex not opened; call .open() or use 'with'")

        tokens = tokenize_query(query)
        if not tokens:
            return []

        # Build the MATCH expression: phrase = the whole query, plus each
        # discriminative token OR'd in, so both multi-word and single-term hits
        # surface. The whole-query phrase is quoted first (highest weight via
        # FTS5 phrase matching) followed by token alternatives. Stopwords are
        # stripped from the OR list: leaving them in made ``bm25_ranked`` match
        # a large fraction of the corpus and globally sort it (~319 s/query),
        # while the quoted full-query phrase still carries the precision signal.
        phrases = [query.strip()]
        phrases.extend(filter_query_terms(tokens))
        match_expr = build_fts_match_expr(phrases)
        if not match_expr:
            return []

        cols = ", ".join(_META_COLS)
        if self.mode == "bm25_ranked":
            sql = (
                "SELECT c.row_idx, c.chunk_id, c.doc_uid, c.law_id, "
                "c.ten_van_ban, c.dieu_so, bm25(chunks_fts) AS bm25_score "
                "FROM chunks_fts "
                "JOIN chunks c ON c.row_idx = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? "
                "ORDER BY bm25_score "
                "LIMIT ?"
            )
            rows = self._conn.execute(sql, (match_expr, int(top_k))).fetchall()
            out = []
            for r in rows:
                d = {c: r[c] for c in _META_COLS}
                d["bm25_score"] = float(r["bm25_score"])
                out.append(d)
            return out

        # fts_fast: no global sort; rely on FTS5 rowid / relevance heuristics.
        sql = (
            f"SELECT {cols} "
            "FROM chunks_fts "
            "JOIN chunks c ON c.row_idx = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? "
            "LIMIT ?"
        )
        rows = self._conn.execute(sql, (match_expr, int(top_k))).fetchall()
        return [{c: r[c] for c in _META_COLS} for r in rows]

    # -- full-text fetch -------------------------------------------------- #
    def fetch_chunks(self, row_idxs: Sequence[int]) -> List[Dict[str, Any]]:
        """Fetch full ``chunk_text`` + metadata for the given ``row_idx`` list.

        Output order matches the input ``row_idx`` order (rows not found are
        omitted). Each dict includes all ``chunks`` columns.
        """
        if self._conn is None:
            raise RuntimeError("FTSIndex not opened; call .open() or use 'with'")
        idxs = [int(x) for x in row_idxs]
        if not idxs:
            return []
        placeholders = ",".join(["?"] * len(idxs))
        sql = (
            "SELECT row_idx, chunk_id, doc_uid, law_id, ten_van_ban, dieu_so, "
            "chunk_text FROM chunks WHERE row_idx IN "
            f"({placeholders})"
        )
        rows = self._conn.execute(sql, idxs).fetchall()
        by_idx = {int(r["row_idx"]): dict(r) for r in rows}
        return [by_idx[i] for i in idxs if i in by_idx]


# ---------------------------------------------------------------------------
# PureBM25 — in-memory Okapi BM25 (fallback / tests)
# ---------------------------------------------------------------------------


class PureBM25:
    """Minimal pure-Python Okapi BM25 over an in-memory string corpus.

    Tokenisation uses the same ``\\w+`` regex as :func:`tokenize_query`.
    Designed for small corpora (unit tests, smoke checks). Not for the
    636k-chunk production index — use :class:`FTSIndex` there.

    Parameters
    ----------
    corpus:
        List of document strings, indexed by ``row_idx``.
    k1, b:
        BM25 tuning constants. ``k1`` must be ``> 0`` and ``b`` in ``[0, 1]``.
    """

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError(f"PureBM25: k1 must be > 0, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"PureBM25: b must be in [0, 1], got {b}")
        self.k1 = k1
        self.b = b
        self._docs: List[List[str]] = []
        self._tf: List[Dict[str, int]] = []
        self._dl: List[int] = []
        self._df: Dict[str, int] = {}
        for doc in corpus:
            toks = tokenize_query(doc or "")
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self._docs.append(toks)
            self._tf.append(tf)
            self._dl.append(len(toks))
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1
        self._n = len(corpus)
        self._avgdl = (sum(self._dl) / self._n) if self._n else 0.0
        # Precompute idf (with the +1 smoothing to keep it non-negative).
        self._idf = {
            t: math.log(1 + (self._n - df + 0.5) / (df + 0.5))
            for t, df in self._df.items()
        }

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return BM25-ranked hits as ``{row_idx, bm25_score}`` dicts."""
        q_terms = tokenize_query(query)
        if not q_terms or self._n == 0:
            return []
        scores: List[float] = [0.0] * self._n
        for i in range(self._n):
            tf = self._tf[i]
            dl = self._dl[i]
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / self._avgdl if self._avgdl else 1.0))
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                f = tf[term]
                s += idf * (f * (self.k1 + 1)) / (f + denom_norm)
            scores[i] = s
        # Rank by score desc, row_idx asc; stop once scores go non-positive.
        ranked = sorted(
            ((i, sc) for i, sc in enumerate(scores) if sc > 0),
            key=lambda x: (-x[1], x[0]),
        )
        return [
            {"row_idx": i, "bm25_score": float(sc)}
            for i, sc in ranked[:top_k]
        ]
