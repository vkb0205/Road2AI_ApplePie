"""PostgreSQL lexical (tsvector / websearch_to_tsquery) retrieval backend.

This module provides :class:`PGFTSIndex`, a drop-in replacement for the SQLite
:class:`retrieval.bm25_index.FTSIndex` that reads from the Supabase PostgreSQL
database populated by ``upload_data_to_db.py`` instead of the local Stage 6
``chunk_store.sqlite`` bundle.

Why this exists
---------------
The Stage 6 lexical contract is *chunk*-level and keyed on a stable integer
``row_idx`` (see ``data/stage6_data/artifacts_guide.md``). The upload path,
however, lands data in PostgreSQL where ``sql/00_schema.sql`` only defined
*article*-level FTS (``articles.noi_dung_tsv``) and the ``chunks`` table had no
``row_idx``. Migration ``sql/02_chunk_fts.sql`` adds:

  * a chunk-level ``chunk_text_tsv`` tsvector + GIN index, and
  * a stable, dense, non-null ``chunks.row_idx``.

``PGFTSIndex`` exposes the same public surface as ``FTSIndex`` (``open``,
``search``, ``fetch_chunks``, ``n_rows``, context-manager), so
:class:`retrieval.retriever.HybridRetriever` can be constructed with a
``PGFTSIndex`` in place of an ``FTSIndex`` with no other changes.

Query semantics
---------------
``fts_fast`` (default) ranks by PostgreSQL's built-in ts_rank_cd (cover density)
— fast and stable, matching the SQLite bundle's "rank but defer precision to
the reranker" philosophy. ``bm25_ranked`` is not supported at the SQL level in
PG (there is no native BM25 function); instead ``bm25_ranked`` here is an alias
that returns the same ts_rank_cd ranking but *exposes* the rank score under the
``bm25_score`` key for debugging/diagnostics parity with ``FTSIndex``.

The query string is built with ``build_websearch_tsquery`` which joins the
query tokens with `` & `` (AND) for precision; recall-friendly OR semantics are
the job of the dense leg + RRF + reranker, exactly as in the SQLite path.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from .bm25_index import (
    DEFAULT_MODE,
    DEFAULT_TOP_K,
    _simple_tokenize,
)

logger = logging.getLogger(__name__)

# Lazily imported so the rest of the retrieval package works without psycopg2
# (e.g. on a CPU laptop running only the SQLite bundle path).
try:
    import psycopg2  # type: ignore
    from psycopg2.extras import RealDictCursor  # type: ignore
    _HAVE_PSYCOPG2 = True
except Exception:  # pragma: no cover - optional dep
    psycopg2 = None  # type: ignore
    RealDictCursor = None  # type: ignore
    _HAVE_PSYCOPG2 = False


# --------------------------------------------------------------------------- #
# Lexical index Protocol — shared contract for FTSIndex and PGFTSIndex
# --------------------------------------------------------------------------- #


@runtime_checkable
class LexicalIndex(Protocol):
    """Structural contract every lexical backend implements.

    Defined so :class:`retrieval.retriever.HybridRetriever` can accept any
    backend (SQLite ``FTSIndex`` or PostgreSQL ``PGFTSIndex``) without a hard
    import dependency on either.
    """

    n_rows: int

    def open(self) -> "LexicalIndex": ...
    def close(self) -> None: ...
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]: ...
    def fetch_chunks(self, row_indices: Sequence[int]) -> List[Dict[str, Any]]: ...


# --------------------------------------------------------------------------- #
# Query helpers
# --------------------------------------------------------------------------- #


def build_websearch_tsquery(query: str) -> str:
    """Build a ``websearch_to_tsquery``-style AND query from a raw string.

    Tokenises with the same lightweight splitter used by the SQLite path so the
    two backends see the same terms. Tokens are joined with `` & `` (AND) for
    precision; recall is recovered by the dense leg + RRF + reranker. Each
    token is quoted with double quotes so PostgreSQL treats it as a literal
    lexeme and never mis-parses operators (``AND``, ``OR``, ``-``).
    """
    tokens = _simple_tokenize(query)
    if not tokens:
        return ""
    # websearch_to_tsquery supports quoted phrases and boolean operators; we
    # emit quoted lexemes joined by ' & ' (AND) for a precise lexical match.
    quoted = [f'"{t}"' for t in tokens if t]
    return " & ".join(quoted)


# --------------------------------------------------------------------------- #
# PGFTSIndex
# --------------------------------------------------------------------------- #


class PGFTSIndex:
    """Lexical retrieval over the PostgreSQL ``chunks`` table (tsvector backend).

    A drop-in replacement for :class:`retrieval.bm25_index.FTSIndex` that reads
    from the Supabase/PostgreSQL database populated by ``upload_data_to_db.py``.

    Requires migration ``sql/02_chunk_fts.sql`` (chunk-level ``chunk_text_tsv``
    + dense ``row_idx``). Call :meth:`ensure_migration` once after uploading to
    populate ``row_idx`` and build the GIN index.

    Parameters
    ----------
    db_url:
        PostgreSQL connection string. If ``None``, read from the
        ``DATABASE_URL`` environment variable.
    mode:
        ``"fts_fast"`` (default) — ranked by ``ts_rank_cd``; the raw rank is
        reported as ``0.0`` in hits (ranking is delegated to RRF + reranker),
        matching ``FTSIndex`` semantics.
        ``"bm25_ranked"`` — same ``ts_rank_cd`` ranking but the rank score is
        exposed under ``bm25_score`` for diagnostics. (PostgreSQL has no native
        BM25; this mirrors the ``FTSIndex`` API surface for parity.)
    table:
        Base table name (default ``"chunks"``). Override only for tests.
    schema:
        Schema name (default ``"public"``).
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        mode: str = DEFAULT_MODE,
        table: str = "chunks",
        schema: str = "public",
    ) -> None:
        if not _HAVE_PSYCOPG2:
            raise ImportError(
                "PGFTSIndex requires psycopg2. Install with: "
                "pip install psycopg2-binary"
            )
        if mode not in {"fts_fast", "bm25_ranked"}:
            raise ValueError(
                f"PGFTSIndex: mode must be 'fts_fast' or 'bm25_ranked', got {mode!r}"
            )
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        if not self.db_url:
            raise ValueError(
                "PGFTSIndex: no db_url and DATABASE_URL env var is unset; "
                "pass db_url= or export DATABASE_URL"
            )
        self.mode = mode
        self.table = table
        self.schema = schema
        self._conn: Optional[Any] = None
        self._n_rows: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Qualified table helpers (schema-agnostic)
    # ------------------------------------------------------------------ #
    @property
    def _qtable(self) -> str:
        return f'"{self.schema}"."{self.table}"'

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def open(self) -> "PGFTSIndex":
        """Open the PostgreSQL connection (read-only session) and probe row count."""
        self._conn = psycopg2.connect(self.db_url)
        # Force read-only so a retrieval session can never mutate the corpus.
        with self._conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on;")
        self._conn.row_factory = None  # we use RealDictCursor per-query
        self._n_rows = self._count_rows()
        logger.info(
            "PGFTSIndex opened (table=%s, rows=%s, mode=%s)",
            self._qtable,
            f"{self._n_rows:,}",
            self.mode,
        )
        return self

    def _count_rows(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._qtable}")
            return int(cur.fetchone()[0])

    @property
    def conn(self) -> Any:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    @property
    def n_rows(self) -> int:
        if self._n_rows is None:
            self.open()
        assert self._n_rows is not None
        return self._n_rows

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PGFTSIndex":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Migration helper
    # ------------------------------------------------------------------ #
    def ensure_migration(self) -> "PGFTSIndex":
        """Idempotently apply ``sql/02_chunk_fts.sql`` (row_idx + chunk FTS).

        Convenience wrapper so callers don't have to drop to the shell. Must be
        run with a *read-write* connection (it temporarily flips
        ``default_transaction_read_only`` off for the migration statements).
        Safe to call before :meth:`open` or on an already-open index.
        """
        from pathlib import Path

        was_open = self._conn is not None
        if not was_open:
            self._conn = psycopg2.connect(self.db_url)

        sql_path = Path(__file__).resolve().parents[2] / "sql" / "02_chunk_fts.sql"
        if not sql_path.exists():
            raise FileNotFoundError(
                f"PGFTSIndex.ensure_migration: migration not found at {sql_path}"
            )
        sql = sql_path.read_text(encoding="utf-8")
        autocommit_was = self._conn.autocommit
        try:
            self._conn.autocommit = True
            with self._conn.cursor() as cur:
                cur.execute("SET default_transaction_read_only = off;")
                cur.execute(sql)
            logger.info("PGFTSIndex: migration 02 applied (row_idx + chunk FTS)")
        finally:
            self._conn.autocommit = autocommit_was
            if not was_open:
                self._conn.close()
                self._conn = None
            else:
                # refresh row count after migration
                self._n_rows = self._count_rows()
        return self

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        """Run a chunk-level lexical search and return up to ``top_k`` hit dicts.

        Each hit dict has the same keys as :meth:`FTSIndex.search`::

            {
              "row_idx": int,
              "bm25_score": float,        # ts_rank_cd; 0.0 in fts_fast mode
              "bm25_rank": int | None,    # 1-based in bm25_ranked, None in fts_fast
              "chunk_id": str,
              "doc_uid": str,
              "law_id": str,
              "ten_van_ban": str,
              "dieu_so": str,
            }

        ``law_id`` / ``ten_van_ban`` / ``dieu_so`` are joined from ``articles``
        on ``doc_uid`` because the uploaded ``chunks`` table (per
        ``sql/00_schema.sql``) does not denormalise them, unlike the Stage 6
        SQLite bundle. The join is cheap on ``doc_uid`` (indexed).
        """
        tsq = build_websearch_tsquery(query)
        if not tsq:
            logger.debug("PGFTSIndex.search: empty query for %r", query)
            return []

        # ts_rank_cd on the generated chunk_text_tsv column. We join articles
        # for law_id/ten_van_ban/dieu_so (not stored on chunks in the PG schema).
        expose_score = self.mode == "bm25_ranked"
        rank_col = "ts_rank_cd(c.chunk_text_tsv, q) AS rank" if expose_score else "0.0 AS rank"
        sql = f"""
            WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS tsq)
            SELECT c.row_idx,
                   c.chunk_id,
                   c.doc_uid,
                   COALESCE(a.law_id, '')      AS law_id,
                   COALESCE(a.ten_van_ban, '') AS ten_van_ban,
                   COALESCE(a.dieu_so, '')     AS dieu_so,
                   {rank_col}
            FROM {self._qtable} c
            CROSS JOIN q
            LEFT JOIN "{self.schema}"."articles" a
                   ON a.doc_uid = c.doc_uid
            WHERE c.chunk_text_tsv @@ q.tsq
              AND q.tsq <> to_tsvector('simple', '')
            ORDER BY rank DESC, c.row_idx ASC
            LIMIT %s
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (tsq, int(top_k)))
            rows = cur.fetchall()

        if expose_score:
            return [
                {
                    "row_idx": int(r["row_idx"]),
                    "bm25_score": float(r["rank"]),
                    "bm25_rank": rank,
                    "chunk_id": r["chunk_id"],
                    "doc_uid": r["doc_uid"],
                    "law_id": r["law_id"],
                    "ten_van_ban": r["ten_van_ban"],
                    "dieu_so": r["dieu_so"],
                }
                for rank, r in enumerate(rows, start=1)
            ]
        # fts_fast — hide the raw score (0.0), no explicit rank
        return [
            {
                "row_idx": int(r["row_idx"]),
                "bm25_score": 0.0,
                "bm25_rank": None,
                "chunk_id": r["chunk_id"],
                "doc_uid": r["doc_uid"],
                "law_id": r["law_id"],
                "ten_van_ban": r["ten_van_ban"],
                "dieu_so": r["dieu_so"],
            }
            for r in rows
        ]

    def fetch_chunks(self, row_indices: Sequence[int]) -> List[Dict[str, Any]]:
        """Fetch full chunk rows (including ``chunk_text``) for the given row_ids.

        Order follows the input ``row_indices``. Missing row_ids are dropped.
        Mirrors :meth:`FTSIndex.fetch_chunks`.
        """
        if not row_indices:
            return []
        idxs = [int(x) for x in row_indices]
        # bound the IN-list to avoid sending empty/huge placeholders
        placeholders = ",".join(["%s"] * len(idxs))
        sql = f"""
            SELECT c.row_idx, c.chunk_id, c.doc_uid,
                   COALESCE(a.law_id, '')      AS law_id,
                   COALESCE(a.ten_van_ban, '') AS ten_van_ban,
                   COALESCE(a.dieu_so, '')     AS dieu_so,
                   c.chunk_text
            FROM {self._qtable} c
            LEFT JOIN "{self.schema}"."articles" a
                   ON a.doc_uid = c.doc_uid
            WHERE c.row_idx IN ({placeholders})
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, idxs)
            rows = cur.fetchall()
        by_idx = {int(r["row_idx"]): dict(r) for r in rows}
        return [by_idx[i] for i in idxs if i in by_idx]
