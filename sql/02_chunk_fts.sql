-- ============================================================================
-- Migration 02: Chunk-level full-text search for the PostgreSQL retrieval path
--
-- Why this exists
-- ---------------
-- ``sql/00_schema.sql`` only defines article-level FTS (``articles.noi_dung_tsv``)
-- and the upload script (``upload_data_to_db.py``) writes to Supabase PG. The
-- Stage 6 lexical retrieval contract, however, is *chunk*-level: SQLite's
-- ``chunks_fts`` indexes ``chunks.chunk_text`` and returns a stable integer
-- ``row_idx`` that is also the FAISS vector position (see
-- ``data/stage6_data/artifacts_guide.md``).
--
-- ``src/retrieval/pg_index.py`` (``PGFTSIndex``) is a drop-in replacement for
-- the SQLite ``FTSIndex`` that reads from PG. For that it needs:
--   1. a chunk-level tsvector column + GIN index, and
--   2. a stable, dense, non-null integer ``row_idx`` on ``chunks`` so the rest
--      of the pipeline (RRF, graph expansion, ``fetch_chunks``) can key off it
--      exactly like the SQLite bundle.
--
-- This migration is idempotent (safe to re-run). It does NOT touch the existing
-- article-level FTS index.
--
-- Tokeniser: 'simple' config — splits on non-letter runs, lowercases. This is
-- the same fallback tokeniser the SQLite bundle uses and matches the existing
-- ``articles.noi_dung_tsv`` generated column.
-- ============================================================================

-- --------------------------------------------------------------------------
-- 1. Stable dense row_idx on chunks
--    Assigned once, deterministically, ordered by (doc_uid, chunk_id) so the
--    mapping is reproducible across re-runs. Unique + non-null so it can act
--    as a secondary key for the retrieval pipeline.
-- --------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'row_idx'
    ) THEN
        ALTER TABLE chunks ADD COLUMN row_idx BIGINT;
    END IF;
END $$;

-- Populate any NULL row_idx with a deterministic dense rank.
-- Re-running is a no-op once every row is non-null.
WITH ranked AS (
    SELECT chunk_id,
           ROW_NUMBER() OVER (ORDER BY doc_uid, chunk_id)::BIGINT AS rn
    FROM chunks
)
UPDATE chunks c
SET row_idx = r.rn
FROM ranked r
WHERE c.chunk_id = r.chunk_id
  AND c.row_idx IS NULL;

-- Guard: every chunk must now have a row_idx.
-- (Left as a runtime assertion in PGFTSIndex.open() rather than a hard CHECK,
--  so a partially-loaded DB still opens for diagnostics.)
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_row_idx
    ON chunks(row_idx)
    WHERE row_idx IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chunks_doc_uid_row_idx
    ON chunks(doc_uid, row_idx);

COMMENT ON COLUMN chunks.row_idx IS
    'Stable dense integer id (1-based) assigned by migration 02; the retrieval pipeline (RRF, graph, fetch_chunks) keys off this like the SQLite bundle row_idx.';

-- --------------------------------------------------------------------------
-- 2. Chunk-level tsvector + GIN index
--    GENERATED ALWAYS ... STORED keeps it in sync with chunk_text automatically.
-- --------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'chunk_text_tsv'
    ) THEN
        ALTER TABLE chunks
            ADD COLUMN chunk_text_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('simple', chunk_text)) STORED;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON chunks USING GIN(chunk_text_tsv);

COMMENT ON COLUMN chunks.chunk_text_tsv IS
    'Chunk-level full-text search vector (simple config); consumed by PGFTSIndex as the chunk-level lexical backend.';

-- --------------------------------------------------------------------------
-- 3. Convenience: coverage diagnostic view (optional, no data duplication)
--    Helps validate that row_idx is dense and gap-free after migration.
-- --------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_chunk_fts_coverage AS
SELECT
    COUNT(*)                                   AS total_chunks,
    COUNT(row_idx)                             AS chunks_with_row_idx,
    COUNT(*) - COUNT(row_idx)                  AS chunks_missing_row_idx,
    MIN(row_idx)                               AS min_row_idx,
    MAX(row_idx)                               AS max_row_idx,
    COUNT(chunk_text_tsv)                      AS chunks_with_tsv
FROM chunks;

COMMENT ON VIEW v_chunk_fts_coverage IS
    'Migration 02 diagnostic: row_idx population and FTS coverage for chunks.';
