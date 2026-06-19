-- ============================================================================
-- SQL Migration 01: Normalised chunk → concept MENTIONS table
-- Description: Stores individual chunk-to-concept mention relationships so
--              they are never lost after Stage 5.6 pipeline execution.
-- ============================================================================

-- --------------------------------------------------------------------------
-- 1. CHUNK → CONCEPT MENTIONS
--    Written by Stage 5.6 after each chunk is processed (substring or LLM).
--    One row per (chunk, concept) pair; duplicates are silently ignored.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunk_concept_mentions (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chunk_id        TEXT NOT NULL REFERENCES chunks(chunk_id),
    doc_uid         TEXT NOT NULL,
    doc_id          BIGINT NOT NULL REFERENCES documents(id),
    concept_name    TEXT NOT NULL,
    mentions_source TEXT NOT NULL,          -- 'substring' or 'llm'
    created_at      TIMESTAMP DEFAULT now(),
    UNIQUE(chunk_id, concept_name)
);

CREATE INDEX IF NOT EXISTS idx_ccm_chunk_id      ON chunk_concept_mentions(chunk_id);
CREATE INDEX IF NOT EXISTS idx_ccm_concept_name  ON chunk_concept_mentions(concept_name);
CREATE INDEX IF NOT EXISTS idx_ccm_doc_id        ON chunk_concept_mentions(doc_id);
CREATE INDEX IF NOT EXISTS idx_ccm_mentions_source ON chunk_concept_mentions(mentions_source);

COMMENT ON TABLE  chunk_concept_mentions IS 'Stage 5.6: Normalised chunk→concept MENTIONS relationships';
COMMENT ON COLUMN chunk_concept_mentions.chunk_id IS 'FK to chunks(chunk_id) — the source chunk';
COMMENT ON COLUMN chunk_concept_mentions.concept_name IS 'Matched concept display name';
COMMENT ON COLUMN chunk_concept_mentions.mentions_source IS 'Extraction method: substring (deterministic) or llm (model-based)';
