-- ============================================================================
-- SQL Schema: G-LRAG Vietnamese Legal Knowledge Graph
-- Description: Stores docs, articles, chunks, concepts, and tracks
--              pipeline processing state (Stage 5.6 concept extraction).
-- Pipeline: Stage 1 → 2 → 3 → 5 (5.2–5.8)
-- ============================================================================

-- --------------------------------------------------------------------------
-- 1. DOCUMENTS (Stage 1 — SME-filtered legal documents)
--    Source: stage1_sme_docs.parquet
-- --------------------------------------------------------------------------
CREATE TABLE documents (
    id              BIGINT PRIMARY KEY,          -- Unique doc ID from dataset
    law_id          TEXT NOT NULL,                -- e.g. "11/2023/TT-BLĐTBXH"
    ten_van_ban     TEXT NOT NULL,                -- Full Vietnamese title
    loai_van_ban    TEXT,                         -- Document type (Thông tư, Nghị định, ...)
    nganh           TEXT,                         -- Sector (Lao động - Thương binh và Xã hội)
    linh_vuc        TEXT,                         -- Field (tổ chức cán bộ)
    ngay_ban_hanh   DATE,                         -- Issue date
    tinh_trang_hieu_luc TEXT,                     -- Validity status
    ngay_co_hieu_luc    DATE,                     -- Effective date
    ngay_het_hieu_luc   DATE,                     -- Expiry date
    created_at      TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_documents_law_id ON documents(law_id);
CREATE INDEX idx_documents_nganh ON documents(nganh);
CREATE INDEX idx_documents_linh_vuc ON documents(linh_vuc);
CREATE INDEX idx_documents_loai ON documents(loai_van_ban);

COMMENT ON TABLE documents IS 'Stage 1: SME-filtered legal documents from metadata';
COMMENT ON COLUMN documents.id IS 'Unique document identifier (doc_id in downstream stages)';
COMMENT ON COLUMN documents.law_id IS 'Official law number/code, e.g. 11/2023/TT-BLĐTBXH';
COMMENT ON COLUMN documents.ten_van_ban IS 'Full Vietnamese document title, e.g. "Thông tư 11/2023/TT-BLĐTBXH Hướng dẫn..."';
COMMENT ON COLUMN documents.loai_van_ban IS 'Document classification (Luật, Nghị định, Thông tư, ...)';

-- --------------------------------------------------------------------------
-- 2. ARTICLES (Stage 2 — Parsed articles from HTML documents)
--    Source: stage2_articles.parquet
--    Each row = one article (dieu) within a legal document.
--    FK → documents(doc_id).
-- --------------------------------------------------------------------------
CREATE TABLE articles (
    doc_uid         TEXT PRIMARY KEY,             -- Unique article ID: "{law_id}|{title}|{dieu_so}"
    doc_id          BIGINT NOT NULL REFERENCES documents(id),
    law_id          TEXT NOT NULL,
    ten_van_ban     TEXT NOT NULL,
    loai_van_ban    TEXT,
    ngay_ban_hanh   DATE,
    nganh           TEXT,
    linh_vuc        TEXT,
    phan            TEXT,                         -- Part (Phần)
    chuong          TEXT,                         -- Chapter (Chương)
    muc             TEXT,                         -- Section (Mục)
    dieu_so         TEXT NOT NULL,                -- Article number, e.g. "Điều 9"
    dieu_ten        TEXT,                         -- Article title, e.g. "Hiệu lực và trách nhiệm thi hành"
    noi_dung        TEXT NOT NULL,                -- Full HTML-cleaned article body
    start_char      INTEGER,                      -- Start offset in original doc text
    end_char        INTEGER,                      -- End offset in original doc text
    created_at      TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_articles_doc_id ON articles(doc_id);
CREATE INDEX idx_articles_law_id ON articles(law_id);
CREATE INDEX idx_articles_dieu_so ON articles(dieu_so);

COMMENT ON TABLE articles IS 'Stage 2: Parsed articles (điều) from legal documents';
COMMENT ON COLUMN articles.doc_uid IS 'Globally unique article ID: concatenation of law_id|title|article_number';
COMMENT ON COLUMN articles.dieu_so IS 'Article designation, e.g. "Điều 9" or "Điều 9a"';
COMMENT ON COLUMN articles.dieu_ten IS 'Article heading, e.g. "Hiệu lực và trách nhiệm thi hành"';
COMMENT ON COLUMN articles.noi_dung IS 'Full article content after HTML-to-text conversion';

-- --------------------------------------------------------------------------
-- 3. CHUNKS (Stage 3 — Token-bounded overlapping chunks)
--    Source: stage3_chunks.parquet
--    Each row = one chunk, FK → articles(doc_uid).
--    Overlap: chunk N re-appends the last piece of chunk N-1.
-- --------------------------------------------------------------------------
CREATE TABLE chunks (
    chunk_id        TEXT PRIMARY KEY,             -- e.g. "{doc_uid}#{part_idx}"
    doc_uid         TEXT NOT NULL REFERENCES articles(doc_uid),
    doc_id          BIGINT NOT NULL REFERENCES documents(id),
    part_idx        INTEGER NOT NULL,             -- 0-based position within article
    breadcrumb      TEXT,                         -- Hierarchical nav: "Type > Law > Part > Chapter > Article"
    chunk_text      TEXT NOT NULL,                -- Actual chunk content (with breadcrumb prepended)
    created_at      TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_chunks_doc_uid ON chunks(doc_uid);
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX idx_chunks_part_idx ON chunks(doc_uid, part_idx);

COMMENT ON TABLE chunks IS 'Stage 3: Overlapping text chunks from articles — the atomic retrieval unit';
COMMENT ON COLUMN chunks.breadcrumb IS 'Hierarchical breadcrumb for provenance, e.g. "Thông tư > Law Title > Điều 9"';

-- --------------------------------------------------------------------------
-- 4. CONCEPTS (Stage 5.6 — Curated legal concepts for semantic linking)
--    Source: config/legal_concepts.yaml
--    Fixed vocabulary of 50-100 SME-relevant legal terms.
-- --------------------------------------------------------------------------
CREATE TABLE concepts (
    name            TEXT PRIMARY KEY,             -- Canonical concept name, e.g. "vốn điều lệ"
    name_lower      TEXT NOT NULL UNIQUE,         -- Lowercase for case-insensitive matching
    created_at      TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_concepts_name_lower ON concepts(name_lower);

COMMENT ON TABLE concepts IS 'Stage 5.6: Curated legal concept vocabulary for CHUNK concept extraction';
COMMENT ON COLUMN concepts.name IS 'Display name in original Vietnamese casing';

-- --------------------------------------------------------------------------
-- 5. CHUNK PROCESSING TRACKER (Stage 5.6)
--    Records which chunks have been processed for concept extraction,
--    what method was used, and which concepts were found.
--    Used by the Python pipeline to track progress and resume safely.
-- --------------------------------------------------------------------------
CREATE TABLE chunk_processing (
    chunk_id        TEXT PRIMARY KEY,             -- FK to chunks(chunk_id)
    doc_uid         TEXT NOT NULL,                -- Denormalised for easy querying
    doc_id          BIGINT NOT NULL,              -- Denormalised for easy querying
    processed_at    TIMESTAMP DEFAULT now(),      -- When Stage 5.6 processed this chunk
    mentions_source TEXT,                         -- 'substring' or 'llm'
    concepts_found  TEXT[],                       -- Array of matched concept names (may be empty)
    concepts_count  INTEGER DEFAULT 0             -- Length of concepts_found for fast filtering
);

CREATE INDEX idx_chunk_processing_doc_uid ON chunk_processing(doc_uid);
CREATE INDEX idx_chunk_processing_doc_id ON chunk_processing(doc_id);
CREATE INDEX idx_chunk_processing_mentions_source ON chunk_processing(mentions_source);
CREATE INDEX idx_chunk_processing_processed_at ON chunk_processing(processed_at);

COMMENT ON TABLE chunk_processing IS 'Stage 5.6 pipeline tracker: records which chunks have had concept extraction applied';
COMMENT ON COLUMN chunk_processing.chunk_id IS 'FK to chunks(chunk_id) — the processed chunk';
COMMENT ON COLUMN chunk_processing.mentions_source IS 'Extraction method: substring (deterministic) or llm (model-based)';
COMMENT ON COLUMN chunk_processing.concepts_found IS 'Array of concept names matched in this chunk; empty array means none found';

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Full article with breadcrumb from its document
CREATE VIEW v_article_details AS
SELECT
    a.doc_uid,
    a.dieu_so,
    a.dieu_ten,
    a.noi_dung,
    a.phan,
    a.chuong,
    a.muc,
    d.id                AS doc_id,
    d.law_id,
    d.ten_van_ban,
    d.loai_van_ban,
    d.nganh,
    d.linh_vuc,
    d.ngay_ban_hanh,
    d.tinh_trang_hieu_luc
FROM articles a
JOIN documents d ON d.id = a.doc_id;

COMMENT ON VIEW v_article_details IS 'Articles denormalised with document-level metadata';

-- ============================================================================
-- INDEX: Full-text search on article content
-- ============================================================================
ALTER TABLE articles ADD COLUMN noi_dung_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', noi_dung)) STORED;

CREATE INDEX idx_articles_fts ON articles USING GIN(noi_dung_tsv);

COMMENT ON COLUMN articles.noi_dung_tsv IS 'Full-text search vector for article content (simple config)';

-- ============================================================================
-- METADATA
-- ============================================================================
CREATE TABLE pipeline_metadata (
    stage       TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    row_count   BIGINT,
    run_at      TIMESTAMP DEFAULT now()
);

COMMENT ON TABLE pipeline_metadata IS 'Tracks when each pipeline stage was loaded into the DB';
