# Preprocessing To-Do List for Vietnamese Legal Documents

## 1. Set up the workspace
- [ ] Create a clean project folder structure for raw, interim, processed, and artifacts.
- [ ] Install the needed libraries: `pandas`, `pyarrow`, `beautifulsoup4`, `lxml`, `tqdm`, `regex`, `networkx`.
- [ ] Decide where to run the job: CPU for parsing/filtering, GPU later for summarization if needed.
- [ ] Fix a random seed for reproducibility.

## 2. Load the dataset
- [ ] Download or mount the Hugging Face dataset `th1nhng0/vietnamese-legal-documents`.
- [ ] Inspect the available tables: `metadata`, `content`, and `relationships`.
- [ ] Check row counts, column names, and null rates.
- [ ] Verify which rows have HTML content and which are PDF-only or missing content.

## 3. Clean and filter metadata
- [ ] Keep only documents relevant to the SME legal scope.
- [ ] Filter by document validity status, keeping active documents first.
- [ ] Filter by acceptable document types, such as laws, decrees, circulars, and decisions.
- [ ] Normalize important metadata fields: title, issue date, effective date, document type, and sector labels.
- [ ] Create a stable `lawid` or document key for joins across tables.

## 4. Join metadata and content
- [ ] Merge metadata with HTML content on document ID.
- [ ] Drop documents without usable HTML content.
- [ ] Log removed rows so you know how many were excluded and why.
- [ ] Save the joined dataset as a first preprocessing checkpoint.

## 5. Parse the HTML
- [ ] Strip HTML with BeautifulSoup.
- [ ] Remove duplicate text blocks caused by nested tags.
- [ ] Normalize whitespace, punctuation, and line breaks.
- [ ] Preserve document hierarchy markers such as `Phần`, `Chương`, `Mục`, and `Điều`.

## 6. Extract legal articles
- [ ] Split each document into article-level records.
- [ ] Build one row per `Điều` as the main retrieval unit.
- [ ] Extract article number, article title, article body, and parent document info.
- [ ] Drop suspicious extractions, such as articles with very short text.
- [ ] Verify that each article has a unique document ID plus article number combination.

## 7. Chunk long articles
- [ ] Measure token length of every article.
- [ ] Keep short articles as single chunks.
- [ ] Split long articles by clause boundaries (`Khoản`) when they exceed the limit.
- [ ] Use overlap between chunks so context is not lost.
- [ ] Add breadcrumb context to every chunk: document type, document title, part, chapter, section, article.

## 8. Build enriched text
- [ ] Create a cleaned chunk text field that combines breadcrumb, summary, and body.
- [ ] Optionally generate short summaries and key-point summaries for each chunk.
- [ ] Cache summaries in a resumable file such as JSONL.
- [ ] Make sure enriched text is consistent across all chunks.

## 9. Process relationships
- [ ] Clean the `relationships` table.
- [ ] Keep only relation types you plan to use, such as amends, cites, replaces, and details.
- [ ] Normalize directionality so edge labels are consistent.
- [ ] Filter graph edges to documents that survive your SME preprocessing scope.
- [ ] Save the cleaned edge list for graph construction.

## 10. Build the knowledge graph
- [ ] Create document nodes and article nodes.
- [ ] Add `HAS_ARTICLE` edges from documents to extracted articles.
- [ ] Add legal relation edges from the relationships table.
- [ ] Add concept mention edges only if you are using concept extraction.
- [ ] Export the graph in a reusable format such as `gpickle`.

## 11. Create retrieval indexes
- [ ] Build a BM25 index on enriched chunk text.
- [ ] Build a dense embedding index for summaries and full chunks.
- [ ] Store sidecar metadata for each chunk ID.
- [ ] Confirm row-to-index alignment before saving.
- [ ] Keep separate indexes for summary recall and full-text precision if needed.

## 12. Validate quality
- [ ] Check article extraction coverage per document.
- [ ] Sample documents manually to confirm parsing accuracy.
- [ ] Look for malformed titles, empty bodies, or duplicated articles.
- [ ] Measure how many docs were lost in each stage.
- [ ] Save validation reports with counts and error examples.

## 13. Prepare final artifacts
- [ ] Export each stage as parquet or jsonl.
- [ ] Keep file names versioned and deterministic.
- [ ] Store a preprocessing log with timestamps and record counts.
- [ ] Make the pipeline restartable from each stage.
- [ ] Document the schema of every saved artifact.

## 14. Final check
- [ ] Re-run the pipeline on a small sample end to end.
- [ ] Confirm outputs are ready for retrieval and citation.
- [ ] Confirm no stage depends on manual cleanup.
- [ ] Freeze the preprocessing spec before indexing and modeling.