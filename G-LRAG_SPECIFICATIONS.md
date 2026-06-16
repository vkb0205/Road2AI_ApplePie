# Vietnamese Legal AI Assistant for SMEs — Technical Specification

| Field | Value |
|---|---|
| System name | G-LRAG (Graph-enhanced Legal Retrieval Augmented Generation) |
| Document version | 3.1 |
| Last updated | 2026-06-16 |
| Status | Approved for implementation |

> Changes in v3.1: §8 knowledge graph aligned with `KG.md` (canonical-direction-only relations; reverse-only labels and `RELATED_LANGUAGE` dropped; CHUNK node added to §8.2; build procedure split per Stage 5.x; edge-attribute key standardized to `relation`). §10.4 graph expansion now reads canonical edges in both directions instead of separate reverse edges. §11.5 graph-context construction synthesizes reverse labels in code.

---

## Table of Contents

1. [Scope](#1-scope)
2. [Requirements](#2-requirements)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Data Sources](#5-data-sources)
6. [Data Pipeline](#6-data-pipeline)
7. [Preprocessing Specification](#7-preprocessing-specification)
8. [Knowledge Graph Specification](#8-knowledge-graph-specification)
9. [Indexing Specification](#9-indexing-specification)
10. [Retrieval Specification](#10-retrieval-specification)
11. [Generation Specification](#11-generation-specification)
12. [Guardrails Specification](#12-guardrails-specification)
13. [Submission Specification](#13-submission-specification)
14. [Evaluation Specification](#14-evaluation-specification)
15. [Resource Plan](#15-resource-plan)
16. [Repository Structure](#16-repository-structure)
17. [Configuration Reference](#17-configuration-reference)
18. [Glossary](#18-glossary)

---

## 1. Scope

### 1.1 Purpose

Build an AI system that, given a Vietnamese legal question relevant to Small and Medium Enterprises (SMEs), returns:

- A natural-language Vietnamese answer grounded in primary legal sources.
- The list of relevant legal documents.
- The list of relevant articles cited inside the answer.

Output conforms to the competition JSON schema (Section 13).

### 1.2 In-scope legal domains

- Luật Doanh nghiệp and supporting decrees and circulars
- Luật Hỗ trợ Doanh nghiệp nhỏ và vừa
- Luật Thuế: thu nhập doanh nghiệp, giá trị gia tăng, thu nhập cá nhân, hóa đơn
- Luật Lao động and BHXH, BHYT, BHTN
- Luật Đầu tư, Thương mại, Kinh doanh
- Luật Hợp đồng and Dân sự (parts applicable to SMEs)
- Luật Kế toán, Kiểm toán
- Phá sản, Giải thể, Chia tách, Sáp nhập, Hợp nhất

### 1.3 Out-of-scope

- Hình sự (except SME-related administrative violations)
- Tố tụng chi tiết
- Specialized sectors not relevant to general SME operations (defense, healthcare regulations, education).
- Legal advice for non-SME entities.

### 1.4 Success criteria

| Metric | Target |
|---|---|
| Retrieval F2 (macro) on dev set | ≥ 0.55 |
| QA grounding rate (auto) | ≥ 0.80 |
| Inference latency per query (T4 GPU) | ≤ 15 seconds |
| Offline build time (cumulative GPU) | ≤ 30 hours |

---

## 2. Requirements

### 2.1 Functional requirements

| ID    | Requirement                                                                                                                                  |                           |          |
| -------| ----------------------------------------------------------------------------------------------------------------------------------------------| ---------------------------| ----------|
| FR-01 | The system MUST accept a JSON list of `{id, question}` and return a JSON list of `{id, question, answer, relevant_docs, relevant_articles}`. |                           |          |
| FR-02 | The `answer` field MUST cite at least one article using the pattern `Điều X`, where the cited article exists in the retrieved context.       |                           |          |
| FR-03 | Each entry in `relevant_docs` MUST follow the format `<law_id>\                                                                              | <Loại + Mã + Trích yếu>`. |          |
| FR-04 | Each entry in `relevant_articles` MUST follow the format `<law_id>\                                                                          | <Loại + Mã + Trích yếu>\  | Điều X`. |
| FR-05 | The `answer` MUST include a disclaimer that the information is for reference and does not replace formal legal consultation.                 |                           |          |
| FR-06 | The system MUST refuse to fabricate `law_id` values that are not present in the retrieved context.                                           |                           |          |
| FR-07 | The system MUST exclude articles from documents whose `tinh_trang_hieu_luc` indicates expired status.                                        |                           |          |

### 2.2 Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-01 | All language models used MUST have fewer than 14 billion parameters. |
| NFR-02 | All language models used MUST be open-source with downloadable weights. |
| NFR-03 | All language models used MUST be publicly released before 2026-03-01 Vietnam time. |
| NFR-04 | The system MUST NOT use external data beyond the designated legal corpus and the provided test set. |
| NFR-05 | All offline stages MUST be crash-recoverable through file-based checkpointing. |
| NFR-06 | All artifacts MUST be deterministic given a fixed random seed of 42. |
| NFR-07 | The inference pipeline MUST run on a single Kaggle/Colab T4 GPU (16 GB VRAM). |

### 2.3 Constraints

- Maximum 10 submissions per day on the public leaderboard.
- Maximum 5 submissions total during the Private Phase.
- Submission file MUST be `submission.zip` containing `results.json` at the root (no subdirectory).

---

## 3. System Architecture

### 3.1 Architectural overview

The system consists of two phases:

- **Offline indexing phase**: One-time construction of corpus, knowledge graph, and search indexes.
- **Online inference phase**: Per-query retrieval, generation, and validation against built artifacts.

```
┌──────────────── OFFLINE INDEXING ────────────────┐
│                                                  │
│  Legal Corpus ──► Preprocessing (6 stages) ──►   │
│                          │                       │
│                          ▼                       │
│           ┌──────────────┼──────────────┐        │
│           ▼              ▼              ▼        │
│       BM25 Index    Knowledge Graph  Dual FAISS  │
│                                                  │
└──────────────────────┬───────────────────────────┘
                       │ artifacts on disk
┌──────────────────────▼───────────────────────────┐
│                ONLINE INFERENCE                  │
│                                                  │
│  Query ──► Hybrid Retrieval (BM25+FAISS×2)       │
│              │                                   │
│              ▼                                   │
│         RRF Fusion (k=60)                        │
│              │                                   │
│              ▼                                   │
│         Graph Expansion (1-hop)                  │
│              │                                   │
│              ▼                                   │
│         Cross-Encoder Rerank → top-K=5           │
│              │                                   │
│              ▼                                   │
│         LLM Generation                           │
│              │                                   │
│              ▼                                   │
│         Guardrails                               │
│              │                                   │
│              ▼                                   │
│         JSON Submission Record                   │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 3.2 Module list

| Module | Responsibility | Inputs | Outputs |
|---|---|---|---|
| `data.stage1_filter` | Filter SME-scope documents | `metadata` config + `content` IDs | `stage1_sme_docs.parquet` |
| `data.stage2_parse_html` | Parse HTML into article-level records | `content` config + Stage 1 | `stage2_articles.parquet`, `stage2_parse_failures.jsonl` |
| `data.stage3_chunking` | Split long articles with overlap | Stage 2 | `stage3_chunks.parquet` |
| `data.stage4_summarize` | Generate summaries per chunk | Stage 3 | `stage4_enriched.parquet`, `summary_cache.jsonl` |
| `data.stage5_build_graph` | Build NetworkX knowledge graph | Stage 1, Stage 2/3, `relationships` config | `kg.gpickle` |
| `data.stage6_index` | Build BM25 and FAISS indexes | Stage 3 | `bm25.pkl`, `faiss_summary.index`, `faiss_full.index`, `chunk_meta.npy` |
| `retrieval.retriever` | Hybrid retrieval + graph expansion + rerank | query, all indexes, KG | top-K hits |
| `generation.generator` | LLM answer generation with grounded context | query, hits | answer string |
| `generation.guardrails` | Post-generation validation and retry | answer, hits | validated answer or fallback |
| `pipeline` | End-to-end inference orchestration | test set | `results.json` |

### 3.3 Artifact dependency graph

```
metadata ─┐
          ├──► stage1_sme_docs.parquet ─┬──► stage2_articles.parquet
content ──┘                              │     stage2_parse_failures.jsonl
                                         │           │
                                         │           ▼  (+ optional manual fixes)
                                         │     stage3_chunks.parquet
                                         │           │
                                         │           ▼
                                         │     stage4_enriched.parquet  +  summary_cache.jsonl
                                         │           │
                                         │           ├──► bm25.pkl
                                         │           ├──► faiss_summary.index
                                         │           ├──► faiss_full.index
                                         │           └──► chunk_meta.npy
                                         │
relationships ──────────────────────────┴──► kg.gpickle

Note: BM25/FAISS indexes can be built directly from `stage3_chunks.parquet`, with `stage4_enriched.parquet` as optional metadata only.
```

---

## 4. Technology Stack

### 4.1 Models

| Component | Model | Parameters | Quantization |
|---|---|---|---|
| Dense embedder | `BAAI/bge-m3` | 560M | fp16 |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` | 568M | fp16 |
| Summarization LLM | `Qwen/Qwen2.5-3B-Instruct` | 3B | 4-bit NF4 |
| Generation LLM | `google/gemma-3-12b-it` | 12B | 4-bit QAT |

### 4.2 Libraries

```
torch==2.4.1
transformers==4.46.0
accelerate==1.0.1
bitsandbytes==0.44.1
FlagEmbedding==1.3.2
sentence-transformers==3.2.1
faiss-cpu==1.8.0
rank-bm25==0.2.2
pyvi==0.1.1
underthesea==6.8.4
datasets==3.0.1
beautifulsoup4==4.12.3
lxml==5.3.0
pandas==2.2.3
pyarrow==17.0.0
networkx==3.4.2
huggingface-hub==0.26.0
orjson==3.10.7
tqdm==4.66.5
```

### 4.3 Runtime environments

| Environment | Used for | Constraints |
|---|---|---|
| Kaggle Notebook (GPU T4 ×2) | Indexing, summarization, inference | 16 GB VRAM ×2, 30 GPU-hours/week |
| Kaggle Notebook (CPU) | HTML parsing, BM25, graph build | 30 GB RAM, 9 hours/session |
| Google Colab (T4 free) | Prompt experiments, debugging | 12-15 GB VRAM, ~12 hours/session |

---

## 5. Data Sources

### 5.1 Legal corpus

**Source**: `th1nhng0/vietnamese-legal-documents` on Hugging Face.

**Origin**: [vbpl.vn](https://vbpl.vn/) (Vietnamese Government Legal Document Portal, operated by the Ministry of Justice). Crawled with a Scrapy crawler shipped under the `crawler/` directory of the dataset repository.

**License**: CC BY 4.0 (curation); legal documents are public domain under Vietnamese law.

### 5.2 Dataset configurations used

| Config | Split | Rows | Role |
|---|---|---|---|
| `metadata` | `data` | 153,420 | Document-level metadata for filtering and identification |
| `content` | `data` | 178,665 | Raw HTML body keyed by document ID |
| `relationships` | `data` | 897,890 | Directed edges between documents |

The `legacy` config (518k older documents in English field names) is NOT used.

### 5.3 `metadata` schema (full, 17 fields)

All 17 fields are present in every row; fields used by the pipeline are flagged.

| Column | Type | Used | Description |
|---|---|---|---|
| `id` | int | ✓ | Unique document ID; join key to `content` and `relationships` |
| `title` | str | ✓ | Full Vietnamese title |
| `so_ky_hieu` | str | ✓ | Official number, e.g. `13/2017/QH14`; maps to `law_id` |
| `ngay_ban_hanh` | str (DD/MM/YYYY) | ✓ | Issuance date |
| `loai_van_ban` | str | ✓ | Document type (enumerated in §5.5) |
| `ngay_co_hieu_luc` | str | | Effective date |
| `ngay_het_hieu_luc` | str | | Expiry date (empty if still in effect) |
| `nguon_thu_thap` | str | | Collection source (e.g. Công báo) |
| `ngay_dang_cong_bao` | str | | Official Gazette publication date |
| `nganh` | str | ✓ | Sector (broader category) |
| `linh_vuc` | str | ✓ | Legal field / sub-domain (enumerated in §5.6) |
| `co_quan_ban_hanh` | str | | Issuing authority |
| `chuc_danh` | str | | Signatory title (Chủ tịch, Bộ trưởng, …) |
| `nguoi_ky` | str | | Signatory name |
| `pham_vi` | str | | Geographical scope |
| `thong_tin_ap_dung` | str | | Implementation note |
| `tinh_trang_hieu_luc` | str | ✓ | Validity status (enumerated in §5.7) |

### 5.4 `content` schema

| Column | Type | Description |
|---|---|---|
| `id` | int | Document ID; join key to `metadata.id` |
| `content_html` | str | Raw HTML body of the document |

**Important coverage gap**: Not all documents in `metadata` have a corresponding row in `content`. The vbpl.vn portal provides only PDF scans for some documents; those are excluded from `content`. Expected coverage: ~149k of 153k metadata rows.

### 5.5 `loai_van_ban` enumerated values

Known values (from corpus statistics):

```
Quyết định, Nghị quyết, Kế hoạch, Thông tư, Thông báo, Chỉ thị,
Nghị định, Luật, Văn bản khác, Pháp lệnh, Thông tư liên tịch,
Văn bản hợp nhất, Hướng dẫn, Báo cáo, Điều ước quốc tế,
Công điện, Sắc lệnh, Lệnh, Văn bản WTO, Hiến pháp, Bộ luật
```

The pipeline retains only the regulatory-binding subset listed as `VALID_DOCUMENT_TYPES` in §7.1.

### 5.6 `linh_vuc` enumerated values

Known values (27 categories):

```
Bộ máy hành chính, Tài chính nhà nước, Văn hóa - Xã hội,
Tài nguyên - Môi trường, Thương mại, Xây dựng - Đô thị,
Bất động sản, Thể thao - Y tế, Thuế - Phí - Lệ Phí, Giáo dục,
Giao thông - Vận tải, Lao động - Tiền lương, Doanh nghiệp,
Đầu tư, Công nghệ thông tin, Xuất nhập khẩu, Quyền dân sự,
Tiền tệ - Ngân hàng, Bảo hiểm, Dịch vụ pháp lý, Thủ tục Tố tụng,
Vi phạm hành chính, Kế toán - Kiểm toán, Trách nhiệm hình sự,
Sở hữu trí tuệ, Chứng khoán, Lĩnh vực khác
```

The pipeline accepts the SME-relevant subset listed as `SME_LINH_VUC` in §7.1.

### 5.7 `tinh_trang_hieu_luc` enumerated values

Known values include:

```
Còn hiệu lực, Hết hiệu lực toàn bộ, Hết hiệu lực một phần,
Chưa có hiệu lực, Không xác định, Hết hiệu lực
```

The pipeline retains only documents whose status string contains the substring `"Còn hiệu lực"`.

### 5.8 `relationships` schema

| Column | Type | Description |
|---|---|---|
| `doc_id` | int (stored as str) | Source document ID; join key to `metadata.id` |
| `other_doc_id` | int (stored as str) | Target document ID |
| `relationship` | str | Vietnamese label; full enumeration in §8.4 |

Each relationship is stored as a directed edge `doc_id → other_doc_id`. Reciprocal relations (e.g. "Văn bản sửa đổi bổ sung" vs. "Văn bản bị sửa đổi bổ sung") appear as separate edges with distinct labels.

### 5.9 Test set (provided by organizers)

```json
{
  "id": <int>,
  "question": "<Vietnamese legal question>"
}
```

---

## 6. Data Pipeline

### 6.1 Pipeline stages

The data pipeline executes six stages sequentially plus one optional manual review stage. Each stage reads from disk, writes to disk, and is independently re-runnable.

| Stage | Name | Input artifact | Output artifact |
|---|---|---|---|
| 1 | Scope filter | `metadata` config | `stage1_sme_docs.parquet` |
| 2 | HTML parsing | `content` config + Stage 1 | `stage2_articles.parquet` + `stage2_parse_failures.jsonl` |
| 2.5 | Manual review (optional) | `stage2_parse_failures.jsonl` | `stage2_manual_fixes.json` |
| 3 | Chunking | Stage 2 (+ manual fixes if present) | `stage3_chunks.parquet` |
| 4 | Summary injection | Stage 3 | `stage4_enriched.parquet` + `summary_cache.jsonl` |
| 5 | Graph construction | Stage 1 + Stage 2/3 + `relationships` config | `kg.gpickle` |
| 6 | Indexing | Stage 3 | `bm25.pkl`, `faiss_summary.index`, `faiss_full.index`, `chunk_meta.npy` |

### 6.2 Expected volumes

| Stage | Output volume |
|---|---|
| 1 | 3,000 – 8,000 documents (SME scope, in effect, regulatory-binding types) |
| 2 | 30,000 – 80,000 articles; parse failures logged for review |
| 3 | 40,000 – 100,000 chunks (some long articles split by Khoản) |
| 4 | One enriched record per Stage 3 chunk + matching cache lines |
| 5 | ~50,000 nodes, ~200,000 – 350,000 edges |
| 6 | 3 indexes, ~3 GB total on disk |

Volume ranges reflect uncertainty about HTML parse success rate and final SME keyword tuning.

### 6.3 Determinism

- All random seeds: `42`.
- BM25 parameters: `k1=1.5, b=0.75`.
- FAISS index type: `IndexFlatIP` (exact search, no approximation).
- LLM decoding: `do_sample=False, temperature=0.1`.

### 6.4 Checkpointing

- Parquet files are written atomically using a temporary path followed by rename.
- The summary cache is appended line by line in JSONL format; restart resumes from the last completed chunk.
- The inference loop persists `results.json` every 50 records.

---

## 7. Preprocessing Specification

### 7.1 Stage 1 — Scope filter

**Input**: `metadata` config (153,420 rows).

**Output**: `stage1_sme_docs.parquet`.

**Filter rule**: a document is retained if and only if ALL of the following hold.

1. SME relevance — at least ONE of the following is true:
   - The lowercased `title` contains at least one keyword from `SME_TITLE_KEYWORDS`.
   - `linh_vuc` exactly matches one of `SME_LINH_VUC`.
   - `nganh` exactly matches one of `SME_NGANH`.
2. The `tinh_trang_hieu_luc` field contains the substring `"Còn hiệu lực"`.
3. `loai_van_ban` is in `VALID_DOCUMENT_TYPES`.
4. There exists a row in the `content` config with matching `id` (PDF-only documents excluded).

**Constants** (defined in `config/default.yaml`):

```yaml
SME_TITLE_KEYWORDS:
  - "doanh nghiệp"
  - "hỗ trợ doanh nghiệp nhỏ và vừa"
  - "đầu tư"
  - "kinh doanh"
  - "thuế"
  - "hóa đơn"
  - "kế toán"
  - "kiểm toán"
  - "phá sản"
  - "thương mại"
  - "lao động"
  - "bảo hiểm xã hội"
  - "bhxh"
  - "bhyt"
  - "việc làm"
  - "công đoàn"
  - "hợp đồng"
  - "sở hữu trí tuệ"
  - "góp vốn"
  - "cổ phần"
  - "chi nhánh"

# Exact-match values from the canonical 27-category linh_vuc enumeration (§5.6).
SME_LINH_VUC:
  - "Doanh nghiệp"
  - "Đầu tư"
  - "Thương mại"
  - "Thuế - Phí - Lệ Phí"
  - "Kế toán - Kiểm toán"
  - "Lao động - Tiền lương"
  - "Bảo hiểm"
  - "Tài chính nhà nước"
  - "Tiền tệ - Ngân hàng"
  - "Xuất nhập khẩu"
  - "Sở hữu trí tuệ"
  - "Chứng khoán"
  - "Quyền dân sự"
  - "Vi phạm hành chính"

# nganh is a broader sector classification; matched as exact strings.
SME_NGANH:
  - "Doanh nghiệp"
  - "Tài chính"
  - "Lao động"
  - "Thương mại"
  - "Đầu tư"

VALID_DOCUMENT_TYPES:
  - Sắc lệnh
  - Quyết định
  - Chỉ thị
  - Nghị quyết
  - Thông tư liên tịch
  - Thông tư
  - Nghị định
  - Pháp lệnh
  - Lệnh
  - Luật
  - Sắc luật
  - Chương trình
  - Công ước
  - Nghị định thư
  - Hiến pháp
  - Bộ luật
  - Nghị quyết liên tịch
  - Thông báo
  - Hiệp định
  - Văn bản hợp nhất
  - Công văn
  - Bản ghi nhớ
  - None
  - Thỏa thuận
  - Nghị Quyết
  - Thông tư liên bộ
  - Văn bản khác
  - Văn bản liên quan
```

**Derived fields**:

- `law_id = so_ky_hieu`
- `ten_van_ban = f"{loai_van_ban} {so_ky_hieu} {trich_yeu}"`, where `trich_yeu` is the `title` with any leading occurrence of `"{loai_van_ban} (số )?{so_ky_hieu}"` removed (case-insensitive) and stripped of leading punctuation `-–:`.

**Quality gates**:

- Assert `3,000 ≤ output_rows ≤ 8,000`. If outside this range, audit the constant lists.
- Assert no duplicate `id`.
- Assert no duplicate `(law_id, ten_van_ban)` pair.
- Assert every retained `id` exists in the `content` config (no PDF-only documents).

### 7.2 Stage 2 — HTML parsing

**Input**: `content` config (streaming over the full dataset) joined to Stage 1 document IDs.

**Output**:

- `stage2_articles.parquet` — successfully parsed articles.
- `stage2_parse_failures.jsonl` — documents for which parsing produced zero articles or fewer than two articles when the title suggests a multi-article document.

**Output schema** (`stage2_articles.parquet`):

| Column | Type | Description |
|---|---|---|
| `doc_id` | int | Source document ID |
| `law_id` | str | `so_ky_hieu` |
| `ten_van_ban` | str | Loại + Mã + Trích yếu |
| `loai_van_ban` | str | Document type |
| `ngay_ban_hanh` | str | Issuance date |
| `nganh` | str | Sector |
| `linh_vuc` | str | Sub-field |
| `phan` | str | Phần header text or empty |
| `chuong` | str | Chương header text or empty |
| `muc` | str | Mục header text or empty |
| `dieu_so` | str | Article identifier, e.g. `Điều 13` or `Điều 13a` |
| `dieu_ten` | str | Article title (text after the period on the Điều line) |
| `noi_dung` | str | Article body (excluding the Điều header line) |
| `start_char` | int | Start offset of the article in the plain-text body |
| `end_char` | int | End offset of the article in the plain-text body |
| `doc_uid` | str | `f"{law_id}\|{ten_van_ban}\|{dieu_so}"` |

**Regex patterns** (compiled with `re.MULTILINE`):

```python
RE_PHAN   = r"^\s*(Phần(?:\s+thứ)?\s+[\dIVXLCDM\w]+.*)$"
RE_CHUONG = r"^\s*(Chương\s+[\dIVXLCDM\w]+.*)$"
RE_MUC    = r"^\s*(Mục\s+[\dIVXLCDM\w]+.*)$"
RE_DIEU   = r"^\s*Điều\s+(\d+)([a-zđ]?)\s*(?:[\.:\-–—)]\s*)?(.*)$"
RE_KHOAN  = r"^\s*(\d+)\.\s+(.*)$"
RE_DIEM   = r"^\s*([a-zđ])\)\s+(.*)$"
```

Notes on the Vietnamese legal-text format that drive these patterns:

- "Phần" may appear as `Phần thứ X` or `Phần X`, and documents can use Arabic digits, Roman numerals, or word-like counters.
- "Mục" may be numbered with Arabic digits, Roman numerals, or other word-like tokens.
- "Điều" is numbered with Arabic digits; amended documents may use suffixes such as `Điều 13a`, `Điều 13b`, or `Điều 32đ`.
- `RE_DIEU` accepts optional separator characters after the article number: `. : - – — )`.
- The parser works on the cleaned plain-text projection of the HTML body, not on CSS classes.

**Algorithm**:

1. For each `(doc_id, content_html)` whose `doc_id` is in Stage 1:
   a. Parse HTML with `BeautifulSoup(content_html, "html.parser")`.
   b. Remove `script` and `style` tags.
   c. Extract text with `soup.get_text("\n", strip=True)`.
   d. Normalize whitespace: `re.sub(r"[ \t]+", " ", text)` and `re.sub(r"\n{2,}", "\n", text)`.
   e. Deduplicate consecutive identical lines.
2. Locate all `Điều` boundaries with `RE_DIEU` and build spans from each boundary start to the next boundary or end of text.
3. Assign hierarchy context to each article by recording the most recent preceding `Phần`, `Chương`, and `Mục` headers.
4. For each article span, emit one row with:
   - `dieu_so = f"Điều {dieu_num}{dieu_suffix}"`
   - `dieu_ten = dieu_title.strip()`
   - `noi_dung` = text after the Điều header through the next article boundary or document end
   - `phan`, `chuong`, `muc` from the nearest preceding headers
   - `doc_uid = f"{law_id}|{ten_van_ban}|{dieu_so}"`
5. Drop rows where `len(noi_dung) < 30`.
6. If no `Điều` boundaries are found, build a single fallback article from the whole cleaned text.
   - If the cleaned text is shorter than 30 characters, emit failure `text_too_short_after_cleaning`.
7. If the title is law-like (`Luật`, `Bộ luật`, `Nghị định`) and the parser produces zero articles after fallback, emit failure `zero_dieu_law_like_doc`.
8. If the title is law-like and the parser finds boundaries but all articles are dropped for short content, emit failure `zero_parsable_dieu_law_like_doc`.
9. If exactly one article remains and the title is law-like, emit failure `single_dieu_law_like_doc`.
10. Save successful article rows to `stage2_articles.parquet` and failures to `stage2_parse_failures.jsonl`.

**Final keep/drop policy (Decided 2026-06-10)**

- Keep fallback document-level articles produced when no `Điều` markers are found. Fallback rows use `dieu_so = "Điều VB"` and are emitted to `stage2_articles.parquet` (they are also logged as `zero_dieu_law_like_doc` when the title is law-like).
- Keep `single_dieu_law_like_doc` article rows (emit the single parsed article) but log the failure for manual review.
- Drop (do not emit article rows) for `text_too_short_after_cleaning` (cleaned text < 30 characters) and for `zero_parsable_dieu_law_like_doc` (title suggests a multi-article law but parser produced no parsable article rows). Both are recorded in `stage2_parse_failures.jsonl` for audit.

**Quality gates**:

- Drop articles where `len(noi_dung) < 30` characters.
- Sort parsed articles by `noi_dung` length descending, then deduplicate by `(doc_id, dieu_so)`, keeping the longest row.
- Assert each `doc_uid` is unique in the final output.
- Assert no duplicate `(doc_id, dieu_so)` pairs remain after deduplication.

### 7.3 Stage 3 — Chunking

**Input**: `stage2_articles.parquet` (merged with `stage2_manual_fixes.json` if present).

**Output**: `stage3_chunks.parquet`.

**Parameters**:

- `MAX_TOKENS = 1024`
- `OVERLAP_TOKENS = 128`
- Tokenizer: `AutoTokenizer.from_pretrained("google/gemma-3-12b-it", token=hf_token, trust_remote_code=True)`, where `hf_token` is loaded from `--hf-token`, `HF_TOKEN`, or `HUGGINGFACE_HUB_TOKEN`.

**Implementation note**:

- Stage 3 explicitly passes the Hugging Face auth token into tokenizer loading so that gated repos like `google/gemma-3-12b-it` are accessed with authentication instead of using unauthenticated default requests.
- If the token is missing or invalid, tokenizer loading may still fail even when the environment variable is present.

**Algorithm**:

1. Compute `breadcrumb`: join non-empty levels with ` > `:
   ```
   f"{loai_van_ban} > {ten_van_ban}"
   + (f" > {phan}" if phan else "")
   + (f" > {chuong}" if chuong else "")
   + (f" > {muc}" if muc else "")
   + f" > {dieu_so}. {dieu_ten}"
   ```
2. Tokenize `noi_dung` and count tokens.
3. If `n_tokens ≤ MAX_TOKENS`: emit a single chunk with
   `chunk_text = breadcrumb + "\n" + noi_dung`, `chunk_id = f"{doc_uid}#0"`, `part_idx = 0`.
4. Otherwise:
   - Split `noi_dung` into Khoản pieces using `re.split(r"(?=^\s*\d+\.\s)", noi_dung, flags=re.M)`.
   - Pack Khoản pieces into chunks using a greedy first-fit policy: while adding the next Khoản keeps the cumulative token count ≤ `MAX_TOKENS`, append it; otherwise flush the current chunk and start a new one.
   - At chunk boundary, prepend the last Khoản of the previous chunk to the new chunk (overlap). If a single Khoản exceeds `MAX_TOKENS`, split it further at sentence boundaries.
   - Each emitted chunk receives `chunk_id = f"{doc_uid}#{part_idx}"` with `part_idx` increasing from 0.
5. Every chunk carries the breadcrumb as a prefix in `chunk_text`.

**Validation note**:

- Verified run output: `stage2_articles.parquet` with `56,269` rows produced `stage3_chunks.parquet` with `74,107` chunks and `12,633` unique documents.
- Observed warnings are expected for this pipeline stage: PyTorch is disabled on the local environment (`torch==2.2.1`), Windows symlink caching is degraded, and Gemma BPE tokenizer emits a cleanup warning.

**Output schema**: Stage 2 schema + `breadcrumb`, `chunk_id`, `chunk_text`, `part_idx`.

### 7.4 Stage 4 — Summary injection (optional)

**Input**: `stage3_chunks.parquet`.

**Output**: `stage4_enriched.parquet`, `summary_cache.jsonl`.

**Note**: This stage is optional. The main KG and index pipeline can run without it using Stage 2 and Stage 3 artifacts.

**If used**:
- Model: `Qwen/Qwen2.5-3B-Instruct`, loaded with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)`.
- Generation parameters: `batch_size=4, max_new_tokens=256, do_sample=False, temperature=0.1, repetition_penalty=1.05`.
- Prompt template: generate `short` (summary) and `key` (key bullet points).
- Cache format: `summary_cache.jsonl`
- Enriched text format: optional `enriched_text` field.

**Fallback**: If this stage is not run, the pipeline should still build:
- KG from `stage2_articles.parquet` + `relationships`
- BM25/FAISS from `stage3_chunks.parquet`

---

## 8. Knowledge Graph Specification

### 8.1 Graph type

`networkx.MultiDiGraph` — directed multigraph allowing multiple edges between the same pair of nodes with different relations.

### 8.2 Node types

| Type | Key format | Attributes |
|---|---|---|
| Document | `DOC:{doc_id}` | `type`, `doc_id`, `law_id`, `ten`, `loai`, `nganh`, `linh_vuc`, `ngay_ban_hanh`, `tinh_trang_hieu_luc`, `ngay_co_hieu_luc`, `ngay_het_hieu_luc` |
| Article | `ART:{doc_uid}` | `type`, `doc_uid`, `doc_id`, `law_id`, `ten_van_ban`, `dieu_so`, `dieu_ten`, `phan`, `chuong`, `muc`, `short` (optional), `enriched_text` (optional) |
| Chunk | `CHUNK:{chunk_id}` | `type`, `chunk_id`, `doc_uid`, `doc_id`, `rowidx`, `part_idx`, `breadcrumb`, `law_id` (optional), `dieu_so` (optional) |
| Concept | `CONCEPT:{name_lower}` | `type`, `name`, `name_lower` |

`DOC` is the central node: every original legal relation lives at the document layer. `ART` is the citation unit (answers cite `Điều X`) and bridges a document to its enforceable content. `CHUNK` is the retrieval unit and the primary source for concept extraction, sitting as an intermediate layer between `ART` and `CONCEPT` without becoming a central legal node. `CONCEPT` is an auxiliary semantic layer that groups recurring legal topics across documents via their chunks.

### 8.3 Edge types

Only canonical-direction edges are stored. Reverse traversal is handled at query time in code. All `DOC → DOC` edges come from the source `relationships` dataset; no LLM-generated doc-doc edges.

| Edge label | From | To | Source |
|---|---|---|---|
| `HAS_ARTICLE` | Document | Article | Derived from Stage 2 (mandatory) |
| `HAS_CHUNK` | Article | Chunk | Derived from Stage 3 (`stage3_chunks.parquet`) |
| `AMENDS` | Document | Document | `relationships` dataset (canonical direction) |
| `REPLACES` | Document | Document | `relationships` dataset (canonical direction) |
| `DETAILS` | Document | Document | `relationships` dataset (canonical direction) |
| `CITES_REF` | Document | Document | `relationships` dataset (canonical direction) |
| `BASED_ON` | Document | Document | `relationships` dataset (canonical direction) |
| `CONSOLIDATES` | Document | Document | `relationships` dataset (canonical direction) |
| `CORRECTS` | Document | Document | `relationships` dataset (canonical direction) |
| `RELATED_CONTENT` | Document | Document | `relationships` dataset (keep only if label is clean) |
| `MENTIONS` | Chunk | Concept | Rule-based string match on chunk text (primary); optional `enriched_text` fallback if Stage 4 exists |

**Not stored:** reverse-only labels (`AMENDED_BY`, `REPLACED_BY`, `DETAILED_BY`, `CITED_BY_REF`, `CONSOLIDATED_BY`, `CORRECTED_BY`) and `RELATED_LANGUAGE`. Reverse hops are computed in code at query time.

### 8.4 Relationship label mapping

The `relationship` field in the source dataset uses Vietnamese labels. Only the canonical-direction labels are mapped; reverse-direction labels and `RELATED_LANGUAGE` are deliberately excluded so they fall into the dropped set during graph build. The full mapping is maintained in [`config/relationship_mapping.yaml`](config/relationship_mapping.yaml).

```yaml
RELATIONSHIP_MAP:
  # Amendment (canonical direction: the amending document points to the amended one)
  "Văn bản sửa đổi bổ sung": "AMENDS"

  # Replacement
  "Văn bản thay thế": "REPLACES"

  # Guidance / detailing (most valuable for procedure/condition queries)
  "Văn bản hướng dẫn": "DETAILS"

  # Cross-reference / citation
  "Văn bản được dẫn chiếu": "CITES_REF"

  # Legal basis
  "Văn bản được căn cứ": "BASED_ON"

  # Consolidation
  "Văn bản hợp nhất": "CONSOLIDATES"

  # Correction
  "Văn bản đính chính": "CORRECTS"

  # Same-content (retain only if experimentally clean)
  "Văn bản liên quan cùng nội dung": "RELATED_CONTENT"

# Labels explicitly dropped — reverse-only or low-value:
DROPPED_LABELS:
  - "Văn bản bị sửa đổi bổ sung"   # reverse of AMENDS
  - "Văn bản bị thay thế"          # reverse of REPLACES
  - "Văn bản được hướng dẫn"       # reverse of DETAILS
  - "Văn bản được hợp nhất"        # reverse of CONSOLIDATES
  - "Văn bản bị đính chính"        # reverse of CORRECTS
  - "Văn bản liên quan ngôn ngữ"   # RELATED_LANGUAGE — dropped
```

Labels not in `RELATIONSHIP_MAP` are logged as warnings and skipped during graph build.

**Edges used by graph expansion at retrieval time** (Section 10.4):

```
{DETAILS, AMENDS, REPLACES, CITES_REF, BASED_ON}
```

Reverse hops over these same edges are also traversed in code (e.g. following an incoming `DETAILS` edge to find the more-specific document). `CONSOLIDATES`, `CORRECTS`, `RELATED_CONTENT` are stored but not traversed during inference.

### 8.5 Concept seed list

Curated list of approximately 50-100 SME-relevant legal concepts, stored in `config/legal_concepts.yaml`. Matching is case-insensitive substring search against chunk text (or optional `enriched_text` when available), with Stage 3 chunks as the primary evidence source.

```yaml
LEGAL_CONCEPTS:
  - "vốn điều lệ"
  - "vốn pháp định"
  - "doanh nghiệp tư nhân"
  - "công ty cổ phần"
  - "công ty TNHH"
  - "công ty hợp danh"
  - "hợp đồng lao động"
  - "hợp đồng kinh tế"
  - "bảo hiểm xã hội"
  - "bảo hiểm y tế"
  - "bảo hiểm thất nghiệp"
  - "thuế giá trị gia tăng"
  - "thuế thu nhập doanh nghiệp"
  - "thuế thu nhập cá nhân"
  - "hóa đơn điện tử"
  - "hóa đơn giá trị gia tăng"
  - "người lao động"
  - "người sử dụng lao động"
  - "đại hội đồng cổ đông"
  - "hội đồng quản trị"
  - "ban kiểm soát"
  - "kiểm soát viên"
  - "giám đốc"
  - "tổng giám đốc"
  - "chi nhánh"
  - "văn phòng đại diện"
  - "địa điểm kinh doanh"
  - "sáp nhập"
  - "chia tách"
  - "hợp nhất"
  - "giải thể"
  - "phá sản"
  - "góp vốn"
  - "chuyển nhượng vốn"
  - "cổ phần ưu đãi"
  - "cổ phần phổ thông"
  - "thời giờ làm việc"
  - "thời giờ nghỉ ngơi"
  - "tiền lương"
  - "phụ cấp"
  - "kỷ luật lao động"
  - "sa thải"
  - "chấm dứt hợp đồng lao động"
  - "đầu tư trong nước"
  - "đầu tư nước ngoài"
  - "ưu đãi đầu tư"
```

### 8.6 Build procedure

The build runs as five sequential sub-stages. Each sub-stage appends to the same persisted `kg.gpickle`.

**Step 1 (Stage 5.2)** — Document nodes from `stage1_sme_docs.parquet`:

```python
for row in sme_docs.itertuples():
    G.add_node(f"DOC:{row.id}",
               type="Document",
               doc_id=str(row.id),
               law_id=row.law_id,
               ten=row.ten_van_ban,
               loai=row.loai_van_ban,
               nganh=row.nganh,
               linh_vuc=row.linh_vuc,
               ngay_ban_hanh=row.ngay_ban_hanh,
               tinh_trang_hieu_luc=row.tinh_trang_hieu_luc,
               ngay_co_hieu_luc=row.ngay_co_hieu_luc,
               ngay_het_hieu_luc=row.ngay_het_hieu_luc)
```

**Step 2 (Stage 5.3)** — Article nodes and `HAS_ARTICLE` edges from `stage2_articles.parquet`:

```python
for art in articles.drop_duplicates("doc_uid").itertuples():
    art_id = f"ART:{art.doc_uid}"
    G.add_node(art_id, type="Article",
               doc_uid=art.doc_uid,
               doc_id=str(art.doc_id),
               law_id=art.law_id,
               ten_van_ban=art.ten_van_ban,
               dieu_so=art.dieu_so,
               dieu_ten=art.dieu_ten,
               phan=art.phan,
               chuong=art.chuong,
               muc=art.muc)
    # Optional Stage 4 fields — only if present
    if hasattr(art, "short"):
        G.nodes[art_id]["short"] = art.short
    if hasattr(art, "enriched_text"):
        G.nodes[art_id]["enriched_text"] = art.enriched_text
    G.add_edge(f"DOC:{art.doc_id}", art_id, relation="HAS_ARTICLE")
```

**Step 3 (Stage 5.4)** — Chunk nodes and `HAS_CHUNK` edges from `stage3_chunks.parquet`:

```python
for ch in chunks.itertuples():
    chunk_key = f"CHUNK:{ch.chunk_id}"
    G.add_node(chunk_key,
               type="Chunk",
               chunk_id=ch.chunk_id,
               doc_uid=ch.doc_uid,
               doc_id=str(ch.doc_id),
               rowidx=ch.rowidx,
               part_idx=ch.part_idx,
               breadcrumb=ch.breadcrumb)
    G.add_edge(f"ART:{ch.doc_uid}", chunk_key, relation="HAS_CHUNK")
```

**Step 4 (Stage 5.5)** — Cross-document edges from `relationships` dataset:

```python
RELATIONSHIP_MAP = load_yaml("config/relationship_mapping.yaml")["RELATIONSHIP_MAP"]
RELATION_WHITELIST = set(load_yaml("config/relationship_mapping.yaml")["RELATION_WHITELIST"])

rels = load_dataset("th1nhng0/vietnamese-legal-documents",
                    "relationships", split="data").to_pandas()
sme_ids = set(sme_docs["id"].astype(str))
rels = rels[rels["doc_id"].astype(str).isin(sme_ids)
          & rels["other_doc_id"].astype(str).isin(sme_ids)]

for r in rels.itertuples():
    rel_enum = RELATIONSHIP_MAP.get(r.relationship)
    if rel_enum is None or rel_enum not in RELATION_WHITELIST:
        # Log unmapped / dropped labels and skip — no reverse edges created
        continue
    G.add_edge(f"DOC:{r.doc_id}", f"DOC:{r.other_doc_id}", relation=rel_enum)
```

**Step 5 (Stage 5.6)** — Concept nodes and `MENTIONS` edges (chunk-first extraction):

```python
for c in LEGAL_CONCEPTS:
    norm = c.lower()
    G.add_node(f"CONCEPT:{norm}", type="Concept", name=c, name_lower=norm)

# Primary source: chunk_text from stage3_chunks.parquet
for ch in chunks.itertuples():
    text_lower = ch.chunk_text.lower() if hasattr(ch, "chunk_text") else None

    # Fallback to enriched_text only when chunk_text is absent
    if text_lower is None and hasattr(ch, "enriched_text"):
        text_lower = ch.enriched_text.lower()

    if text_lower is None:
        continue

    concepts_matched = 0
    for c in LEGAL_CONCEPTS:
        if concepts_matched >= 3:          # max 3 concepts per chunk
            break
        if c.lower() in text_lower:
            chunk_key  = f"CHUNK:{ch.chunk_id}"
            concept_key = f"CONCEPT:{c.lower()}"
            if not G.has_edge(chunk_key, concept_key):  # dedup (chunk_id, concept_id)
                G.add_edge(chunk_key, concept_key, relation="MENTIONS")
                concepts_matched += 1
```

### 8.7 Persistence

```python
with open("kg.gpickle", "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
```

### 8.8 Expected statistics

| Component | Count |
|---|---|
| Document nodes | ~4,000 |
| Article nodes | ~50,000 |
| Chunk nodes | ~74,107 |
| Concept nodes | ~50–100 |
| `HAS_ARTICLE` edges | ~50,000 |
| `HAS_CHUNK` edges | ~74,107 |
| Cross-document edges (canonical direction only) | ~50,000–175,000 |
| `MENTIONS` edges | ~30,000–50,000 |

Cross-document edge count is lower than the old spec because reverse-direction labels and `RELATED_LANGUAGE` are no longer stored (KG.md §6). The old spec's range of 150k–350k included both directions of every pair; the new range reflects only the canonical half of each pair.

---

## 9. Indexing Specification

### 9.1 BM25 index

- Library: `rank_bm25.BM25Okapi`.
- Input text: `chunk_text` from `stage3_chunks.parquet` or optional `enriched_text` from `stage4_enriched.parquet` if available.
- Tokenization: `pyvi.ViTokenizer.tokenize(text).lower().split()`.
- Parameters: `k1=1.5, b=0.75`.

**Output** (`bm25.pkl`):

```python
{
  "bm25": BM25Okapi,
  "doc_uids": np.ndarray[str],
  "chunk_ids": np.ndarray[str]
}
```

### 9.2 FAISS summary index

- Embedder: `BAAI/bge-m3`, `use_fp16=True`.
- Input text: `(short + " " + " ".join(key)).strip()` if Stage 4 metadata is available; otherwise use article/chunk title/text.
- Encoding parameters: `batch_size=32, max_length=256, return_dense=True`.
- Normalization: L2-normalize before insertion.
- Index type: `faiss.IndexFlatIP`.
- Dimension: 1024.

**Output**: `faiss_summary.index`.

### 9.3 FAISS full index

- Embedder: `BAAI/bge-m3`, `use_fp16=True`.
- Input text: `chunk_text` from `stage3_chunks.parquet` or optional `enriched_text` if Stage 4 is available.
- Encoding parameters: `batch_size=8, max_length=1024, return_dense=True`.
- Normalization: L2-normalize before insertion.
- Index type: `faiss.IndexFlatIP`.
- Dimension: 1024.

**Output**: `faiss_full.index`.

### 9.4 Metadata sidecar

`chunk_meta.npy` — structured NumPy array with columns:

```
chunk_id (str), doc_uid (str), law_id (str), ten_van_ban (str),
dieu_so (str), doc_id (int), row_idx (int)
```

The `row_idx` of each chunk equals its row index in `bm25.pkl`, in `faiss_summary.index`, and in `faiss_full.index`. If `stage4_enriched.parquet` is available, its row ordering should match the indexed chunks, but the core pipeline does not require it.

---

## 10. Retrieval Specification

### 10.1 Pipeline stages

```
query
  │
  ▼
[1] Encode query (BM25 tokens + bge-m3 vector)
  │
  ▼
[2] Parallel retrieval: BM25, FAISS-summary, FAISS-full (top-50 each)
  │
  ▼
[3] RRF fusion → top-30 candidates
  │
  ▼
[4] Graph expansion → top-50 candidates
  │
  ▼
[5] Cross-encoder rerank → top-K hits
  │
  ▼
final hits
```

### 10.2 Parameters

| Parameter                | Value |
| --------------------------| -------|
| `TOP_BM25`               | 50    |
| `TOP_DENSE_SUMMARY`      | 50    |
| `TOP_DENSE_FULL`         | 50    |
| `RRF_K`                  | 60    |
| `FUSED_TOP`              | 30    |
| `GRAPH_DISCOUNT_DOC`     | 0.6   |
| `GRAPH_DISCOUNT_CONCEPT` | 0.3   |
| `EXPANDED_TOP`           | 50    |
| `RERANK_MAX_INPUT_CHARS` | 2000  |
| `FINAL_TOP_K`            | 5     |

### 10.3 RRF fusion formula

For a candidate `d` appearing in retrievers `R_1, ..., R_m` at ranks `r_1, ..., r_m`:

```
score(d) = Σ_{i: d ∈ R_i} 1 / (RRF_K + r_i)
```

Candidates are sorted by score descending; top `FUSED_TOP` are kept.

### 10.4 Graph expansion algorithm

The KG stores only the canonical direction of each cross-document relation (KG.md §2.1). Reverse hops are reconstructed at query time by reading incoming edges over the same canonical relations.

```python
# Cross-document relations traversed during expansion (canonical direction).
# Reverse hops are walked via G.in_edges over the same set, so there is no
# separate AMENDED_BY / REPLACED_BY / DETAILED_BY / CITED_BY_REF / BASIS_OF
# entry — those are derived in code.
EXPANSION_DOC_RELS = {
    "DETAILS",
    "AMENDS",
    "REPLACES",
    "CITES_REF",
    "BASED_ON",
}


def graph_expand(candidates, discount_doc=0.6, discount_concept=0.3, top_n=50):
    expanded = dict(candidates)
    for idx, score in candidates:
        art_uid = df.iloc[idx].doc_uid
        art_node = f"ART:{art_uid}"
        if art_node not in G:
            continue

        # 1-hop A: find parent document via incoming HAS_ARTICLE edge
        doc_node = next(
            (p for p, _, d in G.in_edges(art_node, data=True)
             if d.get("relation") == "HAS_ARTICLE"),
            None,
        )
        if doc_node is None:
            continue

        # 1-hop B: traverse cross-document relations in BOTH directions
        # (canonical out + canonical in) since reverse edges are not stored.
        neighbor_docs = set()
        for _, nb_doc, edge in G.out_edges(doc_node, data=True):
            if edge.get("relation") in EXPANSION_DOC_RELS:
                neighbor_docs.add(nb_doc)
        for nb_doc, _, edge in G.in_edges(doc_node, data=True):
            if edge.get("relation") in EXPANSION_DOC_RELS:
                neighbor_docs.add(nb_doc)

        for nb_doc in neighbor_docs:
            for _, nb_art, edge in G.out_edges(nb_doc, data=True):
                if edge.get("relation") != "HAS_ARTICLE":
                    continue
                if G.nodes[nb_art].get("type") != "Article":
                    continue
                nb_uid = G.nodes[nb_art]["doc_uid"]
                nb_idx = uid_to_idx.get(nb_uid)
                if nb_idx is None:
                    continue
                expanded[nb_idx] = max(
                    expanded.get(nb_idx, 0.0),
                    score * discount_doc,
                )

        # 1.5-hop: concept co-mention via the chunk layer (CHUNK -> CONCEPT).
        for _, chunk_node, edge in G.out_edges(art_node, data=True):
            if edge.get("relation") != "HAS_CHUNK":
                continue
            for _, concept, edge2 in G.out_edges(chunk_node, data=True):
                if edge2.get("relation") != "MENTIONS":
                    continue
                for sibling_chunk, _, edge3 in G.in_edges(concept, data=True):
                    if edge3.get("relation") != "MENTIONS":
                        continue
                    if sibling_chunk == chunk_node or not sibling_chunk.startswith("CHUNK:"):
                        continue
                    sib_art = next(
                        (p for p, _, d in G.in_edges(sibling_chunk, data=True)
                         if d.get("relation") == "HAS_CHUNK"),
                        None,
                    )
                    if sib_art is None or sib_art == art_node:
                        continue
                    sib_uid = G.nodes[sib_art]["doc_uid"]
                    sib_idx = uid_to_idx.get(sib_uid)
                    if sib_idx is None or sib_idx in expanded:
                        continue
                    expanded[sib_idx] = score * discount_concept

    return sorted(expanded.items(), key=lambda x: -x[1])[:top_n]
```

### 10.5 Rerank

- Model: `BAAI/bge-reranker-v2-m3`, loaded via `FlagReranker`, `use_fp16=True`.
- Pair construction: `(query, enriched_text[:RERANK_MAX_INPUT_CHARS])`.
- Scoring: `compute_score(pairs, normalize=True)` returns relevance scores in `[0, 1]`.
- Output: sorted by score descending, top `FINAL_TOP_K` kept.

### 10.6 Hit record schema

Each final hit is a dictionary containing all columns of `stage4_enriched.parquet` for that row, plus:

```python
{
  "row_idx": int,
  "rerank_score": float
}
```

---

## 11. Generation Specification

### 11.1 Model configuration

- Model ID: `google/gemma-3-12b-it`.
- Quantization: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`.
- Attention implementation: `attn_implementation="eager"`.
- Device map: `device_map="auto"`.

### 11.2 Decoding parameters

| Parameter | Value |
|---|---|
| `max_new_tokens` | 900 |
| `do_sample` | False |
| `temperature` | 0.1 |
| `repetition_penalty` | 1.05 |

### 11.3 Prompt structure

**System message**:

```
Bạn là Trợ lý Pháp lý cho doanh nghiệp nhỏ và vừa (SME) tại Việt Nam.
Bạn CHỈ được dùng các CĂN CỨ PHÁP LUẬT được cung cấp; tuyệt đối không
bịa ra điều luật hay văn bản khác. Luôn trích dẫn dưới dạng "Điều X của
<Tên văn bản>". Khi căn cứ không đủ → nói rõ và đề xuất tham vấn chuyên gia.
```

**User message**:

```
CÂU HỎI: {question}

CĂN CỨ PHÁP LUẬT TOP-{K}:
{contexts}

QUAN HỆ VĂN BẢN LIÊN QUAN (từ knowledge graph):
{graph_context}

Hãy trả lời theo cấu trúc:
1) KẾT LUẬN NGẮN GỌN (2-3 câu)
2) PHÂN TÍCH CHI TIẾT — mỗi luận điểm trích dẫn "Điều X của <Tên văn bản>"
3) LƯU Ý TUÂN THỦ / RỦI RO (nếu có)
4) CẢNH BÁO: "Thông tin mang tính tham khảo, không thay thế tư vấn pháp lý chính thức."
```

### 11.4 Context construction

For each hit `i` (1-indexed) in the top-K:

```
[{i}] {ten_van_ban} — {dieu_so}: {dieu_ten}
Tóm tắt: {short}
Nội dung: {noi_dung[:1500]}
```

Items are joined with double newlines.

### 11.5 Graph context construction

For each unique document found in the top-K hits, list its `DETAILS`, `AMENDS`, `REPLACES` neighbors. Because only canonical-direction edges are stored (KG.md §2.1), both outgoing and incoming edges are read so the surfaced context covers the bi-directional view (e.g. "this decree is detailed by …" and "this decree details …"):

```python
INTERESTING_RELS = {"DETAILS", "AMENDS", "REPLACES"}

for _, nb_doc, edge in G.out_edges(doc_node, data=True):
    if edge.get("relation") in INTERESTING_RELS:
        emit(edge["relation"], G.nodes[nb_doc]["ten"])

for nb_doc, _, edge in G.in_edges(doc_node, data=True):
    if edge.get("relation") in INTERESTING_RELS:
        emit(edge["relation"] + "_BY", G.nodes[nb_doc]["ten"])  # synthesized reverse label
```

Emit one bullet per neighbor:

```
• {REL_ENUM}: {neighbor_document_title}
```

If no relevant neighbors exist, emit: `(Không có quan hệ chéo đáng chú ý)`.

### 11.6 Post-generation citation augmentation

After the LLM returns an answer, for every hit `h` in the top-K whose `dieu_so` does not appear verbatim in the answer:

```python
answer += f"\n(Căn cứ bổ sung: {h['dieu_so']} của {h['ten_van_ban']})"
```

This ensures the automated grader can extract all relevant articles via the `Điều X` regex.

---

## 12. Guardrails Specification

### 12.1 Validators

Each validator returns `(passed: bool, reason: str)` based on the generated answer and the retrieved hits.

| Validator ID | Rule |
|---|---|
| `V01_has_citation` | At least one `Điều X` substring in the answer matches a `dieu_so` value in the hits. |
| `V02_no_fabricated_law_id` | Every `\d+/\d{4}/[A-ZĐ-]+` pattern found in the answer must match a `law_id` value in the hits. |
| `V03_has_disclaimer` | The answer contains the substring `"tham khảo"` or `"không thay thế"`. If absent, the standard disclaimer is auto-appended (no retry). |
| `V04_min_length` | `len(answer) ≥ 200` characters. |

### 12.2 Retry policy

- Maximum retries: 2.
- On retry, re-invoke the LLM with the same prompt; temperature remains 0.1, `do_sample=False`.
- If all retries fail any validator other than `V03`, emit the extractive fallback (Section 12.3).

### 12.3 Extractive fallback

```
**Tổng hợp từ căn cứ pháp luật:**

- **{dieu_so_1} của {ten_van_ban_1}**: {short_1 or noi_dung_1[:200]}
- **{dieu_so_2} của {ten_van_ban_2}**: {short_2 or noi_dung_2[:200]}
- **{dieu_so_3} của {ten_van_ban_3}**: {short_3 or noi_dung_3[:200]}

*Thông tin mang tính tham khảo, không thay thế tư vấn pháp lý chính thức.*
```

The fallback uses the top-3 hits and is deterministic.

---

## 13. Submission Specification

### 13.1 Output file

- Filename: `results.json`.
- Encoding: UTF-8.
- Format: JSON array of records, one per test question.

### 13.2 Record schema

```json
{
  "id": <int, matches test set ID>,
  "question": "<str, copied from test set>",
  "answer": "<str, generated answer with embedded 'Điều X' citations and disclaimer>",
  "relevant_docs": [
    "<law_id>|<ten_van_ban>",
    ...
  ],
  "relevant_articles": [
    "<law_id>|<ten_van_ban>|<dieu_so>",
    ...
  ]
}
```

### 13.3 Field rules

- `id`: integer, must match an ID from the test set, must be unique across the file.
- `question`: string, copied verbatim from the test set.
- `answer`: string, length ≥ 200 characters, contains at least one `Điều X` reference present in `relevant_articles`.
- `relevant_docs`: list of strings, each containing exactly one `|` separator. No duplicates within the same record.
- `relevant_articles`: list of strings, each containing exactly two `|` separators. The third segment must start with `Điều`. No duplicates within the same record.

### 13.4 Packaging

```bash
zip submission.zip results.json
```

The zip file must contain `results.json` at its root with no subdirectory.

### 13.5 Validation script

A pre-submission validator must run successfully before every upload. The validator asserts:

- The JSON parses as a list.
- Every record contains the five required fields with correct types.
- All IDs are unique and match the test set.
- All `relevant_docs` items have exactly one `|`.
- All `relevant_articles` items have exactly two `|` and a third segment starting with `Điều`.
- Every `answer` contains at least one `Điều X` matching `relevant_articles`.

---

## 14. Evaluation Specification

### 14.1 Official metrics

**Retrieval metrics** (macro-averaged across queries):

```
Precision_q = |correct_articles_q| / |predicted_articles_q|
Recall_q    = |correct_articles_q| / |gold_articles_q|
F2_q        = 5 * Precision_q * Recall_q / (4 * Precision_q + Recall_q)
```

The official grader extracts predicted articles from the `answer` field using the `Điều X` pattern, then matches against the gold `relevant_articles` set under the full identifier format `<law_id>|<ten_van_ban>|Điều X` (normalized to `Điều X`).

**QA criteria** (graded weekly on the promoted submission):

| Criterion | Method |
|---|---|
| Legal Grounding | Automatic — proportion of questions with at least one correctly cited article. |
| Content Accuracy | Human evaluation. |
| Completeness | Human evaluation. |
| Practicality | Human evaluation. |
| Clarity | Human evaluation. |

### 14.2 Internal dev set

A locally maintained `dev_set.json` of 50-100 SME-relevant questions with gold `relevant_articles`. Used for offline F2 evaluation before every public-leaderboard or Private-Phase submission.

### 14.3 Local evaluation procedure

```python
def f2_macro(predictions, ground_truth):
    f2_scores = []
    for pred, gt in zip(predictions, ground_truth):
        pred_set = set(pred["relevant_articles"])
        gt_set = set(gt["relevant_articles"])
        if not pred_set and not gt_set:
            f2_scores.append(1.0)
            continue
        if not pred_set or not gt_set:
            f2_scores.append(0.0)
            continue
        tp = len(pred_set & gt_set)
        p = tp / len(pred_set)
        r = tp / len(gt_set)
        if p + r == 0:
            f2_scores.append(0.0)
            continue
        f2_scores.append(5 * p * r / (4 * p + r))
    return sum(f2_scores) / len(f2_scores)
```

---

## 15. Resource Plan

### 15.1 Notebook allocation

| Notebook | Purpose | Models loaded | Approximate runtime |
|---|---|---|---|
| `kaggle_01_preprocessing.ipynb` | Stages 1, 2, 3 | None (CPU only) | 2.5 hours |
| `kaggle_02_summarization.ipynb` | Stage 4 | Qwen2.5-3B 4-bit | 17 hours (split across sessions) |
| `kaggle_03_graph_and_index.ipynb` | Stages 5, 6 | bge-m3 | 4 hours |
| `kaggle_04_inference.ipynb` | Online inference | bge-m3 + bge-reranker-v2-m3 + Gemma-3-12B-IT QAT | 1.5 hours per 500 questions |

### 15.2 Session sequencing rules

- The inference notebook loads the embedder and reranker first, batch-encodes all queries, then unloads them before loading the generation LLM.
- Each summarization session is bounded by the JSONL cache; sessions can resume mid-corpus without redoing completed work.
- All artifacts produced by notebooks 1-3 are uploaded as Kaggle Datasets to be mounted in notebook 4.

### 15.3 Checkpoint frequency

- Parquet files: written at the end of each stage (atomic rename).
- Summary cache: appended after each generation.
- Results: persisted every 50 inference records.

---

## 16. Repository Structure

```
vietnamese-legal-rag/
├── README.md
├── SPEC.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── default.yaml
│   ├── sme_keywords.yaml
│   ├── legal_concepts.yaml
│   └── relationship_mapping.yaml
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── stage1_filter.py
│   │   ├── stage2_parse_html.py
│   │   ├── stage3_chunking.py
│   │   ├── stage4_summarize.py
│   │   ├── stage5_build_graph.py
│   │   ├── stage6_index.py
│   │   └── validators.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   ├── bm25_index.py
│   │   ├── faiss_index.py
│   │   ├── rrf.py
│   │   └── graph_expand.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt.py
│   │   ├── generator.py
│   │   └── guardrails.py
│   │
│   ├── pipeline.py
│   └── utils/
│       ├── io.py
│       ├── tokenization.py
│       └── logging.py
│
├── notebooks/
│   ├── kaggle_01_preprocessing.ipynb
│   ├── kaggle_02_summarization.ipynb
│   ├── kaggle_03_graph_and_index.ipynb
│   └── kaggle_04_inference.ipynb
│
├── scripts/
│   ├── build_all.sh
│   ├── infer.py
│   ├── validate_submission.py
│   └── make_submission_zip.sh
│
├── tests/
│   ├── test_stage2_parser.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_guardrails.py
│
├── dev_set/
│   ├── questions.json
│   ├── ground_truth.json
│   └── eval.py
│
└── artifacts/
    ├── stage1_sme_docs.parquet
    ├── stage2_articles.parquet
    ├── stage3_chunks.parquet
    ├── stage4_enriched.parquet
    ├── summary_cache.jsonl
    ├── kg.gpickle
    ├── bm25.pkl
    ├── faiss_summary.index
    ├── faiss_full.index
    └── chunk_meta.npy
```

---

## 17. Configuration Reference

`config/default.yaml`:

```yaml
seed: 42

data:
  hf_dataset: "th1nhng0/vietnamese-legal-documents"
  artifacts_dir: "./artifacts"

filter:
  drop_expired: true
  required_status_substring: "Còn hiệu lực"

chunking:
  max_tokens: 1024
  overlap_tokens: 128
  tokenizer: "google/gemma-3-12b-it"
  inject_breadcrumb: true

summarization:
  model: "Qwen/Qwen2.5-3B-Instruct"
  load_in_4bit: true
  batch_size: 4
  max_input_chars: 3000
  max_new_tokens: 256
  temperature: 0.1
  repetition_penalty: 1.05

embedding:
  model: "BAAI/bge-m3"
  dim: 1024
  use_fp16: true
  batch_summary: 32
  batch_full: 8
  max_len_summary: 256
  max_len_full: 1024

bm25:
  k1: 1.5
  b: 0.75

retrieval:
  top_bm25: 50
  top_dense_summary: 50
  top_dense_full: 50
  rrf_k: 60
  fused_top: 30
  graph_discount_doc: 0.6
  graph_discount_concept: 0.3
  expanded_top: 50
  rerank_model: "BAAI/bge-reranker-v2-m3"
  rerank_max_input_chars: 2000
  final_top_k: 5

generation:
  model: "google/gemma-3-12b-it"
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  attn_implementation: "eager"
  max_new_tokens: 900
  temperature: 0.1
  repetition_penalty: 1.05
  max_retries: 2
  context_max_chars_per_item: 1500

guardrails:
  require_citation: true
  reject_fabricated_law_id: true
  min_answer_chars: 200
  append_disclaimer_if_missing: true

submission:
  output_filename: "results.json"
  zip_filename: "submission.zip"
```

---

## 18. Glossary

| Term | Definition |
|---|---|
| SME | Small and Medium Enterprise (doanh nghiệp nhỏ và vừa) |
| Điều | Article — the atomic retrieval and citation unit in Vietnamese law |
| Khoản | Numbered clause within an article |
| Điểm | Lettered sub-clause within a Khoản |
| `law_id` | Official document number, e.g. `13/2017/QH14`; same as `so_ky_hieu` |
| `ten_van_ban` | Document title following the formula `Loại + Mã + Trích yếu` |
| `doc_uid` | Unique article identifier: `law_id|ten_van_ban|dieu_so` |
| `enriched_text` | Chunk text with summary, key points, breadcrumb, and full body |
| G-LRAG | Project codename: Graph-enhanced Legal Retrieval Augmented Generation |
| RRF | Reciprocal Rank Fusion |
| F2 | F-score with β=2 (recall weighted four times more than precision) |
| QAT | Quantization-Aware Training |
| Hit | A retrieved chunk record returned by the retriever |
