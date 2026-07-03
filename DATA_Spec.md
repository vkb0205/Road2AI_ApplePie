# DATA_Spec.md — Stage 1 SME Legal Documents

| Field | Value |
|---|---|
| Artifact | `data/untracked_data/stage1_sme_docs.parquet` |
| Pipeline stage | Stage 1 — SME-scope legal document metadata |
| Producer | `src/data/stage1_filter.py` |
| Format | Apache Parquet |
| Rows | 114,861 |
| Columns | 19 |
| Primary row grain | 1 legal document metadata record |
| Primary key | `id` |
| Retrieval identity | `law_id` + `ten_van_ban` |
| Downstream users | Stage 2 article parsing, Stage 5 KG document nodes, DB upload, retrieval citation metadata |

---

## 1. Purpose

`stage1_sme_docs.parquet` is the first curated corpus artifact for the Vietnamese SME legal RAG system. It stores document-level metadata after filtering the raw Vietnamese legal documents dataset to documents usable by the downstream legal retrieval pipeline.

This file does **not** contain article text/body content. It contains metadata and canonical identifiers used to join, cite, parse, graph, and filter documents.

---

## 2. Source and derivation

### 2.1 Input sources

Producer script defaults:

- Metadata: `hf://datasets/th1nhng0/vietnamese-legal-documents/data/metadata.parquet`
- Content IDs: `hf://datasets/th1nhng0/vietnamese-legal-documents/data/content.parquet`
- Config: `config/default.yaml`

### 2.2 Stage-1 transform

Implemented in `src/data/stage1_filter.py`:

1. Load raw metadata.
2. Load content IDs.
3. Keep records whose `ngay_het_hieu_luc` is null/blank-like.
4. Create:
   - `law_id = so_ky_hieu`
   - `ten_van_ban = <loai_van_ban> <so_ky_hieu> <cleaned title>`
5. Drop duplicate pairs by `law_id, ten_van_ban`.
6. Assert:
   - output row count > 0
   - `id` unique
   - `(law_id, ten_van_ban)` unique
7. Write Parquet without index.

---

## 3. Dataset profile

### 3.1 Shape

| Metric | Value |
|---|---:|
| Rows | 114,861 |
| Columns | 19 |
| Unique `id` | 114,861 |
| Unique `ten_van_ban` | 114,861 |
| Unique `law_id` | 54,287 |
| Duplicate `law_id` count | 60,574 |
| Duplicate `(law_id, ten_van_ban)` | 0 |

`law_id`/`so_ky_hieu` is **not globally unique**. Use `(law_id, ten_van_ban)` or `id` for stable identity.

### 3.2 Common document types

| `loai_van_ban` | Count |
|---|---:|
| Quyết định | 65,511 |
| Nghị quyết | 19,490 |
| Thông tư | 14,878 |
| Chỉ thị | 5,982 |
| Nghị định | 4,136 |
| Thông tư liên tịch | 2,808 |
| Sắc lệnh | 934 |
| Luật | 479 |

Total distinct document types: 19.

### 3.3 Effect status distribution

| `tinh_trang_hieu_luc` | Count |
|---|---:|
| Còn hiệu lực | 63,143 |
| Hết hiệu lực toàn bộ | 44,669 |
| Hết hiệu lực một phần | 4,432 |
| Blank string | 2,033 |
| Không còn phù hợp | 250 |
| Ngưng hiệu lực | 210 |
| Chưa có hiệu lực | 84 |
| Chưa xác định | 38 |

Distinct statuses: 9.

Note: despite the Stage-1 `ngay_het_hieu_luc` null filter, `tinh_trang_hieu_luc` may still contain expired-like labels. Downstream retrieval MUST check `tinh_trang_hieu_luc` before citation/use.

---

## 4. Column specification

| Column | Type | Nullable | Non-null | Unique | Description | Validation / notes |
|---|---|---:|---:|---:|---|---|
| `id` | int64 | No | 114,861 | 114,861 | Raw document ID. | Primary key. Must be unique. Used to join raw content and downstream docs. |
| `title` | string | No | 114,861 | 109,106 | Raw/metadata title or abstract title. | Can repeat across many documents. |
| `so_ky_hieu` | string | No | 114,861 | 54,287 | Official document number/code. | Same value copied into `law_id`. Not unique. |
| `ngay_ban_hanh` | string | No | 114,861 | 13,115 | Issuance date. | Expected mostly `DD/MM/YYYY`; stored as string. |
| `loai_van_ban` | string | No | 114,861 | 19 | Document type. | Must be in configured valid type set. |
| `ngay_co_hieu_luc` | string | No | 114,861 | 14,561 | Effective date. | String; may be placeholder `...`. |
| `ngay_het_hieu_luc` | object/null | Yes | 0 | 0 | Expiry date. | Entire column null in inspected artifact. |
| `nguon_thu_thap` | string | Yes | 61,493 | 11,426 | Collection/source note. | Free text; many nulls. |
| `ngay_dang_cong_bao` | string | No | 114,861 | 5,579 | Gazette publication date. | Mostly placeholder `...` (86,062 rows). |
| `nganh` | string | Yes | 66,481 | 675 | Sector/ministry domain. | Free-text categorical; casing may vary. |
| `linh_vuc` | string | Yes | 32,745 | 1,442 | Legal field/subdomain. | Sparse, free-text categorical. |
| `co_quan_ban_hanh` | string | Yes | 113,517 | 535 | Issuing authority. | Important for authority ranking/KG. |
| `chuc_danh` | string | Yes | 109,964 | 181 | Signer title. | Free-text categorical. |
| `nguoi_ky` | string | Yes | 112,611 | 2,811 | Signer name. | Free-text person/entity field. |
| `pham_vi` | string | Yes | 101,611 | 402 | Geographic/application scope. | Includes national/provincial/local values; casing varies. |
| `thong_tin_ap_dung` | float64/null | Yes | 0 | 0 | Application info. | Entire column null in inspected artifact. |
| `tinh_trang_hieu_luc` | string | No | 114,861 | 9 | Legal effect status. | Must be used for validity filtering. |
| `law_id` | string | No | 114,861 | 54,287 | Canonical citation code. | Derived from `so_ky_hieu`; not unique alone. |
| `ten_van_ban` | string | No | 114,861 | 114,861 | Canonical document title for citations. | Derived as `<loai_van_ban> <so_ky_hieu> <cleaned title>`. |

---

## 5. Nullability profile

| Column | Null count | Null % |
|---|---:|---:|
| `ngay_het_hieu_luc` | 114,861 | 100.00% |
| `thong_tin_ap_dung` | 114,861 | 100.00% |
| `linh_vuc` | 82,116 | 71.49% |
| `nguon_thu_thap` | 53,368 | 46.46% |
| `nganh` | 48,380 | 42.12% |
| `pham_vi` | 13,250 | 11.54% |
| `chuc_danh` | 4,897 | 4.26% |
| `nguoi_ky` | 2,250 | 1.96% |
| `co_quan_ban_hanh` | 1,344 | 1.17% |
| All other columns | 0 | 0.00% |

---

## 6. Identity and joins

### 6.1 Preferred keys

| Use case | Key |
|---|---|
| Unique document row | `id` |
| Citation/export document identity | `law_id|ten_van_ban` |
| Article identity downstream | `law_id|ten_van_ban|dieu_so` |
| Stage 2 raw content join | `id` / `doc_id` |
| KG DOC node identity | Usually `DOC:{law_id}|{ten_van_ban}` or equivalent implementation key |

### 6.2 Key caveats

- `law_id` duplicates are expected.
- `so_ky_hieu` duplicates are expected.
- `ten_van_ban` is unique in this artifact.
- `(law_id, ten_van_ban)` is unique by producer assertion.
- Do not use title-only joins.

---

## 7. Data quality rules

### 7.1 Required hard checks

- `id` MUST be non-null and unique.
- `law_id` MUST be non-null and non-empty.
- `ten_van_ban` MUST be non-null, non-empty, and unique with `law_id`.
- `loai_van_ban` MUST be non-null and recognized.
- `ngay_ban_hanh` SHOULD parse as date or be flagged.
- `ngay_co_hieu_luc` SHOULD parse as date unless value is placeholder `...`.
- `tinh_trang_hieu_luc` MUST be normalized before effect-status filtering.

### 7.2 Soft quality flags

Flag but do not necessarily drop:

- `ngay_co_hieu_luc == "..."`
- `ngay_dang_cong_bao == "..."`
- Null `co_quan_ban_hanh`
- Null `nganh` / `linh_vuc`
- Expired-like `tinh_trang_hieu_luc`
- Blank-string `tinh_trang_hieu_luc`
- Inconsistent casing in `pham_vi`, `nganh`, `linh_vuc`

---

## 8. Downstream contract

### 8.1 Stage 2 parser

Consumes Stage 1 rows to locate and parse body HTML/text from content dataset.

Required fields:

- `id`
- `law_id`
- `ten_van_ban`
- `loai_van_ban`
- `ngay_ban_hanh`
- `title`

Outputs article-level records with `doc_id`, `doc_uid`, `dieu_so`, etc.

### 8.2 Stage 5 knowledge graph

Required Stage-1 columns per implementation:

- `id`
- `law_id`
- `ten_van_ban`
- `loai_van_ban`
- `nganh`
- `linh_vuc`
- `ngay_ban_hanh`
- `tinh_trang_hieu_luc`
- `ngay_co_hieu_luc`
- `ngay_het_hieu_luc`

Used to create DOC nodes and propagate legal effect metadata to article/chunk nodes.

### 8.3 Retrieval/generation

Citation outputs must use:

- Relevant docs: `law_id|ten_van_ban`
- Relevant articles: `law_id|ten_van_ban|Điều X`

Retrieval MUST avoid grounding final answers in invalid/expired legal texts unless explicitly needed for historical comparison.

---

## 9. Recommended normalized schema

For relational DB upload / analytics, normalize to:

```sql
CREATE TABLE documents (
  id BIGINT PRIMARY KEY,
  law_id TEXT NOT NULL,
  ten_van_ban TEXT NOT NULL,
  loai_van_ban TEXT NOT NULL,
  title TEXT NOT NULL,
  so_ky_hieu TEXT NOT NULL,
  ngay_ban_hanh DATE NULL,
  ngay_co_hieu_luc DATE NULL,
  ngay_het_hieu_luc DATE NULL,
  ngay_dang_cong_bao DATE NULL,
  nguon_thu_thap TEXT NULL,
  nganh TEXT NULL,
  linh_vuc TEXT NULL,
  co_quan_ban_hanh TEXT NULL,
  chuc_danh TEXT NULL,
  nguoi_ky TEXT NULL,
  pham_vi TEXT NULL,
  thong_tin_ap_dung TEXT NULL,
  tinh_trang_hieu_luc TEXT NOT NULL,
  UNIQUE (law_id, ten_van_ban)
);
```

Date parser should map `...`, empty strings, `nan`, `nat`, `none`, `null` to SQL `NULL`.

---

## 10. Example row

```json
{
  "id": 72,
  "title": "Sắc lệnh quy định liên hệ giữa UBKCHC và các cơ quan chuyên môn",
  "so_ky_hieu": "103/SL",
  "ngay_ban_hanh": "05/06/1950",
  "loai_van_ban": "Sắc lệnh",
  "ngay_co_hieu_luc": "20/06/1950",
  "ngay_het_hieu_luc": null,
  "nguon_thu_thap": "Công báo số 7/1950;",
  "ngay_dang_cong_bao": "...",
  "nganh": null,
  "linh_vuc": null,
  "co_quan_ban_hanh": "Chủ tịch nước",
  "chuc_danh": "Chủ tịch nước",
  "nguoi_ky": "Hồ Chí Minh",
  "pham_vi": null,
  "thong_tin_ap_dung": null,
  "tinh_trang_hieu_luc": "Hết hiệu lực toàn bộ",
  "law_id": "103/SL",
  "ten_van_ban": "Sắc lệnh 103/SL Sắc lệnh quy định liên hệ giữa UBKCHC và các cơ quan chuyên môn"
}
```

---

## 11. Known limitations

- The artifact is metadata-only; no article/body text.
- `ngay_het_hieu_luc` and `thong_tin_ap_dung` are fully null in the inspected file.
- Date fields are strings and include placeholders.
- `law_id` is not unique.
- `tinh_trang_hieu_luc` may conflict with null expiry date.
- Categorical fields are not fully normalized for casing/variants.
- SME scope is approximated through configured document-type and expiry filters; it is not a semantic SME-only classifier.

---

## 12. Acceptance checklist

Before using this artifact downstream:

- [ ] File exists at `data/untracked_data/stage1_sme_docs.parquet` or configured path.
- [ ] Row count within KG bound: 3,000–120,000.
- [ ] `id` unique and non-null.
- [ ] `(law_id, ten_van_ban)` unique and non-null.
- [ ] Required Stage-2 fields present.
- [ ] Required Stage-5 fields present.
- [ ] Date placeholders handled.
- [ ] Effect status normalized for retrieval filtering.
- [ ] Null-heavy columns tolerated by consumers.
