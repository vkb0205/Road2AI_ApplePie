# KG Build Plan

## 1. Mục tiêu

Knowledge Graph này được xây để hỗ trợ pipeline truy hồi cho SME Legal QA, ưu tiên các truy vấn dạng:
- hỏi một điều luật cụ thể,
- hỏi thủ tục / điều kiện / chế tài,
- hỏi multi-hop giữa nhiều văn bản,
- hỏi cross-document reasoning giữa luật, nghị định, thông tư.
***(những ý này đúc kết dựa trên nhận xét bộ public test set)***

Mục tiêu của KG không phải là mô hình hóa toàn bộ pháp luật Việt Nam, mà là tạo một lớp **liên kết pháp lý đủ sạch, đủ gọn, đủ hữu dụng** cho retrieval và graph expansion.

***

## 2. Nguyên tắc thiết kế

1. **Không lưu relation chiều ngược như một relation riêng.**  
   Mỗi relation chỉ giữ một hướng canonical; khi cần traversal ngược thì xử lý bằng code ở tầng truy vấn, không nhân đôi cạnh.

2. **Ưu tiên relation có tác dụng trực tiếp cho competition.**  
   Các relation nên phục vụ đúng kiểu câu hỏi trong public test: điều kiện, thủ tục, trách nhiệm, xử phạt, hồ sơ, hiệu lực, ưu đãi, và cross-doc multi-hop.

3. **Doc-level và article-level vẫn là backbone pháp lý.**  
   `DOC -> ART` vẫn là trục chuẩn để đi từ văn bản sang điều luật; chunk được đưa vào KG như tầng trung gian để neo evidence, nhưng không trở thành node pháp lý trung tâm.

4. **Concept là tầng ngữ nghĩa phụ trợ.**  
   Concept giúp gom các chủ đề pháp lý lặp lại giữa nhiều văn bản, nhưng không thay thế relation pháp lý gốc.

5. **Không dùng LLM để bịa relation pháp lý giữa các văn bản.**  
   Relation pháp lý chính phải đến từ dataset gốc hoặc rule-based extraction. LLM chỉ là phương án cuối cho concept normalization hoặc label cleanup nếu thật sự cần.

***

## 3. Node types

### 3.1 `DOC` node

Đây là node đại diện cho một văn bản pháp luật hoàn chỉnh.

**Key format**
```text
DOC:{doc_id}
```

**Attributes**
- `type = "Document"`.
- `doc_id`.
- `law_id` = số hiệu văn bản.
- `ten` = tên văn bản chuẩn hóa.
- `loai` = loại văn bản.
- `nganh`.
- `linh_vuc`.
- `ngay_ban_hanh`.
- `tinh_trang_hieu_luc`.
- `ngay_co_hieu_luc`.
- `ngay_het_hieu_luc` nếu có.

**Vai trò**
- Đây là node trung tâm của KG.
- Mọi relation pháp lý gốc đều đặt ở tầng document.
- Retrieval graph expansion sẽ chủ yếu đi qua node này.

***

### 3.2 `ART` node

Node này đại diện cho một **Điều** cụ thể trong một văn bản.

**Key format**
```text
ART:{doc_uid}
```

Trong đó `doc_uid = {law_id}|{ten_van_ban}|{dieu_so}`.

**Attributes**
- `type = "Article"`.
- `doc_uid`.
- `doc_id`.
- `law_id`.
- `ten_van_ban`.
- `dieu_so`.
- `dieu_ten`.
- `phan`.
- `chuong`.
- `muc`.
- `short` (optional; generated only if Stage 4 summary injection is available).
- `enriched_text` (optional; available only if Stage 4 enriched chunks are produced).

**Vai trò**
- Là đơn vị để answer cite bằng `Điều X`.
- Là bridge từ document xuống nội dung thực thi.
- Giúp retrieval đúng article thay vì chỉ document-level.

***

### 3.3 `CHUNK` node

Node này đại diện cho một **chunk triển khai** được sinh từ Stage 3 để bám sát ngữ cảnh cục bộ phục vụ concept extraction và retrieval tracing.

**Key format**
```text
CHUNK:{chunk_id}
```

**Attributes**
- `type = "Chunk"`.
- `chunk_id`.
- `doc_uid`.
- `doc_id`.
- `rowidx`.
- `part_idx`.
- `breadcrumb`.
- `law_id` (khuyến nghị giữ thêm để tiện join / debug).
- `dieu_so` (khuyến nghị giữ thêm để tiện audit từ chunk về article).

**Vai trò**
- Là node trung gian nối `ART -> CHUNK -> CONCEPT`.
- Giữ trace rõ ràng từ retrieval unit về article gốc.
- Cho phép concept extraction bám vào evidence nhỏ, sạch hơn article full text.

***

### 3.4 `CONCEPT` node

Node này đại diện cho **khái niệm pháp lý hoặc chủ đề pháp lý** được trích từ nội dung văn bản.

**Key format**
```text
CONCEPT:{name_lower}
```

**Attributes**
- `type = "Concept"`.
- `name`: tên concept tiếng Việt có dấu, ví dụ `hóa đơn điện tử`.
- `name_lower`: bản chuẩn hóa không dấu, lowercase, dùng cho matching và dedup.

**Ví dụ**
- `CONCEPT:hoa_don_dien_tu`.
- `CONCEPT:bao_hiem_xa_hoi`.
- `CONCEPT:doanh_nghiep_nho_va_vua`.
- `CONCEPT:quyen_tac_gia`.
- `CONCEPT:thue_gtgt`.
- `CONCEPT:hop_dong_lao_dong`.

**Vai trò**
- Gom các điều luật thuộc cùng chủ đề.
- Hỗ trợ câu hỏi có wording khác nhau nhưng cùng bản chất pháp lý.
- Làm graph expansion từ chunk sang các chunk/article cùng concept.

***

## 4. Concept design and extraction rules

1. Concept layer chỉ nên làm nhiệm vụ **gom chủ đề pháp lý chuẩn hóa** để hỗ trợ retrieval và graph expansion, không nên biến thành một ontology quá nặng.
2. Ưu tiên controlled vocabulary do team định nghĩa trước; LLM chỉ được phép map về danh sách đó, không được phép sinh concept tự do.
3. Chuẩn hoá concept theo ba thuộc tính:
   - `display_name`: tiếng Việt có dấu, ví dụ `hóa đơn điện tử`.
   - `norm_name`: lowercase, bỏ dấu, bỏ ký tự thừa để chuẩn hoá matching.
   - `node_id`: key kỹ thuật, ví dụ `CONCEPT:hoa_don_dien_tu`.
4. Giới hạn tối đa **3 concept/chunk** trong luồng chính; khi aggregate ngược lên article thì có thể giữ union của các concept từ các chunk con.
5. Concept chỉ được gán khi có bằng chứng rõ ràng trong chunk text; article text chỉ dùng để tổng hợp ngược, sanity-check, hoặc fallback kiểm tra.
6. Không dùng LLM để tạo concept tự do; chỉ dùng rule/dictionary hoặc mapping vào danh sách concept có sẵn.
7. Dedup cần hai tầng:
   - dedup theo `norm_name` trước khi tạo node.
   - dedup theo `(chunk_id, concept_id)` trước khi tạo edge.
8. Chunk là nguồn chính để gán concept; article chỉ là lớp tổng hợp ngược hoặc kiểm tra lại.
9. Bản v1 không cần `ART–ART`; concept layer sẽ đóng vai trò semantic bridge giữa các article thông qua các chunk con.
10. Ghi lại reason/mapping cho mỗi concept edge để audit và review thủ công.

***

## 5. Edge types chốt

### 4.1 Nguyên tắc edge
Chỉ giữ các edge thật sự có ích cho retrieval:
- `DOC -> DOC`.
- `DOC -> ART`.
- `ART -> CHUNK`.
- `CHUNK -> CONCEPT`.

Không tạo `CHUNK -> CHUNK`.  
Không tạo reverse edge riêng.  
Không tạo edge mơ hồ chỉ để “cho đẹp graph”.

***

## 5. Bảng relation chốt

| Relation | Nối node gì với node gì | Có sẵn trong dataset relationship gốc chưa? | Cách tạo | Vì sao giữ / dùng | Ghi chú triển khai |
|---|---|---:|---|---|---|
| `HAS_ARTICLE` | `DOC -> ART` | Không | Tạo từ Stage 2 | Bắt buộc để đi từ văn bản sang điều luật; đây là lớp nền cho citation và graph expansion | Deterministic, không dùng LLM. |
| `HAS_CHUNK` | `ART -> CHUNK` | Không | Tạo từ Stage 3 | Neo chunk về article gốc; bắt buộc nếu chunk là retrieval unit và là nguồn concept extraction chính | Deterministic, không dùng LLM. |
| `AMENDS` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Cực quan trọng vì văn bản pháp luật Việt Nam sửa đổi liên tục; multi-hop qua văn bản sửa đổi là case rất thường gặp | Chỉ giữ canonical 1 chiều. |
| `REPLACES` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Dùng để ưu tiên bản thay thế / bản mới hơn khi có chồng chéo hiệu lực | Chỉ giữ canonical 1 chiều. |
| `DETAILS` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Rất hữu ích cho câu hỏi thủ tục, điều kiện, hồ sơ, cách thực hiện; đây là relation đáng giá nhất cho hỏi đáp | Chỉ giữ canonical 1 chiều. |
| `CITES_REF` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Hỗ trợ hỏi “căn cứ pháp lý nào”, và giúp mở rộng bối cảnh giữa các văn bản | Canonical 1 chiều. |
| `BASED_ON` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Hữu ích khi truy vấn hỏi về cơ sở ban hành, cơ sở áp dụng, hoặc nền tảng pháp lý | Canonical 1 chiều. |
| `CONSOLIDATES` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Quan trọng để ưu tiên bản hợp nhất, tránh trích sai bản cũ | Canonical 1 chiều. |
| `CORRECTS` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Giữ để xử lý trường hợp văn bản đính chính / sửa lỗi nội dung | Canonical 1 chiều, mức ưu tiên thấp hơn. |
| `RELATED_CONTENT` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc nếu label sạch | Có thể giúp tăng recall khi câu hỏi liên quan gần chủ đề nhưng không khớp hoàn toàn | Chỉ giữ nếu thực nghiệm cho thấy sạch; nếu nhiễu thì bỏ. |
| `MENTIONS` | `CHUNK -> CONCEPT` | Không | Tạo từ chunk text bằng rule-based extraction; optional `enriched_text` nếu Stage 4 exists nhưng chunk text vẫn là nguồn chính | Hỗ trợ gom chủ đề pháp lý, làm cầu nối semantic giữa nhiều điều luật thông qua các chunk con | Không dùng LLM nếu rule/dictionary đủ tốt. |

***

## 6. Relation nào không giữ

Không nên đưa vào KG:
- `RELATED_LANGUAGE`.
- Reverse relation riêng kiểu `AMENDED_BY`, `REPLACED_BY`, `DETAILED_BY`, `CITED_BY_REF`, `BASIS_OF`, `CONSOLIDATED_BY`, `CORRECTED_BY`.
- Chunk-level relation như `CHUNK -> CHUNK`.
- Relation sinh ngẫu nhiên từ LLM giữa các văn bản.

Lý do là các relation này либо quá ít giá trị cho retrieval, либо chỉ là chiều ngược kỹ thuật, либо dễ làm graph phình và nhiễu.

***

## 7. Nguồn tạo từng relation

### 7.1 Từ dataset gốc
Các relation `DOC -> DOC` nên lấy trực tiếp từ `relationships` dataset gốc:
- `AMENDS`
- `REPLACES`
- `DETAILS`
- `CITES_REF`
- `BASED_ON`
- `CONSOLIDATES`
- `CORRECTS`
- `RELATED_CONTENT`.

### 7.2 Từ Stage 2
- `HAS_ARTICLE` từ `stage2_articles.parquet`.
- `ART` nodes từ Stage 2 articles.

### 7.3 Từ Stage 3
- `HAS_CHUNK` từ `stage3_chunks.parquet`, nối `ART -> CHUNK`.
- `CHUNK` nodes từ Stage 3 chunks với metadata tối thiểu: `chunk_id`, `doc_uid`, `doc_id`, `rowidx`, `part_idx`, `breadcrumb`.
- **Luồng concept extraction chính: chunk text → concept** (Section 7.4 bên dưới).

### 7.4 Từ rule-based concept extraction trên chunk
- **`MENTIONS` từ chunk text (`stage3_chunks.parquet`) là nguồn chính ở luồng mới.**
- Optional `enriched_text` nếu Stage 4 có sẵn, nhưng **chunk text vẫn là primary source**.
- `stage2_articles.parquet` chỉ dùng để aggregate ngược / kiểm tra lại.
- Cách làm:
  - Tạo vocabulary khái niệm pháp lý từ controlled list.
  - Normalize về lowercase, bỏ dấu, loại bỏ ký tự thừa.
  - Match string (substring) hoặc synonym list trên chunk text.
  - Gán edge `CHUNK -> CONCEPT` nếu khái niệm xuất hiện rõ trong chunk.

### 7.5 Khi nào mới dùng LLM
Chỉ dùng LLM nếu:
- muốn mở rộng concept vocabulary,
- muốn chuẩn hóa synonym phức tạp,
- hoặc cần review thủ công cho concept khó.

**Không nên** dùng LLM để:
- sinh relation doc–doc,
- đoán quan hệ sửa đổi / thay thế,
- tạo edge từ ngữ cảnh mơ hồ.

***

## 8. Cách build KG

### 8.1 Build Document nodes
Input:
- `stage1_sme_docs.parquet`.

Procedure:
1. Duyệt từng document.
2. Tạo `DOC:{doc_id}`.
3. Gán metadata chuẩn hóa.
4. Lưu mapping `doc_id -> DOC node`.

***

### 8.2 Build Article nodes
Input:
- `stage2_articles.parquet`.
- Optional: `stage4_enriched.parquet` nếu summary injection đã được chạy.

Procedure:
1. Duyệt từng article.
2. Tạo `ART:{doc_uid}`.
3. Gán `doc_id`, `law_id`, `ten_van_ban`, `dieu_so`, `dieu_ten`.
4. Tạo cạnh `DOC -> ART` với relation `HAS_ARTICLE`.

***

### 8.3 Build doc-doc edges
Input:
- `relationships` dataset gốc.

Procedure:
1. Lọc ra các doc nằm trong `stage1_sme_docs`.
2. Map label gốc sang canonical relation.
3. Chỉ giữ labels thuộc whitelist.
4. Bỏ các label không nằm trong whitelist.
5. Không tạo reverse relation riêng.

***

### 8.4 Build chunk nodes
Input:
- `stage3_chunks.parquet`.

Procedure:
1. Duyệt từng chunk.
2. Tạo `CHUNK:{chunk_id}`.
3. Gán metadata tối thiểu: `doc_uid`, `doc_id`, `rowidx`, `part_idx`, `breadcrumb`.
4. Tạo cạnh `ART -> CHUNK` với relation `HAS_CHUNK`.

***

### 8.5 Build concept nodes
Input:
- `stage3_chunks.parquet`.
- Optional: `stage4_enriched.parquet` if available.

Procedure:
1. Tạo dictionary concept chuẩn.
2. Duyệt text từng chunk.
3. Match concept bằng string / synonym / regex.
4. Tạo `CONCEPT` node nếu chưa có.
5. Tạo edge `CHUNK -> CONCEPT` bằng `MENTIONS`.

***

## 9. Retrieval use-case mapping

Graph này cần phục vụ 4 loại truy vấn chính:

1. **Single-doc, single-article**
- Ví dụ: “Công ty muốn sử dụng hóa đơn điện tử không mã thì cần điều kiện gì?”
- Cần `DOC -> ART -> CHUNK` và `MENTIONS`.

2. **Cross-doc, same topic**
- Ví dụ: luật + nghị định + thông tư về cùng vấn đề.
- Cần `DETAILS`, `CITES_REF`, `BASED_ON`, `CONSOLIDATES`.

3. **Procedure / condition / sanction**
- Ví dụ: “bị phạt thế nào”, “nộp hồ sơ gì”, “thời hạn bao lâu”.
- Cần `DETAILS`, `AMENDS`, `REPLACES` để đi tới văn bản đang có hiệu lực hoặc chi tiết hóa.

4. **Topic bridging**
- Ví dụ: một query nhắc “bảo hiểm xã hội” nhưng văn bản có thể gọi theo cách khác.
- Cần `CHUNK -> CONCEPT` để tăng recall, sau đó aggregate ngược về article/doc phục vụ cite.

***

## 10. Ghi chú về chunk

Chunk là tầng phục vụ:
- BM25 indexing,
- FAISS dense search,
- rerank via cross-encoder,
- context construction cho LLM,
- **concept extraction (luồng chính của KG)**.

Chunk **được đưa vào KG như một tầng trung gian** giữa article và concept:
- **Là retrieval unit chính** (thay vì article hoặc doc).
- **Là source chính cho concept extraction** thay vì article text.
- Nhưng **vẫn không trở thành node pháp lý trung tâm**, vì:
  - Chunk là cấu trúc triển khai / evidence layer, không phải đơn vị cite cuối cùng.
  - Citation và legal grounding vẫn quay về article/doc.
  - Chunk giúp neo tín hiệu semantic ở độ hạt nhỏ, nhưng article là unit để trích dẫn trong answer.

Metadata tối thiểu cần giữ trên node chunk:
- `chunk_id`.
- `doc_uid`.
- `doc_id`.
- `rowidx`.
- `part_idx`.
- `breadcrumb`.

Có thể giữ thêm để tiện trace / audit:
- `law_id`.
- `dieu_so`.

***

## 11. Quality gates

Trước khi chốt KG, cần kiểm tra:
- Không có duplicate node key.
- `HAS_ARTICLE` luôn nối từ `DOC` sang `ART`.
- `HAS_CHUNK` luôn nối từ `ART` sang `CHUNK`.
- Tất cả `ART` đều có parent `DOC`.
- Mọi `CHUNK` đều có parent `ART`.
- Tất cả doc-doc edges đều thuộc whitelist canonical relations.
- `MENTIONS` chỉ nối `CHUNK -> CONCEPT`.
- `CHUNK -> CONCEPT` là edge duy nhất cho concept extraction ở luồng chính.
- Không có `ART -> CONCEPT` nếu pipeline mới đã chuyển hẳn sang chunk-first.
- Không có `RELATED_LANGUAGE` trong final KG nếu không có bằng chứng thực nghiệm tốt.

***

## 12. Khuyến nghị triển khai cuối

### Giữ
- `HAS_ARTICLE`
- `HAS_CHUNK`
- `AMENDS`
- `REPLACES`
- `DETAILS`
- `CITES_REF`
- `BASED_ON`
- `CONSOLIDATES`
- `CORRECTS`
- `RELATED_CONTENT` nếu sạch
- `MENTIONS`

### Bỏ
- reverse relation riêng
- `RELATED_LANGUAGE`
- chunk-level edges
- LLM-generated doc-doc edges 

### Ưu tiên cao nhất
Nếu phải chọn ít relation nhất mà vẫn hiệu quả, mình đề xuất theo thứ tự:
1. `HAS_ARTICLE`
2. `HAS_CHUNK`
3. `DETAILS`
4. `AMENDS`
5. `REPLACES`
6. `CITES_REF`
7. `BASED_ON`
8. `CONSOLIDATES`
9. `MENTIONS`