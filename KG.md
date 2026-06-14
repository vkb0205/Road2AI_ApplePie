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

3. **Doc-level là tầng quan trọng nhất cho graph reasoning.**  
   Article-level dùng để trỏ xuống điều luật; chunk-level chỉ dùng để index và rerank, không cần đưa vào KG.

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

### 3.3 `CONCEPT` node

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
- Làm graph expansion từ article sang các article cùng concept.

***

## 4. Concept design and extraction rules

1. Concept layer chỉ nên làm nhiệm vụ **gom chủ đề pháp lý chuẩn hóa** để hỗ trợ retrieval và graph expansion, không nên biến thành một ontology quá nặng.
2. Ưu tiên controlled vocabulary do team định nghĩa trước; LLM chỉ được phép map về danh sách đó, không được phép sinh concept tự do.
3. Chuẩn hoá concept theo ba thuộc tính:
   - `display_name`: tiếng Việt có dấu, ví dụ `hóa đơn điện tử`.
   - `norm_name`: lowercase, bỏ dấu, bỏ ký tự thừa để chuẩn hoá matching.
   - `node_id`: key kỹ thuật, ví dụ `CONCEPT:hoa_don_dien_tu`.
4. Giới hạn tối đa **3 concept/article**.
5. Concept chỉ được gán khi có bằng chứng rõ ràng trong article hoặc chunk text, ưu tiên concept xuất hiện ở tiêu đề điều, phần mở đầu, hoặc nội dung chính.
6. Không dùng LLM để tạo concept tự do; chỉ dùng rule/dictionary hoặc mapping vào danh sách concept có sẵn.
7. Dedup cần hai tầng:
   - dedup theo `norm_name` trước khi tạo node.
   - dedup theo `(article_id, concept_id)` trước khi tạo edge.
8. Article là nguồn chính để gán concept; chunk chỉ dùng như bằng chứng phụ hoặc fallback khi cần.
9. Bản v1 không cần ART–ART; concept layer sẽ đóng vai trò semantic bridge giữa articles.
10. Ghi lại reason/mapping cho mỗi concept edge để audit và review thủ công.

***

## 5. Edge types chốt

### 4.1 Nguyên tắc edge
Chỉ giữ các edge thật sự có ích cho retrieval:
- `DOC -> DOC`.
- `DOC -> ART`.
- `ART -> CONCEPT`.

Không tạo `CHUNK -> CHUNK`.  
Không tạo reverse edge riêng.  
Không tạo edge mơ hồ chỉ để “cho đẹp graph”.

***

## 5. Bảng relation chốt

| Relation | Nối node gì với node gì | Có sẵn trong dataset relationship gốc chưa? | Cách tạo | Vì sao giữ / dùng | Ghi chú triển khai |
|---|---|---:|---|---|---|
| `HAS_ARTICLE` | `DOC -> ART` | Không | Tạo từ Stage 2 / Stage 4 | Bắt buộc để đi từ văn bản sang điều luật; đây là lớp nền cho citation và graph expansion | Deterministic, không dùng LLM. |
| `AMENDS` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Cực quan trọng vì văn bản pháp luật Việt Nam sửa đổi liên tục; multi-hop qua văn bản sửa đổi là case rất thường gặp | Chỉ giữ canonical 1 chiều. |
| `REPLACES` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Dùng để ưu tiên bản thay thế / bản mới hơn khi có chồng chéo hiệu lực | Chỉ giữ canonical 1 chiều. |
| `DETAILS` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Rất hữu ích cho câu hỏi thủ tục, điều kiện, hồ sơ, cách thực hiện; đây là relation đáng giá nhất cho hỏi đáp | Chỉ giữ canonical 1 chiều. |
| `CITES_REF` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Hỗ trợ hỏi “căn cứ pháp lý nào”, và giúp mở rộng bối cảnh giữa các văn bản | Canonical 1 chiều. |
| `BASED_ON` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Hữu ích khi truy vấn hỏi về cơ sở ban hành, cơ sở áp dụng, hoặc nền tảng pháp lý | Canonical 1 chiều. |
| `CONSOLIDATES` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Quan trọng để ưu tiên bản hợp nhất, tránh trích sai bản cũ | Canonical 1 chiều. |
| `CORRECTS` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc | Giữ để xử lý trường hợp văn bản đính chính / sửa lỗi nội dung | Canonical 1 chiều, mức ưu tiên thấp hơn. |
| `RELATED_CONTENT` | `DOC -> DOC` | Có | Lấy từ dataset `relationships` gốc nếu label sạch | Có thể giúp tăng recall khi câu hỏi liên quan gần chủ đề nhưng không khớp hoàn toàn | Chỉ giữ nếu thực nghiệm cho thấy sạch; nếu nhiễu thì bỏ. |
| `MENTIONS` | `ART -> CONCEPT` | Không | Tạo từ article/chunk text bằng rule-based extraction; optional `enriched_text` nếu Stage 4 exists | Hỗ trợ gom chủ đề pháp lý, làm cầu nối semantic giữa nhiều điều luật | Không dùng LLM nếu rule/dictionary đủ tốt.

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

### 7.2 Từ Stage 2 / Stage 3
- `HAS_ARTICLE` từ `stage2_articles.parquet` hoặc optional `stage4_enriched.parquet`.
- `ART` nodes từ stage 2 articles.
- `short`, `key`, `enriched_text` chỉ có khi Stage 4 được chạy; không bắt buộc cho KG.

### 7.3 Từ rule-based concept extraction
- `MENTIONS` từ article text (`stage2_articles.parquet`) hoặc chunk text (`stage3_chunks.parquet`), với `stage4_enriched.parquet` là tùy chọn bổ sung.
- Cách làm:
  - tạo vocabulary khái niệm pháp lý,
  - normalize về lowercase,
  - match string hoặc synonym list,
  - gán edge `ART -> CONCEPT` nếu khái niệm xuất hiện đủ rõ.

### 7.4 Khi nào mới dùng LLM
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

### 8.4 Build concept nodes
Input:
- `stage2_articles.parquet` hoặc `stage3_chunks.parquet`.
- Optional: `stage4_enriched.parquet` if available.

Procedure:
1. Tạo dictionary concept chuẩn.
2. Duyệt text từng article hoặc chunk.
3. Match concept bằng string / synonym / regex.
4. Tạo `CONCEPT` node nếu chưa có.
5. Tạo edge `ART -> CONCEPT` bằng `MENTIONS`.

***

## 9. Retrieval use-case mapping

Graph này cần phục vụ 4 loại truy vấn chính:

1. **Single-doc, single-article**
- Ví dụ: “Công ty muốn sử dụng hóa đơn điện tử không mã thì cần điều kiện gì?”
- Cần `DOC -> ART` và `MENTIONS`.

2. **Cross-doc, same topic**
- Ví dụ: luật + nghị định + thông tư về cùng vấn đề.
- Cần `DETAILS`, `CITES_REF`, `BASED_ON`, `CONSOLIDATES`.

3. **Procedure / condition / sanction**
- Ví dụ: “bị phạt thế nào”, “nộp hồ sơ gì”, “thời hạn bao lâu”.
- Cần `DETAILS`, `AMENDS`, `REPLACES` để đi tới văn bản đang có hiệu lực hoặc chi tiết hóa.

4. **Topic bridging**
- Ví dụ: một query nhắc “bảo hiểm xã hội” nhưng văn bản có thể gọi theo cách khác.
- Cần `CONCEPT` và `MENTIONS` để tăng recall.

***

## 10. Ghi chú về chunk

Chunk là tầng phục vụ:
- BM25,
- FAISS,
- rerank,
- context construction cho LLM.

Chunk **không cần** trở thành node của KG, vì:
- chunk là cấu trúc triển khai, không phải thực thể pháp lý,
- chunk đã mang breadcrumb và summary,
- article node đã đủ để bridge từ doc sang nội dung. 

Nếu muốn trace từ chunk về graph, chỉ cần metadata:
- `chunk_id`.
- `doc_uid`.
- `doc_id`.
- `rowidx`.
- `law_id`.
- `dieu_so`. 
***

## 11. Quality gates

Trước khi chốt KG, cần kiểm tra:
- Không có duplicate node key.
- `HAS_ARTICLE` luôn nối từ `DOC` sang `ART`.
- Tất cả `ART` đều có parent `DOC`.
- Tất cả doc-doc edges đều thuộc whitelist canonical relations.
- `MENTIONS` chỉ nối `ART -> CONCEPT`.
- Không có `RELATED_LANGUAGE` trong final KG nếu không có bằng chứng thực nghiệm tốt.

***

## 12. Khuyến nghị triển khai cuối

### Giữ
- `HAS_ARTICLE`
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
2. `DETAILS`
3. `AMENDS`
4. `REPLACES`
5. `CITES_REF`
6. `BASED_ON`
7. `CONSOLIDATES`
8. `MENTIONS` 