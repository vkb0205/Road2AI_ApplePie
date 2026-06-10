# Chi tiết đề thi: Vietnamese Legal Information Retrieval & Question Answering

## Bối cảnh bài toán
Doanh nghiệp SME tại Việt Nam thường gặp khó khăn trong việc tra cứu và áp dụng các quy định pháp lý liên quan đến Luật Doanh nghiệp, thuế, lao động, hợp đồng... Trợ lý pháp lý AI cho doanh nghiệp được xây dựng nhằm hỗ trợ chủ doanh nghiệp, kế toán, nhân sự tra cứu nhanh các điều luật, hỏi đáp tình huống pháp lý cụ thể và nhận tư vấn sơ bộ dựa trên hệ thống văn bản pháp luật chính thống.

Trong bối cảnh trí tuệ nhân tạo phát triển mạnh mẽ, đặc biệt với sự xuất hiện của các mô hình ngôn ngữ lớn như ChatGPT, DeepSeek và Qwen, nhu cầu xây dựng các hệ thống AI hỗ trợ xử lý văn bản pháp luật ngày càng trở nên quan trọng. Tuy nhiên, so với các ngôn ngữ như tiếng Anh, tiếng Nhật hay tiếng Trung, nguồn tài nguyên và các nghiên cứu về Vietnamese Legal NLP vẫn còn hạn chế.

Nhằm thúc đẩy nghiên cứu và phát triển trong lĩnh vực này, chúng tôi tổ chức cuộc thi về Truy hồi và Hỏi đáp Văn bản Pháp luật Tiếng Việt (Vietnamese Legal Information Retrieval & Question Answering). Cuộc thi hướng tới việc xây dựng các hệ thống AI có khả năng tìm kiếm điều luật liên quan và tự động trả lời các câu hỏi pháp lý dựa trên căn cứ pháp luật.

## Truy hồi thông tin (Information Retrieval - IR)
Truy hồi thông tin (Information Retrieval - IR) là một nhiệm vụ cốt lõi trong NLP, liên quan đến việc xác định thông tin nào phù hợp nhất với một truy vấn cho trước. Trong lĩnh vực pháp luật, nhiệm vụ Truy hồi Văn bản Pháp luật tập trung vào việc xác định điều luật nào có liên quan đến một câu hỏi pháp lý cụ thể. 

Nhiệm vụ có thể được hình thức hóa như sau: Cho một tập câu hỏi Q = {q1, q2, ..., qn} và một kho điều luật A = {a1, a2, ..., an}, nhiệm vụ yêu cầu xác định một tập con A′ ⊂ A trong đó mỗi điều luật ai ∈ A′ được coi là "liên quan" đến câu hỏi tương ứng q. Chúng tôi gọi một điều luật là "Liên quan" đến một truy vấn nếu câu truy vấn có thể được trả lời Có/Không, được suy ra từ ý nghĩa của điều luật đó.

## Hỏi đáp pháp luật (Legal Question Answering - QA)
Dựa trên các điều luật đã được truy hồi, hệ thống cần sinh ra câu trả lời cho câu hỏi pháp lý tương ứng. Mục tiêu của nhiệm vụ là xây dựng các hệ thống AI có khả năng không chỉ tìm đúng căn cứ pháp luật mà còn hiểu và suy luận nội dung pháp lý để hỗ trợ trả lời tự động cho người dùng.

## Mục tiêu cuộc thi
Các đội thi cần xây dựng hệ thống AI có khả năng:

1. **Tra cứu pháp lý chính xác**
   - Tra cứu điều khoản trong Luật Doanh nghiệp và các văn bản liên quan đến SME.
   - Tìm kiếm và truy xuất thông tin pháp luật chính xác từ kho dữ liệu được cung cấp.
   - Ưu tiên khả năng retrieval và grounding chính xác.
2. **Hỏi đáp pháp lý bằng tiếng Việt**
   - Hiểu ngôn ngữ tự nhiên tiếng Việt.
   - Hỏi đáp các tình huống pháp lý thường gặp.
3. **Dẫn nguồn điều luật**
   - Trích dẫn điều/khoản/văn bản liên quan.
   - Hiển thị rõ nguồn tham chiếu để đảm bảo khả năng kiểm chứng thông tin.
   - Hạn chế việc trả lời không có căn cứ pháp lý.
4. **Tư vấn sơ bộ & cảnh báo giới hạn**
   - Đưa ra hướng dẫn pháp lý sơ bộ cho người dùng.
   - Nhắc nhở các rủi ro tuân thủ trong các tình huống phổ biến.
   - Hiển thị cảnh báo giới hạn AI.
5. **Kiểm soát nội dung sai lệch**
   - Hạn chế việc AI sinh ra thông tin sai lệch.
   - Tránh bịa điều luật hoặc nguồn tham chiếu không tồn tại.
   - Tăng độ tin cậy của câu trả lời dựa trên dữ liệu được cung cấp.

## Các mốc thời gian quan trọng
- **03 tháng 6, 2026:** Ngày khai mạc, phát hành tập dữ liệu kiểm thử
- **30 tháng 6, 2026:** Chính thức đóng cổng hệ thống, các đội phải hoàn thành nộp bài
- **05 tháng 7, 2026:** Công bố kết quả Top 10, tiến vào DemoDay
- **11 tháng 7, 2026:** Ngày DemoDay, công bố kết quả chung cuộc

*Lưu ý: Tất cả các hạn chót đều là 23:59 theo giờ Việt Nam (UTC+07:00).*

## Quy định về dữ liệu bên ngoài và mô hình ngôn ngữ huấn luyện trước (PLMs)
- Để đảm bảo sự công bằng trong cuộc thi, người tham gia không được sử dụng dữ liệu bên ngoài trong bất kỳ bước xử lý nào.
- Bạn thể sử dụng các mô hình ngôn ngữ huấn luyện trước và các LLM có dữ liệu huấn luyện và/hoặc mô hình được công khai có kích trước dưới 14B (ví dụ: Huggingface hoặc các trang tương tự), nhưng bạn không được sử dụng các LLM có mô hình đóng (ví dụ: GPT-4o, Gemini, ...). 
- Ngoài ra, bạn chỉ được sử dụng các mô hình được phát hành trước ngày 1 tháng 3 năm 2026 (giờ Việt Nam).
- Vì mục đích tái lập kết quả, vui lòng đưa thông tin về cách thức lấy mô hình vào bài báo.

## Phương pháp đánh giá
Hiệu năng của hệ thống được đánh giá bằng các chỉ số tự động và thủ công. Chúng tôi sử dụng trung bình macro (chỉ số đánh giá được tính cho từng truy vấn rồi lấy trung bình) để tính điểm đánh giá cuối cùng.

### 3.1 Truy hồi thông tin
Hiệu suất hệ thống trên nhiệm vụ truy hồi được đánh giá bằng các chỉ số Độ chính xác (Precision), Độ bao phủ (Recall) và điểm F2 macro. Chúng tôi sử dụng macro-average (tính chỉ số đánh giá cho từng truy vấn rồi lấy trung bình) để tính điểm đánh giá cuối cùng.

**Cách trích xuất điều luật từ câu trả lời:**
Hệ thống chấm điểm tự động tìm các pattern "Điều X" trong trường `relevant_docs` `relevant_articles` của bài nộp, sau đó so sánh với các điều luật trong đáp án (định danh đầy đủ dạng `law_id|tên văn bản|Điều X` được chuẩn hóa về `Điều X`).

- **Precision (Độ chính xác):**
  Precision = trung bình của (số điều luật truy hồi đúng cho mỗi truy vấn) / (số điều luật đã truy hồi cho mỗi truy vấn)
- **Recall (Độ bao phủ):**
  Recall = trung bình của (số điều luật truy hồi đúng cho mỗi truy vấn) / (số điều luật liên quan của mỗi truy vấn)
- **Chỉ số F2:**
  F2 = (5 × Precision × Recall) / (4 × Precision + Recall)

### 3.2 Hỏi đáp pháp luật
Bộ tiêu chí đánh giá bao gồm 5 nhóm:
1. **Căn cứ chính xác pháp luật:** Tỷ lệ câu hỏi có ít nhất một điều luật được trích xuất đúng từ câu trả lời. Đánh giá tự động.
2. **Tính chính xác nội dung:** Đánh giá mức độ chính xác của nội dung câu trả lời so với quy định pháp luật.
3. **Tính đầy đủ & toàn diện:** Đánh giá câu trả lời có bao quát đầy đủ các khía cạnh liên quan của câu hỏi không.
4. **Tính thực tiễn – khả năng áp dụng:** Đánh giá câu trả lời có thể áp dụng thực tế trong bối cảnh pháp lý không.
5. **Tính rõ ràng – dễ hiểu:** Đánh giá câu trả lời có diễn đạt rõ ràng, dễ hiểu cho người đọc không chuyên không.

#### 3.2.1 Đánh giá tự động
Chúng tôi sử dụng các mô hình ngôn ngữ lớn (LLMs) đóng vai trò là giám khảo tự động (LLM-as-a-Judge) để chấm điểm câu trả lời của hệ thống theo bộ tiêu chí 5 nhóm nêu trên. Với mỗi câu trả lời, LLM được cung cấp câu hỏi, câu trả lời tham chiếu, các điều luật căn cứ và câu trả lời của hệ thống cần đánh giá. LLM sẽ chấm điểm từng nhóm tiêu chí kèm theo lý do giải thích cụ thể.

#### 3.2.2 Con người đánh giá
Song song với đánh giá tự động, một tập con các câu trả lời sẽ được đánh giá độc lập bởi các chuyên gia pháp luật theo cùng bộ tiêu chí 5 nhóm. Mỗi chuyên gia chấm điểm từng nhóm tiêu chí theo thang điểm quy định, kèm theo nhận xét về ưu điểm và hạn chế của câu trả lời. Điểm Human Evaluation cuối cùng là trung bình cộng của các chuyên gia tham gia đánh giá.

*Lưu ý: 4 chỉ số đánh giá thủ công (Tính chính xác nội dung, Tính đầy đủ & toàn diện, Tính thực tiễn, Tính rõ ràng) hiện được đặt giá trị 0.0 và sẽ được cập nhật điểm số sau khi ban giám khảo hoàn thành đánh giá.*

## Dashboard kết quả
Các đội thi nộp kết quả dự đoán trực tiếp trên hệ thống Dashboard chính thức của cuộc thi. Mỗi lần nộp bài cần đảm bảo các yêu cầu sau:
- **Định dạng file:** kết quả được nộp dưới dạng file chuẩn theo mẫu do Ban Tổ chức quy định, với cấu trúc trường dữ liệu tuân thủ đúng đặc tả.
- **Nội dung file:** bao gồm kết quả dự đoán cho toàn bộ câu hỏi trong bộ dữ liệu kiểm thử. Các câu hỏi bị thiếu hoặc sai định dạng sẽ bị tính là dự đoán không hợp lệ.
- **Số lần nộp:** mỗi đội được giới hạn số lần nộp bài mỗi ngày (chi tiết sẽ được công bố trên Dashboard) nhằm đảm bảo tính công bằng và tránh hiện tượng dò đáp án.

## Định dạng nộp bài
Bạn phải nộp một file dự đoán duy nhất tên là `results.json`, sau đó nén thành file `.zip` phẳng (không chứa thư mục con). File JSON phải tuân theo cấu trúc sau:

```json
[
  {
    "id": <id_câu_hỏi>,
    "question": "<câu_hỏi>",
    "answer": "<câu_trả_lời>",
    "relevant_docs": ["<mã văn bản>|<tên văn bản>"],
    "relevant_articles": ["<mã văn bản>|<tên văn bản>|<điều>"]
  },
  ...
]

```

**Giải thích:**

* **id:** Định danh của câu hỏi (khớp với id trong tập dữ liệu kiểm thử).
* **question:** Nội dung câu hỏi pháp lý.
* **answer:** Câu trả lời văn bản của bạn. Hệ thống chấm điểm sẽ tự động trích xuất các điều luật được đề cập trong câu trả lời để đánh giá.
* **relevant_docs:** Danh sách văn bản pháp luật liên quan. Mỗi phần tử có định dạng: `<mã văn bản>|<tên văn bản>`. Trường `<tên văn bản>` phải được viết theo công thức: Loại văn bản + Mã văn bản + Trích yếu.
* **relevant_articles:** Danh sách điều luật liên quan. Mỗi phần tử có định dạng: `<mã văn bản>|<tên văn bản>|<điều>`. Trường `<tên văn bản>` phải được viết theo công thức: Loại văn bản + Mã văn bản + Trích yếu.

**Ví dụ bài nộp:**

```json
[
  {
    "id": 1,
    "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?",
    "answer": "Doanh nghiệp được hỗ trợ khi được thành lập, tổ chức và hoạt động theo pháp luật về doanh nghiệp; đáp ứng tiêu chí doanh nghiệp nhỏ và vừa, gồm số lao động tham gia bảo hiểm xã hội bình quân năm không quá 200 người và đáp ứng một trong hai tiêu chí: tổng nguồn vốn không quá 100 tỷ đồng hoặc tổng doanh thu của năm trước liền kề không quá 300 tỷ đồng. Ngoài ra, doanh nghiệp phải đáp ứng điều kiện cụ thể của từng nội dung hỗ trợ và thực hiện đầy đủ nghĩa vụ, trách nhiệm theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa và pháp luật có liên quan.",
    "relevant_docs": [
      "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
      "80/2021/NĐ-CP|Nghị định Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
    ],
    "relevant_articles": [
      "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4",
      "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 5",
      "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 5"
    ]
  }
]

```

## Cách tạo file nộp bài

Đặt tên file là `results.json`, sau đó nén trực tiếp file đó vào zip (không bọc thêm thư mục):

**Linux / macOS:**

```bash
zip submission.zip results.json

```

**Windows (PowerShell):**

```powershell
Compress-Archive -Path results.json -DestinationPath submission.zip

```

**Cấu trúc zip hợp lệ:** `results.json` phải nằm ngay ở gốc, không nằm trong thư mục con.

```
submission.zip
└── results.json

```

Vào mục My Submissions trên http://leaderboard.aiguru.com.vn/ và tải lên file `submission.zip`.

**Lưu ý:**

* File bắt buộc phải có tên `results.json` — sai tên sẽ không được chấm điểm.
* File zip chỉ chứa duy nhất `results.json` ở gốc, không nằm trong thư mục con.
* Các bài nộp thiếu câu hoặc sai định dạng sẽ không được đánh giá và không bị tính vào số lần nộp tối đa.

## Quy định mô hình

Các đội thi phải tuân thủ các ràng buộc sau đối với mô hình ngôn ngữ được sử dụng:

* **Kích thước:** mô hình có số tham số dưới 14B.
* **Thời điểm ra mắt:** mô hình phải được công bố chính thức trước ngày 01/03/2026.
* **Giấy phép:** mô hình phải là mã nguồn mở (open-source), trọng số được phép tải xuống và sử dụng tự do cho mục đích nghiên cứu.

Ban Tổ chức có quyền yêu cầu các đội cung cấp thông tin xác nhận mô hình sử dụng. Bài nộp không đáp ứng các ràng buộc trên sẽ bị loại khỏi bảng xếp hạng.

## Quy định nộp bài

* Mỗi đội được phép nộp tối đa 10 bài mỗi ngày.
* Số bài nộp tối đa cho mỗi người dùng trong Vòng Riêng (Private Phase) là 5 bài tổng cộng. Vì vậy, hãy chọn lựa các bài nộp ở Vòng Riêng một cách cẩn thận.
* Vui lòng chọn một tên người dùng đại diện cho đội của bạn.
* Kết quả cuối cùng sẽ không được xem là chính thức cho đến khi một bài báo mô tả phương pháp (working notes paper) với mô tả đầy đủ về các phương pháp được nộp.
* Ban Tổ chức Cuộc thi có toàn quyền, theo quyết định riêng của mình, loại bất kỳ thí sinh nào có bài nộp không tuân thủ tất cả các yêu cầu.

## Đánh giá QA

Phần đánh giá QA (Question Answering) không tự động chấm mọi bài nộp. Thay vào đó, mỗi đội sẽ tự chọn một bài nộp trong số các bài đã nộp và đẩy (promote) bài đó lên bảng xếp hạng. Chỉ những bài được đẩy lên leaderboard mới được đưa vào kỳ chấm QA.

* **Chọn bài để chấm QA:** từ danh sách bài đã nộp, đội chọn bài mong muốn và đẩy lên leaderboard. Đây là bài đại diện cho đội trong kỳ đánh giá QA tương ứng.
* **Chu kỳ đánh giá:** Ban Tổ chức chấm điểm QA cho các bài đã được đẩy lên leaderboard định kỳ mỗi tuần một lần. Đội có thể thay đổi bài được đẩy lên trước mỗi kỳ chấm; bài đang ở trên leaderboard tại thời điểm chấm sẽ được sử dụng.
* **Nội dung đánh giá:** hệ thống chấm điểm dựa trên chất lượng câu trả lời (answer) cùng độ chính xác của danh sách văn bản và điều luật liên quan (relevant_docs, relevant_articles).
* **Công bố kết quả:** sau mỗi kỳ đánh giá hằng tuần, điểm số và thứ hạng cập nhật của các đội sẽ được hiển thị trên http://leaderboard.aiguru.com.vn/.

## Dữ liệu cuộc thi

Ban Tổ chức cung cấp duy nhất bộ dữ liệu kiểm thử (test set): tập câu hỏi pháp lý, được sử dụng làm căn cứ chấm điểm và đánh giá hệ thống của các đội thi. Không cung cấp bất kỳ tập dữ liệu huấn luyện (train) hay tập phát triển (dev) nào.

Bộ đáp án chuẩn: được Ban Tổ chức giữ kín, chỉ phục vụ quá trình chấm điểm nhằm đảm bảo tính khách quan và công bằng.

Ban Tổ chức không cung cấp dữ liệu huấn luyện. Các đội thi được toàn quyền chủ động trong việc thu thập và khai thác dữ liệu, bao gồm:

* Văn bản pháp luật, thông tư, nghị định từ các nguồn chính thống.
* Dữ liệu liên quan đến doanh nghiệp SME (quy định về thuế, lao động, hợp đồng, v.v.).
* Các tập dữ liệu mở (open dataset) phục vụ bài toán Legal NLP.
* Mọi nguồn dữ liệu hợp pháp khác mà đội thi có thể tiếp cận.

Cuộc thi khuyến khích các đội phát huy tối đa sự sáng tạo trong toàn bộ quy trình xây dựng giải pháp, chẳng hạn:

* Thu thập và tiền xử lý dữ liệu.
* Thiết kế chiến lược chia nhỏ và biểu diễn dữ liệu (chunking, embedding...).
* Tối ưu hóa cơ chế truy hồi thông tin.
* Xây dựng pipeline AI hoàn chỉnh phù hợp với kiến trúc hệ thống riêng.

### 2.1 Dữ liệu cung cấp

Dữ liệu đầu vào bao gồm:

* **id:** Mã định danh của câu hỏi, kiểu số nguyên (integer).
* **question:** Nội dung câu hỏi pháp lý, kiểu chuỗi (string).

```json
{
  "id": <id_câu_hỏi>,
  "question": "<câu_hỏi>"
}

```

**Ví dụ dữ liệu:**

```json
{
  "id": 1,
  "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?"
}

```

### 2.2 Bài nộp lên hệ thống

Dữ liệu nộp lên hệ thống phải đầy đủ các trường sau:

* **id:** Mã định danh của câu hỏi, kiểu số nguyên (integer).
* **question:** Nội dung câu hỏi pháp lý, kiểu chuỗi (string).
* **answer:** Nội dung câu trả lời cho câu hỏi, kiểu chuỗi (string).
* **relevant_docs:** Danh sách văn bản pháp luật liên quan. Mỗi phần tử có định dạng: `<mã văn bản>|<tên văn bản>`
* **relevant_articles:** Danh sách điều luật liên quan. Mỗi phần tử có định dạng: `<mã văn bản>|<tên văn bản>|<điều>`

*Lưu ý: Trong cả relevant_docs và relevant_articles, trường `<tên văn bản>` phải được viết theo công thức: Loại văn bản + Mã văn bản + Trích yếu*

```json
{
  "id": <id_câu_hỏi>,
  "question": "<câu_hỏi>",
  "answer": "<câu_trả_lời>",
  "relevant_docs": ["<mã văn bản>|<tên văn bản>"],
  "relevant_articles": ["<mã văn bản>|<tên văn bản>|<điều>"]
}

```

**Ví dụ bài nộp:**

```json
{
  "id": 1,
  "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?",
  "answer": "Doanh nghiệp được hỗ trợ khi được thành lập, tổ chức và hoạt động theo pháp luật về doanh nghiệp; đáp ứng tiêu chí doanh nghiệp nhỏ và vừa, gồm số lao động tham gia bảo hiểm xã hội bình quân năm không quá 200 người và đáp ứng một trong hai tiêu chí: tổng nguồn vốn không quá 100 tỷ đồng hoặc tổng doanh thu của năm trước liền kề không quá 300 tỷ đồng. Ngoài ra, doanh nghiệp phải đáp ứng điều kiện cụ thể của từng nội dung hỗ trợ và thực hiện đầy đủ nghĩa vụ, trách nhiệm theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa và pháp luật có liên quan.",
  "relevant_docs": [
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
    "80/2021/NĐ-CP|Nghị định Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
  ],
  "relevant_articles": [
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4",
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 5",
    "80/2021/NĐ-CP|Nghị định Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 5"
  ]
}

```

*Lưu ý: Hệ thống chấm điểm sẽ tự động trích xuất các điều luật được đề cập trong trường `answer` của bài nộp để so sánh với `relevant_articles` trong đáp án.*

## Quy định chung

* **Quyền hủy bỏ, sửa đổi hoặc loại tư cách.** Ban Tổ chức Cuộc thi có toàn quyền theo quyết định riêng của mình để chấm dứt, sửa đổi hoặc tạm ngưng cuộc thi.
* Khi nộp kết quả cho cuộc thi này, bạn đồng ý cho phép công bố công khai điểm số của mình tại hội thảo của Cuộc thi và trong các kỷ yếu liên quan, theo quyết định của ban tổ chức nhiệm vụ. Điểm số có thể bao gồm nhưng không giới hạn ở các đánh giá định lượng tự động và thủ công, các đánh giá định tính, cùng các chỉ số khác mà ban tổ chức nhiệm vụ cho là phù hợp. Bạn chấp nhận rằng quyết định cuối cùng về việc lựa chọn chỉ số và giá trị điểm số thuộc về ban tổ chức nhiệm vụ.
* Khi tham gia cuộc thi, bạn đã chấp nhận các điều khoản và điều kiện của Điều khoản Tham gia và Thỏa thuận Sử dụng Dữ liệu của Nhiệm vụ chung R2AI2026 BUILD AI LEGAL ASSISTANT, đã được gửi đến email của bạn.
* Khi tham gia cuộc thi, bạn khẳng định và thừa nhận rằng bạn đồng ý tuân thủ các luật và quy định hiện hành, và bạn không được xâm phạm bất kỳ bản quyền, tài sản trí tuệ hoặc bằng sáng chế nào của bên khác đối với phần mềm bạn phát triển trong quá trình diễn ra cuộc thi, và sẽ không vi phạm bất kỳ luật và quy định hiện hành nào liên quan đến kiểm soát xuất khẩu cũng như quyền riêng tư và bảo vệ dữ liệu.
* Giải thưởng phụ thuộc vào việc Ban Tổ chức Cuộc thi xem xét và xác minh tư cách hợp lệ của thí sinh cũng như sự tuân thủ các quy định này, cùng với sự tuân thủ của các bài nộp thắng giải đối với các yêu cầu nộp bài.
* Người tham gia trao cho Ban Tổ chức Cuộc thi quyền sử dụng các bài nộp thắng giải cùng mã nguồn và dữ liệu được tạo ra cho và dùng để tạo ra bài nộp đó vì bất kỳ mục đích nào và không cần phê duyệt thêm.

## Điều kiện tham gia

* Mỗi người tham gia phải tạo một tài khoản để nộp giải pháp cho cuộc thi. Chỉ cho phép một tài khoản cho mỗi người dùng.
* Cuộc thi là công khai, nhưng Ban Tổ chức Cuộc thi có thể quyết định không cho phép tham gia theo cân nhắc riêng của mình.
* Ban Tổ chức Cuộc thi có quyền loại bất kỳ thí sinh nào khỏi cuộc thi nếu, theo quyết định riêng của Ban Tổ chức, họ có cơ sở hợp lý để tin rằng thí sinh đã cố gắng làm suy yếu hoạt động hợp pháp của cuộc thi thông qua gian lận, lừa dối hoặc các hành vi chơi không công bằng khác.

## Đội thi

* Người tham gia được phép lập đội.
* Bạn không được tham gia nhiều hơn một đội. Mỗi thành viên trong đội phải là một cá nhân riêng biệt vận hành một tài khoản riêng.
* Chỉ một tài khoản cho mỗi đội được phê duyệt để nộp kết quả.
