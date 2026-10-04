# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Mai Phan Anh Tùng (MSSV 2A202602980)  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8583 | 0.8542 | -0.0042 |
| Answer Relevancy | 0.7641 | 0.7968 | +0.0327 |
| Context Precision | 0.9250 | 0.9667 | +0.0417 |
| Context Recall | 0.9000 | 0.9500 | +0.0500 |

Điểm Production lần chạy đầu (trả **child 256 ký tự** cho LLM): faithfulness 0.7433, answer_relevancy 0.7361, precision 0.9458, recall 0.8167. Sau khi sửa `pipeline.py` để **retrieve child → trả parent** cho LLM, thêm `temperature=0` và siết system prompt, faithfulness +0.11 và recall +0.13.

Lưu ý: test set chỉ có 20 câu và RAGAS dùng LLM-judge nên điểm dao động vài % giữa các lần chạy (baseline cũng đổi 0.90 → 0.86 giữa hai lần). Δ so với baseline nhỏ nên không nên diễn giải quá mức.

### Latency breakdown (lần chạy cuối)

| Bước | Thời gian |
|------|-----------|
| M1 Chunking (104 chunks / 26 docs) | 0.1s |
| M5 Enrichment (104 calls gpt-4o-mini, tuần tự) | 409.3s |
| M2 Index (BM25 + bge-m3 → Qdrant) | 44.0s |
| M3 Rerank (20 candidates, CPU) | ~1266ms / query |
| RAGAS (4 metrics × 20 câu) | 34.8s |

## Bottom-5 Failures

### #1
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép (15 + 9÷3 = 3) theo v2024; lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** 18 ngày phép (đúng), nhưng lương: "không có thông tin cụ thể trong context".
- **Worst metric:** answer_relevancy (0.0); context_recall 0.5; avg 0.5625
- **Error Tree:** Output sai (thiếu nửa sau) → Context đúng? **Thiếu một phần (recall 0.5): có thông tin nghỉ phép, nhiều khả năng thiếu `bang_luong_2024.md`** → Query cần viết lại? **Có**: câu multi-hop 2 chủ đề nên tách thành 2 sub-query → Fix ở bước **Retrieval (M2)**.
- **Root cause (suy luận từ recall 0.5 và nội dung câu trả lời, chưa kiểm tra trực tiếp contexts):** Một query gộp 2 ý (nghỉ phép + lương) nên top-3 sau rerank bị chiếm bởi chunk nghỉ phép; chunk bảng lương không lọt vào. Cross-encoder chấm cả query nên cả hai chủ đề cạnh tranh nhau.
- **Suggested fix:** Query decomposition (tách thành 2 sub-query rồi merge context); hoặc tăng `RERANK_TOP_K` lên 4-5 cho câu multi-hop.

### #2
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn thanh toán 15 ngày, quá hạn 5 ngày, phí 2%/tháng trên 15 triệu = 300.000 VNĐ/tháng (pro-rata ~50.000 VNĐ cho 5 ngày).
- **Got:** "Phạt 600.000 VNĐ (2% của 15.000.000 cho 20 ngày)" — sai.
- **Worst metric:** faithfulness (0.0); context_recall 1.0, precision 1.0
- **Error Tree:** Output sai → Context đúng? **Có (recall 1.0)** → Query cần viết lại? Không, câu hỏi rõ ràng → LLM suy luận sai → Fix ở bước **Generation**.
- **Root cause:** Retrieval hoàn toàn đúng nhưng LLM tính nhầm: lấy 20 ngày thay vì 5 ngày quá hạn và nhân 2% hai lần. Đây là lỗi suy luận số học, không phải lỗi retrieval.
- **Suggested fix:** Prompt yêu cầu liệt kê từng bước (thời hạn → số ngày quá hạn → công thức) trước khi kết luận; hoặc dùng model mạnh hơn `gpt-4o-mini` cho câu numeric.

### #3
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director phê duyệt (5-50 triệu) + xác nhận cấu hình từ CNTT + ≥3 báo giá vì trên 10 triệu.
- **Got:** Director + xác nhận cấu hình CNTT; thiếu yêu cầu 3 báo giá.
- **Worst metric:** faithfulness (0.667); avg 0.8126
- **Error Tree:** Output thiếu → Context đúng? Có (recall 1.0) → Query cần viết lại? Không, câu đã nêu rõ 2 ý (ai phê duyệt + cần gì từ CNTT) → LLM bỏ sót điều kiện phụ → Fix ở bước **Generation / Prompt**.
- **Root cause:** Điều kiện "trên 10 triệu cần 3 báo giá" nằm ở đoạn khác với bảng hạn mức phê duyệt; LLM trả lời ngắn gọn (do prompt "ngắn gọn") nên bỏ điều kiện phụ. Siết prompt về độ ngắn đã đánh đổi completeness.
- **Suggested fix:** Prompt: "liệt kê đầy đủ mọi điều kiện/ yêu cầu liên quan trong context"; hoặc thêm bước verify câu trả lời với checklist điều kiện.

### #4
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo v2024 hiện hành: từ 3 năm, +1 ngày/3 năm (v2023 cũ yêu cầu 5 năm).
- **Got:** "Từ 3 năm trở lên, +1 ngày cho mỗi 3 năm" — **đúng**.
- **Worst metric:** context_precision (0.5); avg 0.823
- **Error Tree:** Output đúng → Context đúng? Có, nhưng precision 0.5 cho thấy có chunk không liên quan trong top-3 (nhiều khả năng chunk v2023 đã hết hiệu lực) → Query cần viết lại? Có thể thêm "theo chính sách hiện hành", nhưng nên xử lý ở hệ thống thay vì bắt người dùng → Fix ở bước **Rerank / Metadata filter**.
- **Root cause (suy luận từ precision 0.5, chưa kiểm tra trực tiếp contexts):** `nghi_phep_nam_v2023.md` và `v2024.md` gần như trùng nội dung, cross-encoder không phân biệt được bản còn hiệu lực. Câu trả lời đúng chỉ nhờ prompt "dùng bản mới nhất".
- **Suggested fix:** Gắn metadata `version`/`status: superseded` (M5 `extract_metadata`) rồi filter hoặc hạ điểm tài liệu superseded trước rerank.

### #5
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết 1 năm; nghỉ sau 8 tháng → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** "Hoàn trả 100%, tức 25 triệu VNĐ" — đúng.
- **Worst metric:** faithfulness (0.5); avg 0.825
- **Error Tree:** Output đúng → Context đúng? Có (recall 1.0, precision 1.0) → Query cần viết lại? Không → Faithfulness thấp vì **câu trả lời không nêu căn cứ "cam kết 1 năm"**, judge không verify được từng claim → Fix ở bước **Prompt**.
- **Root cause:** Kết luận đúng nhưng suy luận ngầm (8 tháng < 12 tháng) không được nêu rõ nên RAGAS chia claim và coi một phần là không được context hậu thuẫn. Đây một phần là artefact của LLM-judge.
- **Suggested fix:** Yêu cầu trích dẫn điều khoản/điều kiện làm căn cứ ("theo quy định cam kết 12 tháng...") trong câu trả lời.

## Case Study (cho presentation)

**Question chọn phân tích:** "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?" (và câu "Nhân viên thử việc có được hưởng bảo hiểm PVI không?") — ở lần chạy đầu cả hai đều trả "Không tìm thấy" dù context đã có đáp án, avg 0.25 / 0.5. Ở lần chạy cuối cả hai đã được trả lời đúng.

**Error Tree walkthrough:**
1. Output đúng? → Không, "Không tìm thấy."
2. Context đúng? → Chunk được retrieve nhưng chỉ là **child 256 ký tự**: đoạn ngưỡng 50 triệu / CEO bị cắt rời khỏi tiêu đề và bảng hạn mức nên LLM không đủ ngữ cảnh để kết luận.
3. Query rewrite OK? → Query rõ ràng, không cần rewrite.
4. Fix ở bước: **Pipeline (child → parent)**. Parent 2048 ký tự giữ nguyên bảng ngưỡng phê duyệt; kèm `temperature=0` và prompt chỉ nói "Không tìm thấy" khi context hoàn toàn không liên quan. Kết quả: faithfulness 0.74 → 0.85, recall 0.82 → 0.95.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Query decomposition cho câu multi-hop (#1).
- Metadata `version/status` + filter tài liệu superseded (#4).
- Cache kết quả enrichment ra file và chạy song song bằng `asyncio`/thread pool (409s → ~40s).
- OCR 2 PDF scan (`BCTC.pdf`, Nghị định 13) hiện đang bị bỏ qua; chạy reranker trên GPU hoặc dùng `FlashrankReranker` (1266ms → vài ms).
