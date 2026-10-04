# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Mai Phan Anh Tùng (MSSV 2A202602980)  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 2026-10-04

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 trên toàn corpus tạo **208 chunks** (avg 99 ký tự, min 6) vs basic **51 chunks** (avg 410). Ngưỡng 0.85 quá cao với MiniLM nên câu liên tiếp hiếm khi đủ giống để gộp → chunk rất nhỏ, vụn. Không dùng cho pipeline chính. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | 99 child (avg 210, max 256) / parent 2048. Child cho precision khi search, **parent trả cho LLM**. Lần đầu quên trả parent → LLM nhận đoạn 256 ký tự và trả "Không tìm thấy"; sau khi sửa faithfulness 0.74 → 0.85. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | 106 chunks theo header `#`, giữ header + bảng nguyên vẹn (max 788 ký tự), có `section` metadata. Phù hợp với corpus quy chế dạng markdown. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()`, `segment_vietnamese()` | RRF chỉ dựa trên thứ hạng nên không cần chuẩn hóa điểm BM25 vs cosine. Phải `replace("_", " ")` sau underthesea, nếu không BM25 coi "nghỉ_phép" là một token và query "nghỉ phép" không khớp. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 trên CPU: **~1266ms** cho 20 candidate (vượt mục tiêu 150ms). Context precision đạt 0.9667. Giải pháp production: GPU hoặc `FlashrankReranker`. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Final: faithfulness 0.854, relevancy 0.797, precision 0.967, recall 0.950. Relevancy thấp nhất vì các câu multi-hop trả lời thiếu một nửa nên điểm 0. Diagnostic Tree chỉ ra worst metric cho từng câu. |
| Contextual embeddings / Enrichment | M5 | `_enrich_single_call()` | 1 call/chunk (104 calls, 409s) trả summary + questions + context + metadata. Context line được prepend vào chunk trước khi embed. Chi phí thời gian lớn nhất của pipeline vì gọi tuần tự. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `SyntaxError: unterminated string literal (detected at line 100)` tại `re.split(r'(?<=[.!?])\s+|` trong `m1_chunking.py`: khi chèn code bằng script, `\n` bị chuyển thành xuống dòng thật ngay trong chuỗi.
  - `ModuleNotFoundError` (test `test_compare_all_strategies`): chưa cài `pypdf` vì chưa chạy `pip install -r requirements.txt`.
  - Chạy lần đầu `main.py`: pipeline exit 0 nhưng Production **không hơn baseline** (faithfulness 0.7433 < 0.8222, recall 0.8167 < 0.9250); câu "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?" trả `Không tìm thấy.`.
  - `pip install` báo xung đột: `langgraph-prebuilt 1.1.0 requires langchain-core>=1.3.1, but you have langchain-core 0.2.43`, `scikit-survival 0.27.0 requires numpy>=2.0.0` (cài vào Python global).
- **Nguyên nhân gốc rễ & Cách debug:**
  - SyntaxError: in lại các dòng chứa `\n` trong file, sửa bằng cách ghi lại escape đúng (`\\n`) rồi chạy lại pytest → 23/23 pass cho M1-M3.
  - Production kém baseline: đọc `failures` trong `ragas_report.json`, thấy các câu điểm thấp có `context_precision=1.0` nhưng `faithfulness=0` và answer = "Không tìm thấy". Nghĩa là retrieval đúng nhưng LLM không đủ ngữ cảnh. `pipeline.py` truyền child 256 ký tự vào LLM thay vì parent → sửa `run_query` map child → parent (`parent_map`), dedupe, `temperature=0`, siết prompt. Kết quả: faithfulness 0.74 → 0.85, recall 0.82 → 0.95.
  - Xung đột pip: do cài vào môi trường global; không ảnh hưởng lab nhưng lần sau nên dùng `.venv` riêng.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Chưa rõ vì sao điểm RAGAS dao động giữa các lần (baseline 0.90 → 0.86). Đọc cách RAGAS dùng LLM-judge: điểm không deterministic, test set 20 câu nhỏ → chỉ nên so sánh xu hướng, không so từng số.
  - Chưa có OCR cho 2 PDF scan (`BCTC.pdf`, Nghị định 13) nên bị bỏ qua; cần tìm hiểu thêm OCR (Tesseract / VLM).

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Production RAG — Hỏi đáp quy chế nội bộ (chính project của Lab 18)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Hierarchical chunking (child 256 / parent 2048, 104 child từ 26 tài liệu) → Enrichment `_enrich_single_call()` (gpt-4o-mini, 1 call/chunk) → Hybrid Search (BM25 underthesea + bge-m3/Qdrant + RRF, top-20) → Cross-encoder bge-reranker-v2-m3 (top-3) → map child → parent → gpt-4o-mini (`temperature=0`) → RAGAS. Điểm hiện tại: faithfulness 0.854, answer_relevancy 0.797, context_precision 0.967, context_recall 0.950.
- **Vấn đề / Bottlenecks đang gặp** (từ `failure_analysis.md` và latency đo được):
  - Câu **multi-hop** (nghỉ phép + lương Senior) chỉ trả lời được một nửa: top-3 bị một chủ đề chiếm hết.
  - **Xung đột phiên bản:** chunk `nghi_phep_nam_v2023.md` (hết hiệu lực) vẫn lọt top-3 (precision 0.5 ở câu thâm niên); trả lời đúng chỉ nhờ prompt.
  - **Suy luận số học sai** dù context đúng (tạm ứng quá hạn: trả 600.000 thay vì tính trên 5 ngày quá hạn).
  - Câu trả lời ngắn bỏ sót điều kiện phụ (thiếu "3 báo giá" khi mua laptop 30 triệu).
  - **Latency:** enrichment 409s (tuần tự), rerank ~1266ms/query trên CPU (mục tiêu < 150ms).
  - 2 PDF scan (`BCTC.pdf`, Nghị định 13) bị bỏ qua vì chưa OCR.
  - Test set chỉ 20 câu, điểm RAGAS dao động giữa các lần chạy nên khó đánh giá cải tiến nhỏ.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Giữ hierarchical (retrieve child → trả parent, đã chứng minh tăng faithfulness 0.74 → 0.85), nhưng tạo parent theo **structure-aware** (mỗi section `#`/`##` là một parent) để bảng ngưỡng phê duyệt và điều kiện đi kèm không bị tách. Bỏ semantic chunking vì ngưỡng 0.85 tạo chunk quá vụn (avg 99 ký tự).
2. **Search retrieval:** Giữ Hybrid BM25 + dense + RRF. Thêm **query decomposition**: LLM tách câu multi-hop thành sub-query, search từng cái rồi gộp context.
3. **Reranking:** Giữ bge-reranker-v2-m3 nhưng (a) tăng top_k lên 4–5 cho câu multi-hop, (b) thử `FlashrankReranker` / chạy GPU để đưa latency về < 150ms, so sánh precision giữa hai reranker.
4. **Evaluation:** Mở rộng test set lên ~50 câu (thêm version-conflict, numeric, multi-hop), chạy RAGAS 3 lần lấy trung bình để giảm nhiễu; lưu `contexts` vào report để kiểm chứng trực tiếp root cause thay vì suy luận từ điểm.
5. **Enrichment:** Dùng `metadata` từ `_enrich_single_call()` để gắn `version` / `status: superseded` và **lọc tài liệu hết hiệu lực** trước rerank. Cache kết quả enrichment ra file JSON + chạy song song (thread pool) để giảm 409s. Generation: prompt yêu cầu nêu căn cứ + tính từng bước cho câu số liệu, liệt kê đủ mọi điều kiện.

#### 3. Timeline triển khai
- **Tuần 1:** Mở rộng test set lên ~50 câu, lưu contexts vào report, chạy RAGAS 3 lần để có baseline ổn định. Cache + song song hóa enrichment.
- **Tuần 2:** Metadata `version/status` + filter tài liệu superseded; parent theo structure-aware. Đo lại precision ở các câu version-conflict.
- **Tuần 3:** Query decomposition cho multi-hop + cải tiến prompt (tính từng bước, nêu căn cứ, đủ điều kiện). Mục tiêu answer_relevancy ≥ 0.85.
- **Tuần 4:** Tối ưu latency rerank (Flashrank/GPU), OCR 2 PDF scan, chạy đánh giá cuối và cập nhật `failure_analysis.md`.
