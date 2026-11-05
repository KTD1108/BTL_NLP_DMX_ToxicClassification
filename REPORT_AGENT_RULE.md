# SYSTEM RULE: AI AGENT AUTOMATED ACADEMIC REPORT WRITER (GOOGLE DOCS MCP)
**Bối cảnh dự án:** Bài tập lớn 1 (PTIT) - Hệ thống phân loại bình luận độc hại (Toxic vs Non-toxic) trên dữ liệu đánh giá sản phẩm Điện Máy Xanh.

---

## PHẦN 1: QUY TẮC VẬN HÀNH GOOGLE DOCS MCP (OPERATIONAL PROTOCOL)

1. **Nguyên tắc Khởi tạo & Định danh (Init & Read-First):**
   - Trước khi ghi nội dung, Agent bắt buộc gọi tool đọc cấu trúc tài liệu (`get_document` / `read-tab` / `list-tabs`) nếu thao tác trên file có sẵn để lấy thông tin Tab ID chính xác (ví dụ Thẻ 3: `t.lqauesz2nqyq`).
   - Tuyệt đối không ghi đè làm mất nội dung các thẻ khác nếu chưa được phép; thao tác đúng trên Tab mục tiêu (`--tab <TAB_ID>`).

2. **Quy chuẩn Định dạng Văn bản Học thuật (Formatting Standards):**
   - **Hệ thống phân cấp tiêu đề:** Sử dụng native Google Docs Heading styles (`HEADING_1` cho Chương/Phần lớn, `HEADING_2` cho mục con `1.1`, `1.2`, `HEADING_3` cho mục `1.1.1`) để tự động sinh Mục lục (Table of Contents).
   - **Phông chữ & Căn lề:** Times New Roman hoặc Arial, cỡ chữ `12pt` cho nội dung chính (`NORMAL_TEXT`), giãn dòng `1.15` - `1.5`, căn đều hai bên (`JUSTIFIED`).
   - **Bảng biểu (Tables):** Mọi bảng so sánh số liệu (Dataset, Kết quả thực nghiệm, Confusion Matrix) phải được dựng bằng native Table của Google Docs API (không dùng bảng ASCII/Markdown thô). Hàng tiêu đề (Header row) phải in đậm và tô màu nền nhạt (`#F3F3F3`).
   - **Công thức Toán học:** Trình bày bằng ký hiệu Unicode toán học chuẩn xác (ví dụ: `wᵀx + b = 0`, `ξᵢ²`, `χ² = 1.6875`, `∑`, `∈ ℝ²⁵¹²⁰`) kèm giải thích rõ từng biến số bên dưới.

3. **Nguyên tắc Trung thực Dữ liệu (Zero Hallucination & Ground-Truth Binding):**
   - Tuyệt đối không tự bịa đặt số liệu. Mọi con số đưa vào Google Docs phải khớp 100% với tệp kết quả thực nghiệm mới nhất của dự án:
     - **Quy mô dữ liệu phân tầng:** Tổng 10.000 mẫu sau khi lọc trùng và phân chia độc lập (Train: 7.011 | Val: 997 | Test: 1.992). Tỷ lệ nhãn Toxic tự nhiên ~4% (Train: 301 toxic - 4.29% | Val: 45 toxic - 4.51% | Test: 80 toxic - 4.02%).
     - **Đặc trưng (Feature Extraction):**
       1. Bag-of-Words (BoW, n-gram 1-1 & 1-2).
       2. TF-IDF đơn (n-gram 1-2, sublinear TF).
       3. `FeatureUnion` 3 nhánh (25.120 chiều = 10.000 Word TF-IDF 1-2 gram + 15.000 `char_wb` TF-IDF 3-5 gram + 120 Toxic Lexicon).
     - **Mô hình (Models):**
       1. **Naive Bayes** (MultinomialNB kết hợp BoW và TF-IDF, tối ưu ngưỡng trên validation).
       2. **Linear SVM** (`C = 0.1`, `loss='squared_hinge'`, `class_weight={0: 1, 1: 15}`).
       3. **PhoBERT Fine-tuned** (`vinai/phobert-base-v2`, 135 triệu tham số, end-to-end fine-tuning với weighted cross-entropy, early stopping patience=2, chạy 3 seeds 42, 13, 7; nộp seed 13 có PR-AUC validation cao nhất).
       4. **Kiến trúc phối hợp 2 tầng (Two-Stage Routing):** Phối hợp tốc độ CPU của SVM FeatureUnion và độ sâu ngữ cảnh GPU của PhoBERT.

---

## PHẦN 2: KHUNG CẤU TRÚC BÁO CÁO BẮT BUỘC (REPORT BLUEPRINT)

Agent phải viết đầy đủ và tuần tự theo đúng cấu trúc đề bài PTIT yêu cầu:

### [TRANG BÌA & TÊN ĐỀ TÀI]
- **Tiêu đề:** XÂY DỰNG HỆ THỐNG PHÂN LOẠI BÌNH LUẬN ĐỘC HẠI (TOXIC COMMENT CLASSIFICATION) TRÊN DỮ LIỆU ĐÁNH GIÁ SẢN PHẨM ĐIỆN MÁY XANH.

### CHƯƠNG 1: GIỚI THIỆU BÀI TOÁN
1. **Bối cảnh và tính cấp thiết:** Bài toán kiểm duyệt bình luận thương mại điện tử tại Điện Máy Xanh (lọc chửi thề, công kích, cáo buộc gian lận, phân biệt với khiếu nại kỹ thuật hợp lệ).
2. **Phát biểu bài toán:** Phân loại nhị phân (Binary Classification): Nhãn `1` (Toxic) và Nhãn `0` (Non-toxic).
3. **Thách thức thực tế:** Dữ liệu mất cân bằng nghiêm trọng (Imbalanced Data ~4% Toxic), ngôn ngữ mạng Việt Nam nhiều teencode, viết tắt lách luật (`vl`, `đm`, `l.ừ.a đ.ả.o`) và câu chê gián tiếp, châm biếm ngầm.
4. **Mục tiêu và bố cục:** Mục tiêu hệ thống, phạm vi thử nghiệm và cấu trúc tài liệu.

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT
1. **Tiền xử lý văn bản tiếng Việt:** Chuẩn hóa Unicode NFC, xóa ký tự lặp, chuẩn hóa Teencode, tách từ bằng `PyVi`, xử lý Stopwords.
2. **Các phương pháp Trích chọn đặc trưng:**
   - Túi từ (Bag-of-Words - BoW).
   - Tần suất từ - Nghịch đảo tần suất văn bản (TF-IDF & Sublinear TF scaling).
   - Kiến trúc ghép nối song song `FeatureUnion` 3 nhánh (Word N-gram + Character Within-Word-Boundary `char_wb` + Từ điển Toxic Lexicon) đạt 25.120 chiều. Lý giải tại sao không dùng Mean Pooling Word2Vec/FastText cho bài toán phát hiện từ khóa thưa.
3. **Các thuật toán Phân loại truyền thống:**
   - **Naive Bayes (MultinomialNB):** Công thức xác suất hậu nghiệm Bayes và làm mịn Laplace $\alpha$. Tinh chỉnh ngưỡng phân loại trên tập Validation.
   - **Support Vector Machine (Linear SVM):** Siêu phẳng phân cách cực đại, bài toán tối ưu biên mềm nhạy chi phí (Cost-Sensitive Soft-Margin với Squared Hinge Loss và `class_weight`), lý giải Định lý Cover tại sao chọn Linear SVM cho không gian đặc trưng siêu chiều thay vì Kernel RBF.
4. **Mô hình Ngôn ngữ Tiền huấn luyện Transformer (PhoBERT):**
   - Kiến trúc Transformer `vinai/phobert-base-v2` (135 triệu tham số), cơ chế Multi-Head Self-Attention và thuật toán tách từ BPE tiếng Việt.
   - Chiến lược End-to-End Fine-Tuning với hàm mất mát Weighted Cross-Entropy, bộ lập lịch Linear Warmup Scheduler và cơ chế Early Stopping.
5. **Hệ thống chỉ số Đánh giá:** Accuracy, Precision, Recall, F1-Score (Toxic), Macro-F1, PR-AUC, Confusion Matrix.

### CHƯƠNG 3: THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ
1. **Bước 1 - Thu thập và Phân chia dữ liệu:** Quy trình thu thập đánh giá Điện Máy Xanh, gán nhãn, chia tập Stratified (Train: 7.011, Val: 997, Test: 1.992) và cam kết chống rò rỉ dữ liệu (**Zero Data Leakage**: kiểm toán 0% trùng lặp ID và văn bản).
2. **Bước 2 - Tiền xử lý dữ liệu:** Pipeline chuẩn hóa tiếng Việt qua module `nlp_utils`.
3. **Bước 3 & 4 - Thiết lập Trích chọn đặc trưng và Huấn luyện mô hình:**
   - Cấu hình của 6 biến thể mô hình (Naive Bayes BoW, Naive Bayes TF-IDF, SVM BoW, SVM TF-IDF đơn, SVM FeatureUnion 3 nhánh, PhoBERT Fine-tuned).
   - Dò siêu tham số GridSearchCV 5-Fold Stratified và tối ưu hóa ngưỡng trên tập Validation.
4. **Bước 5 - Đánh giá và So sánh kết quả thực nghiệm:**
   - **Bảng tổng hợp kết quả trên tập Test (1.992 mẫu):**
     | STT | Mô hình | Accuracy | Precision (Toxic) | Recall (Toxic) | F1 (Toxic) | Macro-F1 | PR-AUC | TP | FP | FN | TN |
     | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
     | 1 | **PhoBERT (`vinai/phobert-base-v2`)** | **98.29%** | **78.05%** | **80.00%** | **0.7901** | **0.8906** | **0.7580** | **64** | 18 | **16** | 1894 |
     | 2 | Linear SVM (FeatureUnion 3 nhánh) | 97.14% | 70.91% | 48.75% | 0.5778 | 0.7815 | 0.6495 | 39 | **16** | 41 | 1896 |
     | 3 | Linear SVM (TF-IDF đơn) | 96.69% | 60.61% | 50.00% | 0.5479 | 0.7654 | 0.5574 | 40 | 26 | 40 | 1886 |
     | 4 | Naive Bayes (TF-IDF) | 94.33% | 37.01% | 58.75% | 0.4541 | 0.7121 | 0.4530 | 47 | 80 | 33 | 1832 |
     | 5 | Linear SVM (BoW) | 96.23% | 54.55% | 37.50% | 0.4444 | 0.7125 | 0.4810 | 30 | 25 | 50 | 1887 |
     | 6 | Naive Bayes (BoW) | 94.53% | 37.17% | 52.50% | 0.4352 | 0.7032 | 0.3541 | 42 | 71 | 38 | 1841 |
   - **Phân tích Confusion Matrix:** Chi tiết thế mạnh của PhoBERT (bắt đúng 64/80 ca vi phạm, chỉ sót 16 ca) so với Linear SVM (chỉ báo nhầm 16 ca FP).
   - **Nghiên cứu triệt tiêu (Ablation Study):** Phân tích bước nhảy từ BoW $\rightarrow$ TF-IDF đơn $\rightarrow$ FeatureUnion $\rightarrow$ PhoBERT Transformer.
   - **Phân tích lỗi định tính (Error Analysis):** Khảo sát 169 ca phân loại sai trong `tv4_error_cases.csv`, phân tích các ca khó đặc trưng (Review ID `59177191`, `59425217`, `56417329`).
   - **Đề xuất Kiến trúc phối hợp 2 tầng (Two-Stage Pipeline):** Fast-Path CPU bằng Linear SVM FeatureUnion xử lý 95% bình luận rõ ràng (độ trễ <1ms), Deep-Path GPU bằng PhoBERT thẩm định 5% bình luận nằm trong vùng ranh giới bất định.

### CHƯƠNG 4: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
- Tổng kết đóng góp khoa học và thực tiễn của dự án.
- Hướng mở rộng: Tăng cường dữ liệu bằng LLM sinh mẫu đối kháng (Adversarial Data Augmentation) và nén mô hình PhoBERT bằng kiến trúc chưng cất tri thức (Knowledge Distillation) để triển khai thực tế.

### TÀI LIỆU THAM KHẢO
- Trích dẫn chuẩn IEEE các bài báo gốc: Cortes & Vapnik (1995 - SVM), Fan et al. (2008 - LIBLINEAR), Nguyen & Nguyen (2020 - PhoBERT), Manning et al. (Introduction to Information Retrieval).

---

## PHẦN 3: GIAO THỨC TỰ KIỂM THỬ & NGHIỆM THU BÁO CÁO (AUTOMATED QA & VERIFICATION PROTOCOL)

> **[BẮT BUỘC]** Sau khi hoàn tất việc ghi nội dung vào Google Docs, Agent **KHÔNG ĐƯỢC kết thúc nhiệm vụ ngay**. Agent phải gọi tool đọc lại toàn bộ nội dung file Google Docs vừa tạo (`get_document` / `read-tab`) và chạy kịch bản kiểm thử 4 tầng dưới đây.

### Tầng 1: Kiểm thử Cấu trúc Đề cương (Structural Verification)
- [ ] **Check 1.1:** Có mặt đầy đủ 4 phần chính theo yêu cầu đề bài: `Tiêu đề`, `Chương 1. Giới thiệu bài toán`, `Chương 2. Cơ sở lý thuyết`, `Chương 3. Thực nghiệm và kết quả`, `Chương 4. Kết luận`, `Tài liệu tham khảo`.
- [ ] **Check 1.2:** Có mặt đầy đủ 5 bước thực hiện trong Chương 3: (1) Thu thập dữ liệu, (2) Tiền xử lý dữ liệu, (3) Trích chọn đặc trưng, (4) Xây dựng & huấn luyện mô hình, (5) Đánh giá đối sánh mô hình.

### Tầng 2: Kiểm thử Đáp ứng Yêu cầu Đề bài PTIT (Requirement Traceability Audit)
- [ ] **Check 2.1 (Nguồn dữ liệu & Bài toán):** Xác nhận rõ nguồn dữ liệu là *Review sản phẩm Điện Máy Xanh* và dạng bài toán là *Phân loại Toxic và Non-toxic*.
- [ ] **Check 2.2 (Số lượng phương pháp trích chọn đặc trưng $\ge 2$):** Kiểm tra báo cáo có trình bày và so sánh ít nhất 2 phương pháp (BoW, TF-IDF, và mở rộng FeatureUnion).
- [ ] **Check 2.3 (Thuật toán bắt buộc):** Kiểm tra có đủ cả 2 thuật toán **Naive Bayes** và **SVM**.
- [ ] **Check 2.4 (Yêu cầu bổ sung ý 4 - Mô hình hiện đại):** Kiểm tra có phần trình bày và thực nghiệm mô hình hiện đại (**PhoBERT** & **Kiến trúc phối hợp 2 tầng**).
- [ ] **Check 2.5 (Độ đo bắt buộc):** Kiểm tra sự xuất hiện đầy đủ của cả 5 chỉ số: `Accuracy`, `Precision`, `Recall`, `F1-Score`, và `Confusion Matrix`.

### Tầng 3: Kiểm thử Đối khớp Số liệu (Data Integrity & Anti-Hallucination Test)
Agent thực hiện kiểm tra đối chiếu chuỗi (Exact Value Matching) trên văn bản Google Docs:
- [ ] **Check 3.1:** Quy mô tập dữ liệu = `10.000` (Train `7.011`, Val `997`, Test `1.992`), tỷ lệ Toxic tập Test = `4.02%` (80 mẫu).
- [ ] **Check 3.2:** Số chiều FeatureUnion = `25.120` (`10.000` + `15.000` + `120`).
- [ ] **Check 3.3:** Chỉ số của Linear SVM (FeatureUnion): Accuracy = `97.14%`, Precision = `70.91%`, Recall = `48.75%`, F1 = `0.5778`, TP = `39`, FP = `16`, FN = `41`, TN = `1896`.
- [ ] **Check 3.4:** Chỉ số của PhoBERT: Accuracy = `98.29%`, Precision = `78.05%`, Recall = `80.00%`, F1 = `0.7901`, Macro-F1 = `0.8906`, PR-AUC = `0.7580`, TP = `64`, FP = `18`, FN = `16`, TN = `1894`.
- [ ] **Check 3.5:** Số ca phân loại sai trong Error Analysis = `169` ca (`tv4_error_cases.csv`), 12 ca bị tất cả 6 mô hình đoán sai.

### Tầng 4: Kiểm thử Định dạng & Trình bày (Formatting & Artifact Inspection)
- [ ] **Check 4.1:** Không còn sót các ký tự Markdown thô chưa được parse (như `###`, `**`, `|---|---|`, ```` ``` ````) nằm lộ thiên trong văn bản Google Docs.
- [ ] **Check 4.2:** Các bảng số liệu đã được chuyển thành đối tượng Table thực sự của Google Docs.

### Quy tắc Tự sửa lỗi (Self-Healing Loop) & Báo cáo Nghiệm thu:
1. Nếu **BẤT KỲ** chốt kiểm tra nào ở 4 tầng trên bị `FAIL`: Agent phải tự động cập nhật để bổ sung hoặc sửa lại ngay phần bị thiếu/sai trên Google Docs, sau đó chạy lại bước kiểm thử cho đến khi đạt `100% PASS`.
2. Khi tất cả các chốt kiểm tra đều `PASS`, Agent xuất **Biên bản Nghiệm thu Báo cáo (QA Verification Report)** ra màn hình chat cho người dùng.
