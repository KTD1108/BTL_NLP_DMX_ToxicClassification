# HỆ THỐNG PHÂN LOẠI ĐÁNH GIÁ ĐỘC HẠI TRÊN SÀN TMĐT ĐIỆN MÁY XANH
## (Toxic Review Classification in E-Commerce Platforms)

> **Môn học:** Xử lý ngôn ngữ tự nhiên / Khai phá dữ liệu văn bản (NLP / Text Mining)  
> **Đơn vị:** Khoa Công nghệ Thông tin 1 – Học viện Công nghệ Bưu chính Viễn thông (PTIT)  

---

## 1. GIỚI THIỆU ĐỀ TÀI

Trong các sàn thương mại điện tử lớn như Điện Máy Xanh, hàng chục nghìn đánh giá sản phẩm được người dùng gửi về mỗi ngày. Đề tài này xây dựng hệ thống **Phân loại văn bản nhị phân có giám sát (Supervised Binary Text Classification)** nhằm tự động phát hiện và kiểm duyệt các bình luận độc hại:

* **Nhãn 0 (Non-toxic - Bình thường):** Bình luận khen ngợi, trung tính, hỏi đáp hoặc **phản ánh lỗi kỹ thuật sản phẩm thông thường** (*"máy giặt kêu to", "tủ lạnh không mát", "giao hàng chậm", "nấu cơm bị khét"*).
* **Nhãn 1 (Toxic - Độc hại):** Bình luận chứa ngôn từ thô tục, chửi thề, công kích cá nhân, xúc phạm nhân viên hoặc vu khống, quy chụp lừa đảo (*"lừa đảo", "ăn cướp", "đồ đểu", "bố láo", "mất dạy", "vcl"*).

### Thách thức chính của bài toán:
1. **Mất cân bằng dữ liệu tự nhiên cực lớn (Imbalanced Data):** Tỷ lệ bình luận độc hại trong thực tế chỉ chiếm khoảng **3.4% – 3.7%**.
2. **Nhiễu ngôn ngữ mạng:** Người dùng sử dụng nhiều từ lóng, teencode, viết tắt, cố tình biến âm để lách kiểm duyệt (*"đm"*, *"vcl"*, *"bịp"*...).
3. **Ranh giới mong manh giữa bức xúc kỹ thuật và xúc phạm:** Dễ nhầm lẫn giữa khách hàng giận dữ vì lỗi sản phẩm với khách hàng chửi bới độc hại.

---

## 2. BỘ DỮ LIỆU CHUẨN VÀNG (`data/`)

Dữ liệu được quy về một đầu mối duy nhất tại thư mục [`data/`](./data/), gồm đúng **10.000 đánh giá người dùng thực tế** từ 29 ngành hàng của hệ thống Điện Máy Xanh (không chứa dữ liệu nhân tạo, không có dòng trống NaN) và tuân thủ kiểm toán chống rò rỉ dữ liệu (Zero Data Leakage):

| Tập dữ liệu | File lưu trữ | Số mẫu | Non-toxic (0) | Toxic (1) | Tỷ lệ Toxic | Vai trò kỹ thuật |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | [`data/train.csv`](./data/train.csv) | **7.000** | 6.760 | 240 | **3.43%** | Huấn luyện mô hình phân loại |
| **Validation** | [`data/val.csv`](./data/val.csv) | **1.000** | 955 | 45 | **4.50%** | Tối ưu siêu tham số & hiệu chuẩn ngưỡng |
| **Test (Gold)** | [`data/test.csv`](./data/test.csv) | **2.000** | 1.925 | 75 | **3.75%** | **Kiểm thử độc lập đánh giá hiệu năng** |

*Cấu trúc dữ liệu chuẩn:* `review_id`, `product_id`, `nganh_hang`, `so_sao`, `noi_dung_goc`, `noi_dung_tien_xu_ly`, `nguon_nhan`, `label`.

---

## 3. KIẾN TRÚC MÔ HÌNH & PHÂN CÔNG NHIỆM VỤ

* **Thành viên 1: Kiều Tiến Đạt (B23DCKH020) - Data Architect & SVM Specialist:**
  * Xây dựng pipeline tiền xử lý tiếng Việt [`nlp_utils.py`](./nlp_utils.py): chuẩn hóa teencode/từ lóng, tách từ tiếng Việt.
  * Huấn luyện mô hình **Linear SVM** ([`3b_train_svm.py`](./3b_train_svm.py)): kết hợp đặc trưng đa nhánh `FeatureUnion` (Word TF-IDF + Char n-gram 3-5 + Toxic Lexicon) và Cost-Sensitive Learning (`class_weight={0: 1, 1: 15}`).
  * Đạt độ chính xác kiểm duyệt cao nhất: **Precision = 75.00%**, **F1-Score = 0.6412**, **PR-AUC = 0.7197**.

* **Thành viên 2: NLP Preprocessing & Naive Bayes Specialist:**
  * Huấn luyện mô hình **Multinomial Naive Bayes** ([`3a_train_naive_bayes.py`](./3a_train_naive_bayes.py)) trên BoW và TF-IDF.
  * Dò lưới tham số làm mịn Laplace $\alpha$ và ngưỡng quyết định Bayes trên 5-Fold Cross-Validation.
  * Đạt kết quả TF-IDF Naive Bayes: **Recall = 58.67%**, **F1-Score = 0.5207**, **PR-AUC = 0.5669**.

* **Thành viên 3: Deep Learning Specialist (PhoBERT Contextual Transformer):**
  * Xây dựng mô hình Transformer hiện đại dựa trên backbone **VinAI PhoBERT** (`vinai/phobert-base-v2`, 135M tham số) ([`3c_train_phobert.py`](./3c_train_phobert.py)).
  * Thực hiện **End-to-End Fine-Tuning** với hàm mất mát có trọng số phạt lỗi Toxic trên GPU T4 ([`TV3_PhoBERT_FineTuning_Colab.ipynb`](./TV3_PhoBERT_FineTuning_Colab.ipynb)).
  * Đạt độ bao quát vi phạm cao nhất toàn đề tài: **Recall = 60.00% (bắt đúng 45/75 câu toxic)**, **Precision = 62.50%**, **F1-Score = 0.6122**, **PR-AUC = 0.6357**.

* **Thành viên 4: Evaluation Lead & Statistical Analysis:**
  * Thực hiện đánh giá đối sánh độc lập 6 mô hình trên tập Test chuẩn vàng ([`4_model_evaluation.py`](./4_model_evaluation.py)).
  * Thực hiện kiểm định ý nghĩa thống kê **McNemar Test** ($p = 0.194$, khẳng định tính bổ trợ hoàn hảo giữa SVM và PhoBERT).
  * Xuất trực quan hóa kết quả 300 DPI: ma trận nhầm lẫn 6 mô hình, đồ thị đối sánh Macro F1 & Recall, bảng số liệu CSV và trích xuất 141 ca lỗi phân tích sâu.

---

## 4. KẾT QUẢ THỰC NGHIỆM TRÊN TẬP TEST (2.000 MẪU)

| STT | Thành viên | Mô hình phân loại | Accuracy | Precision (Toxic) | Recall (Toxic) | F1-Score (Toxic) | Macro F1 | PR-AUC | TP / 75 | FP |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **TV1** | **Word TF-IDF + Char n-gram + Lexicon + Linear SVM** | **97.65%** | **75.00%** | 56.00% | **0.6412** | **0.8145** | **0.7197** | 42 | 14 |
| 2 | **TV3** | **PhoBERT (vinai/phobert-base-v2, Fine-Tuned GPU T4)** | **97.15%** | **62.50%** | **60.00%** | **0.6122** | **0.7987** | **0.6357** | **45** | **27** |
| 3 | **TV2** | **TF-IDF + Naive Bayes (MultinomialNB)** | 95.95% | 46.81% | 58.67% | **0.5207** | 0.7498 | 0.5669 | 44 | 50 |
| 4 | TV1 | TF-IDF đơn + Linear SVM | 96.85% | 62.50% | 40.00% | 0.4878 | 0.7358 | 0.5232 | 30 | 18 |
| 5 | TV2 | BoW + Naive Bayes (MultinomialNB) | 95.45% | 41.49% | 52.00% | 0.4615 | 0.7189 | 0.4252 | 39 | 55 |
| 6 | TV1 | BoW + Linear SVM | 96.90% | 66.67% | 34.67% | 0.4561 | 0.7201 | 0.4489 | 26 | 13 |

---

## 5. HƯỚNG DẪN CÀI ĐẶT & THỰC THI

### Cài đặt môi trường:
```bash
git clone https://github.com/KTD1108/BTL_NLP_DMX_ToxicClassification.git
cd BTL_NLP_DMX_ToxicClassification

pip install scikit-learn pandas numpy matplotlib seaborn pyvi scipy transformers torch
```

### Chạy các mô hình thực nghiệm:
```bash
# 1. Huấn luyện và đánh giá Naive Bayes (Thành viên 2)
python 3a_train_naive_bayes.py

# 2. Huấn luyện và đánh giá Linear SVM (Thành viên 1)
python 3b_train_svm.py

# 3. Đánh giá kiểm thử PhoBERT (Thành viên 3)
python 3c_train_phobert.py

# 4. Đánh giá đối sánh toàn bộ mô hình, vẽ ma trận nhầm lẫn và biểu đồ (Thành viên 4)
python 4_model_evaluation.py
```

---

## 6. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
├── data/                             # Thư mục dữ liệu chuẩn mực duy nhất
│   ├── train.csv                     # Tập Train (7.000 mẫu)
│   ├── val.csv                       # Tập Validation (1.000 mẫu)
│   ├── test.csv                      # Tập Test chuẩn vàng (2.000 mẫu)
│   └── dmx_reviews_10k_raw.csv       # 10.000 bình luận gốc
├── results/                          # Kết quả đầu ra chính thức
│   ├── predictions/                  # File dự đoán nộp cho TV4 theo quy ước
│   │   ├── nb_bow.csv & _info.json
│   │   ├── nb_tfidf.csv & _info.json
│   │   ├── svm_bow.csv & _info.json
│   │   ├── svm_tfidf_single.csv & _info.json
│   │   ├── svm_tfidf.csv & _info.json
│   │   └── phobert.csv & _info.json
│   ├── 5_confusion_matrices.png      # Ma trận nhầm lẫn 6 mô hình (300 DPI)
│   ├── 6_model_comparison.png        # Biểu đồ cột Macro F1 & Recall (300 DPI)
│   ├── tv4_model_comparison.csv      # Bảng số liệu đối sánh chính thức
│   └── tv4_error_cases.csv           # Danh sách 141 ca lỗi phân tích sâu
├── 1_sampling_and_eda.py             # Script phân tích khám phá dữ liệu (EDA)
├── 3a_train_naive_bayes.py           # Mã nguồn Thành viên 2 (Naive Bayes)
├── 3b_train_svm.py                   # Mã nguồn Thành viên 1 (Linear SVM)
├── 3c_train_phobert.py               # Mã nguồn Thành viên 3 (PhoBERT Transformer)
├── 4_model_evaluation.py             # Mã nguồn Thành viên 4 (Đánh giá đối sánh & Thống kê)
├── TV3_PhoBERT_FineTuning_Colab.ipynb# Notebook Google Colab GPU T4 huấn luyện PhoBERT
├── nlp_utils.py                      # Module tiền xử lý văn bản tiếng Việt dùng chung
└── raw.zip                           # Dữ liệu cào gốc ban đầu
```
