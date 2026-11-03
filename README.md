# BÀI TẬP LỚN 1: HỆ THỐNG PHÂN LOẠI BÌNH LUẬN ĐỘC HẠI (TOXIC VS NON-TOXIC)
## DỮ LIỆU ĐÁNH GIÁ SẢN PHẨM HỆ THỐNG ĐIỆN MÁY XANH

> **Học phần:** Xử lý ngôn ngữ tự nhiên / Khai phá dữ liệu văn bản (NLP / Text Mining)  
> **Đơn vị:** Khoa Công nghệ Thông tin 1 – Học viện Công nghệ Bưu chính Viễn thông (PTIT)  
> **Repository:** [https://github.com/KTD1108/BTL_NLP_DMX_ToxicClassification](https://github.com/KTD1108/BTL_NLP_DMX_ToxicClassification)  

---

## 1. TỔNG QUAN BÀI TOÁN & BỘ DỮ LIỆU CHUẨN VÀNG (10.000 MẪU)

Đề tài giải quyết bài toán **Phân loại văn bản nhị phân có giám sát (Supervised Binary Text Classification)** trên các đánh giá của khách hàng từ 29 ngành hàng của hệ thống Điện Máy Xanh:
* **Nhãn 0 (Non-toxic - Bình thường):** Các bình luận khen ngợi, trung tính, hỏi đáp hoặc **phản ánh lỗi kỹ thuật sản phẩm thông thường** (*"máy giặt kêu to", "tủ lạnh không mát", "hơi đắt", "nấu cơm bị khét"*).
* **Nhãn 1 (Toxic - Độc hại):** Bình luận chứa ngôn từ thô tục, công kích, xúc phạm nhân viên bán hàng hoặc quy chụp gian dối, vu khống lừa đảo (*"bố láo", "lừa đảo", "ăn bớt 1m dây", "đồ đểu", "cạch mặt", "vcl"*).

### Bộ dữ liệu chuẩn mực chống rò rỉ (Zero Data Leakage) trong thư mục `data/`:
Toàn bộ 10.000 mẫu đều là **đánh giá từ người dùng thực tế trên sàn Điện Máy Xanh** (100% người thật, không dữ liệu nhân tạo, không dòng trống NaN), được kiểm toán 4 tầng chống rò rỉ dữ liệu (0% trùng lặp ID, 0% trùng lặp văn bản gốc, 0% trùng lặp văn bản sau tiền xử lý):

| Tập dữ liệu | Đường dẫn lưu trữ | Quy mô | Non-toxic (0) | Toxic (1) | Tỷ lệ Toxic | Vai trò kỹ thuật |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | [`data/train.csv`](./data/train.csv) | **7.000** | 6.760 | 240 | **3.43%** | Huấn luyện mô hình phân loại |
| **Validation** | [`data/val.csv`](./data/val.csv) | **1.000** | 955 | 45 | **4.50%** | Dò ngưỡng quyết định & Early Stopping |
| **Test (Gold)** | [`data/test.csv`](./data/test.csv) | **2.000** | 1.925 | 75 | **3.75%** | **Kiểm thử độc lập đánh giá chuẩn vàng** |

*Cấu trúc 8 cột đồng nhất:* `['review_id', 'product_id', 'nganh_hang', 'so_sao', 'noi_dung_goc', 'noi_dung_tien_xu_ly', 'nguon_nhan', 'label']`.

---

## 2. PHÂN CÔNG NHIỆM VỤ NHÓM 4 THÀNH VIÊN

* **Thành viên 1: Kiều Tiến Đạt (B23DCKH020) - Data Architect & SVM Specialist:**
  * Khai phá, làm sạch và xây dựng bộ 3 Train/Val/Test 10.000 mẫu phân tầng chuẩn hóa.
  * Xây dựng pipeline tiền xử lý tiếng Việt dùng chung [`nlp_utils.py`](./nlp_utils.py).
  * Huấn luyện mô hình **Linear SVM** (`3b_train_svm.py`): BoW baseline, TF-IDF đơn baseline, và FeatureUnion nâng cấp 3 nhánh đặc trưng (Word TF-IDF + Char_wb 3-5 + Toxic Lexicon) kết hợp Cost-Sensitive Weighting (`class_weight={0: 1, 1: 15}`).
  * Đạt kết quả xuất sắc: **Toxic Precision = 75.00%**, **Toxic F1 = 0.6412**, **Macro F1 = 0.8145**, **PR-AUC = 0.7197** trên tập Test.

* **Thành viên 2: NLP Preprocessing & Naive Bayes Specialist:**
  * Huấn luyện mô hình **Multinomial Naive Bayes** (`3a_train_naive_bayes.py`) trên BoW và TF-IDF.
  * Tinh chỉnh tham số làm mịn Laplace $\alpha$ và dò ngưỡng quyết định tối ưu trên Cross-Validation.
  * Đạt kết quả TF-IDF Naive Bayes: **Toxic Recall = 58.67%**, **Toxic F1 = 0.5207**, **PR-AUC = 0.5669**.

* **Thành viên 3: Deep Learning Specialist (PhoBERT Contextual Transformer):**
  * Xây dựng mô hình Transformer hiện đại dựa trên backbone **VinAI PhoBERT** (`vinai/phobert-base-v2`, 135M tham số) (`3c_train_phobert.py`).
  * Thực hiện **End-to-End Fine-Tuning** 3 epochs trên GPU T4 với chiến lược Weighted Cross-Entropy Loss (`TV3_PhoBERT_FineTuning_Colab.ipynb`).
  * Đạt kết quả dẫn đầu về độ bao quát: **Toxic Recall = 60.00% (bắt đúng 45/75 câu toxic)**, **Toxic Precision = 62.50%**, **Toxic F1 = 0.6122**, **Macro F1 = 0.7987**, **PR-AUC = 0.6357**.

* **Thành viên 4: Evaluation Lead & Statistical Analysis:**
  * Thực hiện đánh giá kiểm thử độc lập cả 6 mô hình trên tập Test chuẩn vàng (`4_model_evaluation.py`).
  * Phân tích kiểm định ý nghĩa thống kê **McNemar Test** ($p = 0.194$, chứng minh sự tiệm cận và bổ trợ giữa SVM và PhoBERT).
  * Xuất báo cáo trực quan hóa 300 DPI: ma trận nhầm lẫn 6 mô hình, đồ thị đối sánh Macro F1 & Recall, bảng số liệu CSV và trích xuất 141 ca lỗi phân tích chuyên sâu.

---

## 3. BẢNG TỔNG HỢP KẾT QUẢ ĐỐI SÁNH CÁC MÔ HÌNH (TẬP TEST 2.000 MẪU)

| STT | Thành viên | Mô hình phân loại | Accuracy | Precision (Toxic) | Recall (Toxic) | F1-Score (Toxic) | Macro F1 | PR-AUC | TP / 75 | FP |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **TV1** | **Word TF-IDF + Char n-gram + Lexicon + Linear SVM** | **97.65%** | **75.00%** | 56.00% | **0.6412** | **0.8145** | **0.7197** | 42 | 14 |
| 2 | **TV3** | **PhoBERT (vinai/phobert-base-v2, Fine-Tuned GPU T4)** | **97.15%** | **62.50%** | **60.00%** | **0.6122** | **0.7987** | **0.6357** | **45** | **27** |
| 3 | **TV2** | **TF-IDF + Naive Bayes (MultinomialNB)** | 95.95% | 46.81% | 58.67% | **0.5207** | 0.7498 | 0.5669 | 44 | 50 |
| 4 | TV1 | TF-IDF đơn + Linear SVM | 96.85% | 62.50% | 40.00% | 0.4878 | 0.7358 | 0.5232 | 30 | 18 |
| 5 | TV2 | BoW + Naive Bayes (MultinomialNB) | 95.45% | 41.49% | 52.00% | 0.4615 | 0.7189 | 0.4252 | 39 | 55 |
| 6 | TV1 | BoW + Linear SVM | 96.90% | 66.67% | 34.67% | 0.4561 | 0.7201 | 0.4489 | 26 | 13 |

---

## 4. HƯỚNG DẪN CÀI ĐẶT & CHẠY THỬ NGHIỆM

### Cài đặt môi trường:
```bash
git clone https://github.com/KTD1108/BTL_NLP_DMX_ToxicClassification.git
cd BTL_NLP_DMX_ToxicClassification

# Cài đặt thư viện cần thiết
pip install scikit-learn pandas numpy matplotlib seaborn pyvi scipy transformers torch
```

### Các lệnh chạy chính:
```bash
# 1. Tái lập bộ dữ liệu chuẩn vàng 10.000 mẫu
python build_reconstructed_dataset.py

# 2. Huấn luyện mô hình Naive Bayes (Thành viên 2)
python 3a_train_naive_bayes.py

# 3. Huấn luyện mô hình Linear SVM (Thành viên 1)
python 3b_train_svm.py

# 4. Huấn luyện / Đánh giá PhoBERT (Thành viên 3)
python 3c_train_phobert.py

# 5. Đánh giá toàn diện, vẽ đồ thị đối sánh và ma trận nhầm lẫn (Thành viên 4)
python 4_model_evaluation.py
```

---

## 5. CẤU TRÚC THƯ MỤC REPOSITORY

```text
├── data/                             # Thư mục dữ liệu chuẩn vàng
│   ├── train.csv                     # Tập Train (7.000 mẫu)
│   ├── val.csv                       # Tập Validation (1.000 mẫu)
│   ├── test.csv                      # Tập Test chuẩn vàng (2.000 mẫu)
│   └── dmx_reviews_10k_raw.csv       # 10.000 bình luận gốc
├── data_reconstructed/               # Thư mục kiểm toán đối chiếu tái cấu trúc
├── results/                          # Toàn bộ kết quả đầu ra
│   ├── predictions/                  # Các file dự đoán chuẩn hóa của 4 thành viên
│   │   ├── nb_bow.csv & _info.json
│   │   ├── nb_tfidf.csv & _info.json
│   │   ├── svm_bow.csv & _info.json
│   │   ├── svm_tfidf_single.csv & _info.json
│   │   ├── svm_tfidf.csv & _info.json
│   │   └── phobert.csv & _info.json
│   ├── 5_confusion_matrices.png      # Lưới ma trận nhầm lẫn 6 mô hình (300 DPI)
│   ├── 6_model_comparison.png        # Đồ thị so sánh Macro F1 & Recall (300 DPI)
│   ├── tv4_model_comparison.csv      # Bảng số liệu đối sánh chính thức
│   └── tv4_error_cases.csv           # 141 trường hợp đoán sai để phân tích lỗi
├── 1_sampling_and_eda.py             # Script EDA và phân tích dữ liệu ban đầu
├── 3a_train_naive_bayes.py           # Mã nguồn Thành viên 2 (Naive Bayes)
├── 3b_train_svm.py                   # Mã nguồn Thành viên 1 (Linear SVM)
├── 3c_train_phobert.py               # Mã nguồn Thành viên 3 (PhoBERT Transformer)
├── 4_model_evaluation.py             # Mã nguồn Thành viên 4 (Đánh giá & Thống kê)
├── TV3_PhoBERT_FineTuning_Colab.ipynb# Notebook Google Colab GPU T4 huấn luyện PhoBERT
├── nlp_utils.py                      # Module tiện ích tiền xử lý tiếng Việt dùng chung
├── build_reconstructed_dataset.py    # Script tái cấu trúc dữ liệu không rò rỉ
└── raw.zip                           # Dữ liệu nguồn ban đầu
```
