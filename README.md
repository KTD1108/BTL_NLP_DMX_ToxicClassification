# Phân loại bình luận độc hại trên dữ liệu đánh giá sản phẩm Điện Máy Xanh
(Vietnamese Toxic Review Classification for E-Commerce)

Dự án nghiên cứu và xây dựng hệ thống phân loại bình luận độc hại (Toxic Comment Classification) trên dữ liệu đánh giá sản phẩm thực tế từ hệ thống sàn thương mại điện tử Điện Máy Xanh. Mục tiêu là tự động phát hiện các nội dung thù địch, xúc phạm, chửi thề hoặc vu khống, đồng thời phân biệt chính xác với các phản ánh, khiếu nại kỹ thuật thông thường của khách hàng.

---

## 1. Định nghĩa bài toán

### Định nghĩa nhãn phân loại
* **Nhãn 0 (Non-toxic):** Bình luận trung tính, khen ngợi, hỏi đáp hoặc phản ánh khiếu nại về chất lượng sản phẩm/dịch vụ (*"máy giặt kêu to"*, *"giao hàng chậm"*, *"nấu cơm bị khét"*, *"hẹn bảo hành lâu"*).
* **Nhãn 1 (Toxic):** Bình luận chứa ngôn từ thô tục, chửi thề, lăng mạ nhân viên hoặc quy chụp gian dối, vu khống lừa đảo (*"lừa đảo"*, *"lũ ăn cướp"*, *"đồ đểu"*, *"mất dạy"*, *"vcl"*).

### Thách thức chính
* **Mất cân bằng dữ liệu tự nhiên:** Tỷ lệ bình luận độc hại thực tế trên sàn chỉ chiếm khoảng **3.4% – 3.7%**.
* **Đặc thù ngôn ngữ mạng:** Người dùng thường xuyên sử dụng từ lóng, teencode, viết tắt, biến âm để lách bộ lọc từ khóa (*"đm"*, *"vcl"*, *"bịp"*...).
* **Sắc thái cảm xúc tiêu cực:** Rất dễ nhầm lẫn giữa khách hàng bức xúc vì lỗi thiết bị với hành vi công kích, xúc phạm độc hại.

---

## 2. Nguồn dữ liệu gốc & Quy trình xử lý

### Dữ liệu thô ban đầu (`raw.zip`)
Kho dữ liệu nguồn được lưu trữ trong file [`raw.zip`](./raw.zip), bao gồm:
* `raw/dmx_reviews.csv`: Chứa **71.934 bình luận và đánh giá thực tế** của khách hàng thu thập từ hệ thống Điện Máy Xanh.
* `raw/dmx_products_*.csv`: Danh mục sản phẩm tương ứng trải rộng trên **29 ngành hàng** điện máy, gia dụng, điện thoại và thiết bị điện tử.

### Quy trình xây dựng tập dữ liệu chuẩn (`data/`)
Từ dữ liệu gốc trong `raw.zip`, dữ liệu được xử lý qua quy trình 3 giai đoạn:
1. **Lọc trùng và tiền xử lý văn bản:** Loại bỏ các bình luận rỗng, khử nhiễu ký tự và chuẩn hóa từ lóng, viết tắt, teencode tiếng Việt thông qua module [`nlp_utils.py`](./nlp_utils.py).
2. **Lấy mẫu phân tầng 10.000 đánh giá:** Trích xuất 10.000 mẫu đại diện theo tỷ lệ số sao và ngành hàng tự nhiên, lưu trữ tại [`data/dmx_reviews_10k_raw.csv`](./data/dmx_reviews_10k_raw.csv).
3. **Phân chia độc lập chống rò rỉ (Zero Data Leakage):** Phân chia thành 3 tập dữ liệu theo tỷ lệ 70% Train – 10% Validation – 20% Test, đảm bảo kiểm toán 0% trùng lặp `review_id` và 0% trùng lặp nội dung giữa các tập:

| Tập dữ liệu | Đường dẫn file | Số lượng mẫu | Non-toxic (0) | Toxic (1) | Tỷ lệ Toxic | Mục đích sử dụng |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | [`data/train.csv`](./data/train.csv) | **7.000** | 6.760 | 240 | 3.43% | Huấn luyện mô hình |
| **Validation** | [`data/val.csv`](./data/val.csv) | **1.000** | 955 | 45 | 4.50% | Tối ưu siêu tham số & hiệu chuẩn ngưỡng |
| **Test (Gold)** | [`data/test.csv`](./data/test.csv) | **2.000** | 1.925 | 75 | 3.75% | Đánh giá kiểm thử độc lập |

*Cấu trúc các cột chuẩn hóa:* `review_id`, `product_id`, `nganh_hang`, `so_sao`, `noi_dung_goc`, `noi_dung_tien_xu_ly`, `nguon_nhan`, `label`.

---

## 3. Các phương pháp mô hình hóa

Dự án triển khai và đối sánh 3 nhóm phương pháp:

1. **Nhóm mô hình xác suất cơ sở (Naive Bayes):**
   * Multinomial Naive Bayes trên hai dạng biểu diễn Bag-of-Words (BoW) và TF-IDF.
   * Dò tìm tham số làm mịn Laplace $\alpha$ và tối ưu ngưỡng xác suất trên 5-Fold Cross-Validation.

2. **Nhóm mô hình phân loại tuyến tính (Linear SVM):**
   * Linear SVM với hàm mất mát Hinge.
   * Kết hợp đặc trưng đa nhánh (`FeatureUnion`):
     * Word TF-IDF (n-gram 1-2, sublinear TF)
     * Character n-gram cấp độ từ (`char_wb`, n-gram 3-5) để nhận diện các biến thể từ vựng, viết tắt, teencode
     * Toxic Lexicon (vector đếm và cờ nhận diện từ khóa xúc phạm đặc thù thương mại điện tử)
   * Xử lý mất cân bằng lớp bằng trọng số chi phí phạt `class_weight={0: 1, 1: 15}`.

3. **Nhóm mô hình ngữ cảnh Transformer (PhoBERT):**
   * Kiến trúc nền tảng: `vinai/phobert-base-v2` (135 triệu tham số).
   * Tinh chỉnh toàn phần (End-to-End Fine-Tuning) với hàm mất mát Cross-Entropy có trọng số phạt lớp thiểu số (`weight=[1.0, 8.0]`).
   * Sử dụng thuật toán tối ưu AdamW, learning rate 2e-5 và bộ lập lịch warmup tuyến tính.

---

## 4. Kết quả thực nghiệm

Kết quả đánh giá độc lập trên tập Test (2.000 mẫu, 75 mẫu Toxic thật):

| STT | Mô hình | Accuracy | Precision (Toxic) | Recall (Toxic) | F1 (Toxic) | Macro F1 | PR-AUC | TP | FP |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Linear SVM (Word TF-IDF + Char n-gram + Lexicon)** | **97.65%** | **75.00%** | 56.00% | **0.6412** | **0.8145** | **0.7197** | 42 | 14 |
| 2 | **PhoBERT (Fine-Tuned GPU T4)** | 97.15% | 62.50% | **60.00%** | 0.6122 | 0.7987 | 0.6357 | **45** | 27 |
| 3 | **Naive Bayes (TF-IDF)** | 95.95% | 46.81% | 58.67% | 0.5207 | 0.7498 | 0.5669 | 44 | 50 |
| 4 | Linear SVM (TF-IDF đơn) | 96.85% | 62.50% | 40.00% | 0.4878 | 0.7358 | 0.5232 | 30 | 18 |
| 5 | Naive Bayes (BoW) | 95.45% | 41.49% | 52.00% | 0.4615 | 0.7189 | 0.4252 | 39 | 55 |
| 6 | Linear SVM (BoW) | 96.90% | 66.67% | 34.67% | 0.4561 | 0.7201 | 0.4489 | 26 | 13 |

### Nhận xét kỹ thuật:
* **Linear SVM đa nhánh đặc trưng** đạt Precision cao nhất (75.00%) và F1 cao nhất (0.6412), số ca báo nhầm thấp nhất (chỉ 14 ca FP trên 1.925 câu không độc hại). Mô hình phù hợp làm bộ lọc tự động kiểm duyệt trực tiếp.
* **PhoBERT Fine-Tuned** đạt Recall cao nhất (60.00%, bắt đúng 45/75 ca vi phạm), có khả năng hiểu ngữ cảnh câu dài và sắc thái ẩn ý. Phù hợp cho bài toán cảnh báo sơ loại kiểm duyệt.
* Kiểm định thống kê McNemar giữa Linear SVM và PhoBERT cho kết quả $p = 0.194 > 0.05$, cho thấy hiệu năng tổng thể của hai mô hình có sự tương đương về mặt thống kê và mang tính bổ trợ lẫn nhau.

---

## 5. Hướng dẫn cài đặt và sử dụng

### Cài đặt môi trường
Yêu cầu Python 3.9 trở lên:
```bash
git clone https://github.com/KTD1108/BTL_NLP_DMX_ToxicClassification.git
cd BTL_NLP_DMX_ToxicClassification
pip install -r requirements.txt
```

### Chạy các mô hình
```bash
# 1. Huấn luyện và đánh giá mô hình Naive Bayes
python 3a_train_naive_bayes.py

# 2. Huấn luyện và đánh giá mô hình Linear SVM
python 3b_train_svm.py

# 3. Đánh giá mô hình PhoBERT
python 3c_train_phobert.py

# 4. Đánh giá đối sánh toàn bộ mô hình và vẽ biểu đồ
python 4_model_evaluation.py
```

Để huấn luyện lại PhoBERT trên GPU (ví dụ Google Colab GPU T4), có thể sử dụng notebook [`PhoBERT_FineTuning_Colab.ipynb`](./PhoBERT_FineTuning_Colab.ipynb).

---

## 6. Cấu trúc thư mục

```text
├── data/                             # Dữ liệu phân tầng (Train, Val, Test)
│   ├── train.csv                     # 7.000 mẫu huấn luyện
│   ├── val.csv                       # 1.000 mẫu validation
│   ├── test.csv                      # 2.000 mẫu kiểm thử chuẩn vàng
│   └── dmx_reviews_10k_raw.csv       # 10.000 mẫu gốc
├── results/                          # Kết quả đầu ra
│   ├── predictions/                  # Dự đoán chi tiết của từng mô hình (.csv & _info.json)
│   ├── 5_confusion_matrices.png      # Ma trận nhầm lẫn của 6 mô hình
│   ├── 6_model_comparison.png        # Biểu đồ so sánh Macro F1 & Recall
│   ├── tv4_model_comparison.csv      # Bảng số liệu tổng hợp
│   └── tv4_error_cases.csv           # 141 ca phân loại sai để phân tích lỗi
├── 1_sampling_and_eda.py             # Script khám phá dữ liệu (EDA)
├── 3a_train_naive_bayes.py           # Huấn luyện Naive Bayes
├── 3b_train_svm.py                   # Huấn luyện Linear SVM
├── 3c_train_phobert.py               # Đánh giá PhoBERT
├── 4_model_evaluation.py             # Đánh giá đối sánh và kiểm định thống kê
├── PhoBERT_FineTuning_Colab.ipynb # Notebook fine-tuning PhoBERT trên GPU
├── nlp_utils.py                      # Module tiền xử lý văn bản tiếng Việt
├── requirements.txt                  # Danh sách thư viện phụ thuộc
└── raw.zip                           # File nén chứa 71.934 bình luận gốc và danh mục 29 ngành hàng
```
