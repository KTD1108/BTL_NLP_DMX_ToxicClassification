# Phân loại bình luận độc hại trên dữ liệu đánh giá sản phẩm Điện Máy Xanh
(Vietnamese Toxic Review Classification for E-Commerce)

Dự án nghiên cứu và xây dựng hệ thống phân loại bình luận độc hại (Toxic Comment Classification) trên dữ liệu đánh giá sản phẩm thực tế từ hệ thống sàn thương mại điện tử Điện Máy Xanh. Mục tiêu là tự động phát hiện các nội dung thù địch, xúc phạm, chửi thề hoặc vu khống, đồng thời phân biệt chính xác với các phản ánh, khiếu nại kỹ thuật thông thường của khách hàng.

---

## 1. Định nghĩa bài toán

### Định nghĩa nhãn phân loại
* **Nhãn 0 (Non-toxic):** Bình luận khen ngợi, hỏi đáp, hoặc chê, báo lỗi, phàn nàn bằng giọng kể sự việc (*"máy giặt kêu to"*, *"giao hàng chậm, rất thất vọng"*, *"hẹn bảo hành lâu"*); câu chỉ nghi ngờ, chưa khẳng định bị lừa.
* **Nhãn 1 (Toxic):** Bình luận chửi thề, thô tục, xúc phạm; cáo buộc gian dối, lừa đảo; miệt thị, hạ thấp; kêu gọi người khác né hoặc tẩy chay cửa hàng/thương hiệu; hoặc thể hiện thái độ thù địch rõ rệt (*"lừa đảo"*, *"mất dạy"*, *"vcl"*, *"mọi người né ra nha"*, *"thợ ăn gian ống đồng"*, *"làm ăn tệ bạc"*).

Luật chi tiết, ví dụ ranh giới và số liệu độ tin cậy nhãn: [`LUAT_GAN_NHAN.md`](./LUAT_GAN_NHAN.md).

### Thách thức chính
* **Mất cân bằng dữ liệu:** Ước tính trên toàn bộ `raw.zip`, bình luận độc hại chỉ chiếm khoảng **1%**; bộ dữ liệu được lấy dư mẫu độc hại thật lên khoảng **4%** để đánh giá ổn định.
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
| **Train** | [`data/train.csv`](./data/train.csv) | **7.011** | 6.710 | 301 | 4.29% | Huấn luyện mô hình |
| **Validation** | [`data/val.csv`](./data/val.csv) | **997** | 952 | 45 | 4.51% | Tối ưu siêu tham số & hiệu chuẩn ngưỡng |
| **Test** | [`data/test.csv`](./data/test.csv) | **1.992** | 1.907 | 85 | 4.27% | Đánh giá kiểm thử độc lập |

*Cấu trúc các cột chuẩn hóa:* `review_id`, `product_id`, `nganh_hang`, `so_sao`, `noi_dung_goc`, `noi_dung_tien_xu_ly`, `nguon_nhan`, `label`.

*Gán nhãn:*
* **Train / Validation:** Gemini gán theo [`LUAT_GAN_NHAN.md`](./LUAT_GAN_NHAN.md) cho toàn bộ câu 1–3 sao, người rà soát chốt mọi câu đổi nhãn; câu 4–5 sao mặc định non-toxic.
* **Test:** 4 thành viên gán mù lại toàn bộ 1.992 câu (không xem nhãn cũ, nhãn Gemini), mỗi người ~648 câu, trong đó 200 câu chung để đo đồng thuận. Đồng thuận giữa 4 người: **Fleiss' κ = 0,68**, đồng ý thuần 96,7%.

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
   * Tinh chỉnh toàn phần (End-to-End Fine-Tuning) với hàm mất mát Weighted Cross-Entropy có trọng số phạt thích ứng với độ mất cân bằng lớp (`weight=[1.0, 10.0]`).
   * Sử dụng thuật toán tối ưu AdamW (learning rate $2 \times 10^{-5}$, weight decay 0.01) kết hợp bộ lập lịch warmup tuyến tính và cơ chế Early Stopping (patience = 2).
   * Chạy huấn luyện đa seed (seeds: 42, 13, 7) để đảm bảo tính khách quan và chọn mô hình có PR-AUC trên validation cao nhất.

---

## 4. Kết quả thực nghiệm

> **Lưu ý:** Nhãn test đã được gán mù lại (85 mẫu Toxic, đổi 45 nhãn so với bản cũ 80 mẫu Toxic). Toàn bộ 6 mô hình đã được đánh giá lại trên nhãn mới.

Kết quả đánh giá độc lập trên tập Test (1.992 mẫu):

| STT | Mô hình | Accuracy | Precision (Toxic) | Recall (Toxic) | F1 (Toxic) | Macro F1 | PR-AUC | TP | FP | FN |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **PhoBERT (`vinai/phobert-base-v2`)** | **97.04%** | **67.57%** | **58.82%** | **0.6289** | **0.8068** | **0.6484** | **50** | **24** | **35** |
| 2 | Linear SVM (TF-IDF đa nhánh) | 96.69% | 63.77% | 51.76% | 0.5714 | 0.7771 | 0.6026 | 44 | 25 | 41 |
| 3 | Linear SVM (TF-IDF đơn) | 96.29% | 58.46% | 44.71% | 0.5067 | 0.7437 | 0.5465 | 38 | 27 | 47 |
| 4 | Naive Bayes (TF-IDF) | 93.67% | 33.86% | 50.59% | 0.4057 | 0.6861 | 0.3777 | 43 | 84 | 42 |
| 5 | Linear SVM (BoW) | 96.03% | 55.56% | 35.29% | 0.4317 | 0.7056 | 0.4334 | 30 | 24 | 55 |
| 6 | Naive Bayes (BoW) | 93.88% | 33.63% | 44.71% | 0.3838 | 0.6758 | 0.2900 | 38 | 75 | 47 |

### Nhận xét kỹ thuật:
* **PhoBERT (`vinai/phobert-base-v2`)** vẫn dẫn đầu mọi chỉ số (seed 7, chọn theo PR-AUC trên val): F1 Toxic 0.6289, Macro F1 0.8068, PR-AUC 0.6484. Trung bình 3 seed: F1 Toxic 0.639 ± 0.020. So với nhãn test cũ (F1 0.7901), điểm giảm chủ yếu do nhãn: trên val không đổi F1 gần như giữ nguyên (0.71 → 0.72), và trên 45 câu đổi nhãn PhoBERT đoán trùng nhãn cũ khoảng 70%, tức mô hình đã học theo cách gán của nhãn train (Gemini + rà soát).
* **Kiểm định thống kê** (bootstrap 10.000 lần, McNemar, hiệu chỉnh Holm; xem `results/tv4_pairwise_tests.csv`): chênh lệch PhoBERT − SVM FeatureUnion là +0.058 F1 Toxic, CI đồng thời [−0.067, 0.188], p = 0.73 → **không có ý nghĩa thống kê** trên tập test hiện tại. PhoBERT và SVM FeatureUnion đều tốt hơn có ý nghĩa so với cả hai Naive Bayes.
* **Linear SVM đa nhánh đặc trưng (FeatureUnion)** trên nhãn mới tiếp tục là mô hình truyền thống tốt nhất: F1 Toxic đạt 0.5714, Macro F1 đạt 0.7771, Precision đạt 63.77%, bắt đúng 44/85 ca toxic và số ca báo nhầm thấp nhất trong các mô hình chạy trên CPU (chỉ 25 ca FP so với 75-84 ca của Naive Bayes).
* **Naive Bayes** có ưu thế tốc độ huấn luyện và suy luận cực nhanh. Trên nhãn test gán mù, NB TF-IDF đạt Recall 50.59% (43/85) và F1 Toxic 0.4057, giảm so với nhãn cũ (F1 0.4541), nhiều khả năng do nhãn test gán mù không còn bám theo cách gán của Gemini như nhãn train; tỷ lệ báo nhầm (84 FP) vẫn cao hơn SVM và PhoBERT.

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

# 3. Đánh giá đối sánh toàn bộ mô hình và vẽ biểu đồ
python 4_model_evaluation.py
```

Để huấn luyện và tinh chỉnh mô hình **PhoBERT** trên Kaggle (GPU T4 hoặc P100), sử dụng notebook chuẩn của dự án: [`phobert.ipynb`](./phobert.ipynb). Notebook sẽ tự động lưu kết quả dự đoán đạt chuẩn vào `results/predictions/phobert.csv` và `phobert_info.json`.

---

## 6. Cấu trúc thư mục

```text
├── data/                             # Dữ liệu phân tầng (Train, Val, Test)
│   ├── train.csv                     # 7.011 mẫu huấn luyện
│   ├── val.csv                       # 997 mẫu validation
│   ├── test.csv                      # 1.992 mẫu kiểm thử chuẩn vàng
│   └── dmx_reviews_10k_raw.csv       # 10.000 mẫu gốc
├── results/                          # Kết quả đầu ra
│   ├── predictions/                  # 6 file dự đoán chuẩn hóa (.csv & _info.json)
│   ├── phobert_v2/                   # Chi tiết đường học, metrics từng seed, báo cáo PhoBERT
│   ├── 5_confusion_matrices.png      # Ma trận nhầm lẫn của 6 mô hình
│   ├── 6_model_comparison.png        # Biểu đồ so sánh Macro F1 & Recall 6 mô hình
│   ├── tv4_model_comparison.csv      # Bảng số liệu tổng hợp đối sánh 6 mô hình
│   └── tv4_error_cases.csv           # 169 ca phân loại sai để phân tích lỗi
├── 1_sampling_and_eda.py             # Script khám phá dữ liệu (EDA)
├── 3a_train_naive_bayes.py           # Huấn luyện Naive Bayes
├── 3b_train_svm.py                   # Huấn luyện Linear SVM
├── 4_model_evaluation.py             # Đánh giá đối sánh và kiểm định thống kê
├── phobert.ipynb                     # Notebook huấn luyện và đánh giá PhoBERT (Kaggle)
├── LUAT_GAN_NHAN.md                  # Luật gán nhãn và độ tin cậy nhãn
├── nlp_utils.py                      # Module tiền xử lý văn bản tiếng Việt
├── requirements.txt                  # Danh sách thư viện phụ thuộc
└── raw.zip                           # File nén chứa 71.934 bình luận gốc và danh mục 29 ngành hàng
```
