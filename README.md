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
| **Test** | [`data/test.csv`](./data/test.csv) | **1.992** | 1.912 | 80 | 4.02% | Đánh giá kiểm thử độc lập |

*Cấu trúc các cột chuẩn hóa:* `review_id`, `product_id`, `nganh_hang`, `so_sao`, `noi_dung_goc`, `noi_dung_tien_xu_ly`, `nguon_nhan`, `label`.

*Gán nhãn:* Gemini gán theo [`LUAT_GAN_NHAN.md`](./LUAT_GAN_NHAN.md) cho toàn bộ câu 1–3 sao, người rà soát chốt mọi câu đổi nhãn; câu 4–5 sao mặc định non-toxic.

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

Kết quả đánh giá độc lập trên tập Test (1.992 mẫu, 80 mẫu Toxic):

| STT | Mô hình | Accuracy | Precision (Toxic) | Recall (Toxic) | F1 (Toxic) | Macro F1 | PR-AUC | TP | FP | FN |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **PhoBERT (`vinai/phobert-base-v2`)** | **98.29%** | **78.05%** | **80.00%** | **0.7901** | **0.8906** | **0.7580** | **64** | 18 | **16** |
| 2 | Linear SVM (TF-IDF đa nhánh) | 97.14% | 70.91% | 48.75% | 0.5778 | 0.7815 | 0.6495 | 39 | **16** | 41 |
| 3 | Linear SVM (TF-IDF đơn) | 96.69% | 60.61% | 50.00% | 0.5479 | 0.7654 | 0.5574 | 40 | 26 | 40 |
| 4 | Naive Bayes (TF-IDF) | 94.33% | 37.01% | 58.75% | 0.4541 | 0.7121 | 0.4530 | 47 | 80 | 33 |
| 5 | Linear SVM (BoW) | 96.23% | 54.55% | 37.50% | 0.4444 | 0.7125 | 0.4810 | 30 | 25 | 50 |
| 6 | Naive Bayes (BoW) | 94.53% | 37.17% | 52.50% | 0.4352 | 0.7032 | 0.3541 | 42 | 71 | 38 |

### Nhận xét kỹ thuật:
* **PhoBERT (`vinai/phobert-base-v2`)** đạt hiệu năng dẫn đầu tuyệt đối ở mọi chỉ số: Accuracy (98.29%), Precision (78.05%), Recall (80.00%), F1 Toxic (0.7901) và Macro F1 (0.8906). Nhờ kiến trúc Transformer tiền huấn luyện chuyên biệt cho tiếng Việt, PhoBERT hiểu sâu ngữ cảnh câu dài, từ lóng và sắc thái chê bai gián tiếp, bắt đúng 64/80 ca vi phạm và chỉ bỏ sót 16 ca.
* **Linear SVM đa nhánh đặc trưng (FeatureUnion)** là mô hình truyền thống tốt nhất với Precision đạt 70.91%, số ca báo nhầm thấp nhất (chỉ 16 ca FP trên 1.912 câu sạch). Đây là lựa chọn tối ưu về mặt chi phí tính toán khi cần triển khai nhẹ trên CPU.
* **Naive Bayes** có ưu thế tốc độ huấn luyện và suy luận cực nhanh, đạt Recall tương đối tốt (~58.75%) khi tinh chỉnh ngưỡng trên tập validation, nhưng tỷ lệ báo nhầm (FP) còn cao hơn so với SVM và PhoBERT.

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

Để huấn luyện và tinh chỉnh mô hình **PhoBERT** trên GPU (Google Colab hoặc Kaggle GPU T4), sử dụng notebook chuẩn của dự án: [`btl-nlpv2.ipynb`](./btl-nlpv2.ipynb). Notebook sẽ tự động lưu kết quả dự đoán đạt chuẩn vào `results/predictions/phobert.csv` và `phobert_info.json`.

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
├── btl-nlpv2.ipynb                   # Notebook huấn luyện và đánh giá PhoBERT (Colab/Kaggle/Local)
├── LUAT_GAN_NHAN.md                  # Luật gán nhãn và độ tin cậy nhãn
├── nlp_utils.py                      # Module tiền xử lý văn bản tiếng Việt
├── requirements.txt                  # Danh sách thư viện phụ thuộc
└── raw.zip                           # File nén chứa 71.934 bình luận gốc và danh mục 29 ngành hàng
```
