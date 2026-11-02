# BÀI TẬP LỚN 1: HỆ THỐNG PHÂN LOẠI BÌNH LUẬN ĐỘC HẠI (TOXIC VS NON-TOXIC)
## DỮ LIỆU ĐÁNH GIÁ SẢN PHẨM HỆ THỐNG ĐIỆN MÁY XANH

> **Học phần:** Xử lý ngôn ngữ tự nhiên / Khai phá dữ liệu văn bản (NLP / Text Mining)  
> **Đơn vị:** Khoa Công nghệ Thông tin 1 – Học viện Công nghệ Bưu chính Viễn thông (PTIT)  
> **Repository:** [https://github.com/KTD1108/BTL_NLP_DMX_ToxicClassification](https://github.com/KTD1108/BTL_NLP_DMX_ToxicClassification)  
> **Báo cáo Google Docs chính thức:** [Xem tại đây](https://docs.google.com/document/d/17pGPsIW-CI-ilbRXc8MWG0yqayHl31qRMqTeNiW9_60/edit)

---

## 1. TỔNG QUAN BÀI TOÁN & DỮ LIỆU CHUẨN HÓA

Đề tài giải quyết bài toán **Phân loại văn bản nhị phân có giám sát (Supervised Binary Text Classification)** trên các đánh giá của khách hàng từ 29 ngành hàng của hệ thống Điện Máy Xanh:
* **Nhãn 0 (Non-toxic - Bình thường):** Các bình luận khen ngợi, trung tính, hỏi đáp hoặc **phản ánh lỗi kỹ thuật sản phẩm thông thường** (*"máy giặt kêu to", "tủ lạnh không mát", "hơi đắt", "nấu cơm bị khét"*).
* **Nhãn 1 (Toxic - Độc hại):** Bình luận chứa ngôn từ thô tục, công kích, xúc phạm nhân viên bán hàng hoặc quy chụp gian dối, vu khống lừa đảo (*"bố láo", "lừa đảo", "ăn bớt 1m dây", "đồ đểu", "cạch mặt"*).

### Bộ 3 dữ liệu cốt lõi đã sẵn sàng (`data/`):
Toàn bộ dữ liệu đã được tiền xử lý chuẩn hóa qua pipeline [`nlp_utils.py`](./nlp_utils.py) và chia tách phân tầng (Stratified Split) với tỷ lệ cân đối tuyệt đối:

| Tập dữ liệu | File lưu trữ | Số lượng mẫu | Non-toxic (0) | Toxic (1) | Tỷ lệ Toxic | Nguồn gốc nhãn |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | [`data/train.csv`](./data/train.csv) | **6.048** | 5.925 | 123 | **2.03%** | Bán tự động + Lọc trùng |
| **Validation** | [`data/val.csv`](./data/val.csv) | **879** | 862 | 17 | **1.93%** | Dành riêng cho BiLSTM (Early Stopping/LR) |
| **Test (Gold)** | [`data/test.csv`](./data/test.csv) | **2.000** | 1.960 | 40 | **2.00%** | **Gán tay thủ công 4 thành viên (Chuẩn vàng)** |

*Cấu trúc 8 cột đồng nhất:* `['review_id', 'product_id', 'nganh_hang', 'so_sao', 'noi_dung_goc', 'noi_dung_tien_xu_ly', 'nguon_nhan', 'label']`.

---

## 2. PHÂN CÔNG NHIỆM VỤ NHÓM 4 THÀNH VIÊN

Kế hoạch chi tiết được quy định tại: 👉 [**`KE_HOACH_CHI_TIET_BTL_4_NGUOI.md`**](./KE_HOACH_CHI_TIET_BTL_4_NGUOI.md)

* **Thành viên 1: Kiều Tiến Đạt (Data Architect & SVM Specialist):**
  * Lấy mẫu phân tầng 10k từ kho dữ liệu gốc 72k dòng, xây dựng bộ 3 Train/Val/Test.
  * Xây dựng pipeline tiền xử lý tiếng Việt chung [`nlp_utils.py`](./nlp_utils.py).
  * Huấn luyện mô hình **Linear SVM** đối sánh 3 biến thể (`3b_train_svm.py`): BoW baseline, TF-IDF đơn baseline, và FeatureUnion nâng cấp 3 nhánh đặc trưng (Word TF-IDF + Char_wb TF-IDF + Lexicon) kết hợp GridSearchCV 5-Folds (`C=0.1`, `class_weight={0: 1, 1: 15}`). Đạt **Toxic F1 = 0.4000**, **Macro F1 = 0.6950**, **PR-AUC = 0.4061** trên tập Test.
  * Xuất chuẩn hóa 3 cặp file dự đoán vào `results/predictions/` (`svm_bow`, `svm_tfidf_single`, `svm_tfidf`) theo quy ước của TV4.
  * Báo cáo chi tiết: [`BAO_CAO_TIEN_DO_VA_KET_QUA_TV1.md`](./BAO_CAO_TIEN_DO_VA_KET_QUA_TV1.md).

* **Thành viên 2 (NLP Preprocessing & Naive Bayes Specialist):**
  * Sử dụng [`nlp_utils.py`](./nlp_utils.py) và bộ dữ liệu trong `data/`.
  * Viết mã nguồn huấn luyện mô hình **Multinomial Naive Bayes** (`3a_train_naive_bayes.py`) trên BoW và TF-IDF.
  * Tinh chỉnh tham số làm mịn Laplace $\alpha$ và ngưỡng quyết định (Threshold Tuning).

* **Thành viên 3 (Deep Learning Specialist - BiLSTM PyTorch):**
  * Sử dụng `data/train.csv` (huấn luyện), `data/val.csv` (theo dõi validation loss, early stopping), `data/test.csv` (đánh giá cuối).
  * Viết mã nguồn mạng nơ-ron hồi quy hai chiều **BiLSTM kết hợp Word Embeddings** (`3c_train_bilstm.py`).
  * Xử lý mất cân bằng dữ liệu bằng hàm mất mát có trọng số `pos_weight`.

* **Thành viên 4 (Evaluation Lead & Report Author):**
  * Đánh giá đối sánh 5 mô hình trên cùng tập Test chuẩn vàng `data/test.csv`.
  * Tổng hợp kết quả dự đoán (đã có kết quả của SVM tại `results/tv1_svm_test_predictions.csv`).
  * Script đánh giá chung [`4_model_evaluation.py`](./4_model_evaluation.py). **TV1, TV2, TV3 nộp file dự đoán theo quy ước tại 👉 [`QUY_UOC_FILE_DU_DOAN.md`](./QUY_UOC_FILE_DU_DOAN.md).**
  * Hoàn thiện Báo cáo tổng kết và Notebook tổng thể (`BTL_NLP_DienMayXanh.ipynb`).

---

## 3. HƯỚNG DẪN CÀI ĐẶT & CHẠY MÃ NGUỒN

### Cài đặt môi trường:
```bash
# Clone repository
git clone https://github.com/KTD1108/BTL_NLP_DMX_ToxicClassification.git
cd BTL_NLP_DMX_ToxicClassification

# Cài đặt thư viện phụ thuộc
pip install scikit-learn pandas numpy matplotlib seaborn pyvi openpyxl tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Chạy kiểm tra mô hình Linear SVM (Thành viên 1):
```bash
python 3b_train_svm.py
```
Kết quả sẽ tự động lưu vào thư mục `results/`:
* `results/tv1_svm_metrics.csv`: Bảng số liệu chi tiết Accuracy, Precision, Recall, Toxic F1, Macro F1.
* `results/tv1_svm_test_predictions.csv`: Kết quả dự đoán nhãn trên 2.000 mẫu Test.
* `results/tv1_svm_confusion_matrices.png`: Biểu đồ ma trận nhầm lẫn ở 2 ngưỡng (0.0 và -0.10).
* `results/tv1_svm_top_features.png`: Biểu đồ Top 15 đặc trưng phân tách có trọng số lớn nhất.

---

## 4. CẤU TRÚC THƯ MỤC REPOSITORY

```text
├── data/
│   ├── train.csv                     # Tập Train chuẩn hóa (6.048 mẫu)
│   ├── val.csv                       # Tập Validation độc lập (879 mẫu)
│   ├── test.csv                      # Tập Test chuẩn vàng (2.000 mẫu gán tay 4 TV)
│   ├── tap_test_2000_can_gan_tay.csv # Lịch sử gán nhãn 4 thành viên
│   └── tap_test_2000_can_gan_tay.xlsx
├── results/
│   ├── predictions/                  # Thư mục nộp file dự đoán cho TV4
│   │   ├── svm_bow.csv & _info.json
│   │   ├── svm_tfidf_single.csv & _info.json
│   │   └── svm_tfidf.csv & _info.json
│   ├── tv1_svm_metrics.csv           # Bảng điểm đối sánh SVM trên tập Test
│   ├── tv1_svm_test_predictions.csv  # Nhãn dự đoán SVM chi tiết trên tập Test
│   ├── tv1_svm_confusion_matrices.png# Biểu đồ Ma trận nhầm lẫn SVM
│   ├── tv1_svm_top_features.png      # Biểu đồ Top trọng số đặc trưng SVM
│   ├── tv4_model_comparison.csv      # Bảng đối sánh tổng hợp của TV4
│   └── human_vs_ai_agreement.csv     # Phân tích đối sánh Con người vs AI (2.000 mẫu)
├── nlp_utils.py                      # Pipeline tiền xử lý văn bản tiếng Việt dùng chung
├── build_full_datasets.py            # Kịch bản chia tách bộ 3 Train/Val/Test
├── 3b_train_svm.py                   # Huấn luyện & đánh giá đối sánh Linear SVM (TV1)
├── 4_model_evaluation.py             # Script đánh giá chung & vẽ biểu đồ đối sánh (TV4)
├── QUY_UOC_FILE_DU_DOAN.md           # Quy ước định dạng nộp file dự đoán cho TV4
├── KE_HOACH_CHI_TIET_BTL_4_NGUOI.md  # Kế hoạch chi tiết phân công 4 thành viên
├── BAO_CAO_TIEN_DO_VA_KET_QUA_TV1.md # Báo cáo tiến độ & kết quả chuyên sâu TV1
└── README.md                         # Tài liệu hướng dẫn chung
```
