# BÀI TẬP LỚN 1: XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)
## CHỦ ĐỀ 2c: PHÂN LOẠI BÌNH LUẬN TOXIC VÀ NON-TOXIC TRÊN DỮ LIỆU ĐÁNH GIÁ SẢN PHẨM ĐIỆN MÁY XANH

---

## 📌 Giới thiệu dự án

Hệ thống phân loại tự động bình luận độc hại (**Toxic**) và bình luận thông thường (**Non-toxic**) được xây dựng phục vụ bài toán thực tế trong lĩnh vực Thương mại Điện tử và Bán lẻ điện máy tại Việt Nam.

Toàn bộ dữ liệu trong dự án được thu thập thực tế 100% từ hệ sinh thái đánh giá sản phẩm của **Điện Máy Xanh** (`dienmayxanh.com`), hoàn toàn không sử dụng dữ liệu giả lập (synthetic) hay các bộ dữ liệu mạng xã hội bên ngoài không tương thích miền (domain mismatch).

### Mục tiêu kỹ thuật
1. Thu thập và tiền xử lý tập dữ liệu đánh giá sản phẩm tiếng Việt quy mô lớn (**15.183 mẫu**).
2. Xây dựng quy chuẩn gán nhãn chuyên biệt cho thương mại điện tử:
   - **Toxic (1)**: Lăng mạ, chửi thề xúc phạm nhân viên/thợ kỹ thuật, vu khống lừa đảo/gian lận/trộm cắp, kích động tẩy chay thù địch.
   - **Non-toxic (0)**: Khen ngợi, góp ý trung tính, hoặc khiếu nại lỗi kỹ thuật/giao hàng chậm nhưng giữ ngôn từ chuẩn mực (kể cả đánh giá 1 sao).
3. Hiện thực hóa và so sánh có hệ thống giữa các phương pháp trích chọn đặc trưng cổ điển (**Bag of Words**, **TF-IDF**) kết hợp các thuật toán học máy (**Naive Bayes**, **SVM**) và mô hình học sâu hiện đại (**PhoBERT-style Contextual Transformer**).
4. Đảm bảo quy trình thực nghiệm nghiêm ngặt: Stratified Split $70:15:15$, ngăn chặn hoàn toàn rò rỉ dữ liệu (data leakage).

---

## 📂 Cấu trúc thư mục dự án

```text
btlvb1/
├── data/
│   ├── raw/
│   │   └── dmx_reviews_raw.csv           # 15.183 đánh giá thô cào từ Điện Máy Xanh
│   └── processed/
│       ├── dmx_reviews_processed.csv     # Dữ liệu sau chuẩn hóa & gán nhãn
│       ├── train.csv                     # Tập huấn luyện (10.627 mẫu - 70%)
│       ├── val.csv                       # Tập kiểm định (2.278 mẫu - 15%)
│       └── test.csv                      # Tập kiểm thử độc lập (2.278 mẫu - 15%)
├── notebooks/
│   └── Chu_De_2c_Toxic_DienMayXanh.ipynb # Jupyter Notebook toàn bộ pipeline đã thực thi
├── outputs/
│   ├── error_analysis.csv                # Phân tích định tính các mẫu dự đoán sai
│   ├── results_summary.csv               # Bảng tổng hợp số liệu đo lường 5 mô hình
│   ├── figures/                          # 5 biểu đồ trực quan hóa độ phân giải cao
│   │   ├── 01_eda_distribution.png       # Phân bố sao & nhãn theo 6 ngành hàng
│   │   ├── 02_word_length_distribution.png # Phân bố độ dài từ
│   │   ├── 03_model_comparison_metrics.png# Biểu đồ cột so sánh Accuracy, F1, AUC
│   │   ├── 04_confusion_matrices_grid.png# Ma trận nhầm lẫn 5 mô hình
│   │   └── 05_roc_curves.png             # Đường cong ROC & chỉ số AUC
│   └── models/                           # Trọng số mô hình và bộ vectorizer đã lưu
│       ├── bow_extractor.pkl
│       ├── tfidf_extractor.pkl
│       ├── best_traditional_svm_tfidf.pkl
│       └── phobert_contextual_transformer.pt
├── src/
│   ├── __init__.py
│   ├── crawler.py                        # Bộ cào dữ liệu tối ưu đa luồng từ API DMX
│   ├── preprocess.py                     # Chuẩn hóa Unicode NFC, teencode, tách từ tiếng Việt
│   ├── features.py                       # Trích xuất đặc trưng BoW và TF-IDF
│   ├── train_ml.py                       # Huấn luyện Naive Bayes & LinearSVC
│   ├── train_phobert.py                  # Huấn luyện PhoBERT-style Contextual Transformer
│   └── evaluate.py                       # Tính toán metrics, vẽ đồ thị & phân tích lỗi
├── .gitignore
├── requirements.txt                      # Các thư viện phụ thuộc
├── run_pipeline.py                       # Script thực thi toàn bộ luồng tự động
└── README.md
```

---

## 📊 Kết quả thực nghiệm trên tập Test (2.278 mẫu)

Tập Test độc lập gồm 2.076 mẫu Non-toxic (91,13%) và 202 mẫu Toxic (8,87%).

| STT | Phương pháp trích xuất | Mô hình | Accuracy | Precision (Toxic) | Recall (Toxic) | F1-Score (Toxic) | Macro-F1 | ROC-AUC | Thời gian |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Bag of Words (BoW) | Multinomial NB | 95.13% | 66.91% | 89.11% | 76.43% | 86.86% | 97.10% | 0.11s |
| 2 | Bag of Words (BoW) | SVM (LinearSVC) | 98.16% | 95.45% | 83.17% | 88.89% | 93.94% | 99.69% | 2.02s |
| 3 | TF-IDF (Word + Char) | Multinomial NB | 94.42% | 63.44% | 87.62% | 73.60% | 85.24% | 97.65% | 0.53s |
| **4** | **TF-IDF (Word + Char)** | **SVM (LinearSVC)** | **98.51%** | **96.15%** | **86.63%** | **91.15%** | **95.17%** | **99.76%** | **0.26s** |
| **5** | **Contextual Subword** | **PhoBERT Transformer** | **95.08%** | **66.42%** | **90.10%** | **76.47%** | **86.86%** | **98.35%** | **168.12s** |

### Nhận xét & Đánh giá chuyên sâu:
*   **Mô hình cân bằng và tối ưu vận hành thực tế**: **TF-IDF + LinearSVC** đạt hiệu năng vượt trội nhất với **Accuracy = 98,51%**, **Precision = 96,15%**, **F1-Score (Toxic) = 91,15%**, **Macro-F1 = 95,17%**, **ROC-AUC = 99,76%** cùng tốc độ suy luận cực nhanh (0,26 giây).
*   **Mô hình cảnh báo sớm & an toàn tối đa**: **PhoBERT-style Transformer** đạt **Recall (Toxic) = 90,10%** (phát hiện được 182/202 bình luận độc hại), đóng vai trò quan trọng như một màng lọc quét sâu cho bài toán kiểm duyệt nội dung.
*   **Tính toàn vẹn (Không rò rỉ dữ liệu - No Data Leakage)**: Việc chia tập `train/val/test` được thực hiện trước khi khớp vectorizer (`fit_transform` chỉ chạy trên tập Train). Tỉ lệ nhãn được phân tầng nghiêm ngặt (Stratified Sampling).

---

## 🚀 Hướng dẫn cài đặt và sử dụng

### 1. Yêu cầu hệ thống
*   Python 3.10 trở lên.
*   Hệ điều hành: Windows, macOS, hoặc Linux.

### 2. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 3. Chạy toàn bộ quy trình (Pipeline)
Để chạy lại toàn bộ quy trình tiền xử lý, trích xuất đặc trưng, huấn luyện 5 mô hình, sinh biểu đồ và trích xuất báo cáo:
```bash
python run_pipeline.py --no-live-crawl
```
*(Nếu muốn cào dữ liệu mới trực tiếp từ website Điện Máy Xanh, bỏ cờ `--no-live-crawl`).*

### 4. Khám phá Jupyter Notebook
Khởi chạy notebook tương tác để theo dõi trực quan từng bước:
```bash
jupyter notebook notebooks/Chu_De_2c_Toxic_DienMayXanh.ipynb
```

---

## 📑 Báo cáo nộp bài

Dự án đã biên soạn đầy đủ báo cáo học thuật theo quy chuẩn:
1.  **Định dạng Markdown**: [`report/BAO_CAO_BTL1_CHU_DE_2C.md`](report/BAO_CAO_BTL1_CHU_DE_2C.md)
2.  **Định dạng Microsoft Word (.docx)**: [`report/BAO_CAO_BTL1_CHU_DE_2C.docx`](report/BAO_CAO_BTL1_CHU_DE_2C.docx) (được định dạng chuẩn font Times New Roman, dãn dòng 1.2pt, căn lề 2cm, nhúng toàn bộ 5 biểu đồ đồ họa phân giải cao và bảng phân tích lỗi chi tiết).

---
*Dự án Bài tập lớn 1 - Xử lý ngôn ngữ tự nhiên (NLP) - Nhóm tác giả: Kiều Tiến Đạt.*
