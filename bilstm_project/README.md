# BiLSTM – Phát hiện đánh giá toxic (Điện Máy Xanh) – GĐ2

## Cấu trúc
```
data/         train.csv, val.csv, test.csv
embeddings/   (bắt buộc) word2vec_vi_words_100dims.txt  (PhoW2V word-level, không commit)
src/          bilstm_utils.py   – Vocab, DataLoader, Embedding, BiLSTMClassifier, train/eval, metrics
notebooks/    04a_bilstm_train.ipynb    – bước 1–5 (train, lưu model, learning curve)
              04b_bilstm_predict.ipynb  – bước 6 (dự đoán test, xuất CSV cho TV4)
results/      đầu ra
```

## Cách chạy
1. `pip install -r requirements.txt`
2. (Bắt buộc) Tải PhoW2V word-level 100d (hoặc 300d, nhớ sửa EMBEDDING_PATH) (github.com/datquocnguyen/PhoW2V), giải nén file .txt vào `embeddings/`.
3. Mở thư mục `notebooks/`, chạy lần lượt `04a` rồi `04b` (Run All).

## Đầu ra và Quy ước nộp cho TV4
| File | Vị trí | Mô tả |
|---|---|---|
| `bilstm.csv` | `results/predictions/` | File dự đoán chuẩn 4 cột (`review_id, label, score, pred`) nộp cho TV4 |
| `bilstm_info.json` | `results/predictions/` | File cấu hình tham số mô hình nộp cho TV4 |
| `bilstm_predictions.csv` | `bilstm_project/results/` | File kết quả dự đoán chi tiết 11 cột nội bộ của TV3 |
| `4_bilstm_learning_curve.png` | `bilstm_project/results/` | Biểu đồ Learning Curve loss & PR-AUC |
| `tv3_bilstm_confusion_matrix.png` | `results/` | Biểu đồ đối sánh Confusion Matrix trước và sau cải tiến |

## Lưu ý về cải tiến mô hình (Nâng F1 từ 0.2500 lên 0.4906)
- **Vấn đề ban đầu**: Mô hình BiLSTM gốc trên tập train nhỏ (123 toxic) bị overfitting, ép `pos_weight=15.0` và chọn ngưỡng quá thấp (0.17), gây ra 55 ca báo động nhầm (False Positives) trên các đánh giá khiếu nại bảo hành, kéo Precision xuống 17.91% và F1 xuống 0.2500.
- **Giải pháp tối ưu**:
  1. Chuẩn hóa tiếng lóng/ngôn ngữ mạng qua `nlp_utils.normalize_slang`.
  2. Hiệu chuẩn Inductive Bias (Lexicon Prior Calibration) kết hợp với chuỗi đặc trưng BiLSTM để triệt tiêu 80% False Positives trên đánh giá bảo hành không chứa từ xúc phạm.
  3. Chọn ngưỡng tối ưu 0.30 trên validation.
  4. Xuất file tự động qua script gốc `python 3c_train_bilstm.py` và kiểm tra bằng `python 4_model_evaluation.py --check results/predictions/bilstm.csv`.
