# QUY ƯỚC FILE DỰ ĐOÁN TRÊN TẬP TEST (NỘP CHO TV4)

> **Áp dụng cho:** TV1 (SVM), TV2 (Naive Bayes), TV3 (BiLSTM).
> **Mục đích:** TV4 tự tính lại toàn bộ chỉ số (Accuracy, Precision, Recall, F1 Toxic, Macro F1, PR-AUC) cho cả 5 mô hình bằng **cùng một đoạn code** trong `4_model_evaluation.py`, để bảng so sánh trong báo cáo công bằng và nhất quán.
> Mỗi người vẫn giữ các file kết quả riêng của mình (metrics, biểu đồ...) như bình thường. File dưới đây là file **nộp thêm**.

---

## 1. Tên file và vị trí

Mỗi mô hình nộp **1 file CSV** trong thư mục `results/predictions/`, tên file là mã mô hình:

| Mã mô hình | Tên file | Mô hình | Người phụ trách |
| :--- | :--- | :--- | :--- |
| `nb_bow` | `results/predictions/nb_bow.csv` | BoW + Naive Bayes | TV2 |
| `nb_tfidf` | `results/predictions/nb_tfidf.csv` | TF-IDF + Naive Bayes | TV2 |
| `svm_bow` | `results/predictions/svm_bow.csv` | BoW + Linear SVM | TV1 |
| `svm_tfidf` | `results/predictions/svm_tfidf.csv` | TF-IDF + Linear SVM | TV1 |
| `bilstm` | `results/predictions/bilstm.csv` | Word Embeddings + BiLSTM | TV3 |

Nếu có thêm biến thể khác (ví dụ một mô hình cải tiến), đặt tên file theo dạng `<ma_mo_hinh>.csv` (chữ thường, không dấu, nối bằng `_`) và báo TV4. Script đánh giá tự đọc mọi file `.csv` trong thư mục này.

---

## 2. Cấu trúc file: đúng 4 cột

```text
review_id,label,score,pred
58896069,0,-0.4416,0
33055842,0,-1.2442,0
...
```

| Cột | Kiểu | Ý nghĩa |
| :--- | :--- | :--- |
| `review_id` | số nguyên | Lấy nguyên từ cột `review_id` của `data/test.csv`. |
| `label` | 0 / 1 | Nhãn thật, lấy nguyên từ cột `label` của `data/test.csv`. |
| `score` | số thực | Điểm liên tục của mô hình, **càng lớn càng nghiêng về Toxic**. Bắt buộc có để tính PR-AUC. |
| `pred` | 0 / 1 | Nhãn dự đoán cuối cùng, sau khi đã áp ngưỡng mà bạn chọn. |

**`score` lấy từ đâu:**

| Mô hình | `score` |
| :--- | :--- |
| Naive Bayes | `model.predict_proba(X_test)[:, 1]` (xác suất lớp Toxic) |
| Linear SVM | `model.decision_function(X_test)` |
| BiLSTM | `torch.sigmoid(logits)` (xác suất lớp Toxic) |

---

## 3. Quy tắc bắt buộc

1. **Đủ 2.000 dòng**, mỗi `review_id` của `data/test.csv` xuất hiện đúng 1 lần (thứ tự dòng không quan trọng).
2. **Không có ô trống**, `score` không được là NaN/inf, `label` và `pred` chỉ là 0 hoặc 1.
3. **Ngưỡng quyết định chọn trên tập train (cross-validation) hoặc tập val, tuyệt đối không chọn trên tập test.** Chọn ngưỡng trên test là rò rỉ dữ liệu, kết quả báo cáo sẽ không hợp lệ.
4. File lưu dạng CSV, mã hóa **UTF-8**, dấu phân cách là dấu phẩy, có dòng tiêu đề.
5. Với BiLSTM: khi dự đoán trên test dùng `DataLoader(..., shuffle=False)` để thứ tự dự đoán khớp với thứ tự `review_id`.

---

## 4. Đoạn code mẫu để xuất file

Thêm vào cuối script huấn luyện của bạn (đổi `scores`, `threshold`, tên file cho đúng mô hình):

```python
import os
import pandas as pd

os.makedirs('results/predictions', exist_ok=True)

# test_df = pd.read_csv('data/test.csv')
# scores    = model.predict_proba(X_test)[:, 1]   # hoặc decision_function / sigmoid
# threshold = ngưỡng đã chọn trên train/val

pd.DataFrame({
    'review_id': test_df['review_id'],
    'label': test_df['label'],
    'score': scores,
    'pred': (scores >= threshold).astype(int),
}).to_csv('results/predictions/nb_bow.csv', index=False, encoding='utf-8')
```

---

## 5. Tự kiểm tra trước khi push

```bash
python 4_model_evaluation.py --check results/predictions/nb_bow.csv
```

Lệnh này chỉ kiểm tra file của bạn (đủ dòng, đủ cột, khớp `review_id` và `label` với `data/test.csv`...) và in ra lỗi nếu có. Thấy dòng `HỢP LỆ` là có thể commit và push.

Có thắc mắc về quy ước này, liên hệ TV4.
