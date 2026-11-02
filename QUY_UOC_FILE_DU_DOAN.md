# QUY ƯỚC FILE DỰ ĐOÁN TRÊN TẬP TEST (NỘP CHO TV4)

> **Áp dụng cho:** TV1 (SVM), TV2 (Naive Bayes), TV3 (BiLSTM).
> **Mục đích:** TV4 tự tính lại toàn bộ chỉ số (Accuracy, Precision, Recall, F1 Toxic, Macro F1, PR-AUC) cho cả 5 mô hình bằng **cùng một đoạn code** trong `4_model_evaluation.py`, để bảng so sánh trong báo cáo công bằng và nhất quán.
> Mỗi người vẫn giữ các file kết quả riêng của mình (metrics, biểu đồ...) như bình thường. File dưới đây là file **nộp thêm**.

---

## 1. Tên file và vị trí

Mỗi mô hình nộp **1 file CSV** (dự đoán) và **1 file JSON** (thông tin mô hình, xem mục 4) trong thư mục `results/predictions/`, tên file theo mã mô hình:

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

## 4. File thông tin mô hình đi kèm: `<ma_mo_hinh>_info.json`

Mỗi file dự đoán phải có **1 file JSON đi kèm cùng thư mục**, ví dụ `results/predictions/nb_bow.csv` đi với `results/predictions/nb_bow_info.json`. TV4 dùng file này để ghi cấu hình vào Chương 3 của báo cáo và để kiểm tra `pred` có đúng bằng `score >= threshold` không.

```json
{
  "nguoi_phu_trach": "TV2",
  "threshold": 0.3,
  "chon_nguong_tren": "val",
  "sieu_tham_so": {"alpha": 0.5},
  "random_seed": 42,
  "ghi_chu": ""
}
```

| Trường | Bắt buộc | Ý nghĩa |
| :--- | :---: | :--- |
| `nguoi_phu_trach` | ✔ | `TV1` / `TV2` / `TV3`. |
| `threshold` | ✔ | Ngưỡng đã dùng để tạo cột `pred` (đúng quy tắc `pred = score >= threshold`). |
| `chon_nguong_tren` | ✔ | Ngưỡng được chọn ở đâu: `"val"`, `"train_cv"` (cross-validation trên train) hoặc `"mac_dinh"` (giữ ngưỡng mặc định, không tinh chỉnh). |
| `sieu_tham_so` | ✔ | Bộ siêu tham số tốt nhất của mô hình cuối cùng, ví dụ `{"alpha": 0.5}`, `{"C": 0.25, "class_weight": {"0": 1, "1": 15}}`, `{"epochs": 7, "lr": 0.001, "pos_weight": 48}`. |
| `random_seed` | ✔ | Seed đã dùng; ghi `null` nếu mô hình không có yếu tố ngẫu nhiên. |
| `ghi_chu` | | Ghi chú thêm nếu cần (đặc trưng dùng, lưu ý đặc biệt...). |

---

## 5. Đoạn code mẫu để xuất 2 file

Thêm vào cuối script huấn luyện của bạn (đổi `scores`, `threshold`, `MODEL_ID` và thông tin cho đúng mô hình):

```python
import os
import json
import pandas as pd

MODEL_ID = 'nb_bow'
os.makedirs('results/predictions', exist_ok=True)

# test_df = pd.read_csv('data/test.csv')
# scores    = model.predict_proba(X_test)[:, 1]   # hoặc decision_function / sigmoid
# threshold = ngưỡng đã chọn trên train/val

pd.DataFrame({
    'review_id': test_df['review_id'],
    'label': test_df['label'],
    'score': scores,
    'pred': (scores >= threshold).astype(int),
}).to_csv(f'results/predictions/{MODEL_ID}.csv', index=False, encoding='utf-8')

with open(f'results/predictions/{MODEL_ID}_info.json', 'w', encoding='utf-8') as f:
    json.dump({
        'nguoi_phu_trach': 'TV2',
        'threshold': float(threshold),
        'chon_nguong_tren': 'val',
        'sieu_tham_so': {'alpha': 0.5},
        'random_seed': 42,
        'ghi_chu': '',
    }, f, ensure_ascii=False, indent=2)
```

---

## 6. Tự kiểm tra trước khi push

```bash
python 4_model_evaluation.py --check results/predictions/nb_bow.csv
```

Lệnh này kiểm tra file dự đoán và file `_info.json` đi kèm (đủ dòng, đủ cột, khớp `review_id` và `label` với `data/test.csv`, `pred` khớp với `threshold`...) và in ra lỗi nếu có. Thấy dòng `HỢP LỆ` là có thể commit và push.

Có thắc mắc về quy ước này, liên hệ TV4.
