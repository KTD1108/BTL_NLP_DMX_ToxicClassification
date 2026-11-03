# BiLSTM – Phát hiện đánh giá toxic (Điện Máy Xanh) – GĐ2

## Cấu trúc
```
data/         train.csv, val.csv, test.csv
embeddings/   (tuỳ chọn) word2vec_vi_words_100dims.txt  (PhoW2V word-level, không commit)
src/          bilstm_utils.py   – Vocab, DataLoader, Embedding, BiLSTMClassifier, train/eval, metrics
notebooks/    04a_bilstm_train.ipynb    – bước 1–5 (train, lưu model, learning curve)
              04b_bilstm_predict.ipynb  – bước 6 (dự đoán test, xuất CSV cho TV4)
results/      đầu ra
```

## Cách chạy
1. `pip install -r requirements.txt`
2. (Khuyến nghị) Tải PhoW2V word-level 100d (hoặc 300d, nhớ sửa EMBEDDING_PATH) (github.com/datquocnguyen/PhoW2V), giải nén file .txt vào `embeddings/`.
3. Mở thư mục `notebooks/`, chạy lần lượt `04a` rồi `04b` (Run All).

## Đầu ra trong results/
| File | Tạo bởi |
|---|---|
| bilstm_best_model.pt  (state_dict + config + ngưỡng) | 04a |
| bilstm_vocab.pkl | 04a |
| 4_bilstm_learning_curve.png | 04a |
| bilstm_history.csv | 04a |
| bilstm_predictions.csv | 04b |
