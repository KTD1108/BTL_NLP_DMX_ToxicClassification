# Tải PhoBERT thật về máy

PhoBERT được công bố miễn phí trên Hugging Face tại https://huggingface.co/vinai/phobert-base-v2. Mô hình nặng khoảng 540 MB và không cần đăng ký tài khoản để tải.

Có 2 cách dùng:

- **Cách A (đơn giản nhất):** không làm gì thêm. Mở notebook và chạy; lần chạy đầu tiên notebook sẽ tự tải mô hình về bộ nhớ đệm của Hugging Face.
- **Cách B (khuyên dùng):** tải trước vào thư mục `models/` bằng script. Sau đó chạy được offline, dễ mang sang máy khác, và biết chắc mô hình đã tải xong.

## Cách B – từng bước

### Bước 1. Cài thư viện (một lần)

```bash
cd toxic_classification
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Nếu máy có GPU NVIDIA, cài PyTorch bản CUDA **trước** khi chạy lệnh trên. Lệnh cài phù hợp với máy bạn có tại https://pytorch.org/get-started/locally/. Ví dụ với CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Bước 2. Tải mô hình

```bash
python download_models.py
```

Khi chạy thành công, màn hình sẽ in ra tương tự như sau:

```
>>> Đang tải vinai/phobert-base-v2 về models/phobert-base-v2 ...
    xong sau 60 giây, 540 MB: bpe.codes, config.json, pytorch_model.bin, vocab.txt ...
>>> Kiểm tra phobert trong models/phobert-base-v2
    câu thử    : Bọn bán hàng lừa_đảo , bảo_hành thì hẹn lên hẹn xuống
    vocab      : 64001 | tham số: 135 triệu | đầu ra: (1, 2)
    OK – mô hình chạy được.
```

Nếu đã tải từ trước, có thể kiểm tra lại bằng `python download_models.py --check_only`.

### Bước 3. Chạy notebook

```bash
jupyter notebook train_phobert_v2.ipynb
```

Bạn cũng có thể mở notebook bằng VS Code. Notebook tự nhận ra thư mục `models/phobert-base-v2`: dòng đầu tiên của cell Cấu hình sẽ in `Mô hình: models/phobert-base-v2`. Sau đó chạy lần lượt các cell từ trên xuống.

Nên đặt `QUICK_TEST = True` để chạy thử khoảng 1 epoch trước, sau đó đổi về `False` để chạy thật.

Sau khi chạy xong, cấu trúc thư mục sẽ như sau (notebook đọc `data/` và ghi `results/` ở gốc repo):

```
BTL_NLP_DMX_ToxicClassification/
├── data/                        ← train/val/test.csv chung của dự án
├── results/
│   ├── predictions/
│   │   └── phobert.csv, phobert_info.json   ← file nộp TV4
│   └── phobert_v2/              ← kết quả chi tiết sau khi train
│       ├── metrics.json, ket_qua_test.csv
│       ├── confusion_matrix_pr_curve.png, learning_curve.png
│       ├── errors_test.csv, predictions_test.csv
│       └── best_model/          ← mô hình đã fine-tune (không đưa vào git)
└── toxic_classification/
    ├── download_models.py
    ├── train_phobert_v2.ipynb
    └── models/
        └── phobert-base-v2/     ← mô hình gốc đã tải về
```

## Các cách tải khác (nếu không dùng script)

**Dùng dòng lệnh của Hugging Face:**

```bash
pip install -U huggingface_hub
huggingface-cli download vinai/phobert-base-v2 --local-dir models/phobert-base-v2 --exclude "*.h5" "*.msgpack"
```

**Tải thủ công qua trình duyệt:** vào https://huggingface.co/vinai/phobert-base-v2/tree/main và tải các file `config.json`, `vocab.txt`, `bpe.codes` và `pytorch_model.bin` (hoặc `model.safetensors` nếu có) vào thư mục `models/phobert-base-v2/`.

**Trên Kaggle:** không cần tải trước. Bật **Internet: On** trong Settings của notebook, mô hình sẽ tự được tải về. Nếu không bật được Internet, tải thư mục `models/phobert-base-v2` lên làm dataset và gắn vào notebook; notebook tự nhận ra (tìm file `bpe.codes` trong `/kaggle/input`).

## Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| Tải chậm hoặc bị ngắt giữa chừng | Chạy lại lệnh, script sẽ tải tiếp phần còn thiếu. Có thể thử dùng mirror bằng cách đặt biến môi trường `HF_ENDPOINT=https://hf-mirror.com` (Windows: `set HF_ENDPOINT=...`, Mac/Linux: `export HF_ENDPOINT=...`). |
| `ProxyError`, `SSLError`, `ConnectionError` | Mạng trường hoặc công ty có thể chặn huggingface.co. Hãy thử mạng khác (ví dụ 4G), tải thủ công qua trình duyệt, hoặc dùng Kaggle. |
| `401` / `403` | Chạy `huggingface-cli login` và dán token tạo tại https://huggingface.co/settings/tokens. |
| `No module named 'pyvi'` | Chạy `pip install pyvi`. |
| `ImportError` liên quan đến tokenizer | Chạy `pip install -U transformers protobuf`. |
| `CUDA out of memory` khi train | Trong cell Cấu hình, đặt `BATCH_SIZE = 8`. |
| Train trên CPU quá chậm | Đặt `MAX_LEN = 96` và `EPOCHS = 3`, hoặc chuyển sang chạy trên Kaggle. |
