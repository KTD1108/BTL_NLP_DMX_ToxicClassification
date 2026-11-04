# Chạy notebook PhoBERT (`train_phobert_v2.ipynb`)

## 1. Chạy trên Kaggle (khuyên dùng)

Kaggle cho dùng GPU miễn phí khoảng 30 giờ/tuần. Với GPU T4, mỗi seed mất khoảng 3–5 phút, cả 3 seed chưa tới 20 phút.

1. **Tạo dataset:** *Datasets → New Dataset*, tải lên 3 file `train.csv`, `val.csv`, `test.csv` trong thư mục `data/` ở gốc repo, đặt tên `DATA-NLPV2`.
2. **Tạo notebook:** *File → Import Notebook*, chọn `train_phobert_v2.ipynb`, rồi gắn dataset `DATA-NLPV2` qua **Add Input**.
3. **Settings** (khung bên phải):
   - **Accelerator:** **GPU T4 x2**. Không chọn P100, vì bản PyTorch mới trên Kaggle có thể không còn hỗ trợ GPU này.
   - **Internet:** **On**, để cài `pyvi` và tải `vinai/phobert-base-v2`. Tài khoản Kaggle cần xác minh số điện thoại mới bật được.
4. Bấm **Save Version → Save & Run All (Commit)**. Notebook chạy nền, tắt trình duyệt cũng không sao.
5. Khi chạy xong, vào tab **Output**:
   - tải `results_phobert_v2.zip`, **giải nén ở gốc repo** (file sẽ nằm đúng ở `results/predictions/` và `results/phobert_v2/`);
   - tải notebook đã chạy (có output) nếu muốn lưu lại, ví dụ `btl-nlpv2.ipynb`.
6. Ở gốc repo, kiểm tra file nộp: `python 4_model_evaluation.py --check results/predictions/phobert.csv`.

Cell ngay sau mục 1 của notebook tự tìm dataset có đường dẫn chứa `data-nlpv2` và `test.csv` đúng 2.000 câu. Nếu gắn nhầm dataset khác (ví dụ bản `data_v3`), notebook báo lỗi ngay. Dòng in ra phải là `Dữ liệu: /kaggle/input/.../data-nlpv2`.

## 2. Chạy trên máy cá nhân

| Phần cứng | Chạy được? | Thời gian ước tính (1 seed, 5 epoch) |
|---|---|---|
| GPU NVIDIA ≥ 4 GB VRAM (GTX 1650, RTX 3050 trở lên) | Tốt nhất | khoảng 5–15 phút |
| Mac Apple Silicon (M1/M2/M3/M4), RAM ≥ 8 GB | Được (dùng MPS) | khoảng 30–90 phút |
| Chỉ có CPU, RAM ≥ 8 GB | Được, nhưng chậm | vài giờ |
| RAM < 8 GB | Không nên, hãy dùng Kaggle | — |

Sau khoảng 20 batch đầu, notebook tự đo tốc độ và in thời gian dự kiến.

```bash
cd toxic_classification
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

# Nếu có GPU NVIDIA, cài PyTorch bản CUDA trước theo hướng dẫn ở https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python download_models.py          # tuỳ chọn: tải trước PhoBERT để chạy offline
jupyter notebook train_phobert_v2.ipynb
```

Khi chạy trong thư mục `toxic_classification/`, notebook tự đọc `../data` và ghi kết quả vào `../results` (gốc repo). **Lưu ý:** việc này ghi đè `results/predictions/phobert.csv` đang có.

## 3. Các tham số trong cell Cấu hình

- `QUICK_TEST = True`: chạy thử nhanh (1 epoch, câu ngắn) để kiểm tra code.
- `SEEDS = [42, 13, 7]`: mặc định 3 seed. Chạy trên CPU thì nên để `[42]`.
- `BATCH_SIZE = 8`: khi bị lỗi hết bộ nhớ (out of memory).
- `MAX_LEN`: mặc định `None` = 256 trên GPU, 128 trên CPU/MPS. Đặt 96 để chạy nhanh hơn trên CPU.
- `EPOCHS`: mặc định 5, có early stopping.
- `INPUT_MODE`: `'pyvi'` (mặc định, tách từ văn bản gốc) hoặc `'tien_xu_ly'` (dùng cột `noi_dung_tien_xu_ly` có sẵn).

## 4. Kết quả

**File nộp TV4** (`results/predictions/`), theo `QUY_UOC_FILE_DU_DOAN.md`:

| File | Nội dung |
|---|---|
| `phobert.csv` | 2.000 dòng, 4 cột `review_id, label, score, pred`; `score` là xác suất toxic, `pred = score >= threshold` |
| `phobert_info.json` | Ngưỡng (chọn trên `val`), siêu tham số, seed nộp, chỉ số trên test, trung bình ± độ lệch chuẩn qua các seed |

Notebook có cell tự kiểm tra file nộp ngay sau khi xuất; thấy `HỢP LỆ` là đạt.

**Kết quả chi tiết** (`results/phobert_v2/`):

| File | Dùng cho báo cáo |
|---|---|
| `metrics.json` | Cấu hình, các chỉ số của từng seed, kèm trung bình ± độ lệch chuẩn |
| `ket_qua_test.csv` | Bảng kết quả trên test theo từng seed |
| `confusion_matrix_pr_curve.png` | Confusion matrix và đường Precision-Recall của seed tốt nhất |
| `learning_curve.png` | Loss trên train và PR-AUC trên val theo từng epoch |
| `errors_test.csv` | Các câu bỏ sót hoặc báo nhầm, dùng cho phần phân tích lỗi |
| `predictions_test.csv` | Xác suất toxic của từng câu trong tập test |
| `best_model/` | Mô hình đã fine-tune và `threshold.json` (khoảng 540 MB, không đưa vào git) |

## 5. Lỗi thường gặp

- **`Không thấy bộ dữ liệu có test.csv 2000 câu`:** dataset gắn vào không phải bản `data/` chung của dự án. Tạo lại dataset từ 3 file trong `data/`.
- **`CUDA out of memory`:** đặt `BATCH_SIZE = 8` hoặc `MAX_LEN = 128`.
- **Không tải được mô hình:** kiểm tra Internet đã bật, hoặc xem `HUONG_DAN_TAI_MO_HINH.md`.
- **`No module named 'pyvi'`:** chạy `pip install pyvi` (trên Kaggle cần bật Internet).
- **Lỗi khi tải tokenizer:** notebook tự thử lại với `trust_remote_code=True`. Nếu vẫn lỗi, chạy `pip install -U transformers`.
- **Trên Mac gặp lỗi MPS:** đặt biến môi trường `PYTORCH_ENABLE_MPS_FALLBACK=1` trước khi mở notebook, hoặc dùng Kaggle.
