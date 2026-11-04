# PhoBERT fine-tune – Phân loại bình luận Toxic / Không toxic

Fine-tune end-to-end `vinai/phobert-base-v2` trên **bộ dữ liệu chung của dự án** (`data/` ở gốc repo) và xuất file nộp TV4 theo `QUY_UOC_FILE_DU_DOAN.md`, để so sánh trực tiếp với Naive Bayes và SVM trong `4_model_evaluation.py`.

## Cấu trúc thư mục

```
toxic_classification/
├── README.md                    ← file này
├── train_phobert_v2.ipynb       ← notebook train PhoBERT (bản sạch, chưa chạy)
├── btl-nlpv2.ipynb              ← cùng notebook, đã chạy trên Kaggle (GPU T4), có đầy đủ output
├── download_models.py           ← tải PhoBERT thật về thư mục models/ và kiểm tra
├── HUONG_DAN.md                 ← chạy trên Kaggle / máy cá nhân, cấu hình, lỗi thường gặp
├── HUONG_DAN_TAI_MO_HINH.md     ← hướng dẫn tải mô hình từng bước
└── requirements.txt             ← danh sách thư viện cần cài
```

Notebook đọc dữ liệu và ghi kết quả ở gốc repo:

| Đường dẫn | Nội dung |
|---|---|
| `data/train.csv`, `val.csv`, `test.csv` | 7.000 / 1.000 / 2.000 câu; 240 / 45 / 75 câu toxic |
| `results/predictions/phobert.csv`, `phobert_info.json` | File nộp TV4 (4 cột `review_id, label, score, pred`) |
| `results/phobert_v2/` | Metrics, hình, phân tích lỗi, mô hình đã fine-tune (`best_model/` không đưa vào git) |

## Chạy nhanh

**Trên Kaggle (khuyên dùng):** tạo dataset `DATA-NLPV2` từ 3 file trong `data/`, import `train_phobert_v2.ipynb`, gắn dataset qua *Add Input*, bật **GPU T4 x2** và **Internet: On**, rồi *Save & Run All*. Tải `results_phobert_v2.zip` ở tab Output và giải nén vào gốc repo. Chi tiết xem `HUONG_DAN.md`.

**Trên máy cá nhân:**

```bash
cd toxic_classification
pip install -r requirements.txt
python download_models.py                    # tải PhoBERT vào models/ (khoảng 540 MB)
jupyter notebook train_phobert_v2.ipynb      # đọc ../data, ghi ../results
```

Sau khi có kết quả, kiểm tra file nộp: `python 4_model_evaluation.py --check results/predictions/phobert.csv` (chạy ở gốc repo).

## Thiết lập huấn luyện

- **Đầu vào:** `noi_dung_goc`, chuẩn hoá Unicode và teencode, tách từ bằng pyvi (PhoBERT bắt buộc đầu vào đã tách từ).
- **Tối ưu hoá:** AdamW, lr 2e-5, warmup 10%, batch 16, `max_len` 256 (GPU), tối đa 5 epoch, gradient clipping 1.0, fp16 trên GPU.
- **Mất cân bằng lớp:** weighted cross-entropy, trọng số lớp toxic = số câu không toxic / số câu toxic, giới hạn 10.
- **Chọn epoch:** theo PR-AUC trên val, early stopping patience 2.
- **Ngưỡng:** chọn trên val sao cho F1 lớp toxic cao nhất, giữ nguyên khi đánh giá trên test.
- **Seed:** 42, 13, 7. File nộp TV4 lấy seed có PR-AUC trên val cao nhất.

## Kết quả trên Kaggle (`btl-nlpv2.ipynb`, GPU T4)

| Seed | Accuracy | Precision | Recall | F1 (toxic) | Macro-F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 42 | 0,979 | 0,770 | 0,627 | 0,691 | 0,840 | 0,691 | 0,941 |
| 13 (nộp TV4) | 0,980 | 0,740 | 0,720 | **0,730** | 0,860 | **0,804** | 0,979 |
| 7 | 0,973 | 0,636 | 0,653 | 0,645 | 0,815 | 0,747 | 0,977 |
| **Trung bình ± độ lệch chuẩn** | 0,977 ± 0,003 | 0,716 ± 0,057 | 0,667 ± 0,039 | 0,689 ± 0,035 | 0,838 ± 0,018 | 0,747 ± 0,046 | 0,966 ± 0,018 |

Seed 13 được chọn vì có PR-AUC trên val cao nhất (0,839), không chọn theo test. Ngưỡng của seed này là 0,981: mô hình rất tự tin nên ngưỡng tối ưu trên val nằm sát 1.

## Lưu ý khi viết báo cáo

- **Kết quả dao động khá nhiều giữa các seed** (F1 toxic 0,645–0,730), vì val chỉ có 45 câu toxic và test có 75 câu toxic. Nên báo cáo trung bình ± độ lệch chuẩn thay vì chỉ một seed.
- **Ngưỡng được chọn trên val**, nên con số trên test của seed nộp vẫn là đánh giá độc lập. Tuy vậy, việc chọn seed tốt nhất theo val có thể làm kết quả nộp nhỉnh hơn trung bình một chút.
- **So với kết quả PhoBERT trước đó** (F1 toxic 0,612, PR-AUC 0,643; 3 epoch, `max_len` 64, trọng số [1, 8]), bản này dùng `max_len` dài hơn, early stopping theo PR-AUC val và chạy nhiều seed.
