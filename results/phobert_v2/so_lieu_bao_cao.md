# Số liệu PhoBERT cho báo cáo

## Bảng kết quả trên tập test (2.000 câu, 75 toxic)

| Dòng | Accuracy | Precision | Recall | F1 (toxic) | Macro-F1 | PR-AUC | ROC-AUC | Kappa | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PhoBERT, seed nộp 13 (ngưỡng 0.006) | 98.29% | 78.05% | 80.00% | 0.7901 | 0.8906 | 0.7580 | 0.9399 | 0.7812 | 64 | 18 | 16 | 1894 |
| PhoBERT, trung bình ± độ lệch chuẩn 3 seed | 97.98 ± 0.23% | 77.80 ± 1.91% | 69.58 ± 8.19% | 0.7316 ± 0.0436 | 0.8605 ± 0.0223 | 0.7738 ± 0.0295 | 0.9646 ± 0.0176 | 0.7212 ± 0.0446 | 55.7 | 16.0 | 24.3 | 1896.0 |

## Kết quả từng seed (test)

|               |   Accuracy |   Precision (toxic) |   Recall (toxic) |   F1 (toxic) |   Macro-F1 |   PR-AUC |   ROC-AUC |   Cohen's Kappa |      TP |      FP |      FN |        TN |
|:--------------|-----------:|--------------------:|-----------------:|-------------:|-----------:|---------:|----------:|----------------:|--------:|--------:|--------:|----------:|
| seed 42       |     0.9784 |              0.7534 |           0.6875 |       0.719  |     0.8539 |   0.7481 |    0.9744 |          0.7078 | 55      | 18      | 25      | 1894      |
| seed 13 (nộp) |     0.9829 |              0.7805 |           0.8    |       0.7901 |     0.8906 |   0.758  |    0.9399 |          0.7812 | 64      | 18      | 16      | 1894      |
| seed 7        |     0.9779 |              0.8    |           0.6    |       0.6857 |     0.8371 |   0.8152 |    0.9796 |          0.6745 | 48      | 12      | 32      | 1900      |
| Trung bình    |     0.9798 |              0.778  |           0.6958 |       0.7316 |     0.8605 |   0.7738 |    0.9646 |          0.7212 | 55.6667 | 16      | 24.3333 | 1896      |
| Độ lệch chuẩn |     0.0023 |              0.0191 |           0.0819 |       0.0436 |     0.0223 |   0.0295 |    0.0176 |          0.0446 |  6.549  |  2.8284 |  6.549  |    2.8284 |

## Cấu hình huấn luyện

- Backbone: `vinai/phobert-base-v2` (135 triệu tham số, 517 MB)
- Đầu vào: INPUT_MODE = `pyvi` (chuẩn hoá NFC, tách chữ dính, teencode = True, tách từ pyvi; giữ chữ hoa, dấu câu, stopword)
- Fine-tune toàn bộ mô hình, AdamW (lr 2e-05, weight decay 0.01), warmup tuyến tính 10%, gradient clipping 1.0, fp16 = True
- Weighted Cross-Entropy, trọng số [1.0, 10.0] (tỉ lệ thật ≈ 22.3, giới hạn 10.0)
- Batch 16, max_len 256, tối đa 5 epoch, dừng sớm theo PR-AUC val (patience 2)
- Seed: [42, 13, 7]; seed nộp 13 (PR-AUC val cao nhất), epoch tốt nhất 5
- Ngưỡng chọn trên val (F1 toxic cao nhất): seed 42: 0.026, seed 13: 0.006, seed 7: 0.997

## Môi trường

- nen_tang: Colab
- python: 3.13.16
- torch: 2.11.0+cu130
- transformers: 5.22.0
- scikit_learn: 1.6.1
- pyvi: 0.1.1
- thiet_bi: cuda
- gpu: Tesla T4
- so_luong_cpu: 2

## Chi phí tính toán

- so_tham_so: 134999810
- dung_luong_mo_hinh_mb: 516.9448766708374
- thoi_gian_train_phut: {42: 5.244119576613108, 13: 4.879935733477274, 7: 4.874565653006235}
- vram_dinh_khi_train_gb: {42: 3.8264427185058594, 13: 4.845277786254883, 7: 4.85704231262207}
- tach_tu_ms_moi_cau: 0.42872958232932945
- gpu: Tesla T4
- gpu_batch_size: 32
- gpu_fp16: True
- gpu_test_giay: 2.736906480000016
- gpu_cau_moi_giay: 727.8290341875285
- gpu_ms_moi_cau: 1.3739490361445865
- gpu_vram_suy_luan_gb: 1.3972554206848145
- cpu_so_luong_dung: 1
- cpu_so_cau_do: 200
- cpu_ms_moi_cau: 122.40616317499985
- cpu_cau_moi_giay: 8.169523282666207

## Phân tích lỗi (seed nộp)

- Bỏ sót (FN): 16 câu, báo nhầm (FP): 18 câu (chi tiết: errors_test.csv)
- Review ID báo cáo đang trích dẫn: xem bang_kiem_tra_review_id.csv