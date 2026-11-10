# Số liệu PhoBERT cho báo cáo

## Bảng kết quả trên tập test (1992 câu, 85 toxic)

| Dòng | Accuracy | Precision | Recall | F1 (toxic) | Macro-F1 | PR-AUC | ROC-AUC | Kappa | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PhoBERT, seed nộp 7 (ngưỡng 0.022) | 97.04% | 67.57% | 58.82% | 0.6289 | 0.8068 | 0.6484 | 0.9623 | 0.6136 | 50 | 24 | 35 | 1883 |
| PhoBERT, trung bình ± độ lệch chuẩn 3 seed | 96.99 ± 0.07% | 65.42 ± 1.53% | 62.75 ± 4.74% | 0.6392 ± 0.0196 | 0.8117 ± 0.0099 | 0.6443 ± 0.0029 | 0.9631 ± 0.0015 | 0.6235 ± 0.0198 | 53.3 | 28.3 | 31.7 | 1878.7 |

## Kết quả từng seed (test)

|               |   Accuracy |   Precision (toxic) |   Recall (toxic) |   F1 (toxic) |   Macro-F1 |   PR-AUC |   ROC-AUC |   Cohen's Kappa |      TP |      FP |      FN |        TN |
|:--------------|-----------:|--------------------:|-----------------:|-------------:|-----------:|---------:|----------:|----------------:|--------:|--------:|--------:|----------:|
| seed 42       |     0.9704 |              0.6413 |           0.6941 |       0.6667 |     0.8256 |   0.642  |    0.9619 |          0.6512 | 59      | 33      | 26      | 1874      |
| seed 13       |     0.9689 |              0.6456 |           0.6    |       0.622  |     0.8029 |   0.6425 |    0.9651 |          0.6057 | 51      | 28      | 34      | 1879      |
| seed 7 (nộp)  |     0.9704 |              0.6757 |           0.5882 |       0.6289 |     0.8068 |   0.6484 |    0.9623 |          0.6136 | 50      | 24      | 35      | 1883      |
| Trung bình    |     0.9699 |              0.6542 |           0.6275 |       0.6392 |     0.8117 |   0.6443 |    0.9631 |          0.6235 | 53.3333 | 28.3333 | 31.6667 | 1878.67   |
| Độ lệch chuẩn |     0.0007 |              0.0153 |           0.0474 |       0.0196 |     0.0099 |   0.0029 |    0.0015 |          0.0198 |  4.0277 |  3.6818 |  4.0277 |    3.6818 |

## Cấu hình huấn luyện

- Backbone: `vinai/phobert-base-v2` (135 triệu tham số, 517 MB)
- Đầu vào: INPUT_MODE = `pyvi` (chuẩn hoá NFC, tách chữ dính, teencode = True, tách từ pyvi; giữ chữ hoa, dấu câu, stopword)
- Fine-tune toàn bộ mô hình, AdamW (lr 2e-05, weight decay 0.01), warmup tuyến tính 10%, gradient clipping 1.0, fp16 = True
- Weighted Cross-Entropy, trọng số [1.0, 10.0] (tỉ lệ thật ≈ 22.3, giới hạn 10.0)
- Batch 16, max_len 256, tối đa 5 epoch, dừng sớm theo PR-AUC val (patience 2)
- Seed: [42, 13, 7]; seed nộp 7 (PR-AUC val cao nhất), epoch tốt nhất 4
- Ngưỡng chọn trên val (F1 toxic cao nhất): seed 42: 0.951, seed 13: 0.992, seed 7: 0.022

## Môi trường

- nen_tang: Kaggle
- python: 3.13.16
- torch: 2.11.0+cu130
- transformers: 5.20.2
- scikit_learn: 1.6.1
- pyvi: 0.1.1
- thiet_bi: cuda
- gpu: Tesla T4
- so_luong_cpu: 4

## Chi phí tính toán

- so_tham_so: 134999810
- dung_luong_mo_hinh_mb: 516.9448747634888
- thoi_gian_train_phut: {42: 3.868719784418742, 13: 3.7989279985427857, 7: 4.779845869541168}
- vram_dinh_khi_train_gb: {42: 3.8233604431152344, 13: 4.846841812133789, 7: 4.844568252563477}
- tach_tu_ms_moi_cau: 0.41418533835341864
- gpu: Tesla T4
- gpu_batch_size: 32
- gpu_fp16: True
- gpu_test_giay: 2.8411236709998775
- gpu_cau_moi_giay: 701.131042035546
- gpu_ms_moi_cau: 1.4262669031123882
- gpu_vram_suy_luan_gb: 1.3958592414855957
- cpu_so_luong_dung: 2
- cpu_so_cau_do: 200
- cpu_ms_moi_cau: 71.18473258499989
- cpu_cau_moi_giay: 14.047956123258947

## Phân tích lỗi (seed nộp)

- Bỏ sót (FN): 35 câu, báo nhầm (FP): 24 câu (chi tiết: errors_test.csv)
- Review ID báo cáo đang trích dẫn: xem bang_kiem_tra_review_id.csv