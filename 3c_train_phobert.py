"""
3c_train_phobert.py
THÀNH VIÊN 3: MÔ HÌNH HÓA DEEP LEARNING HIỆN ĐẠI (PhoBERT Contextual Transformer)
PHÁT HIỆN ĐÁNH GIÁ ĐỘC HẠI TRÊN DỮ LIỆU ĐIỆN MÁY XANH

Mục tiêu kỹ thuật:
1. Sử dụng mô hình VinAI PhoBERT (vinai/phobert-base-v2, 135M tham số) trích xuất biểu diễn ngữ cảnh 768 chiều.
2. Huấn luyện bộ phân loại phân tầng tối ưu trên tập Train (7.000 mẫu).
3. Tối ưu ngưỡng quyết định threshold trên tập Validation (1.000 mẫu, 45 toxic).
4. Đánh giá kiểm thử độc lập trên tập Test (2.000 mẫu, 75 toxic).
5. Xuất file kết quả chuẩn quy ước nộp TV4:
   - results/predictions/phobert.csv
   - results/predictions/phobert_info.json
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix
)
from transformers import AutoTokenizer, AutoModel

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / os.environ.get("DATA_DIR", "data")
PRED_DIR = ROOT / "results" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 80, flush=True)
print("THÀNH VIÊN 3: HUẤN LUYỆN MÔ HÌNH HIỆN ĐẠI PhoBERT (vinai/phobert-base-v2)", flush=True)
print(f"Thư mục dữ liệu: {DATA_DIR.name}/", flush=True)
print("=" * 80, flush=True)

import argparse

parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá PhoBERT (vinai/phobert-base-v2)")
parser.add_argument("--threshold", type=float, default=None, help="Ngưỡng phân định thủ công (nếu không set sẽ tự động tối ưu trên Val)")
parser.add_argument("--mode", type=str, default="balanced", choices=["balanced", "high_recall", "high_precision"], help="Chiến lược chọn ngưỡng trên Val")
parser.add_argument("--force", action="store_true", help="Buộc ghi đè kết quả dự đoán kể cả khi đã có kết quả Fine-tuning GPU")
args = parser.parse_args()
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df   = pd.read_csv(DATA_DIR / "val.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")

TEXT_COL = "noi_dung_tien_xu_ly"
LABEL_COL = "label"

train_texts = train_df[TEXT_COL].fillna("").tolist()
val_texts   = val_df[TEXT_COL].fillna("").tolist()
test_texts  = test_df[TEXT_COL].fillna("").tolist()

y_train = train_df[LABEL_COL].values
y_val   = val_df[LABEL_COL].values
y_test  = test_df[LABEL_COL].values

print(f"Train: {len(train_df):,} mẫu | Toxic: {y_train.sum()} ({y_train.mean()*100:.2f}%)", flush=True)
print(f"Val  : {len(val_df):,} mẫu | Toxic: {y_val.sum()} ({y_val.mean()*100:.2f}%)", flush=True)
print(f"Test : {len(test_df):,} mẫu | Toxic: {y_test.sum()} ({y_test.mean()*100:.2f}%)", flush=True)

# 2. Trích xuất PhoBERT Embeddings (có cache để tái lập nhanh)
EMB_CACHE_FILE = CACHE_DIR / "phobert_embeddings.pt"

if EMB_CACHE_FILE.exists():
    print(f"\n[CACHE] Nạp biểu diễn PhoBERT Embeddings từ {EMB_CACHE_FILE.name}...", flush=True)
    cached = torch.load(EMB_CACHE_FILE, map_location='cpu', weights_only=False)
    X_train_emb = cached["train"]
    X_val_emb   = cached["val"]
    X_test_emb  = cached["test"]
    print(" -> Nạp cache thành công!", flush=True)
else:
    print("\n[BƯỚC 1] Đang nạp PhoBERT (vinai/phobert-base-v2) từ HuggingFace...", flush=True)
    MODEL_NAME = "vinai/phobert-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    transformer = AutoModel.from_pretrained(MODEL_NAME)
    transformer.eval()

    def extract_features(texts, desc=""):
        print(f" -> Đang trích xuất đặc trưng {desc} ({len(texts):,} mẫu)...", flush=True)
        batch_size = 64
        all_embeddings = []
        t0 = time.time()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=48,
                return_tensors='pt'
            )
            with torch.no_grad():
                out = transformer(**encoded)
                # Dùng vector [CLS] (token 0) làm biểu diễn câu
                cls_rep = out.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_rep)
            if (i // batch_size + 1) % 10 == 0 or (i + batch_size >= len(texts)):
                elapsed = time.time() - t0
                pct = min(100, int((i + len(batch_texts)) / len(texts) * 100))
                print(f"    [{desc}] Tiến độ: {pct}% ({i + len(batch_texts):,}/{len(texts):,}) [{elapsed:.1f}s]", flush=True)
        return np.vstack(all_embeddings)

    X_train_emb = extract_features(train_texts, "Train")
    X_val_emb   = extract_features(val_texts, "Val")
    X_test_emb  = extract_features(test_texts, "Test")

    torch.save({
        "train": X_train_emb,
        "val": X_val_emb,
        "test": X_test_emb
    }, EMB_CACHE_FILE)
    print(f" -> Đã lưu cache đặc trưng vào {EMB_CACHE_FILE.name}!", flush=True)

# 3. Huấn luyện PhoBERT Classifier Head
print("\n[BƯỚC 2] Huấn luyện Phân loại PhoBERT (L2 Normalized + LinearSVC balanced)...", flush=True)
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

X_tr = normalize(X_train_emb)
X_va = normalize(X_val_emb)
X_te = normalize(X_test_emb)

clf = LinearSVC(
    C=0.2,
    class_weight='balanced',
    max_iter=2000,
    random_state=SEED
)
clf.fit(X_tr, y_train)

val_scores = clf.decision_function(X_va)
test_scores = clf.decision_function(X_te)

# 4. Tìm và hiệu chuẩn ngưỡng tối ưu trên tập Validation (45 toxic)
print("\n[BƯỚC 3] Hiệu chuẩn ngưỡng quyết định (Threshold Calibration) trên tập Validation...", flush=True)
thresholds = np.arange(-0.50, 0.82, 0.02)

val_results = []
for th in thresholds:
    th = round(th, 2)
    v_pred = (val_scores >= th).astype(int)
    vf1 = f1_score(y_val, v_pred, pos_label=1, zero_division=0)
    vp = precision_score(y_val, v_pred, pos_label=1, zero_division=0)
    vr = recall_score(y_val, v_pred, pos_label=1, zero_division=0)
    val_results.append({"th": th, "f1": vf1, "p": vp, "r": vr})

val_res_df = pd.DataFrame(val_results)

if args.threshold is not None:
    best_th = float(args.threshold)
    reason = f"Thủ công chỉ định bởi người dùng (--threshold {best_th:+.2f})"
elif args.mode == "high_precision":
    # Tối ưu F0.5 để ưu tiên triệt tiêu báo oan FP
    f05_scores = [
        (1.25 * r["p"] * r["r"]) / (0.25 * r["p"] + r["r"]) if (0.25 * r["p"] + r["r"]) > 0 else 0
        for _, r in val_res_df.iterrows()
    ]
    best_idx = int(np.argmax(f05_scores))
    best_th = val_res_df.iloc[best_idx]["th"]
    reason = f"Tối ưu F0.5 trên Val (ưu tiên Precision, giảm tối đa False Positives)"
elif args.mode == "high_recall":
    # Tối ưu ngưỡng nhạy để bao quát vi phạm
    subset = val_res_df[(val_res_df["th"] >= 0.10) & (val_res_df["th"] <= 0.35)]
    best_idx = subset["f1"].idxmax()
    best_th = val_res_df.loc[best_idx, "th"]
    reason = f"Tối ưu độ phủ (High-Recall mode, Th={best_th:+.2f})"
else: # balanced
    # Tối ưu F1 toàn diện trên tập Validation
    best_idx = val_res_df["f1"].idxmax()
    best_th = val_res_df.loc[best_idx, "th"]
    reason = f"Tối ưu F1 cực đại trên Validation (Full range [-0.5, +0.8])"

row_sel = val_res_df[val_res_df["th"] == round(best_th, 2)].iloc[0]
print(f" -> Chiến lược chọn ngưỡng: {reason}", flush=True)
print(f" -> Ngưỡng hiệu chuẩn: Threshold = {best_th:+.2f} | Val F1 = {row_sel['f1']:.4f} (Val P = {row_sel['p']*100:.1f}%, Val R = {row_sel['r']*100:.1f}%)", flush=True)

# 5. Đánh giá trên tập Test (2.000 mẫu, 75 toxic)
test_preds = (test_scores >= best_th).astype(int)

acc = accuracy_score(y_test, test_preds)
prec = precision_score(y_test, test_preds, pos_label=1, zero_division=0)
rec = recall_score(y_test, test_preds, pos_label=1, zero_division=0)
f1_tox = f1_score(y_test, test_preds, pos_label=1, zero_division=0)
macro_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
pr_auc = average_precision_score(y_test, test_scores)
tn, fp, fn, tp = confusion_matrix(y_test, test_preds, labels=[0, 1]).ravel()

print("\n" + "=" * 80, flush=True)
print("             KẾT QUẢ ĐÁNH GIÁ PhoBERT TRÊN TẬP TEST (2.000 MẪU)", flush=True)
print("=" * 80, flush=True)
print(f"Accuracy         : {acc*100:.2f}%", flush=True)
print(f"Precision (Toxic): {prec*100:.2f}%", flush=True)
print(f"Recall (Toxic)   : {rec*100:.2f}% (Bắt đúng {tp}/75 câu)", flush=True)
print(f"F1-Score (Toxic) : {f1_tox:.4f}", flush=True)
print(f"Macro F1-Score   : {macro_f1:.4f}", flush=True)
print(f"PR-AUC           : {pr_auc:.4f}", flush=True)
print(f"Ma trận nhầm lẫn : TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}", flush=True)

# 6. Xuất file kết quả nộp TV4
pred_csv_path = PRED_DIR / "phobert.csv"
info_json_path = PRED_DIR / "phobert_info.json"

if info_json_path.exists() and not args.force:
    try:
        with open(info_json_path, "r", encoding="utf-8") as f_chk:
            ex_data = json.load(f_chk)
        if "Fine-Tuning" in str(ex_data.get("sieu_tham_so", {})):
            print("\n[BẢO VỆ DỮ LIỆU] Phát hiện file phobert.csv hiện tại là kết quả Fine-tuning GPU T4 (F1=0.6122).", flush=True)
            print(" -> Giữ nguyên kết quả tối ưu GPU T4. (Dùng thêm cờ --force nếu muốn ghi đè bằng kết quả CPU).", flush=True)
            sys.exit(0)
    except Exception:
        pass

pd.DataFrame({
    "review_id": test_df["review_id"],
    "label": y_test,
    "score": np.round(test_scores, 4),
    "pred": test_preds
}).to_csv(pred_csv_path, index=False, encoding="utf-8-sig")

with open(info_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "nguoi_phu_trach": "TV3",
        "threshold": float(best_th),
        "chon_nguong_tren": "tap_validation",
        "sieu_tham_so": {
            "backbone": "vinai/phobert-base-v2",
            "feature_dim": 768,
            "classifier": "LinearSVC(C=0.2, class_weight='balanced')",
            "max_len": 48
        },
        "random_seed": SEED,
        "metrics_test": {
            "accuracy": float(acc),
            "precision_toxic": float(prec),
            "recall_toxic": float(rec),
            "f1_toxic": float(f1_tox),
            "macro_f1": float(macro_f1),
            "pr_auc": float(pr_auc),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
        }
    }, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Đã xuất file dự đoán: {pred_csv_path}", flush=True)
print(f"[OK] Đã xuất file metadata: {info_json_path}", flush=True)
