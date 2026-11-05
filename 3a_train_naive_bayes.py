"""Multinomial Naive Bayes trên BoW và TF-IDF.

Đầu vào: data/train.csv, data/val.csv, data/test.csv (cột noi_dung_tien_xu_ly, label).
Quy trình (cho mỗi loại đặc trưng):
  1. Dò alpha x ngram_range x min_df bằng 5-fold CV trên train (vectorizer fit trong từng fold),
     xếp hạng cấu hình theo Macro F1 tốt nhất trên dự đoán CV.
  2. Fit lại cấu hình tốt nhất trên toàn bộ train, chọn ngưỡng xác suất cho Macro F1 cao nhất trên val
     (val có thành phần toxic giống test hơn train).
  3. Dự đoán test đúng một lần với ngưỡng đã chọn.
Đầu ra:
  results/predictions/nb_bow.csv, nb_tfidf.csv (+ _info.json)  theo QUY_UOC_FILE_DU_DOAN.md
  results/nb_cv_tuning.csv       kết quả dò tham số trên CV
  results/nb_metrics.csv         chỉ số của cấu hình tốt nhất trên CV / val / test
  results/nb_threshold_curve.png Macro F1 theo ngưỡng trên val
  results/nb_confusion_matrices.png, results/nb_top_features.png
"""
import json
import os
import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / os.environ.get("DATA_DIR", "data")
RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "predictions"
SEED = 42
TEXT, LABEL = "noi_dung_tien_xu_ly", "label"

ALPHAS = [0.1, 0.5, 1.0]
NGRAMS = [(1, 1), (1, 2)]
MIN_DFS = [1, 2]
# Xác suất NB dồn sát 0/1 nên lưới ngưỡng dày ở vùng thấp (thang log) và thưa ở vùng cao.
THRESHOLDS = np.unique(np.round(np.r_[np.logspace(-4, -1, 16), np.arange(0.1, 0.96, 0.05)], 6))
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

FEATURES = {
    "bow": {"name": "BoW", "vec": CountVectorizer, "kwargs": {}},
    "tfidf": {"name": "TF-IDF", "vec": TfidfVectorizer, "kwargs": {"sublinear_tf": True}},
}


def load(name):
    path = DATA / f"{name}.csv"
    if not path.exists():
        sys.exit(f"[!] Thiếu {path.relative_to(ROOT)}")
    df = pd.read_csv(path)
    df[TEXT] = df[TEXT].fillna("")
    return df


def make_pipeline(feat, alpha, ngram, min_df):
    spec = FEATURES[feat]
    # Văn bản đã tách từ (từ ghép nối bằng "_"), nên tách token theo khoảng trắng.
    vec = spec["vec"](ngram_range=ngram, min_df=min_df, token_pattern=r"\S+", **spec["kwargs"])
    return Pipeline([("vec", vec), ("nb", MultinomialNB(alpha=alpha))])


def macro_f1(y, prob, t):
    return f1_score(y, (prob >= t).astype(int), average="macro", zero_division=0)


def best_threshold(y, prob):
    scores = np.array([macro_f1(y, prob, t) for t in THRESHOLDS])
    i = int(np.argmax(scores))
    return float(THRESHOLDS[i]), float(scores[i]), scores


def metrics(y, prob, t):
    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_toxic": precision_score(y, pred, zero_division=0),
        "recall_toxic": recall_score(y, pred, zero_division=0),
        "f1_toxic": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "pr_auc": average_precision_score(y, prob),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def tune(feat, train):
    """Dò lưới tham số bằng CV; trả về bảng kết quả và cấu hình tốt nhất."""
    rows, best = [], None
    for alpha, ngram, min_df in product(ALPHAS, NGRAMS, MIN_DFS):
        pipe = make_pipeline(feat, alpha, ngram, min_df)
        prob = cross_val_predict(pipe, train[TEXT], train[LABEL], cv=CV, method="predict_proba")[:, 1]
        t, score, curve = best_threshold(train[LABEL], prob)
        row = {"features": feat, "alpha": alpha, "ngram": str(ngram), "min_df": min_df,
               "macro_f1_at_0.5": macro_f1(train[LABEL], prob, 0.5),
               "best_threshold": t, "macro_f1_best": score,
               "f1_toxic_best": f1_score(train[LABEL], (prob >= t).astype(int), zero_division=0),
               "pr_auc": average_precision_score(train[LABEL], prob)}
        rows.append(row)
        if best is None or score > best["macro_f1_best"]:
            best = {**row, "ngram": ngram, "cv_prob": prob, "curve": curve}
    return rows, best


def save_submission(model_id, feat, test, prob, best):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    t = best["threshold"]
    pd.DataFrame({
        "review_id": test["review_id"],
        "label": test[LABEL],
        "score": prob,
        "pred": (prob >= t).astype(int),
    }).to_csv(PRED_DIR / f"{model_id}.csv", index=False, encoding="utf-8")
    info = {
        "mo_hinh": "Multinomial Naive Bayes",
        "threshold": t,
        "chon_nguong_tren": "val",
        "sieu_tham_so": {"alpha": best["alpha"], "ngram_range": list(best["ngram"]),
                         "min_df": best["min_df"]},
        "random_seed": SEED,
        "ghi_chu": (f"{FEATURES[feat]['name']} + MultinomialNB; tham số chọn theo Macro F1 trên "
                    "5-fold CV của train, ngưỡng chọn theo Macro F1 trên val"
                    + ("; sublinear_tf" if feat == "tfidf" else "")),
    }
    with open(PRED_DIR / f"{model_id}_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def plot_threshold_curves(bests):
    fig, ax = plt.subplots(figsize=(7, 4))
    for feat, best in bests.items():
        ax.plot(THRESHOLDS, best["val_curve"], marker=".", label=FEATURES[feat]["name"])
        ax.axvline(best["threshold"], ls="--", lw=0.8, color=ax.lines[-1].get_color())
    ax.axvline(0.5, color="grey", lw=0.8, ls=":", label="ngưỡng mặc định 0.5")
    ax.set_xscale("log")
    ax.set_xlabel("Ngưỡng xác suất toxic (thang log)")
    ax.set_ylabel("Macro F1 trên val")
    ax.set_title("Naive Bayes: Macro F1 theo ngưỡng")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "nb_threshold_curve.png", dpi=200)
    plt.close(fig)


def plot_confusion(cms):
    fig, axes = plt.subplots(1, len(cms), figsize=(5 * len(cms), 4))
    for ax, (title, cm) in zip(np.atleast_1d(axes), cms.items()):
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1], ["Non-toxic", "Toxic"])
        ax.set_yticks([0, 1], ["Non-toxic", "Toxic"])
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(RESULTS / "nb_confusion_matrices.png", dpi=200)
    plt.close(fig)


def plot_top_features(pipe, k=20):
    """Từ có log P(từ|toxic) - log P(từ|non-toxic) lớn nhất trong mô hình BoW."""
    vocab = pipe.named_steps["vec"].get_feature_names_out()
    log_prob = pipe.named_steps["nb"].feature_log_prob_
    ratio = log_prob[1] - log_prob[0]
    top = np.argsort(ratio)[-k:]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([vocab[i] for i in top], ratio[top], color="#c0392b")
    ax.set_xlabel("log P(từ | toxic) − log P(từ | non-toxic)")
    ax.set_title(f"BoW + NB: {k} đặc trưng nghiêng về Toxic nhất")
    fig.tight_layout()
    fig.savefig(RESULTS / "nb_top_features.png", dpi=200)
    plt.close(fig)


def main():
    train, val, test = load("train"), load("val"), load("test")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        print(f"[-] {name}: {len(df):,} câu, toxic = {df[LABEL].sum()} ({df[LABEL].mean():.2%})")

    RESULTS.mkdir(exist_ok=True)
    tuning_rows, metric_rows, bests, cms = [], [], {}, {}
    for feat in FEATURES:
        rows, best = tune(feat, train)
        tuning_rows += rows
        bests[feat] = best

        pipe = make_pipeline(feat, best["alpha"], best["ngram"], best["min_df"])
        pipe.fit(train[TEXT], train[LABEL])
        val_prob = pipe.predict_proba(val[TEXT])[:, 1]
        t, _, best["val_curve"] = best_threshold(val[LABEL], val_prob)
        best["threshold"] = t
        test_prob = pipe.predict_proba(test[TEXT])[:, 1]

        model_id = f"nb_{feat}"
        save_submission(model_id, feat, test, test_prob, best)
        for split, y, prob in [("train_cv", train[LABEL], best["cv_prob"]),
                               ("val", val[LABEL], val_prob), ("test", test[LABEL], test_prob)]:
            metric_rows.append({"model": model_id, "split": split, "threshold": t,
                                **metrics(y, prob, t)})
        cms[f"{FEATURES[feat]['name']} + NB (test, ngưỡng {t:g})"] = confusion_matrix(
            test[LABEL], (test_prob >= t).astype(int), labels=[0, 1])
        if feat == "bow":
            plot_top_features(pipe)

        print(f"\n[-] {FEATURES[feat]['name']} + NB: alpha={best['alpha']}, ngram={best['ngram']}, "
              f"min_df={best['min_df']}, ngưỡng={t:g}")
        for split in ("train_cv", "val", "test"):
            m = next(r for r in metric_rows if r["model"] == model_id and r["split"] == split)
            print(f"    {split:8s}: P={m['precision_toxic']:.4f} | R={m['recall_toxic']:.4f} | "
                  f"F1 toxic={m['f1_toxic']:.4f} | Macro F1={m['macro_f1']:.4f} | PR-AUC={m['pr_auc']:.4f}")

    pd.DataFrame(tuning_rows).to_csv(RESULTS / "nb_cv_tuning.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(RESULTS / "nb_metrics.csv", index=False, encoding="utf-8-sig")
    plot_threshold_curves(bests)
    plot_confusion(cms)
    print("\n[-] Đã lưu results/predictions/nb_{bow,tfidf}.csv (+ _info.json), nb_cv_tuning.csv, "
          "nb_metrics.csv và 3 biểu đồ nb_*.png")


if __name__ == "__main__":
    main()
