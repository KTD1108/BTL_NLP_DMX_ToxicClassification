"""
Bước 5: Đánh giá mô hình và trực quan hóa kết quả thực nghiệm.
Sử dụng đầy đủ các chỉ số:
- Accuracy
- Precision (lớp Toxic & Macro)
- Recall (lớp Toxic & Macro)
- F1-Score (lớp Toxic & Macro)
- Confusion Matrix (Ma trận nhầm lẫn)
- Đường cong ROC-AUC và Phân tích lỗi (Error Analysis)
"""

import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def evaluate_all_models(
    y_test: np.ndarray,
    experiment_outputs: List[Dict[str, Any]],
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Tính toán toàn bộ chỉ số đánh giá cho từng tổ hợp (Đặc trưng × Thuật toán) và lưu bảng kết quả.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for idx, exp in enumerate(experiment_outputs, start=1):
        y_pred = exp["y_pred"]
        y_prob = exp["y_prob"]

        acc = accuracy_score(y_test, y_pred)
        prec_toxic = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_toxic = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_toxic = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.0

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "STT": idx,
            "Phương pháp trích chọn đặc trưng": exp["feature_name"],
            "Nhóm thuật toán": exp["algorithm_group"],
            "Mô hình chi tiết": exp["model_name"],
            "Tên tổ hợp": exp["full_name"],
            "Accuracy": round(acc * 100, 2),
            "Precision (Toxic %)": round(prec_toxic * 100, 2),
            "Recall (Toxic %)": round(rec_toxic * 100, 2),
            "F1-Score (Toxic %)": round(f1_toxic * 100, 2),
            "Macro-F1 (%)": round(f1_macro * 100, 2),
            "ROC-AUC (%)": round(auc * 100, 2),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "Thời gian train (s)": exp["train_time_sec"],
        })

    df_res = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "results_summary.csv")
    df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df_res


def plot_eda_figures(df: pd.DataFrame, fig_dir: str = "outputs/figures") -> None:
    """Vẽ biểu đồ thống kê mô tả bộ dữ liệu bình luận Điện Máy Xanh."""
    os.makedirs(fig_dir, exist_ok=True)

    # Hình 1: Phân bố nhãn Toxic vs Non-toxic và theo Ngành hàng Điện Máy Xanh
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    label_counts = df["label"].map({0: "Non-toxic (0)", 1: "Toxic (1)"}).value_counts()
    colors = ["#2b8cbe", "#e34a33"]
    bars = axes[0].bar(label_counts.index, label_counts.values, color=colors, width=0.5, edgecolor="black")
    axes[0].set_title("Phân bố nhãn bình luận Điện Máy Xanh", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Số lượng bình luận")
    for bar in bars:
        h = bar.get_height()
        pct = h / len(df) * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            h + 20,
            f"{int(h)} ({pct:.1f}%)",
            ha="center",
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(label_counts.values) * 1.15)

    group_col = "source" if "source" in df.columns else "category"
    group_label = df.groupby([group_col, "label"]).size().unstack(fill_value=0)
    group_label.columns = ["Non-toxic (0)", "Toxic (1)"]
    group_label.plot(kind="bar", stacked=False, ax=axes[1], color=colors, edgecolor="black")
    axes[1].set_title(f"Phân bố nhãn theo {group_col}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Nguồn / Ngành hàng")
    axes[1].set_ylabel("Số lượng mẫu")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(title="Nhãn")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "01_eda_distribution.png"), bbox_inches="tight")
    plt.close(fig)

    # Hình 2: Phân bố độ dài bình luận (số từ sau tách từ)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    sns.histplot(
        data=df,
        x="word_count",
        hue=df["label"].map({0: "Non-toxic (0)", 1: "Toxic (1)"}),
        kde=True,
        bins=25,
        palette={"Non-toxic (0)": "#2b8cbe", "Toxic (1)": "#e34a33"},
        ax=ax,
    )
    ax.set_title("Phân bố độ dài câu bình luận (sau khi tách từ tiếng Việt)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Số lượng từ (Word tokens)")
    ax.set_ylabel("Tần suất")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "02_word_length_distribution.png"), bbox_inches="tight")
    plt.close(fig)


def plot_evaluation_figures(
    y_test: np.ndarray,
    experiment_outputs: List[Dict[str, Any]],
    df_results: pd.DataFrame,
    fig_dir: str = "outputs/figures",
) -> None:
    """Vẽ biểu đồ so sánh hiệu suất, Grid Confusion Matrix và đường cong ROC-AUC."""
    os.makedirs(fig_dir, exist_ok=True)

    # Hình 3: Biểu đồ so sánh các chỉ số Accuracy, Precision, Recall, F1-Score
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    short_names = [
        f"{r['Phương pháp trích chọn đặc trưng'].split('(')[0].strip()}\n+ {r['Mô hình chi tiết']}"
        for _, r in df_results.iterrows()
    ]
    x = np.arange(len(short_names))
    w = 0.2

    ax.bar(x - 1.5 * w, df_results["Accuracy"], width=w, label="Accuracy (%)", color="#3182bd")
    ax.bar(x - 0.5 * w, df_results["Precision (Toxic %)"], width=w, label="Precision - Toxic (%)", color="#31a354")
    ax.bar(x + 0.5 * w, df_results["Recall (Toxic %)"], width=w, label="Recall - Toxic (%)", color="#fd8d3c")
    ax.bar(x + 1.5 * w, df_results["F1-Score (Toxic %)"], width=w, label="F1-Score - Toxic (%)", color="#e6550d")

    ax.set_title(
        "So sánh hiệu suất giữa các phương pháp trích chọn đặc trưng và thuật toán phân loại",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9.5)
    ax.set_ylabel("Điểm số (%)")
    ax.set_ylim(50, 103)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    for i, v in enumerate(df_results["F1-Score (Toxic %)"]):
        ax.text(i + 1.5 * w, v + 0.7, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "03_model_comparison_metrics.png"), bbox_inches="tight")
    plt.close(fig)

    # Hình 4: Lưới Confusion Matrices (3x3 cho toàn bộ 9 mô hình)
    n_models = len(experiment_outputs)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows), dpi=150)
    axes = axes.flatten()

    for idx, exp in enumerate(experiment_outputs):
        cm = confusion_matrix(y_test, exp["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Non-toxic (0)", "Toxic (1)"],
            yticklabels=["Non-toxic (0)", "Toxic (1)"],
            ax=axes[idx],
            annot_kws={"size": 12, "weight": "bold"},
        )
        title_str = f"({idx+1}) {exp['feature_name'].split('(')[0].strip()}\n{exp['model_name']}"
        axes[idx].set_title(title_str, fontsize=10.5, fontweight="bold")
        axes[idx].set_xlabel("Nhãn dự đoán (Predicted)")
        axes[idx].set_ylabel("Nhãn thực tế (Actual)")

    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Ma trận nhầm lẫn (Confusion Matrix) trên tập Kiểm tra (Test Set)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "04_confusion_matrices_grid.png"), bbox_inches="tight")
    plt.close(fig)

    # Hình 5: Đường cong ROC-AUC
    fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=150)
    for exp in experiment_outputs:
        fpr, tpr, _ = roc_curve(y_test, exp["y_prob"])
        auc_val = roc_auc_score(y_test, exp["y_prob"])
        label_curve = f"{exp['feature_name'].split('(')[0].strip()} + {exp['model_name']} (AUC = {auc_val*100:.2f}%)"
        ax.plot(fpr, tpr, lw=2, label=label_curve)

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Ngẫu nhiên (AUC = 50.0%)")
    ax.set_title("Đường cong ROC (Receiver Operating Characteristic) trên tập Test", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate (Tỷ lệ báo động giả)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "05_roc_curves.png"), bbox_inches="tight")
    plt.close(fig)


def export_error_analysis(
    df_test: pd.DataFrame,
    experiment_outputs: List[Dict[str, Any]],
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Xuất bảng phân tích các mẫu dự đoán sai (False Positives & False Negatives)
    giữa BoW+NB, TF-IDF+SVM và PhoBERT Transformer để phục vụ thảo luận báo cáo.
    """
    available_cols = [c for c in ["comment_id", "category", "product_name", "source", "sub_group", "raw_text", "clean_text", "label"] if c in df_test.columns]
    df_err = df_test[available_cols].copy()

    for exp in experiment_outputs:
        key = exp["full_name"]
        df_err[f"pred_{key}"] = exp["y_pred"]

    # Lọc các câu mà ít nhất 1 trong 3 đại diện chính dự đoán sai
    bow_nb_col = "pred_Bag of Words (BoW) + Multinomial Naive Bayes"
    tfidf_svm_col = "pred_TF-IDF (Word + Char N-gram) + SVM (LinearSVC)"
    phobert_col = [c for c in df_err.columns if "PhoBERT" in c][0]

    mask_err = (
        (df_err[bow_nb_col] != df_err["label"])
        | (df_err[tfidf_svm_col] != df_err["label"])
        | (df_err[phobert_col] != df_err["label"])
    )
    df_err_subset = df_err[mask_err].copy()
    err_path = os.path.join(output_dir, "error_analysis.csv")
    df_err_subset.to_csv(err_path, index=False, encoding="utf-8-sig")
    return df_err_subset
