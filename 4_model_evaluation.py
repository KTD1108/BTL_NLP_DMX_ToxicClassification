# -*- coding: utf-8 -*-
"""
THÀNH VIÊN 4: EVALUATION LEAD & REPORT AUTHOR
Script: 4_model_evaluation.py

Nhiệm vụ:
1. Đọc các file dự đoán chuẩn hóa trong `results/predictions/` (quy ước tại QUY_UOC_FILE_DU_DOAN.md),
   kiểm tra tính hợp lệ so với tập Test chuẩn vàng `data/test.csv` (2.000 mẫu).
2. Tính lại 6 chỉ số cho mọi mô hình bằng cùng một đoạn code:
   Accuracy, Precision (Toxic), Recall (Toxic), F1 (Toxic), Macro F1, PR-AUC.
3. Xuất kết quả (300 DPI):
   - results/tv4_model_comparison.csv : Bảng đối sánh các mô hình.
   - results/5_confusion_matrices.png  : Lưới ma trận nhầm lẫn.
   - results/6_model_comparison.png    : Biểu đồ cột Macro F1 và Recall (Toxic).
   - results/tv4_error_cases.csv       : Các câu bị ít nhất 1 mô hình đoán sai (FP/FN) để phân tích lỗi.

Cách chạy:
   python 4_model_evaluation.py                                        # Đánh giá toàn bộ
   python 4_model_evaluation.py --check results/predictions/nb_bow.csv # Chỉ kiểm tra 1 file
"""

import os
import sys
import glob
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

TEST_PATH = os.path.join('data', 'test.csv')
REQUIRED_COLS = ['review_id', 'label', 'score', 'pred']

# Thứ tự khai báo = thứ tự hiển thị trong bảng và biểu đồ
MODEL_NAMES = {
    'nb_bow': 'BoW + Naive Bayes',
    'nb_tfidf': 'TF-IDF + Naive Bayes',
    'svm_bow': 'BoW + Linear SVM',
    'svm_tfidf': 'TF-IDF + Linear SVM',
    'bilstm': 'Word Embeddings + BiLSTM',
}

# Bảng màu đồ thị
COLOR_MACRO_F1 = '#2a78d6'
COLOR_RECALL = '#eb6834'
COLOR_TEXT = '#0b0b0b'
COLOR_TEXT_MUTED = '#52514e'
COLOR_GRID = '#e4e3df'
CM_CMAP = LinearSegmentedColormap.from_list(
    'cm_blues', ['#f4f8fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#104281']
)


# -----------------------------------------------------------------------------
# 1. KIỂM TRA TÍNH HỢP LỆ CỦA FILE DỰ ĐOÁN
# -----------------------------------------------------------------------------
def validate_prediction_file(path, test_df):
    """Trả về (DataFrame đã sắp theo thứ tự test.csv, danh sách lỗi). DataFrame là None nếu có lỗi."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, [f"Không đọc được file: {e}"]

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        return None, [f"Thiếu cột {missing_cols}. File phải có đúng các cột {REQUIRED_COLS}."]

    df = df[REQUIRED_COLS].copy()
    errors = []

    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        errors.append(f"Có ô trống ở cột {null_cols}.")

    for col in ['review_id', 'label', 'pred', 'score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if len(df) != len(test_df):
        errors.append(f"File có {len(df)} dòng, cần đúng {len(test_df)} dòng.")

    n_dup = df['review_id'].duplicated().sum()
    if n_dup:
        errors.append(f"Có {n_dup} review_id bị lặp.")

    test_ids = set(test_df['review_id'])
    pred_ids = set(df['review_id'].dropna())
    if test_ids - pred_ids:
        errors.append(f"Thiếu {len(test_ids - pred_ids)} review_id có trong data/test.csv.")
    if pred_ids - test_ids:
        errors.append(f"Có {len(pred_ids - test_ids)} review_id không thuộc data/test.csv.")

    for col in ['label', 'pred']:
        bad = ~df[col].isin([0, 1])
        if bad.any():
            errors.append(f"Cột `{col}` có {bad.sum()} giá trị khác 0/1.")

    bad_score = ~np.isfinite(df['score'])
    if bad_score.any():
        errors.append(f"Cột `score` có {bad_score.sum()} giá trị không phải số hữu hạn (NaN/inf/chữ).")

    if errors:
        return None, errors

    df[['review_id', 'label', 'pred']] = df[['review_id', 'label', 'pred']].astype(int)
    df = df.set_index('review_id').loc[test_df['review_id']].reset_index()

    n_label_mismatch = (df['label'].values != test_df['label'].values).sum()
    if n_label_mismatch:
        return None, [f"Có {n_label_mismatch} dòng `label` không khớp với data/test.csv "
                      f"(có thể đang dùng nhầm file test cũ)."]

    return df, []


# -----------------------------------------------------------------------------
# 2. TÍNH CHỈ SỐ ĐÁNH GIÁ
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, scores):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision (Toxic)': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Recall (Toxic)': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'F1 (Toxic)': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Macro F1': f1_score(y_true, y_pred, average='macro'),
        'PR-AUC': average_precision_score(y_true, scores),
        'TP': cm[1, 1],
        'FP': cm[0, 1],
        'FN': cm[1, 0],
        'TN': cm[0, 0],
        '_cm': cm,
    }


def run_check(path, test_df):
    print(f"Kiểm tra file: {path}")
    df, errors = validate_prediction_file(path, test_df)
    if errors:
        print(" -> KHÔNG HỢP LỆ:")
        for err in errors:
            print(f"    - {err}")
        return 1
    m = compute_metrics(df['label'], df['pred'], df['score'])
    print(" -> HỢP LỆ.")
    print(f"    Dự đoán Toxic: {df['pred'].sum()} câu | TP={m['TP']} FP={m['FP']} FN={m['FN']} "
          f"| Macro F1={m['Macro F1']:.4f} | PR-AUC={m['PR-AUC']:.4f}")
    return 0


# -----------------------------------------------------------------------------
# 3. TRỰC QUAN HÓA
# -----------------------------------------------------------------------------
def plot_confusion_matrices(results, out_path):
    n = len(results)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
    tick_labels = ['Non-toxic (0)', 'Toxic (1)']

    for ax, r in zip(axes.flat, results):
        cm = r['_cm']
        # Tô màu theo tỷ lệ trên từng hàng (tỷ lệ nhận đúng của mỗi lớp) để lớp Toxic ít mẫu vẫn nhìn rõ
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(cm_norm, cmap=CM_CMAP, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)",
                        ha='center', va='center', fontsize=12,
                        color='white' if cm_norm[i, j] > 0.55 else COLOR_TEXT)
        ax.set_xticks([0, 1], tick_labels)
        ax.set_yticks([0, 1], tick_labels, rotation=90, va='center')
        ax.set_xlabel('Dự đoán', color=COLOR_TEXT_MUTED)
        ax.set_ylabel('Thực tế', color=COLOR_TEXT_MUTED)
        ax.set_title(f"{r['Mô hình']}\nMacro F1 = {r['Macro F1']:.4f}", fontsize=11, color=COLOR_TEXT)
        ax.grid(False)

    for ax in list(axes.flat)[n:]:
        ax.axis('off')

    fig.suptitle('Ma trận nhầm lẫn trên tập Test (2.000 mẫu)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_model_comparison(results, out_path):
    names = [r['Mô hình'].replace(' + ', '\n+ ') for r in results]
    macro_f1 = [r['Macro F1'] for r in results]
    recall = [r['Recall (Toxic)'] for r in results]

    x = np.arange(len(results))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(6, 2.0 * len(results) + 1.5), 5))

    for offset, values, label, color in [
        (-width / 2, macro_f1, 'Macro F1', COLOR_MACRO_F1),
        (width / 2, recall, 'Recall (Toxic)', COLOR_RECALL),
    ]:
        bars = ax.bar(x + offset, values, width, label=label, color=color,
                      edgecolor='white', linewidth=2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha='center', va='bottom', fontsize=9, color=COLOR_TEXT_MUTED)

    ax.set_xticks(x, names, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Điểm số', color=COLOR_TEXT_MUTED)
    ax.set_title('So sánh Macro F1 và Recall (Toxic) giữa các mô hình', fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.tick_params(axis='y', length=0, colors=COLOR_TEXT_MUTED)
    ax.legend(loc='upper left', ncols=2, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. CHƯƠNG TRÌNH CHÍNH
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Đánh giá đối sánh các mô hình trên tập Test.')
    parser.add_argument('--check', metavar='FILE', help='Chỉ kiểm tra tính hợp lệ của 1 file dự đoán.')
    parser.add_argument('--pred-dir', default=os.path.join('results', 'predictions'),
                        help='Thư mục chứa các file dự đoán (mặc định: results/predictions).')
    parser.add_argument('--out-dir', default='results', help='Thư mục lưu kết quả (mặc định: results).')
    args = parser.parse_args()

    test_df = pd.read_csv(TEST_PATH)

    if args.check:
        sys.exit(run_check(args.check, test_df))

    print("=" * 85)
    print("     THÀNH VIÊN 4: ĐÁNH GIÁ ĐỐI SÁNH CÁC MÔ HÌNH TRÊN TẬP TEST (2.000 MẪU)")
    print("=" * 85)

    # --- Nạp và kiểm tra các file dự đoán ---
    paths = glob.glob(os.path.join(args.pred_dir, '*.csv'))
    known = list(MODEL_NAMES)
    paths.sort(key=lambda p: (known.index(os.path.splitext(os.path.basename(p))[0])
                              if os.path.splitext(os.path.basename(p))[0] in known else len(known),
                              os.path.basename(p)))
    if not paths:
        print(f"Không tìm thấy file dự đoán nào trong `{args.pred_dir}`. Xem QUY_UOC_FILE_DU_DOAN.md.")
        sys.exit(1)

    print(f"\n[BƯỚC 1] Kiểm tra {len(paths)} file dự đoán trong `{args.pred_dir}`...")
    predictions = {}
    for path in paths:
        model_id = os.path.splitext(os.path.basename(path))[0]
        df, errors = validate_prediction_file(path, test_df)
        if errors:
            print(f" [BỎ QUA] {model_id}: " + ' | '.join(errors))
            continue
        predictions[model_id] = df
        print(f" [OK] {model_id}")

    missing_models = [m for m in MODEL_NAMES if m not in predictions]
    if missing_models:
        print(f" -> Chưa có (hoặc không hợp lệ): {missing_models}")
    if not predictions:
        print("Không có file dự đoán hợp lệ nào để đánh giá.")
        sys.exit(1)

    # --- Tính chỉ số ---
    print("\n[BƯỚC 2] Tính chỉ số đánh giá...")
    results = []
    for model_id, df in predictions.items():
        m = compute_metrics(df['label'], df['pred'], df['score'])
        m['Mã mô hình'] = model_id
        m['Mô hình'] = MODEL_NAMES.get(model_id, model_id)
        results.append(m)

    table_cols = ['Mã mô hình', 'Mô hình', 'Accuracy', 'Precision (Toxic)', 'Recall (Toxic)',
                  'F1 (Toxic)', 'Macro F1', 'PR-AUC', 'TP', 'FP', 'FN', 'TN']
    df_metrics = pd.DataFrame(results)[table_cols]
    print("\n" + df_metrics.drop(columns='Mã mô hình').to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, 'tv4_model_comparison.csv')
    df_metrics.round(4).to_csv(metrics_path, index=False, encoding='utf-8-sig')

    # --- Biểu đồ ---
    print("\n[BƯỚC 3] Vẽ biểu đồ...")
    cm_path = os.path.join(args.out_dir, '5_confusion_matrices.png')
    cmp_path = os.path.join(args.out_dir, '6_model_comparison.png')
    plot_confusion_matrices(results, cm_path)
    plot_model_comparison(results, cmp_path)

    # --- Trích xuất các câu bị đoán sai để phân tích lỗi ---
    print("\n[BƯỚC 4] Trích xuất các câu bị đoán sai (FP/FN)...")
    errors_df = test_df[['review_id', 'nganh_hang', 'so_sao', 'noi_dung_goc', 'label']].copy()
    wrong = np.zeros(len(test_df), dtype=int)
    for model_id, df in predictions.items():
        errors_df[f'pred_{model_id}'] = df['pred'].values
        errors_df[f'score_{model_id}'] = df['score'].round(4).values
        wrong += (df['pred'].values != test_df['label'].values)
    errors_df.insert(5, 'so_mo_hinh_sai', wrong)
    errors_df.insert(5, 'loai_loi', np.where(test_df['label'] == 1, 'FN (bỏ sót)', 'FP (bắt nhầm)'))
    errors_df = errors_df[errors_df['so_mo_hinh_sai'] > 0]
    errors_df = errors_df.sort_values(['loai_loi', 'so_mo_hinh_sai'], ascending=[True, False])
    errors_path = os.path.join(args.out_dir, 'tv4_error_cases.csv')
    errors_df.to_csv(errors_path, index=False, encoding='utf-8-sig')

    n_models = len(predictions)
    all_wrong = errors_df[errors_df['so_mo_hinh_sai'] == n_models]
    print(f" -> {len(errors_df)} câu bị ít nhất 1 mô hình đoán sai, "
          f"{len(all_wrong)} câu bị tất cả {n_models} mô hình đoán sai "
          f"({(all_wrong['label'] == 1).sum()} FN, {(all_wrong['label'] == 0).sum()} FP).")

    print("\nĐã lưu:")
    for p in [metrics_path, cm_path, cmp_path, errors_path]:
        print(f" - {p}")


if __name__ == '__main__':
    main()
