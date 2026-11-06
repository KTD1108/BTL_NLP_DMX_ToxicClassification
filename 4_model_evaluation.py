# -*- coding: utf-8 -*-
"""
HỆ THỐNG ĐÁNH GIÁ ĐỐI SÁNH HIỆU NĂNG MÔ HÌNH (MODEL EVALUATION PIPELINE)
Script: 4_model_evaluation.py

Nhiệm vụ:
1. Đọc các file dự đoán chuẩn hóa và file thông tin `_info.json` đi kèm trong `results/predictions/`,
   kiểm tra tính hợp lệ so với tập Test chuẩn vàng `data/test.csv`.
2. Tính lại 6 chỉ số cho mọi mô hình bằng cùng một đoạn code:
   Accuracy, Precision (Toxic), Recall (Toxic), F1 (Toxic), Macro F1, PR-AUC.
3. Xuất kết quả (300 DPI):
   - results/tv4_model_comparison.csv : Bảng đối sánh các mô hình.
   - results/5_confusion_matrices.png  : Lưới ma trận nhầm lẫn.
   - results/6_model_comparison.png    : Biểu đồ cột Macro F1 và Recall (Toxic).
   - results/tv4_error_cases.csv       : Các câu bị ít nhất 1 mô hình đoán sai (FP/FN) để phân tích lỗi.
   - results/tv4_bootstrap_ci.csv      : Khoảng tin cậy 95% (paired bootstrap) cho F1 Toxic, Macro F1, PR-AUC.
   - results/tv4_pairwise_tests.csv    : So sánh từng cặp mô hình, cả hai cách đều hiệu chỉnh nhiều phép so sánh:
                                         (1) chênh lệch F1 Toxic bằng paired bootstrap: CI 95%, CI đồng thời
                                             (Bonferroni), p bootstrap hiệu chỉnh Holm -> căn cứ chính;
                                         (2) McNemar chính xác trên số câu đúng/sai, hiệu chỉnh Holm -> bổ trợ.
   - results/tv4_shortcut_baselines.csv: Baseline đối chứng chỉ dùng độ dài câu / số sao (ngưỡng chọn trên val).
   - results/7_pr_curves.png           : Đường cong Precision-Recall của các mô hình và baseline.
   - results/tv4_slice_analysis.csv    : Recall/Precision/F1 (Toxic) của từng mô hình theo nhóm dữ liệu
                                         (độ dài câu, số sao, có/không từ khóa toxic, ngành hàng).
   - results/8_slice_analysis.png      : Heatmap Recall và Precision theo nhóm dữ liệu.

Cách chạy:
   python 4_model_evaluation.py                                        # Đánh giá toàn bộ
   python 4_model_evaluation.py --check results/predictions/nb_bow.csv # Chỉ kiểm tra 1 file
"""

import os
import re
import sys
import glob
import json
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

DATA_DIR = os.environ.get('DATA_DIR', 'data')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
REQUIRED_COLS = ['review_id', 'label', 'score', 'pred']
REQUIRED_INFO_KEYS = ['mo_hinh', 'threshold', 'chon_nguong_tren', 'sieu_tham_so', 'random_seed']
THRESHOLD_SOURCES = ['val', 'train_cv', 'mac_dinh', 'tap_validation']

# Thứ tự khai báo = thứ tự hiển thị trong bảng và biểu đồ
MODEL_NAMES = {
    'nb_bow': 'BoW + Naive Bayes',
    'nb_tfidf': 'TF-IDF + Naive Bayes',
    'svm_bow': 'BoW + Linear SVM',
    'svm_tfidf_single': 'TF-IDF đơn + Linear SVM',
    'svm_tfidf': 'FeatureUnion + Linear SVM',
    'phobert': 'PhoBERT (vinai/phobert-base-v2)',
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


def validate_info_file(info_path, df):
    """Kiểm tra file `<ma_mo_hinh>_info.json` đi kèm. Trả về (dict thông tin, danh sách lỗi)."""
    if not os.path.exists(info_path):
        return None, [f"Thiếu file thông tin `{os.path.basename(info_path)}` đi kèm."]
    try:
        with open(info_path, encoding='utf-8') as f:
            info = json.load(f)
    except Exception as e:
        return None, [f"Không đọc được `{os.path.basename(info_path)}`: {e}"]
    if not isinstance(info, dict):
        return None, ["File thông tin phải là một object JSON."]

    missing_keys = [k for k in REQUIRED_INFO_KEYS if k not in info]
    if missing_keys:
        return None, [f"File thông tin thiếu trường {missing_keys}."]

    errors = []
    if info['chon_nguong_tren'] not in THRESHOLD_SOURCES:
        errors.append(f"`chon_nguong_tren` = {info['chon_nguong_tren']!r}, chỉ chấp nhận {THRESHOLD_SOURCES}.")
    if not isinstance(info['sieu_tham_so'], dict):
        errors.append("`sieu_tham_so` phải là một object JSON, ví dụ {\"alpha\": 0.5}.")
    if isinstance(info['threshold'], bool) or not isinstance(info['threshold'], (int, float)):
        errors.append(f"`threshold` = {info['threshold']!r} không phải số.")
    elif df is not None:
        n_mismatch = ((df['score'] >= info['threshold']).astype(int) != df['pred']).sum()
        if n_mismatch:
            errors.append(f"Có {n_mismatch} dòng `pred` không khớp với `score >= threshold` "
                          f"(threshold = {info['threshold']}).")

    return (info if not errors else None), errors


def validate_submission(path, test_df):
    """Kiểm tra cặp file dự đoán + file thông tin. Trả về (DataFrame, dict thông tin, danh sách lỗi)."""
    df, errors = validate_prediction_file(path, test_df)
    info_path = os.path.splitext(path)[0] + '_info.json'
    info, info_errors = validate_info_file(info_path, df)
    errors = errors + info_errors
    if errors:
        return None, None, errors
    return df, info, []


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


# -----------------------------------------------------------------------------
# 2b. KIỂM ĐỊNH THỐNG KÊ: BOOTSTRAP CI VÀ McNEMAR
# -----------------------------------------------------------------------------
def _f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def bootstrap_metrics(predictions, y_true, n_boot, seed):
    """Paired bootstrap: mọi mô hình dùng chung các mẫu lặp lại để so sánh cặp được.
    Trả về {model_id: {'F1 (Toxic)': arr, 'Macro F1': arr, 'PR-AUC': arr}}."""
    y = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    yb = y[idx]
    valid = yb.sum(axis=1) > 0  # bỏ mẫu lặp không có câu Toxic nào (PR-AUC không xác định)
    idx, yb = idx[valid], yb[valid]

    out = {}
    for model_id, df in predictions.items():
        pb = df['pred'].values[idx]
        sb = df['score'].values[idx]
        tp = ((yb == 1) & (pb == 1)).sum(axis=1)
        fp = ((yb == 0) & (pb == 1)).sum(axis=1)
        fn = ((yb == 1) & (pb == 0)).sum(axis=1)
        tn = ((yb == 0) & (pb == 0)).sum(axis=1)
        f1_pos = _f1_from_counts(tp, fp, fn)
        f1_neg = _f1_from_counts(tn, fn, fp)
        pr_auc = np.array([average_precision_score(yb[i], sb[i]) for i in range(len(yb))])
        out[model_id] = {'F1 (Toxic)': f1_pos, 'Macro F1': (f1_pos + f1_neg) / 2, 'PR-AUC': pr_auc}
    return out


def holm_adjust(pvalues):
    """Hiệu chỉnh Holm-Bonferroni cho nhiều phép so sánh cặp."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def bootstrap_pvalue(diff):
    """p hai phía của giả thuyết 'chênh lệch = 0', tính từ phân phối bootstrap của chênh lệch.
    Cộng 1 ở tử và mẫu để p không bao giờ bằng 0 tuyệt đối với số lần lặp hữu hạn."""
    n = len(diff)
    tail = min((diff <= 0).sum(), (diff >= 0).sum())
    return min(1.0, 2 * (tail + 1) / (n + 1))


def pairwise_tests(predictions, y_true, boot, names):
    """So sánh từng cặp mô hình bằng 2 cách, mỗi cách đều hiệu chỉnh cho nhiều phép so sánh:
    - McNemar chính xác (binomial) trên số câu đoán đúng/sai -> hiệu chỉnh Holm.
    - Paired bootstrap trên chênh lệch F1 (Toxic) -> p bootstrap hiệu chỉnh Holm,
      kèm CI 95% thường và CI đồng thời (Bonferroni) cho toàn bộ các cặp."""
    from scipy.stats import binomtest
    y = np.asarray(y_true)
    ids = list(predictions)
    n_pairs = len(ids) * (len(ids) - 1) // 2
    alpha_bonf = 0.05 / max(n_pairs, 1)
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            ca = predictions[a]['pred'].values == y
            cb = predictions[b]['pred'].values == y
            n_a_only = int((ca & ~cb).sum())  # A đúng, B sai
            n_b_only = int((~ca & cb).sum())  # A sai, B đúng
            n_disc = n_a_only + n_b_only
            p = binomtest(min(n_a_only, n_b_only), n_disc, 0.5).pvalue if n_disc else 1.0
            diff = boot[a]['F1 (Toxic)'] - boot[b]['F1 (Toxic)']
            lo, hi = np.percentile(diff, [2.5, 97.5])
            lo_adj, hi_adj = np.percentile(diff, [100 * alpha_bonf / 2, 100 * (1 - alpha_bonf / 2)])
            rows.append({
                'Mô hình A': names[a], 'Mô hình B': names[b],
                'A đúng - B sai': n_a_only, 'A sai - B đúng': n_b_only,
                'p McNemar': p,
                'Chênh F1 Toxic (A-B)': f1_score(y, predictions[a]['pred'], zero_division=0)
                                         - f1_score(y, predictions[b]['pred'], zero_division=0),
                'CI 95% thấp': lo, 'CI 95% cao': hi,
                'CI đồng thời thấp': lo_adj, 'CI đồng thời cao': hi_adj,
                'p bootstrap F1': bootstrap_pvalue(diff),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df['p McNemar (Holm)'] = holm_adjust(df['p McNemar'])
        df['p bootstrap F1 (Holm)'] = holm_adjust(df['p bootstrap F1'])
        df['Đúng/sai khác biệt (McNemar-Holm)'] = np.where(df['p McNemar (Holm)'] < 0.05, 'Có', 'Không')
        df['F1 Toxic khác biệt (Bootstrap-Holm)'] = np.where(df['p bootstrap F1 (Holm)'] < 0.05, 'Có', 'Không')
        df = df[['Mô hình A', 'Mô hình B',
                 'Chênh F1 Toxic (A-B)', 'CI 95% thấp', 'CI 95% cao', 'CI đồng thời thấp', 'CI đồng thời cao',
                 'p bootstrap F1', 'p bootstrap F1 (Holm)', 'F1 Toxic khác biệt (Bootstrap-Holm)',
                 'A đúng - B sai', 'A sai - B đúng',
                 'p McNemar', 'p McNemar (Holm)', 'Đúng/sai khác biệt (McNemar-Holm)']]
    return df


# -----------------------------------------------------------------------------
# 2c. BASELINE ĐỐI CHỨNG (SHORTCUT): CHỈ DÙNG ĐỘ DÀI CÂU / SỐ SAO
# -----------------------------------------------------------------------------
# Câu Toxic trong dữ liệu dài hơn hẳn và gần như toàn 1 sao. Hai baseline này đo xem
# riêng các tín hiệu "lối tắt" đó đạt được bao nhiêu, để biết mô hình thật học thêm được gì.
SHORTCUT_BASELINES = {
    'baseline_do_dai': ('Đối chứng: chỉ độ dài câu', lambda d: d['noi_dung_goc'].fillna('').astype(str).str.len()),
    'baseline_so_sao': ('Đối chứng: chỉ số sao', lambda d: -d['so_sao'].astype(float)),
}


def best_threshold_on_val(y_val, s_val):
    """Chọn ngưỡng tối đa F1 (Toxic) trên tập val, dự đoán Toxic khi score >= ngưỡng."""
    best_th, best_f1 = None, -1.0
    for th in np.unique(s_val):
        f1 = f1_score(y_val, (s_val >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_th, best_f1 = th, f1
    return float(best_th)


def shortcut_baselines(test_df, val_df):
    results, preds = [], {}
    for model_id, (name, score_fn) in SHORTCUT_BASELINES.items():
        th = best_threshold_on_val(val_df['label'].values, score_fn(val_df).values)
        s_test = score_fn(test_df).values
        pred = (s_test >= th).astype(int)
        m = compute_metrics(test_df['label'], pred, s_test)
        m.update({'Mã mô hình': model_id, 'Mô hình': name, 'Ngưỡng': th, 'Chọn ngưỡng trên': 'val'})
        results.append(m)
        preds[model_id] = pd.DataFrame({'review_id': test_df['review_id'], 'label': test_df['label'],
                                        'score': s_test, 'pred': pred})
    return results, preds


# -----------------------------------------------------------------------------
# 2d. PHÂN TÍCH THEO NHÓM DỮ LIỆU (SLICE ANALYSIS)
# -----------------------------------------------------------------------------
# Danh sách từ khóa toxic lấy nguyên từ TOXIC_KEYWORDS trong 3b_train_svm.py (TV1),
# bỏ các bản có dấu "_" vì ở đây so khớp trên nội dung gốc chưa tách từ.
TOXIC_KEYWORDS = [
    'đầu buồi', 'như hạch', 'vãi', 'đéo', 'lừa đảo', 'bịp', 'ăn bớt', 'đồ đểu', 'tệ bạc',
    'vô học', 'cùi bắp', 'rác rưởi', 'bố láo', 'khốn nạn', 'mất dạy', 'lừa dối', 'cẩu thả',
    'ăn cướp', 'giải tán đi', 'cút', 'chó', 'lởm', 'như cặc', 'đcm', 'dcm', 'đkm', 'dkm',
    'bịp bợm', 'lừa gạt',
]
# So khớp nguyên từ để tránh bắt nhầm (ví dụ "chó" trong "nhanh chóng")
_KEYWORD_RE = re.compile(r'(?<!\w)(' + '|'.join(re.escape(k) for k in TOXIC_KEYWORDS) + r')(?!\w)')
MIN_TOXIC_PER_CATEGORY = 5


def build_slices(test_df):
    """Trả về danh sách (nhóm tiêu chí, tên nhóm, mặt nạ boolean) trên tập test."""
    text = test_df['noi_dung_goc'].fillna('').astype(str)
    n_chars = text.str.len()
    has_kw = text.str.lower().str.contains(_KEYWORD_RE)
    stars = test_df['so_sao'].astype(int)

    slices = [
        ('Tất cả', 'Toàn bộ tập test', np.ones(len(test_df), dtype=bool)),
        ('Độ dài câu gốc', '≤ 100 ký tự', n_chars <= 100),
        ('Độ dài câu gốc', '101–300 ký tự', (n_chars > 100) & (n_chars <= 300)),
        ('Độ dài câu gốc', '> 300 ký tự', n_chars > 300),
        ('Số sao', '1 sao', stars == 1),
        ('Số sao', '2–5 sao', stars >= 2),
        ('Từ khóa toxic (lexicon TV1)', 'Có từ khóa', has_kw),
        ('Từ khóa toxic (lexicon TV1)', 'Không có từ khóa', ~has_kw),
    ]
    toxic_cat = test_df.loc[test_df['label'] == 1, 'nganh_hang'].value_counts()
    big_cats = toxic_cat[toxic_cat >= MIN_TOXIC_PER_CATEGORY].index.tolist()
    for cat in big_cats:
        slices.append(('Ngành hàng', cat, (test_df['nganh_hang'] == cat).values))
    slices.append(('Ngành hàng', f'Ngành khác (< {MIN_TOXIC_PER_CATEGORY} câu toxic)',
                   ~test_df['nganh_hang'].isin(big_cats).values))
    return [(g, name, np.asarray(mask)) for g, name, mask in slices]


def slice_analysis(predictions, test_df, names):
    y = test_df['label'].values
    rows = []
    for group, slice_name, mask in build_slices(test_df):
        n, n_tox = int(mask.sum()), int(y[mask].sum())
        for model_id, df in predictions.items():
            p = df['pred'].values[mask]
            yt = y[mask]
            tp = int(((yt == 1) & (p == 1)).sum())
            fp = int(((yt == 0) & (p == 1)).sum())
            fn = int(((yt == 1) & (p == 0)).sum())
            rows.append({
                'Tiêu chí': group, 'Nhóm': slice_name, 'Số câu': n, 'Số câu toxic': n_tox,
                'Mã mô hình': model_id, 'Mô hình': names[model_id],
                'TP': tp, 'FN': fn, 'FP': fp,
                'Recall (Toxic)': tp / n_tox if n_tox else np.nan,
                'Precision (Toxic)': tp / (tp + fp) if (tp + fp) else np.nan,
                'F1 (Toxic)': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
                'Ít mẫu (< 10 câu toxic)': 'Có' if n_tox < 10 else '',
            })
    return pd.DataFrame(rows)


def plot_slice_heatmaps(df_slice, out_path):
    """Hai heatmap cạnh nhau: Recall và Precision theo nhóm (hàng) x mô hình (cột)."""
    df_slice = df_slice.copy()
    df_slice['Hàng'] = (df_slice['Tiêu chí'] + ': ' + df_slice['Nhóm']
                        + ' (n=' + df_slice['Số câu'].astype(str)
                        + ', toxic=' + df_slice['Số câu toxic'].astype(str) + ')')
    row_order = list(dict.fromkeys(df_slice['Hàng']))
    col_order = list(dict.fromkeys(df_slice['Mô hình']))
    fig, axes = plt.subplots(1, 2, figsize=(22, 0.55 * len(row_order) + 2.5), sharey=True)
    for ax, metric in zip(axes, ['Recall (Toxic)', 'Precision (Toxic)']):
        mat = df_slice.pivot(index='Hàng', columns='Mô hình', values=metric).loc[row_order, col_order]
        vals = mat.values.astype(float)
        ax.imshow(np.nan_to_num(vals, nan=0.0), cmap=CM_CMAP, vmin=0, vmax=1, aspect='auto')
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                ax.text(j, i, '–' if np.isnan(v) else f"{v:.2f}", ha='center', va='center', fontsize=9,
                        color='white' if (not np.isnan(v) and v > 0.55) else COLOR_TEXT)
        ax.set_xticks(range(len(col_order)), [c.replace(' + ', '\n+ ').replace(' (', '\n(') for c in col_order],
                      fontsize=8.5)
        ax.set_yticks(range(len(row_order)), row_order, fontsize=9)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.tick_params(length=0, colors=COLOR_TEXT_MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle('Recall và Precision (Toxic) theo từng nhóm dữ liệu trên tập Test', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_check(path, test_df):
    print(f"Kiểm tra file: {path}")
    df, info, errors = validate_submission(path, test_df)
    if errors:
        print(" -> KHÔNG HỢP LỆ:")
        for err in errors:
            print(f"    - {err}")
        return 1
    m = compute_metrics(df['label'], df['pred'], df['score'])
    print(" -> HỢP LỆ.")
    print(f"    Ngưỡng = {info['threshold']} (chọn trên: {info['chon_nguong_tren']}) "
          f"| Siêu tham số: {json.dumps(info['sieu_tham_so'], ensure_ascii=False)}")
    print(f"    Dự đoán Toxic: {df['pred'].sum()} câu | TP={m['TP']} FP={m['FP']} FN={m['FN']} "
          f"| Macro F1={m['Macro F1']:.4f} | PR-AUC={m['PR-AUC']:.4f}")
    return 0


# -----------------------------------------------------------------------------
# 3. TRỰC QUAN HÓA
# -----------------------------------------------------------------------------
def plot_confusion_matrices(results, out_path, n_test):
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

    fig.suptitle(f'Ma trận nhầm lẫn trên tập Test ({n_test:,} mẫu)'.replace(',', '.'),
                 fontsize=14, fontweight='bold')
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


# Màu cố định theo mô hình (không theo thứ hạng), nét đường theo họ mô hình để không phụ thuộc màu
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
COLOR_BASELINE = '#8a8983'


def _line_style(model_id):
    if model_id.startswith('nb_'):
        return ':'
    if model_id.startswith('svm_'):
        return '-'
    return '-.'


def plot_pr_curves(predictions, names, baseline_preds, y_true, out_path):
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(11, 6))
    known = list(MODEL_NAMES)
    for k, (model_id, df) in enumerate(predictions.items()):
        slot = known.index(model_id) if model_id in known else k
        p, r, _ = precision_recall_curve(df['label'], df['score'])
        ap = average_precision_score(df['label'], df['score'])
        ax.plot(r, p, _line_style(model_id), color=SERIES_COLORS[slot % len(SERIES_COLORS)],
                linewidth=2, label=f"{names[model_id]} (PR-AUC {ap:.3f})", drawstyle='steps-post')
    baseline_dashes = [(0, (5, 2)), (0, (1, 1.5))]
    for k, (model_id, df) in enumerate(baseline_preds.items()):
        p, r, _ = precision_recall_curve(df['label'], df['score'])
        ap = average_precision_score(df['label'], df['score'])
        ax.plot(r, p, linestyle=baseline_dashes[k % 2], color=COLOR_BASELINE, linewidth=1.4,
                drawstyle='steps-post', label=f"{SHORTCUT_BASELINES[model_id][0]} (PR-AUC {ap:.3f})")
    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, color=COLOR_TEXT_MUTED, linewidth=0.8)
    ax.text(0.01, prevalence - 0.012, f"Đoán ngẫu nhiên = {prevalence:.3f}", ha='left', va='top',
            fontsize=9, color=COLOR_TEXT_MUTED)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('Recall (Toxic)', color=COLOR_TEXT_MUTED)
    ax.set_ylabel('Precision (Toxic)', color=COLOR_TEXT_MUTED)
    ax.set_title('Đường cong Precision–Recall trên tập Test', fontsize=13, fontweight='bold')
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT_MUTED, length=0)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=False)
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
    # CI đồng thời cho 15 cặp cần phân vị ~0.17% hai đuôi, nên cần nhiều lần lặp để ổn định
    parser.add_argument('--n-boot', type=int, default=10000, help='Số lần lặp bootstrap (mặc định: 10000).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed cho bootstrap (mặc định: 42).')
    args = parser.parse_args()

    test_df = pd.read_csv(TEST_PATH)

    if args.check:
        sys.exit(run_check(args.check, test_df))

    print("=" * 85)
    print(f"     ĐÁNH GIÁ ĐỐI SÁNH CÁC MÔ HÌNH TRÊN TẬP TEST ({len(test_df):,} MẪU)".replace(',', '.'))
    print("=" * 85)

    # --- Nạp và kiểm tra các file dự đoán ---
    paths = glob.glob(os.path.join(args.pred_dir, '*.csv'))
    known = list(MODEL_NAMES)
    paths.sort(key=lambda p: (known.index(os.path.splitext(os.path.basename(p))[0])
                              if os.path.splitext(os.path.basename(p))[0] in known else len(known),
                              os.path.basename(p)))
    if not paths:
        print(f"Không tìm thấy file dự đoán nào trong `{args.pred_dir}`.")
        sys.exit(1)

    print(f"\n[BƯỚC 1] Kiểm tra {len(paths)} file dự đoán trong `{args.pred_dir}`...")
    predictions = {}
    infos = {}
    for path in paths:
        model_id = os.path.splitext(os.path.basename(path))[0]
        df, info, errors = validate_submission(path, test_df)
        if errors:
            print(f" [BỎ QUA] {model_id}: " + ' | '.join(errors))
            continue
        predictions[model_id] = df
        infos[model_id] = info
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
        m['Nhóm mô hình'] = infos[model_id].get('mo_hinh', infos[model_id].get('nguoi_phu_trach', 'NLP Model'))
        m['Ngưỡng'] = infos[model_id]['threshold']
        m['Chọn ngưỡng trên'] = infos[model_id]['chon_nguong_tren']
        m['Siêu tham số'] = json.dumps(infos[model_id]['sieu_tham_so'], ensure_ascii=False)
        m['Random seed'] = infos[model_id]['random_seed']
        results.append(m)

    metric_cols = ['Accuracy', 'Precision (Toxic)', 'Recall (Toxic)',
                   'F1 (Toxic)', 'Macro F1', 'PR-AUC', 'TP', 'FP', 'FN', 'TN']
    config_cols = ['Nhóm mô hình', 'Ngưỡng', 'Chọn ngưỡng trên', 'Siêu tham số', 'Random seed']
    df_metrics = pd.DataFrame(results)[['Mã mô hình', 'Mô hình'] + metric_cols + config_cols]
    print("\n" + df_metrics[['Mô hình'] + metric_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nCấu hình từng mô hình:")
    print(df_metrics[['Mô hình'] + config_cols].to_string(index=False))

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, 'tv4_model_comparison.csv')
    df_metrics.round(4).to_csv(metrics_path, index=False, encoding='utf-8-sig')

    # --- Biểu đồ ---
    print("\n[BƯỚC 3] Vẽ biểu đồ...")
    cm_path = os.path.join(args.out_dir, '5_confusion_matrices.png')
    cmp_path = os.path.join(args.out_dir, '6_model_comparison.png')
    plot_confusion_matrices(results, cm_path, len(test_df))
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

    # --- Kiểm định thống kê ---
    print(f"\n[BƯỚC 5] Bootstrap {args.n_boot} lần (seed={args.seed}) và kiểm định McNemar từng cặp...")
    names = {mid: MODEL_NAMES.get(mid, mid) for mid in predictions}
    y_test = test_df['label'].values
    boot = bootstrap_metrics(predictions, y_test, args.n_boot, args.seed)
    ci_rows = []
    for r in results:
        row = {'Mã mô hình': r['Mã mô hình'], 'Mô hình': r['Mô hình']}
        for metric in ['F1 (Toxic)', 'Macro F1', 'PR-AUC']:
            lo, hi = np.percentile(boot[r['Mã mô hình']][metric], [2.5, 97.5])
            row[metric] = r[metric]
            row[f'{metric} CI thấp'] = lo
            row[f'{metric} CI cao'] = hi
        ci_rows.append(row)
    df_ci = pd.DataFrame(ci_rows)
    for _, row in df_ci.iterrows():
        print(f"    {row['Mô hình']:<34} F1 Toxic {row['F1 (Toxic)']:.4f} "
              f"[{row['F1 (Toxic) CI thấp']:.3f}, {row['F1 (Toxic) CI cao']:.3f}] | "
              f"PR-AUC {row['PR-AUC']:.4f} [{row['PR-AUC CI thấp']:.3f}, {row['PR-AUC CI cao']:.3f}]")
    ci_path = os.path.join(args.out_dir, 'tv4_bootstrap_ci.csv')
    df_ci.round(4).to_csv(ci_path, index=False, encoding='utf-8-sig')

    df_mcn = pairwise_tests(predictions, y_test, boot, names)
    mcn_path = os.path.join(args.out_dir, 'tv4_pairwise_tests.csv')
    df_mcn.round(4).to_csv(mcn_path, index=False, encoding='utf-8-sig')
    if len(df_mcn):
        n_f1 = (df_mcn['F1 Toxic khác biệt (Bootstrap-Holm)'] == 'Có').sum()
        n_mcn = (df_mcn['Đúng/sai khác biệt (McNemar-Holm)'] == 'Có').sum()
        print(f" -> {len(df_mcn)} cặp. Sau hiệu chỉnh Holm: {n_f1} cặp khác biệt về F1 Toxic (bootstrap), "
              f"{n_mcn} cặp khác biệt về số câu đúng/sai (McNemar).")
        show = df_mcn[['Mô hình A', 'Mô hình B', 'Chênh F1 Toxic (A-B)', 'CI đồng thời thấp', 'CI đồng thời cao',
                       'p bootstrap F1 (Holm)', 'p McNemar (Holm)']]
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- Baseline đối chứng ---
    print("\n[BƯỚC 6] Baseline đối chứng (ngưỡng chọn trên tập val)...")
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    base_results, base_preds = shortcut_baselines(test_df, val_df)
    base_cols = ['Mã mô hình', 'Mô hình', 'Ngưỡng', 'Chọn ngưỡng trên'] + metric_cols
    df_base = pd.DataFrame(base_results)[base_cols]
    print(df_base[['Mô hình', 'Ngưỡng'] + metric_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("    (Baseline số sao dùng score = -số sao: ngưỡng -1 nghĩa là đoán Toxic khi số sao <= 1.)")
    base_path = os.path.join(args.out_dir, 'tv4_shortcut_baselines.csv')
    df_base.round(4).to_csv(base_path, index=False, encoding='utf-8-sig')

    pr_path = os.path.join(args.out_dir, '7_pr_curves.png')
    plot_pr_curves(predictions, names, base_preds, y_test, pr_path)

    # --- Phân tích theo nhóm dữ liệu ---
    print("\n[BƯỚC 7] Phân tích theo nhóm dữ liệu (độ dài, số sao, từ khóa, ngành hàng)...")
    df_slice = slice_analysis(predictions, test_df, names)
    slice_path = os.path.join(args.out_dir, 'tv4_slice_analysis.csv')
    df_slice.round(4).to_csv(slice_path, index=False, encoding='utf-8-sig')
    recall_view = df_slice.pivot_table(index=['Tiêu chí', 'Nhóm', 'Số câu', 'Số câu toxic'], columns='Mã mô hình',
                                       values='Recall (Toxic)', sort=False)[list(predictions)]
    print(" Recall (Toxic) theo nhóm:")
    print(recall_view.to_string(float_format=lambda v: f"{v:.2f}"))
    slice_png = os.path.join(args.out_dir, '8_slice_analysis.png')
    plot_slice_heatmaps(df_slice, slice_png)

    print("\nĐã lưu:")
    for p in [metrics_path, cm_path, cmp_path, errors_path, ci_path, mcn_path, base_path, pr_path,
              slice_path, slice_png]:
        print(f" - {p}")


if __name__ == '__main__':
    main()
