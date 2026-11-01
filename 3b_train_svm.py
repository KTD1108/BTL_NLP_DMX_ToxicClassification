# -*- coding: utf-8 -*-
"""
THÀNH VIÊN 1: DATA ARCHITECT & SVM SPECIALIST
Script: 3b_train_svm.py (Phiên bản nâng cấp FeatureUnion & Auto Threshold Tuning)

Nhiệm vụ theo 5 yêu cầu kỹ thuật:
1. Chuẩn hóa từ lóng & Teencode e-commerce + Trích xuất đặc trưng từ khóa độc hại (ToxicLexiconFeatureExtractor).
2. FeatureUnion 3 nhánh:
   - word_tfidf: Word n-grams (1, 2), max_features=10000, sublinear_tf=True, min_df=2.
   - char_tfidf: Character n-grams char_wb (3, 5), max_features=15000, sublinear_tf=True, min_df=2.
   - lexicon_feat: Bộ đếm từ khóa độc hại (Lexicon count & binary flag).
3. GridSearchCV với StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='f1_macro':
   - 'C': [0.05, 0.1, 0.25, 0.5, 1.0]
   - 'class_weight': [{0: 1, 1: 5}, {0: 1, 1: 10}, {0: 1, 1: 15}, {0: 1, 1: 20}, 'balanced']
4. Tự động tìm ngưỡng quyết định tối ưu (Auto Threshold Tuning) trên tập Train bằng cross_val_predict:
   - Quét threshold từ -0.50 đến +0.50 (bước nhảy 0.02) để tối ưu Macro F1 trên tập Train (Zero data leakage).
5. Xuất kết quả đối sánh toàn diện trên tập Test độc lập (test.csv - 2.000 mẫu):
   - So sánh trực quan Ngưỡng mặc định (0.0) vs Ngưỡng tối ưu tự động.
   - Lưu trữ Model Pipeline (.joblib), CSV dự đoán, CSV chỉ số và biểu đồ Confusion Matrix sắc nét (300 DPI).
"""

import os
import sys
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình thẩm mỹ đồ thị Matplotlib
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font='DejaVu Sans')

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("=" * 85)
print("   THÀNH VIÊN 1: NÂNG CẤP LINEAR SVM - FEATUREUNION (WORD + CHAR + LEXICON)")
print("                 & AUTO THRESHOLD TUNING TRÊN DỮ LIỆU MẤT CÂN BẰNG (1:49)")
print("=" * 85)

# -----------------------------------------------------------------------------
# 1. TÍCH HỢP MODULE TIỀN XỬ LÝ NLP_UTILS & BỘ TRÍCH XUẤT LEXICON FEATURE
# -----------------------------------------------------------------------------
import nlp_utils

def normalize_slang(text):
    """Sử dụng hàm chuẩn hóa từ lóng & teencode đồng bộ từ module dùng chung nlp_utils."""
    return nlp_utils.normalize_slang(text)

TOXIC_KEYWORDS = [
    'đầu buồi', 'như hạch', 'vãi', 'đéo', 'lừa đảo', 'lừa_đảo', 'bịp',
    'ăn bớt', 'ăn_bớt', 'đồ đểu', 'đồ_đểu', 'tệ bạc', 'tệ_bạc',
    'vô học', 'vô_học', 'cùi bắp', 'cùi_bắp', 'rác rưởi', 'rác_rưởi',
    'bố láo', 'bố_láo', 'khốn nạn', 'khốn_nạn', 'mất dạy', 'mất_dạy',
    'lừa dối', 'lừa_dối', 'cẩu thả', 'cẩu_thả', 'ăn cướp', 'ăn_cướp',
    'giải tán đi', 'giải_tán', 'cút', 'chó', 'lởm', 'như cặc',
    'đcm', 'dcm', 'đkm', 'dkm', 'bịp bợm', 'lừa gạt'
]

class ToxicLexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    """CustomTransformer trích xuất số lượng và cờ nhận diện từ khóa độc hại/chửi thề."""
    def __init__(self, keywords=None, scale=2.5):
        self.keywords = keywords or TOXIC_KEYWORDS
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            t = str(text).lower()
            count = sum(1 for kw in self.keywords if kw in t)
            has_kw = 1.0 if count > 0 else 0.0
            features.append([float(count) * self.scale, has_kw * self.scale])
        return csr_matrix(features)

# -----------------------------------------------------------------------------
# 2. NẠP DỮ LIỆU & ÁP DỤNG CHUẨN HÓA TỪ LÓNG
# -----------------------------------------------------------------------------
print("\n[BƯỚC 1] Đang nạp và áp dụng tiền xử lý từ lóng trên các tập dữ liệu...")
df_train = pd.read_csv('data/train.csv')
df_val = pd.read_csv('data/val.csv')
df_test = pd.read_csv('data/test.csv')

X_train_raw = df_train['noi_dung_tien_xu_ly'].fillna('').astype(str)
y_train = df_train['label'].astype(int)

X_test_raw = df_test['noi_dung_tien_xu_ly'].fillna('').astype(str)
y_test = df_test['label'].astype(int)

# Áp dụng chuẩn hóa từ lóng
X_train = X_train_raw.apply(normalize_slang)
X_test = X_test_raw.apply(normalize_slang)

print(f" -> Tập Train: {len(X_train)} mẫu (Toxic: {y_train.sum()} ~ {y_train.mean()*100:.2f}%)")
print(f" -> Tập Test:  {len(X_test)} mẫu (Toxic: {y_test.sum()} ~ {y_test.mean()*100:.2f}%)")
print(f" -> Danh sách từ khóa độc hại theo dõi: {len(TOXIC_KEYWORDS)} cụm từ")

# -----------------------------------------------------------------------------
# 3. THIẾT KẾ PIPELINE FEATUREUNION (WORD + CHAR + LEXICON)
# -----------------------------------------------------------------------------
print("\n[BƯỚC 2] Khởi tạo Pipeline FeatureUnion 3 nhánh đặc trưng...")

feature_union = FeatureUnion([
    ('word_tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2
    )),
    ('char_tfidf', TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=15000,
        sublinear_tf=True,
        min_df=2
    )),
    ('lexicon_feat', ToxicLexiconFeatureExtractor(keywords=TOXIC_KEYWORDS, scale=2.5))
])

full_pipeline = Pipeline([
    ('features', feature_union),
    ('clf', LinearSVC(random_state=42, max_iter=5000))
])

# -----------------------------------------------------------------------------
# 4. TỐI ƯU SIÊU THAM SỐ BẰNG GRIDSEARCHCV
# -----------------------------------------------------------------------------
print("\n[BƯỚC 3] Thực thi GridSearchCV với StratifiedKFold (5-folds, scoring='f1_macro')...")

param_grid = {
    'clf__C': [0.05, 0.1, 0.25, 0.5, 1.0],
    'clf__class_weight': [{0: 1, 1: 5}, {0: 1, 1: 10}, {0: 1, 1: 15}, {0: 1, 1: 20}, 'balanced']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f" -> Bộ siêu tham số tối ưu nhất: {grid_search.best_params_}")
print(f" -> Điểm Macro F1 trung bình tốt nhất trên CV 5-folds (Train): {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_

# -----------------------------------------------------------------------------
# 5. TỰ ĐỘNG TÌM NGƯỠNG QUYẾT ĐỊNH TỐI ƯU (AUTO THRESHOLD TUNING) TRÊN TRAIN
# -----------------------------------------------------------------------------
print("\n[BƯỚC 4] Tự động tìm ngưỡng quyết định tối ưu (Auto Threshold Tuning) trên tập Train...")
print(" -> Sử dụng cross_val_predict(method='decision_function') để tính điểm không thiên vị...")

cv_decision_scores = cross_val_predict(
    best_pipeline,
    X_train,
    y_train,
    cv=cv,
    method='decision_function',
    n_jobs=-1
)

threshold_range = np.arange(-0.50, 0.52, 0.02)
best_threshold = 0.0
best_cv_macro_f1 = -1.0
cv_tuning_history = []

for th in threshold_range:
    th = round(th, 2)
    preds_th = (cv_decision_scores >= th).astype(int)
    f1_macro_th = f1_score(y_train, preds_th, average='macro', zero_division=0)
    f1_toxic_th = f1_score(y_train, preds_th, pos_label=1, zero_division=0)
    recall_toxic_th = recall_score(y_train, preds_th, pos_label=1, zero_division=0)
    precision_toxic_th = precision_score(y_train, preds_th, pos_label=1, zero_division=0)

    cv_tuning_history.append({
        'threshold': th,
        'macro_f1': f1_macro_th,
        'toxic_f1': f1_toxic_th,
        'toxic_recall': recall_toxic_th,
        'toxic_precision': precision_toxic_th
    })

    if f1_macro_th > best_cv_macro_f1:
        best_cv_macro_f1 = f1_macro_th
        best_threshold = th

print(f" -> Ngưỡng tối ưu tự động tìm được trên Train: Threshold = {best_threshold:+.2f}")
print(f" -> Điểm Train Macro F1 tại ngưỡng tối ưu: {best_cv_macro_f1:.4f} (Không tiếp xúc tập Test!)")

# -----------------------------------------------------------------------------
# 6. ĐÁNH GIÁ ĐỐI SÁNH TRỰC QUAN TRÊN TẬP TEST ĐỘC LẬP (2.000 MẪU)
# -----------------------------------------------------------------------------
print("\n" + "=" * 85)
print("     [BƯỚC 5] ĐÁNH GIÁ ĐỐI SÁNH TRÊN TẬP TEST ĐỘC LẬP (2.000 MẪU - TEST.CSV)")
print("=" * 85)

# Tính decision_function trên tập Test một lần duy nhất
test_decision_scores = best_pipeline.decision_function(X_test)

def compute_metrics(y_true, scores, threshold, setting_name):
    preds = (scores >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    prec0 = precision_score(y_true, preds, pos_label=0)
    rec0 = recall_score(y_true, preds, pos_label=0)
    f1_0 = f1_score(y_true, preds, pos_label=0)

    prec1 = precision_score(y_true, preds, pos_label=1, zero_division=0)
    rec1 = recall_score(y_true, preds, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, preds, pos_label=1, zero_division=0)

    macro_f1 = f1_score(y_true, preds, average='macro')
    weighted_f1 = f1_score(y_true, preds, average='weighted')
    cm = confusion_matrix(y_true, preds)

    return {
        'Cấu hình Ngưỡng': setting_name,
        'Ngưỡng (Threshold)': f"{threshold:+.2f}",
        'Accuracy': f"{acc*100:.2f}%",
        'Precision (Toxic)': f"{prec1*100:.2f}%",
        'Recall (Toxic)': f"{rec1*100:.2f}%",
        'F1-Score (Toxic)': f"{f1_1:.4f}",
        'Macro F1-Score': f"{macro_f1:.4f}",
        'Weighted F1-Score': f"{weighted_f1:.4f}",
        'TP (Bắt đúng)': cm[1,1],
        'FP (Bắt nhầm)': cm[0,1],
        'FN (Bỏ sót)': cm[1,0],
        'TN': cm[0,0],
        '_raw_preds': preds,
        '_raw_cm': cm
    }

eval_default = compute_metrics(y_test, test_decision_scores, 0.0, "Ngưỡng mặc định (Default 0.0)")
eval_optimal = compute_metrics(y_test, test_decision_scores, best_threshold, f"Ngưỡng tối ưu tự động (Tuned {best_threshold:+.2f})")

# Bảng so sánh trực quan
df_compare = pd.DataFrame([eval_default, eval_optimal])
display_cols = [
    'Cấu hình Ngưỡng', 'Ngưỡng (Threshold)', 'Accuracy',
    'Precision (Toxic)', 'Recall (Toxic)', 'F1-Score (Toxic)',
    'Macro F1-Score', 'Weighted F1-Score'
]
print("\n" + df_compare[display_cols].to_string(index=False))

print("\nChi tiết Ma trận nhầm lẫn (Confusion Matrix):")
print(f"1. {eval_default['Cấu hình Ngưỡng']}:")
print(f"   - Bắt đúng Toxic (TP): {eval_default['TP (Bắt đúng)']}/40 câu | Bắt nhầm (FP): {eval_default['FP (Bắt nhầm)']} câu | Bỏ sót (FN): {eval_default['FN (Bỏ sót)']} câu")
print(f"2. {eval_optimal['Cấu hình Ngưỡng']}:")
print(f"   - Bắt đúng Toxic (TP): {eval_optimal['TP (Bắt đúng)']}/40 câu | Bắt nhầm (FP): {eval_optimal['FP (Bắt nhầm)']} câu | Bỏ sót (FN): {eval_optimal['FN (Bỏ sót)']} câu")

# -----------------------------------------------------------------------------
# 7. TRỰC QUAN HÓA MA TRẬN NHẦM LẪN (300 DPI)
# -----------------------------------------------------------------------------
print("\n[BƯỚC 6] Đang vẽ và xuất biểu đồ Ma trận nhầm lẫn đối sánh 300 DPI...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

for ax, res_dict in zip(axes, [eval_default, eval_optimal]):
    cm = res_dict['_raw_cm']
    title = f"{res_dict['Cấu hình Ngưỡng']}\nF1(Toxic)={res_dict['F1-Score (Toxic)']} | Macro-F1={res_dict['Macro F1-Score']}"
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        ax=ax,
        annot_kws={'size': 14, 'weight': 'bold'},
        xticklabels=['Non-toxic (0)', 'Toxic (1)'],
        yticklabels=['Non-toxic (0)', 'Toxic (1)']
    )
    ax.set_title(title, fontsize=11, pad=12, fontweight='bold')
    ax.set_xlabel('Predicted Label (Nhãn Dự Đoán)', fontsize=11, labelpad=8)
    ax.set_ylabel('True Label (Nhãn Thực Tế)', fontsize=11, labelpad=8)

plt.suptitle('So Sánh Ma Trận Nhầm Lẫn Trên Tập Test (2.000 Mẫu) - Linear SVM Nâng Cấp', fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
cm_plot_path = 'results/tv1_svm_confusion_matrices.png'
plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f" -> Đã lưu biểu đồ: {cm_plot_path}")

# -----------------------------------------------------------------------------
# 8. LƯU ARTIFACTS: MODEL PIPELINE, DỰ ĐOÁN VÀ CHỈ SỐ
# -----------------------------------------------------------------------------
print("\n[BƯỚC 7] Đang lưu trữ toàn bộ Artifacts...")

# Lưu toàn bộ Pipeline
model_save_path = 'models/best_svm_pipeline.joblib'
joblib.dump(best_pipeline, model_save_path)
print(f" -> Đã lưu Pipeline hoàn chỉnh vào: {model_save_path}")

# Lưu kết quả dự đoán chi tiết 2.000 mẫu
df_pred_test = pd.DataFrame({
    'stt': range(1, len(df_test) + 1),
    'review_id': df_test['review_id'],
    'so_sao': df_test['so_sao'],
    'noi_dung_goc': df_test['noi_dung_goc'],
    'noi_dung_chuan_hoa_slang': X_test,
    'true_label': y_test,
    'decision_score': np.round(test_decision_scores, 4),
    'pred_default_th0': eval_default['_raw_preds'],
    'pred_optimal_th': eval_optimal['_raw_preds']
})
pred_path = 'results/tv1_svm_test_predictions.csv'
df_pred_test.to_csv(pred_path, index=False, encoding='utf-8')
print(f" -> Đã lưu kết quả dự đoán 2.000 mẫu tập Test: {pred_path}")

# Lưu file tóm tắt chỉ số số liệu
summary_path = 'results/tv1_svm_metrics.csv'
df_save_metrics = df_compare.drop(columns=['_raw_preds', '_raw_cm'])
df_save_metrics.to_csv(summary_path, index=False, encoding='utf-8')
print(f" -> Đã lưu bảng chỉ số hiệu năng: {summary_path}")

print("\n" + "=" * 85)
print("     HOÀN TẤT NÂNG CẤP TOÀN BỘ 5 YÊU CẦU KỸ THUẬT CỦA THÀNH VIÊN 1!")
print("=" * 85)
