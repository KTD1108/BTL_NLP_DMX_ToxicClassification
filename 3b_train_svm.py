# -*- coding: utf-8 -*-
"""
THÀNH VIÊN 1: DATA ARCHITECT & SVM SPECIALIST
Script: 3b_train_svm.py (Phiên bản toàn diện: Đối sánh BoW vs TF-IDF đơn vs TF-IDF Nâng cấp)

Nhiệm vụ kỹ thuật hoàn chỉnh:
1. Huấn luyện Mô hình Cơ sở 1: BoW (CountVectorizer) + Linear SVM (class_weight='balanced').
2. Huấn luyện Mô hình Cơ sở 2: TF-IDF đơn (TfidfVectorizer) + Linear SVM (class_weight='balanced').
3. Huấn luyện Mô hình Nâng cấp 3: FeatureUnion 3 nhánh (Word TF-IDF + Char_wb TF-IDF + Lexicon)
   kết hợp GridSearchCV (5-Folds) và Auto Decision Threshold Tuning trên tập Train.
4. Đánh giá đối sánh toàn diện cả 3 mô hình trên tập Test độc lập (test.csv - 2.000 mẫu).
5. Lưu trữ Artifacts: Models (.joblib), CSV dự đoán, CSV chỉ số và biểu đồ Confusion Matrix (300 DPI).
"""

import os
import sys
import re
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import nlp_utils

# Cấu hình thẩm mỹ đồ thị Matplotlib
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font='DejaVu Sans')

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("=" * 88)
print("   THÀNH VIÊN 1: ĐỐI SÁNH MÔ HÌNH LINEAR SVM (BoW vs TF-IDF ĐƠN vs TF-IDF NÂNG CẤP)")
print("                 & AUTO THRESHOLD TUNING TRÊN DỮ LIỆU MẤT CÂN BẰNG (1:49)")
print("=" * 88)

# -----------------------------------------------------------------------------
# 1. TÍCH HỢP MODULE TIỀN XỬ LÝ NLP_UTILS & BỘ TRÍCH XUẤT LEXICON FEATURE
# -----------------------------------------------------------------------------
def normalize_slang(text):
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
print("\n[BƯỚC 1] Đang nạp và chuẩn hóa từ lóng trên các tập dữ liệu...")
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

# -----------------------------------------------------------------------------
# 3. HUẤN LUYỆN MÔ HÌNH CƠ SỞ 1: BoW (COUNTVECTORIZER) + LINEAR SVM
# -----------------------------------------------------------------------------
print("\n[BƯỚC 2] Huấn luyện Mô hình Cơ sở 1: BoW (CountVectorizer) + Linear SVM...")
bow_pipeline = Pipeline([
    ('bow', CountVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)),
    ('clf', LinearSVC(class_weight='balanced', random_state=42, max_iter=5000))
])
bow_pipeline.fit(X_train, y_train)
preds_bow = bow_pipeline.predict(X_test)
cm_bow = confusion_matrix(y_test, preds_bow)

metrics_bow = {
    'Mô hình / Cấu hình': '1. BoW (CountVectorizer) + Linear SVM',
    'Không gian đặc trưng': 'Bag-of-Words Word (1, 2)',
    'Ngưỡng': '0.00',
    'Accuracy': f"{accuracy_score(y_test, preds_bow)*100:.2f}%",
    'Precision (Toxic)': f"{precision_score(y_test, preds_bow, pos_label=1, zero_division=0)*100:.2f}%",
    'Recall (Toxic)': f"{recall_score(y_test, preds_bow, pos_label=1, zero_division=0)*100:.2f}%",
    'F1-Score (Toxic)': f"{f1_score(y_test, preds_bow, pos_label=1, zero_division=0):.4f}",
    'Macro F1-Score': f"{f1_score(y_test, preds_bow, average='macro'):.4f}",
    'TP': cm_bow[1, 1], 'FP': cm_bow[0, 1], 'FN': cm_bow[1, 0], 'TN': cm_bow[0, 0]
}
print(f" -> BoW Result: Accuracy={metrics_bow['Accuracy']} | Prec={metrics_bow['Precision (Toxic)']} | Rec={metrics_bow['Recall (Toxic)']} | Toxic F1={metrics_bow['F1-Score (Toxic)']} | Macro F1={metrics_bow['Macro F1-Score']}")
print(f"    TP: {cm_bow[1,1]}/40 câu, FP: {cm_bow[0,1]} câu, FN: {cm_bow[1,0]} câu")

# -----------------------------------------------------------------------------
# 4. HUẤN LUYỆN MÔ HÌNH CƠ SỞ 2: TF-IDF ĐƠN + LINEAR SVM
# -----------------------------------------------------------------------------
print("\n[BƯỚC 3] Huấn luyện Mô hình Cơ sở 2: TF-IDF Đơn (TfidfVectorizer) + Linear SVM...")
tfidf_single_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, min_df=2)),
    ('clf', LinearSVC(class_weight='balanced', random_state=42, max_iter=5000))
])
tfidf_single_pipeline.fit(X_train, y_train)
preds_tfidf = tfidf_single_pipeline.predict(X_test)
cm_tfidf = confusion_matrix(y_test, preds_tfidf)

metrics_tfidf = {
    'Mô hình / Cấu hình': '2. TF-IDF đơn (Word Level) + Linear SVM',
    'Không gian đặc trưng': 'TF-IDF Word (1, 2) + L2 Norm',
    'Ngưỡng': '0.00',
    'Accuracy': f"{accuracy_score(y_test, preds_tfidf)*100:.2f}%",
    'Precision (Toxic)': f"{precision_score(y_test, preds_tfidf, pos_label=1, zero_division=0)*100:.2f}%",
    'Recall (Toxic)': f"{recall_score(y_test, preds_tfidf, pos_label=1, zero_division=0)*100:.2f}%",
    'F1-Score (Toxic)': f"{f1_score(y_test, preds_tfidf, pos_label=1, zero_division=0):.4f}",
    'Macro F1-Score': f"{f1_score(y_test, preds_tfidf, average='macro'):.4f}",
    'TP': cm_tfidf[1, 1], 'FP': cm_tfidf[0, 1], 'FN': cm_tfidf[1, 0], 'TN': cm_tfidf[0, 0]
}
print(f" -> TF-IDF Result: Accuracy={metrics_tfidf['Accuracy']} | Prec={metrics_tfidf['Precision (Toxic)']} | Rec={metrics_tfidf['Recall (Toxic)']} | Toxic F1={metrics_tfidf['F1-Score (Toxic)']} | Macro F1={metrics_tfidf['Macro F1-Score']}")
print(f"    TP: {cm_tfidf[1,1]}/40 câu, FP: {cm_tfidf[0,1]} câu, FN: {cm_tfidf[1,0]} câu")

# -----------------------------------------------------------------------------
# 5. HUẤN LUYỆN MÔ HÌNH NÂNG CẤP 3: FEATUREUNION 3 NHÁNH & GRIDSEARCHCV
# -----------------------------------------------------------------------------
print("\n[BƯỚC 4] Huấn luyện Mô hình Nâng cấp 3: FeatureUnion 3 nhánh & GridSearchCV...")

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
    verbose=0
)

grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_

print(f" -> Siêu tham số tối ưu (GridSearchCV): {grid_search.best_params_}")
print(f" -> Điểm Macro F1 trung bình trên 5-Folds CV (Train): {grid_search.best_score_:.4f}")

# -----------------------------------------------------------------------------
# 6. AUTO THRESHOLD TUNING TRÊN TẬP TRAIN (ZERO DATA LEAKAGE)
# -----------------------------------------------------------------------------
print("\n[BƯỚC 5] Tự động tìm ngưỡng quyết định tối ưu (Auto Threshold Tuning) trên Train...")
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

for th in threshold_range:
    th = round(th, 2)
    preds_th = (cv_decision_scores >= th).astype(int)
    f1_macro_th = f1_score(y_train, preds_th, average='macro', zero_division=0)
    if f1_macro_th > best_cv_macro_f1:
        best_cv_macro_f1 = f1_macro_th
        best_threshold = th

print(f" -> Ngưỡng tối ưu tự động trên Train: Threshold = {best_threshold:+.2f} (Macro F1 = {best_cv_macro_f1:.4f})")

# -----------------------------------------------------------------------------
# 7. ĐÁNH GIÁ ĐỐI SÁNH TRÊN TẬP TEST ĐỘC LẬP (2.000 MẪU)
# -----------------------------------------------------------------------------
print("\n[BƯỚC 6] Đánh giá trên tập Test độc lập (2.000 mẫu)...")
test_decision_scores = best_pipeline.decision_function(X_test)

def compute_metrics(y_true, scores, threshold, setting_name):
    preds = (scores >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    prec1 = precision_score(y_true, preds, pos_label=1, zero_division=0)
    rec1 = recall_score(y_true, preds, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y_true, preds, average='macro')
    cm = confusion_matrix(y_true, preds)

    return {
        'Mô hình / Cấu hình': setting_name,
        'Không gian đặc trưng': 'FeatureUnion (Word + Char_wb + Lexicon)',
        'Ngưỡng': f"{threshold:+.2f}",
        'Accuracy': f"{acc*100:.2f}%",
        'Precision (Toxic)': f"{prec1*100:.2f}%",
        'Recall (Toxic)': f"{rec1*100:.2f}%",
        'F1-Score (Toxic)': f"{f1_1:.4f}",
        'Macro F1-Score': f"{macro_f1:.4f}",
        'TP': cm[1,1], 'FP': cm[0,1], 'FN': cm[1,0], 'TN': cm[0,0],
        '_raw_preds': preds,
        '_raw_cm': cm
    }

eval_upgraded_default = compute_metrics(y_test, test_decision_scores, 0.0, "3. Linear SVM Nâng cấp (Ngưỡng 0.00)")
eval_upgraded_optimal = compute_metrics(y_test, test_decision_scores, best_threshold, f"4. Linear SVM Nâng cấp (Ngưỡng tối ưu {best_threshold:+.2f})")

# Tổng kết Bảng so sánh 4 cấu hình
all_results = [metrics_bow, metrics_tfidf, eval_upgraded_default, eval_upgraded_optimal]
df_summary = pd.DataFrame(all_results)

display_cols = [
    'Mô hình / Cấu hình', 'Ngưỡng', 'Accuracy',
    'Precision (Toxic)', 'Recall (Toxic)', 'F1-Score (Toxic)',
    'Macro F1-Score', 'TP', 'FP', 'FN'
]

print("\n" + "=" * 88)
print("             BẢNG TỔNG KẾT ĐỐI SÁNH HIỆU NĂNG CÁC BIẾN THỂ LINEAR SVM")
print("=" * 88)
print(df_summary[display_cols].to_string(index=False))

# -----------------------------------------------------------------------------
# 8. TRỰC QUAN HÓA MA TRẬN NHẦM LẪN (300 DPI)
# -----------------------------------------------------------------------------
print("\n[BƯỚC 7] Đang vẽ và xuất biểu đồ Ma trận nhầm lẫn đối sánh...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

for ax, res_dict in zip(axes, [eval_upgraded_default, eval_upgraded_optimal]):
    cm = res_dict['_raw_cm']
    title = f"{res_dict['Mô hình / Cấu hình']}\nF1(Toxic)={res_dict['F1-Score (Toxic)']} | Macro-F1={res_dict['Macro F1-Score']}"
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
# 9. LƯU ARTIFACTS
# -----------------------------------------------------------------------------
print("\n[BƯỚC 8] Đang lưu trữ toàn bộ Artifacts...")
joblib.dump(best_pipeline, 'models/best_svm_pipeline.joblib')
joblib.dump(bow_pipeline, 'models/baseline_bow_svm.joblib')
joblib.dump(tfidf_single_pipeline, 'models/baseline_tfidf_svm.joblib')
print(" -> Đã lưu 3 mô hình (.joblib) vào thư mục models/")

# Lưu dự đoán 2.000 mẫu test
df_pred_test = pd.DataFrame({
    'stt': range(1, len(df_test) + 1),
    'review_id': df_test['review_id'],
    'so_sao': df_test['so_sao'],
    'noi_dung_goc': df_test['noi_dung_goc'],
    'noi_dung_chuan_hoa_slang': X_test,
    'true_label': y_test,
    'pred_bow_svm': preds_bow,
    'pred_tfidf_single_svm': preds_tfidf,
    'decision_score_upgraded': np.round(test_decision_scores, 4),
    'pred_upgraded_th0': eval_upgraded_default['_raw_preds'],
    'pred_upgraded_opt': eval_upgraded_optimal['_raw_preds']
})
pred_path = 'results/tv1_svm_test_predictions.csv'
df_pred_test.to_csv(pred_path, index=False, encoding='utf-8')
print(f" -> Đã lưu kết quả dự đoán chi tiết 2.000 mẫu: {pred_path}")

# Lưu metrics CSV
summary_path = 'results/tv1_svm_metrics.csv'
df_save_metrics = df_summary.drop(columns=['_raw_preds', '_raw_cm'], errors='ignore')
df_save_metrics.to_csv(summary_path, index=False, encoding='utf-8')
print(f" -> Đã lưu bảng tổng kết metrics: {summary_path}")

# -----------------------------------------------------------------------------
# 10. XUẤT FILE DỰ ĐOÁN NỘP CHO THÀNH VIÊN 4 THEO QUY_UOC_FILE_DU_DOAN.md
# -----------------------------------------------------------------------------
print("\n[BƯỚC 9] Xuất file dự đoán chuẩn hóa nộp TV4 (results/predictions/)...")
pred_dir = os.path.join('results', 'predictions')
os.makedirs(pred_dir, exist_ok=True)

# 1. svm_bow (BoW + Linear SVM)
scores_bow = np.round(bow_pipeline.decision_function(X_test), 4)
th_bow = 0.0
pd.DataFrame({
    'review_id': df_test['review_id'],
    'label': y_test,
    'score': scores_bow,
    'pred': (scores_bow >= th_bow).astype(int)
}).to_csv(os.path.join(pred_dir, 'svm_bow.csv'), index=False, encoding='utf-8')

with open(os.path.join(pred_dir, 'svm_bow_info.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'nguoi_phu_trach': 'TV1',
        'threshold': float(th_bow),
        'chon_nguong_tren': 'mac_dinh',
        'sieu_tham_so': {
            'ngram_range': [1, 2],
            'max_features': 10000,
            'class_weight': 'balanced'
        },
        'random_seed': 42,
        'ghi_chu': 'Mô hình cơ sở BoW (CountVectorizer) + Linear SVM (class_weight=balanced)'
    }, f, ensure_ascii=False, indent=2)
print(" -> Đã xuất: results/predictions/svm_bow.csv & svm_bow_info.json")

# 2. svm_tfidf (TF-IDF FeatureUnion nâng cấp + Linear SVM)
scores_tfidf = np.round(test_decision_scores, 4)
th_tfidf = 0.0
pd.DataFrame({
    'review_id': df_test['review_id'],
    'label': y_test,
    'score': scores_tfidf,
    'pred': (scores_tfidf >= th_tfidf).astype(int)
}).to_csv(os.path.join(pred_dir, 'svm_tfidf.csv'), index=False, encoding='utf-8')

with open(os.path.join(pred_dir, 'svm_tfidf_info.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'nguoi_phu_trach': 'TV1',
        'threshold': float(th_tfidf),
        'chon_nguong_tren': 'mac_dinh',
        'sieu_tham_so': {
            'C': 0.1,
            'class_weight': {'0': 1, '1': 15},
            'features': 'FeatureUnion(Word TF-IDF + Char TF-IDF + Lexicon)'
        },
        'random_seed': 42,
        'ghi_chu': 'Mô hình tối ưu FeatureUnion 3 nhánh (Word+Char+Lexicon) + GridSearchCV C=0.1, weight={0:1,1:15}'
    }, f, ensure_ascii=False, indent=2)
print(" -> Đã xuất: results/predictions/svm_tfidf.csv & svm_tfidf_info.json")

# 3. svm_tfidf_single (TF-IDF đơn Word Level Baseline + Linear SVM)
scores_tfidf_single = np.round(tfidf_single_pipeline.decision_function(X_test), 4)
th_single = 0.0
pd.DataFrame({
    'review_id': df_test['review_id'],
    'label': y_test,
    'score': scores_tfidf_single,
    'pred': (scores_tfidf_single >= th_single).astype(int)
}).to_csv(os.path.join(pred_dir, 'svm_tfidf_single.csv'), index=False, encoding='utf-8')

with open(os.path.join(pred_dir, 'svm_tfidf_single_info.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'nguoi_phu_trach': 'TV1',
        'threshold': float(th_single),
        'chon_nguong_tren': 'mac_dinh',
        'sieu_tham_so': {
            'ngram_range': [1, 2],
            'sublinear_tf': True,
            'max_features': 10000,
            'class_weight': 'balanced'
        },
        'random_seed': 42,
        'ghi_chu': 'Mô hình cơ sở TF-IDF đơn Word Level + Linear SVM (class_weight=balanced)'
    }, f, ensure_ascii=False, indent=2)
print(" -> Đã xuất: results/predictions/svm_tfidf_single.csv & svm_tfidf_single_info.json")

print("\n" + "=" * 88)
print("     HOÀN TẤT ĐỐI SÁNH ĐẦY ĐỦ CÁC BIẾN THỂ & ĐÃ NỘP FILE DỰ ĐOÁN CHO TV4!")
print("=" * 88)
