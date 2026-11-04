"""
Bước 4a: Xây dựng và huấn luyện các thuật toán Học máy truyền thống (Naive Bayes & SVM)
trên 2 phương pháp trích chọn đặc trưng tiêu chuẩn:
1. Bag of Words (BoW)
2. TF-IDF (Word + Char N-grams)
"""

import os
import time
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

from src.features import (
    build_bow_extractor,
    build_tfidf_extractor,
)


def train_and_evaluate_traditional_models(
    train_texts: List[str],
    y_train: np.ndarray,
    val_texts: List[str],
    y_val: np.ndarray,
    test_texts: List[str],
    y_test: np.ndarray,
    save_dir: str = "outputs/models",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Huấn luyện và tinh chỉnh siêu tham số cho Naive Bayes & SVM trên 2 nhóm đặc trưng BoW và TF-IDF.
    Trả về danh sách kết quả dự đoán trên tập Test và các đối tượng mô hình đã huấn luyện.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Trích chọn đặc trưng cho 2 phương pháp (BoW và TF-IDF)
    print("[Bước 3] Đang trích chọn đặc trưng (Bag of Words & TF-IDF Word+Char)...")
    bow_ext = build_bow_extractor(max_features=6000)
    X_train_bow = bow_ext.fit_transform(train_texts)
    X_test_bow = bow_ext.transform(test_texts)

    tfidf_ext = build_tfidf_extractor(max_word_features=6000, max_char_features=4000)
    X_train_tfidf = tfidf_ext.fit_transform(train_texts)
    X_test_tfidf = tfidf_ext.transform(test_texts)

    joblib.dump(bow_ext, os.path.join(save_dir, "bow_extractor.pkl"))
    joblib.dump(tfidf_ext, os.path.join(save_dir, "tfidf_extractor.pkl"))

    experiments = [
        # --- Phương pháp 1: Bag of Words (BoW) ---
        {
            "feature_name": "Bag of Words (BoW)",
            "algorithm_group": "Naive Bayes",
            "model_name": "Multinomial Naive Bayes",
            "estimator": GridSearchCV(
                MultinomialNB(),
                param_grid={"alpha": [0.1, 0.5, 1.0, 2.0]},
                scoring="f1",
                cv=5,
            ),
            "X_train": X_train_bow,
            "X_test": X_test_bow,
        },
        {
            "feature_name": "Bag of Words (BoW)",
            "algorithm_group": "SVM",
            "model_name": "SVM (LinearSVC)",
            "estimator": CalibratedClassifierCV(
                LinearSVC(C=0.5, class_weight="balanced", random_state=42, max_iter=3000),
                cv=3,
            ),
            "X_train": X_train_bow,
            "X_test": X_test_bow,
        },
        # --- Phương pháp 2: TF-IDF (Word + Char N-grams) ---
        {
            "feature_name": "TF-IDF (Word + Char N-gram)",
            "algorithm_group": "Naive Bayes",
            "model_name": "Multinomial Naive Bayes",
            "estimator": GridSearchCV(
                MultinomialNB(),
                param_grid={"alpha": [0.1, 0.3, 0.5, 1.0]},
                scoring="f1",
                cv=5,
            ),
            "X_train": X_train_tfidf,
            "X_test": X_test_tfidf,
        },
        {
            "feature_name": "TF-IDF (Word + Char N-gram)",
            "algorithm_group": "SVM",
            "model_name": "SVM (LinearSVC)",
            "estimator": CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=3000),
                cv=3,
            ),
            "X_train": X_train_tfidf,
            "X_test": X_test_tfidf,
        },
    ]

    experiment_outputs = []
    trained_artifacts = {
        "extractors": {
            "bow": bow_ext,
            "tfidf": tfidf_ext,
        },
        "models": {},
    }

    for exp in experiments:
        full_key = f"{exp['feature_name']} + {exp['model_name']}"
        print(f"[Bước 4a] Đang huấn luyện: {full_key} ...")
        t0 = time.time()
        est = exp["estimator"]
        X_tr = exp["X_train"]
        y_tr = y_train

        est.fit(X_tr, y_tr)
        train_time = time.time() - t0

        y_pred = est.predict(exp["X_test"])
        if hasattr(est, "predict_proba"):
            y_prob = est.predict_proba(exp["X_test"])[:, 1]
        else:
            y_prob = y_pred.astype(float)

        best_params = est.best_params_ if isinstance(est, GridSearchCV) else {}
        trained_artifacts["models"][full_key] = est

        experiment_outputs.append({
            "feature_name": exp["feature_name"],
            "algorithm_group": exp["algorithm_group"],
            "model_name": exp["model_name"],
            "full_name": full_key,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "train_time_sec": round(train_time, 3),
            "best_params": best_params,
        })

    # Lưu mô hình truyền thống tốt nhất (TF-IDF + SVM)
    joblib.dump(
        trained_artifacts["models"]["TF-IDF (Word + Char N-gram) + SVM (LinearSVC)"],
        os.path.join(save_dir, "best_traditional_svm_tfidf.pkl"),
    )
    return experiment_outputs, trained_artifacts
