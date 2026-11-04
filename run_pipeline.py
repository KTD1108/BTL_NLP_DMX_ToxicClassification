"""
Script chạy toàn bộ Pipeline Bài tập lớn 1 - Chủ đề 2c:
Hệ thống phân loại bình luận Toxic và Non-toxic trên dữ liệu review sản phẩm Điện Máy Xanh.
"""

import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.crawler import collect_or_build_dataset
from src.preprocess import VietnameseTextPreprocessor
from src.train_ml import train_and_evaluate_traditional_models
from src.train_phobert import train_modern_transformer_model
from src.evaluate import (
    evaluate_all_models,
    plot_eda_figures,
    plot_evaluation_figures,
    export_error_analysis,
)


def main(n_samples: int = 15000, try_live_crawl: bool = True, use_hf_hub: bool = False):
    print("=" * 80)
    print("BÀI TẬP LỚN 1 - CHỦ ĐỀ 2C: PHÂN LOẠI TOXIC & NON-TOXIC TRÊN ĐIỆN MÁY XANH")
    print("=" * 80)

    # Bước 1: Thu thập dữ liệu
    print("\n[Bước 1] Đang thu thập và khởi tạo bộ dữ liệu bình luận Điện Máy Xanh...")
    raw_csv = "data/raw/dmx_reviews_raw.csv"
    df_raw = collect_or_build_dataset(
        output_csv_path=raw_csv,
        n_samples=n_samples,
        try_live_crawl=try_live_crawl,
    )
    print(f" -> Tổng số mẫu thu thập: {len(df_raw)} mẫu (Non-toxic: {(df_raw['label']==0).sum()}, Toxic: {(df_raw['label']==1).sum()})")

    # Bước 2: Tiền xử lý dữ liệu tiếng Việt
    print("\n[Bước 2] Đang tiền xử lý văn bản tiếng Việt (Chuẩn hóa, Teencode, Tách từ PyVi, Stopwords)...")
    proc_csv = "data/processed/dmx_reviews_processed.csv"
    train_csv = "data/processed/train.csv"
    val_csv = "data/processed/val.csv"
    test_csv = "data/processed/test.csv"

    if os.path.exists(proc_csv) and os.path.exists(train_csv) and os.path.exists(test_csv):
        df_proc = pd.read_csv(proc_csv)
        if len(df_proc) == len(df_raw):
            print(f" -> Đã nạp dữ liệu tiền xử lý từ cache: {len(df_proc):,} mẫu.")
            df_train = pd.read_csv(train_csv)
            df_val = pd.read_csv(val_csv)
            df_test = pd.read_csv(test_csv)
        else:
            preprocessor = VietnameseTextPreprocessor(remove_stopwords=True, lowercase=True)
            df_proc = preprocessor.preprocess_dataframe(df_raw, text_col="raw_text")
            os.makedirs("data/processed", exist_ok=True)
            df_proc.to_csv(proc_csv, index=False, encoding="utf-8-sig")
            df_train, df_temp = train_test_split(df_proc, test_size=0.30, stratify=df_proc["label"], random_state=42)
            df_val, df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp["label"], random_state=42)
            df_train.to_csv(train_csv, index=False, encoding="utf-8-sig")
            df_val.to_csv(val_csv, index=False, encoding="utf-8-sig")
            df_test.to_csv(test_csv, index=False, encoding="utf-8-sig")
    else:
        preprocessor = VietnameseTextPreprocessor(remove_stopwords=True, lowercase=True)
        df_proc = preprocessor.preprocess_dataframe(df_raw, text_col="raw_text")
        os.makedirs("data/processed", exist_ok=True)
        df_proc.to_csv(proc_csv, index=False, encoding="utf-8-sig")
        df_train, df_temp = train_test_split(df_proc, test_size=0.30, stratify=df_proc["label"], random_state=42)
        df_val, df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp["label"], random_state=42)
        df_train.to_csv(train_csv, index=False, encoding="utf-8-sig")
        df_val.to_csv(val_csv, index=False, encoding="utf-8-sig")
        df_test.to_csv(test_csv, index=False, encoding="utf-8-sig")

    print(f" -> Kích thước tập dữ liệu: Train = {len(df_train):,}, Val = {len(df_val):,}, Test = {len(df_test):,}")

    for d in [df_train, df_val, df_test]:
        d["clean_text"] = d["clean_text"].fillna("").astype(str)
        d["phobert_text"] = d["phobert_text"].fillna("").astype(str)

    # Vẽ biểu đồ thống kê dữ liệu (EDA)
    plot_eda_figures(df_proc, fig_dir="outputs/figures")

    # Bước 3 & 4a: Trích chọn đặc trưng + Huấn luyện Naive Bayes & SVM
    ml_outputs, _ = train_and_evaluate_traditional_models(
        train_texts=df_train["clean_text"].tolist(),
        y_train=df_train["label"].values,
        val_texts=df_val["clean_text"].tolist(),
        y_val=df_val["label"].values,
        test_texts=df_test["clean_text"].tolist(),
        y_test=df_test["label"].values,
        save_dir="outputs/models",
    )

    # Bước 4b (Bổ sung ý 4): Huấn luyện mô hình hiện đại PhoBERT / Contextual Transformer
    dl_output = train_modern_transformer_model(
        train_texts=df_train["phobert_text"].tolist(),
        y_train=df_train["label"].values,
        val_texts=df_val["phobert_text"].tolist(),
        y_val=df_val["label"].values,
        test_texts=df_test["phobert_text"].tolist(),
        y_test=df_test["label"].values,
        use_hf_hub=use_hf_hub,
        save_dir="outputs/models",
    )

    all_experiments = ml_outputs + [dl_output]

    # Bước 5: Đánh giá mô hình & Vẽ biểu đồ so sánh
    print("\n[Bước 5] Đang tính toán các chỉ số đánh giá và xuất biểu đồ...")
    df_results = evaluate_all_models(
        y_test=df_test["label"].values,
        experiment_outputs=all_experiments,
        output_dir="outputs",
    )
    plot_evaluation_figures(
        y_test=df_test["label"].values,
        experiment_outputs=all_experiments,
        df_results=df_results,
        fig_dir="outputs/figures",
    )
    df_err = export_error_analysis(
        df_test=df_test,
        experiment_outputs=all_experiments,
        output_dir="outputs",
    )

    print("\n" + "=" * 80)
    print("BẢNG KẾT QUẢ THỰC NGHIỆM TRÊN TẬP KIỂM TRA (TEST SET)")
    print("=" * 80)
    cols_show = [
        "STT",
        "Phương pháp trích chọn đặc trưng",
        "Mô hình chi tiết",
        "Accuracy",
        "Precision (Toxic %)",
        "Recall (Toxic %)",
        "F1-Score (Toxic %)",
        "Macro-F1 (%)",
    ]
    print(df_results[cols_show].to_string(index=False))
    print(f"\n-> Đã lưu bảng tổng hợp tại: outputs/results_summary.csv")
    print(f"-> Đã lưu phân tích lỗi ({len(df_err)} câu khó) tại: outputs/error_analysis.csv")
    print(f"-> Đã lưu 5 biểu đồ chất lượng cao tại: outputs/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15000, help="Số lượng mẫu bình luận")
    parser.add_argument("--no-live-crawl", action="store_true", help="Tắt thử nghiệm crawl trực tiếp")
    parser.add_argument("--use-hf-hub", action="store_true", help="Fine-tune trực tiếp vinai/phobert-base-v2 từ HuggingFace")
    args = parser.parse_args()
    main(
        n_samples=args.samples,
        try_live_crawl=not args.no_live_crawl,
        use_hf_hub=args.use_hf_hub,
    )
