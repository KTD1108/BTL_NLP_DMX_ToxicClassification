"""Ghép nhãn, tiền xử lý và xuất bộ dữ liệu cuối cùng.

    python -X utf8 2b_build_dataset.py

Đầu vào: data/train_7000_raw.csv, data/val_1000_raw.csv, data/tap_test_2000_can_gan_tay.xlsx,
         llm_labeling/llm_labels.csv, llm_labeling/spotcheck_positive.xlsx,
         data/test_adjudication.csv (tùy chọn: review_id,label cho các câu gán chéo bị lệch)
Đầu ra:  data/train.csv, data/val.csv, data/test.csv (test chỉ xuất khi đã đủ nhãn người)

Câu train/val trùng nội dung với test (hoặc train trùng val) bị loại để tránh rò rỉ dữ liệu.
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

from nlp_utils import preprocess

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LLM = ROOT / "llm_labeling"
TEST_XLSX = DATA / "tap_test_2000_can_gan_tay.xlsx"
ADJUDICATION = DATA / "test_adjudication.csv"
COLUMNS = ["review_id", "product_id", "nganh_hang", "so_sao", "noi_dung_goc",
           "noi_dung_tien_xu_ly", "nguon_nhan", "label"]

# Người gán của từng vùng stt trong tập test; stt 1-400 do TV1 và TV2 gán chéo.
CROSS = range(1, 401)
OWNER = [(range(401, 801), "nhan_TV1"), (range(801, 1201), "nhan_TV2"),
         (range(1201, 1601), "nhan_TV3"), (range(1601, 2001), "nhan_TV4")]


def category_map():
    """product_id -> nganh_hang, lấy từ danh mục sản phẩm trong raw.zip."""
    frames = []
    with zipfile.ZipFile(ROOT / "raw.zip") as z:
        for name in z.namelist():
            if name.startswith("raw/dmx_products_") and name.endswith(".csv"):
                with z.open(name) as f:
                    frames.append(pd.read_csv(f, usecols=["product_id", "nganh_hang"]))
    return pd.concat(frames).drop_duplicates("product_id").set_index("product_id")["nganh_hang"]


def llm_labels():
    """Nhãn LLM cho train/val, đã áp dụng các sửa đổi trong file rà soát."""
    labels = pd.read_csv(LLM / "llm_labels.csv")
    labels = labels[labels["split"] != "test"].set_index("review_id")["llm_label"]
    check = pd.read_excel(LLM / "spotcheck_positive.xlsx").dropna(subset=["nhan_sua"])
    labels.loc[check["review_id"]] = check["nhan_sua"].astype(int).values
    print(f"[-] Nhãn LLM: {len(labels):,} câu, sửa {int((check['nhan_sua'] == 0).sum())} câu sau rà soát")
    return labels


def test_labels(test):
    """Nhãn người cho tập test; trả về None kèm lý do nếu chưa đủ."""
    label = pd.Series(pd.NA, index=test.index, dtype="Int64")
    missing = []
    for rng, col in OWNER:
        part = test["stt"].isin(rng)
        label[part] = test.loc[part, col].astype("Int64")
        if test.loc[part, col].isna().any():
            missing.append(f"{col} còn {int(test.loc[part, col].isna().sum())} câu trống")

    cross = test["stt"].isin(CROSS)
    agree = cross & (test["nhan_TV1"] == test["nhan_TV2"])
    label[agree] = test.loc[agree, "nhan_TV1"].astype("Int64")
    disagree = test[cross & test["nhan_TV1"].notna() & test["nhan_TV2"].notna() & ~agree]
    if ADJUDICATION.exists():
        adj = pd.read_csv(ADJUDICATION).set_index("review_id")["label"]
        hit = test["review_id"].isin(adj.index) & cross
        label[hit] = test.loc[hit, "review_id"].map(adj).astype("Int64")
    unresolved = disagree[~disagree["review_id"].isin(adj.index if ADJUDICATION.exists() else [])]
    if len(unresolved):
        missing.append(f"{len(unresolved)} câu gán chéo bị lệch chưa hòa giải "
                       f"(stt {unresolved['stt'].tolist()}) -> ghi vào {ADJUDICATION.relative_to(ROOT)}")
    if test.loc[cross, ["nhan_TV1", "nhan_TV2"]].isna().any().any():
        missing.append("vùng gán chéo stt 1-400 còn câu trống")
    return (None, missing) if missing else (label.astype(int), [])


def main():
    cats = category_map()
    raw10k = pd.read_csv(DATA / "dmx_reviews_10k_raw.csv", usecols=["review_id", "product_id"])
    train = pd.read_csv(DATA / "train_7000_raw.csv")
    val = pd.read_csv(DATA / "val_1000_raw.csv")
    test = pd.read_excel(TEST_XLSX).merge(raw10k, on="review_id", how="left")

    labels = llm_labels()
    for df in (train, val):
        df["label"] = df["review_id"].map(labels)
        df["nguon_nhan"] = "semi_auto"
        assert df["label"].notna().all(), "Thiếu nhãn LLM cho một số câu train/val"

    print("[-] Đang tiền xử lý...")
    for df in (train, val, test):
        df["noi_dung_goc"] = df["noi_dung"]
        df["noi_dung_tien_xu_ly"] = df["noi_dung"].map(preprocess)
        df["nganh_hang"] = df["product_id"].map(cats)

    # Loại rò rỉ: câu rỗng sau xử lý, val trùng test, train trùng test hoặc val.
    test_texts, val_texts = set(test["noi_dung_tien_xu_ly"]), set(val["noi_dung_tien_xu_ly"])
    n_val, n_train = len(val), len(train)
    val = val[(val["noi_dung_tien_xu_ly"] != "") & ~val["noi_dung_tien_xu_ly"].isin(test_texts)]
    val_texts = set(val["noi_dung_tien_xu_ly"])
    train = train[(train["noi_dung_tien_xu_ly"] != "")
                  & ~train["noi_dung_tien_xu_ly"].isin(test_texts | val_texts)]
    print(f"[-] Loại rò rỉ: train {n_train - len(train)} câu, val {n_val - len(val)} câu")

    for name, df in [("train", train), ("val", val)]:
        df["label"] = df["label"].astype(int)
        df[COLUMNS].to_csv(DATA / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"[-] data/{name}.csv: {len(df):,} câu, toxic = {df['label'].sum()} ({df['label'].mean():.2%})")

    label, missing = test_labels(test)
    if label is None:
        print("[!] Chưa xuất data/test.csv:\n    " + "\n    ".join(missing))
        return
    test["label"] = label
    test["nguon_nhan"] = "human_gold"
    test[COLUMNS].to_csv(DATA / "test.csv", index=False, encoding="utf-8-sig")
    print(f"[-] data/test.csv: {len(test):,} câu, toxic = {test['label'].sum()} ({test['label'].mean():.2%})")


if __name__ == "__main__":
    main()
