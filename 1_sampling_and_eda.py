import os
import sys
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_merge_raw_data(zip_path: str = "raw.zip") -> pd.DataFrame:
    # Trích xuất thông tin sản phẩm từ các file danh mục trong raw.zip
    product_map = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        for fname in z.namelist():
            if fname.startswith('raw/dmx_products_') and fname.endswith('.csv'):
                with z.open(fname) as f:
                    df_p = pd.read_csv(f)
                    if 'product_id' in df_p.columns and 'ten_san_pham' in df_p.columns:
                        for _, row in df_p.iterrows():
                            pid = row.get('product_id')
                            pname = row.get('ten_san_pham')
                            if pd.notna(pid) and pd.notna(pname):
                                product_map[int(pid)] = str(pname).strip()

    # Đọc dữ liệu đánh giá thô
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('raw/dmx_reviews.csv') as f:
            df_raw = pd.read_csv(f)

    df_raw['ten_san_pham'] = df_raw['product_id'].map(product_map).fillna("Sản phẩm Điện Máy Xanh")

    # Lọc bỏ đánh giá rỗng và đánh giá dưới 2 từ
    df_valid = df_raw.dropna(subset=['noi_dung']).copy()
    df_valid['noi_dung'] = df_valid['noi_dung'].astype(str).str.strip()
    df_valid['so_tu'] = df_valid['noi_dung'].apply(lambda x: len(x.split()))
    df_valid = df_valid[df_valid['so_tu'] >= 2].copy().reset_index(drop=True)

    print(f"Dữ liệu gốc: {len(df_raw):,} dòng | Sau khi lọc: {len(df_valid):,} dòng hợp lệ.")
    return df_valid


def stratified_sample_10k(df_valid: pd.DataFrame) -> pd.DataFrame:
    # Lấy mẫu phân tầng 10.000 dòng theo số sao
    df_1 = df_valid[df_valid['so_sao'] == 1].sample(n=1500, random_state=42)
    df_2 = df_valid[df_valid['so_sao'] == 2].sample(n=1000, random_state=42)
    df_3 = df_valid[df_valid['so_sao'] == 3].sample(n=1500, random_state=42)
    df_4 = df_valid[df_valid['so_sao'] == 4].sample(n=2500, random_state=42)
    df_5 = df_valid[df_valid['so_sao'] == 5].sample(n=3500, random_state=42)

    df_10k = pd.concat([df_1, df_2, df_3, df_4, df_5]).sample(frac=1, random_state=42).reset_index(drop=True)

    raw_10k_path = os.path.join(DATA_DIR, "dmx_reviews_10k_raw.csv")
    df_10k.to_csv(raw_10k_path, index=False, encoding='utf-8-sig')
    print(f"Đã lưu tập 10.000 mẫu: {raw_10k_path}")
    return df_10k


def plot_eda_charts(df_10k: pd.DataFrame):
    # 1. Biểu đồ phân bổ số sao
    plt.figure(figsize=(8, 5))
    star_counts = df_10k['so_sao'].value_counts().sort_index()
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71']
    bars = plt.bar(star_counts.index.astype(str), star_counts.values, color=colors, edgecolor='black', alpha=0.85)
    plt.title('Phan bo danh gia theo so sao (10.000 mau)', fontsize=13, fontweight='bold')
    plt.xlabel('So sao')
    plt.ylabel('So luong')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 30, f'{int(h):,}', ha='center', va='bottom', fontsize=10)
    plt.ylim(0, max(star_counts.values) + 300)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "1_rating_distribution.png"), dpi=300)
    plt.close()

    # 2. Biểu đồ top ngành hàng
    plt.figure(figsize=(10, 6))
    top_cates = df_10k['nganh'].value_counts().head(12)
    plt.barh(top_cates.index[::-1], top_cates.values[::-1], color='#2980b9', edgecolor='black', alpha=0.85)
    plt.title('Top 12 nganh hang pho bien nhat', fontsize=13, fontweight='bold')
    plt.xlabel('So luong')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "2_category_distribution.png"), dpi=300)
    plt.close()

    # 3. Biểu đồ phân phối độ dài bình luận
    plt.figure(figsize=(9, 5))
    lengths = df_10k['so_tu']
    sns.histplot(lengths[lengths <= 80], bins=30, kde=True, color='#8e44ad', edgecolor='black')
    plt.title('Phan phoi do dai binh luan (so tu)', fontsize=13, fontweight='bold')
    plt.xlabel('So tu')
    plt.ylabel('So luong')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "3_review_length_distribution.png"), dpi=300)
    plt.close()


def split_and_prepare_test_set(df_10k: pd.DataFrame):
    # Tách tập Test (20% = 2.000 mẫu)
    df_train_val, df_test = train_test_split(
        df_10k, test_size=0.20, random_state=42, stratify=df_10k['so_sao']
    )

    # Tách tập Train (70% = 7.000 mẫu) và tập Val (10% = 1.000 mẫu)
    df_train, df_val = train_test_split(
        df_train_val, test_size=0.125, random_state=42, stratify=df_train_val['so_sao']
    )

    train_path = os.path.join(DATA_DIR, "train_7000_raw.csv")
    val_path = os.path.join(DATA_DIR, "val_1000_raw.csv")
    df_train.to_csv(train_path, index=False, encoding='utf-8-sig')
    df_val.to_csv(val_path, index=False, encoding='utf-8-sig')
    print(f"Đã lưu tập Train: {train_path} ({len(df_train):,} mẫu)")
    print(f"Đã lưu tập Val:   {val_path} ({len(df_val):,} mẫu)")

    # Định dạng tập Test 2.000 mẫu để gán nhãn thủ công
    df_test = df_test.reset_index(drop=True)
    df_test['stt'] = range(1, len(df_test) + 1)

    def assign_section(idx):
        if idx <= 400:
            return "Gan_cheo_TV1_va_TV2 (400 mau do Cohen Kappa)"
        elif idx <= 800:
            return "TV1_gan_rieng (400 mau)"
        elif idx <= 1200:
            return "TV2_gan_rieng (400 mau)"
        elif idx <= 1600:
            return "TV3_gan_rieng (400 mau)"
        else:
            return "TV4_gan_rieng (400 mau)"

    df_test['phan_vung_nhiem_vu'] = [assign_section(i) for i in df_test['stt']]
    df_test['nhan_TV1'] = ""
    df_test['nhan_TV2'] = ""
    df_test['nhan_TV3'] = ""
    df_test['nhan_TV4'] = ""

    export_cols = [
        'stt', 'review_id', 'ten_san_pham', 'nganh', 'so_sao',
        'noi_dung', 'phan_vung_nhiem_vu',
        'nhan_TV1', 'nhan_TV2', 'nhan_TV3', 'nhan_TV4'
    ]
    df_test_export = df_test[export_cols]

    excel_path = os.path.join(DATA_DIR, "tap_test_2000_can_gan_tay.xlsx")
    csv_path = os.path.join(DATA_DIR, "tap_test_2000_can_gan_tay.csv")
    df_test_export.to_excel(excel_path, index=False)
    df_test_export.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"Đã lưu tập Test:  {excel_path} ({len(df_test):,} mẫu)")
    return df_train, df_val, df_test


def main():
    df_valid = load_and_merge_raw_data("raw.zip")
    df_10k = stratified_sample_10k(df_valid)
    plot_eda_charts(df_10k)
    split_and_prepare_test_set(df_10k)
    print("Hoàn tất xử lý dữ liệu và tạo tập kiểm tra.")


if __name__ == "__main__":
    main()
