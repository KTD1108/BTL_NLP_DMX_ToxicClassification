# -*- coding: utf-8 -*-
"""
Kịch bản tổng hợp và chuẩn hóa 3 tập dữ liệu Train - Val - Test cho bài toán
phân loại bình luận Toxic trên Điện Máy Xanh (PTIT NLP BTL1).

1. Đồng bộ nlp_utils.py, train.csv (6.048 mẫu), val.csv (879 mẫu).
2. Gộp nhãn thủ công từ 4 thành viên (TV1, TV2, TV3, TV4) theo Chuẩn học thuật (Phương án 1).
3. Tiền xử lý tập Test 2.000 mẫu bằng nlp_utils.preprocess().
4. Xuất data/test.csv đồng nhất 100% định dạng với train.csv và val.csv.
5. Cập nhật data/tap_test_2000_can_gan_tay.xlsx & .csv có đầy đủ nhan_TV1..nhan_TV4 và nhan_chot.
6. Xuất results/human_vs_ai_agreement.csv phục vụ mục Error Analysis.
"""

import sys
import subprocess
import io
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report

sys.stdout.reconfigure(encoding='utf-8')

def get_bytes_from_git(ref, path):
    cmd = ['git', 'show', f'{ref}:{path}']
    res = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    return res.stdout

print("=== BƯỚC 1: ĐỒNG BỘ nlp_utils.py, train.csv, val.csv ===")
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. nlp_utils.py
nlp_bytes = get_bytes_from_git('origin/feature/data-qa', 'nlp_utils.py')
with open('nlp_utils.py', 'wb') as f:
    f.write(nlp_bytes)
print(" -> Đã lưu nlp_utils.py")

import nlp_utils

# 2. train.csv
train_bytes = get_bytes_from_git('origin/feature/data-qa', 'data/train.csv')
with open('data/train.csv', 'wb') as f:
    f.write(train_bytes)
df_train = pd.read_csv('data/train.csv')
print(f" -> Đã lưu data/train.csv: {df_train.shape[0]} dòng, {df_train.shape[1]} cột. Toxic = {df_train['label'].sum()} ({df_train['label'].mean()*100:.2f}%)")

# 3. val.csv
val_bytes = get_bytes_from_git('origin/feature/data-qa', 'data/val.csv')
with open('data/val.csv', 'wb') as f:
    f.write(val_bytes)
df_val = pd.read_csv('data/val.csv')
print(f" -> Đã lưu data/val.csv: {df_val.shape[0]} dòng, {df_val.shape[1]} cột. Toxic = {df_val['label'].sum()} ({df_val['label'].mean()*100:.2f}%)")


print("\n=== BƯỚC 2: GỘP NHÃN THỦ CÔNG 4 THÀNH VIÊN VÀ CHUẨN HÓA PHƯƠNG ÁN 1 ===")
# Đọc file nhãn từ các nhánh remote
qa_excel_bytes = get_bytes_from_git('origin/feature/data-qa', 'data/tap_test_2000_can_gan_tay.xlsx')
df_qa = pd.read_excel(io.BytesIO(qa_excel_bytes))

duong_excel_bytes = get_bytes_from_git('origin/feature/data-labeling-duong', 'data/tap_test_2000_can_gan_tay.xlsx')
df_duong = pd.read_excel(io.BytesIO(duong_excel_bytes))

# Đọc file 10k raw để lấy chính xác product_id
df_10k = pd.read_csv('data/dmx_reviews_10k_raw.csv')
prod_map = dict(zip(df_10k['review_id'], df_10k['product_id']))

# Danh sách STT lỗi kỹ thuật/chê thông thường cần revert về 0 theo Phương án 1
revert_tv3 = [1211, 1263, 1301, 1308, 1335, 1363, 1452, 1475, 1582]
revert_tv4 = [1770, 1803, 1805, 1807, 1840, 1866, 1925, 1985]

# Xây dựng bảng gán nhãn 2000 mẫu đầy đủ
test_rows = []
for i in range(2000):
    stt = i + 1
    rid = int(df_qa.loc[i, 'review_id'])
    pid = prod_map.get(rid, 0)
    sp = df_qa.loc[i, 'ten_san_pham']
    nganh = df_qa.loc[i, 'nganh']
    sao = int(df_qa.loc[i, 'so_sao'])
    nd = str(df_qa.loc[i, 'noi_dung'])
    vung = df_qa.loc[i, 'phan_vung_nhiem_vu']
    
    # Lấy nhãn từng thành viên
    v1 = df_qa.loc[i, 'nhan_TV1']
    v2 = df_qa.loc[i, 'nhan_TV2']
    v3 = df_duong.loc[i, 'nhan_TV3']
    v4 = df_duong.loc[i, 'nhan_TV4']
    
    # Xác định nhan_chot theo chuẩn học thuật
    if stt <= 400:
        # Cross-annotation TV1 & TV2
        if v1 == v2:
            chot = int(v1)
        else:
            # 10 câu bất đồng: 63, 219, 345 giữ 1 (thái độ gay gắt, tố gian lận); còn lại 0
            chot = 1 if stt in [63, 219, 345] else 0
    elif stt <= 800:
        chot = int(v1)
    elif stt <= 1200:
        chot = int(v2)
    elif stt <= 1600:
        val = int(v3)
        if stt in revert_tv3:
            val = 0
        chot = val
    else:
        val = int(v4)
        if stt in revert_tv4:
            val = 0
        chot = val

    test_rows.append({
        'stt': stt,
        'review_id': rid,
        'product_id': pid,
        'ten_san_pham': sp,
        'nganh': nganh,
        'so_sao': sao,
        'noi_dung': nd,
        'phan_vung_nhiem_vu': vung,
        'nhan_TV1': int(v1) if pd.notna(v1) else np.nan,
        'nhan_TV2': int(v2) if pd.notna(v2) else np.nan,
        'nhan_TV3': int(v3) if pd.notna(v3) else np.nan,
        'nhan_TV4': int(v4) if pd.notna(v4) else np.nan,
        'nhan_chot': chot
    })

df_test_full = pd.DataFrame(test_rows)

# Lưu lại tap_test_2000_can_gan_tay.xlsx và .csv
df_test_full.to_excel('data/tap_test_2000_can_gan_tay.xlsx', index=False)
df_test_full.to_csv('data/tap_test_2000_can_gan_tay.csv', index=False, encoding='utf-8')
print(" -> Đã cập nhật data/tap_test_2000_can_gan_tay.xlsx & .csv với đầy đủ nhãn 4 thành viên và nhan_chot!")


print("\n=== BƯỚC 3: TIỀN XỬ LÝ VĂN BẢN VÀ XUẤT data/test.csv ===")
print(" -> Đang chạy nlp_utils.preprocess trên 2.000 đánh giá tập Test...")
preprocessed_texts = [nlp_utils.preprocess(text) for text in df_test_full['noi_dung']]

df_test_clean = pd.DataFrame({
    'review_id': df_test_full['review_id'],
    'product_id': df_test_full['product_id'],
    'nganh_hang': df_test_full['nganh'],
    'so_sao': df_test_full['so_sao'],
    'noi_dung_goc': df_test_full['noi_dung'],
    'noi_dung_tien_xu_ly': preprocessed_texts,
    'nguon_nhan': 'human_consensus',
    'label': df_test_full['nhan_chot']
})

df_test_clean.to_csv('data/test.csv', index=False, encoding='utf-8')
print(f" -> Đã xuất data/test.csv: {df_test_clean.shape[0]} dòng, {df_test_clean.shape[1]} cột. Toxic = {df_test_clean['label'].sum()} ({df_test_clean['label'].mean()*100:.2f}%)")


print("\n=== BƯỚC 4: TÍNH ĐIỂM ĐỐI SÁNH CON NGƯỜI VS AI TRÊN 2.000 MẪU TEST ===")
df_ai = pd.read_csv('results/ai_test_set_toxic_analysis.csv')
y_human = df_test_clean['label']
y_ai = df_ai['ai_pred_toxic_50']
prob_ai = df_ai['ai_prob_toxic']

acc = accuracy_score(y_human, y_ai)
kappa = cohen_kappa_score(y_human, y_ai)
cm = confusion_matrix(y_human, y_ai)

print(f" - Độ đồng thuận (Accuracy): {acc*100:.2f}% (trùng {int(acc*2000)}/2000 câu)")
print(f" - Điểm Cohen's Kappa (Human vs AI): {kappa:.4f}")
print(" - Confusion Matrix (Hàng: Con người, Cột: AI):\n", cm)

agreement_types = []
for h, a in zip(y_human, y_ai):
    if h == a:
        agreement_types.append('Agree_Toxic' if h == 1 else 'Agree_NonToxic')
    else:
        agreement_types.append('Human0_AI1' if h == 0 else 'Human1_AI0')

df_agree = pd.DataFrame({
    'stt': df_test_full['stt'],
    'review_id': df_test_clean['review_id'],
    'so_sao': df_test_clean['so_sao'],
    'noi_dung': df_test_clean['noi_dung_goc'],
    'human_label': y_human,
    'ai_label': y_ai,
    'ai_prob_toxic': prob_ai,
    'is_agreement': (y_human == y_ai).astype(int),
    'agreement_type': agreement_types
})
df_agree.to_csv('results/human_vs_ai_agreement.csv', index=False, encoding='utf-8')
print(" -> Đã lưu results/human_vs_ai_agreement.csv phục vụ báo cáo!")


print("\n=== BƯỚC 5: KIỂM ĐỊNH TỔNG QUAN BỘ 3 TẬP DỮ LIỆU HUẤN LUYỆN ===")
summary_data = [
    {
        'Tập dữ liệu': 'Train (train.csv)',
        'Số mẫu': len(df_train),
        'Non-toxic (0)': (df_train['label'] == 0).sum(),
        'Toxic (1)': (df_train['label'] == 1).sum(),
        'Tỷ lệ Toxic': f"{df_train['label'].mean()*100:.2f}%"
    },
    {
        'Tập dữ liệu': 'Validation (val.csv)',
        'Số mẫu': len(df_val),
        'Non-toxic (0)': (df_val['label'] == 0).sum(),
        'Toxic (1)': (df_val['label'] == 1).sum(),
        'Tỷ lệ Toxic': f"{df_val['label'].mean()*100:.2f}%"
    },
    {
        'Tập dữ liệu': 'Test (test.csv)',
        'Số mẫu': len(df_test_clean),
        'Non-toxic (0)': (df_test_clean['label'] == 0).sum(),
        'Toxic (1)': (df_test_clean['label'] == 1).sum(),
        'Tỷ lệ Toxic': f"{df_test_clean['label'].mean()*100:.2f}%"
    }
]
df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

print("\nHOÀN THÀNH TOÀN BỘ YÊU CẦU! SẴN SÀNG HUẤN LUYỆN MÔ HÌNH!")
