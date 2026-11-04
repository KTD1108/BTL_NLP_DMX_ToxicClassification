import os
from playwright.sync_api import sync_playwright

out_dir = "D:/btlvb1/report_assets"
os.makedirs(out_dir, exist_ok=True)

css_base = """
  * {
    box-sizing: border-box;
  }
  body {
    margin: 0;
    padding: 10px;
    background: #ffffff;
    font-family: 'Times New Roman', Times, serif;
    color: #0f172a;
  }
  table {
    border-collapse: collapse;
    width: 960px;
    background: #ffffff;
    font-size: 13.5px;
    line-height: 1.5;
    border: 1px solid #94a3b8;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  th {
    background-color: #1e293b;
    color: #ffffff;
    font-weight: bold;
    text-align: center;
    padding: 10px 12px;
    border: 1px solid #475569;
    font-size: 14px;
    vertical-align: middle;
  }
  td {
    padding: 9px 12px;
    border: 1px solid #cbd5e1;
    vertical-align: middle;
    word-wrap: break-word;
  }
  tr:nth-child(even) td {
    background-color: #f8fafc;
  }
  .center {
    text-align: center;
  }
  .bold {
    font-weight: bold;
  }
  .quote {
    font-style: italic;
    color: #334155;
    line-height: 1.45;
  }
  .total-row td {
    background-color: #f1f5f9 !important;
    font-weight: bold;
    border-top: 2px solid #64748b;
  }
  .highlight-row td {
    background-color: #ecfdf5 !important;
    font-weight: bold;
  }
"""

def wrap_html(table_html):
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{css_base}</style>
</head>
<body>
{table_html}
</body>
</html>"""

# -----------------------------------------------------------------------------
# TABLE 1: CATEGORY BREAKDOWN (15,183 REAL SAMPLES)
# -----------------------------------------------------------------------------
t1_html = """
<table>
  <thead>
    <tr>
      <th style="width: 22%; text-align: left;">Ngành hàng</th>
      <th style="width: 14%;">Số bình luận</th>
      <th style="width: 14%;">Tỷ lệ (%)</th>
      <th style="width: 50%; text-align: left;">Đặc thù phản hồi & khiếu nại thực tế</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="bold">Gia dụng</td>
      <td class="center">9.101</td>
      <td class="center">59.94%</td>
      <td>Nồi chiên, máy lọc nước, bếp từ; phản ánh lỗi linh kiện, khét, mùi nhựa, hỏng nguồn.</td>
    </tr>
    <tr>
      <td class="bold">Tủ lạnh</td>
      <td class="center">1.615</td>
      <td class="center">10.64%</td>
      <td>Phản ánh không đông đá, ngăn mát yếu, tiếng ồn động cơ inverter, lỗi gioăng cao su.</td>
    </tr>
    <tr>
      <td class="bold">Điện thoại</td>
      <td class="center">1.556</td>
      <td class="center">10.25%</td>
      <td>Phản ánh nóng máy khi dùng, pin tụt nhanh, thái độ nhân viên tư vấn, lỗi cảm ứng.</td>
    </tr>
    <tr>
      <td class="bold">Máy lạnh / Điều hòa</td>
      <td class="center">1.267</td>
      <td class="center">8.34%</td>
      <td>Khiếu nại thợ lắp đặt, rò rỉ khí gas, làm lạnh yếu, trễ hẹn khung giờ giao hàng lắp đặt.</td>
    </tr>
    <tr>
      <td class="bold">Máy giặt</td>
      <td class="center">1.088</td>
      <td class="center">7.17%</td>
      <td>Khiếu nại máy rung lắc mạnh, vắt kêu to, đọng nước lồng giặt, chậm trễ giải quyết đổi trả.</td>
    </tr>
    <tr>
      <td class="bold">Tivi</td>
      <td class="center">556</td>
      <td class="center">3.66%</td>
      <td>Màn hình sọc kẻ, nứt tấm nền trong, lỗi kết nối wifi, ứng dụng xem phim Youtube.</td>
    </tr>
    <tr class="total-row">
      <td>Tổng cộng</td>
      <td class="center">15.183</td>
      <td class="center">100.0%</td>
      <td>Dữ liệu người dùng thực tế 100% thu thập trực tiếp từ dienmayxanh.com</td>
    </tr>
  </tbody>
</table>
"""

# -----------------------------------------------------------------------------
# TABLE 2: DATA SPLIT ZERO DATA LEAKAGE (NATURAL RANDOM SAMPLING RATIOS)
# -----------------------------------------------------------------------------
t2_html = """
<table>
  <thead>
    <tr>
      <th style="width: 15%; text-align: left;">Tập dữ liệu</th>
      <th style="width: 13%;">Tổng số mẫu</th>
      <th style="width: 18%;">Non-toxic (Nhãn 0)</th>
      <th style="width: 15%;">Toxic (Nhãn 1)</th>
      <th style="width: 12%;">Tỷ lệ Toxic</th>
      <th style="width: 27%; text-align: left;">Vai trò trong pipeline học máy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="bold">Train (70%)</td>
      <td class="center">10.628</td>
      <td class="center">9.732 (91.57%)</td>
      <td class="center bold">896 (8.43%)</td>
      <td class="center">8.43%</td>
      <td>Học từ vựng, khớp vectorizer và tối ưu hóa mô hình</td>
    </tr>
    <tr>
      <td class="bold">Validation (15%)</td>
      <td class="center">2.277</td>
      <td class="center">2.103 (92.36%)</td>
      <td class="center bold">174 (7.64%)</td>
      <td class="center">7.64%</td>
      <td>Kiểm định siêu tham số, tối ưu hóa ngưỡng xác suất</td>
    </tr>
    <tr>
      <td class="bold">Test (15%)</td>
      <td class="center">2.278</td>
      <td class="center">2.076 (91.13%)</td>
      <td class="center bold">202 (8.87%)</td>
      <td class="center">8.87%</td>
      <td>Đo lường hiệu năng độc lập khách quan chuẩn Zero Leakage</td>
    </tr>
    <tr class="total-row">
      <td>Tổng cộng</td>
      <td class="center">15.183</td>
      <td class="center">13.911 (91.62%)</td>
      <td class="center">1.272 (8.38%)</td>
      <td class="center">8.38%</td>
      <td>Dữ liệu thực tế Điện Máy Xanh, gán nhãn chuẩn hóa Weak Supervision</td>
    </tr>
  </tbody>
</table>
"""

# -----------------------------------------------------------------------------
# TABLE 3: MODEL COMPARISON (ROBUST TEST RESULTS)
# -----------------------------------------------------------------------------
t3_html = """
<table>
  <thead>
    <tr>
      <th style="width: 5%;">STT</th>
      <th style="width: 16%; text-align: left;">Phương pháp trích chọn</th>
      <th style="width: 18%; text-align: left;">Mô hình phân loại</th>
      <th style="width: 9%;">Accuracy</th>
      <th style="width: 9%;">Precision</th>
      <th style="width: 9%;">Recall</th>
      <th style="width: 9%;">F1 (Toxic)</th>
      <th style="width: 9%;">Macro-F1</th>
      <th style="width: 9%;">ROC-AUC</th>
      <th style="width: 7%;">Thời gian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="center">1</td>
      <td>Bag of Words (BoW)</td>
      <td>Multinomial NB</td>
      <td class="center">95.13%</td>
      <td class="center">66.91%</td>
      <td class="center bold" style="color: #2563eb;">89.11%</td>
      <td class="center">76.43%</td>
      <td class="center">86.86%</td>
      <td class="center">97.10%</td>
      <td class="center">0.110 s</td>
    </tr>
    <tr>
      <td class="center">2</td>
      <td>Bag of Words (BoW)</td>
      <td>SVM (LinearSVC)</td>
      <td class="center">98.16%</td>
      <td class="center bold" style="color: #15803d;">95.45%</td>
      <td class="center">83.17%</td>
      <td class="center">88.89%</td>
      <td class="center">93.94%</td>
      <td class="center">99.69%</td>
      <td class="center">2.016 s</td>
    </tr>
    <tr>
      <td class="center">3</td>
      <td>TF-IDF (Word + Char)</td>
      <td>Multinomial NB</td>
      <td class="center">94.42%</td>
      <td class="center">63.44%</td>
      <td class="center bold" style="color: #2563eb;">87.62%</td>
      <td class="center">73.60%</td>
      <td class="center">85.24%</td>
      <td class="center">97.65%</td>
      <td class="center">0.527 s</td>
    </tr>
    <tr class="highlight-row">
      <td class="center">4</td>
      <td>TF-IDF (Word + Char)</td>
      <td>Linear SVM (Tối ưu)</td>
      <td class="center" style="color: #15803d;">98.51%</td>
      <td class="center" style="color: #15803d; font-weight: bold;">96.15%</td>
      <td class="center" style="color: #15803d;">86.63%</td>
      <td class="center" style="color: #15803d; font-weight: bold;">91.15%</td>
      <td class="center" style="color: #15803d; font-weight: bold;">95.17%</td>
      <td class="center" style="color: #15803d; font-weight: bold;">99.76%</td>
      <td class="center">0.260 s</td>
    </tr>
    <tr>
      <td class="center">5</td>
      <td>Contextual Subword</td>
      <td class="bold">PhoBERT Transformer</td>
      <td class="center">95.08%</td>
      <td class="center">66.42%</td>
      <td class="center bold" style="color: #2563eb;">90.10%</td>
      <td class="center">76.47%</td>
      <td class="center">86.86%</td>
      <td class="center">98.35%</td>
      <td class="center">168.1 s</td>
    </tr>
  </tbody>
</table>
"""

# -----------------------------------------------------------------------------
# TABLE 4: ERROR ANALYSIS (REAL SAMPLES FROM 179 CASES)
# -----------------------------------------------------------------------------
t4_html = """
<table>
  <thead>
    <tr>
      <th style="width: 17%; text-align: left;">Dạng lỗi phân loại</th>
      <th style="width: 36%; text-align: left;">Bình luận thực tế từ Điện Máy Xanh</th>
      <th style="width: 10%;">Nhãn thật</th>
      <th style="width: 11%;">Mô hình sai</th>
      <th style="width: 11%;">Mô hình đúng</th>
      <th style="width: 15%; text-align: left;">Nguyên nhân bản chất</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="bold">Dương tính giả (FP - Khiếu nại dài)</td>
      <td class="quote">"Thợ dmx đi gắn khoan lỗ xong không bít lại, cũng khôg chịu hút chân không cho khách hàng"</td>
      <td class="center bold">0 (Non-toxic)</td>
      <td class="center" style="color: #b91c1c; font-weight: bold;">NB (BoW), NB (TF-IDF)</td>
      <td class="center" style="color: #15803d; font-weight: bold;">Linear SVM, PhoBERT</td>
      <td>Khách phàn nàn thợ lắp đặt kỹ thuật, Naive Bayes bắt từ tiêu cực 'không chịu' báo nhầm; SVM và PhoBERT phân loại chuẩn 0</td>
    </tr>
    <tr>
      <td class="bold">Dương tính giả (FP - Che giấu từ tục)</td>
      <td class="quote">"Làm ăn như *** Đặt hàng xong báo ko giao đc, còn tự ý đổi điểm nhận hàng."</td>
      <td class="center bold">0 (Non-toxic)</td>
      <td class="center" style="color: #b91c1c; font-weight: bold;">NB (BoW), NB (TF-IDF)</td>
      <td class="center" style="color: #15803d; font-weight: bold;">Linear SVM, PhoBERT</td>
      <td>Người dùng tự kiểm duyệt bằng dấu sao '***', Naive Bayes nhạy cảm báo nhầm 1; Linear SVM nhận diện chính xác nhãn 0</td>
    </tr>
    <tr>
      <td class="bold">Âm tính giả (FN - Bỏ sót câu ngắn)</td>
      <td class="quote">"Mua xài chưa được 1 tháng đã có dấu hiệu bong tróc... Thất vọng về sản phẩm. Sản phẩm đồ phế phẩm."</td>
      <td class="center bold">1 (Toxic)</td>
      <td class="center" style="color: #b91c1c; font-weight: bold;">NB (BoW), NB (TF-IDF)</td>
      <td class="center" style="color: #15803d; font-weight: bold;">Linear SVM, PhoBERT</td>
      <td>Cụm từ lăng mạ sản phẩm 'đồ phế phẩm' ở cuối câu ngắn, Naive Bayes chưa đủ ngưỡng xác suất, SVM và PhoBERT bắt chính xác</td>
    </tr>
  </tbody>
</table>
"""

tasks = [
    ("table_categories.png", t1_html),
    ("table_datasplit.png", t2_html),
    ("table_models.png", t3_html),
    ("table_error_analysis.png", t4_html)
]

with sync_playwright() as p:
    browser = p.chromium.launch()
    for filename, html in tasks:
        page = browser.new_page(device_scale_factor=2)
        page.set_content(wrap_html(html))
        table = page.locator("table")
        path = os.path.join(out_dir, filename)
        table.screenshot(path=path)
        print(f"Generated: {path}")
    browser.close()

print("ALL 4 CLEAN TABLES RENDERED FLAWLESSLY WITH PLAYWRIGHT HTML+CSS!")
