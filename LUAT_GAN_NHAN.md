# Luật gán nhãn Toxic / Non-toxic

Áp dụng cho toàn bộ `data/train.csv`, `data/val.csv`, `data/test.csv` (cột `label`).
Nhãn dựa vào **cách dùng ngôn từ và thái độ** của người viết, không dựa vào việc khách hài lòng hay không.

## Nhãn 1 – Toxic

1. **Chửi bới, thô tục, xúc phạm** người khác (nhân viên, cửa hàng, hãng, khách khác), kể cả viết tắt, biến âm:
   "mất dạy", "vô học", "bố láo", "đồ chó", "như cc", "như loz", "như quần què", "đkm", "vcl", "ngu như bò",
   "tụi bây ăn gì tao cúng".
2. **Cáo buộc gian dối / lừa đảo**: "lừa đảo", "treo đầu dê bán thịt chó", "bẫy khách", "bọn ăn cướp", "làm ăn xảo trá".
3. **Miệt thị, hạ thấp, thành ngữ nặng nề**: "đồ rác rưởi", "hàng dõm", "cái của nợ", "như gà mò", "dân chợ búa",
   "đem con bỏ chợ", "tiền mất tật mang", "làm ăn như hạch", "thu ngân xem thường khách hàng".

### Luật bổ sung cho các trường hợp ranh giới (đều là 1)

- **A. Kêu gọi người khác né / tẩy chay cửa hàng hoặc thương hiệu**, dù giọng nhẹ nhàng:
  "mọi người né ra nha", "nên tránh xa điện máy xanh", "mọi người đừng mua hãng này", "Chừa Nanomax ra nhé mọi người".
- **B. Khẳng định bị lừa / gian dối**, dù chỉ kể sự việc, không chửi:
  "thợ ăn gian ống đồng", "mua mới mà giao hàng đã bóc seal", "nhân viên cố tình tư vấn sai",
  "cảm thấy bị lừa", "như kiểu bị lừa đảo" (người viết bày tỏ mình đã bị lừa).
- **C. Thái độ thù địch rõ rệt** với cửa hàng / hãng / nhân viên, dù không có từ thô tục:
  "nhân viên kiểu bất cần đời", "làm ăn tệ bạc quá chừng", "âm điểm vô cực", "cạch mặt ĐMX", "từ bỏ dmx",
  "có ngày bị tẩy chay 😡".

## Nhãn 0 – Non-toxic

1. Khen, hài lòng: "máy dùng êm, làm lạnh nhanh", "giao hàng đúng hẹn".
2. Hỏi đáp, thắc mắc: "máy này có đổi trả được không?".
3. **Chê, báo lỗi, phàn nàn bằng giọng kể sự việc**, kể cả có "tệ", "kém", "thất vọng":
   - "Mới mua 2 ngày đã lỗi không lên nguồn, đã gọi bảo hành"
   - "Gọi bảo hành 3 lần không thấy ai đến, rất thất vọng"
   - "Sản phẩm kém, không đáng tiền, sẽ không mua lại"
4. **Chỉ nghi ngờ hoặc đặt câu hỏi**, chưa khẳng định bị lừa: "hay là bị bịp rồi?", "đmx lừa đảo à?",
   "không biết có phải hàng trưng bày không?".
5. **Từ nhạy cảm nhưng nghĩa khác**: "tiếng bịp" (tiếng kêu), "ĐM Chợ Lớn" / "ĐMCL" (tên cửa hàng),
   "miếng CL" (cường lực), "kẻ lừa đảo mạo danh shipper" (kẻ gian bên ngoài, không cáo buộc cửa hàng).
6. Khuyên về **một sản phẩm** vì lỗi kỹ thuật, giọng bình thường: "loa lỗi, không nên mua dòng này".

## Mẹo khi phân vân

- Hỏi: "Câu này có **xúc phạm** ai, **khẳng định** bị lừa, **kêu gọi** người khác né, hay thể hiện **thù địch** rõ không?"
  Có → 1. Chỉ là không hài lòng → 0.
- Một từ "tệ", "kém", "thất vọng" đơn lẻ chưa đủ để gán 1.
- Đọc cả câu, không gán theo từ khóa đơn lẻ.

## Quy trình gán nhãn

### Train / Validation
1. Gemini gán theo luật này cho toàn bộ câu 1–3 sao và mọi câu đang là toxic.
2. Người rà soát từng câu có nhãn Gemini khác nhãn trước đó và chốt nhãn cuối.
3. Câu 4–5 sao mặc định là 0 (tỷ lệ toxic ở nhóm này gần 0%).

### Test (gán mù lại)
Nhãn test cũ được chốt khi người rà soát đã thấy nhãn Gemini nên không độc lập
(Kappa nhãn test cũ với Gemini lên tới 0,98). Vì vậy toàn bộ 1.992 câu test được gán mù lại:
1. Xáo ngẫu nhiên, lấy 200 câu chung cho cả 4 thành viên, 1.792 câu còn lại chia đều 4 phần
   (mỗi người 648 câu, không biết câu nào là câu chung).
2. File gán chỉ có số sao và nội dung, không có nhãn cũ hay nhãn Gemini; thứ tự xáo riêng từng người.
3. Câu riêng lấy nhãn của người gán; câu chung lấy theo đa số, 4 câu hòa 2–2 lấy theo nhãn test cũ.
4. Kết quả: 85 câu toxic (4,27%), đổi 45 nhãn so với test cũ (25 câu 0→1, 20 câu 1→0).

Script: `experiments/test_blind/make_test_blind.py` (`make` / `score` / `apply`); file gán của
từng người ở `data/test_blind/`.

Cột `nguon_nhan`:
- `human`: nhãn do thành viên nhóm gán tay hoặc đã rà soát lại (toàn bộ câu toxic và toàn bộ tập test).
- `llm`: nhãn do Gemini gán theo luật hoặc mặc định non-toxic (câu 4–5 sao), không có người rà riêng từng câu.

## Độ tin cậy nhãn

### Gán mù tập test: 4 thành viên trên 200 câu chung

| Chỉ số | Giá trị |
|---|---|
| Fleiss' Kappa (4 người) | **0,68** |
| Cohen's Kappa trung bình 6 cặp | 0,67 (từ 0,57 đến 0,77) |
| Đồng ý thuần trung bình các cặp | 96,7% |
| Cả 4 người cùng nhãn | 94,0% |

| Cặp | Kappa | Đồng ý thuần | Đồng ý trên câu toxic |
|---|---|---|---|
| TV1 – TV2 | 0,75 | 97,0% | 76,9% |
| TV1 – TV3 | 0,77 | 97,5% | 78,3% |
| TV1 – TV4 | 0,62 | 96,0% | 63,6% |
| TV2 – TV3 | 0,65 | 96,5% | 66,7% |
| TV2 – TV4 | 0,69 | 97,0% | 70,0% |
| TV3 – TV4 | 0,57 | 96,5% | 58,8% |

Ghi chú: toxic chỉ ~5% nên đồng ý thuần luôn cao; Kappa và "đồng ý trên câu toxic"
(2·số câu cùng gán 1 / tổng số câu mỗi người gán 1) phản ánh đúng hơn. Kappa tính trên nhãn
gán mù lần đầu của mỗi người (TV2 có chỉnh một số câu sau đó; nhãn đã chỉnh dùng làm nhãn cuối).

### Các lần đo trước

| So sánh | Số câu | Kappa |
|---|---|---|
| Hai người gán chéo | 400 | 0,49 |
| Nhãn chốt của nhóm với Gemini (test cũ) | 2.000 | 0,54 |
| Người rà soát với Gemini (gán mù) | 100 | 0,63 |
| Người rà soát với Gemini (test cũ, không mù) | 800 | 0,75 |
| Gemini lần 1 với Gemini lần 2 | 1.002 | 0,92 |
