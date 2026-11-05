"""Gán nhãn train/val bằng LLM (chạy qua chat hoặc agent, không dùng API).

    python -X utf8 2_llm_labeling.py prepare   # tạo prompt trong llm_labeling/prompts/
    python -X utf8 2_llm_labeling.py collect   # kiểm tra kết quả trong llm_labeling/responses/

Batch test_* chỉ dùng để đo độ chính xác của LLM so với nhãn người, không thay nhãn test.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LLM = ROOT / "llm_labeling"
PROMPTS = LLM / "prompts"
RESPONSES = LLM / "responses"
MANIFEST = LLM / "manifest.csv"
LLM_LABELS = LLM / "llm_labels.csv"
SPOTCHECK = LLM / "spotcheck_positive.xlsx"

SPLITS = {
    "train": DATA / "train_7000_raw.csv",
    "val": DATA / "val_1000_raw.csv",
    "test": DATA / "tap_test_2000_can_gan_tay.xlsx",
}
HUMAN_COLS = ["nhan_TV1", "nhan_TV2", "nhan_TV3", "nhan_TV4"]

GUIDELINES = """\
# QUY TẮC GÁN NHÃN TOXIC / NON-TOXIC (dùng chung cho người và Gemini)

Nhiệm vụ: đọc từng bình luận đánh giá sản phẩm trên Điện Máy Xanh và gán 1 nhãn.
Chỉ dựa vào CÁCH DÙNG NGÔN TỪ, không dựa vào việc khách hài lòng hay không.

## Nhãn 1 = TOXIC (độc hại)
1. Chửi bới, thô tục, xúc phạm người khác (nhân viên, cửa hàng, hãng, khách khác):
   "mất dạy", "vô học", "bố láo", "đồ chó", "như cc", "đkm", "vcl", "ngu như bò", "khinh người"...
2. Cáo buộc gian dối / lừa đảo, kêu gọi tẩy chay:
   "lừa đảo", "treo đầu dê bán thịt chó", "bẫy khách", "tiền mất tật mang", "cạch mặt ĐMX",
   "mọi người đừng bao giờ mua ở đây", "bọn ăn cướp"...
3. Bức xúc cực đoan, miệt thị, thù hằn, ngôn từ gay gắt:
   "đồ rác rưởi", "đồ phế thải", "phí tiền vứt đi", "tức điên người", "làm ăn như hạch"...

## Luật bổ sung đã chốt (các trường hợp ranh giới) -> đều là 1
A. KÊU GỌI NGƯỜI KHÁC NÉ / TẨY CHAY cửa hàng hoặc thương hiệu, dù giọng nhẹ nhàng:
   "mọi người né ra nha", "A e né ra", "tốt nhất nên né ra", "người tiêu dùng né điện máy xanh",
   "mọi ng né cái hãng này ra", "mọi người không nên mua thương hiệu X".
   (Khác với khuyên về MỘT SẢN PHẨM vì lỗi kỹ thuật: "loa lỗi, không nên mua dòng này" -> 0.)
B. KHẲNG ĐỊNH bị lừa / gian dối, dù chỉ kể sự việc, không chửi:
   "nhân viên CỐ TÌNH tư vấn sai", "nv nói xạo để bán hàng", "thợ ăn gian ống đồng",
   "hóa đơn ghi chính hãng mà hàng nhận là đồ dỏm", "bán sạc cũ như hàng đã dùng", "dắt khách".
   (Chỉ NGHI NGỜ / HỎI thì vẫn là 0: "không biết có phải hàng trưng bày không?" -> 0.)
C. MIỆT THỊ, hạ thấp cửa hàng / nhân viên, hoặc dùng thành ngữ nặng nề:
   "bán hàng thua kiểu chợ đen", "thu ngân xem thường khách hàng", "tiền mất tật mang"
   (kể cả khi dùng để khuyên người khác).

## Nhãn 0 = NON-TOXIC (bình thường)
1. Khen, hài lòng: "máy dùng êm, làm lạnh nhanh", "giao hàng đúng hẹn".
2. Hỏi đáp, thắc mắc: "máy này có đổi trả được không?", "chỉnh nhiệt độ thế nào?".
3. CHÊ / BÁO LỖI / PHÀN NÀN nhưng LỊCH SỰ, chỉ kể sự việc -> VẪN LÀ 0. Ví dụ:
   - "Máy dùng hơi ồn lúc vắt, mong hãng cải thiện" -> 0
   - "Mới mua 2 ngày đã lỗi không lên nguồn, đã gọi bảo hành" -> 0
   - "Gọi bảo hành 3 lần không thấy ai đến, rất thất vọng" -> 0 (thất vọng nhưng không lăng mạ)
   - "Sản phẩm kém, không đáng tiền, sẽ không mua lại" -> 0 (chê nhưng không xúc phạm)

## Mẹo khi phân vân
- Hỏi: "Câu này có XÚC PHẠM ai, CÁO BUỘC lừa đảo, hay dùng từ THÔ TỤC / MIỆT THỊ không?"
  Có -> 1. Chỉ là không hài lòng -> 0.
- Một từ "tệ", "kém", "thất vọng" đơn lẻ CHƯA đủ để gán 1.
- Mỉa mai cay độc, miệt thị (vd "bán toàn đồ đồng nát", "nhân viên như cái đầu gối") -> 1.
"""

ANSWER_FORMAT = """\
## Định dạng trả lời (BẮT BUỘC)
Chỉ trả về MỘT mảng JSON, không markdown, không giải thích thêm.
Mảng có đúng {n} phần tử, mỗi phần tử ứng với 1 dòng dữ liệu, giữ NGUYÊN "id":
[{{"id": 123, "label": 0}}, {{"id": 456, "label": 1, "ly_do": "chửi nhân viên 'mất dạy'"}}]
- "label" chỉ là 0 hoặc 1.
- Khi label = 1 phải kèm "ly_do" ngắn (tối đa 15 từ). Khi label = 0 không cần.
"""



def load_split(name):
    path = SPLITS[name]
    return pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)


# ----------------------------------------------------------------------------- prepare
def cmd_prepare(args):
    PROMPTS.mkdir(parents=True, exist_ok=True)
    RESPONSES.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split in SPLITS:
        df = load_split(split)
        for b, start in enumerate(range(0, len(df), args.batch_size), start=1):
            chunk = df.iloc[start:start + args.batch_size]
            name = f"{split}_{b:03d}"
            lines = [json.dumps({"id": int(r.review_id), "text": str(r.noi_dung)}, ensure_ascii=False)
                     for r in chunk.itertuples()]
            body = (f"# Batch {name} ({len(chunk)} bình luận)\n\n{GUIDELINES}\n"
                    f"{ANSWER_FORMAT.format(n=len(chunk))}\n## Dữ liệu (mỗi dòng 1 bình luận)\n"
                    + "\n".join(lines) + "\n")
            (PROMPTS / f"{name}.md").write_text(body, encoding="utf-8")
            manifest += [{"split": split, "batch": name, "review_id": int(i)} for i in chunk["review_id"]]
        print(f"[-] {split}: {len(df):,} câu -> {-(-len(df) // args.batch_size)} batch")
    pd.DataFrame(manifest).to_csv(MANIFEST, index=False)
    print(f"[-] Đã tạo prompt trong {PROMPTS.relative_to(ROOT)}/")


# ----------------------------------------------------------------------------- collect
def parse_response(path, expected_ids):
    """Đọc một file kết quả -> (DataFrame, lỗi). Chấp nhận JSON bọc trong ```json```."""
    m = re.search(r"\[.*\]", path.read_text(encoding="utf-8"), flags=re.S)
    if not m:
        return None, "không tìm thấy mảng JSON"
    try:
        out = pd.DataFrame(json.loads(m.group(0)))
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"JSON lỗi: {e}"
    if not {"id", "label"} <= set(out.columns):
        return None, "thiếu khóa id/label"
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    if "ly_do" not in out.columns:
        out["ly_do"] = ""
    got, exp = set(out["id"].dropna().astype(int)), set(expected_ids)
    errs = []
    if not out["label"].isin([0, 1]).all():
        errs.append("có label khác 0/1")
    if out["id"].duplicated().any():
        errs.append("trùng id")
    if exp - got:
        errs.append(f"thiếu {len(exp - got)} id")
    if got - exp:
        errs.append(f"thừa {len(got - exp)} id")
    if errs:
        return None, "; ".join(errs)
    out = out.rename(columns={"id": "review_id", "label": "llm_label"})
    out = out.astype({"review_id": int, "llm_label": int})
    return out[["review_id", "llm_label", "ly_do"]], None


def write_spotcheck(df, path):
    """File rà soát: cột nhan_sua chỉ nhận 0/1."""
    df.to_excel(path, index=False)
    wb = load_workbook(path)
    ws = wb.active
    widths = {"noi_dung": 90, "ly_do": 40}
    dv = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
    ws.add_data_validation(dv)
    for idx, col in enumerate(df.columns, start=1):
        letter = ws.cell(row=1, column=idx).column_letter
        ws.cell(row=1, column=idx).font = Font(bold=True)
        ws.column_dimensions[letter].width = widths.get(col, 12)
        for r in range(2, len(df) + 2):
            if col in widths:
                ws.cell(row=r, column=idx).alignment = Alignment(wrap_text=True, vertical="top")
            if col == "nhan_sua":
                ws.cell(row=r, column=idx).fill = PatternFill("solid", fgColor="FFF2CC")
        if col == "nhan_sua":
            dv.add(f"{letter}2:{letter}{len(df) + 1}")
    ws.freeze_panes = "A2"
    wb.save(path)


def cmd_collect(args):
    manifest = pd.read_csv(MANIFEST)
    frames, missing, invalid = [], [], []
    for (split, name), grp in manifest.groupby(["split", "batch"], sort=True):
        path = RESPONSES / f"{name}.json"
        if not path.exists():
            missing.append(name)
            continue
        df, err = parse_response(path, grp["review_id"].tolist())
        if err:
            invalid.append((name, err))
        else:
            frames.append(df.assign(split=split, batch=name))

    print(f"[-] Kết quả: {len(frames)}/{manifest['batch'].nunique()} batch hợp lệ, "
          f"{len(missing)} chưa có, {len(invalid)} lỗi")
    for name, err in invalid:
        print(f"    [lỗi] {name}.json: {err}")
    if missing:
        print(f"    Chưa có: {', '.join(missing[:15])}{' ...' if len(missing) > 15 else ''}")
    if not frames:
        return

    labels = pd.concat(frames, ignore_index=True)
    labels.to_csv(LLM_LABELS, index=False, encoding="utf-8-sig")
    for split, part in labels.groupby("split", sort=False):
        print(f"    {split}: {len(part):,} câu, toxic = {part['llm_label'].sum()} ({part['llm_label'].mean():.2%})")

    # So với nhãn người trên các câu test đã được gán.
    test = load_split("test")
    test["human"] = test[HUMAN_COLS].bfill(axis=1).iloc[:, 0]
    gold = test.dropna(subset=["human"]).merge(labels, on="review_id")
    if len(gold):
        y, p = gold["human"].astype(int), gold["llm_label"]
        print(f"[-] LLM vs nhãn người ({len(gold)} câu test): Kappa={cohen_kappa_score(y, p):.3f} "
              f"| P={precision_score(y, p, zero_division=0):.3f} | R={recall_score(y, p, zero_division=0):.3f} "
              f"| F1={f1_score(y, p, zero_division=0):.3f}")

    # File rà soát các câu LLM gán 1 trong train/val (tạo khi đã đủ batch, không ghi đè).
    pending = [b for b in missing + [n for n, _ in invalid] if not b.startswith("test")]
    if pending:
        print("[-] Chưa đủ batch train/val, chưa tạo file rà soát.")
    elif SPOTCHECK.exists():
        print(f"[-] {SPOTCHECK.relative_to(ROOT)} đã tồn tại, không ghi đè.")
    else:
        texts = pd.concat([load_split("train"), load_split("val")])[["review_id", "noi_dung"]]
        pos = labels[(labels["split"] != "test") & (labels["llm_label"] == 1)]
        pos = pos.merge(texts, on="review_id")[["review_id", "split", "noi_dung", "ly_do", "llm_label"]]
        pos["nhan_sua"] = pd.NA
        write_spotcheck(pos, SPOTCHECK)
        print(f"[-] Đã tạo {SPOTCHECK.relative_to(ROOT)} ({len(pos)} câu). "
              f"Điền nhan_sua = 0 nếu LLM gán sai, để trống nếu đồng ý.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "collect"])
    ap.add_argument("--batch-size", type=int, default=100)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "collect": cmd_collect}[args.command](args)


if __name__ == "__main__":
    main()
