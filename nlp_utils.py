"""Tiền xử lý văn bản tiếng Việt.

Thứ tự: chuẩn hóa Unicode -> chữ thường -> URL/SĐT -> ký tự lặp -> teencode
-> xóa ký tự đặc biệt -> tách từ (pyvi) -> lọc stop words.
"""
import re
import unicodedata

from pyvi import ViTokenizer

# Chuẩn hóa vị trí dấu ở cuối từ: "hoà" -> "hòa", "thuỷ" -> "thủy".
_TONE_MAP = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}
_TONE_RE = re.compile(r"(" + "|".join(_TONE_MAP) + r")\b")

TEENCODE = {
    "đmx": "điện máy xanh", "dmx": "điện máy xanh",
    "tgdđ": "thế giới di động", "tgdd": "thế giới di động",
    "nv": "nhân viên", "nvien": "nhân viên",
    "sp": "sản phẩm", "sản fẩm": "sản phẩm",
    "bh": "bảo hành", "bhành": "bảo hành",
    "ko": "không", "kh": "không", "khong": "không", "hok": "không", "hông": "không",
    "k0": "không", "kg": "không", "hem": "không",
    "dc": "được", "đc": "được", "duoc": "được", "đk": "được",
    "j": "gì", "gi": "gì",
    "mn": "mọi người", "mng": "mọi người",
    "ok": "ổn", "oke": "ổn", "okela": "ổn", "okie": "ổn",
    "tl": "trả lời", "sd": "sử dụng", "sdung": "sử dụng",
    "vs": "với", "z": "vậy", "v": "vậy", "zậy": "vậy",
    "r": "rồi", "ròi": "rồi", "rùi": "rồi",
    "bt": "bình thường", "bthg": "bình thường",
    "tks": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn", "cám ơn": "cảm ơn",
    "lm": "làm", "ns": "nói", "nch": "nói chuyện",
    "kt": "kỹ thuật", "ktv": "kỹ thuật viên",
    "đt": "điện thoại", "dt": "điện thoại",
    "ad": "quản trị viên", "shop": "cửa hàng",
    "qá": "quá", "wá": "quá", "wa": "quá",
}
_TEENCODE_RE = re.compile(
    r"(?<![\w])(" + "|".join(sorted(map(re.escape, TEENCODE), key=len, reverse=True)) + r")(?![\w])"
)
# Cụm từ pyvi tách sai, ghép sẵn thành một token.
COMPOUNDS = ["điện máy xanh", "thế giới di động", "kỹ thuật viên"]
_COMPOUND_RE = re.compile(r"(?<![\w])(" + "|".join(COMPOUNDS) + r")(?![\w])")
# "k" đứng riêng là "không"; bỏ qua "500k", "500 k", "4k".
_K_RE = re.compile(r"(?<![\w])(?<!\d )k(?![\w])")

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_PHONE_RE = re.compile(r"(?<!\d)(\+84|0)\d{9,10}(?!\d)")
_REPEAT_RE = re.compile(r"(\w)\1{2,}")
_JUNK_RE = re.compile(r"[^\w\s<>]")
_SPACE_RE = re.compile(r"\s+")

STOP_WORDS = {
    "thì", "là", "mà", "và", "của", "những", "các", "cái", "chiếc", "này", "kia",
    "ấy", "đó", "nọ", "với", "cho", "để", "nên", "vì", "do", "nhé", "nha", "ạ",
    "à", "ơi", "ừ", "thế", "vậy", "rồi", "đã", "đang", "sẽ", "cũng", "vẫn", "lại",
    "nữa", "có_thể", "một", "mình", "tôi", "em", "anh", "chị", "bạn", "ở", "tại",
    "trong", "ra", "vào", "lên", "xuống", "khi", "lúc", "nếu", "như", "từ",
}
# Từ phủ định và từ chỉ mức độ/cảm xúc luôn được giữ lại.
WHITELIST = {
    "không", "chẳng", "chưa", "đừng", "chớ", "bị", "kém", "tệ", "quá", "rất",
    "lắm", "hỏng", "lỗi", "lừa",
}


def normalize(text):
    """Chuẩn hóa mức ký tự, chưa tách từ."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    text = _TONE_RE.sub(lambda m: _TONE_MAP[m.group(1)], text)
    text = _URL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" <phone> ", text)
    text = _REPEAT_RE.sub(r"\1", text)
    text = _JUNK_RE.sub(" ", text.replace("_", " "))
    text = _TEENCODE_RE.sub(lambda m: TEENCODE[m.group(1)], text)
    text = _K_RE.sub("không", text)
    text = _COMPOUND_RE.sub(lambda m: m.group(1).replace(" ", "_"), text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text):
    # pyvi tách "<phone>" thành "< phone >", nên thay tạm bằng một từ thường.
    tokens = ViTokenizer.tokenize(text.replace("<phone>", "phonetoken")).split()
    return ["<phone>" if t == "phonetoken" else t for t in tokens]


def remove_stopwords(tokens):
    return [t for t in tokens if t in WHITELIST or t not in STOP_WORDS]


def preprocess(text):
    """Văn bản thô -> chuỗi token đã tách từ, phân cách bởi dấu cách."""
    text = normalize(text)
    if not text:
        return ""
    return " ".join(remove_stopwords(tokenize(text)))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for s in [
        "Nhân viên đmx thái độ bố láo, lừa đảo khách hàng!!!",
        "Máy chạy hơi ồn lúc vắt, nhưng nv lắp đặt nhiệt tình. Giá 500k ok",
        "ko dc bh, gọi 0912345678 ko ai nghe. Hoà cả vốn, thuỷ chung quááá",
        "Tivi 4k đẹp, k có gì để chê https://dienmayxanh.com/abc",
    ]:
        print(s, "\n  ->", preprocess(s))
