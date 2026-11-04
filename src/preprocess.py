"""
Bước 2: Tiền xử lý dữ liệu văn bản tiếng Việt đặc thù cho bình luận Điện Máy Xanh.
Bao gồm:
1. Chuẩn hóa bảng mã Unicode NFC & làm sạch nhiễu (HTML, URL, SĐT, mã đơn hàng).
2. Khử ký tự lặp kéo dài & giải mã từ viết lách kiểm duyệt (obfuscated toxic words: l.ừ.a đ.ả.o -> lừa đảo).
3. Từ điển chuẩn hóa Teencode & từ viết tắt đặc thù Điện Máy Xanh (đmx, nv, bh, sp, vkl, đm...).
4. Ánh xạ Emoji cảm xúc mạnh sang token đặc trưng.
5. Tách từ tiếng Việt (Word Segmentation) sử dụng PyVi / Underthesea.
6. Loại bỏ Stopwords tiếng Việt có chọn lọc (giữ lại từ phủ định & đại từ mang sắc thái Toxic).
"""

import re
import unicodedata
from typing import List, Dict, Optional

import pandas as pd
from pyvi import ViTokenizer


# Từ điển Teencode & Viết tắt đặc thù bình luận Điện Máy Xanh
TEENCODE_DICT: Dict[str, str] = {
    "đmx": "điện máy xanh",
    "dmx": "điện máy xanh",
    "tgdd": "thế giới di động",
    "nv": "nhân viên",
    "nvien": "nhân viên",
    "ktv": "kỹ thuật viên",
    "bh": "bảo hành",
    "sp": "sản phẩm",
    "đh": "đơn hàng",
    "sđt": "số điện thoại",
    "sdt": "số điện thoại",
    "mn": "mọi người",
    "ad": "quản trị viên",
    "admin": "quản trị viên",
    "ko": "không",
    "k": "không",
    "kh": "không",
    "hong": "không",
    "đc": "được",
    "dc": "được",
    "vs": "với",
    "ntn": "như thế nào",
    "ok": "tốt",
    "oke": "tốt",
    # Các từ viết tắt mang sắc thái Toxic / Chửi thề
    "đm": "địt mẹ",
    "dm": "địt mẹ",
    "đkm": "địt con mẹ",
    "dkm": "địt con mẹ",
    "vcl": "vãi cả lồn",
    "vkl": "vãi cả lồn",
    "vl": "vãi lồn",
    "cc": "cục cứt",
    "cl": "cái lồn",
    "deo": "đéo",
}

# Các mẫu viết lách kiểm duyệt (Obfuscated words) thường gặp
OBFUSCATED_PATTERNS = [
    (r"l[\.\*\-_,\s]+ừ[\.\*\-_,\s]+a\s+đ[\.\*\-_,\s]+ả[\.\*\-_,\s]+o", "lừa đảo"),
    (r"m[\.\*\-_,\s]+ấ[\.\*\-_,\s]+t\s+d[\.\*\-_,\s]+ạ[\.\*\-_,\s]+y", "mất dạy"),
    (r"v[\.\*\-_,\s]+ô\s+h[\.\*\-_,\s]+ọ[\.\*\-_,\s]+c", "vô học"),
    (r"b[\.\*\-_,\s]+ố\s+l[\.\*\-_,\s]+á[\.\*\-_,\s]+o", "bố láo"),
    (r"đ[\.\*\-_]+m", "địt mẹ"),
    (r"v[\.\*\-_]+l", "vãi lồn"),
]

# Ánh xạ Emoji sang token văn bản
EMOJI_MAP: Dict[str, str] = {
    "😡": " emoji_tuc_gian ",
    "🤬": " emoji_chui_the ",
    "👎": " emoji_che_bai ",
    "💩": " emoji_rac_ruoi ",
    "😤": " emoji_buc_xuc ",
    "👍": " emoji_hai_long ",
    "❤️": " emoji_yeu_thich ",
    "😊": " emoji_vui_ve ",
}

# Danh sách Stopwords tiếng Việt (đã loại trừ các từ phủ định & đại từ miệt thị quan trọng cho bài toán Toxic)
VIETNAMESE_STOPWORDS = {
    "là", "và", "của", "có", "những", "các", "một", "được", "trong", "đã",
    "cho", "với", "này", "khi", "đến", "từ", "về", "như", "ở", "cũng",
    "ra", "lại", "thì", "mà", "tại", "theo", "do", "vì", "nếu", "nhưng",
    "hoặc", "hay", "rồi", "vào", "lên", "xuống", "qua", "tới", "nơi",
    "lúc", "ngày", "tháng", "năm", "hôm", "qua", "nay", "sáng", "tối",
    "tuần", "trước", "sau", "nữa", "thêm", "cái", "chiếc", "con", "người",
    "mình", "tôi", "bạn", "shop", "ạ", "nhé", "nha", "vậy", "ơi", "luôn",
    "vừa", "mới", "đang", "sẽ", "cần", "muốn", "hỏi", "xin", "phép",
}
# Lưu ý: KHÔNG đưa các từ sau vào stopwords vì chúng quyết định nhãn Toxic/Non-toxic:
# - Từ phủ định/mức độ: 'không', 'chẳng', 'chưa', 'đéo', 'rất', 'quá', 'cực_kỳ'
# - Đại từ miệt thị: 'lũ', 'bọn', 'chúng_mày', 'mày', 'tao', 'chúng_nó', 'thằng', 'đứa'


class VietnameseTextPreprocessor:
    """
    Bộ tiền xử lý văn bản tiếng Việt cho bài toán phân loại bình luận Toxic / Non-toxic.
    """

    def __init__(self, remove_stopwords: bool = True, lowercase: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.obfuscated_regexes = [
            (re.compile(pat, flags=re.IGNORECASE | re.UNICODE), repl)
            for pat, repl in OBFUSCATED_PATTERNS
        ]

    def normalize_unicode_and_noise(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # 1. Chuẩn hóa Unicode dựng sẵn NFC
        text = unicodedata.normalize("NFC", text)
        # 2. Xóa HTML tags & URL
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # 3. Xóa mã đơn hàng (#12345) và số điện thoại
        text = re.sub(r"#\d+", " ", text)
        text = re.sub(r"\b\d{8,11}\b", " ", text)
        # 4. Thay thế Emoji cảm xúc mạnh bằng token
        for emo, token in EMOJI_MAP.items():
            text = text.replace(emo, token)
        return text

    def normalize_teencode_and_obfuscation(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        # 1. Giải mã các từ viết lách kiểm duyệt (l.ừ.a đ.ả.o -> lừa đảo)
        for regex, repl in self.obfuscated_regexes:
            text = regex.sub(repl, text)
        # 2. Thu gọn ký tự lặp kéo dài (tệệệệ -> tệ, quááá -> quá)
        text = re.sub(r"(\w)\1{2,}", r"\1", text, flags=re.UNICODE)
        # 3. Loại bỏ dấu câu/ký tự đặc biệt thừa (giữ lại chữ cái tiếng Việt, số và gạch dưới)
        text = re.sub(r"[^\w\s_]", " ", text, flags=re.UNICODE)
        # 4. Ánh xạ từ điển Teencode & viết tắt
        tokens = text.split()
        normalized_tokens = [TEENCODE_DICT.get(tok, tok) for tok in tokens]
        return " ".join(normalized_tokens)

    def segment_and_filter(self, text: str, remove_stopwords: Optional[bool] = None) -> str:
        use_stop = self.remove_stopwords if remove_stopwords is None else remove_stopwords
        # Tách từ tiếng Việt bằng PyVi (nhân viên -> nhân_viên, lừa đảo -> lừa_đảo)
        segmented = ViTokenizer.tokenize(text)
        tokens = segmented.split()
        if use_stop:
            tokens = [
                t for t in tokens
                if t not in VIETNAMESE_STOPWORDS and not t.isdigit() and len(t) > 1
            ]
        else:
            tokens = [t for t in tokens if len(t) > 0]
        return " ".join(tokens)

    def preprocess_text(self, text: str, for_transformer: bool = False) -> str:
        """
        Thực thi toàn bộ pipeline tiền xử lý cho 1 câu bình luận.
        - Nếu for_transformer=False (dùng cho BoW, TF-IDF, Word2Vec + NB/SVM): xóa stopwords để giảm nhiễu.
        - Nếu for_transformer=True (dùng cho PhoBERT / ViSoBERT): giữ nguyên cấu trúc ngữ pháp + tách từ.
        """
        step1 = self.normalize_unicode_and_noise(text)
        step2 = self.normalize_teencode_and_obfuscation(step1)
        step3 = self.segment_and_filter(step2, remove_stopwords=(not for_transformer))
        return step3

    def preprocess_dataframe(self, df: pd.DataFrame, text_col: str = "raw_text") -> pd.DataFrame:
        """
        Tiền xử lý toàn bộ DataFrame và tạo 2 cột:
        - clean_text: đã tách từ + loại bỏ stopwords (cho ML truyền thống: Naive Bayes, SVM)
        - phobert_text: đã chuẩn hóa + tách từ nhưng giữ ngữ cảnh câu (cho PhoBERT / ViSoBERT)
        """
        df = df.copy()
        df["clean_text"] = df[text_col].apply(
            lambda x: self.preprocess_text(x, for_transformer=False)
        )
        df["phobert_text"] = df[text_col].apply(
            lambda x: self.preprocess_text(x, for_transformer=True)
        )
        df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
        return df


if __name__ == "__main__":
    processor = VietnameseTextPreprocessor()
    sample = "Đm bọn đmx làm ăn l.ừ.a đ.ả.o vkl!! NV thái độ bố láo mất d.ạ.y 😡🤬 sp lỗi tệệệệ quá!"
    print("Câu gốc      :", sample)
    print("Cho NB / SVM :", processor.preprocess_text(sample, for_transformer=False))
    print("Cho PhoBERT  :", processor.preprocess_text(sample, for_transformer=True))
