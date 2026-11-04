"""Tiền xử lý văn bản tiếng Việt cho dữ liệu bình luận."""

import re
import unicodedata
from typing import List, Dict, Optional

import pandas as pd
from pyvi import ViTokenizer


# Từ điển chuẩn hóa viết tắt và teencode
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

OBFUSCATED_PATTERNS = [
    (r"l[\.\*\-_,\s]+ừ[\.\*\-_,\s]+a\s+đ[\.\*\-_,\s]+ả[\.\*\-_,\s]+o", "lừa đảo"),
    (r"m[\.\*\-_,\s]+ấ[\.\*\-_,\s]+t\s+d[\.\*\-_,\s]+ạ[\.\*\-_,\s]+y", "mất dạy"),
    (r"v[\.\*\-_,\s]+ô\s+h[\.\*\-_,\s]+ọ[\.\*\-_,\s]+c", "vô học"),
    (r"b[\.\*\-_,\s]+ố\s+l[\.\*\-_,\s]+á[\.\*\-_,\s]+o", "bố láo"),
    (r"đ[\.\*\-_]+m", "địt mẹ"),
    (r"v[\.\*\-_]+l", "vãi lồn"),
]

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

# Stopwords tiếng Việt cơ bản (giữ lại từ phủ định và đại từ quan trọng)
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


class VietnameseTextPreprocessor:
    """Xử lý chuẩn hóa và làm sạch văn bản tiếng Việt."""

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
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"#\d+", " ", text)
        text = re.sub(r"\b\d{8,11}\b", " ", text)
        for emo, token in EMOJI_MAP.items():
            text = text.replace(emo, token)
        return text

    def normalize_teencode_and_obfuscation(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        for regex, repl in self.obfuscated_regexes:
            text = regex.sub(repl, text)
        text = re.sub(r"(\w)\1{2,}", r"\1", text, flags=re.UNICODE)
        text = re.sub(r"[^\w\s_]", " ", text, flags=re.UNICODE)
        tokens = text.split()
        normalized_tokens = [TEENCODE_DICT.get(tok, tok) for tok in tokens]
        return " ".join(normalized_tokens)

    def segment_and_filter(self, text: str, remove_stopwords: Optional[bool] = None) -> str:
        use_stop = self.remove_stopwords if remove_stopwords is None else remove_stopwords
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
        step1 = self.normalize_unicode_and_noise(text)
        step2 = self.normalize_teencode_and_obfuscation(step1)
        step3 = self.segment_and_filter(step2, remove_stopwords=(not for_transformer))
        return step3

    def preprocess_dataframe(self, df: pd.DataFrame, text_col: str = "raw_text") -> pd.DataFrame:
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
