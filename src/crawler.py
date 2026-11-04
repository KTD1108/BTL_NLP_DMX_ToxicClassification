"""Thu thập và quản lý dữ liệu đánh giá sản phẩm từ Điện Máy Xanh."""

import os
import re
import time
import random
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


DEFAULT_DMX_PRODUCT_URLS = [
    "https://www.dienmayxanh.com/may-lanh/panasonic-cu-cs-pu9akh-8",
    "https://www.dienmayxanh.com/tivi/smart-tivi-samsung-4k-55-inch-ua55du8000",
    "https://www.dienmayxanh.com/may-giat/toshiba-aw-m905bv-mk",
    "https://www.dienmayxanh.com/tu-lanh/aqua-inverter-186-lit-aqr-t219fa-pb",
    "https://www.dienmayxanh.com/dien-thoai/iphone-15-pro-max",
    "https://www.dienmayxanh.com/noi-chien-khong-dau/sunhouse-shd4026",
]


class WeakLabeler:
    """Bộ gán nhãn chuẩn hóa Toxic: Chỉ gán nhãn 1 cho phát ngôn thù địch, chửi bới, xúc phạm, vu khống lừa đảo.
    Tuyệt đối không gán nhãn 1 cho các bình luận khiếu nại sản phẩm/dịch vụ thông thường ('quá tệ', 'không nên mua', 'rất kém')."""

    STRONG_TOXIC_PATTERNS = [
        # 1. Chửi thề, tục tĩu (kể cả viết tắt)
        r"\b(đm|dm|đkm|dkm|vcl|vkl|vl|đéo|mịa|mịe|như hạch|như cc|như cl|như cặc|như lồn|cục cứt|như cứt|đĩ|cụ nội|bà già mày|mẹ mày|đậu má)\b",
        # 2. Xúc phạm nhân phẩm, lăng mạ nhân viên / thợ kỹ thuật / cửa hàng
        r"\b(bố láo|mất dạy|vô học|vô văn hóa|vô liêm sỉ|óc chó|óc lợn|ngu dốt|ngu như chó|ngu như bò|lũ ăn hại|bọn ăn hại|đồ ăn hại|đồ súc vật|đồ vô tích sự|chó chết|khốn nạn|chó má|rác rưởi|khốn kiếp|bố đời|mẹ thiên hạ|thất đức|láo toét|láo xược)\b",
        # 3. Đại từ miệt thị nhân sự (thằng, con, lũ, bọn + nhân viên/thợ)
        r"\b(thằng thợ|thằng nv|thằng nhân viên|thằng kỹ thuật|thằng giao|con nv|con nhân viên|bọn nhân viên|lũ nhân viên|bọn thợ|lũ thợ|con tổng đài|thằng quản lý)\b",
        # 4. Tố cáo lừa đảo, trộm cắp, chiếm đoạt, lươn lẹo
        r"\b(lừa đảo|lừa bịp|bịp bợm|ăn cướp|bọn cướp|lũ cướp|móc túi|hút máu|treo đầu dê|chém đẹp|ăn chặn|gian dối|lươn lẹo|quỵt tiền|bọn lừa|lũ lừa|lừa gạt|lừa lọc)\b",
        # 5. Khinh miệt khách hàng nặng nề
        r"\b(khinh người|khinh khách|coi thường khách|khinh thường khách|hách dịch|coi khách như rác|hành khách|hành hạ khách)\b",
        # 6. Biến thể né kiểm duyệt
        r"(l[\.\*\-_,\s]+ừ[\.\*\-_,\s]+a\s+đ[\.\*\-_,\s]+ả[\.\*\-_,\s]+o|m[\.\*\-_,\s]+ấ[\.\*\-_,\s]+t\s+d[\.\*\-_,\s]+ạ[\.\*\-_,\s]+y|b[\.\*\-_,\s]+ố\s+l[\.\*\-_,\s]+á[\.\*\-_,\s]+o|v[\.\*\-_,\s]+ô\s+h[\.\*\-_,\s]+ọ[\.\*\-_,\s]+c)",
    ]

    NEGATION_PATTERN = r"\b(không hề|chẳng hề|ko hề|k hề|không phải|chả phải|tránh bị|sợ bị|không có chuyện)\s+(lừa|bố láo|mất dạy|ăn cướp)"

    def __init__(self):
        self.strong_patterns = [
            re.compile(pat, flags=re.IGNORECASE | re.UNICODE)
            for pat in self.STRONG_TOXIC_PATTERNS
        ]
        self.negation_re = re.compile(self.NEGATION_PATTERN, flags=re.IGNORECASE | re.UNICODE)

    def predict_label(self, text: str, rating: int = 3) -> int:
        if not isinstance(text, str) or not text.strip():
            return 0
        if self.negation_re.search(text):
            return 0
        for pat in self.strong_patterns:
            if pat.search(text):
                return 1
        return 0


class DienMayXanhCrawler:
    """Crawl đánh giá và bình luận từ website dienmayxanh.com."""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.dienmayxanh.com/",
    }

    def __init__(self, timeout: int = 10, delay: float = 0.3):
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.labeler = WeakLabeler()

    def crawl_product_page(self, url: str) -> List[Dict]:
        records = []
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                return records

            soup = BeautifulSoup(resp.text, "html.parser")
            category = self._infer_category_from_url(url)

            review_selectors = [
                "div.comment-item",
                "li.par",
                "div.rt-list li",
                "p.cmt-txt",
                "div.cmt-content",
            ]
            seen_texts = set()
            for sel in review_selectors:
                for node in soup.select(sel):
                    txt_node = node.select_one(".cmt-txt, .cmt-content, p") or node
                    text = txt_node.get_text(" ", strip=True)
                    if text and len(text) >= 10 and text not in seen_texts:
                        seen_texts.add(text)
                        records.append({
                            "source_url": url,
                            "category": category,
                            "comment_type": "review",
                            "rating": self._extract_star(node),
                            "raw_text": text,
                            "label": self.labeler.predict_label(text),
                        })
            time.sleep(self.delay)
        except Exception:
            pass
        return records

    @staticmethod
    def _infer_category_from_url(url: str) -> str:
        mapping = {
            "may-lanh": "Máy lạnh",
            "tivi": "Tivi",
            "may-giat": "Máy giặt",
            "tu-lanh": "Tủ lạnh",
            "dien-thoai": "Điện thoại",
            "noi-chien": "Gia dụng",
        }
        for key, val in mapping.items():
            if key in url:
                return val
        return "Gia dụng"

    @staticmethod
    def _extract_star(node) -> int:
        stars = node.select(".icon-star, .iconcmt-starbuy, .active")
        if stars and 1 <= len(stars) <= 5:
            return len(stars)
        return 3


def collect_or_build_dataset(
    output_csv_path: str = "data/raw/dmx_reviews_raw.csv",
    n_samples: int = 15000,
    try_live_crawl: bool = True,
) -> pd.DataFrame:
    """Nạp hoặc tạo tệp dữ liệu bình luận từ Điện Máy Xanh."""
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    if os.path.exists(output_csv_path):
        df_existing = pd.read_csv(output_csv_path)
        if len(df_existing) >= 5000:
            print(f"[Dataset] Đã tải dữ liệu từ {output_csv_path}: {len(df_existing):,} mẫu.")
            return df_existing

    crawled_records = []
    if try_live_crawl:
        crawler = DienMayXanhCrawler(timeout=5, delay=0.2)
        for url in DEFAULT_DMX_PRODUCT_URLS[:3]:
            items = crawler.crawl_product_page(url)
            crawled_records.extend(items)

    df_base = pd.DataFrame(crawled_records)
    if not df_base.empty:
        df_base.insert(0, "comment_id", [f"DMX_LIVE_{i+1:04d}" for i in range(len(df_base))])
        df_base.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    return df_base
