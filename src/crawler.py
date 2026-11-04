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
    """Bộ gán nhãn hỗ trợ kết hợp từ điển và luật ngữ cảnh."""

    STRONG_TOXIC_PATTERNS = [
        r"\b(đm|dm|đkm|dkm|vcl|vkl|vl|đéo|méo|như hạch|như cc|như cl|như lồn|như cặc)\b",
        r"\b(bố láo|mất dạy|vô học|vô văn hóa|vô liêm sỉ|óc chó|óc lợn|ngu dốt|ngu ngốc)\b",
        r"\b(lũ ăn hại|bọn ăn hại|đồ vô tích sự|khốn nạn|chó má|rác rưởi|khốn kiếp)\b",
        r"\b(lũ lừa đảo|bọn lừa đảo|lừa đảo|lừa bịp|bịp bợm|ăn cướp|bọn cướp|móc túi|hút máu)\b",
        r"\b(treo đầu dê|chém đẹp|ăn chặn|làm ăn bát nháo|thất đức|bố đời|hách dịch|mẹ thiên hạ)\b",
        r"\b(khinh người|khinh khách|coi thường khách|khinh thường khách)\b",
        r"\b(thằng lắp|thằng nv|thằng nhân viên|thằng thợ|thằng shipper|con nv|con nhân viên)\b",
    ]

    HOSTILE_COMPLAINT_PATTERNS = [
        r"(quá tệ|rất tệ|siêu tệ|cực kỳ tệ|vô cùng tệ|tệ hại|tệ nhất|tệ quá|quá kém|rất kém|cực kỳ kém)",
        r"(quá thất vọng|rất thất vọng|thất vọng cùng cực|thất vọng tràn trề|kinh khủng|kinh hoàng)",
        r"(bực cả mình|quá bực mình|bực mình|bực bội|ức chế|bức xúc|phí tiền|tiếc tiền|phí thời gian)",
        r"(hối hận khi mua|hối hận vì mua|hối hận|không bao giờ mua|k bao giờ mua|ko bao giờ mua)",
        r"(lần cuối cùng mua|lần đầu cũng như lần cuối|cạch mặt|tẩy chay|cút ngay|né gấp|né ra|tránh xa)",
        r"(khuyên mọi người đừng|khuyên mn đừng|khuyên không nên|đừng ai mua|đừng dại mua|đừng bao giờ mua)",
        r"(không nên mua|k nên mua|ko nên mua|chớ mua|đừng mua|làm ăn như trò hề|như trò đùa|chả ra gì|không ra gì)",
        r"(đồ đểu|hàng đểu|dây đểu|rước bực|rước họa|bỏ con giữa chợ|đùn đẩy trách nhiệm|đùn đẩy)",
        r"(vô trách nhiệm|thiếu trách nhiệm|chối bỏ trách nhiệm|chối quanh|hành khách|hành hạ khách)",
        r"(cho leo cây|mất uy tín|đổ lỗi cho khách|đổ lỗi cho người dùng|đổ thừa|làm ăn rất tệ|làm ăn quá tệ)",
    ]

    NEGATION_PATTERN = r"\b(không hề|chẳng hề|ko hề|k hề|không phải|chả phải)\s+(lừa|bố láo|mất dạy|tệ|kém)"

    def __init__(self):
        self.strong_patterns = [
            re.compile(pat, flags=re.IGNORECASE | re.UNICODE)
            for pat in self.STRONG_TOXIC_PATTERNS
        ]
        self.hostile_patterns = [
            re.compile(pat, flags=re.IGNORECASE | re.UNICODE)
            for pat in self.HOSTILE_COMPLAINT_PATTERNS
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
        if rating is not None and rating <= 2:
            for pat in self.hostile_patterns:
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
