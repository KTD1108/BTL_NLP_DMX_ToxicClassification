"""Trích chọn đặc trưng văn bản bằng Bag of Words và TF-IDF."""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_bow_extractor(max_features: int = 5000) -> CountVectorizer:
    """Xây dựng bộ trích xuất đặc trưng Bag of Words."""
    return CountVectorizer(
        ngram_range=(1, 1),
        max_features=max_features,
        min_df=2,
        max_df=0.95,
    )


def build_tfidf_extractor(max_word_features: int = 6000, max_char_features: int = 4000) -> FeatureUnion:
    """Xây dựng bộ trích xuất đặc trưng TF-IDF kết hợp Word và Char N-grams."""
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_word_features,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_char_features,
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    return FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf),
    ])
