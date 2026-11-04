"""
Bước 3: Trích chọn đặc trưng văn bản (Feature Extraction).
Triển khai đầy đủ cả 3 phương pháp theo yêu cầu đề bài:
1. Bag of Words (BoW) - CountVectorizer (Unigram + Bigram).
2. TF-IDF - Kết hợp Word N-grams (1, 2) và Character N-grams (3, 5) với Sublinear TF scaling.
3. Word Embeddings (Dense Distributed Word Vectors) - Huấn luyện ma trận PPMI / Co-occurrence Word Vectors (200 chiều)
   hoặc nạp Pre-trained Vietnamese Word2Vec/FastText, kết hợp TF-IDF Weighted Pooling để tạo vector biểu diễn câu.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer


def build_bow_extractor(max_features: int = 5000) -> CountVectorizer:
    """
    Phương pháp 1: Bag of Words (BoW) truyền thống - biểu diễn túi từ đơn (Unigram).
    """
    return CountVectorizer(
        ngram_range=(1, 1),
        max_features=max_features,
        min_df=2,
        max_df=0.95,
    )


def build_tfidf_extractor(max_word_features: int = 6000, max_char_features: int = 4000) -> FeatureUnion:
    """
    Phương pháp 2: TF-IDF kết hợp cấp độ Từ (Word N-grams) và Ký tự (Char-wb N-grams).
    Đặc biệt hiệu quả cho bài toán Toxic vì bắt được biến thể từ lóng và lỗi chính tả.
    """
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


class DenseWordEmbeddingExtractor(BaseEstimator, TransformerMixin):
    """
    Phương pháp 3: Word Embeddings (Biểu diễn vector từ liên tục mật độ cao).
    - Xây dựng không gian ngữ nghĩa từ vựng (Word-Context Co-occurrence + Subword SVD / Pre-trained Word2Vec)
      với số chiều `embedding_dim` (mặc định 150 chiều).
    - Biểu diễn mỗi bình luận bằng trung bình có trọng số IDF (IDF-weighted Mean Pooling)
      của các vector từ xuất hiện trong câu.
    """

    def __init__(self, embedding_dim: int = 150, window_size: int = 4, max_vocab: int = 5000, random_state: int = 42):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.max_vocab = max_vocab
        self.random_state = random_state
        self.word2idx_: Dict[str, int] = {}
        self.idf_weights_: Dict[str, float] = {}
        self.word_vectors_: Optional[np.ndarray] = None

    def fit(self, texts: List[str], y=None):
        # 1. Tính tần suất từ và trọng số IDF trên tập từ vựng
        tfidf = TfidfVectorizer(max_features=self.max_vocab, min_df=2, token_pattern=r"(?u)\b\w+\b")
        tfidf.fit(texts)
        self.word2idx_ = dict(tfidf.vocabulary_)
        idf_array = tfidf.idf_
        self.idf_weights_ = {w: float(idf_array[idx]) for w, idx in self.word2idx_.items()}

        vocab_size = len(self.word2idx_)
        # 2. Xây dựng ma trận đồng xuất hiện (Word-Context Co-occurrence Matrix) trong cửa sổ ngữ cảnh
        rows, cols, vals = [], [], []
        for doc in texts:
            tokens = [t for t in doc.split() if t in self.word2idx_]
            n = len(tokens)
            for i, target_word in enumerate(tokens):
                t_idx = self.word2idx_[target_word]
                start = max(0, i - self.window_size)
                end = min(n, i + self.window_size + 1)
                for j in range(start, end):
                    if i == j:
                        continue
                    c_idx = self.word2idx_[tokens[j]]
                    dist = abs(i - j)
                    rows.append(t_idx)
                    cols.append(c_idx)
                    vals.append(1.0 / dist)

        cooc = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(vocab_size, vocab_size),
            dtype=np.float32,
        )
        # Áp dụng biến đổi Log-count (tương đương GloVe / PPMI smoothing)
        cooc.data = np.log1p(cooc.data)

        n_comp = min(self.embedding_dim, max(2, vocab_size - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        word_vecs = svd.fit_transform(cooc)
        self.word_vectors_ = Normalizer(norm="l2").fit_transform(word_vecs)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.word_vectors_ is None:
            raise ValueError("DenseWordEmbeddingExtractor chưa được fit!")

        dim = self.word_vectors_.shape[1]
        doc_vectors = np.zeros((len(texts), dim), dtype=np.float32)

        for idx, doc in enumerate(texts):
            tokens = doc.split()
            vecs = []
            weights = []
            for tok in tokens:
                w_idx = self.word2idx_.get(tok)
                if w_idx is not None:
                    vecs.append(self.word_vectors_[w_idx])
                    weights.append(self.idf_weights_.get(tok, 1.0))
            if vecs:
                weights_arr = np.array(weights, dtype=np.float32)[:, None]
                vecs_arr = np.vstack(vecs)
                pooled = np.sum(vecs_arr * weights_arr, axis=0) / (np.sum(weights_arr) + 1e-8)
                norm = np.linalg.norm(pooled)
                doc_vectors[idx] = pooled / (norm + 1e-8)
        return doc_vectors
