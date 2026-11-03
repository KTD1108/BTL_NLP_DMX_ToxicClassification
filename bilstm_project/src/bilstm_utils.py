"""
bilstm_utils.py
Các thành phần dùng chung cho mô hình BiLSTM phát hiện đánh giá toxic (Điện Máy Xanh).

Gồm: set_seed, Vocab, ToxicDataset + DataLoader, ma trận Word Embeddings (PhoW2V),
BiLSTMClassifier, vòng train/eval, metrics, chọn ngưỡng, vẽ learning curve,
lưu/nạp checkpoint.
"""

import os
import random
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score, confusion_matrix,
)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


# =============================================================================
# 1. Reproducibility
# =============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 2. Tokenize + Vocab
# =============================================================================
def tokenize(text) -> list:
    """Cột noi_dung_tien_xu_ly đã được nhóm tiền xử lý sẵn bằng nlp_utils.py
    (chữ thường, teencode, tách từ nối bằng '_', bỏ stopword),
    nên chỉ cần split theo khoảng trắng để lấy danh sách token."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    return str(text).split()


class Vocab:
    """Từ điển word -> index. Index 0 = <pad>, 1 = <unk>. Chỉ build trên tập train."""

    def __init__(self, min_freq: int = 2, max_size: int = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.itos = [PAD_TOKEN, UNK_TOKEN]
        self.stoi = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.freqs = Counter()

    # Build vocab từ danh sách văn bản (tokenize + đếm tần suất)
    def build(self, texts):
        for t in texts:
            self.freqs.update(tokenize(t))
        words = [w for w, c in self.freqs.most_common() if c >= self.min_freq]
        if self.max_size is not None:
            words = words[: self.max_size - 2]
        self.itos = [PAD_TOKEN, UNK_TOKEN] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        return self

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len: int = None) -> list:
        ids = [self.stoi.get(tok, UNK_IDX) for tok in tokenize(text)]
        if max_len is not None:
            ids = ids[:max_len]
        if len(ids) == 0:          # câu rỗng -> 1 token <unk> để LSTM không lỗi length=0
            ids = [UNK_IDX]
        return ids

    def oov_rate(self, texts) -> float:
        total, unk = 0, 0
        for t in texts:
            toks = tokenize(t)
            total += len(toks)
            unk += sum(1 for w in toks if w not in self.stoi)
        return unk / max(total, 1)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"itos": self.itos, "min_freq": self.min_freq,
                         "max_size": self.max_size}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        v = cls(min_freq=d["min_freq"], max_size=d["max_size"])
        v.itos = d["itos"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v


# =============================================================================
# 3. Dataset + DataLoader
# =============================================================================
class ToxicDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocab, max_len: int = 128):
        self.ids = [torch.tensor(vocab.encode(t, max_len), dtype=torch.long) for t in texts]
        if labels is None:
            self.labels = torch.zeros(len(self.ids), dtype=torch.float)
        else:
            self.labels = torch.tensor(np.asarray(labels), dtype=torch.float)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    return padded, lengths, torch.stack(labels)


def make_loader(df: pd.DataFrame, vocab: Vocab, text_col: str, label_col: str = None,
                max_len: int = 128, batch_size: int = 32, shuffle: bool = False,
                seed: int = 42) -> DataLoader:
    labels = df[label_col].values if (label_col and label_col in df.columns) else None
    ds = ToxicDataset(df[text_col].tolist(), labels, vocab, max_len)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, generator=g)


# =============================================================================
# 4. Word Embeddings
# =============================================================================
def _random_matrix(vocab_size, dim, seed):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.1, size=(vocab_size, dim)).astype(np.float32)
    m[PAD_IDX] = 0.0
    return m


def load_pretrained_vectors(path: str, vocab: Vocab, seed: int = 42):
    """Đọc file vector dạng text word2vec/fastText (.txt/.vec), ví dụ PhoW2V word-level.
    Tự nhận dim từ file. Trả về (matrix, dim, số từ tìm thấy)."""
    vectors, dim = {}, None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if i == 0 and len(parts) == 2:          # dòng header "n_words dim"
                continue
            word, vals = parts[0], parts[1:]
            if dim is None:
                dim = len(vals)
            if len(vals) != dim:
                continue
            w = word.lower()
            if w in vocab.stoi and w not in vectors:
                vectors[w] = np.asarray(vals, dtype=np.float32)
    if dim is None:
        raise ValueError(f"Không đọc được vector nào từ {path}")
    m = _random_matrix(len(vocab), dim, seed)
    for w, vec in vectors.items():
        m[vocab.stoi[w]] = vec
    return m, dim, len(vectors)


def build_embedding_matrix(vocab: Vocab, pretrained_path: str, seed: int = 42):
    """Tạo ma trận embedding từ PhoW2V word-level.
    Trả về (matrix np.float32, dim, mô tả nguồn)."""
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Không tìm thấy file PhoW2V: {pretrained_path}\n"
            "Tải word2vec_vi_words_100dims.zip tại https://github.com/datquocnguyen/PhoW2V "
            "và giải nén file .txt vào thư mục embeddings/.")
    m, d, hits = load_pretrained_vectors(pretrained_path, vocab, seed)
    return m, d, f"pretrained ({os.path.basename(pretrained_path)}): {hits}/{len(vocab)} từ có vector ({hits/len(vocab):.1%})"


# =============================================================================
# 5. Model: BiLSTM 2 chiều + Dropout
# =============================================================================
class BiLSTMClassifier(nn.Module):
    """Embedding -> Dropout -> BiLSTM -> [mean-pool ; max-pool] -> Dropout -> Linear(1).
    Xuất ra 1 logit (chưa sigmoid) để dùng với BCEWithLogitsLoss."""

    def __init__(self, vocab_size: int, embed_dim: int = 300, hidden_dim: int = 128,
                 num_layers: int = 1, dropout: float = 0.5,
                 embedding_matrix: np.ndarray = None, freeze_embedding: bool = False):
        super().__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float),
                freeze=freeze_embedding, padding_idx=PAD_IDX)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        self.emb_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,                                  # 2 chiều
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 * 2, 1)               # (2 chiều) x (mean + max)

    def forward(self, ids, lengths):
        emb = self.emb_dropout(self.embedding(ids))              # (B, T, E)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=ids.size(1))  # (B, T, 2H)

        mask = (ids != PAD_IDX).unsqueeze(-1).float()            # (B, T, 1)
        mean_pool = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        max_pool = out.masked_fill(mask == 0, -1e9).max(1).values
        feat = torch.cat([mean_pool, max_pool], dim=1)           # (B, 4H)
        return self.fc(self.dropout(feat)).squeeze(-1)           # (B,)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 6. Train / Evaluate
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, clip: float = 1.0):
    model.train()
    total, n = 0.0, 0
    for ids, lengths, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total += loss.item() * labels.size(0)
        n += labels.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict_proba(model, loader, device, criterion=None):
    """Trả về (probs, labels, avg_loss). avg_loss=None nếu không truyền criterion."""
    model.eval()
    probs, labels_all = [], []
    total, n = 0.0, 0
    for ids, lengths, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids, lengths)
        if criterion is not None:
            total += criterion(logits, labels).item() * labels.size(0)
            n += labels.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    avg_loss = total / n if criterion is not None and n > 0 else None
    return np.concatenate(probs), np.concatenate(labels_all).astype(int), avg_loss


def compute_metrics(y_true, probs, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(probs) >= threshold).astype(int)
    has_both = len(np.unique(y_true)) == 2
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "f1_toxic": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_toxic": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_toxic": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probs)) if has_both else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, probs)) if has_both else float("nan"),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def find_best_threshold(y_true, probs, grid=None):
    """Tìm ngưỡng tối ưu F1 lớp toxic trên tập validation."""
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (probs >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


# =============================================================================
# 7. Learning curve
# =============================================================================
def plot_learning_curve(history: pd.DataFrame, save_path: str, best_epoch: int = None):
    import matplotlib
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ep = history["epoch"]

    axes[0].plot(ep, history["train_loss"], "o-", label="Train loss")
    axes[0].plot(ep, history["val_loss"], "s-", label="Val loss")
    axes[0].set_title("Loss (BCEWithLogits, pos_weight)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

    axes[1].plot(ep, history["val_f1_toxic"], "o-", label="Val F1 (toxic, ngưỡng 0.5)")
    axes[1].plot(ep, history["val_pr_auc"], "s-", label="Val PR-AUC")
    if "train_f1_toxic" in history:
        axes[1].plot(ep, history["train_f1_toxic"], "^--", alpha=0.6, label="Train F1 (toxic)")
    axes[1].set_title("F1 / PR-AUC lớp toxic")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1.02)

    for ax in axes:
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", linestyle=":", label=f"Best epoch = {best_epoch}")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("BiLSTM – Learning curve")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# =============================================================================
# 8. Checkpoint
# =============================================================================
def save_checkpoint(path: str, model, model_config: dict, extra: dict = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {"model_state": model.state_dict(), "model_config": model_config}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, device="cpu"):
    """Nạp checkpoint -> (model ở chế độ eval, dict checkpoint)."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:                      # torch < 1.13 không có tham số weights_only
        ckpt = torch.load(path, map_location=device)
    cfg = ckpt["model_config"]
    model = BiLSTMClassifier(
        vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt
