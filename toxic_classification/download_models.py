"""
Tải PhoBERT thật từ Hugging Face về máy, rồi kiểm tra mô hình chạy được.

Cách dùng:
    python download_models.py                     # tải PhoBERT (vinai/phobert-base-v2)
    python download_models.py --check_only        # chỉ kiểm tra mô hình đã tải

Mô hình được lưu vào thư mục `models/phobert-base-v2/`.
Sau khi tải xong, notebook tự dùng bản trong `models/` (không cần mạng nữa).
"""
import argparse
import os
import sys
import time

REPO_ID = "vinai/phobert-base-v2"
LOCAL_DIR = "models/phobert-base-v2"
# chỉ tải trọng số PyTorch + tokenizer, bỏ bản TensorFlow/Flax cho nhẹ
ALLOW = ["*.json", "*.txt", "bpe.codes", "*.bin", "*.safetensors"]


def download(repo_id, local_dir):
    from huggingface_hub import snapshot_download
    print(f"\n>>> Đang tải {repo_id} về {local_dir} ...")
    t0 = time.time()
    snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=ALLOW)
    files = sorted(os.listdir(local_dir))
    size = sum(os.path.getsize(os.path.join(local_dir, f)) for f in files if os.path.isfile(os.path.join(local_dir, f)))
    print(f"    xong sau {time.time() - t0:.0f} giây, {size / 1e6:.0f} MB: {', '.join(f for f in files if not f.startswith('.'))}")


def load(fn, path, **kw):
    try:
        return fn(path, **kw)
    except Exception:  # noqa: BLE001  một số phiên bản transformers cần trust_remote_code
        return fn(path, trust_remote_code=True, **kw)


def check(local_dir):
    import torch
    from pyvi import ViTokenizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f"\n>>> Kiểm tra PhoBERT trong {local_dir}")
    if not os.path.isdir(local_dir):
        print("    CHƯA CÓ mô hình. Hãy chạy lại script không kèm --check_only.")
        return False
    tok = load(AutoTokenizer.from_pretrained, local_dir)
    model = load(AutoModelForSequenceClassification.from_pretrained, local_dir, num_labels=2)
    # PhoBERT bắt buộc phải tách từ trước
    text = ViTokenizer.tokenize("Bọn bán hàng lừa đảo, bảo hành thì hẹn lên hẹn xuống")
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    câu thử    : {text}")
    print(f"    token      : {tok.convert_ids_to_tokens(enc['input_ids'][0])[:16]}")
    print(f"    vocab      : {len(tok)} | tham số: {n_params:.0f} triệu | đầu ra: {tuple(logits.shape)}")
    print("    OK – mô hình chạy được. (Lớp phân loại chưa train nên đầu ra lúc này chưa có ý nghĩa.)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check_only", action="store_true")
    args = ap.parse_args()

    if not args.check_only:
        try:
            download(REPO_ID, LOCAL_DIR)
        except Exception as e:  # noqa: BLE001
            print(f"    LỖI khi tải {REPO_ID}: {type(e).__name__}: {e}")
            print("    Xem mục 'Lỗi thường gặp' trong HUONG_DAN_TAI_MO_HINH.md")
            sys.exit(1)
    sys.exit(0 if check(LOCAL_DIR) else 1)


if __name__ == "__main__":
    main()
