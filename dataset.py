"""
dataset.py
Dataset classes cho 3 loại dữ liệu: Vietnamese, Vietlish, IEV
Dựa trên Section 4.1 của paper.
"""

import os
import json
import random
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

from phoneme_set import VOCAB, SPECIAL_TOKENS, word_to_phonemes, VIETLISH_MAP, CORPUS


# ── Cấu hình audio ───────────────────────────────────────────────────────────

SAMPLE_RATE   = 16_000       # Hz — theo paper
N_MELS        = 80           # số kênh mel — theo paper
HOP_LENGTH    = 320          # 20ms frame shift
WIN_LENGTH    = 400          # 25ms frame length
N_FFT         = 512

# ── Giới hạn độ dài để tránh CUDA OOM ───────────────────────────────────────
MAX_LABEL_LEN = 256   # số token âm vị tối đa (bao gồm <sot> và <eot>)
MAX_MEL_FRAMES = 800  # số frame mel tối đa (~16s với hop=20ms)

MEL_TRANSFORM = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)


def audio_to_mel(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Chuyển waveform → log mel-spectrogram (T × 80).

    Args:
        waveform: (channels, samples)
        sample_rate: Hz của file gốc

    Returns:
        mel: (T, N_MELS) — log scale
    """
    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample nếu cần
    if sample_rate != SAMPLE_RATE:
        resampler = T.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)

    mel = MEL_TRANSFORM(waveform)          # (1, N_MELS, T)
    mel = mel.squeeze(0).transpose(0, 1)   # (T, N_MELS)
    log_mel = torch.log(mel + 1e-9)
    return log_mel


def encode_phonemes(phoneme_list: list) -> torch.Tensor:
    """
    Mã hóa danh sách âm vị thành tensor index.
    Tự động thêm <sot> và <eot>.

    Returns:
        (seq_len,) LongTensor
    """
    sot = VOCAB[SPECIAL_TOKENS["SOT"]]
    eot = VOCAB[SPECIAL_TOKENS["EOT"]]
    unk = VOCAB[SPECIAL_TOKENS["UNK"]]

    ids = [sot]
    for p in phoneme_list:
        ids.append(VOCAB.get(p, unk))
    ids.append(eot)
    return torch.tensor(ids, dtype=torch.long)


# ── Dataset cơ sở ────────────────────────────────────────────────────────────

class PhonemeDatasetBase(Dataset):
    """
    Base class cho tất cả các dataset.
    Mỗi item trả về:
        mel:    (T, 80)  — log mel-spectrogram
        labels: (L,)     — chuỗi âm vị được mã hóa
        meta:   dict     — thông tin bổ sung
    """

    def __init__(self, manifest_path: str, audio_root: str, mode: str = "standard"):
        """
        Args:
            manifest_path: File JSON/JSONL, mỗi dòng: {"audio": "path.wav", "phonemes": [...]}
            audio_root:    Thư mục gốc chứa audio
            mode:          'standard' | 'vietlish' | 'iev'
        """
        self.audio_root = Path(audio_root)
        self.mode = mode
        self.items = self._load_manifest(manifest_path)

    def _load_manifest(self, path: str) -> list:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        audio_path = self.audio_root / item["audio"]

        import soundfile as sf
        data, sr = sf.read(str(audio_path), dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
        mel = audio_to_mel(waveform, sr)                  # (T, 80)
        phoneme_list = item.get("phonemes", item.get("phoneme", []))
        labels = encode_phonemes(phoneme_list)            # (L,)

        return {
            "mel":    mel,
            "labels": labels,
            "text":   item.get("text", ""),
            "mode":   self.mode,
            "audio":  item["audio"],
        }


# ── Dataset tiếng Việt (VLSP/VIVOS/CmV/FOSD) ────────────────────────────────

class VietnameseDataset(PhonemeDatasetBase):
    """
    Wrapper cho 4 dataset tiếng Việt thuần.
    Manifest format:
        {"audio": "vlsp/utt001.wav", "text": "xin chào", "phonemes": ["s","i","n","$","tʃ","a","w"]}
    """

    DATASET_INFO = {
        "vlsp2020": {"hours": 48.0,  "samples": 56000},
        "common_voice": {"hours": 30.0, "samples": 12000},
        "vivos":    {"hours": 15.0,  "samples": 11660},
        "fosd":     {"hours": 26.38, "samples": 9073},
    }

    def __init__(self, manifest_path: str, audio_root: str, dataset_name: str = "vivos"):
        super().__init__(manifest_path, audio_root, mode="standard")
        self.dataset_name = dataset_name


# ── Dataset Vietlish ──────────────────────────────────────────────────────────

class VietlishDataset(PhonemeDatasetBase):
    """
    Dataset tiếng Anh được phát âm theo kiểu tiếng Việt.
    Manifest format:
        {"audio": "vietlish/inbox_001.wav", "word": "inbox",
         "viet_syllables": ["in","bóc"], "phonemes": ["ɪ","n","$","b","o","-4","k"]}

    Theo paper: 3,349 training / 3,137 testing — tổng ~0.91h train / 0.81h test
    """

    def __init__(self, manifest_path: str, audio_root: str):
        super().__init__(manifest_path, audio_root, mode="vietlish")

    @staticmethod
    def word_to_vietlish_phonemes(word: str) -> list:
        return word_to_phonemes(word, mode="vietlish")


# ── Dataset IEV (Interleaved English–Vietnamese) ──────────────────────────────

class IEVDataset(PhonemeDatasetBase):
    """
    Dataset câu trộn lẫn tiếng Việt và tiếng Anh.
    Manifest format:
        {"audio": "iev/sent_0001.wav",
         "text": "Anh có mét xịt cho em.",
         "phonemes": [...],
         "language_tags": ["vi","vi","vi+en","vi+en","vi","vi"]}

    Theo paper: 5,678 training / 3,110 testing — tổng ~4.42h train / 2.36h test
    """

    def __init__(self, manifest_path: str, audio_root: str):
        super().__init__(manifest_path, audio_root, mode="iev")

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        item   = self.items[idx]
        sample["language_tags"] = item.get("language_tags", [])
        return sample


# ── Synthetic dataset builder (không cần file audio thật) ────────────────────

class SyntheticVietlishDataset(Dataset):
    """
    Tạo dataset tổng hợp từ VIETLISH_MAP mà không cần audio thật.
    Dùng để kiểm tra pipeline mà không cần tải dataset.

    Mô phỏng quy trình paper: dùng TTS service tạo audio Vietlish.
    Ở đây thay bằng noise ngẫu nhiên để test pipeline.
    """

    def __init__(self, n_samples: int = 200, seq_len: int = 160):
        self.n_samples = n_samples
        self.seq_len   = seq_len
        self.words     = list(CORPUS.keys())

        # Tạo samples
        self.data = []
        for i in range(n_samples):
            word = self.words[i % len(self.words)]
            _, phonemes = CORPUS[word]
            self.data.append({
                "word":     word,
                "phonemes": phonemes,
            })

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        item = self.data[idx]

        # Giả lập mel-spectrogram (trong thực tế: load audio thật)
        T_frames = random.randint(self.seq_len // 2, self.seq_len)
        mel = torch.randn(T_frames, N_MELS) * 0.5   # (T, 80)

        labels = encode_phonemes(item["phonemes"])

        return {
            "mel":    mel,
            "labels": labels,
            "text":   item["word"],
            "mode":   "vietlish",
        }


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    """
    Pad mel và labels về cùng độ dài trong batch.
    Áp dụng cắt ngắn (truncation) theo MAX_MEL_FRAMES và MAX_LABEL_LEN
    để tránh CUDA OOM do cross-attention trên chuỗi quá dài.

    Returns dict:
        mel_padded:    (B, T_max, 80)
        mel_lengths:   (B,)
        label_padded:  (B, L_max)
        label_lengths: (B,)
    """
    pad_mel   = VOCAB[SPECIAL_TOKENS["PAD"]]   # 0
    pad_label = VOCAB[SPECIAL_TOKENS["PAD"]]

    mels   = [item["mel"]    for item in batch]
    labels = [item["labels"] for item in batch]

    # ── Truncate để tránh OOM ────────────────────────────────────────────────
    mels   = [m[:MAX_MEL_FRAMES]   for m in mels]
    # Giữ <sot> + nội dung + <eot>: cắt giữa, giữ lại eot ở cuối
    def _truncate_label(l: torch.Tensor) -> torch.Tensor:
        if l.shape[0] <= MAX_LABEL_LEN:
            return l
        return torch.cat([l[:MAX_LABEL_LEN - 1], l[-1:]])   # giữ <eot>
    labels = [_truncate_label(l) for l in labels]
    # ────────────────────────────────────────────────────────────────────────

    mel_lengths   = torch.tensor([m.shape[0] for m in mels],   dtype=torch.long)
    label_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)

    T_max = mel_lengths.max().item()
    L_max = label_lengths.max().item()
    B     = len(batch)

    mel_padded   = torch.zeros(B, T_max, N_MELS)
    label_padded = torch.full((B, L_max), pad_label, dtype=torch.long)

    for i, (m, l) in enumerate(zip(mels, labels)):
        mel_padded[i, :m.shape[0], :]  = m
        label_padded[i, :l.shape[0]]   = l

    return {
        "mel":           mel_padded,
        "mel_lengths":   mel_lengths,
        "labels":        label_padded,
        "label_lengths": label_lengths,
        "texts":         [item.get("text", "") for item in batch],
        "modes":         [item.get("mode", "")  for item in batch],
    }


if __name__ == "__main__":
    print("=== Synthetic Dataset Demo ===")
    ds = SyntheticVietlishDataset(n_samples=32)
    print(f"Dataset size: {len(ds)}")

    sample = ds[0]
    print(f"Word:     {sample['text']}")
    print(f"Mel:      {sample['mel'].shape}")
    print(f"Labels:   {sample['labels'].tolist()}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch  = next(iter(loader))
    print(f"\nBatch mel:    {batch['mel'].shape}")
    print(f"Batch labels: {batch['labels'].shape}")
    print(f"Texts:        {batch['texts']}")
