"""
model.py
Dùng PhoWhisper để encode file .wav → chuỗi index âm vị.

Pipeline:
    .wav file
        ↓  torchaudio / soundfile
    waveform (float32, 16 kHz)
        ↓  log mel-spectrogram (80 mel bins)
    mel (T × 80)
        ↓  PhoWhisper encoder  (transformers.WhisperModel hoặc bản custom)
    encoder_hidden_states (T' × d_model)
        ↓  Linear projection → VOCAB_SIZE
        ↓  CTC greedy decode  (argmax + collapse blanks)
    indices  [int, ...]
        ↓  INV_VOCAB lookup
    phonemes [str, ...]

Yêu cầu:
    pip install torch torchaudio transformers soundfile

Ghi chú:
    - File này dùng PhoWhisper THẬT từ HuggingFace (vinai/PhoWhisper-base).
    - Nếu muốn dùng kiến trúc custom (WhisperTransformerPhoneme trong model.py gốc),
      hãy thay hàm _load_encoder() tương ứng.
    - CTC head được khởi tạo ngẫu nhiên — cần fine-tune để cho kết quả tốt.
      Để chạy thử pipeline không cần train, kết quả phoneme sẽ là ngẫu nhiên.
"""

import os
import argparse
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

# Phoneme vocabulary từ phoneme_set.py trong cùng project
from phoneme_set import (
    VOCAB, INV_VOCAB, VOCAB_SIZE,
    PAD_IDX, UNK_IDX, SOT_IDX, EOT_IDX,
    SPECIAL_TOKENS,
    decode_sequence,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CẤU HÌNH
# ═══════════════════════════════════════════════════════════════════════════════

PHOWHISPER_MODEL_ID = "vinai/PhoWhisper-base"   # HuggingFace model hub ID

SAMPLE_RATE  = 16_000   # Hz — PhoWhisper yêu cầu 16 kHz
N_MELS       = 80       # mel bins — theo Whisper gốc
HOP_LENGTH   = 160      # 10ms
WIN_LENGTH   = 400      # 25ms
N_FFT        = 512

_MEL_TRANSFORM = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD AUDIO → MEL
# ═══════════════════════════════════════════════════════════════════════════════

def load_wav_to_mel(wav_path: str | Path) -> torch.Tensor:
    """
    Đọc file .wav → log mel-spectrogram.

    Args:
        wav_path: Đường dẫn đến file .wav (hoặc .flac, .mp3 nếu torchaudio hỗ trợ).

    Returns:
        mel: Tensor (1, T, N_MELS) — batch_size=1, ready to feed model.
    """
    wav_path = str(wav_path)

    # Dùng torchaudio; fallback sang soundfile nếu cần
    try:
        waveform, sr = torchaudio.load(wav_path)   # (channels, samples)
    except Exception:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(wav_path, dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample nếu cần
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform  = resampler(waveform)

    # Log mel: (1, N_MELS, T) → (1, T, N_MELS)
    mel = _MEL_TRANSFORM(waveform)               # (1, N_MELS, T)
    log_mel = torch.log(mel + 1e-9)
    log_mel = log_mel.squeeze(0).transpose(0, 1) # (T, N_MELS)
    return log_mel.unsqueeze(0)                   # (1, T, N_MELS)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHOWHISPER ENCODER + CTC HEAD
# ═══════════════════════════════════════════════════════════════════════════════

class PhoWhisperCTCModel(nn.Module):
    """
    PhoWhisper encoder (frozen) + 1 Linear CTC head.

    Kiến trúc:
        mel (B, T, 80)
            ↓  WhisperEncoder (từ HuggingFace)
        hidden (B, T', d_model=512)   ← T' = T sau 2x stride trong Conv layers
            ↓  Linear(d_model, VOCAB_SIZE)
        logits (B, T', VOCAB_SIZE)
            ↓  CTC greedy decode
        index sequence  [int, ...]

    Ghi chú về CTC:
        - Index 0 (PAD_IDX) được dùng làm CTC blank token.
        - Sau argmax, cần collapse các frame liên tiếp giống nhau,
          sau đó loại bỏ blank để lấy chuỗi phoneme cuối cùng.
    """

    def __init__(
        self,
        model_id:      str  = PHOWHISPER_MODEL_ID,
        freeze_encoder: bool = True,
        device:        str  = "cpu",
    ):
        super().__init__()
        self.device_str = device

        # Load PhoWhisper encoder từ HuggingFace
        print(f"[PhoWhisperCTCModel] Loading encoder from '{model_id}' ...")
        self.encoder = _load_phowhisper_encoder(model_id)
        self.d_model = self.encoder.config.d_model   # thường là 512 (base)

        # Freeze encoder weights (theo paper)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("[PhoWhisperCTCModel] Encoder frozen.")

        # CTC projection head
        self.ctc_head = nn.Linear(self.d_model, VOCAB_SIZE)

        self.to(device)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, T, N_MELS)  — log mel-spectrogram

        Returns:
            logits: (B, T', VOCAB_SIZE)  — CTC logits
        """
        # WhisperEncoder expects input_features: (B, N_MELS, T)
        input_features = mel.transpose(1, 2)   # (B, 80, T)

        encoder_out = self.encoder(input_features=input_features)
        hidden = encoder_out.last_hidden_state   # (B, T', d_model)

        logits = self.ctc_head(hidden)           # (B, T', VOCAB_SIZE)
        return logits

    # ── Inference ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def wav_to_indices(self, wav_path: str | Path) -> list[int]:
        """
        Đọc file .wav và trả về chuỗi index âm vị.

        Args:
            wav_path: Đường dẫn file .wav.

        Returns:
            indices: Danh sách int, mỗi phần tử là index trong VOCAB.
        """
        self.eval()
        mel = load_wav_to_mel(wav_path).to(self.device_str)   # (1, T, 80)

        logits = self.forward(mel)                            # (1, T', V)
        indices = ctc_greedy_decode(logits[0])                # list[int]
        return indices

    @torch.no_grad()
    def wav_to_phonemes(self, wav_path: str | Path) -> list[str]:
        """
        Đọc file .wav và trả về chuỗi phoneme (string).

        Args:
            wav_path: Đường dẫn file .wav.

        Returns:
            phonemes: Danh sách string IPA phoneme.

        Ví dụ:
            ["tʃ", "a", "-2", "$", "b", "a", "-1", "n"]  # "chào bạn"
        """
        indices  = self.wav_to_indices(wav_path)
        phonemes = decode_sequence(indices, skip_special=True)
        return phonemes


# ═══════════════════════════════════════════════════════════════════════════════
#  CTC GREEDY DECODE
# ═══════════════════════════════════════════════════════════════════════════════

def ctc_greedy_decode(logits: torch.Tensor, blank_id: int = PAD_IDX) -> list[int]:
    """
    CTC greedy decode: argmax → collapse repeats → remove blank.

    Args:
        logits:   (T, VOCAB_SIZE) — logit tensor cho 1 sample.
        blank_id: Index của blank token (mặc định = PAD_IDX = 0).

    Returns:
        Danh sách index âm vị sau khi đã loại blank và collapse.
    """
    # Argmax trên mỗi frame
    raw = logits.argmax(dim=-1).tolist()   # list[int], độ dài T

    # Collapse các frame liên tiếp giống nhau
    collapsed = [raw[0]]
    for tok in raw[1:]:
        if tok != collapsed[-1]:
            collapsed.append(tok)

    # Loại bỏ blank
    result = [tok for tok in collapsed if tok != blank_id]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD PHOWHISPER ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

def _load_phowhisper_encoder(model_id: str):
    """
    Load Whisper encoder từ HuggingFace.

    PhoWhisper (vinai/PhoWhisper-base) là Whisper được fine-tune trên tiếng Việt
    → dùng WhisperModel.encoder để lấy phần encoder.

    Tham chiếu: https://huggingface.co/vinai/PhoWhisper-base
    """
    try:
        from transformers import WhisperModel
    except ImportError as e:
        raise ImportError(
            "Thiếu thư viện transformers. Chạy: pip install transformers"
        ) from e

    model = WhisperModel.from_pretrained(model_id)
    return model.encoder


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT  save / load
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model: PhoWhisperCTCModel, path: str, epoch: int = 0, loss: float = 0.0):
    """Lưu checkpoint (chỉ lưu CTC head vì encoder đã frozen)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "ctc_head_state": model.ctc_head.state_dict(),
        "vocab_size":     VOCAB_SIZE,
        "d_model":        model.d_model,
        "epoch":          epoch,
        "loss":           loss,
    }, path)
    print(f"[checkpoint] Saved → {path}")


def load_checkpoint(model: PhoWhisperCTCModel, path: str):
    """Khôi phục CTC head từ checkpoint."""
    ckpt = torch.load(path, map_location=model.device_str, weights_only=False)
    model.ctc_head.load_state_dict(ckpt["ctc_head_state"])
    print(f"[checkpoint] Loaded ← {path}  (epoch={ckpt.get('epoch',0)}, loss={ckpt.get('loss',0):.4f})")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI  — chạy trực tiếp: python model.py --wav path/to/file.wav
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PhoWhisper → phoneme index sequence"
    )
    parser.add_argument("--wav",        type=str, required=True,
                        help="Đường dẫn file .wav cần encode")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="(Tùy chọn) checkpoint file .pt để load CTC head")
    parser.add_argument("--model_id",   type=str, default=PHOWHISPER_MODEL_ID,
                        help=f"HuggingFace model ID (mặc định: {PHOWHISPER_MODEL_ID})")
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not Path(args.wav).exists():
        print(f"[ERROR] Không tìm thấy file: {args.wav}")
        return

    # Khởi tạo model
    model = PhoWhisperCTCModel(
        model_id=args.model_id,
        freeze_encoder=True,
        device=args.device,
    )

    # Load checkpoint CTC head nếu có
    if args.checkpoint and Path(args.checkpoint).exists():
        load_checkpoint(model, args.checkpoint)
    else:
        print("[INFO] Không có checkpoint — CTC head dùng trọng số ngẫu nhiên.")
        print("[INFO] Kết quả phoneme sẽ là ngẫu nhiên cho đến khi fine-tune.\n")

    # Encode
    print(f"[INFO] Đang encode: {args.wav}")
    indices  = model.wav_to_indices(args.wav)
    phonemes = decode_sequence(indices, skip_special=True)

    print(f"\n{'='*55}")
    print(f"  File    : {args.wav}")
    print(f"  Indices : {indices}")
    print(f"  Phonemes: {phonemes}")
    print(f"  Length  : {len(phonemes)} phonemes")
    print(f"{'='*55}")

    # In bảng index → phoneme
    if indices:
        print("\n  Index → Phoneme mapping:")
        print(f"  {'Idx':>6}  Phoneme")
        print(f"  {'-'*20}")
        for idx in indices:
            ph = INV_VOCAB.get(idx, "<unk>")
            print(f"  [{idx:4d}]  {ph}")


if __name__ == "__main__":
    main()
