"""
model.py
PhoWhisper encoder → phoneme sequence.

Two model variants are provided:

1.  PhoWhisperCTCModel  (original)
        mel → [WhisperEncoder] → Linear → CTC greedy decode → phonemes

2.  PhoWhisperSeq2SeqModel  (new, stronger)
        mel → [WhisperEncoder (frozen)] → Transformer Decoder → phonemes
        Decoder: d_model=512, nhead=8, 6 layers, FFN=2048,
                 learned positional embeddings, tied input/output embeddings.

Yêu cầu:
    pip install torch torchaudio transformers soundfile

Ghi chú:
    - PhoWhisper là Whisper fine-tune trên tiếng Việt (vinai/PhoWhisper-base).
    - Encoder luôn được freeze; chỉ train decoder.
"""

import os
import argparse
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

from phoneme_set import (
    VOCAB, INV_VOCAB, VOCAB_SIZE,
    PAD_IDX, UNK_IDX, SOT_IDX, EOT_IDX,
    SPECIAL_TOKENS,
    decode_sequence,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CẤU HÌNH
# ═══════════════════════════════════════════════════════════════════════════════

PHOWHISPER_MODEL_ID = "vinai/PhoWhisper-base"

SAMPLE_RATE  = 16_000
N_MELS       = 80
HOP_LENGTH   = 160
WIN_LENGTH   = 400
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

    Returns:
        mel: Tensor (1, 3000, N_MELS) — ready to feed model.
    """
    wav_path = str(wav_path)

    try:
        waveform, sr = torchaudio.load(wav_path)
    except Exception:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(wav_path, dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform  = resampler(waveform)

    mel     = _MEL_TRANSFORM(waveform)
    log_mel = torch.log(mel + 1e-9)
    log_mel = log_mel.squeeze(0).transpose(0, 1)   # (T, N_MELS)

    MAX_FRAMES = 3000
    n_frames   = log_mel.shape[0]
    if n_frames < MAX_FRAMES:
        pad     = torch.zeros(MAX_FRAMES - n_frames, N_MELS)
        log_mel = torch.cat([log_mel, pad], dim=0)
    else:
        log_mel = log_mel[:MAX_FRAMES]

    return log_mel.unsqueeze(0)   # (1, 3000, N_MELS)


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD PHOWHISPER ENCODER  (shared helper)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_phowhisper_encoder(model_id: str):
    """Load frozen PhoWhisper encoder from HuggingFace."""
    try:
        from transformers import WhisperModel
    except ImportError as e:
        raise ImportError(
            "Thiếu thư viện transformers. Chạy: pip install transformers"
        ) from e
    model = WhisperModel.from_pretrained(model_id)
    return model.encoder


# ═══════════════════════════════════════════════════════════════════════════════
#  (1)  CTC MODEL  —  original, kept for backward compatibility
# ═══════════════════════════════════════════════════════════════════════════════

class PhoWhisperCTCModel(nn.Module):
    """
    PhoWhisper encoder (frozen) + 1 Linear CTC head.

    Kiến trúc:
        mel (B, T, 80) → WhisperEncoder → Linear(d_model, VOCAB_SIZE) → CTC
    """

    def __init__(
        self,
        model_id:       str  = PHOWHISPER_MODEL_ID,
        freeze_encoder: bool = True,
        device:         str  = "cpu",
    ):
        super().__init__()
        self.device_str = device

        print(f"[PhoWhisperCTCModel] Loading encoder from '{model_id}' ...")
        self.encoder = _load_phowhisper_encoder(model_id)
        self.d_model = self.encoder.config.d_model

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("[PhoWhisperCTCModel] Encoder frozen.")

        self.ctc_head = nn.Linear(self.d_model, VOCAB_SIZE)
        self.to(device)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        input_features = mel.transpose(1, 2)
        encoder_out    = self.encoder(input_features=input_features)
        hidden         = encoder_out.last_hidden_state
        return self.ctc_head(hidden)

    @torch.no_grad()
    def wav_to_indices(self, wav_path: str | Path) -> list[int]:
        self.eval()
        mel    = load_wav_to_mel(wav_path).to(self.device_str)
        logits = self.forward(mel)
        return ctc_greedy_decode(logits[0])

    @torch.no_grad()
    def wav_to_phonemes(self, wav_path: str | Path) -> list[str]:
        return decode_sequence(self.wav_to_indices(wav_path), skip_special=True)


def ctc_greedy_decode(logits: torch.Tensor, blank_id: int = PAD_IDX) -> list[int]:
    """CTC greedy decode: argmax → collapse repeats → remove blank."""
    raw       = logits.argmax(dim=-1).tolist()
    collapsed = [raw[0]]
    for tok in raw[1:]:
        if tok != collapsed[-1]:
            collapsed.append(tok)
    return [tok for tok in collapsed if tok != blank_id]


# CTC checkpoint helpers
def save_checkpoint(model: PhoWhisperCTCModel, path: str, epoch: int = 0, loss: float = 0.0):
    """Lưu checkpoint CTC head."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "ctc_head_state": model.ctc_head.state_dict(),
        "vocab_size":     VOCAB_SIZE,
        "d_model":        model.d_model,
        "epoch":          epoch,
        "loss":           loss,
    }, path)
    print(f"[checkpoint] CTC saved → {path}")


def load_checkpoint(model: PhoWhisperCTCModel, path: str):
    """Khôi phục CTC head từ checkpoint."""
    ckpt = torch.load(path, map_location=model.device_str, weights_only=False)
    model.ctc_head.load_state_dict(ckpt["ctc_head_state"])
    print(f"[checkpoint] CTC loaded ← {path}  "
          f"(epoch={ckpt.get('epoch', 0)}, loss={ckpt.get('loss', 0):.4f})")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  (2)  PHONEME TRANSFORMER DECODER
# ═══════════════════════════════════════════════════════════════════════════════

class PhonemeTransformerDecoder(nn.Module):
    """
    Strong autoregressive Transformer Decoder for phoneme generation.

    Features:
        - Learned (not sinusoidal) positional embeddings
        - Tied input / output embeddings  →  output_proj.weight = token_emb.weight
        - Pre-LayerNorm (norm_first=True) for stable deep stacks
        - Proper causal self-attention mask + target padding mask
        - Cross-attention over encoder memory (with memory padding mask)

    Default config (matches user spec):
        d_model=512, nhead=8, num_layers=6, dim_feedforward=2048
    """

    def __init__(
        self,
        vocab_size:      int   = VOCAB_SIZE,
        d_model:         int   = 512,
        nhead:           int   = 8,
        num_layers:      int   = 6,
        dim_feedforward: int   = 2048,
        dropout:         float = 0.1,
        max_len:         int   = 2048,
    ):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size

        # ── Embeddings ────────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_emb   = nn.Embedding(max_len, d_model)    # learned positional
        self.emb_drop  = nn.Dropout(dropout)

        # ── Transformer Decoder layers ────────────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,             # Pre-LN: more stable for deep stacks
        )
        self.transformer = nn.TransformerDecoder(
            dec_layer,
            num_layers = num_layers,
            norm       = nn.LayerNorm(d_model),
        )

        # ── Output projection — weight-tied to token embedding ─────────────────
        self.output_proj        = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_emb.weight   # tie weights

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   mean=0.0, std=0.02)
        # output_proj.weight is tied → already initialised above

    # ── Mask helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (True = attend-to is forbidden). Shape: (sz, sz)."""
        return torch.triu(
            torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1
        )

    @staticmethod
    def _pad_mask(ids: torch.Tensor) -> torch.Tensor:
        """True where token == PAD_IDX (position should be ignored). Shape: (B, L)."""
        return ids.eq(PAD_IDX)

    # ── Forward (teacher-forcing) ─────────────────────────────────────────────

    def forward(
        self,
        tgt_ids: torch.Tensor,                             # (B, L)  int64
        memory:  torch.Tensor,                             # (B, T', d_model)
        memory_key_padding_mask: torch.Tensor | None = None,   # (B, T')
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        Args:
            tgt_ids:  Decoder input IDs, SOT-prefixed, PAD-padded.  (B, L)
            memory:   Encoder output.                                (B, T', d_model)
            memory_key_padding_mask: True where encoder frame is padding. (B, T')

        Returns:
            logits: (B, L, vocab_size)
        """
        B, L   = tgt_ids.shape
        device = tgt_ids.device

        # Token + positional embeddings
        pos = torch.arange(L, device=device).unsqueeze(0)             # (1, L)
        x   = self.emb_drop(self.token_emb(tgt_ids) + self.pos_emb(pos))  # (B, L, d)

        # Masks
        causal_mask  = self._causal_mask(L, device)   # (L, L)
        tgt_pad_mask = self._pad_mask(tgt_ids)         # (B, L)

        # Transformer decoder
        out = self.transformer(
            tgt                     = x,
            memory                  = memory,
            tgt_mask                = causal_mask,
            tgt_key_padding_mask    = tgt_pad_mask,
            memory_key_padding_mask = memory_key_padding_mask,
        )  # (B, L, d_model)

        return self.output_proj(out)   # (B, L, vocab_size)

    # ── Autoregressive inference ───────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        memory:  torch.Tensor,   # (1, T', d_model)  — batch_size MUST be 1
        max_len: int = 200,
    ) -> list[int]:
        """
        Greedy autoregressive decode for a single sample.

        Args:
            memory:  Encoder output.  Batch dimension must be 1.
            max_len: Maximum tokens to generate before stopping.

        Returns:
            Token indices, with leading SOT and trailing EOT stripped.
        """
        device = memory.device
        ids    = torch.tensor([[SOT_IDX]], dtype=torch.long, device=device)  # (1,1)

        for _ in range(max_len):
            logits  = self.forward(ids, memory)          # (1, cur_len, V)
            next_id = logits[0, -1].argmax(dim=-1).item()

            if next_id == EOT_IDX:
                break

            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )

        return ids[0, 1:].tolist()   # strip leading SOT


# ═══════════════════════════════════════════════════════════════════════════════
#  (2)  PHOWHISPER SEQ2SEQ MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class PhoWhisperSeq2SeqModel(nn.Module):
    """
    PhoWhisper Encoder (frozen) + PhonemeTransformerDecoder (trainable).

    Architecture:
        mel (B, T, 80)
            ↓  WhisperEncoder  [frozen, d_enc=512]
        memory (B, T', 512)
            ↓  enc_proj  Linear(enc_dim → d_model)  — nn.Identity when dims match
        (B, T', d_model)
            ↕  cross-attention inside PhonemeTransformerDecoder
        logits (B, L, VOCAB_SIZE)

    Training (handled in train.py):
        - Teacher forcing with scheduled-sampling decay  (1.0 → 0.75)
        - Loss: CrossEntropyLoss(label_smoothing=0.1, ignore_index=PAD_IDX)
        - Optimizer: AdamW(lr=3e-4, betas=(0.9,0.98), warmup ~4k steps)
        - Grad clipping at 1.0

    Inference:
        model.wav_to_phonemes(wav_path) → list[str]   (greedy autoregressive)
    """

    def __init__(
        self,
        model_id:        str   = PHOWHISPER_MODEL_ID,
        freeze_encoder:  bool  = True,
        device:          str   = "cpu",
        # Decoder config
        d_model:         int   = 512,
        nhead:           int   = 8,
        num_layers:      int   = 6,
        dim_feedforward: int   = 2048,
        dropout:         float = 0.1,
        max_dec_len:     int   = 2048,
    ):
        super().__init__()
        self.device_str      = device
        self._encoder_frozen = freeze_encoder

        # ── Encoder ──────────────────────────────────────────────────────────
        print(f"[PhoWhisperSeq2SeqModel] Loading encoder from '{model_id}' ...")
        self.encoder = _load_phowhisper_encoder(model_id)
        enc_dim      = self.encoder.config.d_model   # 512 for PhoWhisper-base

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("[PhoWhisperSeq2SeqModel] Encoder frozen.")

        # ── Encoder → Decoder dimension bridge ──────────────────────────────
        # nn.Identity when enc_dim == d_model (default: both 512)
        self.enc_proj: nn.Module = (
            nn.Linear(enc_dim, d_model) if enc_dim != d_model else nn.Identity()
        )

        # ── Transformer Decoder ───────────────────────────────────────────────
        self.decoder = PhonemeTransformerDecoder(
            vocab_size      = VOCAB_SIZE,
            d_model         = d_model,
            nhead           = nhead,
            num_layers      = num_layers,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            max_len         = max_dec_len,
        )

        self.to(device)

    # ── Encode ───────────────────────────────────────────────────────────────

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Run frozen encoder on mel spectrogram.

        Encoder is wrapped in torch.no_grad() to save activation memory.
        (Remove the context manager if you want to fine-tune the encoder.)

        Args:
            mel: (B, T, N_MELS)
        Returns:
            memory: (B, T', d_model)
        """
        input_features = mel.transpose(1, 2)    # (B, N_MELS, T)
        with torch.no_grad():
            hidden = self.encoder(input_features=input_features).last_hidden_state
        return self.enc_proj(hidden)             # (B, T', d_model)

    # ── Forward  (teacher-forcing) ────────────────────────────────────────────

    def forward(
        self,
        mel:     torch.Tensor,   # (B, T, N_MELS)
        tgt_ids: torch.Tensor,   # (B, L)  SOT-prefixed, PAD-padded
    ) -> torch.Tensor:
        """
        Teacher-forcing forward.

        Returns:
            logits: (B, L, VOCAB_SIZE)
        """
        memory = self.encode(mel)              # (B, T', d_model)
        return self.decoder(tgt_ids, memory)   # (B, L, VOCAB_SIZE)

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def wav_to_indices(self, wav_path: str | Path) -> list[int]:
        """Load .wav → autoregressive greedy decode → list of token indices."""
        mel    = load_wav_to_mel(wav_path).to(self.device_str)
        memory = self.encode(mel)
        return self.decoder.generate(memory)

    @torch.no_grad()
    def wav_to_phonemes(self, wav_path: str | Path) -> list[str]:
        """Load .wav → list of IPA phoneme strings."""
        return decode_sequence(self.wav_to_indices(wav_path), skip_special=True)


# ── Seq2Seq checkpoint helpers ────────────────────────────────────────────────

def save_seq2seq_checkpoint(
    model:       PhoWhisperSeq2SeqModel,
    path:        str,
    epoch:       int   = 0,
    per:         float = 1.0,
    global_step: int   = 0,
):
    """
    Save trainable parts of PhoWhisperSeq2SeqModel.
    Only decoder + enc_proj are saved (encoder is frozen and not modified).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    enc_proj_sd = (
        None if isinstance(model.enc_proj, nn.Identity)
        else model.enc_proj.state_dict()
    )
    torch.save({
        "decoder_state":   model.decoder.state_dict(),
        "enc_proj_state":  enc_proj_sd,
        "vocab_size":      VOCAB_SIZE,
        "d_model":         model.decoder.d_model,
        "epoch":           epoch,
        "per":             per,
        "global_step":     global_step,
    }, path)
    print(f"[checkpoint] Seq2Seq saved → {path}")


def load_seq2seq_checkpoint(model: PhoWhisperSeq2SeqModel, path: str):
    """Restore decoder (and optionally enc_proj) weights from a checkpoint."""
    ckpt = torch.load(path, map_location=model.device_str, weights_only=False)
    model.decoder.load_state_dict(ckpt["decoder_state"])
    if ckpt.get("enc_proj_state") is not None and not isinstance(model.enc_proj, nn.Identity):
        model.enc_proj.load_state_dict(ckpt["enc_proj_state"])
    print(
        f"[checkpoint] Seq2Seq loaded ← {path}  "
        f"(epoch={ckpt.get('epoch', 0)}, PER={ckpt.get('per', 1.0) * 100:.2f}%)"
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI  — python model.py --wav path/to/file.wav  [--model ctc|seq2seq]
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PhoWhisper → phoneme sequence"
    )
    parser.add_argument("--wav",        type=str, required=True,
                        help="Đường dẫn file .wav")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint .pt để load")
    parser.add_argument("--model",      type=str, default="seq2seq",
                        choices=["ctc", "seq2seq"],
                        help="Model variant: ctc (legacy) hoặc seq2seq (default)")
    parser.add_argument("--model_id",   type=str, default=PHOWHISPER_MODEL_ID)
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not Path(args.wav).exists():
        print(f"[ERROR] Không tìm thấy file: {args.wav}")
        return

    if args.model == "ctc":
        model = PhoWhisperCTCModel(model_id=args.model_id, device=args.device)
        if args.checkpoint and Path(args.checkpoint).exists():
            load_checkpoint(model, args.checkpoint)
        phonemes = model.wav_to_phonemes(args.wav)
    else:
        model = PhoWhisperSeq2SeqModel(model_id=args.model_id, device=args.device)
        if args.checkpoint and Path(args.checkpoint).exists():
            load_seq2seq_checkpoint(model, args.checkpoint)
        model.eval()
        phonemes = model.wav_to_phonemes(args.wav)

    print(f"\n{'='*55}")
    print(f"  File    : {args.wav}")
    print(f"  Model   : {args.model}")
    print(f"  Phonemes: {phonemes}")
    print(f"  Length  : {len(phonemes)} phonemes")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
