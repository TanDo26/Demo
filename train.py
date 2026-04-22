"""
train.py
Fine-tune PhoWhisperSeq2SeqModel on a phoneme-labelled audio dataset.

Architecture:
    PhoWhisper Encoder (frozen, d=512)
        ↓  cross-attention
    Transformer Decoder  d_model=512, nhead=8, 6 layers, FFN=2048
        ↓
    CrossEntropyLoss(label_smoothing=0.1)

Training recipe:
    Optimizer  : AdamW  lr=3e-4, betas=(0.9, 0.98), weight_decay=0.01
    Schedule   : Linear warmup (4 000 steps) → constant
    Grad clip  : 1.0
    TF ratio   : linear decay 1.0 → 0.75 over epochs
                 (token-level scheduled sampling)
    Label smth : 0.1
"""

import json
import os
import random
import logging
import threading

# Suppress the harmless HuggingFace Hub background-thread 403 warning.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)

_original_thread_excepthook = threading.excepthook
def _silent_hf_thread_errors(args):
    """Swallow harmless HuggingFace Hub background-thread errors."""
    module = getattr(args.exc_type, "__module__", "") or ""
    if "huggingface_hub" in module or "safetensors_conversion" in module:
        return
    _original_thread_excepthook(args)
threading.excepthook = _silent_hf_thread_errors

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import (
    PhoWhisperSeq2SeqModel,
    load_wav_to_mel,
    save_seq2seq_checkpoint,
    load_seq2seq_checkpoint,
)
from phoneme_set import (
    encode_sequence, decode_sequence,
    PAD_IDX, SOT_IDX, EOT_IDX, VOCAB_SIZE,
)


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def levenshtein(ref: list, hyp: list) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    n, m = len(ref), len(hyp)
    dp   = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if ref[i - 1] == hyp[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev  = temp
    return dp[m]


def compute_per(ref_seqs: list[list[str]], hyp_seqs: list[list[str]]) -> float:
    """Phoneme Error Rate = total edit distance / total reference length."""
    total_edits = sum(levenshtein(r, h) for r, h in zip(ref_seqs, hyp_seqs))
    total_ref   = sum(len(r) for r in ref_seqs)
    return total_edits / max(total_ref, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

def load_manifest(manifest_path: str) -> list[dict]:
    """Read a .jsonl manifest file and return a list of record dicts."""
    entries = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def split_manifest(
    entries: list[dict], train_ratio: float = 0.8, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split entries into (train, eval) sets."""
    entries = entries.copy()
    random.seed(seed)
    random.shuffle(entries)
    cut = int(len(entries) * train_ratio)
    return entries[:cut], entries[cut:]


class AudioDataset(Dataset):
    """Dataset built from a list of manifest dicts."""

    def __init__(self, entries: list[dict], audio_root: str = "dataset"):
        self.data       = entries
        self.audio_root = audio_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item     = self.data[idx]
        wav      = os.path.join(self.audio_root, item["audio"])
        phonemes = item["phoneme"]
        mel      = load_wav_to_mel(wav)                                # (1, 3000, 80)
        target   = torch.tensor(encode_sequence(phonemes), dtype=torch.long)
        return mel.squeeze(0), target, phonemes                        # (3000,80) tensor list


# ══════════════════════════════════════════════════════════════════════════════
#  COLLATE
# ══════════════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    """
    Collate function for seq2seq training.

    Builds:
        tgt_in  = [SOT, t_1, ..., t_N]       (decoder input,  SOT-prefixed)
        tgt_out = [t_1, ..., t_N, EOT]       (decoder target, EOT-appended)

    Sequences padded with PAD_IDX to the length of the longest in the batch.

    Returns:
        mels        (B, 3000, 80)
        tgt_in      (B, L_max)   int64
        tgt_out     (B, L_max)   int64
        phoneme_lists  list[list[str]]
    """
    mels, targets, phoneme_lists = zip(*batch)

    # All mels are exactly (3000, 80) from load_wav_to_mel → simple stack
    mels = torch.stack(mels)   # (B, 3000, 80)

    sot = torch.tensor([SOT_IDX], dtype=torch.long)
    eot = torch.tensor([EOT_IDX], dtype=torch.long)

    tgt_in_list  = [torch.cat([sot, t]) for t in targets]   # [SOT, t1..tN]
    tgt_out_list = [torch.cat([t, eot]) for t in targets]   # [t1..tN, EOT]

    tgt_in  = nn.utils.rnn.pad_sequence(
        tgt_in_list,  batch_first=True, padding_value=PAD_IDX
    )
    tgt_out = nn.utils.rnn.pad_sequence(
        tgt_out_list, batch_first=True, padding_value=PAD_IDX
    )

    return mels, tgt_in, tgt_out, list(phoneme_lists)


# ══════════════════════════════════════════════════════════════════════════════
#  TEACHER FORCING  +  SCHEDULED SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def get_tf_ratio(
    epoch:     int,
    num_epochs: int,
    tf_start:  float = 1.0,
    tf_end:    float = 0.75,
) -> float:
    """Linear decay of the teacher-forcing ratio: tf_start → tf_end."""
    if num_epochs <= 1:
        return tf_start
    return tf_start - (tf_start - tf_end) * epoch / (num_epochs - 1)


def scheduled_sampling_forward(
    model:    PhoWhisperSeq2SeqModel,
    mel:      torch.Tensor,   # (B, 3000, 80)
    tgt_in:   torch.Tensor,   # (B, L)
    tf_ratio: float,
    device:   str,
) -> torch.Tensor:
    """
    Token-level scheduled sampling.

    Step 1 — Pure teacher-forcing forward (always runs; provides reference
              predictions and is the only pass when tf_ratio == 1.0).

    Step 2 — If tf_ratio < 1.0, construct a mixed decoder input:
              For each position i > 0 independently:
                with prob  tf_ratio  → keep ground-truth token  (tgt_in[:, i])
                with prob 1-tf_ratio → use model's prediction   (pred_ids[:, i-1])
              Then run a second forward pass with this mixed input.

    Returns:
        logits: (B, L, VOCAB_SIZE)  — from the (possibly mixed) forward pass.
    """
    # Pass 1 — pure teacher forcing
    logits = model(mel, tgt_in)   # (B, L, V)

    if tf_ratio >= 1.0:
        return logits

    # Build mixed input from pass-1 predictions (no grad needed)
    with torch.no_grad():
        pred_ids = logits.detach().argmax(dim=-1)   # (B, L)

    B, L  = tgt_in.shape

    # At position i (i ≥ 1): ground-truth is tgt_in[:,i],
    # model's candidate is pred_ids[:,i-1]  (prediction for the *next* token).
    keep_gt = torch.bernoulli(
        torch.full((B, L - 1), tf_ratio, device=device)
    ).bool()   # (B, L-1)  True → keep ground truth

    mixed_input = tgt_in.clone()
    mixed_input[:, 1:] = torch.where(keep_gt, tgt_in[:, 1:], pred_ids[:, :-1])

    # Pass 2 — forward with mixed input
    return model(mel, mixed_input)


# ══════════════════════════════════════════════════════════════════════════════
#  LR SCHEDULE  (linear warmup → constant)
# ══════════════════════════════════════════════════════════════════════════════

def warmup_schedule(current_step: int, warmup_steps: int = 4_000) -> float:
    """
    Linear warmup for the first `warmup_steps`, then hold at 1.0.
    Pass as `lr_lambda` to torch.optim.lr_scheduler.LambdaLR.
    """
    if current_step < warmup_steps:
        return current_step / max(warmup_steps, 1)
    return 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATE  (autoregressive greedy decode → PER)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model:       PhoWhisperSeq2SeqModel,
    entries:     list[dict],
    audio_root:  str,
    device:      str,
    max_samples: int | None = None,
) -> float:
    """
    Greedy autoregressive decode on held-out entries → Phoneme Error Rate.

    Note: slower than CTC eval because each sample requires sequential
          token generation. Use max_samples to cap evaluation time.
    """
    model.eval()
    ref_seqs, hyp_seqs = [], []
    sample_entries = entries if max_samples is None else entries[:max_samples]

    for item in tqdm(sample_entries, desc="  Evaluating", unit="file",
                     leave=False, colour="yellow"):
        wav_path     = os.path.join(audio_root, item["audio"])
        ref_phonemes = item["phoneme"]
        try:
            hyp_phonemes = model.wav_to_phonemes(wav_path)
        except Exception as e:
            print(f"\n    [warn] skipped {wav_path}: {e}")
            hyp_phonemes = []
        ref_seqs.append(ref_phonemes)
        hyp_seqs.append(hyp_phonemes)

    model.train()
    return compute_per(ref_seqs, hyp_seqs)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(
    # ── Data ──────────────────────────────────────────────────────────────────
    num_epochs:       int   = 30,
    manifest:         str   = "dataset/manifest_all.jsonl",
    audio_root:       str   = "dataset",
    train_ratio:      float = 0.8,
    seed:             int   = 42,
    batch_size:       int   = 8,
    # ── Model ─────────────────────────────────────────────────────────────────
    d_model:          int   = 512,
    nhead:            int   = 8,
    num_layers:       int   = 6,
    dim_feedforward:  int   = 2048,
    dropout:          float = 0.1,
    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr:               float = 3e-4,
    warmup_steps:     int   = 4_000,
    weight_decay:     float = 0.01,
    max_grad_norm:    float = 1.0,
    # ── Teacher forcing ───────────────────────────────────────────────────────
    tf_start:         float = 1.0,
    tf_end:           float = 0.75,
    # ── Checkpoint & Eval ─────────────────────────────────────────────────────
    checkpoint_path:  str         = "checkpoint_seq2seq.pt",
    resume_from:      str | None  = None,     # path to existing checkpoint to resume
    eval_max_samples: int | None  = None,     # cap eval files (None = all held-out)
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'═' * 60}")
    print(f"  PhoWhisper → Transformer Decoder  (Seq2Seq Training)")
    print(f"{'═' * 60}")
    print(f"  Device          : {device}")
    print(f"  d_model         : {d_model}")
    print(f"  nhead / layers  : {nhead} / {num_layers}")
    print(f"  FFN dim         : {dim_feedforward}")
    print(f"  Dropout         : {dropout}")
    print(f"  lr / warmup     : {lr} / {warmup_steps} steps")
    print(f"  weight_decay    : {weight_decay}")
    print(f"  grad clip       : {max_grad_norm}")
    print(f"  TF ratio        : {tf_start:.2f} → {tf_end:.2f}")
    print(f"  Label smoothing : 0.1")
    print(f"{'═' * 60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    all_entries                  = load_manifest(manifest)
    train_entries, eval_entries  = split_manifest(all_entries, train_ratio, seed)
    print(f"Dataset split (seed={seed}): "
          f"{len(train_entries)} train  |  {len(eval_entries)} eval  "
          f"({train_ratio:.0%} / {1-train_ratio:.0%})\n")

    train_ds = AudioDataset(train_entries, audio_root)
    loader   = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PhoWhisperSeq2SeqModel(
        device          = device,
        d_model         = d_model,
        nhead           = nhead,
        num_layers      = num_layers,
        dim_feedforward = dim_feedforward,
        dropout         = dropout,
    )

    if resume_from and os.path.exists(resume_from):
        load_seq2seq_checkpoint(model, resume_from)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    print(f"Trainable parameters : {n_params:,}\n")

    # ── Optimizer & LR Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable,
        lr           = lr,
        betas        = (0.9, 0.98),
        eps          = 1e-9,
        weight_decay = weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda step: warmup_schedule(step, warmup_steps),
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        label_smoothing = 0.1,
        ignore_index    = PAD_IDX,
    )

    best_per    = float("inf")
    global_step = 0

    # ══════════════════════════════════════════════════════════════════════════
    for epoch in range(num_epochs):
        model.train()
        tf_ratio    = get_tf_ratio(epoch, num_epochs, tf_start, tf_end)
        total_loss  = 0
        num_batches = 0

        progress_bar = tqdm(
            loader,
            desc          = f"Epoch [{epoch + 1:02d}/{num_epochs}]  TF={tf_ratio:.3f}",
            unit          = "batch",
            dynamic_ncols = True,
            colour        = "cyan",
        )

        for mels, tgt_in, tgt_out, _ in progress_bar:
            mels    = mels.to(device)
            tgt_in  = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            # ── Scheduled-sampling forward → (B, L, V) ────────────────────────
            logits = scheduled_sampling_forward(model, mels, tgt_in, tf_ratio, device)

            B, L, V = logits.shape
            loss = criterion(logits.view(B * L, V), tgt_out.view(B * L))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss  += loss.item()
            num_batches += 1
            global_step += 1

            progress_bar.set_postfix({
                "loss" : f"{loss.item():.4f}",
                "avg"  : f"{total_loss / num_batches:.4f}",
                "lr"   : f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = total_loss / max(num_batches, 1)
        print(
            f"  ✓ Epoch [{epoch + 1:02d}/{num_epochs}]"
            f" — Avg Loss: {avg_loss:.4f}"
            f"  |  TF: {tf_ratio:.3f}"
            f"  |  Step: {global_step}"
        )

        # ── PER Evaluation on held-out 20 % ───────────────────────────────────
        per = evaluate(model, eval_entries, audio_root, device,
                       max_samples=eval_max_samples)
        print(f"  Eval PER: {per * 100:.2f}%", end="")

        if per < best_per:
            best_per = per
            save_seq2seq_checkpoint(
                model, checkpoint_path,
                epoch=epoch + 1, per=per, global_step=global_step,
            )
            print(f"  ✅ New best — checkpoint saved → {checkpoint_path}")
        else:
            print(f"  (best so far: {best_per * 100:.2f}% — not saved)")
        print()


if __name__ == "__main__":
    train()
