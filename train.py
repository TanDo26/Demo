import json
import os
import random
import logging
import threading

# Suppress the harmless 403 "Discussions disabled" background-thread warning.
# It's raised by transformers' safetensors_conversion daemon thread and is not
# a real error — the model loads fine from pytorch_model.bin.
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

from model import PhoWhisperCTCModel, load_wav_to_mel, ctc_greedy_decode
from phoneme_set import encode_sequence, decode_sequence, PAD_IDX


# ================= PER METRIC =================
def levenshtein(ref: list, hyp: list) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def compute_per(ref_seqs: list[list[str]], hyp_seqs: list[list[str]]) -> float:
    """
    Phoneme Error Rate = total edit distance / total reference length.
    Returns a float in [0, 1] (0 = perfect).
    """
    total_edits = sum(levenshtein(r, h) for r, h in zip(ref_seqs, hyp_seqs))
    total_ref   = sum(len(r) for r in ref_seqs)
    return total_edits / max(total_ref, 1)


# ================= DATASET =================
def load_manifest(manifest_path: str) -> list[dict]:
    """Read a .jsonl manifest file and return a list of record dicts."""
    entries = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def split_manifest(entries: list[dict], train_ratio: float = 0.8, seed: int = 42
                   ) -> tuple[list[dict], list[dict]]:
    """Shuffle and split entries into (train, test) by train_ratio."""
    entries = entries.copy()
    random.seed(seed)
    random.shuffle(entries)
    cut = int(len(entries) * train_ratio)
    return entries[:cut], entries[cut:]


class AudioDataset(Dataset):
    """Dataset built from a list of manifest dicts (already split)."""
    def __init__(self, entries: list[dict], audio_root: str = "dataset"):
        self.data       = entries
        self.audio_root = audio_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item     = self.data[idx]
        wav      = os.path.join(self.audio_root, item["audio"])
        phonemes = item["phoneme"]
        mel      = load_wav_to_mel(wav)            # (1, 3000, 80)
        target   = torch.tensor(encode_sequence(phonemes), dtype=torch.long)
        return mel.squeeze(0), target, phonemes    # raw phonemes for PER


# ================= COLLATE =================
def collate_fn(batch):
    mels, targets, phoneme_lists = zip(*batch)

    mel_lens    = [m.shape[0] for m in mels]
    target_lens = [len(t)     for t in targets]

    mels    = nn.utils.rnn.pad_sequence(mels,    batch_first=True)
    targets = torch.cat(targets)

    return mels, targets, mel_lens, target_lens, list(phoneme_lists)


# ================= EVALUATE =================
@torch.no_grad()
def evaluate(model, entries: list[dict], audio_root: str, device: str) -> float:
    """Run greedy CTC decode on a list of manifest entries and return PER."""
    model.eval()
    ref_seqs, hyp_seqs = [], []

    for item in tqdm(entries, desc="  Evaluating", unit="file", leave=False, colour="yellow"):
        wav_path     = os.path.join(audio_root, item["audio"])
        ref_phonemes = item["phoneme"]

        try:
            mel          = load_wav_to_mel(wav_path).to(device)  # (1, 3000, 80)
            logits       = model(mel)                             # (1, T', V)
            indices      = ctc_greedy_decode(logits[0])
            hyp_phonemes = decode_sequence(indices, skip_special=True)
        except Exception as e:
            print(f"    [warn] skipped {wav_path}: {e}")
            hyp_phonemes = []

        ref_seqs.append(ref_phonemes)
        hyp_seqs.append(hyp_phonemes)

    model.train()
    return compute_per(ref_seqs, hyp_seqs)


# ================= TRAIN =================
def train(
    num_epochs:      int   = 30,
    manifest:        str   = "dataset/manifest_all.jsonl",
    audio_root:      str   = "dataset",
    train_ratio:     float = 0.8,
    seed:            int   = 42,
    checkpoint_path: str   = "checkpoint.pt",
    batch_size:      int   = 8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load & split manifest 80 / 20 ────────────────────────────────────────
    all_entries = load_manifest(manifest)
    train_entries, eval_entries = split_manifest(all_entries, train_ratio=train_ratio, seed=seed)
    print(f"Dataset split (seed={seed}): "
          f"{len(train_entries)} train  |  {len(eval_entries)} eval  "
          f"(ratio {train_ratio:.0%} / {1-train_ratio:.0%})\n")

    train_ds = AudioDataset(train_entries, audio_root=audio_root)
    loader   = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = PhoWhisperCTCModel(device=device)
    optimizer = torch.optim.Adam(model.ctc_head.parameters(), lr=1e-3)
    ctc_loss  = nn.CTCLoss(blank=PAD_IDX, zero_infinity=True)

    best_per = float("inf")   # track best PER for checkpoint saving

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        model.train()
        total_loss  = 0
        num_batches = 0

        progress_bar = tqdm(
            loader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}]",
            unit="batch",
            dynamic_ncols=True,
            colour="cyan",
        )

        for mels, targets, mel_lens, target_lens, _ in progress_bar:
            mels = mels.to(device)

            logits    = model(mels)                 # (B, T', V)
            log_probs = logits.log_softmax(dim=-1)
            log_probs = log_probs.permute(1, 0, 2)  # (T', B, V)

            input_lengths  = torch.tensor([log_probs.size(0)] * log_probs.size(1))
            target_lengths = torch.tensor(target_lens)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

            progress_bar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss":   f"{total_loss / num_batches:.4f}",
            })

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  ✓ Epoch [{epoch + 1}/{num_epochs}] — Avg Loss: {avg_loss:.4f}")

        # ── PER Evaluation on held-out 20 % ───────────────────────────────────
        per = evaluate(model, eval_entries, audio_root, device)
        print(f"  Eval PER : {per * 100:.2f}%", end="")

        if per < best_per:
            best_per = per
            torch.save(model.ctc_head.state_dict(), checkpoint_path)
            print(f"  ✅ New best — checkpoint saved → {checkpoint_path}")
        else:
            print(f"  (best so far: {best_per * 100:.2f}% — not saved)")
        print()


if __name__ == "__main__":
    train()
