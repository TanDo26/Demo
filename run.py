"""
run.py
Apply the trained checkpoint to wav files listed in data/manifest.jsonl,
show per-file results (text / true phoneme / predicted phoneme),
and report the overall Phoneme Error Rate (PER).

Usage:
    python run.py
    python run.py --manifest data/manifest.jsonl --checkpoint checkpoint.pt
"""

import argparse
import json
import os
import logging
import threading

# ── Suppress harmless HF Hub background-thread noise ──────────────────────────
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)

_orig_excepthook = threading.excepthook
def _silent_hf(args):
    module = getattr(args.exc_type, "__module__", "") or ""
    if "huggingface_hub" in module or "safetensors_conversion" in module:
        return
    _orig_excepthook(args)
threading.excepthook = _silent_hf
# ──────────────────────────────────────────────────────────────────────────────

import torch
from tqdm import tqdm

from model import PhoWhisperCTCModel, load_wav_to_mel, ctc_greedy_decode
from phoneme_set import decode_sequence


# ================= PER HELPERS =================
def levenshtein(ref: list, hyp: list) -> int:
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if ref[i-1] == hyp[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[m]


def compute_per(ref_seqs: list, hyp_seqs: list) -> float:
    total_edits = sum(levenshtein(r, h) for r, h in zip(ref_seqs, hyp_seqs))
    total_ref   = sum(len(r) for r in ref_seqs)
    return total_edits / max(total_ref, 1)


# ================= PHONEME → TEXT (inverse mapper) =================
_TONES    = {"-1", "-2", "-3", "-4", "-5", "-6"}
_INITIALS = {"b","k","tʃ","z","d","g","h","x","l","m","n","ŋ","ɲ","p","f","r","s","t","tʰ","v","ʔ"}
_NUCLEI   = {"a","ɤ","ɛ","e","i","ɔ","o","u","ɯ","iə","uə","ɯə"}
_CODAS    = {"p","t","k","m","n","ŋ","j","w"}

_INV_INITIAL = {
    "b":"b",  "k":"k",  "tʃ":"ch", "z":"d",  "d":"đ",
    "g":"g",  "h":"h",  "x":"kh",  "l":"l",  "m":"m",
    "n":"n",  "ŋ":"ng", "ɲ":"nh",  "p":"p",  "f":"ph",
    "r":"r",  "s":"x",  "t":"t",   "tʰ":"th","v":"v",  "ʔ":"",
}
_INV_NUCLEUS = {
    "a":"a",  "ɤ":"ơ",  "ɛ":"e",  "e":"ê",  "i":"i",
    "ɔ":"o",  "o":"ô",  "u":"u",  "ɯ":"ư",
    "iə":"ia","uə":"ua","ɯə":"ưa",
}
_INV_CODA = {
    "p":"p","t":"t","k":"c","m":"m","n":"n","ŋ":"ng","j":"i","w":"u",
}

# Tone diacritics applied to the main vowel letter
_TONE_TABLE: dict[str, dict[str, str]] = {
    "-1": {},   # ngang — no mark
    "-2": {"a":"à","ă":"ằ","â":"ầ","e":"è","ê":"ề","i":"ì","o":"ò","ô":"ồ","ơ":"ờ","u":"ù","ư":"ừ","y":"ỳ"},
    "-3": {"a":"ả","ă":"ẳ","â":"ẩ","e":"ẻ","ê":"ể","i":"ỉ","o":"ỏ","ô":"ổ","ơ":"ở","u":"ủ","ư":"ử","y":"ỷ"},
    "-4": {"a":"á","ă":"ắ","â":"ấ","e":"é","ê":"ế","i":"í","o":"ó","ô":"ố","ơ":"ớ","u":"ú","ư":"ứ","y":"ý"},
    "-5": {"a":"ạ","ă":"ặ","â":"ậ","e":"ẹ","ê":"ệ","i":"ị","o":"ọ","ô":"ộ","ơ":"ợ","u":"ụ","ư":"ự","y":"ỵ"},
    "-6": {"a":"ã","ă":"ẵ","â":"ẫ","e":"ẽ","ê":"ễ","i":"ĩ","o":"õ","ô":"ỗ","ơ":"ỡ","u":"ũ","ư":"ữ","y":"ỹ"},
}
_VOWELS = set("aăâeêioôơuưy")


def _apply_tone(syl: str, tone: str) -> str:
    """Apply Vietnamese tone diacritic to the last vowel in the syllable string."""
    marks = _TONE_TABLE.get(tone, {})
    if not marks:
        return syl
    chars = list(syl)
    for i in range(len(chars) - 1, -1, -1):
        if chars[i] in marks:
            chars[i] = marks[chars[i]]
            break
    return "".join(chars)


def _syl_phonemes_to_text(phones: list[str]) -> str:
    """Convert one syllable's phoneme list back to a Vietnamese syllable string."""
    if not phones:
        return ""

    # Extract trailing tone token
    tone = "-1"
    if phones[-1] in _TONES:
        tone, phones = phones[-1], phones[:-1]

    idx, result = 0, ""

    # Initial consonant cluster
    if idx < len(phones) and phones[idx] in _INITIALS:
        result += _INV_INITIAL.get(phones[idx], phones[idx])
        idx += 1

    # Optional medial /w/ (only when followed by a nucleus)
    if idx < len(phones) and phones[idx] == "w" and idx + 1 < len(phones) and phones[idx + 1] in _NUCLEI:
        result += "u"
        idx += 1

    # Nucleus (vowel)
    if idx < len(phones) and phones[idx] in _NUCLEI:
        result += _INV_NUCLEUS.get(phones[idx], phones[idx])
        idx += 1

    # Coda
    if idx < len(phones) and phones[idx] in _CODAS:
        coda = phones[idx]
        result += "i" if coda == "j" else "u" if coda == "w" else _INV_CODA.get(coda, coda)
        idx += 1

    return _apply_tone(result, tone) if result else ""


def phonemes_to_text(phonemes: list[str]) -> str:
    """
    Convert a flat IPA phoneme list (with '$' syllable boundaries) back to
    approximate Vietnamese text.
    """
    # Split into per-syllable groups
    syllable_groups: list[list[str]] = []
    current: list[str] = []
    for ph in phonemes:
        if ph == "$":
            syllable_groups.append(current)
            current = []
        else:
            current.append(ph)
    if current:
        syllable_groups.append(current)

    words = [_syl_phonemes_to_text(syl) for syl in syllable_groups]
    return " ".join(w for w in words if w)


# ================= MAIN =================
def run(
    manifest_path: str  = "data/manifest.jsonl",
    audio_root:    str  = "data",
    checkpoint:    str  = "checkpoint.pt",
    device:        str  | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Using device : {device}")
    print(f"Checkpoint   : {checkpoint}")
    print(f"Manifest     : {manifest_path}\n")

    model = PhoWhisperCTCModel(device=device)
    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        model.ctc_head.load_state_dict(ckpt)
        print(f"  ✓ Loaded checkpoint: {checkpoint}\n")
    else:
        print(f"  ⚠ No checkpoint found at '{checkpoint}' — using random weights.\n")

    model.eval()

    # ── Load manifest ─────────────────────────────────────────────────────────
    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]

    print(f"{'='*70}")
    print(f"  Running inference on {len(entries)} file(s)")
    print(f"{'='*70}\n")

    ref_seqs, hyp_seqs = [], []
    per_file_results   = []

    for item in tqdm(entries, desc="Inferring", unit="file", colour="cyan"):
        wav_path     = os.path.join(audio_root, item["audio"])
        text         = item.get("text", "")
        ref_phonemes = item["phoneme"]

        try:
            with torch.no_grad():
                mel    = load_wav_to_mel(wav_path).to(device)   # (1, 3000, 80)
                logits = model(mel)                              # (1, T', V)
                indices      = ctc_greedy_decode(logits[0])
                hyp_phonemes = decode_sequence(indices, skip_special=True)
        except Exception as e:
            print(f"\n  [warn] skipped {wav_path}: {e}")
            hyp_phonemes = []

        file_per = compute_per([ref_phonemes], [hyp_phonemes])
        ref_seqs.append(ref_phonemes)
        hyp_seqs.append(hyp_phonemes)
        per_file_results.append((wav_path, text, ref_phonemes, hyp_phonemes, file_per))

    # ── Print per-file results ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Results per file")
    print(f"{'='*70}")

    for wav_path, text, ref, hyp, file_per in per_file_results:
        fname = os.path.basename(wav_path)
        pred_text = phonemes_to_text(hyp) if hyp else "<empty>"
        print(f"\n  File      : {fname}  (PER: {file_per*100:.1f}%)")
        print(f"  Text      : {text}")
        print(f"  True phon : {' '.join(ref)}")
        print(f"  Pred phon : {' '.join(hyp) if hyp else '<empty>'}")
        print(f"  Pred text : {pred_text}")

    # ── Overall PER ───────────────────────────────────────────────────────────
    overall_per = compute_per(ref_seqs, hyp_seqs)
    print(f"\n{'='*70}")
    print(f"  Overall PER : {overall_per * 100:.2f}%  ({len(entries)} files)")
    print(f"{'='*70}\n")


# ================= CLI =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply checkpoint and compute PER")
    parser.add_argument("--manifest",   default="data/manifest.jsonl",
                        help="Path to manifest .jsonl  (default: data/manifest.jsonl)")
    parser.add_argument("--audio_root", default="data",
                        help="Root folder prepended to audio paths in manifest  (default: dataset)")
    parser.add_argument("--checkpoint", default="checkpoint.pt",
                        help="Checkpoint .pt file  (default: checkpoint.pt)")
    parser.add_argument("--device",     default=None,
                        help="cpu / cuda  (default: auto-detect)")
    args = parser.parse_args()

    run(
        manifest_path = args.manifest,
        audio_root    = args.audio_root,
        checkpoint    = args.checkpoint,
        device        = args.device,
    )
