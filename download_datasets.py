# -*- coding: utf-8 -*-
"""
download_datasets.py
Tải toàn bộ 4 dataset (VLSP2020, CommonVoice, FOSD, VIVOS)
và lưu vào thư mục `dataset/` theo định dạng thống nhất:
    dataset/
        vlsp2020/         audio/*.wav  +  manifest.jsonl
        common_voice_vi/  audio/*.wav  +  manifest.jsonl
        common_voice_en/  audio/*.wav  +  manifest.jsonl
        fosd/             audio/*.wav  +  manifest.jsonl
        vivos/            audio/*.wav  +  manifest.jsonl

Mỗi dòng trong manifest.jsonl:
    {"audio": "audio/utt.wav", "text": "xin chào", "phoneme": ["s","i","n",...], "source": "vlsp2020"}

Speed-ups:
  - HF_HUB_ENABLE_HF_TRANSFER=1  → multi-part parallel shard download
    (requires: pip install hf-transfer)
  - ThreadPoolExecutor            → parallel WAV file writes
  - Streaming manifest writes     → no large in-memory list accumulation
"""

# ── Enable hf_transfer BEFORE importing huggingface_hub / datasets ───────────
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

import sys
import io
import json
import shutil
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force line-buffered UTF-8 stdout so Windows terminal never appears frozen
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# ── Local phoneme converter ──────────────────────────────────────────────────
from text2phoneme import text_to_phoneme

# ── Parallel IO workers ──────────────────────────────────────────────────────
_NUM_IO_WORKERS = min(8, (os.cpu_count() or 4))

# ── Dataset root ─────────────────────────────────────────────────────────────
DATASET_ROOT = Path("dataset")
DATASET_ROOT.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _write_wav(audio_array, sample_rate: int, out_path: Path):
    """Write one WAV — called from a worker thread."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(audio_array, bytes):
        out_path.write_bytes(audio_array)
    else:
        if hasattr(audio_array, "numpy"):
            audio_array = audio_array.numpy()
        audio_array = np.asarray(audio_array, dtype=np.float32)
        sf.write(str(out_path), audio_array, sample_rate)


def write_manifest(records: list, manifest_path: Path):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  >> Saved manifest: {manifest_path} ({len(records)} records)", flush=True)


def _process_hf_split_parallel(split_ds, split_name, audio_dir,
                                prefix, manifest_fh, extra_fields_fn,
                                start_idx=0):
    """
    Parallel-write WAVs for one HuggingFace split, stream-write manifest lines.
    Returns number of samples processed.
    """
    idx = start_idx
    pending = {}

    with ThreadPoolExecutor(max_workers=_NUM_IO_WORKERS) as pool:
        pbar = tqdm(total=len(split_ds),
                    desc=f"    {prefix}/{split_name}",
                    unit="sample", dynamic_ncols=True, file=sys.stdout)

        for item in split_ds:
            fname = f"{prefix}_{split_name}_{idx:06d}.wav"
            fpath = audio_dir / fname
            audio = item["audio"]
            fut   = pool.submit(_write_wav, audio["array"], audio["sampling_rate"], fpath)
            rec   = {"audio": f"audio/{fname}", **extra_fields_fn(item, split_name)}
            pending[fut] = rec
            idx += 1

            # Drain completed futures to keep memory low
            if len(pending) >= _NUM_IO_WORKERS * 4:
                done = [f for f in list(pending) if f.done()]
                for f in done:
                    f.result()
                    manifest_fh.write(
                        json.dumps(pending.pop(f), ensure_ascii=False) + "\n"
                    )
                    pbar.update(1)

        for fut in as_completed(pending):
            fut.result()
            manifest_fh.write(
                json.dumps(pending[fut], ensure_ascii=False) + "\n"
            )
            pbar.update(1)

        pbar.close()

    return idx - start_idx


# ─────────────────────────────────────────────────────────────────────────────
#  1. VLSP 2020  (Kaggle: tuannguyenvananh/vin-big-data-vlsp-2020-100h)
#     Layout: pairs of <name>.wav + <name>.txt in the same folder tree
# ─────────────────────────────────────────────────────────────────────────────

def download_vlsp2020():
    print("\n" + "="*60, flush=True)
    print("  [1/4] Downloading VLSP 2020 (Kaggle) ...")
    print("="*60, flush=True)

    try:
        import kagglehub
    except ImportError:
        print("  [ERROR] Missing 'kagglehub'. Run: pip install kagglehub")
        return

    try:
        raw_path = kagglehub.dataset_download(
            "tuannguyenvananh/vin-big-data-vlsp-2020-100h"
        )
    except Exception as e:
        print(f"  [ERROR] KaggleHub download failed: {e}")
        print("  Make sure kaggle.json is configured (~/.kaggle/kaggle.json)")
        return

    raw_path = Path(raw_path)
    print(f"  Raw files at: {raw_path}", flush=True)

    out_dir   = DATASET_ROOT / "vlsp2020"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # Collect all .wav files (paired with a same-name .txt)
    wav_files = sorted(raw_path.rglob("*.wav"))
    print(f"  Found {len(wav_files)} WAV files", flush=True)

    records = []

    def _copy_wav(args):
        src, dst = args
        shutil.copy2(src, dst)

    tasks = []
    for idx, src_wav in enumerate(wav_files):
        fname   = f"vlsp2020_{idx:06d}.wav"
        dst_wav = audio_dir / fname
        tasks.append((src_wav, dst_wav, fname))

    with ThreadPoolExecutor(max_workers=_NUM_IO_WORKERS) as pool:
        fut_map = {
            pool.submit(_copy_wav, (src, dst)): (src, fname)
            for src, dst, fname in tasks
        }
        for fut in tqdm(as_completed(fut_map), total=len(fut_map),
                        desc="    VLSP2020", unit="file",
                        dynamic_ncols=True, file=sys.stdout):
            src_wav, fname = fut_map[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"  [WARN] Copy failed for {src_wav.name}: {e}", flush=True)
                continue

            # Read paired .txt transcript
            txt_path = src_wav.with_suffix(".txt")
            text = ""
            if txt_path.exists():
                try:
                    text = txt_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

            phoneme = text_to_phoneme(text, mode="vi") if text else []

            records.append({
                "audio":   f"audio/{fname}",
                "text":    text,
                "phoneme": phoneme,
                "source":  "vlsp2020",
                "original_file": src_wav.name,
            })

    write_manifest(records, manifest_path)
    print(f"  [VLSP2020] Done: {len(records)} samples → {out_dir}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  2. COMMON VOICE  (HuggingFace: vi + en)
# ─────────────────────────────────────────────────────────────────────────────

def download_common_voice():
    print("\n" + "="*60, flush=True)
    print("  [2/4] Downloading Common Voice (vi + en) ...")
    print("="*60, flush=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] Missing 'datasets'. Run: pip install datasets")
        return

    for lang_code, lang_name, phoneme_mode in [
        ("vi", "common_voice_vi", "vi"),
        ("en", "common_voice_en", "en"),
    ]:
        print(f"\n  -- Language: {lang_code} --", flush=True)

        out_dir   = DATASET_ROOT / lang_name
        audio_dir = out_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.jsonl"

        try:
            ds = load_dataset(
                "mozilla-foundation/common_voice_17_0",
                lang_code,
                trust_remote_code=True,
                token=True,
            )
        except Exception as e:
            print(f"  [WARN] common_voice_17_0/{lang_code} failed: {e}")
            print("  Trying common_voice_11_0 ...")
            try:
                ds = load_dataset(
                    "mozilla-foundation/common_voice_11_0",
                    lang_code,
                    trust_remote_code=True,
                )
            except Exception as e2:
                print(f"  [ERROR] Failed to load Common Voice {lang_code}: {e2}")
                continue

        total = 0
        with open(manifest_path, "w", encoding="utf-8") as mf:
            for split_name, split_ds in ds.items():
                print(f"    Split: {split_name} ({len(split_ds)} samples)", flush=True)

                def fields(item, sn, lc=lang_code, pm=phoneme_mode):
                    text = item.get("sentence", "")
                    return {
                        "text":     text,
                        "phoneme":  text_to_phoneme(text, mode=pm) if text else [],
                        "source":   f"common_voice_{lc}",
                        "split":    sn,
                        "language": lc,
                    }

                n = _process_hf_split_parallel(
                    split_ds, split_name, audio_dir,
                    f"cv_{lang_code}", mf, fields,
                    start_idx=total,
                )
                total += n

        print(f"  [CommonVoice-{lang_code}] Done: {total} samples → {out_dir}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  3. FOSD  (Kaggle: thinh127/fpt-open-speech-dataset-fosd-vietnamese)
#     Layout:
#       <root>/mp3/*.mp3          ← audio files
#       <root>/transcriptAll.txt  ← pipe-delimited: filename|text|time_range
# ─────────────────────────────────────────────────────────────────────────────

def download_fosd():
    print("\n" + "="*60, flush=True)
    print("  [3/4] Downloading FOSD (Kaggle) ...")
    print("="*60, flush=True)

    try:
        import kagglehub
    except ImportError:
        print("  [ERROR] Missing 'kagglehub'. Run: pip install kagglehub")
        return

    try:
        raw_path = kagglehub.dataset_download(
            "thinh127/fpt-open-speech-dataset-fosd-vietnamese"
        )
    except Exception as e:
        print(f"  [ERROR] KaggleHub download failed: {e}")
        print("  Make sure kaggle.json is configured (~/.kaggle/kaggle.json)")
        return

    raw_path = Path(raw_path)
    print(f"  Raw files at: {raw_path}", flush=True)

    out_dir   = DATASET_ROOT / "fosd"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # ── Parse transcriptAll.txt ──────────────────────────────────────────────
    # Format per line: FPTOpenSpeechData_Set001_V0.1_000001.mp3|cách để đi|0.00000-1.27298
    transcript_map = {}  # stem (without ext) → text
    transcript_files = list(raw_path.rglob("transcriptAll.txt"))
    if not transcript_files:
        # Fallback: any .txt that looks like a transcript
        transcript_files = list(raw_path.rglob("transcript*.txt"))
    for tf in transcript_files:
        print(f"  Reading transcript: {tf}", flush=True)
        with open(tf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 2:
                    fname_key = Path(parts[0].strip()).stem  # remove extension
                    text      = parts[1].strip()
                    transcript_map[fname_key] = text

    print(f"  Transcript entries loaded: {len(transcript_map)}", flush=True)

    # ── Find audio files in mp3/ folder ──────────────────────────────────────
    mp3_dir    = raw_path / "mp3"
    audio_exts = {".mp3", ".wav", ".flac", ".ogg"}
    if mp3_dir.exists():
        audio_files = sorted([f for f in mp3_dir.rglob("*") if f.suffix.lower() in audio_exts])
    else:
        audio_files = sorted([f for f in raw_path.rglob("*") if f.suffix.lower() in audio_exts])

    print(f"  Found {len(audio_files)} audio files", flush=True)

    def _convert_audio(args):
        src, dst = args
        if src.suffix.lower() == ".wav":
            shutil.copy2(src, dst)
        else:
            data, sr = sf.read(str(src))
            sf.write(str(dst), data, sr)

    tasks     = []
    stem_map  = {}  # new fname → original stem
    for idx, src in enumerate(audio_files):
        fname    = f"fosd_{idx:06d}.wav"
        dst      = audio_dir / fname
        tasks.append((src, dst, fname, src.stem))

    records = []
    with ThreadPoolExecutor(max_workers=_NUM_IO_WORKERS) as pool:
        fut_map = {
            pool.submit(_convert_audio, (src, dst)): (src, fname, stem)
            for src, dst, fname, stem in tasks
        }
        for fut in tqdm(as_completed(fut_map), total=len(fut_map),
                        desc="    FOSD", unit="file",
                        dynamic_ncols=True, file=sys.stdout):
            src, fname, stem = fut_map[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"  [WARN] Convert failed for {src.name}: {e}", flush=True)
                continue

            text    = transcript_map.get(stem, "")
            phoneme = text_to_phoneme(text, mode="vi") if text else []

            records.append({
                "audio":         f"audio/{fname}",
                "text":          text,
                "phoneme":       phoneme,
                "source":        "fosd",
                "original_file": src.name,
            })

    write_manifest(records, manifest_path)
    print(f"  [FOSD] Done: {len(records)} samples → {out_dir}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  4. VIVOS  (Kaggle: kynthesis/vivos-vietnamese-speech-corpus-for-asr)
#     Layout:
#       train/
#         waves/<speaker>/<speaker>_<utt>.wav
#         prompts.txt   →  VIVOSSPK01_R001 KHÁCH SẠN (uppercase)
#       test/
#         same structure
# ─────────────────────────────────────────────────────────────────────────────

def download_vivos():
    print("\n" + "="*60, flush=True)
    print("  [4/4] Downloading VIVOS (Kaggle) ...")
    print("="*60, flush=True)

    try:
        import kagglehub
    except ImportError:
        print("  [ERROR] Missing 'kagglehub'. Run: pip install kagglehub")
        return

    try:
        raw_path = kagglehub.dataset_download(
            "kynthesis/vivos-vietnamese-speech-corpus-for-asr"
        )
    except Exception as e:
        print(f"  [ERROR] KaggleHub download failed: {e}")
        print("  Make sure kaggle.json is configured (~/.kaggle/kaggle.json)")
        return

    raw_path = Path(raw_path)
    print(f"  Raw files at: {raw_path}", flush=True)

    out_dir   = DATASET_ROOT / "vivos"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    records = []

    for split_name in ["train", "test"]:
        split_dir = raw_path / split_name
        if not split_dir.exists():
            # Some archives nest under vivos/
            split_dir = raw_path / "vivos" / split_name
        if not split_dir.exists():
            print(f"  [WARN] Split folder not found: {split_name}", flush=True)
            continue

        # ── Read prompts.txt ─────────────────────────────────────────────────
        # Format: VIVOSSPK01_R001 KHÁCH SẠN (key is utt-id = filename stem)
        prompts_file = split_dir / "prompts.txt"
        utt2text = {}
        if prompts_file.exists():
            with open(prompts_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(None, 1)  # split on first whitespace
                    if len(parts) == 2:
                        utt_id, text = parts
                        utt2text[utt_id] = text.lower()  # convert to lowercase
        else:
            print(f"  [WARN] prompts.txt not found in {split_dir}", flush=True)

        print(f"  Split '{split_name}': {len(utt2text)} prompts loaded", flush=True)

        # ── Collect WAV files ────────────────────────────────────────────────
        waves_dir = split_dir / "waves"
        if not waves_dir.exists():
            wav_files = sorted(split_dir.rglob("*.wav"))
        else:
            wav_files = sorted(waves_dir.rglob("*.wav"))

        print(f"  Split '{split_name}': {len(wav_files)} WAV files found", flush=True)

        def _copy_wav(args):
            src, dst = args
            shutil.copy2(src, dst)

        tasks = []
        for idx, src_wav in enumerate(wav_files):
            fname   = f"vivos_{split_name}_{idx:06d}.wav"
            dst_wav = audio_dir / fname
            tasks.append((src_wav, dst_wav, fname, src_wav.stem))

        with ThreadPoolExecutor(max_workers=_NUM_IO_WORKERS) as pool:
            fut_map = {
                pool.submit(_copy_wav, (src, dst)): (src, fname, stem)
                for src, dst, fname, stem in tasks
            }
            for fut in tqdm(as_completed(fut_map), total=len(fut_map),
                            desc=f"    VIVOS/{split_name}", unit="file",
                            dynamic_ncols=True, file=sys.stdout):
                src, fname, stem = fut_map[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"  [WARN] Copy failed for {src.name}: {e}", flush=True)
                    continue

                text    = utt2text.get(stem, "")
                phoneme = text_to_phoneme(text, mode="vi") if text else []

                records.append({
                    "audio":   f"audio/{fname}",
                    "text":    text,
                    "phoneme": phoneme,
                    "source":  "vivos",
                    "split":   split_name,
                    "utt_id":  stem,
                })

    write_manifest(records, manifest_path)
    print(f"  [VIVOS] Done: {len(records)} samples → {out_dir}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MERGE all sub-manifests → manifest_all.jsonl
# ─────────────────────────────────────────────────────────────────────────────

def merge_all_manifests():
    print("\n" + "="*60, flush=True)
    print("  [MERGE] Combining all manifests ...")
    print("="*60, flush=True)

    merged_path = DATASET_ROOT / "manifest_all.jsonl"
    total = 0

    with open(merged_path, "w", encoding="utf-8") as out_f:
        for sub_dir in sorted(DATASET_ROOT.iterdir()):
            if not sub_dir.is_dir():
                continue
            manifest = sub_dir / "manifest.jsonl"
            if not manifest.exists():
                continue
            count = 0
            with open(manifest, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rec["audio"] = f"{sub_dir.name}/{rec['audio']}"
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
            print(f"  {sub_dir.name:25s}: {count:6d} records", flush=True)
            total += count

    print(f"\n  >> Merged manifest: {merged_path}", flush=True)
    print(f"     Total records   : {total}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    # Verify hf_transfer
    try:
        import hf_transfer  # noqa: F401
        print("  [hf_transfer] ✓ active (fast multi-part HF download)", flush=True)
    except ImportError:
        print("  [hf_transfer] ✗ not installed — run: pip install hf-transfer", flush=True)

    parser = argparse.ArgumentParser(description="Download all 4 ASR datasets")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["vlsp2020", "common_voice", "fosd", "vivos"],
        choices=["vlsp2020", "common_voice", "fosd", "vivos"],
        help="Datasets to download (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_NUM_IO_WORKERS,
        help=f"Parallel WAV write threads (default: {_NUM_IO_WORKERS})",
    )
    args = parser.parse_args()
    _NUM_IO_WORKERS = args.workers

    print("\n" + "=" * 60, flush=True)
    print("  DATASET DOWNLOADER")
    print(f"  Save location : {DATASET_ROOT.resolve()}")
    print(f"  IO workers    : {_NUM_IO_WORKERS}")
    print(f"  hf_transfer   : {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0')}")
    print("=" * 60, flush=True)

    if "vlsp2020"     in args.sources: download_vlsp2020()
    if "common_voice" in args.sources: download_common_voice()
    if "fosd"         in args.sources: download_fosd()
    if "vivos"        in args.sources: download_vivos()

    merge_all_manifests()

    print("\n" + "=" * 60, flush=True)
    print("  DONE! Dataset structure:")
    print(f"  {DATASET_ROOT}/")
    for d in sorted(DATASET_ROOT.iterdir()):
        if d.is_dir():
            n = sum(1 for _ in (d / "audio").glob("*.wav")) if (d / "audio").exists() else 0
            print(f"    {d.name}/  ({n} wav files)")
    print(f"    manifest_all.jsonl  (merged)")
    print("=" * 60, flush=True)
