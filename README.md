# Vietnamese ASR Phoneme Pipeline

A rule-based, dialect-aware **Grapheme-to-Phoneme (G2P)** pipeline for Vietnamese ASR training data.  
Converts raw text in speech manifests into Latin-encoded phoneme sequences.

---

## Repository Structure

```
Demo_Whisper/
├── phoneme_set.py        # Phoneme vocabulary definition & utilities
├── text2phoneme.py       # G2P engine + manifest processor
├── download_datasets.py  # Dataset downloader & manifest builder
└── dataset/              # Output directory (auto-created)
    ├── vlsp2020/  audio/*.wav + manifest.jsonl
    ├── fosd/      audio/*.wav + manifest.jsonl
    ├── vivos/     audio/*.wav + manifest.jsonl
    ├── tedlium/   audio/*.wav + manifest.jsonl
    └── manifest_all.jsonl  ← merged manifest
```

---

## `phoneme_set.py` — Phoneme Vocabulary

Defines the complete **Latin-based phoneme set** used throughout the pipeline. All symbols are ASCII-only for compatibility with standard ASR tokenizers.

### Phoneme Categories

| Category | Symbols | Notes |
|---|---|---|
| **Special tokens** | `<pad>` `<unk>` `<sot>` `<eot>` `$` | `$` = syllable boundary |
| **Initials** (âm đầu) | `b c ch tr d z j g h kh l m n ng nh p ph r s x t th v qu` | Dialect variants split (e.g. `z`/`j` for Northern/Southern `d/gi`) |
| **Medials** (âm đệm) | `u o` | Labio-velar glides before the nucleus |
| **Nuclei** (nguyên âm) | `a aw aa e ee i o oo ow u uw ie ua uo uwa uow` | Short, long, diphthong and triphthong nuclei |
| **Codas** (âm cuối) | `p t c m n ng nh y u o` | Final consonants |
| **Tones** (thanh điệu) | `t1 t2 t3 t4 t5 t6` | ngang / huyền / hỏi / sắc / nặng / ngã |

## `text2phoneme.py` — G2P Engine & Manifest Processor

### Overview

Converts Vietnamese (and mixed-language) text into sequences of Latin phoneme tokens for both the **Northern** and **Southern** dialects.

**Pipeline per syllable:**

```
raw text
  ↓ VietnameseNormalizer.normalize()   (numbers, dates, abbreviations → words)
  ↓ text_to_graphemes()               (word segmentation + Vietlish / EN handling)
  ↓ vn_syllable_to_phonemes()         (tone extraction → initial + rhyme mapping)
  ↓
[initial, ...rhyme_phonemes..., tone]  × {north, south}
```

### Tone Encoding

Six tones are encoded by Unicode NFD diacritic detection. Tone `t1` (level/ngang) has no diacritic and is the implicit default.

| Token | Tone | Diacritic |
|---|---|---|
| `t1` | ngang (level) | *(none)* |
| `t2` | huyền (falling) | `\u0300` grave |
| `t3` | hỏi (dipping-rising) | `\u0309` hook above |
| `t4` | sắc (rising) | `\u0301` acute |
| `t5` | nặng (heavy-falling) | `\u0323` dot below |
| `t6` | ngã (broken-rising) | `\u0303` tilde |

### Dialect Differences

Dialect variants are handled at two levels:

**1. Initials** — mapped in `vn_syllable_to_phonemes()`:

| Grapheme | North | South |
|---|---|---|
| `d`, `gi` | `z` | `j` |
| `r` | `z` | `r` |
| `tr` | `ch` | `tr` |
| `s` | `x` | `s` |
| `v` | `v` | `j` |

**2. Rhymes** — stored in `_VN_RHYME_MAP_LATIN`. When a rhyme has dialect-specific pronunciation variants, its value is a **tuple of lists** (positional, index 0 = north, index 1 = south):

```python
# Single pronunciation — shared by both dialects
"an": ["a", "n"],

# Two variants — (north, south)
"au": (["a", "u"], ["a", "o"]),
```

### Language Detection & Modes

`detect_language(text)` classifies text as one of four modes:

| Mode | Description |
|---|---|
| `vi` | Pure Vietnamese (≥ 80 % VN words) |
| `en` | Pure English (≥ 80 % EN words, no Vietlish) |
| `vietlish` | English words pronounced Vietnamese-style |
| `iev` | Code-switching (mixed VN + EN) |

English OOV words are transliterated to Vietnamese syllables via `vietnormalizer.transliterate_word()`.

### Key Public API

```python
from text2phoneme import text_to_phoneme, detect_language, process_manifest

# Single utterance
ph = text_to_phoneme("xin chào bạn", mode="vi")
# → {"north": ["s","i","n","t1","$","ch","a","w","t2","$","b","a","n","t5"],
#    "south": ["x","i","n","t1","$","ch","a","w","t2","$","b","a","n","t5"]}

# Manifest processing
process_manifest("data/manifest.jsonl", output_path="data/manifest_all.jsonl")
```

### Manifest Processor

`process_manifest()` reads a `.jsonl` manifest, applies the full pipeline to each record, and writes two output records per input — one per dialect — to `manifest_all.jsonl`:

```jsonc
// Input
{"audio": "clip001.wav", "text": "xin chào bạn"}

// Output (2 records)
{"audio": "clip001.wav", "text_raw": "xin chào bạn", "text": "xin chào bạn",
 "normalized_text": "xin chào bạn", "dialect": "north",
 "phonemes": ["x","i","n","t1","$","ch","a","w","t2","$","b","a","n","t5"]}

{"audio": "clip001.wav", "text_raw": "xin chào bạn", "text": "xin chào bạn",
 "normalized_text": "xin chào bạn", "dialect": "south",
 "phonemes": ["s","i","n","t1","$","ch","a","w","t2","$","b","a","n","t5"]}
```

### CLI Usage

```bash
# Process a manifest (writes manifest_all.jsonl next to input by default)
python text2phoneme.py --manifest data/manifest.jsonl

# Specify output path and language mode
python text2phoneme.py --manifest data/manifest.jsonl --mode vi --output data/out.jsonl

# Dry-run: print first 5 records without writing
python text2phoneme.py --manifest data/manifest.jsonl --dry_run

# Run built-in demo
python text2phoneme.py
```

---

## `download_datasets.py` — Dataset Downloader

Downloads, converts, and organises **4 ASR datasets** into a unified `dataset/` directory structure. Each dataset gets its own subfolder with WAV files and a `manifest.jsonl`.

### Datasets

| # | Dataset | Source | Language | Notes |
|---|---|---|---|---|
| 1 | **VLSP 2020** | Kaggle (`tuannguyenvananh/vin-big-data-vlsp-2020-100h`) | Vietnamese | ~100 h, paired `.wav`/`.txt` |
| 2 | **FOSD** | Kaggle (`thinh127/fpt-open-speech-dataset-fosd-vietnamese`) | Vietnamese | MP3 → WAV converted |
| 3 | **VIVOS** | Kaggle (`kynthesis/vivos-vietnamese-speech-corpus-for-asr`) | Vietnamese | Train + test splits |
| 4 | **TEDLIUM-1** | Kaggle (`peterokonma/tedlium-1-cleaned`) | English | TED talk recordings |

### Manifest Schema (per record)

```jsonc
{
  "audio":         "audio/vlsp2020_000001.wav",  // relative path within dataset dir
  "text":          "xin chào",
  "phoneme":       ["x","i","n","t1","$","ch","a","w","t2"],
  "source":        "vlsp2020",
  "split":         "train",          // (if applicable)
  "language":      "vi",             // (if applicable)
  "original_file": "original.wav"    // (if applicable)
}
```

### Performance Features

- **`ThreadPoolExecutor`** — parallel WAV file writes (default: `min(8, cpu_count)` workers)
- **Streaming manifest writes** — records flushed incrementally, no large in-memory lists

### CLI Usage

```bash
# Download all datasets
python download_datasets.py

# Download only specific datasets
python download_datasets.py --sources vlsp2020 vivos

# Control parallel workers
python download_datasets.py --workers 4
```

After all downloads, `merge_all_manifests()` is called automatically to produce:

```
dataset/manifest_all.jsonl   ← merged from all sub-manifests
```

### Prerequisites

```bash
pip install kagglehub soundfile numpy tqdm
# Kaggle credentials: ~/.kaggle/kaggle.json
```

---

## Dependencies Summary

| Package | Used by | Purpose |
|---|---|---|
| `vietnamesenormalizer` | `text2phoneme` | Normalize numbers, dates, abbreviations |
| `vietnormalizer` | `text2phoneme` | VN word detection, EN→VN transliteration |
| `kagglehub` | `download_datasets` | Kaggle dataset download |
| `soundfile` | `download_datasets` | MP3/FLAC → WAV conversion |
| `tqdm` | `download_datasets` | Progress bars |
