"""
text2phoneme.py
Đọc manifest.jsonl, chuyển trường "text" → chuỗi âm vị IPA,
rồi ghi lại file manifest với trường "phonemes" được bổ sung.

Sử dụng:
    python text2phoneme.py --manifest data/manifest.jsonl
    python text2phoneme.py --manifest data/manifest.jsonl --mode vi --output data/manifest_ph.jsonl
    python text2phoneme.py --manifest data/manifest.jsonl --dry_run

Input (mỗi dòng manifest.jsonl):
    {"audio": "clip001.wav", "text": "xin chào bạn"}

Output (ghi đè hoặc file mới):
    {"audio": "clip001.wav", "text": "xin chào bạn",
     "phonemes": ["s","i","n","-1","$","tʃ","a","-2","$","b","a","-5","n"]}

Chế độ (--mode):
    auto      Tự phát hiện ngôn ngữ (mặc định)
    vi        Tiếng Việt thuần
    vietlish  Tiếng Anh đọc kiểu Việt
    iev       Code-switching (xen kẽ VN + EN)
    en        Tiếng Anh native (bảng đơn giản)
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

from phoneme_set import SPECIAL_TOKENS, VIETLISH_MAP

# ───────────────────────────────────────────────────────────────────────────────
#  BẢNG CHUYỂN ĐỔI
# ───────────────────────────────────────────────────────────────────────────────

_TONE_MAP = {
    "COMBINING GRAVE ACCENT": "-2",
    "COMBINING HOOK ABOVE":   "-3",
    "COMBINING ACUTE ACCENT": "-4",
    "COMBINING DOT BELOW":    "-5",
    "COMBINING TILDE":        "-6",
}
_TONE_RE = re.compile(r"[\u0300\u0301\u0303\u0309\u0323]")

_VN_INITIAL_MAP = {
    "ngh":"ŋ","gh":"g","gi":"z","ch":"tʃ","kh":"x","ng":"ŋ","nh":"ɲ",
    "ph":"f","qu":"k","th":"tʰ","tr":"tʃ","b":"b","c":"k","d":"z",
    "đ":"d","g":"g","h":"h","k":"k","l":"l","m":"m","n":"n","p":"p",
    "r":"r","s":"s","t":"t","v":"v","x":"s",
}

_VN_RHYME_MAP = {
    "uyên":["w","i","e","n"],"uynh":["w","i","ŋ"],
    "ươn":["ɯ","o","n"],"ương":["ɯ","o","ŋ"],"ươc":["ɯ","o","k"],"ươm":["ɯ","o","m"],"ươp":["ɯ","o","p"],
    "ươi":["ɯ","o","j"],"ươt":["ɯ","o","t"],"ươu":["ɯ","o","w"],
    "uôn":["u","o","n"],"uông":["u","o","ŋ"],"uôt":["u","o","t"],"uôc":["u","o","k"],
    "uôm":["u","o","m"],"uôp":["u","o","p"],"uôi":["u","o","j"],
    "iêng":["i","e","ŋ"],"iêm":["i","e","m"],"iên":["i","e","n"],"yên":["i","e","n"],"yêng":["i","e","ŋ"],"yêm":["i","e","m"],
    "iêp":["i","e","p"],"iêt":["i","e","t"],"iêc":["i","e","k"],"yêt":["i","e","t"],
    "oan":["w","a","n"],"oăn":["w","a","n"],"oen":["w","ɛ","n"],
    "oang":["w","a","ŋ"],"oat":["w","a","t"],"oac":["w","a","k"],
    "uân":["w","ɤ","n"],"uât":["w","ɤ","t"],"uyn":["w","i","n"],
    "anh":["a","ŋ"],"ênh":["e","ŋ"],"inh":["i","ŋ"],
    "ach":["a","k"],"êch":["e","k"],"ich":["i","k"],
    "ang":["a","ŋ"],"ăng":["a","ŋ"],"âng":["ɤ","ŋ"],
    "eng":["ɛ","ŋ"],"êng":["e","ŋ"],"ing":["i","ŋ"],
    "ong":["ɔ","ŋ"],"ông":["o","ŋ"],"ơng":["ɤ","ŋ"],"oong":["o","o","ŋ"],
    "ung":["u","ŋ"],"ưng":["ɯ","ŋ"],
    "an":["a","n"],"ăn":["a","n"],"ân":["ɤ","n"],
    "en":["ɛ","n"],"ên":["e","n"],"in":["i","n"],
    "on":["ɔ","n"],"ôn":["o","n"],"ơn":["ɤ","n"],
    "un":["u","n"],"ưn":["ɯ","n"],
    "am":["a","m"],"ăm":["a","m"],"âm":["ɤ","m"],
    "em":["ɛ","m"],"êm":["e","m"],"im":["i","m"],
    "om":["ɔ","m"],"ôm":["o","m"],"ơm":["ɤ","m"],
    "um":["u","m"],"ưm":["ɯ","m"],
    "at":["a","t"],"ăt":["a","t"],"ât":["ɤ","t"],
    "et":["ɛ","t"],"êt":["e","t"],"it":["i","t"],
    "ot":["ɔ","t"],"ôt":["o","t"],"ơt":["ɤ","t"],
    "ut":["u","t"],"ưt":["ɯ","t"],
    "ac":["a","k"],"ăc":["a","k"],"âc":["ɤ","k"],
    "ec":["ɛ","k"],"oc":["ɔ","k"],"ôc":["o","k"],
    "uc":["u","k"],"ưc":["ɯ","k"],
    "ap":["a","p"],"ăp":["a","p"],"âp":["ɤ","p"],
    "ep":["ɛ","p"],"êp":["e","p"],"ip":["i","p"],
    "op":["ɔ","p"],"ôp":["o","p"],"ưp":["ɯ","p"],"up":["u","p"],
    "ôi":["o","j"],"ơi":["ɤ","j"],"ai":["a","j"],
    "ay":["a","j"],"ây":["ɤ","j"],"oi":["ɔ","j"],
    "ui":["u","j"],"uy":["w","i"],"iu":["i","w"],
    "ao":["a","w"],"âu":["ɤ","w"],"au":["a","w"],
    "eo":["ɛ","w"],"êu":["e","w"],"ưu":["ɯ","w"],
    "oa":["w","a"],"oă":["w","a"],"oe":["w","ɛ"],
    "ia":["i","a"],"iê":["i","e"],"ie":["i","e"],
    "ua":["u","a"],"uô":["u","o"],"uo":["u","o"],
    "iêu": ["i", "e", "w"], "yêu": ["i", "e", "w"], "ơp": ["ɤ", "p"], "ooc": ["ɔ", "k"],
    "oai": ["w", "a", "j"], "oay": ["w", "a", "j"], "uây": ["w", "ɤ", "j"],
    "oao": ["w", "a", "w"], "oeo": ["w", "ɛ", "w"], "uyu": ["w", "i", "w"],
    "oam": ["w", "a", "m"], "oăm": ["w", "a", "m"], "oanh": ["w", "a", "ŋ"],
    "uên": ["w", "e", "n"], "uênh": ["w", "e", "ŋ"], "uâng": ["w", "ɤ", "ŋ"],
    "oap": ["w", "a", "p"], "oăp": ["w", "a", "p"], "oach": ["w", "a", "k"],
    "oet": ["w", "ɛ", "t"], "oăc": ["w", "a", "k"], "uêch": ["w", "e", "k"],
    "uêt": ["w", "e", "t"], "uyêt": ["w", "i", "e", "t"], "uych": ["w", "i", "k"],
    "uyt": ["w", "i", "t"], "uyp": ["w", "i", "p"],
    "oăng": ["w", "a", "ŋ"], "oăt": ["w", "a", "t"], "uâc": ["w", "ɤ", "k"], 
    "oep": ["w", "ɛ", "p"], "uêu": ["w", "e"],
    "uê": ["w", "e"], "uya": ["w", "i", "a"],
    "ưa":["ɯ","a"],"ươ":["ɯ","o"],"uơ":["ɯ","o"],
    "a":["a"],"ă":["a"],"â":["ɤ"],"e":["ɛ"],"ê":["e"],"i":["i"],
    "o":["ɔ"],"ô":["o"],"ơ":["ɤ"],"u":["u"],"ư":["ɯ"],"y":["i"],
}

_VN_CHARS = set(
    "àáảãạăắặẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
)
_VN_COMMON = {
    "tôi","em","anh","chị","bạn","họ","mình","chúng","các","và","là","có",
    "không","được","với","cho","của","trong","từ","khi","thì","mà","để",
    "hay","hoặc","nhưng","vì","nếu","đã","sẽ","đang","nhà","xe",
}

# ───────────────────────────────────────────────────────────────────────────────
#  VIETNAMESE RULE-BASED G2P
# ───────────────────────────────────────────────────────────────────────────────

def _get_tone(nfd: str) -> str:
    for ch in nfd:
        t = _TONE_MAP.get(unicodedata.name(ch, ""))
        if t:
            return t
    return "-1"

def _remove_tone(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC",
           "".join(ch for ch in nfd if not _TONE_RE.match(ch)))

def vn_syllable_to_phonemes(syllable: str) -> list[str]:
    syllable = syllable.strip().lower()
    if not syllable:
        return []
    tone = _get_tone(unicodedata.normalize("NFD", syllable))
    base = _remove_tone(syllable)
    phonemes: list[str] = []
    remainder = base

    for init in sorted(_VN_INITIAL_MAP, key=len, reverse=True):
        if base.startswith(init):
            ph = _VN_INITIAL_MAP[init]
            remainder = base[len(init):]
            if init == "qu":
                phonemes += [ph, "w"]
                remainder = remainder.lstrip("u")
            else:
                phonemes.append(ph)
            break

    rhyme_ph = None
    for rhyme in sorted(_VN_RHYME_MAP, key=len, reverse=True):
        if remainder == rhyme or remainder.startswith(rhyme):
            rhyme_ph = _VN_RHYME_MAP[rhyme]
            break

    if rhyme_ph:
        phonemes += rhyme_ph
    else:
        for ch in remainder:
            phonemes += _VN_RHYME_MAP.get(ch, [ch if ch != "đ" else "d"])

    phonemes.append(tone)
    return phonemes

def text_to_graphemes(text: str) -> list[str]:
    """
    Chuyển văn bản thành danh sách các âm tiết tiếng Việt (grapheme).
    Nếu gặp từ tiếng Anh có trong VIETLISH_MAP, tách thành các âm tiết Vietlish.
    """
    words = [re.sub(r"[^\w\u00C0-\u1EF9]","",w) for w in text.strip().lower().split()]
    words = [w for w in words if w]
    graphemes = []
    for w in words:
        if w in VIETLISH_MAP:
            graphemes.extend(VIETLISH_MAP[w][0])
        else:
            graphemes.append(w)
    return graphemes

def graphemes_to_phonemes(graphemes: list[str]) -> list[str]:
    """
    Chuyển danh sách grapheme (âm tiết tiếng Việt) thành chuỗi phoneme.
    """
    sep = SPECIAL_TOKENS.get("VN_SEP", "$")
    phonemes = []
    for i, g in enumerate(graphemes):
        phonemes.extend(vn_syllable_to_phonemes(g))
        if i < len(graphemes) - 1:
            phonemes.append(sep)
    return phonemes

def vn_text_to_phonemes(text: str) -> list[str]:
    return graphemes_to_phonemes(text_to_graphemes(text))

# ───────────────────────────────────────────────────────────────────────────────
#  VIETLISH / EN / IEV (Tất cả đều quy về 1 luồng grapheme -> phoneme)
# ───────────────────────────────────────────────────────────────────────────────

def vietlish_word_to_phonemes(word: str) -> list[str]:
    return graphemes_to_phonemes(text_to_graphemes(word))

def vietlish_text_to_phonemes(text: str) -> list[str]:
    return graphemes_to_phonemes(text_to_graphemes(text))

def iev_text_to_phonemes(text: str) -> list[str]:
    return graphemes_to_phonemes(text_to_graphemes(text))

def en_text_to_phonemes(text: str) -> list[str]:
    return graphemes_to_phonemes(text_to_graphemes(text))

def _is_vn(w: str) -> bool:
    return any(c in _VN_CHARS for c in w) or w.lower() in _VN_COMMON

def _is_en(w: str) -> bool:
    return bool(re.match(r"^[a-zA-Z]+$", w)) and w.lower() not in _VN_COMMON

# ───────────────────────────────────────────────────────────────────────────────
#  AUTO-DETECT + DISPATCHER
# ───────────────────────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    words = text.strip().split()
    if not words:
        return "vi"
    vn = sum(1 for w in words if _is_vn(w))
    en = sum(1 for w in words if _is_en(w))
    t  = len(words)
    if vn/t >= 0.8:
        return "vi"
    if en/t >= 0.8:
        return "vietlish" if any(w.lower() in VIETLISH_MAP for w in words) else "en"
    return "iev"

Mode = Literal["auto","vi","en","vietlish","iev"]

def text_to_phoneme(text: str, mode: Mode = "auto") -> list[str]:
    """
    Hàm chính: chuyển text → chuỗi phoneme IPA.

    Args:
        text: Văn bản đầu vào.
        mode: "auto" | "vi" | "en" | "vietlish" | "iev"

    Returns:
        Danh sách phoneme string.

    Ví dụ:
        text_to_phonemes("xin chào")   → ["s","i","n","-1","$","tʃ","a","-2"]
        text_to_phonemes("inbox")      → ["ɪ","n","$","b","o","-4","k"]
    """
    text = text.strip()
    if not text:
        return []
    if mode == "auto":
        mode = detect_language(text)
    return {
        "vi":       vn_text_to_phonemes,
        "en":       en_text_to_phonemes,
        "vietlish": vietlish_text_to_phonemes,
        "iev":      iev_text_to_phonemes,
    }.get(mode, vn_text_to_phonemes)(text)

# ───────────────────────────────────────────────────────────────────────────────
#  MANIFEST PROCESSOR
# ───────────────────────────────────────────────────────────────────────────────

def process_manifest(
    manifest_path: str,
    output_path:   str | None = None,
    mode:          Mode       = "auto",
    text_key:      str        = "text",
    verbose:       bool       = True,
) -> None:
    """
    Đọc manifest.jsonl, thêm trường "phonemes" vào mỗi record, rồi ghi lại.

    Args:
        manifest_path: Đường dẫn file manifest.jsonl.
        output_path:   Nếu None → ghi đè file gốc.
        mode:          Chế độ chuyển đổi ("auto" | "vi" | "en" | "vietlish" | "iev").
        text_key:      Tên trường văn bản (mặc định: "text").
        verbose:       In tiến trình.

    Ghi chú:
        - Nếu record đã có trường "phonemes" → sẽ bị ghi đè.
        - Record không có trường text_key → bỏ qua (giữ nguyên record).
    """
    src = Path(manifest_path)
    dst = Path(output_path) if output_path else src

    if not src.exists():
        raise FileNotFoundError(f"Không tìm thấy: {src}")

    records: list[dict] = []
    with src.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Dòng {lineno} lỗi JSON, bỏ qua: {e}")

    if verbose:
        print(f"[text2phoneme] Đọc {len(records)} records từ '{src}'")

    n_updated, n_skipped = 0, 0
    for i, rec in enumerate(records):
        text = rec.get(text_key, "").strip()
        if not text:
            n_skipped += 1
            continue

        rec_mode = detect_language(text) if mode == "auto" else mode
        rec["phonemes"] = text_to_phonemes(text, mode=rec_mode)
        n_updated += 1

        if verbose and (i+1) % 500 == 0:
            print(f"  ... {i+1}/{len(records)} records")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if verbose:
        action = "ghi đè file gốc" if dst == src else f"ghi ra '{dst}'"
        print(f"[text2phoneme] {n_updated} records đã thêm phonemes, "
              f"{n_skipped} bỏ qua. {action}.")

# ───────────────────────────────────────────────────────────────────────────────
#  CLI + DEMO
# ───────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Thêm trường 'phonemes' vào manifest.jsonl"
    )
    parser.add_argument("--manifest",  required=True, help="Đường dẫn manifest.jsonl")
    parser.add_argument("--output",    default=None,  help="File output (mặc định: ghi đè)")
    parser.add_argument("--mode",      default="auto",
                        choices=["auto","vi","en","vietlish","iev"])
    parser.add_argument("--text_key",  default="text", help="Tên trường văn bản")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Chỉ in kết quả 5 dòng đầu, không ghi file")
    args = parser.parse_args()

    if args.dry_run:
        src = Path(args.manifest)
        print(f"[dry_run] {src}\n")
        with src.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5: break
                rec  = json.loads(line)
                text = rec.get(args.text_key, "")
                ph   = text_to_phonemes(text, mode=args.mode)
                lang = detect_language(text) if args.mode == "auto" else args.mode
                print(f"  [{i+1}] text     : {text}")
                print(f"       lang     : {lang}")
                print(f"       phonemes : {ph}\n")
        return

    process_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        mode=args.mode,
        text_key=args.text_key,
        verbose=True,
    )


def _demo():
    print("="*60)
    print("   text_to_phonemes  —  Demo")
    print("="*60)
    cases = [
        ("xin chào",                    "vi",       "Tiếng Việt"),
        ("tôi đang đi học",             "vi",       "Câu VN"),
        ("say",                         "en",       "Vietlish"),
        ("xấy",                         "vi",       "Câu VN"),
        ("inbox",                       "vietlish", "Vietlish"),
        ("message coffee",              "vietlish", "Vietlish nhiều từ"),
        ("anh đang dùng laptop ở nhà",  "iev",      "Code-switching"),
        ("mình sẽ check email ngay",    "auto",     "Auto-detect"),
    ]
    for text, mode, desc in cases:
        ph   = text_to_phonemes(text, mode=mode)
        lang = detect_language(text) if mode == "auto" else mode
        print(f"\n[{desc}]")
        print(f"  input   : {text!r}")
        print(f"  mode    : {lang}")
        print(f"  phonemes: {ph}")

    # Demo manifest processor
    import tempfile, os
    sample = [
        {"audio":"clip001.wav","text":"xin chào bạn"},
        {"audio":"clip002.wav","text":"mình sẽ check email ngay"},
        {"audio":"clip003.wav","text":"inbox coffee"},
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        for r in sample: tmp.write(json.dumps(r, ensure_ascii=False)+"\n")
        tmp_path = tmp.name

    print(f"\n{'='*60}")
    print("   process_manifest  —  Demo")
    print(f"{'='*60}")
    process_manifest(tmp_path, verbose=True)
    with open(tmp_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            print(f"\n  audio   : {r['audio']}")
            print(f"  text    : {r['text']}")
            print(f"  phonemes: {r['phonemes']}")
    os.unlink(tmp_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        _cli()
    else:
        _demo()
