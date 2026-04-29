"""
text2phoneme.py
Đọc manifest.jsonl, chuyển trường "text" → chuỗi âm vị Latin (cho 2 giọng Bắc/Nam),
rồi ghi lại file manifest với trường "phonemes_north" và "phonemes_south" được bổ sung.

Sử dụng:
    python text2phoneme.py --manifest data/manifest.jsonl
    python text2phoneme.py --manifest data/manifest.jsonl --mode vi --output data/manifest_ph.jsonl
    python text2phoneme.py --manifest data/manifest.jsonl --dry_run

Input (mỗi dòng manifest.jsonl):
    {"audio": "clip001.wav", "text": "xin chào bạn"}

Output:
    {"audio": "clip001.wav", "text": "xin chào bạn",
     "phonemes_north": ["x","i","n","t1","$","ch","a","w","t2","$","b","a","n","t5"],
     "phonemes_south": ["s","i","n","t1","$","ch","a","w","t2","$","b","a","n","t5"]}
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

from phoneme_set import SPECIAL_TOKENS, VIETLISH_MAP
from vietnormalizer import is_vietnamese_word, transliterate_word
from vietnamesenormalizer import VietnameseNormalizer

_normalizer = VietnameseNormalizer()

# ───────────────────────────────────────────────────────────────────────────────
#  BẢNG CHUYỂN ĐỔI (LATIN MATCHING)
# ───────────────────────────────────────────────────────────────────────────────

_TONE_MAP = {
    "COMBINING GRAVE ACCENT": "t2", # huyền
    "COMBINING HOOK ABOVE":   "t3", # hỏi
    "COMBINING ACUTE ACCENT": "t4", # sắc
    "COMBINING DOT BELOW":    "t5", # nặng
    "COMBINING TILDE":        "t6", # ngã
}
_TONE_RE = re.compile(r"[\u0300\u0301\u0303\u0309\u0323]")

# Rhymes in Latin format
_VN_RHYME_MAP_LATIN = {
    "uyên":["u","ie","n"],"uynh":["u","i","nh"],
    "ươn":["uow","n"],"ương":["uow","ng"],"ươc":["uow","c"],"ươm":["uow","m"],"ươp":["uow","p"],
    "ươi":["uow","i"],"ươt":["uow","t"],"ươu":["uow","u"],
    "uôn":["uo","n"],"uông":["uo","ng"],"uôt":["uo","t"],"uôc":["uo","c"],
    "uôm":["uo","m"],"uôp":["uo","p"],"uôi":["uo","i"],
    "iêng":["ie","ng"],"iêm":["ie","m"],"iên":["ie","n"],"yên":["ie","n"],"yêng":["ie","ng"],"yêm":["ie","m"],
    "iêp":["ie","p"],"iêt":["ie","t"],"iêc":["ie","c"],"yêt":["ie","t"],
    # ─── oa/oe rhymes (medial = 'o') ───
    "oăng":["o","aw","ng"],"oăt":["o","aw","t"],"oăc":["o","aw","c"],"oăp":["o","aw","p"],
    "oăm":["o","aw","m"],"oăn":["o","aw","n"],
    "oanh":["o","a","nh"],"oang":["o","a","ng"],"oam":["o","a","m"],
    "oan":["o","a","n"],"oay":["o","aw","y"],"oai":["o","a","i"],
    "oat":["o","a","t"],"oac":["o","a","c"],"oap":["o","a","p"],
    "oao":["o","a","o"],"oeo":["o","e","o"],"oet":["o","e","t"],"oep":["o","e","p"],
    "oach":["o","a","c"],
    "ooc":["o","o","c"],"oong":["o","o","ng"],
    "oa":["o","a"],"oă":["o","aw"],"oe":["o","e"],
    # ─── u-prefix rhymes (medial = 'u') ───
    "uâng":["u","aa","ng"],"uâc":["u","aa","c"],
    "uân":["u","aa","n"],"uât":["u","aa","t"],
    "uây":["u","aa","y"],
    "uênh":["u","ee","nh"],"uêch":["u","ee","c"],"uêt":["u","ee","t"],
    "uên":["u","ee","n"],"uêu":["u","ee","w"],
    "uê":["u","ee"],
    "uyêt":["u","ie","t"],"uych":["u","i","c"],
    "uyt":["u","i","t"],"uyp":["u","i","p"],"uyn":["u","i","n"],
    "uyu":["u","i","w"],"uya":["u","ie","a"],
    "uy":["u","i"],
    # ─── -anh/-inh/-ênh (coda nh) ───
    "anh":["a","nh"],"ênh":["ee","nh"],"inh":["i","nh"],
    "ach":["a","c"],"êch":["ee","c"],"ich":["i","c"],
    # ─── -ng codas ───
    "ang":["a","ng"],"ăng":["aw","ng"],"âng":["aa","ng"],
    "eng":["e","ng"],"êng":["ee","ng"],"ing":["i","ng"],
    "ong":["o","ng"],"ông":["oo","ng"],"ơng":["ow","ng"],"oong":["o","o","ng"],
    "ung":["u","ng"],"ưng":["uw","ng"],
    # ─── -n codas ───
    "an":["a","n"],"ăn":["aw","n"],"ân":["aa","n"],
    "en":["e","n"],"ên":["ee","n"],"in":["i","n"],
    "on":["o","n"],"ôn":["oo","n"],"ơn":["ow","n"],
    "un":["u","n"],"ưn":["uw","n"],
    # ─── -m codas ───
    "am":["a","m"],"ăm":["aw","m"],"âm":["aa","m"],
    "em":["e","m"],"êm":["ee","m"],"im":["i","m"],
    "om":["o","m"],"ôm":["oo","m"],"ơm":["ow","m"],
    "um":["u","m"],"ưm":["uw","m"],
    # ─── -t codas ───
    "at":["a","t"],"ăt":["aw","t"],"ât":["aa","t"],
    "et":["e","t"],"êt":["ee","t"],"it":["i","t"],
    "ot":["o","t"],"ôt":["oo","t"],"ơt":["ow","t"],
    "ut":["u","t"],"ưt":["uw","t"],
    # ─── -c codas ───
    "ac":["a","c"],"ăc":["aw","c"],"âc":["aa","c"],
    "ec":["e","c"],"oc":["o","c"],"ôc":["oo","c"],
    "uc":["u","c"],"ưc":["uw","c"],
    # ─── -p codas ───
    "ap":["a","p"],"ăp":["aw","p"],"âp":["aa","p"],
    "ep":["e","p"],"êp":["ee","p"],"ip":["i","p"],
    "op":["o","p"],"ôp":["oo","p"],"ưp":["uw","p"],"up":["u","p"],"ơp":["ow","p"],
    # ─── -y/-w glides ───
    "ôi":["oo","y"],"ơi":["ow","y"],"ai":["a","y"],
    "ay":["a","y"],"ây":["aa","y"],"oi":["o","y"],
    "ui":["u","y"],"iu":["i","u"],
    "ao":["a","o"],"âu":["aa","u"],"au":(["a","u"], ["a","o"], ["aa","u"]),
    "eo":["e","o"],"êu":["ee","u"],"ưu":["uwu"],
    # ─── diphthongs ───
    "iêu":["ie","u"],"yêu":["ie","u"],
    "ia":["ia"],"iê":["ie"],"ie":["ie"],
    "ua":["ua"],"uô":["uo"],"uo":["uo"],
    "ưa":["uwa"],"ươ":["uow"],
    # ─── singles ───
    "a":["a"],"ă":["aw"],"â":["aa"],"e":["e"],"ê":["ee"],"i":["i"],
    "o":["o"],"ô":["oo"],"ơ":["ow"],"u":["u"],"ư":["uw"],"y":["i"],
}

_VN_RHYME_MAP_LATIN = dict(sorted(_VN_RHYME_MAP_LATIN.items(), key=lambda x: len(x[0]), reverse=True))

# ───────────────────────────────────────────────────────────────────────────────
#  VIETNAMESE RULE-BASED G2P (MULTI-DIALECT)
# ───────────────────────────────────────────────────────────────────────────────

def _get_tone(nfd: str) -> str:
    for ch in nfd:
        t = _TONE_MAP.get(unicodedata.name(ch, ""))
        if t: return t
    return "t1"

def _remove_tone(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(ch for ch in nfd if not _TONE_RE.match(ch)))

def vn_syllable_to_phonemes(syllable: str) -> dict[str, list[str]]:
    """Returns {"north": [...], "south": [...]}"""
    syllable = syllable.strip().lower()
    if not syllable:
        return {"north": [], "south": []}
        
    nfd = unicodedata.normalize("NFD", syllable)
    tone = _get_tone(nfd)
    base = unicodedata.normalize("NFC", _remove_tone(syllable))

    prefixes = ["ngh", "gi", "ch", "kh", "ng", "nh", "ph", "qu", "th", "tr", "b", "c", "d", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x"]
    
    init = ""
    for p in prefixes:
        if base.startswith(p):
            init = p
            base = base[len(p):]
            break

    # Dialect mapping for initials
    north_init = ""
    south_init = ""
    
    if init == "gi" or init == "d":
        north_init, south_init = "z", "j"
    elif init == "r":
        north_init, south_init = "z", "r"
    elif init == "tr":
        north_init, south_init = "ch", "tr"
    elif init == "s":
        north_init, south_init = "x", "s"
    elif init == "v":
        north_init, south_init = "v", "j"
    elif init == "đ":
        north_init = south_init = "d"
    elif init == "c" or init == "k":
        north_init = south_init = "c"
    elif init == "g" or init == "gh":
        north_init = south_init = "g"
    elif init == "ng" or init == "ngh":
        north_init = south_init = "ng"
    elif init == "qu":
        north_init = south_init = "qu"
    elif init:
        north_init = south_init = init
    else:
        north_init = south_init = ""

    north_ph: list[str] = []
    south_ph: list[str] = []
    for r, ph in _VN_RHYME_MAP_LATIN.items():
        if base == r or base.startswith(r):
            if isinstance(ph, tuple):
                north_ph = ([north_init] if north_init else []) + list(ph[0]) + [tone]
                south_ph = ([south_init] if south_init else []) + list(ph[1] if len(ph) > 1 else ph[0]) + [tone]
            else:
                north_ph = ([north_init] if north_init else []) + ph + [tone]
                south_ph = ([south_init] if south_init else []) + ph + [tone]
            break

    if not north_ph:
        fallback = list(base) if base else []
        north_ph = ([north_init] if north_init else []) + fallback + [tone]
        south_ph = ([south_init] if south_init else []) + fallback + [tone]

    return {
        "north": north_ph,
        "south": south_ph
    }

def text_to_graphemes(text: str) -> list[str]:
    """Tách văn bản thành danh sách graphemes (âm tiết VN hoặc Vietlish)."""
    words = [re.sub(r"[^\w\u00C0-\u1EF9]","",w) for w in text.strip().lower().split()]
    words = [w for w in words if w]
    graphemes = []
    for w in words:
        if w in VIETLISH_MAP:
            graphemes.extend(VIETLISH_MAP[w][0])
        elif not is_vietnamese_word(w) and re.match(r"^[a-z]+$", w):
            # Transliterate english word
            transliterated = transliterate_word(w).replace("-", " ")
            graphemes.extend(transliterated.split())
        else:
            graphemes.append(w)
    return graphemes

def graphemes_to_phonemes_dict(graphemes: list[str]) -> dict[str, list[str]]:
    """Chuyển đổi danh sách graphemes thành chuỗi phoneme cho mỗi dialect."""
    sep = SPECIAL_TOKENS.get("VN_SEP", "$")
    out_north = []
    out_south = []
    
    for i, g in enumerate(graphemes):
        ph_dict = vn_syllable_to_phonemes(g)
        out_north.extend(ph_dict["north"])
        out_south.extend(ph_dict["south"])
        
        if i < len(graphemes) - 1:
            out_north.append(sep)
            out_south.append(sep)
            
    return {"north": out_north, "south": out_south}

def vn_text_to_phonemes(text: str) -> dict[str, list[str]]:
    return graphemes_to_phonemes_dict(text_to_graphemes(text))

def vietlish_text_to_phonemes(text: str) -> dict[str, list[str]]:
    return graphemes_to_phonemes_dict(text_to_graphemes(text))

def iev_text_to_phonemes(text: str) -> dict[str, list[str]]:
    return graphemes_to_phonemes_dict(text_to_graphemes(text))

def en_text_to_phonemes(text: str) -> dict[str, list[str]]:
    return graphemes_to_phonemes_dict(text_to_graphemes(text))

def _is_vn(w: str) -> bool:
    return is_vietnamese_word(w)

def _is_en(w: str) -> bool:
    return not is_vietnamese_word(w) and bool(re.match(r"^[a-zA-Z]+$", w))

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

def text_to_phoneme(text: str, mode: Mode = "auto") -> dict[str, list[str]]:
    """
    Hàm chính: chuyển text → chuỗi phoneme LATIN theo dialect.
    """
    text = text.strip()
    if not text:
        return {"north": [], "south": []}
    if mode == "auto":
        mode = detect_language(text)
        
    dispatcher = {
        "vi":       vn_text_to_phonemes,
        "en":       en_text_to_phonemes,
        "vietlish": vietlish_text_to_phonemes,
        "iev":      iev_text_to_phonemes,
    }
    return dispatcher.get(mode, vn_text_to_phonemes)(text)

# ───────────────────────────────────────────────────────────────────────────────
#  MANIFEST PROCESSOR
# ───────────────────────────────────────────────────────────────/────────────────

def process_manifest(
    manifest_path: str,
    output_path:   str | None = None,
    mode:          Mode       = "auto",
    text_key:      str        = "text",
    verbose:       bool       = True,
) -> None:
    src = Path(manifest_path)
    dst = Path(output_path) if output_path else src.parent / "manifest_all.jsonl"

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
                print(f"  [WARN] Dong {lineno} loi JSON, bo qua: {e}")

    if verbose:
        print(f"[text2phoneme] Doc {len(records)} records tu '{src}'")

    out_records: list[dict] = []
    n_updated, n_skipped = 0, 0
    for i, rec in enumerate(records):
        raw_text = rec.get(text_key, "").strip()
        if not raw_text:
            n_skipped += 1
            continue

        # 1. Chuẩn hoá văn bản (số → chữ, ngày tháng, viết tắt, v.v.)
        text = _normalizer.normalize(raw_text)

        rec_mode = detect_language(text) if mode == "auto" else mode
        ph_dict = text_to_phoneme(text, mode=rec_mode)
        
        # Giữ lại văn bản gốc, lưu văn bản đã chuẩn hoá
        rec["text_raw"] = raw_text
        rec[text_key]   = text

        # Thêm output transcript đã tách grapheme
        rec["normalized_text"] = " ".join(text_to_graphemes(text))
        
        # Thêm output 2 dialect
        rec["phonemes_north"] = ph_dict["north"]
        rec["phonemes_south"] = ph_dict["south"]
        
        # Loại bỏ trường phonemes cũ nếu tồn tại để tránh nhầm lẫn
        if "phonemes" in rec:
            del rec["phonemes"]
            
        n_updated += 1

        if verbose and (i+1) % 500 == 0:
            print(f"  ... {i+1}/{len(records)} records")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if verbose:
        action = "ghi đè file gốc" if dst == src else f"ghi ra '{dst}'"
        print(f"[text2phoneme] {n_updated} records đã thêm dialects phonemes, "
              f"{n_skipped} bỏ qua. {action}.")

# ───────────────────────────────────────────────────────────────────────────────
#  CLI + DEMO
# ───────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Thêm trường 'phonemes_north' và 'phonemes_south' vào manifest.jsonl")
    parser.add_argument("--manifest",  required=True, help="Đường dẫn manifest.jsonl")
    parser.add_argument("--output",    default=None,  help="File output (mặc định: ghi đè)")
    parser.add_argument("--mode",      default="auto", choices=["auto","vi","en","vietlish","iev"])
    parser.add_argument("--text_key",  default="text", help="Tên trường văn bản")
    parser.add_argument("--dry_run",   action="store_true", help="Chỉ in kết quả 5 dòng đầu, không ghi file")
    args = parser.parse_args()

    if args.dry_run:
        src = Path(args.manifest)
        print(f"[dry_run] {src}\n")
        with src.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5: break
                rec  = json.loads(line)
                text = rec.get(args.text_key, "")
                norm = " ".join(text_to_graphemes(text))
                ph   = text_to_phoneme(text, mode=args.mode)
                lang = detect_language(text) if args.mode == "auto" else args.mode
                print(f"  [{i+1}] text     : {text}")
                print(f"       lang     : {lang}")
                print(f"       norm     : {norm}")
                print(f"       ph_north : {ph['north']}")
                print(f"       ph_south : {ph['south']}\n")
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
    print("   text_to_phoneme  —  Demo (Multi-Dialect)")
    print("="*60)
    cases = [
        ("quốc",                        "vi",       "Từ có 'qu' đầu"),
        ("vào",                         "vi",       "Phương ngữ Bắc/Nam (v)"),
        ("gió",                         "vi",       "Phương ngữ Bắc/Nam (gi)"),
        ("trời",                        "vi",       "Phương ngữ Bắc/Nam (tr)"),
        ("xin chào",                    "vi",       "Tiếng Việt cơ bản"),
        ("inbox",                       "vietlish", "Vietlish"),
        ("anh đang dùng laptop ở nhà",  "iev",      "Code-switching"),
    ]
    for text, mode, desc in cases:
        norm = " ".join(text_to_graphemes(text))
        ph   = text_to_phoneme(text, mode=mode)
        lang = detect_language(text) if mode == "auto" else mode
        print(f"\n[{desc}]")
        print(f"  input   : {text!r}")
        print(f"  norm    : {norm!r}")
        print(f"  mode    : {lang}")
        print(f"  north   : {ph['north']}")
        print(f"  south   : {ph['south']}")

    import tempfile, os
    sample = [
        {"audio":"clip001.wav","text":"quà và đùa"},
        {"audio":"clip002.wav","text":"mình sẽ check email ngay"},
        {"audio":"clip003.wav","text":"we need to test the algorithm and transformer"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for r in sample: tmp.write(json.dumps(r, ensure_ascii=False)+"\n")
        tmp_path = tmp.name

    print(f"\n{'='*60}")
    print("   process_manifest  —  Demo")
    print(f"{'='*60}")
    process_manifest(tmp_path, verbose=True)
    with open(tmp_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            print(f"\n  audio    : {r['audio']}")
            print(f"  text     : {r['text']}")
            print(f"  norm     : {r.get('normalized_text')}")
            print(f"  ph_north : {r.get('phonemes_north')}")
            print(f"  ph_south : {r.get('phonemes_south')}")
    os.unlink(tmp_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        _cli()
    else:
        _demo()
