# ═══════════════════════════════════════════════════════════════════════════════
#  SPECIAL TOKENS
# ═══════════════════════════════════════════════════════════════════════════════

SPECIAL_TOKENS = {
    "PAD":    "<pad>",   # 0  — padding
    "UNK":    "<unk>",   # 1  — unknown phoneme
    "SOT":    "<sot>",   # 2  — start of transcript
    "EOT":    "<eot>",   # 3  — end of transcript
    "VN_SEP": "$",       # 4  — Vietnamese syllable boundary
}

# ═══════════════════════════════════════════════════════════════════════════════
#  VIETNAMESE PHONEMES (LATIN)
# ═══════════════════════════════════════════════════════════════════════════════

# Âm đầu (Initials)
VN_INITIALS = [
    "b",    # b
    "c",    # c, k
    "ch",   # ch / tr 
    "tr",   # tr 
    "d",    # đ
    "z",    # d, gi, r 
    "j",    # d, gi, v
    "g",    # g, gh
    "h",    # h
    "kh",   # kh
    "l",    # l
    "m",    # m
    "n",    # n
    "ng",   # ng, ngh
    "nh",   # nh
    "p",    # p
    "ph",   # ph
    "r",    # r 
    "s",    # s 
    "x",    # x / s 
    "t",    # t
    "th",   # th
    "v",    # v 
    "qu",    # qu
]

# Âm đệm (Medials)
VN_MEDIALS = [
    "u",    # u-
    "o",    # o-
]

# Nguyên âm / Hạt nhân (Nuclei)
VN_NUCLEI = [
    "a",    # a
    "aw",   # ă
    "aa",   # â
    "e",    # e
    "ee",   # ê
    "i",    # i, y
    "o",    # o
    "oo",   # ô
    "ow",   # ơ
    "u",    # u
    "uw",   # ư
    "ie",   # ia, iê, ya, yê
    "ua",   # ua
    "uo",   # uo, uô
    "uwa",  # ưa
    "uow",  # ươ
]

# Âm cuối (Codas)
VN_CODAS = [
    "p",    # -p
    "t",    # -t
    "c",    # -c / -ch
    "m",    # -m
    "n",    # -n
    "ng",   # -ng
    "nh",   # -nh
    "y",    # -i / -y glide
    "u",    # -u glide
    "o",    # -o glide
]

# Thanh điệu (Tones) — 6 tones
VN_TONES = [
    "t1",   # ngang  (level, mid)
    "t2",   # huyền  (falling)
    "t3",   # hỏi    (dipping-rising)
    "t4",   # sắc    (rising)
    "t5",   # nặng   (heavy-falling, glottalised)
    "t6",   # ngã    (broken-rising, creaky)
]


def _build_vocab() -> dict[str, int]:
    """
    Xây dựng từ điển ánh xạ phoneme → index theo thứ tự cố định.
    """
    tokens: list[str] = []

    def _add(group):
        for ph in group:
            if ph not in tokens:
                tokens.append(ph)

    _add(SPECIAL_TOKENS.values())
    _add(VN_INITIALS)
    _add(VN_MEDIALS)
    _add(VN_NUCLEI)
    _add(VN_CODAS)
    _add(VN_TONES)
    return {tok: idx for idx, tok in enumerate(tokens)}


VOCAB:      dict[str, int] = _build_vocab()
INV_VOCAB:  dict[int, str] = {idx: tok for tok, idx in VOCAB.items()}
VOCAB_SIZE: int = len(VOCAB)

PAD_IDX: int = VOCAB[SPECIAL_TOKENS["PAD"]]
UNK_IDX: int = VOCAB[SPECIAL_TOKENS["UNK"]]
SOT_IDX: int = VOCAB[SPECIAL_TOKENS["SOT"]]
EOT_IDX: int = VOCAB[SPECIAL_TOKENS["EOT"]]

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def phoneme_to_index(phoneme: str) -> int:
    return VOCAB.get(phoneme, UNK_IDX)

def index_to_phoneme(index: int) -> str:
    return INV_VOCAB.get(index, SPECIAL_TOKENS["UNK"])

def encode_sequence(phonemes: list[str]) -> list[int]:
    return [phoneme_to_index(p) for p in phonemes]

def decode_sequence(indices: list[int], skip_special: bool = True) -> list[str]:
    skip_ids = {PAD_IDX, SOT_IDX, EOT_IDX} if skip_special else set()
    return [index_to_phoneme(i) for i in indices if i not in skip_ids]

# ═══════════════════════════════════════════════════════════════════════════════
#  VIETLISH MAP  (English word → Vietnamese-style pronunciation + LATIN phonemes)
# ═══════════════════════════════════════════════════════════════════════════════

VIETLISH_MAP: dict[str, tuple[list[str], list[str]]] = {
    "message":      (["mét", "xịt"],              ["m", "e", "t", "t4", "$", "x", "i", "t", "t5"]),
    "inbox":        (["in", "bóc"],               ["i", "n", "t1", "$", "b", "o", "c", "t4"]),
    "coffee":       (["cà", "phê"],               ["c", "a", "t2", "$", "ph", "ee", "t1"]),
    "laptop":       (["láp", "tóp"],              ["l", "a", "p", "t4", "$", "t", "o", "p", "t4"]),
    "online":       (["on", "lai"],               ["o", "n", "t1", "$", "l", "a", "y", "t1"]),
    "meeting":      (["mít", "tinh"],             ["m", "i", "t", "t4", "$", "t", "i", "ng", "t1"]),
    "email":        (["i", "meo"],                ["i", "t1", "$", "m", "e", "w", "t1"]),
    "video":        (["vi", "đê", "ô"],           ["v", "i", "t1", "$", "d", "ee", "t1", "$", "oo", "t1"]),
    "check":        (["chéc"],                    ["ch", "e", "c", "t4"]),
    "share":        (["se"],                      ["s", "e", "t1"]),
    "like":         (["lai"],                     ["l", "a", "y", "t1"]),
    "download":     (["đao", "lót"],              ["d", "a", "w", "t1", "$", "l", "o", "t", "t4"]),
    "upload":       (["ắp", "lót"],               ["aw", "p", "t4", "$", "l", "o", "t", "t4"]),
    "phone":        (["phôn"],                    ["ph", "oo", "n", "t1"]),
    "wifi":         (["quai", "phai"],            ["qu", "a", "y", "t1", "$", "ph", "a", "y", "t1"]),
    "app":          (["áp"],                      ["a", "p", "t4"]),
    "post":         (["pốt"],                     ["p", "oo", "t", "t4"]),
    "call":         (["côn"],                     ["c", "oo", "n", "t1"]),
    "sale":         (["sêu"],                     ["s", "ee", "w", "t1"]),
    "story":        (["sờ", "to", "ri"],          ["s", "ow", "t2", "$", "t", "o", "t1", "$", "r", "i", "t1"]),
    "review":       (["rì", "viu"],               ["r", "i", "t2", "$", "v", "i", "w", "t1"]),
    "content":      (["còn", "ten"],              ["c", "o", "n", "t2", "$", "t", "e", "n", "t1"]),
    "channel":      (["chen", "nồ"],              ["ch", "e", "n", "t1", "$", "n", "oo", "t2"]),
    "follow":       (["fo", "lô"],                ["ph", "o", "t1", "$", "l", "oo", "t1"]),
    "cancel":       (["can", "xồ"],               ["c", "a", "n", "t1", "$", "x", "oo", "t2"]),
    "update":       (["ắp", "đết"],               ["aw", "p", "t4", "$", "d", "ee", "t", "t4"]),
    "backup":       (["béc", "cúp"],              ["b", "e", "c", "t4", "$", "c", "u", "p", "t4"]),
    "feature":      (["phi", "chờ"],              ["ph", "i", "t1", "$", "ch", "ow", "t2"]),
    "website":      (["uép", "xai"],              ["w", "e", "p", "t4", "$", "x", "a", "y", "t1"]),
    "deploy":       (["đì", "pờ", "loi"],         ["d", "i", "t2", "$", "p", "ow", "t2", "$", "l", "o", "y", "t1"]),
    "submit":       (["sụp", "mít"],              ["s", "u", "p", "t5", "$", "m", "i", "t", "t4"]),
    "stream":       (["sì", "trim"],              ["s", "i", "t2", "$", "tr", "i", "m", "t1"]),
    "trend":        (["tren"],                    ["tr", "e", "n", "t1"]),
    "version":      (["vơ", "sình"],              ["v", "ow", "t1", "$", "s", "i", "ng", "t2"]),
    "fix":          (["phích"],                   ["ph", "i", "ch", "t4"]),
    "edit":         (["ê", "đít"],                ["ee", "t1", "$", "d", "i", "t", "t4"]),
    "search":       (["xớc"],                     ["x", "ow", "c", "t4"]),
    "google":       (["gu", "gồ"],                ["g", "u", "t1", "$", "g", "oo", "t2"]),
    "presentation": (["pờ", "rì", "sen", "tấy", "sần"], ["p", "ow", "t2", "$", "r", "i", "t2", "$", "s", "e", "n", "t1", "$", "t", "aa", "y", "t4", "$", "s", "aa", "n", "t2"]),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"{'='*58}")
    print(f"  Vietnamese Latin Phoneme Set  |  vocab size = {VOCAB_SIZE}")
    print(f"{'='*58}\n")

    _cat: dict[str, str] = {}
    for v in SPECIAL_TOKENS.values(): _cat[v] = "special"
    for p in VN_INITIALS:  _cat[p] = "VN initial"
    for p in VN_MEDIALS:   _cat[p] = "VN medial"
    for p in VN_NUCLEI:    _cat[p] = "VN nucleus"
    for p in VN_CODAS:     _cat[p] = "VN coda"
    for p in VN_TONES:     _cat[p] = "VN tone"

    print(f"  {'Idx':>4}  {'Phoneme':<8}  Category")
    print(f"  {'-'*35}")
    for tok, idx in VOCAB.items():
        print(f"  [{idx:3d}]  {tok:<8}  {_cat.get(tok, '?')}")

    print(f"\n{'='*58}")
    print("  Encode / Decode round-trip")
    print(f"{'='*58}")
    examples = [
        ["x", "i", "n", "t1", "$", "ch", "a", "w", "t2"],   # "xin chào"
        ["ch", "u", "ng", "t4", "$", "t", "oo", "y", "t1"], # "chúng tôi"
    ]
    for ph_list in examples:
        indices = encode_sequence(ph_list)
        decoded = decode_sequence(indices, skip_special=False)
        print(f"\n  phonemes : {ph_list}")
        print(f"  indices  : {indices}")
        print(f"  decoded  : {decoded}")
        assert decoded == ph_list, "Round-trip failed!"
    print("\n  All round-trip checks passed.")
