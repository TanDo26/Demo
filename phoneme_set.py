"""
phoneme_set.py
Bộ âm vị IPA tiếng Việt (và tiếng Anh bổ sung) với ánh xạ phoneme ↔ index.

Cấu trúc VOCAB:
    [0..6]   special tokens  (<pad>, <unk>, <sot>, <eot>, $, |, —)
    [7..27]  VN initials     (IPA)
    [28]     VN medial       /w/
    [29..40] VN nuclei       (IPA nguyên âm đơn + đôi)
    [41..48] VN codas        (p t k m n ŋ j w)
    [49..54] VN tones        (-1 .. -6)
    [55..]   EN-only phonemes

Dùng VOCAB[phoneme] để lấy index, INV_VOCAB[index] để lấy phoneme.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  SPECIAL TOKENS
# ═══════════════════════════════════════════════════════════════════════════════

SPECIAL_TOKENS = {
    "PAD":    "<pad>",   # 0  — padding
    "UNK":    "<unk>",   # 1  — unknown phoneme
    "SOT":    "<sot>",   # 2  — start of transcript
    "EOT":    "<eot>",   # 3  — end of transcript
    "VN_SEP": "$",       # 4  — Vietnamese syllable boundary
    "EN_SEP": "|",       # 5  — English syllable boundary
    "LINK":   "—",       # 6  — English linking sound
}

# ═══════════════════════════════════════════════════════════════════════════════
#  VIETNAMESE PHONEMES (IPA)
# ═══════════════════════════════════════════════════════════════════════════════

# Âm đầu (Initials) — 21 IPA symbols
# Chính tả → IPA được xử lý trong text2phoneme.py
VN_INITIALS = [
    "b",    # b
    "k",    # c / k / q
    "tʃ",   # ch / tr
    "z",    # d / gi  (miền Nam /z/, miền Bắc /j/ — dùng /z/ làm chuẩn)
    "d",    # đ
    "g",    # g / gh
    "h",    # h
    "x",    # kh
    "l",    # l
    "m",    # m
    "n",    # n
    "ŋ",    # ng / ngh
    "ɲ",    # nh
    "p",    # p  (vay mượn / phương ngữ)
    "f",    # ph
    "r",    # r
    "s",    # s / x
    "t",    # t
    "tʰ",   # th
    "v",    # v
    "ʔ",    # zero-initial / glottal stop (âm tiết bắt đầu bằng nguyên âm)
]

# Âm đệm (Medials) — 1 IPA symbol
VN_MEDIALS = [
    "w",    # labio-velar glide trong oa, oe, uê, uy...
]

# Nguyên âm / Hạt nhân (Nuclei) — 12 IPA symbols
VN_NUCLEI = [
    "a",    # a / ă
    "ɤ",    # â / ơ   (unrounded mid-back)
    "ɛ",    # e
    "e",    # ê
    "i",    # i / y
    "ɔ",    # o
    "o",    # ô
    "u",    # u
    "ɯ",    # ư       (unrounded high back)
    "iə",   # ia / iê (falling diphthong)
    "uə",   # ua / uô (falling diphthong)
    "ɯə",   # ưa / ươ (falling diphthong)
]

# Âm cuối (Codas) — 8 IPA symbols
VN_CODAS = [
    "p",    # -p  (allophone của /p/ cuối âm tiết, không bật hơi)
    "t",    # -t  (unreleased)
    "k",    # -c / -ch cuối
    "m",    # -m
    "n",    # -n
    "ŋ",    # -ng / -nh cuối
    "j",    # -i / -y glide
    "w",    # -u / -o glide
]

# Thanh điệu (Tones) — 6 tones
VN_TONES = [
    "-1",   # ngang  (level, mid)
    "-2",   # huyền  (falling)
    "-3",   # hỏi    (dipping-rising)
    "-4",   # sắc    (rising)
    "-5",   # nặng   (heavy-falling, glottalised)
    "-6",   # ngã    (broken-rising, creaky)
]

# ═══════════════════════════════════════════════════════════════════════════════
#  ENGLISH-ONLY PHONEMES (IPA)  — không có trong tiếng Việt
# ═══════════════════════════════════════════════════════════════════════════════

EN_VOWELS = [
    "æ",    # cat, bad
    "ɒ",    # hot, dog (British)
    "ʌ",    # cup, bus
    "ɪ",    # bit, sit
    "ʊ",    # put, book
    "ə",    # schwa
    "iː",   # see, feet
    "uː",   # too, blue
    "ɜː",   # bird, word
    "ɑː",   # car, father
    "ɔː",   # more, door
]

EN_DIPHTHONGS = [
    "eɪ",   # say, day
    "aɪ",   # my, fly
    "ɔɪ",   # boy, coin
    "oʊ",   # go, show
    "aʊ",   # how, now
    "ɪə",   # near, here
    "eə",   # care, bear
    "ʊə",   # tour, pure
]

EN_CONSONANTS = [
    "dʒ",   # judge, age
    "ʒ",    # vision, measure
    "θ",    # think, bath
    "ð",    # this, bathe
    "ʃ",    # she, wash
    "tz",   # affricative dùng trong Vietlish (mét xịt → "tz")
]

# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD VOCABULARY  {phoneme_str → int}
# ═══════════════════════════════════════════════════════════════════════════════

def _build_vocab() -> dict[str, int]:
    """
    Xây dựng từ điển ánh xạ phoneme → index theo thứ tự cố định.

    CẢNH BÁO: Thứ tự này KHÔNG được thay đổi sau khi đã huấn luyện model.
    Nếu cần bổ sung phoneme mới, hãy thêm vào cuối danh sách tương ứng.
    """
    tokens: list[str] = []

    def _add(group):
        for ph in group:
            if ph not in tokens:
                tokens.append(ph)

    _add(SPECIAL_TOKENS.values())   # 0–6
    _add(VN_INITIALS)               # 7–27
    _add(VN_MEDIALS)                # 28
    _add(VN_NUCLEI)                 # 29–40
    _add(VN_CODAS)                  # 41–48  (p/t/k/m/n/ŋ/j/w có thể trùng với initials về ký hiệu,
                                    #         nhưng index KHÁC nhau — context xác định vai trò)
    _add(VN_TONES)                  # 49–54
    _add(EN_VOWELS)                 # 55–65
    _add(EN_DIPHTHONGS)             # 66–73
    _add(EN_CONSONANTS)             # 74–79

    return {tok: idx for idx, tok in enumerate(tokens)}


VOCAB:      dict[str, int] = _build_vocab()
INV_VOCAB:  dict[int, str] = {idx: tok for tok, idx in VOCAB.items()}
VOCAB_SIZE: int = len(VOCAB)

# Thường dùng — tránh gọi VOCAB[...] nhiều lần
PAD_IDX: int = VOCAB[SPECIAL_TOKENS["PAD"]]
UNK_IDX: int = VOCAB[SPECIAL_TOKENS["UNK"]]
SOT_IDX: int = VOCAB[SPECIAL_TOKENS["SOT"]]
EOT_IDX: int = VOCAB[SPECIAL_TOKENS["EOT"]]

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def phoneme_to_index(phoneme: str) -> int:
    """Trả về index của phoneme, hoặc UNK_IDX nếu không tìm thấy."""
    return VOCAB.get(phoneme, UNK_IDX)


def index_to_phoneme(index: int) -> str:
    """Trả về phoneme tương ứng với index, hoặc '<unk>' nếu không hợp lệ."""
    return INV_VOCAB.get(index, SPECIAL_TOKENS["UNK"])


def encode_sequence(phonemes: list[str]) -> list[int]:
    """
    Chuyển danh sách phoneme → danh sách index.
    KHÔNG thêm SOT / EOT — gọi thủ công nếu cần.

    Ví dụ:
        encode_sequence(["tʃ", "a", "-2"]) → [9, 30, 51]   # "chào"
    """
    return [phoneme_to_index(p) for p in phonemes]


def decode_sequence(indices: list[int], skip_special: bool = True) -> list[str]:
    """
    Chuyển danh sách index → danh sách phoneme.

    Args:
        indices:      Dãy index (output của model hoặc encode_sequence).
        skip_special: Nếu True, bỏ qua PAD / SOT / EOT trong kết quả.

    Ví dụ:
        decode_sequence([2, 9, 30, 51, 3]) → ["tʃ", "a", "-2"]
    """
    skip_ids = {PAD_IDX, SOT_IDX, EOT_IDX} if skip_special else set()
    return [index_to_phoneme(i) for i in indices if i not in skip_ids]


# ═══════════════════════════════════════════════════════════════════════════════
#  VIETLISH MAP  (English word → Vietnamese-style pronunciation + IPA phonemes)
#  Được dùng bởi text2phoneme.py khi gặp từ tiếng Anh đọc kiểu Việt
# ═══════════════════════════════════════════════════════════════════════════════

VIETLISH_MAP: dict[str, tuple[list[str], list[str]]] = {
    # "english_word": (["viet-syllable", ...], ["ph", "o", "n", "e", "m", "e", "s"])
    "message":      (["mét", "xịt"],              ["m","e","-4","tz","$","s","i","-5","tz"]),
    "inbox":        (["in", "bóc"],               ["ɪ","n","$","b","o","-4","k"]),
    "coffee":       (["cà", "phê"],               ["k","a","-2","$","f","e","-1"]),
    "laptop":       (["lép", "tóp"],              ["l","e","-5","$","t","o","-4"]),
    "online":       (["on", "lai"],               ["o","n","$","l","a","j"]),
    "meeting":      (["mít", "tinh"],             ["m","i","-5","$","t","i","ŋ"]),
    "email":        (["i", "meo"],                ["i","$","m","ɛ","w"]),
    "video":        (["vi", "đê", "ô"],           ["v","i","$","d","e","-1","$","o","-1"]),
    "check":        (["chéc"],                    ["tʃ","e","-4","k"]),
    "share":        (["sẹ"],                      ["ʃ","ɛ","-5"]),
    "like":         (["lai"],                     ["l","a","j"]),
    "download":     (["đao", "lôt"],              ["d","a","w","$","l","o","-4","t"]),
    "upload":       (["ắp", "lôt"],               ["a","-4","p","$","l","o","-4","t"]),
    "phone":        (["phôn"],                    ["f","o","n","-1"]),
    "wifi":         (["oai", "phai"],             ["w","a","j","$","f","a","j"]),
    "app":          (["ép"],                      ["a","-4","p"]),
    "post":         (["pốt"],                     ["p","o","-4","t"]),
    "call":         (["col"],                     ["k","ɔ","l","-1"]),
    "sale":         (["xên"],                     ["s","ɛ","n","-1"]),
    "story":        (["xto", "ri"],               ["s","t","o","-1","$","r","i","-1"]),
    "review":       (["ri", "viu"],               ["r","i","-1","$","v","i","u","-1"]),
    "content":      (["con", "ten"],              ["k","ɔ","n","-1","$","t","ɛ","n","-1"]),
    "channel":      (["chen", "nồ"],              ["tʃ","ɛ","n","-1","$","n","o","-2"]),
    "follow":       (["fo", "lô"],                ["f","ɔ","-1","$","l","o","-1"]),
    "cancel":       (["can", "xen"],              ["k","a","n","-1","$","s","ɛ","n","-1"]),
    "update":       (["ắp", "đết"],               ["a","-4","p","$","d","ɛ","-4","t"]),
    "backup":       (["béc", "ắp"],               ["b","ɛ","-4","k","$","a","-4","p"]),
    "feature":      (["phi", "chờ"],              ["f","i","-1","$","tʃ","ɤ","-1"]),
    "website":      (["uép", "xai"],              ["u","-4","p","$","s","a","j","-1"]),
    "deploy":       (["đi", "ploi"],              ["d","i","-1","$","p","l","ɔ","j","-1"]),
    "submit":       (["xớp", "mít"],              ["s","ɤ","-4","p","$","m","i","-4","t"]),
    "stream":       (["xtrim"],                   ["s","t","r","i","m","-1"]),
    "trend":        (["tren"],                    ["t","r","ɛ","n","-1"]),
    "version":      (["vờ", "sinh"],              ["v","ɤ","-1","$","ʃ","i","ŋ","-1"]),
    "fix":          (["phích"],                   ["f","i","-4","k"]),
    "edit":         (["é", "đit"],                ["ɛ","-4","$","d","i","t","-1"]),
    "search":       (["xớc"],                     ["s","ɤ","-4","k"]),
    "google":       (["gù", "gồ"],                ["g","u","-2","$","g","o","-2"]),
    "presentation": (["pre","sen","tây","shần"],   ["p","r","ɛ","-1","$","s","ɛ","n","-1","$",
                                                    "t","a","-1","j","$","ʃ","ɤ","n","-1"]),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"{'='*58}")
    print(f"  Vietnamese IPA Phoneme Set  |  vocab size = {VOCAB_SIZE}")
    print(f"{'='*58}\n")

    # Category lookup
    _cat: dict[str, str] = {}
    for v in SPECIAL_TOKENS.values(): _cat[v] = "special"
    for p in VN_INITIALS:  _cat[p] = "VN initial"
    for p in VN_MEDIALS:   _cat[p] = "VN medial"
    for p in VN_NUCLEI:    _cat[p] = "VN nucleus"
    for p in VN_CODAS:     _cat[p] = "VN coda"
    for p in VN_TONES:     _cat[p] = "VN tone"
    for p in EN_VOWELS:    _cat[p] = "EN vowel"
    for p in EN_DIPHTHONGS:_cat[p] = "EN diphthong"
    for p in EN_CONSONANTS:_cat[p] = "EN consonant"

    print(f"  {'Idx':>4}  {'Phoneme':<8}  Category")
    print(f"  {'-'*35}")
    for tok, idx in VOCAB.items():
        print(f"  [{idx:3d}]  {tok:<8}  {_cat.get(tok, '?')}")

    print(f"\n{'='*58}")
    print("  Encode / Decode round-trip")
    print(f"{'='*58}")
    examples = [
        ["s", "i", "n", "-1", "$", "tʃ", "a", "-2"],   # "xin chào"
        ["tʃ", "u", "-4", "ŋ", "$", "t", "o", "-1", "j"],  # "chúng tôi"
        ["m", "i", "-5", "$", "t","i","ŋ"],             # "meeting" (vietlish)
    ]
    for ph_list in examples:
        indices = encode_sequence(ph_list)
        decoded = decode_sequence(indices, skip_special=False)
        print(f"\n  phonemes : {ph_list}")
        print(f"  indices  : {indices}")
        print(f"  decoded  : {decoded}")
        assert decoded == ph_list, "Round-trip failed!"
    print("\n  All round-trip checks passed.")
