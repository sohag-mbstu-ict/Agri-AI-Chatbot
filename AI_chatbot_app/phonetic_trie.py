# phonetic_trie.py
# --------------------------------------
# Trie-based Banglish → Bangla Phonetic Engine
# Inspired by Avro Keyboard (simplified)
# --------------------------------------

# -----------------------------
# 1. Phonetic Mapping Rules
# -----------------------------
PHONETIC_RULES = {

# =============================
# Consonant clusters
# =============================
"kkh": "ক্ষ",
"ksh": "ক্ষ",
"chh": "ছ",
"ngk": "ঙ্ক",
"ngg": "ঙ্গ",
"ngc": "ঞ্চ",
"ngj": "ঞ্জ",
"nch": "ঞ্ছ",
"nth": "ন্থ",
"ndh": "ন্ধ",
"mph": "ম্ফ",

"kh": "খ",
"gh": "ঘ",
"ch": "চ",
"jh": "ঝ",
"th": "থ",
"dh": "ধ",
"ph": "ফ",
"bh": "ভ",
"sh": "শ",
"ss": "ষ",
"ng": "ঙ",
"ny": "ঞ",
# =============================
# Independent vowels
# =============================
"aa": "আ",
"ii": "ঈ",
"uu": "ঊ",
"ri": "ঋ",
"oi": "ঐ",
"ou": "ঔ",

"a": "অ",
"i": "ই",
"u": "উ",
"e": "এ",
"o": "ও",
# =============================
# Vowel signs (kar)
# =============================
"aa_kar": "া",
"i_kar": "ি",
"ii_kar": "ী",
"u_kar": "ু",
"uu_kar": "ূ",
"e_kar": "ে",
"oi_kar": "ৈ",
"o_kar": "ো",
"ou_kar": "ৌ",
"ri_kar": "ৃ",
# =============================
# Consonants
# =============================
"k": "ক",
"g": "গ",
"c": "চ",
"j": "জ",
"t": "ত",
"d": "দ",
"n": "ন",
"p": "প",
"b": "ব",
"m": "ম",
"l": "ল",
"r": "র",
"s": "স",
"h": "হ",

"y": "য",
"z": "জ",
"f": "ফ",
"q": "ক",
"v": "ভ",
"x": "ক্স",
# =============================
# Retroflex / special
# =============================
"T": "ট",
"D": "ড",
"N": "ণ",
"R": "ড়",
"Rh": "ঢ়",
# =============================
# Bengali digits
# =============================
"0": "০",
"1": "১",
"2": "২",
"3": "৩",
"4": "৪",
"5": "৫",
"6": "৬",
"7": "৭",
"8": "৮",
"9": "৯",
# =============================
# Punctuation
# =============================
".": "।",
",": ",",
"?": "?",
"!": "!",

}


# -----------------------------
# 2. Trie Node
# -----------------------------
class TrieNode:
    def __init__(self):
        self.children = {}
        self.output = None  # Bangla character if rule ends here


# -----------------------------
# 3. Phonetic Trie
# -----------------------------
class PhoneticTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, key: str, value: str):
        node = self.root
        for ch in key:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.output = value

    def transliterate(self, text: str) -> str:
        """
        Convert Banglish text into Bangla using longest match
        """
        text = text.lower()
        i = 0
        result = ""

        while i < len(text):
            node = self.root
            last_match = None
            last_pos = i

            j = i
            while j < len(text) and text[j] in node.children:
                node = node.children[text[j]]
                if node.output:
                    last_match = node.output
                    last_pos = j + 1
                j += 1

            if last_match:
                result += last_match
                i = last_pos
            else:
                result += text[i]
                i += 1

        return result


# -----------------------------
# 4. Build Phonetic Engine
# -----------------------------
def build_phonetic_engine() -> PhoneticTrie:
    trie = PhoneticTrie()
    for key, value in PHONETIC_RULES.items():
        trie.insert(key, value)
    return trie


# -----------------------------
# 5. Demo / Test
# -----------------------------
if __name__ == "__main__":
    engine = build_phonetic_engine()

    examples = [
        "ami",
        "tumi",
        "bangla",
        "chad",
        "khub bhalo",
    ]

    for word in examples:
        print(f"{word}  →  {engine.transliterate(word)}")


