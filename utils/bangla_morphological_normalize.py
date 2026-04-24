import re
import unicodedata
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)


class BanglaMorphNormalizer:
    def __init__(self):
        # ✅ Ordered suffix list (longest first handled dynamically)
        self.suffixes: List[str] = [
            "গুলোর", "গুলো", "গুলি",
            "দেরকে", "দের",
            "ের", "র",
            "তে", "কে",
            "টা", "টি"
        ]

        # ✅ Phonetic / spelling normalization map
        self.phonetic_map: Dict[str, str] = {
            "পেয়াজ": "পেঁয়াজ",
            "পিয়াজ": "পেঁয়াজ",
            "পেঁয়াজ": "পেঁয়াজ",
            "মরিচ্": "মরিচ",
        }

        # Precompile regex
        self.non_bangla_pattern = re.compile(r"[^\u0980-\u09FF]")

    # -----------------------------------
    # 1. Unicode normalization
    # -----------------------------------
    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    # -----------------------------------
    # 2. Clean text
    # -----------------------------------
    def clean_text(self, text: str) -> str:
        text = text.strip()
        text = self.non_bangla_pattern.sub("", text)
        return text

    # -----------------------------------
    # 3. Phonetic normalization
    # -----------------------------------
    def normalize_phonetic(self, word: str) -> str:
        return self.phonetic_map.get(word, word)

    # -----------------------------------
    # 4. Multi-step suffix stripping
    # -----------------------------------
    def strip_suffixes(self, word: str) -> str:
        """
        Iteratively removes suffixes until no more can be removed.
        Handles overlapping suffixes like:
        - র, এর
        - গুলো, গুলোর
        """
        while True:
            matched = False

            # sort by length DESC → critical for overlap
            for suffix in sorted(self.suffixes, key=len, reverse=True):
                if word.endswith(suffix) and len(word) > len(suffix) + 1:
                    word = word[: -len(suffix)]
                    matched = True
                    break

            if not matched:
                break

        return word

    # -----------------------------------
    # 5. Final normalize pipeline
    # -----------------------------------
    def normalize(self, word: str) -> str:
        original = word

        word = self.normalize_unicode(word)
        word = self.clean_text(word)
        word = self.normalize_phonetic(word)
        word = self.strip_suffixes(word)

        logging.info(f"[NORMALIZE] {original} -> {word}")
        return word


# -----------------------------------
# Crop Mapping Layer (Your Use Case)
# -----------------------------------
class CropResolver:
    def __init__(self, normalizer: BanglaMorphNormalizer):
        self.normalizer = normalizer

        self.crop_map = {
            "কলা": "banana",
            "শিম": "bean",
            "পেঁয়াজ": "onion",
            "আলু": "potato",
            "মরিচ": "chili",
            "আম":"mango",
            "ধান" : "rice"
        }

    def resolve(self, word: str) -> str:
        root = self.normalizer.normalize(word)
        return self.crop_map.get(root, "Not Found")



def get_crop_name_for_image_path(bangla_crop_name):
    normalizer = BanglaMorphNormalizer()
    resolver = CropResolver(normalizer)
    en_crop_name = resolver.resolve(bangla_crop_name)
    return en_crop_name

# # -----------------------------------
# # Example Usage
# # -----------------------------------
# if __name__ == "__main__":
#     normalizer = BanglaMorphNormalizer()
#     resolver = CropResolver(normalizer)

#     test_words = ["ধান","ভুট্টা","ভুট্টার","শিম","শিমের","আলু","টমেটো","পেঁয়াজের","পেয়াজ","তেঁতুল","ফুলকপির","ফুলকপি","লেবুর","লেবু","পটলের","পটল","কলা","কলার",
#          "কাকরোল","কাকরোলের","পেয়ারার","পেয়ারা","গোলাপ","গোলাপের","কমলা","রসুনের","রসুন","দারুচিনি","তেজপাতার","তেজপাতা","মিষ্টি কুমড়া","চালকুমড়া",
#          "হলুদ","আঙ্গুর","মাল্টা","পানপাতা","পানপাতার","আমড়া","মরিচ","মরিচের","লালশাকের","লালশাক","ডালিম","ডালিমের","পেঁপে","পেঁপের","চাল কুমড়া","মিষ্টিকুমড়া",
#          "নাড়িকেলের","নাড়িকেল","নারিকেল","কচুর","কচু","ড্রাগন","ড্রাগনফল","আমের","আম","ডা‌লি‌মের","ডা‌লি‌ম","ঢেড়ঁসের","ঢেড়ঁস","তরমুজের","তরমুজ","পাট","পাটের",
#          "কাঁঠালের","কাঁঠাল","বেগুনের","বেগুন","লাউ","লিচু","লিচুর","উঁইপোকা","উইভিল","সরিষা","সরিষার","গম","গমের","আখ","আখের","চিনাবাদাম","বরবটি","বরবটির",
#          "মুগডাল","মুগ ডাল","সূর্যমুখী","সূর্যমুখীর","বরই","তিল","তিলর","কাঁকরোল","কাঁকরোলের","ধুন্দল","ধুন্দলের","ঝিঙ্গা","ঝিঙ্গার","চিচিঙ্গা","চিচিঙ্গার",
#          "শিমগুলো", "শিমগুলোর"]
#     for word in test_words:
#         print(f"{word} -> {resolver.resolve(word)}")