import json
import time
from collections import defaultdict
import re
from pathlib import Path
from django.core.cache import cache
from collections import defaultdict
from typing import List, Tuple, Dict
import pickle

# ------------------------------------------------------
# 🔹 Trie Implementation (Prefix Matching)
# ------------------------------------------------------
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word.lower():
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def autocomplete(self, prefix, top_k=10):
        node = self.root
        for ch in prefix.lower():
            if ch not in node.children:
                return []
            node = node.children[ch]

        results = []

        def dfs(curr, path):
            if len(results) >= top_k:
                return
            if curr.is_end:
                results.append(path)
            for c in curr.children:
                dfs(curr.children[c], path + c)

        dfs(node, prefix)
        return results

# ------------------------------------------------------
# Detect if a word is Bengali
# ------------------------------------------------------
def is_bengali(word):
    return any("\u0980" <= ch <= "\u09FF" for ch in word)

# ------------------------------------------------------
# Load JSON dataset (cached)
# ------------------------------------------------------


def load_json(path: str) -> List[Dict]:
    """
    Load JSON data from:
    - a folder containing multiple .json files
    Uses Django cache for performance.
    """
    cache_key = f"rag_json::{path}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data
    path_obj = Path(path)
    all_data: List[Dict] = []
    # Case 1: Folder with multiple JSON files
    if path_obj.is_dir():
        for json_file in sorted(path_obj.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
    # Case 2: Single JSON file (fallback / safety)
    elif path_obj.is_file() and path_obj.suffix == ".json":
        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    # Cache for 1 hour
    cache.set(cache_key, all_data, timeout=60 * 60 * 12)
    return all_data

def load_dataset(base_path: str) -> List[Dict]:
    """
    Load dataset from:
    - rag_data (flat JSON files)
    - crops_data (subfolders → JSON files)
    """
    dataset: List[Dict] = []
    base_dir = Path(base_path)
    if not base_dir.exists():
        raise ValueError(f"❌ Path not found: {base_path}")
    # -----------------------------
    # 1️⃣ Load rag_data (flat)
    # -----------------------------
    rag_data_path = base_dir / "rag_data"

    if rag_data_path.exists():
        print("📂 Loading rag_data...")
        dataset.extend(load_json(str(rag_data_path)))
    # -----------------------------
    # 2️⃣ Load crops_data (nested)
    # -----------------------------
    crops_data_path = base_dir / "crops_data"
    if crops_data_path.exists():
        print("📂 Loading crops_data...")
        for crop_folder in crops_data_path.iterdir():
            if not crop_folder.is_dir():
                continue
            print(f"   🌾 Loading {crop_folder.name}")
            dataset.extend(load_json(str(crop_folder)))
    # -----------------------------
    # FINAL
    # -----------------------------
    print(f"\n✅ Total records loaded: {len(dataset)}")
    return dataset

base_path = "/home/gflml/Chatbot/multi_modal_chatbot_new/dataset/RAG_data"
dataset = load_dataset(base_path)
a=2
# ------------------------------------------------------
# Extract keywords from dataset (Django cached)
# ------------------------------------------------------
def extract_keywords(data: List[Dict], cache_key: str = "dataset_keywords") -> Tuple[List[str], Dict[str, int]]:
    """
    Extract unique words and their frequencies from dataset.
    Caches the result in Django cache.
    """
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    words = []
    freq = defaultdict(int)
    for item in data:
        text = (item.get("title", "") + " " + item.get("content", "")).lower()
        word_list = re.findall(r"[A-Za-z]+|[\u0980-\u09FF]+", text)
        for w in word_list:
            words.append(w)
            freq[w] += 1
    unique_words = sorted(set(words))
    cache.set(cache_key, (unique_words, freq), timeout=60 * 60 * 12)  # cache for 1 hour
    return unique_words, freq

# Usage
dataset_words, frequencies = extract_keywords(dataset)


# ------------------------------------------------------
# Load fallback wordlists (Django cached)
# ------------------------------------------------------
def load_wordlist(path: str, cache_key: str = None) -> List[str]:
    """
    Load a wordlist from a file and cache the result in Django cache.
    """
    if cache_key is None:
        cache_key = f"wordlist::{path}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    path_obj = Path(path)
    words = []
    if path_obj.exists():
        with open(path_obj, "r", encoding="utf-8") as f:
            words = [w.strip().lower() for w in f if w.strip()]
    cache.set(cache_key, words, timeout=60 * 60 * 12)  # cache for 1 hour
    return words

english_words = load_wordlist("/home/gflml/Chatbot/multi_modal_chatbot_new/dataset/bn_en_word_list/479k_English_words.txt")
bengali_words = load_wordlist("/home/gflml/Chatbot/multi_modal_chatbot_new/dataset/bn_en_word_list/200k_Bangla_Words.txt")
fallback_words = sorted(set(english_words + bengali_words))

# ------------------------------------------------------
# Build Tries (Django cached)
# ------------------------------------------------------
def build_trie_cached(words, cache_key: str):
    """
    Build a Trie from a list of words and cache it in Django.
    """
    cached_trie = cache.get(cache_key)
    if cached_trie is not None:
        # Deserialize from pickle
        return pickle.loads(cached_trie)
    # Build trie
    trie = Trie()
    for w in words:
        trie.insert(w)
    # Serialize and cache
    cache.set(cache_key, pickle.dumps(trie), timeout=60 * 60 * 12)  # cache for 1 hour
    return trie
main_trie = build_trie_cached(dataset_words, "trie::dataset_words")
fallback_trie = build_trie_cached(fallback_words, "trie::fallback_words")

# ------------------------------------------------------
# Hybrid Suggestion Function
# ------------------------------------------------------
def words_suggest(prefix, top_k=10):
    prefix = prefix.strip()
    if not prefix:
        return []

    parts = prefix.split()
    prefix = parts[-1].lower()  # last word only
    prefix = prefix.lower().strip()
    if not prefix:
        return []

    # Step 1: dataset trie
    prefix_matches = main_trie.autocomplete(prefix, top_k=top_k)

    # Step 2: fallback trie
    if not prefix_matches:
        prefix_matches = fallback_trie.autocomplete(prefix, top_k=top_k)

    return prefix_matches


