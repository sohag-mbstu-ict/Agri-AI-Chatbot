import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict
from rapidfuzz import process


# ------------------------------------------------------
# Load JSON dataset
# ------------------------------------------------------
def load_json(path: str) -> List[Dict]:
    """
    Load JSON data from:
    - a folder containing multiple .json files
    - or a single JSON file
    """
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

    # Case 2: Single JSON file
    elif path_obj.is_file() and path_obj.suffix == ".json":
        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    return all_data


# ------------------------------------------------------
# Load dataset
# ------------------------------------------------------
dataset = []
dataset.extend(load_json("/home/gflml/Chatbot/multi_modal_chatbot_new/dataset/RAG_data/rag_data"))
dataset.extend(load_json("/home/gflml/Chatbot/multi_modal_chatbot_new/dataset/RAG_data/crops_data"))

# ------------------------------------------------------
# Extract keywords from dataset
# ------------------------------------------------------
def extract_keywords(data: List[Dict]) -> Tuple[List[str], Dict[str, int]]:
    words = []
    freq = defaultdict(int)
    for item in data:
        text = (item.get("title", "") + " " + item.get("content", "")).lower()
        word_list = re.findall(r"[A-Za-z]+|[\u0980-\u09FF]+", text)
        for w in word_list:
            words.append(w)
            freq[w] += 1
    unique_words = sorted(set(words))
    return unique_words, freq
dataset_words, frequencies = extract_keywords(dataset)


# ------------------------------------------------------
# Load wordlists
# ------------------------------------------------------
def load_wordlist(path: str) -> List[str]:
    path_obj = Path(path)
    words = []
    if path_obj.exists():
        with open(path_obj, "r", encoding="utf-8") as f:
            words = [w.strip().lower() for w in f if w.strip()]
    return words

# ------------------------------------------------------
# Spell Corrector
# ------------------------------------------------------
class SpellCorrector:
    def __init__(self, vocabulary):
        self.vocab = list(set(vocabulary))
    def correct(self, query):
        words = query.split()
        corrected = []
        for word in words:
            match = process.extractOne(word, self.vocab)
            if match and match[1] > 80:
                corrected.append(match[0])
            else:
                corrected.append(word)
        return " ".join(corrected)


# ------------------------------------------------------
# Load vocabulary
# ------------------------------------------------------
english_words = load_wordlist(
    "/home/gflml/Chatbot/dataset/bn_en_word_list/479k_English_words.txt")
bengali_words = load_wordlist(
    "/home/gflml/Chatbot/dataset/bn_en_word_list/200k_Bangla_Words.txt")
vocabulary = sorted(set(english_words + bengali_words))


# ------------------------------------------------------
# Run Spell Corrector
# ------------------------------------------------------
spell_correct_obj = SpellCorrector(vocabulary)
while True:
    query = input("Enter your query : ")
    if query.lower() == "q":
        break
    corrected_text = spell_correct_obj.correct(query)
    print("corrected_text :", corrected_text)

