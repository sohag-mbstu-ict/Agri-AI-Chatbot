import re
import torch
import spacy
from typing import List, Tuple
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# Crop List
# -------------------------------
CROPS_EN = [
    "rice", "maize", "bean", "potato", "tomato", "onion", "tamarind",
    "cauliflower", "lemon", "pointed gourd", "banana", "betel leaf",
    "malta", "spine gourd", "guava", "rose", "orange", "garlic",
    "cinnamon", "bay leaf", "pumpkin", "ash gourd", "turmeric",
    "grape", "malta orange", "hog plum", "chili", "red amaranth",
    "pomegranate", "papaya", "coconut", "taro", "dragon", "mango",
    "okra", "watermelon", "jute", "sunflower", "jackfruit",
    "eggplant", "brinjal", "bottle gourd", "lychee", "litchi",
    "mustard", "wheat", "sugarcane", "peanut", "yardlong bean",
    "mung bean", "jujube", "sesame", "snake gourd",
    "ridge gourd", "snake bean"
]

# -------------------------------
# Load Embedding Model
# -------------------------------
model = HuggingFaceEmbeddings(
    model_name="/home/gflml/Chatbot/pretrained_model/embeddings/BAAI_bge_m3",
    encode_kwargs={"normalize_embeddings": True})
# -------------------------------
# Precompute crop embeddings (🔥 VERY IMPORTANT)
# -------------------------------
CROP_EMBEDDINGS = torch.tensor(model.embed_documents(CROPS_EN))
# -------------------------------
# SpaCy Setup
# -------------------------------
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
CUSTOM_MAP = {
    "fruits": "fruit",
    "vegetables": "vegetable",
    "drones": "drone",
    "spraying": "spray",
    "services": "service"}

# -------------------------------
# Text Normalization
# -------------------------------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        lemma = token.lemma_.strip()
        lemma = CUSTOM_MAP.get(lemma, lemma)
        if lemma:
            tokens.append(lemma)
    return " ".join(tokens)
# -------------------------------
# Tokenization
# -------------------------------
def tokenize(text: str):
    return text.split()
# -------------------------------
# Embedding Helpers
# -------------------------------
@lru_cache(maxsize=10000)
def embed_query_cached(text: str):
    return torch.tensor(model.embed_query(text))
# -------------------------------
# Cosine Similarity
# -------------------------------
def cosine_sim(a: torch.Tensor, b: torch.Tensor):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b)
# -------------------------------
# Sentence-level Crop Match
# -------------------------------
def match_crop(text: str, threshold: float = 0.5):
    text = normalize_text(text)
    text_emb = embed_query_cached(text)
    scores = cosine_sim(text_emb, CROP_EMBEDDINGS)
    best_score, best_idx = torch.max(scores, dim=0)
    if best_score.item() >= threshold:
        return CROPS_EN[best_idx], best_score.item()
    return None, best_score.item()
# -------------------------------
# Token-level Crop Match
# -------------------------------
def match_crop_tokens(text: str, threshold: float = 0.6):
    text = normalize_text(text)
    tokens = tokenize(text)
    # 🔥 Add bigrams (VERY IMPORTANT)
    # tokens += [
    #     f"{tokens[i]} {tokens[i+1]}"
    #     for i in range(len(tokens) - 1)]
    matches: List[Tuple[str, str, float]] = []
    for token in tokens:
        emb = embed_query_cached(token)
        scores = cosine_sim(emb, CROP_EMBEDDINGS)
        best_score, best_idx = torch.max(scores, dim=0)
        if best_score.item() >= threshold:
            matches.append((token, CROPS_EN[best_idx], best_score.item()))
    crop_list = []
    if matches:
        for t, c, s in matches:
            if s > 0.8:
                crop_list.append(c)
    return matches, crop_list


# -------------------------------
# MAIN LOOP
# -------------------------------
if __name__ == "__main__":
    print("Type 'q' to quit\n")
    while True:
        text = input("Enter text: ")
        if text.lower() == "q":
            break
        # Sentence Match
        crop, score = match_crop(text)
        print("\n🔍 Sentence Match:")
        print("Crop:", crop)
        print("Score:", round(score, 4))
        # Token Match
        token_matches, crop_list = match_crop_tokens(text)
        print("\n🧩 Token Matches:")
        if token_matches:
            for t, c, s in token_matches:
                print(f"{t} → {c} ({round(s,4)})")
        else:
            print("No strong matches")
        print("🎯 Final Crop List:",crop_list)
        print("\n" + "-" * 50 + "\n")