import re, sys, os
from typing import List, Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .query_preprocess import normalize_text_en, processed_query
from .get_pest_disease import extract_disease_or_pest
# =====================================================
# CONFIG
# =====================================================
CROP_CONFIG = [
    {"crop": "banana", "disease": ["panama", "sigatoka"], "pest": ["beetle spot"]},
    {"crop": "chilli", "disease": ["dieback"], "pest": []},
    {"crop": "potato", "disease": ["bacterial wilt", "blight", "leaf curl", "scab"], "pest": []},
    {"crop": "maize", "disease": ["leaf blight"], "pest": ["fall armyworm", "cutworm"]},
    {"crop": "wheat", "disease": ["leaf blast", "leaf blight", "root rot"], "pest": []},
    {"crop": "mango", "disease": ["mealybug", "stem end rot", "malformation", "dieback", "anthracnose"],# of leaf", "anthracnose of fruit"], 
                                 "pest": ["fruit fly", "weevil"]},
    {"crop": "rice", "disease": ["Neck Blast","Node Blast","Leaf Blast","False Smut","Deadheart","Sheath rot"], 
             "pest": ["BPH","Stem Borer Egg Mass","Leaf Folder","Stem borer Moth","Stem Borer Larva","Rice bug"]},
    ]

def normalize_crop_data(data):
    normalized = []
    for item in data:
        normalized.append({
            "crop": item.get("crop", "").lower(),
            "disease": [d.lower() for d in item.get("disease", [])],
            "pest": [p.lower() for p in item.get("pest", [])],})
    return normalized
CROP_CONFIG = normalize_crop_data(CROP_CONFIG)

INTENT_NOISE = {
    "tell", "about", "explain", "describe", "give", "me",
    "information", "details", "know", "what is",
    "how to", "can you", "please", "can you please"}

# =====================================================
# QUERY UNDERSTANDING
# =====================================================
def detect_crop(query: str):
    for item in CROP_CONFIG:
        if item["crop"] in query:
            return item["crop"]
    return None

# =====================================================
# DEDUPLICATION
# =====================================================
def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for c in chunks:
        key = hash(c["chunk_text"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

# =====================================================
# FILTERING
# =====================================================
def filter_by_crop_chunks(chunks: List[Dict], query_crop: str) -> List[Dict]:
    if not query_crop:
        return chunks
    filtered = []
    for c in chunks:
        text = (c["title"] + " " + c["chunk_text"]).lower()
        text = processed_query(text)
        if query_crop in text:
            filtered.append(c)
    return filtered if filtered else chunks

# =====================================================
# NOISE REMOVAL
# =====================================================
def remove_noise(chunks: List[Dict], threshold: float = 0.05) -> List[Dict]:
    return [c for c in chunks if c["similarity"] >= threshold]

# =====================================================
# BOOSTING
# =====================================================
def apply_boost_chunks(
    chunks: List[Dict], query, query_crop: str,intent_terms: List[str]) -> List[Dict]:
    boosted = []
    tokens = query.split()
    # Remove noise words
    query_token = [t for t in tokens if t not in INTENT_NOISE] # Remove noise words
    query = " ".join(query_token)
    for c in chunks:
        title = c["title"].lower()
        title = processed_query(title)
        content = c["chunk_text"].lower()
        content = processed_query(content)
        base_score = c["similarity"]
        boost = 0.0
        # =========================
        # 🌱 Crop Boost (Title > Content)
        # =========================
        if query_crop: # if crop exist then it will also perform Keyword match
            if query_crop in title:
                boost += 0.1   # strong signal
            elif query_crop in content:
                boost += 0.03   # weaker signal
        # =========================
        # 🦠 Disease / Pest Boost
        # =========================
        for term in intent_terms:
            if term in title:
                boost += 0.17   # strongest signal
            elif term in content:
                boost += 0.09

        #  Keyword match when crop name is not exist in title
        for word in query.split():
            if word == query_crop:
                continue # since we have boosted it in above
            if word in intent_terms:
                continue # since we have boosted it in above
            if word in title:
                boost += 0.07
            elif word in content:
                boost += 0.03

        # =========================
        # Multi-term bonus (very important)
        # =========================
        match_count = sum(
            1 for term in intent_terms if term in (title + " " + content))
        if match_count >= 2:
            boost += 0.07
        # =========================
        # 📏 Length Penalty (remove weak chunks)
        # =========================
        word_count = len(content.split())
        if word_count < 20:
            boost -= 0.05
        # =========================
        # 🧠 Final Score
        # =========================
        final_score = base_score + boost
        new_c = c.copy()
        new_c["similarity"] = final_score
        boosted.append(new_c)
    # =========================
    # 📊 Sort
    # =========================
    boosted = sorted(
        boosted,
        key=lambda x: x["similarity"],
        reverse=True)
    return boosted

# =====================================================
# PHRASE BOOST
# =====================================================
def phrase_boost_chunks(chunks: List[Dict], query: str) -> List[Dict]:
    tokens = query.split()
    # Remove noise words
    query_token = [t for t in tokens if t not in INTENT_NOISE] # Remove noise words
    query = " ".join(query_token)
    boosted = []
    for c in chunks:
        text = (c["title"] + " " + c["chunk_text"]).lower()
        # text = processed_query(text) # No need
        new_c = c.copy()
        if query in c["title"]:
            new_c["similarity"] += 0.17
        elif query in text:
                new_c["similarity"] += 0.1
        boosted.append(new_c)
    return sorted(boosted, key=lambda x: x["similarity"], reverse=True)

# =====================================================
# FINAL SMART RERANKER
# =====================================================
def faiss_rank_chunks_en(
    top_chunks: List[Dict],
    query: str,
    top_k: int = 5) -> List[Dict]:
    if not top_chunks:
        return []
    # 1️⃣ Normalize similarity
    # chunks = normalize_similarity(top_chunks)
    # 2️⃣ Remove noise
    chunks = remove_noise(top_chunks, threshold=0.05)
    if not chunks:
        return [], []
    # 3️⃣ Deduplicate
    chunks = deduplicate_chunks(chunks)
    # 4️⃣ Query understanding
    query_norm = processed_query(query)
    query_crop = detect_crop(query_norm)
    disease_or_pest_name = extract_disease_or_pest(query_norm, CROP_CONFIG)
    # 5️⃣ Crop filter
    chunks = filter_by_crop_chunks(chunks, query_crop)
    # 6️⃣ Boosting
    chunks = apply_boost_chunks(chunks, query_norm, query_crop, disease_or_pest_name)
    # 7️⃣ Phrase boost
    chunks = phrase_boost_chunks(chunks, query_norm)
    # 8️⃣ Final ranking + cleanup
    final_chunks = []
    for i, c in enumerate(chunks[:top_k]):
        new_c = c.copy()
        new_c["rank"] = i + 1
        final_chunks.append(new_c)
    return final_chunks, disease_or_pest_name


