import re, sys, os
from typing import List, Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .query_preprocess import normalize_text_en, processed_query
# =====================================================
# CONFIG
# =====================================================
CROP_CONFIG = [
    {"crop": "কলা", "disease": ["পানামা", "সিগাটোকা"], "pest": ["বিটল পোকার দাগ"]},
    {"crop": "আলু", "disease": ["আলুর মড়ক", "ঢলে পড়া রোগ বা নেতিয়ে পড়া", "পাতা কোঁকড়ানো", "দাউদ বা স্কাব রোগ"], "pest": []},
    {"crop": "ভুট্টা", "disease": ["পাতা ঝলসানো রোগ"], "pest": ["কাটুই পোকা", "ফল আর্মিওয়ার্ম"]},
    {"crop": "গম", "disease":  ["ব্লাস্ট রোগ", "পাতা ঝলসানো", "গোড়া পঁচা"], "pest": []},
    {"crop": "আম", "disease": ["মিলিবাগ","বোঁটা পঁচা","পাতায় লাল মরিচা","অঙ্গ বিকৃতি বা গুচ্ছ মুকুল","আগামরা","পাতায় অ্যানথ্রাকনোজ","ফলের অ্যানথ্রাকনোজ"], 
                               "pest": ["ফলের মাছি পোকা","ভোমরা পোকা"]},
    {"crop":"ধান","disease":["নেক বা শীষ ব্লাস্ট","নেক ব্লাস্ট","শীষ ব্লাস্ট","নোড ব্লাস্ট","গীট ব্লাস্ট","পাতায় ব্লাস্ট","লক্ষীর গু","ভূয়াঝুল","ভূয়া ঝুল","মরাডিগ","সিথ রট","ধানের পাতা মোড়ান"], 
                    "pest": ["মাজরা পোকার মথ","মাজরা পোকার লার্ভ া","মাজরা মথ","মাজরা লার্ভ া","ধানের গান্ধী পোকা","বাদামী গাছ ফড়িং","মাজরা পোকার ডিমের গাদা"]}
    ]

INTENT_NOISE = {
    "কি", "কিভাবে", "বলুন","বল","জানতে চাই", "জানতে", "চাই", "সম্পর্কে", "একটু", "দয়া", "দয়া করে"}

# =====================================================
# QUERY UNDERSTANDING
# =====================================================
def detect_crop(query: str, CROP_CONFIG):
    for item in CROP_CONFIG:
        if item["crop"] in query:
            return item["crop"]
    return None

def detect_disease_or_pest(query: str, CROP_CONFIG):
    found = []
    for item in CROP_CONFIG:
        for d in item["disease"]:
            if d in query:
                found.append(d)
        for p in item["pest"]:
            if p in query:
                found.append(p)
    return list(set(found))

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
    chunks: List[Dict], query, query_crop: str,disease_or_pest: List[str]) -> List[Dict]:
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
                boost += 0.2   # strong signal
            elif query_crop in content:
                boost += 0.1   # weaker signal
        # =========================
        # 🦠 Disease / Pest Boost
        # =========================
        for term in disease_or_pest:
            if term in title:
                boost += 0.2   # strongest signal
            elif term in content:
                boost += 0.1

        #  Keyword match when crop name is not exist in title
        for word in query.split():
            if word == query_crop:
                continue # since we have boosted it in above
            if word in disease_or_pest:
                continue # since we have boosted it in above
            if word in title:
                boost += 0.17
            elif word in content:
                boost += 0.07

        # =========================
        # 🔥 Multi-term bonus (very important)
        # =========================
        match_count = sum(
            1 for term in disease_or_pest if term in (title + " " + content))
        if match_count >= 2:
            boost += 0.10
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
            new_c["similarity"] += 0.25
        elif query in text:
                new_c["similarity"] += 0.1
        boosted.append(new_c)
    return sorted(boosted, key=lambda x: x["similarity"], reverse=True)

# =====================================================
# FINAL SMART RERANKER
# =====================================================
def faiss_rank_chunks_bn(
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
        return []
    # 3️⃣ Deduplicate
    chunks = deduplicate_chunks(chunks)
    # 4️⃣ Query understanding
    query_norm = processed_query(query)
    query_crop = detect_crop(query_norm, CROP_CONFIG)
    disease_or_pest = detect_disease_or_pest(query_norm, CROP_CONFIG)
    # 5️⃣ Crop filter
    chunks = filter_by_crop_chunks(chunks, query_crop)
    # 6️⃣ Boosting
    chunks = apply_boost_chunks(chunks, query_norm, query_crop, disease_or_pest)
    # 7️⃣ Phrase boost
    chunks = phrase_boost_chunks(chunks, query_norm)
    # 8️⃣ Final ranking + cleanup
    final_chunks = []
    for i, c in enumerate(chunks[:top_k]):
        new_c = c.copy()
        new_c["rank"] = i + 1
        final_chunks.append(new_c)
    return final_chunks, disease_or_pest


