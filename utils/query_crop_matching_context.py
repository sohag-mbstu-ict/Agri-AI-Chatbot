from langchain_huggingface import HuggingFaceEmbeddings
import torch
from sentence_transformers import util
import sys, os, re
from typing import List, Tuple
from functools import lru_cache
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .query_preprocess import detect_language, processed_query

# -------------------------------
# Known CROPS_BN
# -------------------------------
#উইভিল পোকা কি কচি পাতার
CROPS_BN = ["ধান","ভুট্টা","ভুট্টার","শিম","শিমের","আলু","টমেটো","পেঁয়াজের","পেয়াজ","তেঁতুল","ফুলকপির","ফুলকপি","লেবুর","লেবু","পটলের","পটল","কলা","কলার",
         "কাকরোল","কাকরোলের","পেয়ারার","পেয়ারা","গোলাপ","গোলাপের","কমলা","রসুনের","রসুন","দারুচিনি","তেজপাতার","তেজপাতা","মিষ্টি কুমড়া","চালকুমড়া",
         "হলুদ","আঙ্গুর","মাল্টা","পানপাতা","পানপাতার","আমড়া","মরিচ","মরিচের","লালশাকের","লালশাক","ডালিম","ডালিমের","পেঁপে","পেঁপের","চাল কুমড়া","মিষ্টিকুমড়া",
         "নাড়িকেলের","নাড়িকেল","নারিকেল","কচুর","কচু","ড্রাগন","ড্রাগনফল","আমের","আম","ডা‌লি‌মের","ডা‌লি‌ম","ঢেড়ঁসের","ঢেড়ঁস","তরমুজের","তরমুজ","পাট","পাটের",
         "কাঁঠালের","কাঁঠাল","বেগুনের","বেগুন","লাউ","লিচু","লিচুর","উঁইপোকা","উইভিল","সরিষা","সরিষার","গম","গমের","আখ","আখের","চিনাবাদাম","বরবটি","বরবটির",
         "মুগডাল","মুগ ডাল","সূর্যমুখী","সূর্যমুখীর","বরই","তিল","তিলর","কাঁকরোল","কাঁকরোলের","ধুন্দল","ধুন্দলের","ঝিঙ্গা","ঝিঙ্গার","চিচিঙ্গা","চিচিঙ্গার"]

CROPS_EN = [
    "rice", "maize", "bean", "potato", "tomato", "onion", "tamarind", "cauliflower", "lemon", "pointed gourd", "banana", "betel leaf", "malta",
    "spine gourd", "guava", "rose", "orange", "garlic", "cinnamon", "bay leaf", "pumpkin", "ash gourd", "turmeric", "grape", "malta orange", 
    "hog plum", "chilli", "red amaranth", "pomegranate", "papaya", "coconut", "taro", "dragon", "mango", "okra", "watermelon", "jute", "sunflower",
    "jackfruit", "eggplant", "brinjal", "bottle gourd", "lychee", "litchi","mustard", "wheat", "sugarcane", "peanut", "yardlong bean", "mung bean", 
    "sunflower", "jujube", "sesame", "snake gourd", "ridge gourd", "snake bean", "corn", "sajeena"]

# -------------------------------
# Tokenization
# -------------------------------
def tokenize(text: str):
    return text.split()

# -------------------------------
# Merge Rules
# -------------------------------
MERGE_RULES = {
    ("মিষ্টি", "কুমড়া"): "মিষ্টিকুমড়া",
    ("চাল", "কুমড়া"): "চালকুমড়া",
    ("লাল", "শাক"): "লালশাক",
    ("মুগ", "ডাল"): "মুগডাল",
    ("ড্রাগন", "ফল"): "ড্রাগনফল",
    ("পান", "পাতা"): "পানপাতা",
    ("তেজ", "পাতা"): "তেজপাতা",
    ("ফুল", "কপি"): "ফুলকপি"
}

def merge_tokens(tokens):
    merged = []
    i = 0
    while i < len(tokens):
        # Check pair (current, next)
        if i < len(tokens) - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in MERGE_RULES:
                merged.append(MERGE_RULES[pair])
                i += 2  # skip next token
                continue
        # Otherwise keep original
        merged.append(tokens[i])
        i += 1
    return merged

# -------------------------------
# Embedding Helper (NO CACHE)
# -------------------------------
def embed_query(model, text: str):
    return torch.tensor(model.embed_query(text))
# Cosine Similarity
# -------------------------------
def cosine_sim(a: torch.Tensor, b: torch.Tensor):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b)

# -------------------------------
# Token-level Crop Match
# -------------------------------
def match_crop_tokens(model, text: str, crops_bn_en, threshold: float = 0.6):
    tokens = tokenize(text)
    tokens = merge_tokens(tokens)
    # 🔥 Add bigrams (VERY IMPORTANT)
    # tokens += [
    #     f"{tokens[i]} {tokens[i+1]}"
    #     for i in range(len(tokens) - 1)]
    matches: List[Tuple[str, str, float]] = []
    CROP_EMBEDDINGS = torch.tensor(model.embed_documents(crops_bn_en))
    for token in tokens:
        emb = embed_query(model,token)
        scores = cosine_sim(emb, CROP_EMBEDDINGS)
        best_score, best_idx = torch.max(scores, dim=0)
        if best_score.item() >= threshold:
            matches.append((token, crops_bn_en[best_idx], best_score.item()))
    crop_list = []
    if matches:
        for t, c, s in matches:
            if s > 0.81:
                crop_list.append(c)
    return matches, crop_list
    
# -------------------------------
# Helper functions
# -------------------------------
def embed_text(model, text):
    """Embed a single text and return as torch tensor."""
    vec = model.embed_query(text)  # returns list/array
    return torch.tensor(vec)

def embed_texts(model, texts):
    """Embed a list of texts and return as torch tensor."""
    vecs = model.embed_documents(texts)
    return torch.tensor(vecs)

def extract_title_from_chunk_text(chunk_text):
    """Extract the title from chunk_text field."""
    match = re.search(r"Title:\s*(.+)", chunk_text)
    if match:
        return match.group(1).strip()
    return None

# -------------------------------
# Preprocess raw context text
# -------------------------------
def preprocess_context(raw_text_blocks):
    """
    Convert raw text blocks into context_chunks list of dicts.
    Each chunk dict has a "chunk_text" key.
    """
    context_chunks = []
    for block in raw_text_blocks:
        # Skip empty blocks
        if not block.strip():
            continue
        context_chunks.append({"chunk_text": block})
    return context_chunks


def filter_bangla_english_chunk(query,top_chunks):
    is_query_bn_or_en = detect_language(query)
    selected_top_chunks = []
    for index_id in range (0,len(top_chunks)):
        title = top_chunks[index_id]['title']
        is_title_bn_or_en = detect_language(title)
        if is_title_bn_or_en == is_query_bn_or_en:    
            selected_top_chunks.append(top_chunks[index_id])
    return selected_top_chunks

crop_groups = {
    "banana": ["কলা", "কলার"],
    "bean": ["শিম", "শিমের"],
    "onion": ["পেঁয়াজ", "পেঁয়াজের", "পেয়াজ"]
}

def get_crop_name(word: str) -> str:
    for eng, variants in crop_groups.items():
        if word in variants:
            return eng
    return "Not Found"

# -------------------------------
# Check if query crop matches any title in context
# -------------------------------
def is_query_crop_matching_context(model, query, top_chunks, threshold=0.6):
    """
    Returns (True, matched_title) if any title matches detected query crop.
    Accepts a single large raw context string.
    """
    # ---------------------------
    # Step 2: Detect crop from query
    # ---------------------------
    selected_top_chunks = []
    is_query_bn_or_en = detect_language(query)
    if is_query_bn_or_en == "english":
        crops_bn_en = CROPS_EN
        query = processed_query(query) # process the query
        threshold=0.7 # keep threshold=0.57 for title and threshold=0.7 for query
    else:
        crops_bn_en = CROPS_BN
        threshold=0.57 
    query_matches, query_crop_list = match_crop_tokens(model, query, crops_bn_en, threshold = 0.6)
    print("query_crop $$$$$$$$$$$$$$$$$$$$$$$ : ",query_crop_list)
    # if not query_crop: # not None return True
    if len(query_crop_list) == 0:
        # if not crop in query then we will take only theose title that have not crop name
        print("No confident crop detected in query.")
        
        for index_id in range (0,len(top_chunks)):
            title = top_chunks[index_id]['title']
            is_title_bn_or_en = detect_language(title)
            if is_title_bn_or_en == "english":
                title = processed_query(title)
            title_matches, title_crop_list = match_crop_tokens(model, title, crops_bn_en, threshold = 0.6)
            print("title_crop $$$$$$$$$$$$$$$$$$$$$$$11111111111111 : ",title_crop_list)
            # if not title_crop: # not title_crop=banana return false
            if len(title_crop_list) == 0: # take only theose title that have not crop name
                # if not crop in query then we will take only theose title that have not crop name
                selected_top_chunks.append(top_chunks[index_id])
                continue
        return False, selected_top_chunks, "No_Crop", []
    print(f"\nDetected crop from query: {query_crop_list[0]} --->>>  (score: {query_matches})")
    # ---------------------------
    # Step 3: Encode crop once
    # ---------------------------
    crop_embedding = embed_text(model, query_crop_list[0])
    # ---------------------------
    # Step 4: Compare with each title
    # ---------------------------
    is_keyword_exist=False
    for index_id in range (0,len(top_chunks)):
        title = top_chunks[index_id]['title']
        is_title_bn_or_en = detect_language(title)
        if is_title_bn_or_en == "english":
            title = processed_query(title)
            threshold=0.57 # keep threshold=0.57 for title and threshold=0.7 for query
        title_matches, title_crop_list = match_crop_tokens(model, title, crops_bn_en, threshold = 0.6)
        # if not title_crop: # Since now query has crop name now we will take chunk(title, content) which has crop name
        if len(title_crop_list) == 0:
            continue
        print("title_crop $$$$$$$$$$$$$$$$$$$$$$$22222222222222 : ",title_crop_list)
        print(f"\nDetected crop from title: {title_crop_list[0]} --->>>  (score: {title_matches})")
        title_embedding = embed_text(model, title_crop_list[0])
        similarity = util.cos_sim(title_embedding, crop_embedding)[0][0].item()
        print(f"\nTitle: {title}")
        print(f"Similarity with query_crop_list : '{query_crop_list[0]}' and title_crop_list : '{title_crop_list[0]}' : {similarity:.4f}")
        if similarity >= 0.7: # that means query crop name == title crop name
            selected_top_chunks.append(top_chunks[index_id])
            is_keyword_exist = True
    if is_keyword_exist:
        return is_keyword_exist, selected_top_chunks, "Has_Crop", query_crop_list[0]
    else:
        return False, selected_top_chunks, "No_Crop", []

