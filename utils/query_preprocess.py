import sys, os, re
from typing import List, Tuple
import spacy

# =====================================================
# LANGUAGE DETECTION
# =====================================================
def detect_language(text: str) -> str:
    english_count = len(re.findall(r"[A-Za-z]", text))
    bangla_count = len(re.findall(r"[\u0980-\u09FF]", text))
    return "english" if english_count > bangla_count else "bangla"

# =====================================================
# COMMON UTILITIES
# =====================================================
def remove_repeated_chars(input_word: str) -> str:
    # Step 1: reduce repeated single chars
    word1 = re.sub(r"(.)\1{2,}", r"\1", input_word)
    special_words = ['banana', 'papaya', 'coconut'] 
    if word1.lower() not in special_words:
        # Step 2: remove repeating clusters like চ্চচ্চচ্চ
        word1 = re.sub(r"(.{2,})\1+", r"\1", word1)
    return word1

def is_valid_bangla_word(word: str) -> bool:
    # Remove very short tokens
    if len(word) <= 2:
        return False
    # Remove tokens with excessive repetition
    if re.search(r"(.)\1{2,}", word):
        return False
    # Remove tokens like চ্চচ, ট্ট্ট, etc.
    if re.fullmatch(r"[\u0980-\u09FF]{1,3}", word):
        # small random clusters
        return False
    # Remove tokens with very low diversity
    unique_chars = len(set(word))
    if unique_chars <= 2 and len(word) >= 4:
        return False
    return True

def remove_duplicate_words(tokens: List[str]) -> List[str]:
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique

def correct_word(word: str) -> str:
    # plug SymSpell / TextBlob here later
    return word

# =====================================================
# ENGLISH PROCESSING
# =====================================================
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

INTENT_NOISE_EN = {
    "tell", "about", "explain", "describe", "give", "me",
    "information", "details", "know", "what is",
    "how to", "can you", "please", "can you please"}

CUSTOM_MAP = {
    "fruits": "fruit",
    "vegetables": "vegetable",
    "drones": "drone",
    "spraying": "spray",
    "services": "service"}

def normalize_text_en(text: str) -> str:
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


def preprocess_en_query(query: str) -> str:
    # Step 1: remove repeated chars
    tokens = query.split()
    tokens = [remove_repeated_chars(t) for t in tokens]
    # Step 2: remove intent noise
    tokens = [t for t in tokens if t.lower() not in INTENT_NOISE_EN]
    query = " ".join(tokens)
    # Step 3: normalize (lemmatization)
    query = normalize_text_en(query)
    tokens = query.split()
    # Step 4: spell correction
    tokens = [correct_word(t) for t in tokens]
    # Step 5: remove duplicates
    tokens = remove_duplicate_words(tokens)
    return " ".join(tokens)

# =====================================================
# BANGLA PROCESSING
# =====================================================

INTENT_NOISE_BN = {
    "কি", "কিভাবে", "বলুন","বল","জানতে চাই", "জানতে", "চাই", "সম্পর্কে", "একটু", "দয়া", "দয়া করে"}

def normalize_text_bn(text: str) -> str:
    text = re.sub(r"[^\u0980-\u09FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_bn_query(query: str) -> str:
    tokens = query.split()
    # Step 1: clean repeated chars
    tokens = [remove_repeated_chars(t) for t in tokens]
    # Step 2: normalize
    query = " ".join(tokens)
    query = normalize_text_bn(query)
    tokens = query.split()
    # Step 3: remove noise words
    tokens = [t for t in tokens if t not in INTENT_NOISE_BN]
    # Step 4: remove garbage tokens
    # tokens = [t for t in tokens if is_valid_bangla_word(t)]
    # Step 5: remove repeated-pattern junk
    tokens = [t for t in tokens if not re.fullmatch(r"(.)\1+", t)]
    # Step 6: remove duplicates
    tokens = remove_duplicate_words(tokens)
    return " ".join(tokens)

# =====================================================
# MAIN FUNCTION
# =====================================================
def processed_query(query: str) -> str:
    lang = detect_language(query)
    if lang == "english":
        return preprocess_en_query(query)
    elif lang == "bangla":
        return preprocess_bn_query(query)
    return query
