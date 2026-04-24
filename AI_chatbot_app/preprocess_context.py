import numpy as np
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import re, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.query_preprocess import detect_language

def preprocess_context(context_text: str):
    """
    Clean large combined context string and structure it properly.
    """
    # Split by Title blocks
    sections = re.split(r"\n\s*Title:\s*", context_text)
    processed_sections = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        # Extract title (first line before newline)
        lines = sec.split("\n", 1)
        title = lines[0].strip()
        body = lines[1] if len(lines) > 1 else ""
        # Remove metadata
        body = re.sub(r"Category:.*?\n", "", body)
        body = re.sub(r"Tags:.*?\n", "", body)
        body = re.sub(r"Content:\s*", "", body)
        body = body.strip()
        formatted = f"""
                    [Section]
                    Topic: {title}
                    {body}
                    """
        processed_sections.append(formatted.strip())
    return "\n\n".join(processed_sections)

# =====================================================
# 🔹 FALLBACK RESPONSE
# =====================================================
def fallback_Not_Found_RAG_answer(query):
    # simple language detection
    if re.search(r"[\u0980-\u09FF]", query):
        return "দয়া করে স্পষ্ট কৃষি সম্পর্কিত প্রশ্ন করুন যাতে আমি আপনাকে সঠিকভাবে সাহায্য করতে পারি।"
    else:
        return "Please ask a clear agriculture-related question so I can help you."
    

import re

# =====================================================
# 🔹 KEYWORDS
# =====================================================
greeting_keywords = [
    "hi", "hello", "hey", "hii", "helo",
    "good morning", "good afternoon", "good evening", "have a good day"
]

identity_keywords = [
    "who are you", "how do you work", "what are you",
    "about you", "tell me about yourself", "introduce yourself",
    "are you dr chashi", "what is dr chashi",
    "chashi", "gfl", "gfl ai", "gap",
    "ডা.চাষী", "ডা.চাষী কাজ", "জিনিয়াস ফার্মস লিমিটেড",
    "স্প্রে ড্রোন", "সয়েল সেন্সর", "আবহাওয়ার পূর্বাভাস",
    "আবহাওয়ার", "কৃষি চর্চা", "প্রশিক্ষণ",
    "ড. চাষী কে", "ডক্টর চাষী কে"
]

capability_keywords = [
    "help", "helpful", "assist", "assistance", "support",
    "how can you help", "how you help me", "how can you help me",
    "what can you do", "what do you do", "what types of help",
    "why should i use you", "why i use you", "why will i use you",
    "will you be helpful", "can you help me", "can you assist me",
    "how do you work", "what services do you provide",
    "কিভাবে সাহায্য করবে", "তুমি কি সাহায্য করতে পারো",
    "আপনি কি সাহায্য করতে পারেন", "কি ধরনের সাহায্য",
    "তুমি কি করো", "আপনি কি করেন", "কিভাবে কাজ করো",
    "কেন তোমাকে ব্যবহার করব", "তুমি কি উপকারী",
    "আপনি কি উপকারী", "সাহায্য করবে"
]

# =====================================================
# 🔹 HELPER: CREATE REGEX PATTERN
# =====================================================
def build_pattern(keywords):
    """
    Build regex pattern with word boundaries for robust matching.
    Handles both single words and phrases.
    """
    escaped_keywords = [re.escape(k.lower()) for k in keywords]

    # For phrases → allow flexible spaces
    escaped_keywords = [
        kw.replace(r"\ ", r"\s+") for kw in escaped_keywords
    ]

    pattern = r"\b(" + "|".join(escaped_keywords) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


# Compile patterns once (FAST 🚀)
greeting_pattern = build_pattern(greeting_keywords)
identity_pattern = build_pattern(identity_keywords)
capability_pattern = build_pattern(capability_keywords)

# =====================================================
# 🔹 DETECT FUNCTIONS
# =====================================================
def is_greeting(query: str) -> bool:
    return bool(greeting_pattern.search(query))


def is_identity(query: str) -> bool:
    return bool(identity_pattern.search(query))


def is_capability(query: str) -> bool:
    return bool(capability_pattern.search(query))


def detect_intent(query: str) -> str:
    query = query.lower().strip()

    if is_greeting(query):
        return "greeting"
    elif is_identity(query):
        return "assistant_identity"
    elif is_capability(query):
        return "assistant_capability"
    else:
        return "agriculture_query"


def is_small_talk(query: str) -> bool:
    intent = detect_intent(query)
    return intent in ["greeting", "assistant_identity", "assistant_capability"]













# # =====================================================
# # 🔹 KEYWORD DEFINITIONS (Intent Anchors)
# # =====================================================
# greeting_keywords = [
#     # English greetings
#     "hi", "hello", "hey", "hii", "helo", "good morning",
#     "good afternoon", "good evening","have a good day",

#     # # Bangla greetings
#     # "হাই", "হ্যালো", "হ্যালোও", "সালাম", "আসসালামু আলাইকুম",
#     # "কেমন আছো", "কেমন আছেন", "কি খবর"
# ]
# identity_keywords = [
#     # English
#     "who are you",
#     "how do you work",
#     "what are you",
#     "about you",
#     "tell me about yourself",
#     "introduce yourself",
#     "are you dr chashi",
#     "what is dr chashi",
#     "chashi",
#     "gfl",
#     "GFL",
#     "gfl ai",
#     "GAP",
#     "ডা.চাষী",
#     "ডা.চাষী কাজ",
#     "জিনিয়াস ফার্মস লিমিটেড",
#     "স্প্রে ড্রোন",
#     "সয়েল সেন্সর",
#     "আবহাওয়ার পূর্বাভাস",
#     "আবহাওয়ার",
#     "কৃষি চর্চা",
#     "প্রশিক্ষণ"
#     # # Bangla
#     # "তুমি কে",
#     # "আপনি কে",
#     # "তুমি কি",
#     # "আপনি কি",
#     # "নিজের সম্পর্কে বল",
#     "ড. চাষী কে",
#     "ডক্টর চাষী কে"
# ]
# capability_keywords = [
#     # English
#     "help",
#     "helpful",
#     "assist",
#     "assistance",
#     "support",
#     "how can you help",
#     "how you help me",
#     "how can you help me",
#     "what can you do",
#     "what do you do",
#     "what types of help",
#     "why should i use you",
#     "why i use you",
#     "why will i use you",
#     "will you be helpful",
#     "can you help me",
#     "can you assist me",
#     "how do you work",
#     "what services do you provide",

#     # Bangla
#     "কিভাবে সাহায্য করবে",
#     "তুমি কি সাহায্য করতে পারো",
#     "আপনি কি সাহায্য করতে পারেন",
#     "কি ধরনের সাহায্য",
#     "তুমি কি করো",
#     "আপনি কি করেন",
#     "কিভাবে কাজ করো",
#     "কেন তোমাকে ব্যবহার করব",
#     "তুমি কি উপকারী",
#     "আপনি কি উপকারী",
#     "সাহায্য করবে"
# ]
# greeting_keywords = [
#     "hi", "hello","hellow","helo", "hey", "good morning", "good evening",
#     "হ্যালো", "সালাম", "আসসালামু আলাইকুম"
# ]

# identity_keywords = [
#     "who are you", "what are you", "introduce yourself",
#     "tell me about yourself", "what is dr chashi",
#     "ডা. চাষী কে", "ডক্টর চাষী কে"
# ]

# capability_keywords = [
#     "what can you do", "how can you help",
#     "can you help me", "what services do you provide",
#     "তুমি কি সাহায্য করতে পারো", "কিভাবে সাহায্য করবে"
# ]

# # =====================================================
# # 🔹 EMBEDDING CACHE (IMPORTANT FOR SPEED 🚀)
# # =====================================================
# def embed_texts(texts: List[str],embedding_model):
#     return embedding_model.embed_documents(texts)



# # =====================================================
# # 🔹 SIMILARITY FUNCTION
# # =====================================================
# def get_max_similarity(query_emb, keyword_embs):
#     scores = cosine_similarity([query_emb], keyword_embs)[0]
#     return np.max(scores)

# # =====================================================
# # 🔹 INTENT DETECTION (SEMANTIC)
# # =====================================================
# def detect_intent(query: str, embedding_model, threshold=0.5):
#     query = query.strip()

#     # 🔹 Embed query
#     query_emb = embedding_model.embed_query(query)
#     greeting_emb = embed_texts(greeting_keywords,embedding_model)
#     identity_emb = embed_texts(identity_keywords,embedding_model)
#     capability_emb = embed_texts(capability_keywords,embedding_model)

#     # 🔹 Compute similarity
#     greet_score = get_max_similarity(query_emb, greeting_emb)
#     identity_score = get_max_similarity(query_emb, identity_emb)
#     capability_score = get_max_similarity(query_emb, capability_emb)

#     print(f"Scores → Greeting: {greet_score:.3f}, Identity: {identity_score:.3f}, Capability: {capability_score:.3f}")

#     # 🔹 Decision logic
#     if greet_score > threshold:
#         return "greeting"
#     elif identity_score > threshold:
#         return "assistant_identity"
#     elif capability_score > threshold:
#         return "assistant_capability"
#     else:
#         return "agriculture_query"

# # =====================================================
# # 🔹 SMALL TALK DETECTION
# # =====================================================
# def is_small_talk(query: str, embedding_model) -> bool:
#     intent = detect_intent(query, embedding_model)

#     if intent in ["greeting", "assistant_identity", "assistant_capability"]:
#         return True
#     return False






























# def detect_intent(query: str):
#     query_lower = query.lower().strip()
#     if any(word in query_lower for word in greeting_keywords):
#         return "greeting"
#     if any(word in query_lower for word in identity_keywords):
#         return "assistant_identity"
#     if any(word in query_lower for word in capability_keywords):
#         return "assistant_capability"
#     return "agriculture_query"

# def is_small_talk(query: str) -> bool:
#     intent = detect_intent(query)
#     if intent in ["greeting", "assistant_identity", "assistant_capability"]:
#         print("greeting, assistant_identity, assistant_capability : ",True)
#         return True
#     else:
#         return False
    
# def fallback_Not_Found_RAG_answer(query):
#     is_bn_or_en = detect_language(query)
#     if is_bn_or_en == "english":
#         outside_ans = "Please ask clear question so that I can help you"
#     else:
#         outside_ans = "দয়া করে স্পষ্ট প্রশ্ন জিজ্ঞাসা করুন যাতে আমি আপনাকে সঠিকভাবে সাহায্য করতে পারি।"
#     return outside_ans
# # torch==2.10.0
# # transformers==5.0.0
# # langchain-community==0.3.31
# # langchain-core==0.3.83
# # langchain_text_splitters==0.3.11
# # langchain_huggingface==0.3.1
# # django==6.0.1
# # djangorestframework==3.16.1
# # accelerate==1.12.0
# # sentence-transformers==5.2.1
# # faiss-cpu==1.13.2