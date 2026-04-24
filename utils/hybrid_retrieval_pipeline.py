import re

# =====================================================
# CONFIG
# =====================================================
CROP_CONFIG = [
    {"crop": "banana", "disease": ["panama", "sigatoka"], "pest": ["beetle spot"]},
    {"crop": "potato", "disease": ["bacterial wilt", "blight", "leaf curl", "scab"], "pest": []},
    {"crop": "maize", "disease": ["leaf blight"], "pest": ["fall armyworm", "cutworm"]},
    {"crop": "wheat", "disease": ["leaf blast", "leaf blight", "root rot"], "pest": []}
]

# =====================================================
# TEXT NORMALIZATION
# =====================================================
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =====================================================
# QUERY UNDERSTANDING
# =====================================================
def detect_crop(query):
    query = normalize_text(query)
    for item in CROP_CONFIG:
        if item["crop"] in query:
            return item["crop"]
    return None


def detect_intent_terms(query):
    query = normalize_text(query)
    found = []
    for item in CROP_CONFIG:
        for d in item["disease"]:
            if d in query:
                found.append(d)
        for p in item["pest"]:
            if p in query:
                found.append(p)
    return found


# =====================================================
# DEDUPLICATION
# =====================================================
def deduplicate(results):
    seen = set()
    unique = []
    for doc, score in results:
        key = hash(doc.page_content)
        if key not in seen:
            seen.add(key)
            unique.append((doc, score))
    return unique


# =====================================================
# FILTERING
# =====================================================
def filter_by_crop(results, query_crop):
    if not query_crop:
        return results
    filtered = []
    for doc, score in results:
        text = (
            doc.metadata.get("title", "") +
            " " +
            doc.page_content
        ).lower()
        if query_crop in text:
            filtered.append((doc, score))
    return filtered if filtered else results


# =====================================================
# BOOSTING
# =====================================================
def apply_boost(results, query_crop, intent_terms):
    boosted = []
    for doc, score in results:
        text = (
            doc.metadata.get("title", "") +
            " " +
            doc.page_content).lower()
        boost = 0
        # Crop boost
        if query_crop and query_crop in text:
            boost += 2
        # Disease / pest boost
        for term in intent_terms:
            if term in text:
                boost += 3
        boosted.append((doc, score + boost))
    return sorted(boosted, key=lambda x: x[1], reverse=True)


def phrase_boost(results, query):
    query = normalize_text(query)
    boosted = []
    for doc, score in results:
        text = (
            doc.metadata.get("title", "") +
            " " +
            doc.page_content).lower()
        if query in text:
            score += 5
        boosted.append((doc, score))
    return sorted(boosted, key=lambda x: x[1], reverse=True)


# =====================================================
# SMART RANKING PIPELINE
# =====================================================
def smart_rank_results(results, query):
    query = normalize_text(query)
    query_crop = detect_crop(query)
    intent_terms = detect_intent_terms(query)
    filter_crop_results = filter_by_crop(results, query_crop)
    boost_results = apply_boost(filter_crop_results, query_crop, intent_terms)
    phrase_boost_results = phrase_boost(boost_results, query)
    return phrase_boost_results

# =====================================================
# RRF FUSION (KEY PART)
# =====================================================
def rrf_fusion(dense_results, sparse_results, k=20):
    rrf_scores = {}
    k_rrf = 60
    # Dense (FAISS)
    for rank, (doc, _) in enumerate(dense_results):
        doc_id = doc.metadata.get("id", str(hash(doc.page_content)))
        score = 1 / (k_rrf + rank + 1)
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"doc": doc, "score": 0}
        rrf_scores[doc_id]["score"] += score
    # Sparse (BM25)
    for rank, (doc, _) in enumerate(sparse_results):
        doc_id = doc.metadata.get("id", str(hash(doc.page_content)))
        score = 1 / (k_rrf + rank + 1)
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"doc": doc, "score": 0}
        rrf_scores[doc_id]["score"] += score
    ranked = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [(d["doc"], d["score"]) for d in ranked[:k]]


# =====================================================
# FINAL HYBRID RETRIEVER
# =====================================================
def hybrid_retrieve(query, vector_store, sparse_results, top_k=10):
    # Step 1: Retrieve
    dense_results = vector_store.similarity_search_with_score(query, k=10)
    # Step 2: RRF Fusion (IMPORTANT)
    fused_results = rrf_fusion(dense_results, sparse_results, k=30)
    # Step 3: Deduplicate
    fused_results = deduplicate(fused_results)
    # Step 4: Smart Ranking
    ranked_results = smart_rank_results(fused_results, query)
    return ranked_results[:top_k]

