

def hybrid_retrieve(query, faiss, bm25, k=10):
    faiss_results = faiss.similarity_search_with_score(query, k=k)
    bm25_results = bm25.search(query, k=k)
    combined = []
    # Normalize FAISS
    for doc, dist in faiss_results:
        score = 1 / (1 + dist)
        combined.append((doc, score, "faiss"))
    # Normalize BM25
    max_bm25 = max(score for _, score in bm25_results)
    for doc, score in bm25_results:
        combined.append((doc, score / max_bm25, "bm25"))
    return combined