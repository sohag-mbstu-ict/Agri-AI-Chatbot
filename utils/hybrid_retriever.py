

def dense_search(vector_store, query, k=10):
    results = vector_store.similarity_search_with_score(query, k=k)
    dense_docs = []
    for doc, distance in results:
        similarity = 1 - float(distance)
        dense_docs.append((doc, similarity))
    return dense_docs



def hybrid_search(query, vector_store, bm25, k=10):
    dense_results = dense_search(vector_store, query, k)
    bm25_results = bm25.search(query, k)
    combined = {}

    for doc, score in dense_results:
        key = doc.metadata["id"]
        combined[key] = {
            "doc": doc,
            "score": score}

    for doc, score in bm25_results:
        key = doc.metadata["id"]
        if key in combined:
            combined[key]["score"] += score
        else:
            combined[key] = {
                "doc": doc,
                "score": score}

    ranked = sorted(
        combined.values(),
        key=lambda x: x["score"],
        reverse=True)
    return ranked[:k]


