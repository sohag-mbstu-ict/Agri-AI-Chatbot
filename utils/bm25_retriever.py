from rank_bm25 import BM25Okapi
import re

STOPWORDS = {"of","the","is","are","a","an","in","on","for","to"}

class BM25Retriever:
    
    def __init__(self, documents):
        self.documents = documents
        corpus = []
        for doc in documents:
            title = doc.metadata.get("title","")
            text = title + " " + title + " " + doc.page_content
            corpus.append(self.tokenize(text))
        self.bm25 = BM25Okapi(corpus)

    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOPWORDS]
        return tokens

    def search(self, query, k=10):
        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        results = []
        query_lower = query.lower()
        for idx, score in enumerate(scores):
            doc = self.documents[idx]
            text = (doc.metadata.get("title","") + " " + doc.page_content).lower()
            if query_lower in text:
                score += 10
            results.append((idx, score))
        ranked = sorted(results, key=lambda x: x[1], reverse=True)
        return [(self.documents[i], s) for i,s in ranked[:k]]




# def search(self, query, k=10):
#         tokens = self.tokenize(query)
#         scores = self.bm25.get_scores(tokens)
#         results = []
#         query_lower = query.lower()
#         for idx, score in enumerate(scores):
#             doc = self.documents[idx]
#             text = (doc.metadata.get("title","") + " " + doc.page_content).lower()
#             if query_lower in text:
#                 score += 10
#             results.append((idx, score))

#         # normalize scores
#         scores_only = [s for _, s in results]
#         min_score = min(scores_only)
#         max_score = max(scores_only)
#         normalized_results = []
#         for idx, score in results:
#             if max_score - min_score == 0:
#                 norm_score = 0
#             else:
#                 norm_score = (score - min_score) / (max_score - min_score)
#             # convert relevance → distance
#             distance = 1 - norm_score
#             normalized_results.append((idx, distance))
#         ranked = sorted(normalized_results, key=lambda x: x[1])
#         return [(self.documents[i], s) for i, s in ranked[:k]]



# from rank_bm25 import BM25Okapi
# import re

# class BM25Retriever:

#     def __init__(self, documents):
#         self.documents = documents
#         corpus = [self.tokenize(doc.page_content) for doc in documents]
#         self.bm25 = BM25Okapi(corpus)

#     def tokenize(self, text):
#         text = text.lower()
#         text = re.sub(r"[^\w\s]", " ", text)
#         return text.split()

#     def search(self, query, k=10):
#         tokens = self.tokenize(query)
#         scores = self.bm25.get_scores(tokens)
#         ranked = sorted(
#             list(enumerate(scores)),
#             key=lambda x: x[1],
#             reverse=True)
#         results = []

#         for idx, score in ranked[:k]:
#             doc = self.documents[idx]
#             results.append((doc, score))
#         return results

        