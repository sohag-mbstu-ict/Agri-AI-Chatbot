import os
import json
from pathlib import Path
import time
import re
from typing import List, Dict
from functools import lru_cache
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
from langchain_core.prompts import ChatPromptTemplate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from .fallback_Qwen3_0_6 import QwenChatbot
from .fallback_Qwen3_1_7b import QwenChatbot
# from .gemma_3_1b_it import GemmaRAGChatbot
 
# =====================================================
# CONFIG
# =====================================================
VECTOR_STORE_DIR = "/home/gflml/Chatbot/chatbot_with_Django/vector_store/query_answer"
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "index.faiss")
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2 "
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)



@lru_cache(maxsize=1)  # Qween Model
def fall_back_llm():
    return QwenChatbot()
fallback_llm_bot = fall_back_llm()

# @lru_cache(maxsize=1)    # Gemma Model
# def fall_back_llm():
#     return GemmaRAGChatbot()
# fallback_llm_bot = fall_back_llm()

# =====================================================
# 1️⃣ LOAD DATA
# =====================================================
def load_rag_data(json_path: str) -> List[Dict]:
    """
    Load raw RAG data from:
    - a single JSON file
    - OR a folder containing multiple JSON files
    """
    path = Path(json_path)
    data: List[Dict] = []
    # Case 1: Single JSON file
    if path.is_file() and path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                data.extend(loaded)
            else:
                data.append(loaded)
    # Case 2: Folder with multiple JSON files
    elif path.is_dir():
        for json_file in sorted(path.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    data.extend(loaded)
                else:
                    data.append(loaded)
    return data

# =====================================================
# 2️⃣ PREPARE DOCUMENTS
# =====================================================
def prepare_documents(data: List[Dict]) -> List[Document]:
    """Convert raw JSON into LangChain Documents."""
    documents = []

    for item in data:
        content = (
            f"Title: {item.get('title', '')}\n"
            f"Category: {item.get('category', '')}\n"
            f"Tags: {', '.join(item.get('tags', []))}\n"
            f"Content: {item.get('content', '')}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "category": item.get("category"),
                    "tags": item.get("tags"),
                }
            )
        )
    return documents

# =====================================================
# 2️⃣A CHUNKING FUNCTION
# =====================================================
def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into semantically meaningful chunks
    while preserving metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)

    chunked_docs = []

    for doc in documents:
        chunks = splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                )
            )

    return chunked_docs


# =====================================================
# 3️⃣ LOAD EMBEDDING MODEL
# =====================================================
@lru_cache(maxsize=1)
def load_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Load and cache embedding model."""
    print("🔹 Loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True}
    )

@lru_cache(maxsize=1)
def load_vector_store_cached(index_path: str) -> FAISS:
    """Load FAISS vector store from disk (cached)."""
    print("🔹 Loading FAISS vector store into memory...")
    
    embeddings = load_embedding_model(EMBEDDING_MODEL_NAME)

    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
# =====================================================
# 4️⃣ LOAD OR CREATE VECTOR STORE
# =====================================================
def load_or_create_vector_store(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    index_path: str
) -> FAISS:
    """Create or load FAISS vector store."""
    if os.path.exists(os.path.join(index_path, "index.faiss")):
        vector_store = load_vector_store_cached(index_path)
    else:
        print("🔹 Creating FAISS index...")
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(index_path)
        print("✅ FAISS index created and saved.")
    return vector_store

# =====================================================
# 6️⃣A CHUNK-LEVEL RETRIEVER (DEBUG FRIENDLY)
# =====================================================
def retrieve_top_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = 5):
    """
    Retrieve top-k most relevant chunks with metadata.
    """
    results = vector_store.similarity_search_with_score(query, k=top_k)
    retrieved_chunks = []
    for rank, (doc, distance) in enumerate(results, start=1):
        similarity = round(1 - float(distance), 4)
        retrieved_chunks.append({
            "rank": rank,
            "title": doc.metadata.get("title"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "total_chunks": doc.metadata.get("total_chunks"),
            "similarity": similarity,
            "chunk_text": doc.page_content})
    return retrieved_chunks


# =====================================================
# 6️⃣ SIMPLE QUERY RECOMMENDER
# =====================================================
def recommend_titles(
    vector_store: FAISS,
    query: str,
    top_k: int = 5):
    results = vector_store.similarity_search_with_score(query, k=top_k)
    recommendations = []
    similarity_score_ = []
    for doc, distance in results:
        similarity = round(float(distance), 4)
        similarity_score_.append(similarity)
        recommendations.append(f"{doc.metadata.get('title')}  →  {similarity}")
    return recommendations, similarity_score_

def answer_with_rag(
    vector_store: FAISS,
    query: str,
    similarity_threshold: float = 1.95,
    top_k: int = 5):

    # 1️⃣ Retrieve chunks
    top_chunks = retrieve_top_chunks(vector_store, query, top_k=top_k)
    if not top_chunks:
        return None, "No chunks retrieved"
    # 2️⃣ Confidence check
    if top_chunks[0]["similarity"] > similarity_threshold:
        return None, "Low similarity"
    
    print("top_chunks -------------------------------------- : ",top_chunks)
    # scores = [float(score) for doc, score in top_chunks]
    context = "\n\n".join(
        f"Title: {chunk['title']}\n{chunk['chunk_text']}"
        for chunk in top_chunks)
    print("1111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
    print("context  : ",context)
    print("1111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
    # # 3️⃣ Build prompt
    # qween_32B_prompt = build_rag_prompt_for_groq_qween_32B(context, query)
    # # For bangla answer
    # # qween_32B_prompt = build_rag_prompt_for_groq_qween_32B_bn(context, query)
    # # 4️⃣ Ask LLM using qween 32B model using groq api
    # qween_32B_llm = load_llm()
    # chain = qween_32B_prompt | qween_32B_llm
    
    # response = chain.invoke({"context": context, "query": query})
    # return response.content.strip(), top_chunks

    # 4️⃣ Ask LLM using local qween_0.6B model
    fall_back_answer = fallback_llm_bot.generate(query,context)
    return fall_back_answer, top_chunks

def split_think(text):
        """
        Splits a message into:
        1. content inside <think>...</think>
        2. content outside <think>...</think>
        Returns a tuple: (inside_think, outside_think)
        """

        # Extract content inside <think>...</think>
        inside_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        inside = inside_match.group(1).strip() if inside_match else ""
        # Remove the <think>...</think> block
        outside = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return inside, outside

def split_query_score(text: str):
    """
    Split a string in the format 'query : score' into a tuple (query, score).
    Returns (str, float)
    """
    if ':' not in text:
        return text, 0.2
    query_part, score_part = text.rsplit(':', 1)  # split only on the last colon
    query = query_part.strip()
    try:
        score = float(score_part.strip())
    except ValueError:
        raise ValueError(f"Cannot convert score '{score_part.strip()}' to float.")
    return query, score

 
# =====================================================
# 7️⃣ Global Declare for fast response
# =====================================================
DATA_FOLDER = "/home/gflml/Chatbot/dataset/RAG_data"
data = load_rag_data(DATA_FOLDER)
# Step 1: Prepare base documents
raw_documents = prepare_documents(data)
# Step 2: Chunk documents
documents = chunk_documents(
    raw_documents,
    chunk_size=1000,
    chunk_overlap=200)
documents
# Step 3: Load embeddings
embeddings = load_embedding_model(EMBEDDING_MODEL_NAME)
# Step 4: Load / create vector store
vector_store = load_or_create_vector_store(
    documents,
    embeddings,
    VECTOR_STORE_DIR)

# =====================================================
# 7️⃣ Get recommend titles
# =====================================================
def get_recommend_titles(selected_word):
    recommendations, similarity_score_ = recommend_titles(vector_store, selected_word)
    return recommendations, similarity_score_

def is_no_answer(answer: str) -> bool:
    if not answer:
        return True

    no_answer_phrases = [
        "not available in the provided data",
        "not found in the provided data",
        "no relevant information",
        "cannot find",
        "not mentioned in the context"
    ]

    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in no_answer_phrases)

def fallback_answer(query):
    print("No related information found in our database")
    fall_back_answer = fallback_llm_bot.generate(
        query,
        "You are a helpful AI assistant")
    outside_ans = (
        "⚠️ This answer is NOT from our database. "
        "It is generated by a general-purpose LLM (Qwen):\n\n"
        + fall_back_answer)
    print(outside_ans)
    return outside_ans


# =====================================================
# 7️⃣ MAIN ENTRY POINT
# =====================================================
def get_answer_using_RAG(query):
    start = time.time()
    recommendations, similarity_score_ = recommend_titles(vector_store, query)
    query, score = split_query_score(query)
    outside_ans = None
    print("Inside query_answer.py -------------------------------------- :", query, "similarity_score_:", similarity_score_)
    if similarity_score_[0] < 1.1:
        answer, chunks = answer_with_rag(vector_store, query)
        if answer and not is_no_answer(answer):
            _, outside_ans = split_think(answer)
            print("\n🤖 RAG Answer:\n")
            print("answer:", outside_ans)
            print("-" * 60)

            print("\n🔍 Top Responsible Chunks:\n")
            for c in chunks:
                print(f"Rank {c['rank']}")
                print(f"Title        : {c['title']}")
                print(f"Chunk        : {c['chunk_id'] + 1}/{c['total_chunks']}")
                print(f"Similarity   : {c['similarity']}")
                print(f"Text Preview : {c['chunk_text'][:500]}...")
                print("-" * 60)

        else:
            # ⬇️ RAG failed → fallback
            outside_ans = fallback_answer(query)

    else:
        outside_ans = fallback_answer(query)

    latency = round(time.time() - start, 3)
    print(f"\n⏱ Time: {latency}s")
    print("-" * 50)

    return outside_ans






#  while True:
#         query = input("Ask something (or 'exit'): ").strip()
#         if query.lower() == "exit":
#             break

#         start = time.time()
#         recommendations, similarity_score_ = recommend_titles(vector_store, query)
#         if similarity_score_[0]>0:
#             answer, chunks = answer_with_rag(vector_store, query)
#             if answer:
#                 inside, outside_ans = split_think(answer)
#                 print("\n🤖 RAG Answer:\n")
#                 print("answer : ",outside_ans)
#                 print("-" * 60)
#                 # print("Chunks : ",chunks)
#                 # print("-" * 60)
#             for r in recommendations:
#                 print("•", r)

#             latency = round(time.time() - start, 3)
#             print(f"\n⏱ Time: {latency}s")
#             print("\n🔍 Top 5 Responsible Chunks:\n")
#             for c in chunks:
#                 print(f"Rank {c['rank']}")
#                 print(f"Title        : {c['title']}")
#                 print(f"Chunk        : {c['chunk_id'] + 1}/{c['total_chunks']}")
#                 print(f"Similarity   : {c['similarity']}")
#                 print(f"Text Preview : {c['chunk_text'][:200]}...")
#                 print("-" * 60)
#         else:
#             print("No related Information found in our database")
#             fallback_llm_bot = fall_back_llm()
#             thinking, fall_back_answer = fallback_llm_bot.generate(query)
#             outside_ans = "From model (ChatGPT) model Not fom our DataBase -------------------------------- :  " + fall_back_answer
#             print(outside_ans)

#         latency = round(time.time() - start, 3)
#         print(f"\n⏱ Time: {latency}s")
#         print("-" * 50)
#     return outside_ans

# ans = get_answer_using_RAG()


