import os
import json
from pathlib import Path
import time
import re, shutil
from typing import List, Dict
import unicodedata
from functools import lru_cache
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from utils.query_crop_matching_context import is_query_crop_matching_context,filter_bangla_english_chunk
from utils.bm25_retriever import BM25Retriever
from utils.hybrid_retriever import dense_search, hybrid_search
from utils.hybrid_retrieval_pipeline import hybrid_retrieve
from utils.faiss_reranker_en import faiss_rank_chunks_en
from utils.faiss_reranker_bn import faiss_rank_chunks_bn
from utils.query_preprocess import processed_query
from .groq_llm import build_rag_prompt, load_llm, build_rag_prompt_for_groq_qween_32B, get_fallback_ans_using_qween_32B
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .preprocess_context import is_small_talk, fallback_Not_Found_RAG_answer
from utils.query_preprocess import detect_language
from utils.rag_data_loader import load_rag_data, load_complete_rag_dataset, prepare_documents, chunk_documents
from utils.search_agent import ask_agent
# from .fallback_Qwen3_0_6 import QwenChatbot
from .Qwen3_1_7b import QwenChatbot
# from .Qwen3_4b_t import QwenChatbot

# =====================================================
# CONFIG
# =====================================================
VECTOR_STORE_DIR = "/home/gflml/Chatbot/multi_modal_chatbot_new/vector_store"
# if os.path.exists(VECTOR_STORE_DIR):
#     shutil.rmtree(VECTOR_STORE_DIR)
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "index.faiss")
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "/home/gflml/Chatbot/pretrained_model/embeddings/BAAI_bge_m3"
# EMBEDDING_MODEL_NAME = 'shihab17/bangla-sentence-transformer'
# EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


import torch
torch.backends.cudnn.benchmark = True # faster kernel selection.
@lru_cache(maxsize=1)
def fall_back_llm():
    return QwenChatbot()
fallback_llm_bot = fall_back_llm()

# =====================================================
# 3️⃣ LOAD EMBEDDING MODEL
# =====================================================
@lru_cache(maxsize=1)
def load_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Load and cache embedding model."""
    print("🔹 Loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True})

@lru_cache(maxsize=1)
def load_vector_store_cached(index_path: str) -> FAISS:
    """Load FAISS vector store from disk (cached)."""
    print("🔹 Loading FAISS vector store into memory...")
    embeddings = load_embedding_model(EMBEDDING_MODEL_NAME)
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True)
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
            "has_images":doc.metadata.get("has_images"),
            "disease_or_pest_name":doc.metadata.get("disease_or_pest_name"),
            "chunk_text": doc.page_content})
    return retrieved_chunks


# =====================================================
# 6️⃣ SIMPLE QUERY RECOMMENDER
# =====================================================
def recommend_titles(
    vector_store: FAISS,
    query: str,
    top_k: int = 13):
    results = vector_store.similarity_search_with_score(query, k=top_k)
    recommendations = []
    similarity_score_ = []
    for doc, distance in results:
        similarity = round(float(distance), 4)
        similarity_score_.append(similarity)
        recommendations.append(f"{doc.metadata.get('title')}  →  {similarity}")
    print("Inside recommend_titles recommendations : ",recommendations)
    return recommendations, similarity_score_

def answer_with_rag(
    vector_store: FAISS,
    query: str,
    small_talk,
    similarity_threshold: float = 0.35,
    top_k: int = 3):
    crop_name =[]
    disease_or_pest_name = []
    # 1️⃣ Retrieve chunks
    top_chunks = retrieve_top_chunks(vector_store, query, top_k=top_k)
    is_bn_or_en = detect_language(query)
    if is_bn_or_en == 'english':
        top_chunks, disease_or_pest_name = faiss_rank_chunks_en(top_chunks, query, top_k=5) # boost the similarity score based on disease name, crop name, phrase existance
    else:
        top_chunks, disease_or_pest_name = faiss_rank_chunks_bn(top_chunks, query, top_k=5) # boost the similarity score based on disease name, crop name, phrase existance
    if not top_chunks:
        return None, "Low similarity", crop_name, disease_or_pest_name
    print("top_chunks[0][similarity] -----------%%%%%%%%%%%% : ",top_chunks[0]["similarity"])
    # 2️⃣ Confidence check
    if (top_chunks[0]["similarity"]) < similarity_threshold:
        return None, "Low similarity", crop_name, disease_or_pest_name
    top_chunks = filter_bangla_english_chunk(query,top_chunks)
    if len(top_chunks)<1:
        return None, "Low similarity", crop_name, disease_or_pest_name
    print("Inside answer_with_rag top_chunks111111 -------------------------------------- : ",top_chunks)
    # scores = [float(score) for doc, score in top_chunks]
    context = "\n\n".join(
        # f"Title: {chunk['title']}\n{chunk['chunk_text']}"
        f"\n{chunk['chunk_text']}"
        for chunk in top_chunks)
    if(small_talk):
        start = time.time()# For small talk no need to match crop name between query and content
        answer_ = fallback_llm_bot.generate(query,context)
        end = time.time()
        print(f"⏱ Inference Time for fallback_llm_bot.generate(query,context) : {round(end - start, 3)} seconds")
        return answer_, top_chunks, crop_name, disease_or_pest_name

    # print("1111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
    # print("Inside answer_with_rag context  : ",context)
    # print("1111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
    embeddings = load_embedding_model(EMBEDDING_MODEL_NAME)
    # suppose query has banana then check in chunk title has banana or not
    flag_, selected_top_chunks, has_crop_, crop_name = is_query_crop_matching_context(embeddings,query,top_chunks)
    selected_top_chunks = [chunk for chunk in selected_top_chunks if chunk.get("similarity", 0) > 0.33] # filter based on similarity
    if flag_==False and len(selected_top_chunks) < 1:
        return None, "Low similarity", crop_name, disease_or_pest_name  # Check crop name is in query or not
    else:
        if has_crop_ == "Has_Crop": 
            if (selected_top_chunks[0]["similarity"]) < 0.3:
                return None, "Low similarity", crop_name, disease_or_pest_name
        if (selected_top_chunks[0]["similarity"]) < 0.1:
            return None, "Low similarity", crop_name, disease_or_pest_name
        context = "\n\n".join(
        # f"Title: {chunk['title']}\n{chunk['chunk_text']}"
        f"\n{chunk['chunk_text']}"
        for chunk in selected_top_chunks)
    print("Inside answer_with_rag top_chunks22222 -------------------------------------- : ",selected_top_chunks)
    # 3️⃣ Build prompt
    start = time.time()
    qween_32B_prompt = build_rag_prompt_for_groq_qween_32B(context, query)
    # For bangla answer
    # qween_32B_prompt = build_rag_prompt_for_groq_qween_32B_bn(context, query)
    # 4️⃣ Ask LLM using qween 32B model using groq api
    qween_32B_llm = load_llm()
    chain = qween_32B_prompt | qween_32B_llm
    response = chain.invoke({"context": context, "query": query})
    end = time.time()
    print(f"⏱ Inference Time for fallback_llm_bot.generate(query,context) : {round(end - start, 3)} seconds")
    return response.content.strip(), top_chunks, crop_name, disease_or_pest_name

    # # 4️⃣ Ask LLM using local qween3_1.7B model
    # start = time.time()
    # # query = processed_query(query) # process the query to get better response
    # print("query   $$$$$$$$$$$$$$$$$$$$$$$$$$$$  : ",query)
    # answer_ = fallback_llm_bot.generate(query,context)
    # end = time.time()
    # print(f"⏱ Inference Time for fallback_llm_bot.generate(query,context) : {round(end - start, 3)} seconds")
    # return answer_, selected_top_chunks, crop_name

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
    If no valid float exists after the last colon, remove ':' and return default score 0.2.
    Returns (str, float)
    """
    if ':' not in text:
        return text.strip(), 0.2
    query_part, score_part = text.rsplit(':', 1)
    try:
        score = float(score_part.strip())
        return query_part.strip(), score
    except ValueError:
        # Remove the last colon and merge properly
        cleaned_query = f"{query_part.strip()} {score_part.strip()}"
        return cleaned_query.strip(), 0.2
 
# =====================================================
# 7️⃣ Global Declare for fast response
# =====================================================
data = []
rag_data_path = "/home/gflml/Chatbot/multi_modal_chatbot_new/dataset/RAG_data/rag_data"
crops_base_path = "/home/gflml/Chatbot/multi_modal_chatbot_new/dataset/RAG_data/crops_data"
data = load_complete_rag_dataset(rag_data_path, crops_base_path)

# Step 1: Prepare base documents
raw_documents = prepare_documents(data)

# BM25Retriever_obj = BM25Retriever(raw_documents)
# result1 = BM25Retriever_obj.search("panama disease of banana")
# result2 = BM25Retriever_obj.search("panama of banana")
# result3 = BM25Retriever_obj.search("banana panama")
# result4 = BM25Retriever_obj.search("tell me about panama disease of banana")
# result5 = BM25Retriever_obj.search("panama disease")
# result6 = BM25Retriever_obj.search("panama disease of mango")
# Step 2: Chunk documents
documents = chunk_documents(
    raw_documents,
    chunk_size=1000,
    chunk_overlap=200)
# Step 3: Load embeddings
embeddings = load_embedding_model(EMBEDDING_MODEL_NAME)
# Step 4: Load / create vector store
vector_store = load_or_create_vector_store(
    documents,
    embeddings,
    VECTOR_STORE_DIR)

# query = "rooftop garden"
# result2 = BM25Retriever_obj.search(query)
# bm25_results = hybrid_retrieve(query,vector_store,result2)
# top_chunks_t = retrieve_top_chunks(vector_store, query, top_k=10)
# faiss_ranked = faiss_rank_chunks_en(top_chunks_t, query, top_k=5)
# =====================================================
# 7️⃣ Get recommend titles
# =====================================================
def get_recommend_titles(selected_word):
    recommendations, similarity_score_ = recommend_titles(vector_store, selected_word)
    return recommendations, similarity_score_


def normalize_text(text):
    return unicodedata.normalize("NFC", text)
def is_no_answer(answer: str) -> bool:
    if not answer:
        return True
    no_answer_phrases = [
        # "প্রদত্ত তথ্যে উত্তর",
        "the information is not available in the provided data",
        "not available in the provided data",
        "not found in the provided data",
        "no relevant information",
        "cannot find",
        "not mentioned in the context"]
    answer_norm = normalize_text(answer).lower()
    return any(
        normalize_text(phrase).lower() in answer_norm
        for phrase in no_answer_phrases )

def fallback_answer(query):
    print("No related information found in our database")
    fall_back_answer = fallback_llm_bot.generate(
        query,
        "You are a helpful AI assistant")
    outside_ans = (
        "⚠️ This answer is NOT from our database. "
        "It is generated by a general-purpose LLM (Qwen) ---------------From ChatGPT---------------- :\n\n"
        + fall_back_answer)
    print(outside_ans)
    return outside_ans


# =====================================================
# 7️⃣ MAIN ENTRY POINT
# =====================================================
def get_answer_using_RAG(query):
    query = processed_query(query)
    start = time.time()
    recommendations, similarity_score_ = recommend_titles(vector_store, query)
    query, score = split_query_score(query)
    outside_ans = None
    chunks = None
    crop_name = []
    disease_or_pest_name = []
    print("Inside query_answer.py -------------------------------------- :", query, "similarity_score_:", similarity_score_)
    if similarity_score_[0] < 1:
        small_talk = False
        # if is_small_talk(query, embeddings):
        #     small_talk = True
        #     # Force very low threshold so greeting always matches
        #     answer, chunks, crop_name, disease_or_pest_name = answer_with_rag(vector_store, query,small_talk, 0.05)
        # else:
        is_bn_or_en = detect_language(query)
        if is_bn_or_en == "english":
            answer, chunks, crop_name, disease_or_pest_name = answer_with_rag(vector_store, query,small_talk, 0.33)
        else:
            answer, chunks, crop_name, disease_or_pest_name = answer_with_rag(vector_store, query,small_talk, 0.1)
        if answer is not None:
            _, outside_ans = split_think(answer)
            if is_no_answer(outside_ans):
                # outside_ans = get_fallback_ans_using_qween_32B(query)
                outside_ans = fallback_Not_Found_RAG_answer(query)
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
    # # ---------------------- get answer using qween 32B ---------------
    #     else:
    #         outside_ans = get_fallback_ans_using_qween_32B(query)
    #         inside, outside_ans = split_think(outside_ans)
    # else:
    #     outside_ans = get_fallback_ans_using_qween_32B(query)
    #     inside, outside_ans = split_think(outside_ans)
    # # ---------------------- get answer using qween 32B --------------- 

        else:
            # outside_ans = fallback_Not_Found_RAG_answer(query)
            outside_ans = ask_agent(query)
    else:
        # outside_ans = fallback_Not_Found_RAG_answer(query)
        outside_ans = ask_agent(query)

    latency = round(time.time() - start, 3)
    print(f"\n⏱ Time: {latency}s")
    print("-" * 50)
    return outside_ans, chunks, crop_name, disease_or_pest_name






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


