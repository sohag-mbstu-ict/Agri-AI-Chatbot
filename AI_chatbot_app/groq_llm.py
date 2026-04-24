from pathlib import Path
import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import time

# ---------------------------------------
# Load .env
# ---------------------------------------
env_path = Path("/home/gflml/Chatbot/.env")
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found")

@lru_cache(maxsize=1)
# def load_llm(model_name="llama-3.1-8b-instant", temperature=0):
def load_llm(model_name="qwen/qwen3-32b", temperature=0):
    print("🔹 Groq LLM loaded")
    return ChatGroq(
        temperature=temperature,
        groq_api_key=GROQ_API_KEY,
        model_name=model_name
    )


def build_rag_prompt(query: str, chunks: list) -> str:
    context = "\n\n".join(
        [
            f"[Chunk {c['rank']} | Title: {c['title']}]\n{c['chunk_text']}"
            for c in chunks
        ]
    )

    prompt = f"""
            You are an expert assistant.
            Answer ONLY using the provided context.
            If the answer is not found, say "Information not available in the knowledge base."

            Context:
            {context}
            Question:
            {query}
            Answer:
            """
    return prompt.strip()


from langchain_core.prompts import ChatPromptTemplate
import re

def is_bangla(text: str) -> bool:
    """
    Check if the text contains Bangla characters.
    """
    return bool(re.search(r'[\u0980-\u09FF]', text))

def detect_query_language(text: str) -> str:
    """
    Detect whether a query is primarily English or Bangla
    based on character count.

    Returns:
        "english" if English letter count > Bangla letter count
        "bangla" otherwise
    """
    # Count English letters (A-Z, a-z)
    english_count = len(re.findall(r"[A-Za-z]", text))
    # Count Bangla Unicode characters
    bangla_count = len(re.findall(r"[\u0980-\u09FF]", text))
    print(f"English letters: {english_count}")
    print(f"Bangla letters : {bangla_count}")
    if english_count > bangla_count:
        return "english"
    else:
        return "bangla"
    
def build_rag_prompt_for_groq_qween_32B(context, query):
    """
    Build a RAG prompt for Groq Qwen-32B.
    Automatically outputs in Bangla if the query contains Bangla,
    otherwise in English.
    Optimized for clear answers, bullet points, and bold keywords.
    """
    if(detect_query_language(query)=="bangla"):
        system_text = (
            "আপনি একজন দক্ষ এবং সহায়ক RAG অ্যাসিস্ট্যান্ট, "
            "বিশেষজ্ঞ শহুরে কৃষি এবং ছাদবাগান। "
            "আপনি শুধুমাত্র প্রদত্ত তথ্য থেকে উত্তর দেবেন। "
            "প্রদত্ত তথ্যের title ব্যবহার করে প্রাসঙ্গিক অংশ খুঁজে নিতে পারেন, "
            "কিন্তু উত্তরে title বা শিরোনাম লিখবেন না। "
            "কোনো বিশ্লেষণ, ব্যাখ্যা, চিন্তাভাবনা উত্তরে  লিখবেন না।\n"
            "যদি তথ্য পাওয়া না যায়, সঠিকভাবে লিখুন: 'প্রদত্ত তথ্যে উত্তর পাওয়া যায়নি।'\n"
            "• শুধুমাত্র প্রদত্ত তথ্য ব্যবহার করুন।\n"
            "• নতুন তথ্য অনুমান করবেন না।\n"
            "• উত্তর স্পষ্ট এবং প্রাঞ্জল হোক।"
        )

        human_text = (
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "নির্দেশাবলী:\n"
            "1. শুধুমাত্র প্রদত্ত তথ্য ব্যবহার করুন।\n"
            "2. প্রাসঙ্গিক অংশের title বা শিরোনাম উত্তরে লিখবেন না।**\n"
            "3. গুরুত্বপূর্ণ শব্দগুলি **বোল্ড** করুন।\n"
            "4. যদি একাধিক পয়েন্ট থাকে, **বুলেট পয়েন্ট** ব্যবহার করুন।\n"
            "5. উত্তর সংক্ষিপ্ত এবং স্পষ্ট রাখুন।\n\n"
            "উত্তর (বাংলায়):"
        )
    else:
        system_text = (
            "You are a knowledgeable and helpful RAG assistant specialized in Urban Agriculture, "
            "Rooftop Gardening, and general factual knowledge.\n"
            "Guidelines:\n"
            "• Prefer information from the context.\n"
            "• Do NOT invent facts or use external knowledge.\n"
            "• Do NOT include the **title** of the relevant section in your answer.\n"
            "Do not write any analysis, explanation, reasoning, or step-by-step discussion.\n"
            "• Respond clearly and naturally.\n"
            "Your task is to answer questions using the provided context as the primary source. "
            "If the answer is not found, reply exactly: "
            "'The information is not available in the provided data.'"
        )

        human_text = (
            "Context:\n{context}\n\n"
            "Question:\n{query}\n\n"
            "Instructions:\n"
            "1. Answer using the provided context.\n"
            "2. Do not include the **title** of the relevant section.\n"
            "3. Use **bold text** for important terms or names.\n"
            "4. If the answer contains multiple points, present them as **bullet points**.\n"
            "5. Keep the answer concise, clear, and well-structured.\n\n"
            "Answer:"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", human_text)
    ])

    return prompt


def get_fallback_ans_using_qween_32B(query):
    context = "You are helpful Agriculture AI Assistant"
    start = time.time()
    qween_32B_prompt = build_rag_prompt_for_groq_qween_32B(context, query)
    # For bangla answer
    # qween_32B_prompt = build_rag_prompt_for_groq_qween_32B_bn(context, query)
    # 4️⃣ Ask LLM using qween 32B model using groq api
    qween_32B_llm = load_llm()
    system_prompt = "You are a helpful assistant. Answer concisely and clearly. Do not write any analysis, explanation, reasoning, or step-by-step discussion."
    messages = [
        ("system", system_prompt),
        ("human", query),]
    # Collect the streamed response
    answer = "ChatGPT   ------------ : \n "
    for chunk in qween_32B_llm.stream(messages):
        answer += chunk.text()
    end = time.time()
    print(f"⏱ Inference Time for fallback_llm_bot.generate(query,context) : {round(end - start, 3)} seconds")
    return answer.strip()
    

def build_rag_prompt_for_groq_qween_32B_bn(context, query):
    """
    Strict RAG prompt that answers ONLY from context
    and outputs the answer in Bangla.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict RAG assistant specialized in Urban Agriculture and Rooftop Gardening. "
         "You MUST answer questions ONLY using the provided context. "
         "DO NOT add, assume, or infer any information not present in the context. "
         "Your task is to TRANSLATE the answer into Bangla based strictly on the context. "
         "If the answer is not found in the context, respond exactly with: "
         "'প্রদত্ত তথ্যে উত্তর পাওয়া যায়নি।'"),

        ("human",
         "Context (English):\n{context}\n\n"
         "Question:\n{query}\n\n"
         "Instructions:\n"
         "1. Find the answer ONLY from the context.\n"
         "2. Translate the answer into clear, natural Bangla.\n"
         "3. Do NOT add extra explanation.\n"
         "4. If not found, reply exactly: 'প্রদত্ত তথ্যে উত্তর পাওয়া যায়নি।'\n\n"
         "Bangla Answer:")
    ])

    return prompt
