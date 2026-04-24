import streamlit as st
from ddgs import DDGS
import re
import requests
from bs4 import BeautifulSoup

# -----------------------------
# CONFIG
# -----------------------------
BLOCKED_KEYWORDS = [
    "youtube", "apk", "google play", "download",
    "app", "прессе", "авторам"
]

MAX_ARTICLES = 2
MAX_CONTENT_LENGTH = 1500


# -----------------------------
# 🔹 LANGUAGE DETECTION
# -----------------------------
def detect_language(text: str):
    bn = len(re.findall(r"[\u0980-\u09FF]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    return "bangla" if bn >= en else "english"


# -----------------------------
# TEXT CLEANING
# -----------------------------
def smart_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\u0980-\u09FFA-Za-z0-9.,:;()%\- ]", "", text)
    return text.strip()


# -----------------------------
# VALIDATION FILTER (UPDATED)
# -----------------------------
def is_valid_result(result, query_lang):
    text = (result["title"] + " " + result["body"]).lower()

    # ❌ Block junk
    for word in BLOCKED_KEYWORDS:
        if word in text:
            return False

    # 🔥 Language adaptive filtering
    bangla_chars = len(re.findall(r"[\u0980-\u09FF]", text))
    english_chars = len(re.findall(r"[A-Za-z]", text))

    if query_lang == "bangla":
        # Prefer Bangla-heavy content
        return bangla_chars >= english_chars
    else:
        # Allow English or mixed
        return english_chars > 0


# -----------------------------
# FETCH FULL PAGE CONTENT
# -----------------------------
def fetch_full_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        text = smart_clean(text)
        return text[:MAX_CONTENT_LENGTH]

    except Exception:
        return ""


# -----------------------------
# MAIN SEARCH FUNCTION
# -----------------------------
@st.cache_data(ttl=600)
def search_web(query):
    context = ""
    articles_count = 0

    query_lang = detect_language(query)

    # 🔥 Improve search query automatically
    if query_lang == "bangla":
        search_query = query + " কৃষি"
        region = "bd-bn"
    else:
        search_query = query + " agriculture farming"
        region = "us-en"

    with DDGS() as ddgs:
        results = ddgs.text(
            search_query,
            region=region,
            max_results=8
        )

        for r in results:
            if not is_valid_result(r, query_lang):
                continue

            full_text = fetch_full_content(r["href"])

            if full_text:
                context += full_text + "\n\n"
                articles_count += 1

            if articles_count >= MAX_ARTICLES:
                break

    return context.strip()


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Agri Web Search", layout="wide")

st.title("🌾 Agricultural Web Search (Bangla + English)")
st.write("Supports both Bangla and English queries.")

query = st.text_input("🔍 Enter your query:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Fetching data..."):
            result = search_web(query)

        if result:
            st.success("✅ Results fetched")
            st.text_area("📄 Context", result, height=400)
        else:
            st.error("❌ No valid content found")