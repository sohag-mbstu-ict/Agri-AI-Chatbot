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
    "app", "прессе", "авторам", "rutube"
]

TRUSTED_DOMAINS = [
    "krishibatayon.gov.bd",
    "bari.gov.bd",
    "dae.gov.bd",
    "brri.gov.bd"
]

MAX_ARTICLES = 2
MAX_CONTENT_LENGTH = 1500


# -----------------------------
# LANGUAGE DETECTION
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
# DOMAIN FILTER
# -----------------------------
def is_trusted_domain(url):
    return any(domain in url for domain in TRUSTED_DOMAINS)


# -----------------------------
# QUERY RELEVANCE CHECK
# -----------------------------
def is_relevant_to_query(text, query):
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    overlap = query_words & text_words
    return len(overlap) >= 1


# -----------------------------
# VALIDATION FILTER
# -----------------------------
def is_valid_result(result, query, query_lang):
    text = (result["title"] + " " + result["body"]).lower()

    # ❌ Block junk keywords
    for word in BLOCKED_KEYWORDS:
        if word in text:
            return False

    # ❌ Block bad domains explicitly
    if "rutube" in result["href"]:
        return False

    # ✅ Allow only trusted domains (STRICT MODE)
    if not is_trusted_domain(result["href"]):
        return False

    # ✅ Language filtering
    bangla_chars = len(re.findall(r"[\u0980-\u09FF]", text))
    english_chars = len(re.findall(r"[A-Za-z]", text))

    if query_lang == "bangla" and bangla_chars < english_chars:
        return False

    if query_lang == "english" and english_chars == 0:
        return False

    # ✅ Relevance filtering
    if not is_relevant_to_query(text, query):
        return False

    return True


# -----------------------------
# FETCH CLEAN CONTENT
# -----------------------------
def fetch_full_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()

        text = soup.get_text(separator=" ")
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

    # 🔥 Query enhancement
    if query_lang == "bangla":
        search_query = query + " কৃষি রোগ প্রতিকার"
        region = "bd-bn"
    else:
        search_query = query + " agriculture disease treatment"
        region = "us-en"

    with DDGS() as ddgs:
        results = ddgs.text(
            search_query,
            region=region,
            max_results=10
        )

        for r in results:
            if not is_valid_result(r, query, query_lang):
                continue

            full_text = fetch_full_content(r["href"])

            if full_text:
                context += f"Source: {r['href']}\n{full_text}\n\n"
                articles_count += 1

            if articles_count >= MAX_ARTICLES:
                break

    return context.strip()


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Agri Web Search", layout="wide")

st.title("🌾 Agricultural Web Search (Production Ready)")
st.write("Supports Bangla & English queries with high-quality filtering.")

query = st.text_input("🔍 Enter your query:")

if st.button("Search"):
    if not query.strip():
        st.warning("⚠️ Please enter a query")
    else:
        with st.spinner("Fetching high-quality agricultural data..."):
            result = search_web(query)

        if result:
            st.success("✅ Clean Results Ready")
            st.text_area("📄 Context", result, height=400)
        else:
            st.error("❌ No high-quality relevant content found")