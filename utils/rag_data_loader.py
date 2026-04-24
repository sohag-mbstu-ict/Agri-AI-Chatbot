import json
import logging
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =====================================================
# 🔧 LOGGER CONFIG
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

# =====================================================
# 1️⃣ LOAD SINGLE / FOLDER DATA
# =====================================================
def load_rag_data(path_input: str) -> List[Dict]:
    """
    Load raw RAG data from:
    - a single JSON file
    - OR a folder containing multiple JSON files
    """
    path = Path(path_input)
    data: List[Dict] = []
    if not path.exists():
        raise ValueError(f"❌ Path does not exist: {path_input}")

    try:
        # 🔹 Case 1: Single JSON file
        if path.is_file() and path.suffix == ".json":
            logging.info(f"📄 Loading file: {path.name}")
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                data.extend(loaded if isinstance(loaded, list) else [loaded])
        # 🔹 Case 2: Folder with JSON files
        elif path.is_dir():
            json_files = list(path.glob("*.json"))
            if not json_files:
                logging.warning(f"⚠️ No JSON files found in {path}")
                return data
            for file in sorted(json_files):
                logging.info(f"📄 Loading file: {file.name}")
                with open(file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    data.extend(loaded if isinstance(loaded, list) else [loaded])
    except Exception as e:
        logging.error(f"❌ Error loading {path}: {e}")
    return data


# =====================================================
# 2️⃣ LOAD ALL CROPS (AUTO DISCOVERY)
# =====================================================
def load_complete_rag_dataset(rag_data_path: str, crops_base_path: str) -> List[Dict]:
    """
    Load full dataset:
    - rag_data (flat JSON files)
    - crops_data (nested folders → JSON files)

    :param rag_data_path: path to rag_data folder
    :param crops_base_path: path to crops_data folder
    :return: combined data list
    """
    data: List[Dict] = []
    # -----------------------------
    # 1️⃣ LOAD rag_data (FLAT)
    # -----------------------------
    rag_path = Path(rag_data_path)

    if not rag_path.exists():
        raise ValueError(f"❌ rag_data path not found: {rag_data_path}")
    print("\n📂 Loading rag_data (flat JSON files)...")

    rag_loaded = load_rag_data(str(rag_path))
    data.extend(rag_loaded)
    print(f"✅ rag_data loaded: {len(rag_loaded)} records")
    # -----------------------------
    # 2️⃣ LOAD crops_data (NESTED)
    # -----------------------------
    crops_path = Path(crops_base_path)
    if not crops_path.exists():
        raise ValueError(f"❌ crops_data path not found: {crops_base_path}")
    print("\n📂 Loading crops_data (nested folders)...")
    for crop_folder in crops_path.iterdir():
        if not crop_folder.is_dir():
            continue
        print(f"   🌾 Crop: {crop_folder.name}")
        try:
            crop_data = load_rag_data(str(crop_folder))

            if crop_data:
                data.extend(crop_data)
                print(f"      ✅ {len(crop_data)} records")
        except Exception as e:
            print(f"      ❌ Error in {crop_folder.name}: {e}")
    # -----------------------------
    # FINAL
    # -----------------------------
    print(f"\n🚀 TOTAL DATA LOADED: {len(data)}")
    return data

# =====================================================
# 3️⃣ PREPARE DOCUMENTS
# =====================================================
def prepare_documents(data: List[Dict]) -> List[Document]:
    """
    Convert raw JSON into LangChain Documents
    with structured metadata.
    """
    documents: List[Document] = []
    for item in data:
        has_images = item.get("has_images", False)
        disease_name = item.get("disease_or_pest_name", "")
        content = (
            f"Title: {item.get('title', '')}\n"
            f"Category: {item.get('category', '')}\n"
            f"Tags: {', '.join(item.get('tags', []))}\n"
            f"Has Images: {has_images}\n"
            f"Disease/Pest: {disease_name}\n"
            f"Content: {item.get('content', '')}")

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "category": item.get("category"),
                    "tags": item.get("tags", []),
                    "has_images": has_images,
                    "disease_or_pest_name": disease_name,
                }
            )
        )
    logging.info(f"📄 Documents prepared: {len(documents)}")
    return documents


# =====================================================
# 4️⃣ CHUNK DOCUMENTS
# =====================================================
def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    chunked_docs: List[Document] = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                    }
                )
            )

    logging.info(f"🧩 Total chunks created: {len(chunked_docs)}")
    return chunked_docs


# # =====================================================
# # ✅ MAIN EXECUTION (OPTIONAL TEST)
# # =====================================================
# if __name__ == "__main__":

#     BASE_PATH = "/home/gflml/Chatbot/multi_modal_chatbot_new/dataset/RAG_data/crops_data"

#     raw_data = load_all_crops_data(BASE_PATH)

#     documents = prepare_documents(raw_data)

#     chunked_docs = chunk_documents(documents)

#     print(f"\n✅ FINAL STATS:")
#     print(f"Raw records: {len(raw_data)}")
#     print(f"Documents: {len(documents)}")
#     print(f"Chunks: {len(chunked_docs)}")