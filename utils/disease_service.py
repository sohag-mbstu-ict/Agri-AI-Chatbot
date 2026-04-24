import json
from functools import lru_cache

@lru_cache(maxsize=1)
def load_disease_db():
    with open(
        "/home/gflml/Chatbot/multi_modal_chatbot/dataset/disease_knowledge.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def get_disease_info(crop: str, disease: str):
    db = load_disease_db()

    crop_data = db.get(crop)
    if not crop_data:
        return None

    disease_data = crop_data.get(disease)
    if not disease_data:
        return None

    return {
        "crop": crop,
        "disease": disease,
        "description": disease_data.get("description", ""),
        "organic_solution": disease_data.get("organic_solution", []),
        "chemical_solution": disease_data.get("chemical_solution", [])
    }
