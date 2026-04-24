# query_image.py
import json
import os
import re
from rapidfuzz import fuzz

IMAGE_INDEX_PATH = "/home/gflml/Chatbot/multi_modal_chatbot_new/dataset/images/image_index.json"
# -----------------------------
# 🔹 Normalize text
# -----------------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[_\-]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def needs_images(query: str) -> bool:
    keywords = [
        "show", "display", "image", "images"
        ,"disease", "symptom", "leaf", "spot"
    ]
    return any(k in query.lower() for k in keywords)

def load_image_db():
    with open(IMAGE_INDEX_PATH) as f:
        return json.load(f)

# ✅ PREVIEW MODE → ONE image per disease
def get_images_only_for_crop_name(crop: str):
    db = load_image_db()
    crop_data = db.get(crop.lower(), {})
    images = []
    for disease, paths in crop_data.items():
        if paths:
            images.append(paths[0])   # ONLY first image
    return images


# -----------------------------
# 🔹 Find BEST match for ONE disease
# -----------------------------
def find_best_match_for_crop(crop: str, disease: str, threshold: int = 80):
    db = load_image_db()
    crop_data = db.get(crop.lower())
    if not crop_data:
        return None
    disease_query = normalize(disease)
    best_match = None
    best_score = 0
    for disease_name in crop_data.keys():
        disease_db = normalize(disease_name)
        score = max(
            fuzz.token_set_ratio(disease_query, disease_db),
            fuzz.partial_ratio(disease_query, disease_db))
        if score > best_score:
            best_score = score
            best_match = disease_name
    if best_score >= threshold:
        return best_match
    return None


# -----------------------------
# 🔹 MULTI-DISEASE HANDLER
# -----------------------------
def get_images_for_crop_and_diseases(crop: str, diseases: list):
    db = load_image_db()
    crop_data = db.get(crop.lower(), {})
    if not crop_data:
        return {
            "matched_diseases": [],
            "images": []}
    matched_diseases = set()
    all_images = []
    for disease in diseases:
        best_match = find_best_match_for_crop(crop, disease)
        if best_match and best_match not in matched_diseases:
            matched_diseases.add(best_match)
            images = crop_data.get(best_match, [])
            all_images.extend(images)
    # Remove duplicate images
    all_images = list(set(all_images))
    return {
        "matched_diseases": list(matched_diseases),
        "images": all_images }


def normalize_strict(text: str) -> str:
    return text.lower().replace("_", "").replace("-", "").replace(" ", "").strip()

def get_disease_images(crop: str, disease: str):
    db = load_image_db()
    crop_data = db.get(crop.lower(), {})
    disease_norm = normalize_strict(disease)
    for disease_key, paths in crop_data.items():
        key_norm = normalize_strict(disease_key)
        # flexible matching
        if disease_norm in key_norm or key_norm in disease_norm:
            return paths
    return []

def check_is_there_image(chunks): # check if there has image with that answer
    there_has_image = False
    if len(chunks) == 0:
        return there_has_image
    if len(chunks)>1:
        if chunks[0]['has_images']==True or chunks[1]['has_images']==True:
            there_has_image = True
    else:
        if chunks[0]['has_images']==True:
            there_has_image = True
    return there_has_image

# ✅ Used by streaming API
def get_disease_images_from_query(chunks:list, crop_name_for_image_path:str, disease_or_pest_name:list):
    if chunks[0]['has_images'] == 'False':
        return []
    if crop_name_for_image_path and len(disease_or_pest_name) == 0:
        # Always show preview images (1 per disease)
        images_path = get_images_only_for_crop_name(crop_name_for_image_path)
    elif crop_name_for_image_path and len(disease_or_pest_name) > 0:
        images_path = get_images_for_crop_and_diseases(crop_name_for_image_path, disease_or_pest_name)
        images_path = images_path['images']
    else:
        images_path = []
    return images_path
