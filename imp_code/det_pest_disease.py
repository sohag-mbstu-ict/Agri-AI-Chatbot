import re
from difflib import get_close_matches
# --------------------------------------------------
# 🔹 TEXT NORMALIZATION
# --------------------------------------------------
def normalize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text
# --------------------------------------------------
# 🔹 DETECT CROP
# --------------------------------------------------
def detect_crop(query: str, CROP_CONFIG):
    query = normalize(query)

    for item in CROP_CONFIG:
        if item["crop"] in query:
            return item
    return None
# --------------------------------------------------
# 🔹 FUZZY MATCH FUNCTION
# --------------------------------------------------
def fuzzy_match(query, candidates, cutoff=0.7):
    matches = []
    for candidate in candidates:
        found = get_close_matches(candidate.lower(), [query], n=1, cutoff=cutoff)
        if found:
            matches.append(candidate)
    return matches

def combine_disease_pest(result_dict):
    return result_dict.get("diseases", []) + result_dict.get("pests", [])

# --------------------------------------------------
# 🔹 MAIN FUNCTION
# --------------------------------------------------
def extract_disease_or_pest(query: str, CROP_CONFIG):
    query_clean = normalize(query)
    crop_item = detect_crop(query_clean, CROP_CONFIG)
    result = {
        "crop": None,
        "diseases": [],
        "pests": []}
    if not crop_item:
        return result
    result["crop"] = crop_item["crop"]
    # 🔹 Exact match (best)
    for disease in crop_item["disease"]:
        if disease.lower() in query_clean:
            result["diseases"].append(disease)
    for pest in crop_item["pest"]:
        if pest.lower() in query_clean:
            result["pests"].append(pest)
    # 🔹 Fallback fuzzy match (for typos)
    if not result["diseases"]:
        result["diseases"] = fuzzy_match(query_clean, crop_item["disease"])
    if not result["pests"]:
        result["pests"] = fuzzy_match(query_clean, crop_item["pest"])
    print("crop disease and pest dictionary : ",result)
    disease_or_pest_list = combine_disease_pest(result) # make a list using disease and pest
    return disease_or_pest_list

# --------------------------------------------------
# 🔹 TEST
# --------------------------------------------------
CROP_CONFIG = [
    {"crop": "banana", "disease": ["panama", "sigatoka"], "pest": ["beetle spot"]},
    {"crop": "chilli", "disease": ["dieback"], "pest": []},
    {"crop": "potato", "disease": ["bacterial wilt", "blight", "leaf curl", "scab"], "pest": []},
    {"crop": "maize", "disease": ["leaf blight"], "pest": ["fall armyworm", "cutworm"]},
    {"crop": "wheat", "disease": ["leaf blast", "leaf blight", "root rot"], "pest": []},
    {"crop": "mango", "disease": ["mealybug", "stem end rot", "malformation", "dieback", "anthracnose"],
                                 "pest": ["fruit fly", "weevil"]},
    {"crop": "rice", "disease": ["Neck Blast","Node Blast","Leaf Blast","False Smut","Deadheart","Sheath rot"],
             "pest": ["BPH","Stem Borer Egg Mass","Leaf Folder","Stem borer Moth","Stem Borer Larva","Rice bug"]},]

queries = [
    "major diseASes of rices",
    "rice slEaf BLasts BPH folder blast leaf ricebug problem",
    "rice folder blast leafblasts ricebug Stem borer egg mas problem",
    "bAnaNas sigaTOkas disease",
    "rice Stem borer",
    "Pest atTack in mAize fall aRmy WOrm",
    "pOTAto BliGHts Issue"  # typo test
]

for q in queries:
    print("\nQuery:", q)
    disease_or_pest_list = extract_disease_or_pest(q,CROP_CONFIG)
    print("disease_or_pest_list : ",disease_or_pest_list)

