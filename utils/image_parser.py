import os

def extract_crop_disease_from_image(image_path: str):
    """
    Example:
    'Beetle_spots Banana.jpg'
    → ('banana', 'beetle spots')
    """
    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]

    if " " not in name:
        return None, None

    disease_part, crop_part = name.split(" ", 1)

    disease = disease_part.replace("_", " ").lower().strip()
    crop = crop_part.lower().strip()

    return crop, disease
