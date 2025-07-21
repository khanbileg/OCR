import easyocr
from passporteye import read_mrz
import os

# Path to the passport image
IMAGE_PATH = 'images/0.jpg'  # Update this path as needed

def extract_mrz(image_path):
    mrz = read_mrz(image_path)
    if mrz is not None:
        mrz_data = mrz.to_dict()
        print("MRZ Data:")
        for k, v in mrz_data.items():
            print(f"  {k}: {v}")
        return mrz_data
    else:
        print("No MRZ found.")
        return None

def extract_text_easyocr(image_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    print("\nEasyOCR Detected Text:")
    for bbox, text, conf in results:
        print(f"  Text: {text} (Confidence: {conf:.2f})")
    return results

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"Image file not found: {IMAGE_PATH}")
    else:
        # 1. Extract MRZ
        extract_mrz(IMAGE_PATH)
        # 2. Extract all text using EasyOCR
        extract_text_easyocr(IMAGE_PATH)
