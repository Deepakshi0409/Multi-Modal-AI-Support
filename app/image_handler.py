import pytesseract
from PIL import Image
from app.text_handler import classify_text
import os

# OPTIONAL: Only needed on Windows to set tesseract path manually
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_and_classify(image_path):
    try:
        # Load image and extract text
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)

        if not extracted_text.strip():
            return {"error": "No readable text found in image."}

        # Reuse the NLP classifier
        result = classify_text(extracted_text)

        return {
            "extracted_text": extracted_text.strip(),
            "classification": result
        }

    except Exception as e:
        return {"error": str(e)}
