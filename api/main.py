from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.text_handler import classify_text
from app.image_handler import extract_and_classify
import shutil
import os

app = FastAPI()

# For text input via POST
class TextRequest(BaseModel):
    message: str

@app.get("/")
def health_check():
    return {"status": "Multi-Modal AI Assistant is live!"}

@app.post("/text")
def handle_text(req: TextRequest):
    result = classify_text(req.message)
    return result

@app.post("/image")
def handle_image(file: UploadFile = File(...)):
    # Save the uploaded image to a temporary file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Pass it to the OCR + classification pipeline
    result = extract_and_classify(temp_path)

    # Delete the file after processing
    os.remove(temp_path)

    return result
