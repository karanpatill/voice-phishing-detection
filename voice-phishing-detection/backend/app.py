from fastapi import FastAPI, UploadFile
import whisper
from classifier.predict import classify
import os

app = FastAPI()
model = whisper.load_model("base")
@app.get("/")
async def root():
    return {"message": "Voice Phishing Detection API"}  

@app.post("/analyze/")
async def analyze(file: UploadFile):
    with open("temp.wav", "wb") as f:
        f.write(await file.read())
    result = model.transcribe("temp.wav")
    prediction = classify(result["text"])
    return {"transcript": result["text"], "prediction": prediction}