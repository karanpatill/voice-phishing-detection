# Voice Phishing Detection 🚨

A real-time AI-powered **voice phishing (vishing) detection system** integrated with Linphone (VoIP softphone) to detect suspicious calls and protect users from fraud.  

This project leverages **OpenAI Whisper** for transcription and **BERT** for phishing classification.

---

## 🛠️ Features

- Real-time detection of voice phishing calls.
- Integration with Linphone to capture live call audio.
- Converts audio into text using Whisper.
- Classifies conversation segments as **phishing** or **normal** using BERT.
- Provides live alerts for suspicious calls.
- Modular architecture: client (Linphone), backend (FastAPI + AI), dashboard (optional).

---

## 📂 Repo Structure

voice-phishing-detection/
├── client/ # Linphone integration & audio chunking
├── backend/ # FastAPI + Whisper + BERT
├── dashboard/ # Web dashboard (optional)
├── data/ # Sample call recordings for testing
├── docs/ # Project docs & demo plan
├── tests/ # Unit & integration tests
├── README.md
└── LICENSE

python
Copy code

---

## 🚀 All-in-One Backend Code

```python
# backend/app.py
from fastapi import FastAPI, UploadFile, File
import shutil, os
import whisper
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI(title="Voice Phishing Detection 🚨")

# -----------------------------
# Setup Whisper Model
# -----------------------------
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

# -----------------------------
# Setup BERT Model
# -----------------------------
print("Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# -----------------------------
# Helper Function
# -----------------------------
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return {"normal": float(probs[0][0]), "phishing": float(probs[0][1])}

# -----------------------------
# Upload & Analyze Endpoint
# -----------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Transcribe audio using Whisper
    result = whisper_model.transcribe(file_path)
    text = result["text"]
    
    # Classify text using BERT
    prediction = classify_text(text)
    
    return {"transcription": text, "prediction": prediction}

# -----------------------------
# Run Server
# -----------------------------
# Command to run: uvicorn app:app --reload
⚡ Installation & Setup
1. Clone the repo
bash
Copy code
git clone https://github.com/<your-username>/voice-phishing-detection.git
cd voice-phishing-detection/backend
2. Setup Python environment
bash
Copy code
python -m venv venv
# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
3. Install dependencies
bash
Copy code
pip install fastapi uvicorn openai-whisper torch torchvision torchaudio transformers
🏃‍♂️ Run Backend Server
bash
Copy code
uvicorn app:app --reload
Server runs at: http://127.0.0.1:8000

Open Swagger docs to test endpoints: http://127.0.0.1:8000/docs

🧪 Testing API
Using Swagger Docs
Go to http://127.0.0.1:8000/docs

Find /predict endpoint.

Click Try it out, upload a .wav file, and click Execute.

Response example:

json
Copy code
{
  "transcription": "Hello, we are calling from your bank...",
  "prediction": {
    "normal": 0.12,
    "phishing": 0.88
  }
}
Using Python Script
python
Copy code
import requests

url = "http://127.0.0.1:8000/predict"
files = {"file": open("sample_audio.wav", "rb")}

response = requests.post(url, files=files)
print(response.json())
🏗️ How It Works
Client (Linphone) records call audio in 10-second chunks.

Backend transcribes audio using Whisper.

BERT model predicts whether the conversation is phishing or normal.

Dashboard (optional) displays live alerts.

📈 Future Improvements
Full real-time RTP audio streaming integration with Linphone SDK.

Enhanced BERT model trained on larger vishing dataset.

Mobile-friendly dashboard for instant alerts.

Integration with notifications for end-users.