from fastapi import FastAPI, UploadFile, Form
from supabase import create_client, Client
import whisper
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import uuid
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ----------------------------
# Supabase Setup
# ----------------------------
SUPABASE_URL = "https://gjwcexivvjhunbdnhepx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdqd2NleGl2dmpodW5iZG5oZXB4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1OTUwMTQ0OSwiZXhwIjoyMDc1MDc3NDQ5fQ.P4nz5LMH1V3qunT-_lnF_65BvqQJsZ0xiBgVGj_tMXQ"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------
# Load Whisper + DistilBERT
# ----------------------------
whisper_model = whisper.load_model("base")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "web")

if not os.path.exists(WEB_DIR):
    raise RuntimeError(f"Directory '{WEB_DIR}' does not exist")

# Serve static files like JS/CSS/images
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

@app.post("/upload_chunk/")
async def upload_chunk(
    file: UploadFile,
    call_id: str = Form(...),         # call identifier (same for all chunks of 1 call)
    chunk_number: int = Form(...),    # chunk order (0, 1, 2, ...)
):
    """
    Uploads a 5s audio chunk:
    - Save locally
    - Upload to Supabase storage
    - Transcribe with Whisper
    - Detect phishing with DistilBERT
    - Store metadata in Supabase table
    """

    # Step 1: Save chunk locally
    filename = f"{call_id}_{chunk_number}_{uuid.uuid4()}.wav"
    filepath = f"temp/{filename}"
    os.makedirs("temp", exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(await file.read())

    # Step 2: Upload chunk to Supabase Storage
    bucket = "audio-chunks"
    supabase.storage.from_(bucket).upload(filename, filepath)
    file_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}"

    # Step 3: Transcribe with Whisper
    result = whisper_model.transcribe(filepath)
    transcript = result["text"]

    # Step 4: Classify with DistilBERT
    inputs = tokenizer(transcript, return_tensors="pt", truncation=True)
    outputs = bert_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    normal_score = float(probs[0][0])
    phishing_score = float(probs[0][1])

    # Step 5: Insert metadata into Supabase table
    supabase.table("chunks").insert({
        "call_id": call_id,
        "chunk_number": chunk_number,
        "chunk_name": filename,
        "file_url": file_url,
        "transcript": transcript,
        "phishing_score": phishing_score
    }).execute()

    # Step 6: Cleanup local file
    os.remove(filepath)

    return {
        "call_id": call_id,
        "chunk_number": chunk_number,
        "transcript": transcript,
        "prediction": {"normal": normal_score, "phishing": phishing_score},
        "file_url": file_url
    }
