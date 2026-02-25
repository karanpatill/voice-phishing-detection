import asyncio
import json
import os
import time
import uuid
from dotenv import load_dotenv
from typing import Dict, List

import torch
torch.set_num_threads(1)       # Limit CPU threads to save memory on free tier
torch.set_num_interop_threads(1)
import whisper
from fastapi import FastAPI, Form, UploadFile, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from supabase import Client, create_client
from transformers import (DistilBertForSequenceClassification, # type: ignore
                          DistilBertTokenizer)

load_dotenv()
# ----------------------------
# Supabase Setup
# ----------------------------
# Load from environment variables (set these on Render dashboard)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️  WARNING: SUPABASE_URL and/or SUPABASE_KEY not set. Database features will fail.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# ----------------------------
# Ensure ffmpeg is available (imageio-ffmpeg bundles a static binary)
# ----------------------------
try:
    import imageio_ffmpeg
    ffmpeg_path = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
    print(f"✅ ffmpeg found via imageio-ffmpeg: {imageio_ffmpeg.get_ffmpeg_exe()}")
except ImportError:
    print("ℹ️  imageio-ffmpeg not installed, using system ffmpeg")

# ----------------------------
# Lazy-load ML Models (saves ~300MB at startup)
# ----------------------------
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "classifier", "saved_model")

_models = {}

def get_whisper_model():
    if "whisper" not in _models:
        print("🔄 Loading Whisper tiny model...")
        _models["whisper"] = whisper.load_model("tiny")
        gc.collect()
        print("✅ Whisper model loaded")
    return _models["whisper"]

def get_bert_model():
    if "tokenizer" not in _models:
        print("🔄 Loading DistilBERT tokenizer + model...")
        _models["tokenizer"] = DistilBertTokenizer.from_pretrained(MODEL_DIR)
        _models["bert"] = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        _models["bert"].eval()  # Set to eval mode — saves memory (no grad tracking)
        gc.collect()
        print("✅ DistilBERT model loaded")
    return _models["tokenizer"], _models["bert"]

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="VoiceShield - Voice Phishing Detection")

# CORS middleware for deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Real-time Call Management
# ----------------------------
call_sessions: Dict[str, Dict] = {}
active_connections: Dict[str, List[WebSocket]] = {}

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "web")
if not os.path.exists(WEB_DIR):
    raise RuntimeError(f"Directory '{WEB_DIR}' does not exist")

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")



def _background_sb_upload(filename: str, file_content: bytes, call_id: str, chunk_number: int, transcript: str, phishing_score: float, file_url: str):
    """Background task for Supabase operations"""
    try:
        # Upload chunk to Supabase Storage
        bucket = "audio-chunks"
        supabase.storage.from_(bucket).upload(filename, file_content)
        
        # Insert metadata into Supabase table
        supabase.table("chunks").insert({
            "call_id": call_id,
            "chunk_number": chunk_number,
            "chunk_name": filename,
            "file_url": file_url,
            "transcript": transcript,
            "phishing_score": phishing_score
        }).execute()
        print(f"✅ [Background] Saved chunk {chunk_number} for call {call_id}")
    except Exception as e:
        print(f"❌ [Background] Error saving chunk {chunk_number}: {e}")

async def _process_and_store_chunk(file: UploadFile, call_id: str, chunk_number: int, background_tasks: BackgroundTasks):
    """
    Optimized core logic:
    1. Transcribe & Classify (Blocking, critical for UI)
    2. Offload Storage/DB to BackgroundTasks (Non-critical for real-time UI)
    """
    import tempfile
    
    # Step 1: Read file content into memory
    filename = f"{call_id}_{chunk_number}_{uuid.uuid4()}.wav"
    file_content = await file.read()
    
    # Pre-calculate file URL (predictable)
    bucket = "audio-chunks"
    file_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}"
    
    # Step 2: Transcribe with Whisper (File I/O + Inference)
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    transcript = ""
    try:
        temp_audio.write(file_content)
        temp_audio.flush()
        temp_audio.close()
        result = get_whisper_model().transcribe(temp_audio.name)
        transcript = result["text"]
    except Exception as e:
        print(f"Transcription error: {e}")
        transcript = "[Error during transcription]"
    finally:
        try:
            os.unlink(temp_audio.name)
        except:
            pass
    
    # Step 3: Classify with DistilBERT (no_grad saves memory)
    tokenizer_inst, bert_inst = get_bert_model()
    with torch.no_grad():
        inputs = tokenizer_inst(transcript, return_tensors="pt", truncation=True)
        outputs = bert_inst(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        normal_score = float(probs[0][0])
        phishing_score = float(probs[0][1])
    
    # Step 4: Offload Supabase operations to background
    background_tasks.add_task(
        _background_sb_upload,
        filename,
        file_content,
        call_id,
        chunk_number,
        transcript,
        phishing_score,
        file_url
    )

    return {
        "call_id": call_id,
        "chunk_number": chunk_number,
        "transcript": transcript,
        "prediction": {"normal": normal_score, "phishing": phishing_score},
        "file_url": file_url
    }


@app.get("/")
async def root():
    return FileResponse(os.path.join(WEB_DIR, "realtime.html"))


@app.get("/upload")
async def upload_page():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/realtime")
async def realtime_dashboard():
    return FileResponse(os.path.join(WEB_DIR, "realtime.html"))


@app.post("/upload_chunk/")
async def upload_chunk(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    call_id: str = Form(...),
    chunk_number: int = Form(...),
):
    """
    Standard endpoint for uploading a single chunk.
    It now simply calls the core processing function.
    """
    return await _process_and_store_chunk(file, call_id, chunk_number, background_tasks)

# ----------------------------
# Real-time Call Monitoring Endpoints
# ----------------------------

@app.post("/start_call_monitoring/")
async def start_call_monitoring(call_id: str = Form(...)):
    """Initialize monitoring for a new call"""
    call_sessions[call_id] = {
        "start_time": time.time(),
        "chunks": [],
        "phishing_alerts": [],
        "status": "active"
    }
    await broadcast_to_call(call_id, {
        "type": "call_started",
        "call_id": call_id,
        "timestamp": time.time()
    })
    return {"status": "monitoring_started", "call_id": call_id}

@app.post("/upload_realtime_chunk/")
async def upload_realtime_chunk(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    call_id: str = Form(...),
    chunk_number: int = Form(...)
):
    """Handle real-time audio chunks with WebSocket notifications"""
    
    # Process chunk using existing logic
    result = await upload_chunk(file, background_tasks, call_id, chunk_number)
    
    # Add timestamp for real-time tracking
    result["timestamp"] = time.time()
    
    # Store in call session
    if call_id in call_sessions:
        call_sessions[call_id]["chunks"].append(result)
    
    # Check for phishing and send alerts if needed
    phishing_score = result["prediction"]["phishing"]
    
    if phishing_score > 0.6:  # High phishing threshold
        alert = await handle_phishing_alert(call_id, result)
        result["alert"] = alert
    
    # Broadcast analysis update to WebSocket clients
    await broadcast_to_call(call_id, {
        "type": "analysis_update",
        "call_id": call_id,
        "chunk_number": chunk_number,
        "transcript": result["transcript"],
        "phishing_score": phishing_score,
        "timestamp": result["timestamp"]
    })
    
    return result


@app.post("/finalize_call/")
async def finalize_call(call_id: str = Form(...)):
    """Finalize a call session"""
    if call_id in call_sessions:
        call_sessions[call_id]["status"] = "completed"
        call_sessions[call_id]["end_time"] = time.time()
        
        chunks = call_sessions[call_id]["chunks"]
        total_chunks = len(chunks)
        avg_phishing_score = sum(chunk["prediction"]["phishing"] for chunk in chunks) / total_chunks if total_chunks > 0 else 0
        
        summary = {
            "call_id": call_id,
            "total_chunks": total_chunks,
            "average_phishing_score": avg_phishing_score,
            "alerts_count": len(call_sessions[call_id]["phishing_alerts"]),
            "duration": call_sessions[call_id].get("end_time", time.time()) - call_sessions[call_id]["start_time"]
        }
        
        await broadcast_to_call(call_id, {
            "type": "call_ended",
            "call_id": call_id,
            "summary": summary,
            "timestamp": time.time()
        })
        
        return {"status": "call_finalized", "summary": summary}
    
    return {"status": "call_not_found"}

@app.get("/call_status/{call_id}")
async def get_call_status(call_id: str):
    """Get status of a specific call"""
    if call_id in call_sessions:
        return call_sessions[call_id]
    return {"error": "Call not found"}


@app.get("/active_calls/")
async def get_active_calls():
    """Get all active calls"""
    active_calls = {
        call_id: session for call_id, session in call_sessions.items()
        if session["status"] == "active"
    }
    return {"active_calls": active_calls}

# ----------------------------
# WebSocket Support (No changes needed here)
# ----------------------------
@app.websocket("/ws/call_monitoring/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()
    if call_id not in active_connections:
        active_connections[call_id] = []
    active_connections[call_id].append(websocket)
    print(f"🔌 WebSocket connected for call: {call_id}")
    try:
        if call_id in call_sessions:
            await websocket.send_text(json.dumps({
                "type": "connection_established",
                "call_id": call_id,
                "session_data": call_sessions[call_id]
            }))
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                try:
                    data = json.loads(message)
                    await handle_websocket_message(call_id, websocket, data)
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON received"
                    }))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": time.time()
                }))
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected for call: {call_id}")
    except Exception as e:
        print(f"❌ WebSocket error for call {call_id}: {e}")
    finally:
        if call_id in active_connections:
            active_connections[call_id] = [
                conn for conn in active_connections[call_id] if conn != websocket
            ]
            if not active_connections[call_id]:
                del active_connections[call_id]


async def handle_websocket_message(call_id: str, websocket: WebSocket, data: dict):
    message_type = data.get("type")
    if message_type == "ping":
        await websocket.send_text(json.dumps({
            "type": "pong",
            "timestamp": time.time()
        }))
    elif message_type == "get_status":
        if call_id in call_sessions:
            await websocket.send_text(json.dumps({
                "type": "status_response",
                "call_id": call_id,
                "data": call_sessions[call_id]
            }))


async def broadcast_to_call(call_id: str, message: dict):
    if call_id in active_connections:
        message_text = json.dumps(message)
        disconnected = []
        for websocket in active_connections[call_id]:
            try:
                await websocket.send_text(message_text)
            except Exception as e:
                print(f"❌ Failed to send to WebSocket: {e}")
                disconnected.append(websocket)
        for ws in disconnected:
            active_connections[call_id].remove(ws)


async def handle_phishing_alert(call_id: str, analysis_result: dict):
    phishing_score = analysis_result["prediction"]["phishing"]
    transcript = analysis_result["transcript"]
    alert = {
        "type": "phishing_alert",
        "call_id": call_id,
        "severity": "high" if phishing_score > 0.8 else "medium",
        "transcript": transcript,
        "confidence": phishing_score,
        "timestamp": time.time(),
        "chunk_number": analysis_result["chunk_number"]
    }
    if call_id in call_sessions:
        call_sessions[call_id]["phishing_alerts"].append(alert)
    await broadcast_to_call(call_id, alert)
    print(f"🚨 PHISHING ALERT for call {call_id}: {phishing_score:.1%} confidence")
    return alert

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)