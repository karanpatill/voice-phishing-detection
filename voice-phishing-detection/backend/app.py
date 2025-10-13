import asyncio
import json
import os
import time
import uuid
from dotenv import load_dotenv
from typing import Dict, List

import torch
import whisper
from fastapi import FastAPI, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from supabase import Client, create_client
from transformers import (DistilBertForSequenceClassification, # type: ignore
                          DistilBertTokenizer)

load_dotenv()
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
tokenizer = DistilBertTokenizer.from_pretrained("./classifier/saved_model")
bert_model = DistilBertForSequenceClassification.from_pretrained("./classifier/saved_model")

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()

# ----------------------------
# Real-time Call Management
# ----------------------------
call_sessions: Dict[str, Dict] = {}
active_connections: Dict[str, List[WebSocket]] = {}

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "web")
if not os.path.exists(WEB_DIR):
    raise RuntimeError(f"Directory '{WEB_DIR}' does not exist")

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# --- Core Logic Helper Function ---
async def _process_and_store_chunk(file: UploadFile, call_id: str, chunk_number: int):
    """
    This single function now contains all the core logic for processing a chunk.
    It's called by both the real-time and standard upload endpoints.
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
    outputs = bert_model(**inputs)  # type: ignore
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


@app.get("/")
async def root():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/realtime")
async def realtime_dashboard():
    return FileResponse(os.path.join(WEB_DIR, "realtime.html"))


@app.post("/upload_chunk/")
async def upload_chunk(
    file: UploadFile,
    call_id: str = Form(...),
    chunk_number: int = Form(...),
):
    """
    Standard endpoint for uploading a single chunk.
    It now simply calls the core processing function.
    """
    return await _process_and_store_chunk(file, call_id, chunk_number)

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
    call_id: str = Form(...),
    chunk_number: int = Form(...)
):
    """
    Handles real-time chunks. It calls the SAME core logic function
    to ensure data is processed and stored in the database, then adds
    real-time WebSocket notifications.
    """
    # Step 1: Process and store the chunk using the shared helper function
    result = await _process_and_store_chunk(file, call_id, chunk_number)

    # Step 2: Add real-time specific data
    result["timestamp"] = time.time()
    if call_id in call_sessions:
        call_sessions[call_id]["chunks"].append(result)

    # Step 3: Check for phishing and broadcast alerts
    phishing_score = result["prediction"]["phishing"]
    if phishing_score > 0.6:  # High phishing threshold
        alert = await handle_phishing_alert(call_id, result)
        result["alert"] = alert

    # Step 4: Broadcast the analysis update to WebSocket clients
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