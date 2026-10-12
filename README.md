# 🛡️ VoiceShield — Real-Time AI Voice Phishing & Scam Detection System

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI_Whisper-00A67E?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace_DistilBERT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
</p>

**VoiceShield** is an AI-powered voice stream analysis platform that listens to live telephone/VoIP audio streams in real-time, transcribes conversations using OpenAI's **Whisper**, and evaluates the threat level using a fine-tuned **DistilBERT** NLP classifier. If fraudulent intent (such as bank OTP theft, urgency triggers, or identity spoofing) is detected, VoiceShield immediately triggers visual alarms and audio siren warnings to alert the victim before they can compromise sensitive credentials.

---

## 🌟 Key Highlights & Demo Features

- ⚡ **Real-Time Speech-to-Text**: High-accuracy local chunk transcription powered by OpenAI Whisper (`tiny`/`base`).
- 🧠 **Contextual Phishing Classification**: Fine-tuned DistilBERT transformer outputs confidence scores (e.g., 99.8% Critical Risk).
- 🚨 **Multi-Tier Threat Alert System**:
  - Safe (`< 30%`): Green status indicator.
  - Suspicious (`30% - 60%`): Amber status indicator.
  - Critical Phishing (`> 60%`): Flashing Red Cyber Warning + Audio Siren Alert.
- 🎯 **1-Click Judge & Hackathon Demo**:
  - Built-in simulation buttons on the web dashboard to instantly test **Bank OTP Scam** vs **Normal Business Call** with zero mic setup needed.
- 🎙️ **Live In-Browser Mic Detection**: Test your own voice directly through any phone or laptop browser via Web Audio API.
- ☁️ **Cloud & Offline Resilient**: Runs in-memory / local fallback when offline, or connects seamlessly to Supabase when credentials are provided.

---

## 🏗️ System Architecture

```
                  ┌───────────────────────────────────────────────┐
                  │           Audio Stream Input                  │
                  │   (Browser Mic / VoIP Stream / Demo Simulator)│
                  └───────────────────────┬───────────────────────┘
                                          │ Audio Chunks (.wav)
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │           FastAPI Backend Engine              │
                  ├───────────────────────────────────────────────┤
                  │ 1. Whisper ASR: Audio ➡️ Text Transcript      │
                  │ 2. DistilBERT: Text ➡️ Phishing Probability   │
                  │ 3. Keyword Heuristics: Flag High-Risk Terms   │
                  │ 4. Storage Engine: Supabase / Local Fallback  │
                  └───────────────────────┬───────────────────────┘
                                          │ WebSocket Broadcast
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │          Real-Time Threat Dashboard           │
                  │  - Live Transcript Streaming                  │
                  │  - Threat Confidence Meter (0 - 100%)         │
                  │  - Trigger Phrase Highlight                   │
                  │  - Real-Time Siren Audio Interception         │
                  └───────────────────────────────────────────────┘
```

---

## 🚀 Live Cloud Deployment (Render / Docker)

### Option 1: 1-Click Blueprint on Render
1. Push this repository to your GitHub account.
2. Sign in to [Render](https://render.com).
3. Click **New +** ➡️ **Blueprint**.
4. Select your `voice-phishing-detection` repository.
5. Render will automatically detect `render.yaml` and build the container using `Dockerfile`.
6. Once deployed, you will get a permanent public HTTPS/WSS URL:  
   `https://voiceshield-production.onrender.com/realtime`

### Option 2: Docker Container Build
```bash
# Build the production Docker image
docker build -t voiceshield .

# Run the container locally
docker run -p 10000:10000 voiceshield
```
Open `http://localhost:10000/realtime` in your browser.

---

## 💻 Local Development Setup

### 1. Prerequisites
- Python 3.11+
- FFmpeg installed (bundled automatically via `imageio-ffmpeg` on Windows)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/karanpatill/voice-phishing-detection.git
cd voice-phishing-detection/voice-phishing-detection

# Create and activate virtual environment
python -m venv backend/venv
# Windows:
backend\venv\Scripts\activate
# Linux/macOS:
source backend/venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Run Development Server
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
Access the dashboard at:
- **Real-Time Monitor**: [http://127.0.0.1:8000/realtime](http://127.0.0.1:8000/realtime)
- **Manual Upload**: [http://127.0.0.1:8000/upload](http://127.0.0.1:8000/upload)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Container health check and model cache status |
| `POST` | `/api/demo/simulate` | 1-Click demo simulation (`scenario=scam` or `scenario=normal`) |
| `POST` | `/start_call_monitoring/` | Initializes a new monitoring session for a call |
| `POST` | `/upload_realtime_chunk/` | Transcribes audio chunk, computes threat score, and broadcasts alert |
| `WS` | `/ws/call_monitoring/{call_id}` | WebSocket stream for live transcript and alert updates |
| `POST` | `/finalize_call/` | Finalizes call and returns risk summary |

---

## 👥 Hackathon Team & Credits
- **Project**: VoiceShield
- **Built for**: Cyber Security & Anti-Fraud Hackathon Innovation
- **License**: MIT
