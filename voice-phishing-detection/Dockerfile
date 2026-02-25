FROM python:3.11-slim

# Install system dependencies (ffmpeg is required by Whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (Render uses $PORT, default 10000)
EXPOSE 10000

# Start the server — Render injects $PORT at runtime
CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-10000}
