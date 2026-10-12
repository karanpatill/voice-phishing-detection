import os
import pytest
from backend.app import get_whisper_model

def test_whisper_model_loading():
    """Verify Whisper model loads successfully without crashing"""
    model = get_whisper_model()
    assert model is not None

def test_transcription_on_sample_audio():
    """Verify Whisper can transcribe a sample WAV file"""
    model = get_whisper_model()
    sample_file = os.path.join(os.path.dirname(__file__), "..", "backend", "data", "phish.wav")
    assert os.path.exists(sample_file), f"Sample file not found at {sample_file}"
    
    result = model.transcribe(sample_file)
    assert "text" in result
    assert len(result["text"].strip()) > 0
