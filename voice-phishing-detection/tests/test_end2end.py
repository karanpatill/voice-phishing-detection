import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health_endpoint():
    """Verify production healthcheck endpoint returns 200 OK"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_cached" in data

def test_realtime_page_serves_html():
    """Verify /realtime serves valid HTML dashboard"""
    response = client.get("/realtime")
    assert response.status_code == 200
    assert "VoiceShield" in response.text
    assert "Demo: Bank Scam (Phishing)" in response.text

def test_demo_simulation_scam():
    """Verify demo simulation for scam call returns high phishing probability"""
    response = client.post("/api/demo/simulate", data={"scenario": "scam"})
    assert response.status_code == 200
    data = response.json()
    assert "call_id" in data
    assert "transcript" in data
    assert data["prediction"]["phishing"] > 0.6
