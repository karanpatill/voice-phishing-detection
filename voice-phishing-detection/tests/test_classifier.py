import pytest
import torch
from backend.app import get_bert_model

def test_bert_model_loading():
    """Verify tokenizer and DistilBERT model load successfully"""
    tokenizer, model = get_bert_model()
    assert tokenizer is not None
    assert model is not None

def test_phishing_text_prediction():
    """Verify obvious phishing text is classified with high risk score"""
    tokenizer, model = get_bert_model()
    text = "Hello, this is your bank fraud department. Your account is compromised. Please give me your one time password immediately."
    
    with torch.inference_mode():
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        phishing_score = float(probs[0][1])
        
    assert phishing_score > 0.5, f"Expected phishing score > 0.5, got {phishing_score}"

def test_normal_text_prediction():
    """Verify normal business conversation has low phishing risk"""
    tokenizer, model = get_bert_model()
    text = "Hi John, just following up on the project status report for tomorrow's standup meeting."
    
    with torch.inference_mode():
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        normal_score = float(probs[0][0])
        
    assert normal_score > 0.5, f"Expected normal score > 0.5, got {normal_score}"
