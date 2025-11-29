# emotion_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

MODEL_PATH = r"D:/python3/doutorado-2-2025-periodo2/mod_ling_neural/go_emotion_model/microsoft/mdeberta-v3-base/20251110-193529/best_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

EMOCOES = [
    'admiração', 'diversão', 'raiva', 'irritação', 'aprovação', 'carinho',
    'confusão', 'curiosidade', 'desejo', 'decepção', 'desaprovação', 'nojo',
    'constrangimento', 'empolgação', 'medo', 'gratidão', 'pesar', 'alegria',
    'amor', 'nervosismo', 'otimismo', 'orgulho', 'percepção', 'alívio',
    'remorso', 'tristeza', 'surpresa', 'neutro'
]


def prever_emocoes(texto: str):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.sigmoid(logits)[0].cpu().numpy()

    return {emo: float(probs[i]) for i, emo in enumerate(EMOCOES)}
