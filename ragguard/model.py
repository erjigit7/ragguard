import torch
from sentence_transformers.cross_encoder import CrossEncoder

BASE_MODEL = "answerdotai/ModernBERT-base"  # long-context (8192 tokens) — RAGTruth passages can be several thousand tokens
MAX_LENGTH = 512  # cut from 2048: long-tail long contexts were making training impractically slow
DEFAULT_MODEL_PATH = "models/ragguard-v1"


def new_model():
    """A fresh, untrained CrossEncoder ready for fine-tuning on RAGTruth."""
    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=MAX_LENGTH, activation_fn=torch.nn.Sigmoid())
    return model


def load_model(model_path=DEFAULT_MODEL_PATH):
    return CrossEncoder(model_path, activation_fn=torch.nn.Sigmoid())


def check_groundedness(model, context, answer, threshold=0.5):
    """Score how well `answer` is supported by `context`. Higher score = more grounded."""
    score = float(model.predict([(context, answer)])[0])
    return {"grounded": score >= threshold, "score": round(score, 4)}
