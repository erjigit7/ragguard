import torch
from sentence_transformers.cross_encoder import CrossEncoder

# The English RagGuard uses answerdotai/ModernBERT-base, but ModernBERT's
# vocabulary has essentially no Kyrgyz Cyrillic tokens -- confirmed
# empirically: a one-sentence Kyrgyz test string tokenized into 65 raw
# UTF-8 byte-fragments ('Ðļ', 'Ñĭ', 'ÑĢ', ...) with ModernBERT, vs 26 clean
# Cyrillic subword tokens ('▁Кыргызстанга', '▁эртең', ...) with
# xlm-roberta-base. Using ModernBERT here would make the model learn
# Kyrgyz essentially from raw bytes with no pretrained representations.
BASE_MODEL = "xlm-roberta-base"
MAX_LENGTH = 512
DEFAULT_MODEL_PATH = "models/ragguard-kyrgyz-v1"


def new_model():
    """A fresh, untrained CrossEncoder ready for fine-tuning on the Kyrgyz dataset."""
    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=MAX_LENGTH, activation_fn=torch.nn.Sigmoid())
    return model


def load_model(model_path=DEFAULT_MODEL_PATH):
    return CrossEncoder(model_path, activation_fn=torch.nn.Sigmoid())


def check_groundedness(model, context, answer, threshold=0.5):
    """Score how well `answer` is supported by `context`. Higher score = more grounded."""
    score = float(model.predict([(context, answer)])[0])
    return {"grounded": score >= threshold, "score": round(score, 4)}
