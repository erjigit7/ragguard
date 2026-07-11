import json
import random
from pathlib import Path

from datasets import Dataset

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "kyrgyz_hallucination.jsonl"


def load_kyrgyz_hallucination(test_ratio=0.15, seed=42):
    """Return (train_dataset, test_dataset) of (context, answer, label) pairs.

    Split by context (article), not by individual example -- a
    grounded/hallucinated pair generated from the same article always
    stays together on one side of the split. Splitting them apart would
    let the model see one half of a pair during training and be
    evaluated on the other, which is a softer version of train/test
    leakage (the context itself, not just the label, would already be
    familiar).
    """
    rows = [json.loads(line) for line in open(DATA_PATH, encoding="utf-8")]
    contexts = sorted(set(r["context"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(contexts)
    n_test = max(1, round(len(contexts) * test_ratio))
    test_contexts = set(contexts[:n_test])

    train_rows = [r for r in rows if r["context"] not in test_contexts]
    test_rows = [r for r in rows if r["context"] in test_contexts]

    def to_dataset(rs):
        return Dataset.from_dict({
            "context": [r["context"] for r in rs],
            "answer": [r["answer"] for r in rs],
            "label": [r["label"] for r in rs],
        })

    return to_dataset(train_rows), to_dataset(test_rows)
