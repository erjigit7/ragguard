"""Дообучение ragguard-kyrgyz на юрдомен + числа словами → ragguard-kyrgyz-v2-legal.

Старт с models/ragguard-kyrgyz-v1 (не с нуля). Данные: legal_hallucination.jsonl
(word-негативы ×3 — их мало и они цель фикса) + весь исходный
kyrgyz_hallucination.jsonl против забывания. Сплит юрдатасета — по статье
(context), как в kyrgyz/data.py.

После обучения — отчёт точности ПО ТИПАМ порчи на held-out юрсплите
(word-негативы до фикса не детектировались вовсе: Δ~0 в guard_sanity)
и на исходном новостном тесте (проверка регрессии).

Usage:  python -m legal.train_legal
"""

import json
import pathlib
import random

from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

from kyrgyz.data import load_kyrgyz_hallucination
from kyrgyz.model import load_model

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "legal_hallucination.jsonl"
OUT = "models/ragguard-kyrgyz-v2-legal"
WORD_OVERSAMPLE = 3


def load_legal(test_ratio=0.15, seed=42):
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines()]
    contexts = sorted({r["context"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(contexts)
    test_ctx = set(contexts[: max(1, round(len(contexts) * test_ratio))])
    train = [r for r in rows if r["context"] not in test_ctx]
    test = [r for r in rows if r["context"] in test_ctx]
    train = train + [r for r in train if r["kind"] == "word"] * (WORD_OVERSAMPLE - 1)
    rng.shuffle(train)
    return train, test


def to_ds(rows):
    return Dataset.from_dict({
        "context": [r["context"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "label": [float(r["label"]) for r in rows],
    })


def report(model, rows, title):
    print(f"\n=== {title} ===")
    by_kind: dict[str, list[tuple[float, int]]] = {}
    scores = model.predict([(r["context"], r["answer"]) for r in rows], batch_size=32, show_progress_bar=False)
    for r, s in zip(rows, scores):
        by_kind.setdefault(r.get("kind", "news"), []).append((float(s), r["label"]))
    for kind, items in sorted(by_kind.items()):
        acc = sum((s >= 0.5) == (lab == 1) for s, lab in items) / len(items)
        neg = [s for s, lab in items if lab == 0]
        pos = [s for s, lab in items if lab == 1]
        print(f"{kind:10s} acc={acc:.3f}  честные avg={sum(pos)/len(pos):.3f}  испорченные avg={sum(neg)/len(neg):.3f}  (n={len(items)})")


def main():
    legal_train, legal_test = load_legal()
    news_train, news_test = load_kyrgyz_hallucination()
    news_test_rows = [{"context": c, "answer": a, "label": l, "kind": "news"}
                      for c, a, l in zip(news_test["context"], news_test["answer"], news_test["label"])]

    model = load_model()
    print("ДО дообучения:")
    report(model, legal_test, "юр-тест по типам (v1)")
    report(model, news_test_rows, "новостной тест (v1)")

    mixed = legal_train + [
        {"context": c, "answer": a, "label": l, "kind": "news"}
        for c, a, l in zip(news_train["context"], news_train["answer"], news_train["label"])
    ]
    random.Random(7).shuffle(mixed)
    print(f"\nTrain: {len(mixed)} примеров (юр {len(legal_train)} + новости {len(news_train)})")

    args = CrossEncoderTrainingArguments(
        output_dir=OUT + "-checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        fp16=True,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
    )
    trainer = CrossEncoderTrainer(
        model=model, args=args,
        train_dataset=to_ds(mixed), eval_dataset=to_ds(legal_test),
        loss=BinaryCrossEntropyLoss(model),
    )
    trainer.train()
    model.save_pretrained(OUT)
    print(f"\nСохранено: {OUT}")

    print("\nПОСЛЕ дообучения:")
    report(model, legal_test, "юр-тест по типам (v2-legal)")
    report(model, news_test_rows, "новостной тест (v2-legal)")


if __name__ == "__main__":
    main()
