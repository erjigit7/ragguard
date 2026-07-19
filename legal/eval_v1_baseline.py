"""Замер v1 на юр-тесте (та же выборка, что в train_legal) — базовая строка для таблицы до/после."""

from kyrgyz.data import load_kyrgyz_hallucination
from kyrgyz.model import load_model
from legal.train_legal import load_legal, report

_, legal_test = load_legal()
_, news_test = load_kyrgyz_hallucination()
news_rows = [{"context": c, "answer": a, "label": l, "kind": "news"}
             for c, a, l in zip(news_test["context"], news_test["answer"], news_test["label"])]

model = load_model("models/ragguard-kyrgyz-v1")
report(model, legal_test, "юр-тест по типам (v1 baseline)")
report(model, news_rows, "новостной тест (v1 baseline)")
