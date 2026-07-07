# RagGuard

A small, fast classifier that checks whether a RAG-generated answer is actually grounded in its retrieved context — or hallucinated.

## Why

RAG systems still hallucinate: they invent a policy that isn't in the source document, contradict it, or make claims it never supported. Most RAG-based products (support bots, internal knowledge assistants, etc.) ship with no automatic check for this before the answer reaches a user. Real products exist specifically to solve this — [Patronus Lynx](https://www.patronus.ai/blog/lynx-state-of-the-art-open-source-hallucination-detection-model) and [Vectara HHEM](https://huggingface.co/vectara/hallucination_evaluation_model) — which confirms it's a real, valued problem, not a hypothetical one.

RagGuard is a lightweight version of the same idea: a cross-encoder fine-tuned to score `(context, answer)` pairs for groundedness, small enough to run as a fast pre-send check in front of any RAG pipeline — not tied to one specific project or domain.

## How it works

- **Base model:** [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) — long-context (up to 8192 tokens, truncated to 512 for training speed), since retrieved passages can run to several thousand characters.
- **Training data:** [RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed) — 15,090 human-annotated (context, answer) pairs across QA, summarization, and data-to-text tasks, MIT licensed. Label = 1 if grounded, 0 if the annotators flagged an evident conflict with the context or a baseless claim not supported by it.
- **Training objective:** binary cross-entropy on the pair, same mechanism used to train the `cross-encoder/ms-marco-*` reranker family.
- **Serving:** a small FastAPI service — `POST /check {"context": ..., "answer": ...}` → `{"grounded": bool, "score": float}`.

## Results

1 epoch, `max_length=512`, ~5.5 minutes on an RTX 4070 Ti. Evaluated on the RAGTruth test split (2,700 examples):

| Metric | Value |
|---|---|
| Accuracy | 79.1% |
| Precision (hallucinated) | 0.74 |
| Recall (hallucinated) | 0.61 |
| Precision (grounded) | 0.81 |
| Recall (grounded) | 0.89 |

## Project layout

```
ragguard/
  data.py    — load RAGTruth, derive (context, answer, label) pairs
  model.py   — build/load the CrossEncoder, run a groundedness check
scripts/
  train.py     — fine-tune on RAGTruth
  evaluate.py  — accuracy/precision/recall on the RAGTruth test split
api/
  main.py    — FastAPI service exposing /check
tests/
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build; see requirements.txt
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py
python scripts/evaluate.py
```

## Run the API

```bash
uvicorn api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"context": "Returns are accepted within 14 days.", "answer": "You can return the item within 30 days."}'
```

## Tests

```bash
pytest -v
```
