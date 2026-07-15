# RagGuard

A small, fast classifier that checks whether a RAG-generated answer is actually grounded in its retrieved context — or hallucinated.

> Used in production by [Myizam](https://github.com/erjigit7/myizam) — a legal RAG assistant for Kyrgyzstan: every answer passes RagGuard before reaching the user.

Two detectors live here: an **English** one trained on human-labeled data (RAGTruth), and a **Kyrgyz** one — likely the first of its kind — trained on a dataset that had to be created from scratch, including fine-tuning a separate LLM just to generate it (see [KyrgyzLLM](https://github.com/erjigit7/kyrgyzllm)).

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

## RagGuard-Kyrgyz

The same idea for the Kyrgyz language — where **nothing existed**: no labeled dataset, and no open model that writes Kyrgyz well enough to generate one (the closest relative, KazLLM, answers in Kazakh even when explicitly told not to).

**How the dataset was made** (`kyrgyz/generate_dataset.py`, output in `data/kyrgyz_hallucination.jsonl` — 3,000 pairs / 6,000 examples):

- **Grounded answers (label=1):** written by [KyrgyzLLM](https://github.com/erjigit7/kyrgyzllm) — KazLLM-8B fine-tuned (LoRA continued pretraining + SFT on XLSum) specifically for this — summarizing real articles from the [Kyrgyz News Corpus](https://huggingface.co/datasets/the-cramer-project/Kyrgyz_News_Corpus).
- **Hallucinated answers (label=0):** built programmatically from the grounded summary, ~50/50 between two corruption types: perturb one number (with special handling so years shift by 1–6 and days of month stay within 1–31), or append a plausible-sounding sentence borrowed from a different article's summary — an unsupported claim, RAGTruth's "baseless info" category. Programmatic, because the fine-tuned generator turned out to be *incapable of lying on request*: asked to "deliberately corrupt one fact", it returns output byte-identical to the truthful prompt.

**The detector** (`kyrgyz/model.py`) is `xlm-roberta-base`, not ModernBERT: ModernBERT's tokenizer shatters Kyrgyz Cyrillic into meaningless byte fragments (65 junk tokens for a sentence XLM-R covers in 26 clean subwords). Train/test split is by article, so a grounded/hallucinated pair never straddles the split.

**Results** — evaluated on 900 held-out examples:

| Metric | Value |
|---|---|
| Accuracy | 97.9% |
| Precision / Recall (hallucinated) | 0.97 / 0.98 |
| Precision / Recall (grounded) | 0.98 / 0.97 |

An honest caveat: these negatives are *synthetic*, which is an easier task than RAGTruth's human-labeled, naturally-occurring hallucinations — so this number is not comparable to the English 79.1%.

```bash
python -m kyrgyz.train       # train the Kyrgyz detector
python -m kyrgyz.evaluate    # metrics on the held-out split
```

## Project layout

```
ragguard/
  data.py    — load RAGTruth, derive (context, answer, label) pairs
  model.py   — build/load the CrossEncoder, run a groundedness check
scripts/
  train.py     — fine-tune on RAGTruth
  evaluate.py  — accuracy/precision/recall on the RAGTruth test split
kyrgyz/
  generate_dataset.py — build the Kyrgyz dataset (KyrgyzLLM + programmatic corruption)
  data.py             — load the dataset, split by article
  model.py            — XLM-R cross-encoder for Kyrgyz
  train.py / evaluate.py
data/
  kyrgyz_hallucination.jsonl — 3,000 (context, answer, label) pairs
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
python -m scripts.train
python -m scripts.evaluate
```

(Run from the repo root — `python scripts/train.py` breaks the `ragguard` package import.)

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

## License

The RagGuard code and the trained detector weights (both English and Kyrgyz — plain `xlm-roberta-base`/`ModernBERT-base` fine-tunes) are free to use. One provenance note: the *Kyrgyz dataset's* grounded summaries were generated by [KyrgyzLLM](https://github.com/erjigit7/kyrgyzllm), a fine-tune of ISSAI's KazLLM-8B, which is licensed **CC BY-NC 4.0 (non-commercial only)**. This doesn't restrict the detector itself, but if you're building on the raw dataset text, be aware of that upstream restriction — see KyrgyzLLM's README for details.
