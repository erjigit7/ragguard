"""Batch end-to-end product check: generate several fresh grounded/
hallucinated pairs (articles NOT in the 3000 used for training) with
KyrgyzLLM, then score them with verify_score.py. If the detector gives
high scores to grounded and low scores to hallucinated on genuinely
unseen articles, the whole pipeline (not just the offline test split)
actually works. For a single hands-on example with your own text, use
verify_generate.py + verify_score.py instead.

Run with KyrgyzLLM's venv (it has unsloth), from RagGuard's repo root:

    ../KyrgyzLLM/.venv/Scripts/python.exe -m kyrgyz.smoke_verify

Then score with RagGuard's own venv:

    python -m kyrgyz.verify_score --batch
"""

import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "KyrgyzLLM"))

from datasets import load_dataset
from unsloth import FastLanguageModel

from kyrgyz.generate_dataset import GROUNDED_PROMPT, corrupt_fact, corrupt_addition

MODEL_PATH = "C:/hf/kazllm"
ADAPTER_PATH = str(Path(__file__).resolve().parent.parent.parent / "KyrgyzLLM" / "outputs" / "kazllm-kyrgyz-sft-v2-final")
OUT_PATH = Path(__file__).resolve().parent / "_verify_batch.json"


def load_fresh_contexts(n=4, min_chars=300, max_chars=1500, seed=42, skip=3000):
    corpus = load_dataset("the-cramer-project/Kyrgyz_News_Corpus")["train"]
    candidates = [
        row["text"] for row in corpus if row["text"] and min_chars <= len(row["text"]) <= max_chars
    ]
    random.Random(seed).shuffle(candidates)
    return candidates[skip:skip + n]


def generate_grounded(model, tokenizer, text, max_new_tokens=150):
    prompt = GROUNDED_PROMPT.format(text=text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    contexts = load_fresh_contexts(n=4)
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL_PATH, max_seq_length=2048, load_in_4bit=True)
    model.load_adapter(ADAPTER_PATH)
    FastLanguageModel.for_inference(model)

    rng = random.Random(7)
    pairs = []
    grounded_answers = [generate_grounded(model, tokenizer, c) for c in contexts]
    for i, (context, grounded) in enumerate(zip(contexts, grounded_answers)):
        others = grounded_answers[:i] + grounded_answers[i + 1:]
        hallucinated = corrupt_fact(grounded, rng) or corrupt_addition(grounded, others, rng)
        pairs.append({"context": context, "grounded": grounded, "hallucinated": hallucinated})

    OUT_PATH.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(pairs)} fresh test pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
