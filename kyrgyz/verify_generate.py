"""Step 1 of 2 for a hands-on product check: generate a grounded summary
and a hallucinated one from a piece of Kyrgyz text, using the fine-tuned
KyrgyzLLM model.

Run this with KyrgyzLLM's venv (it has unsloth):

    cd RagGuard
    ../KyrgyzLLM/.venv/Scripts/python.exe -m kyrgyz.verify_generate

Paste your own Kyrgyz text when prompted, or press Enter on an empty
line to use a random fresh article from the corpus instead (skips the
first 3000 articles, i.e. never one used to train the detector).

Alternatively, pass --text-file path.txt (UTF-8) to skip the prompt --
more reliable than piping text into stdin, which can pick up the
wrong console codepage on Windows depending on how it's invoked.

Writes kyrgyz/_verify_pair.json, which verify_score.py (run from
RagGuard's own venv) reads next.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "KyrgyzLLM"))

from datasets import load_dataset
from unsloth import FastLanguageModel

from kyrgyz.generate_dataset import GROUNDED_PROMPT, corrupt_fact, corrupt_addition

MODEL_PATH = "C:/hf/kazllm"
ADAPTER_PATH = str(Path(__file__).resolve().parent.parent.parent / "KyrgyzLLM" / "outputs" / "kazllm-kyrgyz-sft-v2-final")
OUT_PATH = Path(__file__).resolve().parent / "_verify_pair.json"


def random_fresh_context(seed=42, skip=3000):
    corpus = load_dataset("the-cramer-project/Kyrgyz_News_Corpus")["train"]
    candidates = [row["text"] for row in corpus if row["text"] and 300 <= len(row["text"]) <= 1500]
    random.Random(seed).shuffle(candidates)
    return random.choice(candidates[skip:])


def generate_grounded(model, tokenizer, text, max_new_tokens=150):
    prompt = GROUNDED_PROMPT.format(text=text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", default=None, help="Path to a UTF-8 .txt file with the Kyrgyz text")
    args = parser.parse_args()

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
    else:
        print("Вставьте кыргызский текст (одной строкой) и нажмите Enter.")
        print("Или просто нажмите Enter — возьму случайную свежую статью из корпуса.\n")
        text = input("> ").strip()
    if not text:
        text = random_fresh_context()
        print("\n(взял случайную статью из корпуса)")

    print("\nЗагружаю модель...")
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL_PATH, max_seq_length=2048, load_in_4bit=True)
    model.load_adapter(ADAPTER_PATH)
    FastLanguageModel.for_inference(model)

    grounded = generate_grounded(model, tokenizer, text)
    rng = random.Random()
    # Generic filler claims, unrelated to any specific article, used as the
    # "borrowed sentence" donor pool for the addition-type corruption when
    # there's no batch of other real summaries to draw from (single-context
    # interactive run) -- must NOT include `grounded` itself (degenerates
    # into repeating the same sentence twice), and must NOT start with a
    # connector word themselves (corrupt_addition already prepends one --
    # "Ошондой эле ошондой эле..." otherwise).
    donors = [
        "Бул тууралуу расмий түрдө эч ким жооп берген жок.",
        "Окуя болгон жерге кошумча күчтөр жиберилген.",
        "Бул чечим кийинки жумада күчүнө кире тургандыгы белгиленди.",
    ]
    hallucinated = corrupt_fact(grounded, rng) or corrupt_addition(grounded, donors, rng)

    pair = {"context": text, "grounded": grounded, "hallucinated": hallucinated}
    OUT_PATH.write_text(json.dumps(pair, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== ТЕКСТ ===")
    print(text)
    print("\n=== ПРАВДИВЫЙ ПЕРЕСКАЗ (модель написала сама) ===")
    print(grounded)
    print("\n=== ИСПОРЧЕННЫЙ ПЕРЕСКАЗ (код подменил один факт) ===")
    print(hallucinated)
    print(f"\nСохранено в {OUT_PATH}")
    print("Теперь запустите: (из venv RagGuard) python -m kyrgyz.verify_score")


if __name__ == "__main__":
    main()
