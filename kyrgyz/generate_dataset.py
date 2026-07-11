"""Build a Kyrgyz (context, answer, label) hallucination-detection dataset.

Source of contexts: the-cramer-project/Kyrgyz_News_Corpus (real, natural
Kyrgyz news articles).

History: the first attempt used qwen2.5:7b-instruct via Ollama for BOTH
the grounded and hallucinated summary, but the user (native speaker)
judged qwen2.5's Kyrgyz as not good enough. That's why KyrgyzLLM (a
separate project) exists -- a LoRA continued-pretrain + light SFT of
ISSAI's KazLLM-8B, now writing genuinely fluent, accurate Kyrgyz
summaries (validated on held-out news articles).

That fine-tuned model is used here for the GROUNDED half (label=1) --
it's exactly the "summarize this text accurately" task it was trained
on. It is NOT used for the hallucinated half: a smoke test asking it to
"deliberately insert one wrong fact" produced byte-identical output to
the grounded prompt on all 3 test cases -- the SFT pass narrowed it
onto its one trained behavior so tightly that it ignores instruction
variations entirely. So the hallucinated half (label=0) is built by
programmatically corrupting one number in the already-generated
grounded summary instead -- deterministic, and doesn't depend on the
model doing a task it can't reliably do.

A first version also tried swapping in proper-noun-like capitalized
words (names/places) pulled from other articles' summaries, and swapped
numbers with other numbers from a global pool across all summaries.
Both produced bad examples on manual review: name-swaps often broke
grammatically (a capitalized word grabbed from an unrelated sentence
rarely has the right case/declension to slot in) or grabbed non-names
entirely (a crude "capitalized, not sentence-initial" regex also
matches declined forms of ordinary nouns); pool-based number swaps
sometimes crossed semantic types (a "2200th anniversary" year landing
in a day-of-month slot, producing an impossible date). Perturbing the
original number by a plausible same-magnitude delta avoids both
problems.

A second version used number-perturbation as the ONLY hallucination
type and trained a cross-encoder (xlm-roberta-base) on the 341 pairs it
produced (from 1200 articles -- 859 skipped for having no number to
perturb). Loss sat at ln(2) (0.693, the "always predict 0.5" floor) for
the entire run at multiple learning rates and up to 30 epochs. A sanity
check confirmed the training pipeline itself works fine (loss dropped
0.70 -> 0.09 in 3 epochs on an easy shuffled-answer task) -- the problem
was specifically that single-digit-swap negatives are nearly identical
strings to their positive counterpart, too subtle a signal to learn
from only ~290 training pairs. Two fixes follow from that: a second,
coarser hallucination type (corrupt_addition -- append an unrelated
claim borrowed from a different article's summary, RAGTruth's
"baseless_info" category rather than "evident_conflict"), used for
every article, chosen ~50/50 against the number-swap type when both are
available -- this also means no article is skipped anymore (the number
type needs a number; the addition type never does), roughly quadrupling
the dataset size for free.
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "KyrgyzLLM"))

from datasets import load_dataset
from unsloth import FastLanguageModel

MODEL_PATH = "C:/hf/kazllm"
ADAPTER_PATH = str(Path(__file__).resolve().parent.parent.parent / "KyrgyzLLM" / "outputs" / "kazllm-kyrgyz-sft-v2-final")
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "kyrgyz_hallucination.jsonl"

GROUNDED_PROMPT = (
    "Сен кыргыз тилинде гана жооп берүүчү жардамчысың. Казак тилинде эмес, так кыргыз тилинде жаз.\n\n"
    "Тапшырма: Төмөнкү тексттеги фактыларга гана таянып, кыргызча кыскача жыйынтыкта.\n"
    "Текст: {text}\n"
    "Жыйынтык:"
)

NUMBER_RE = re.compile(r"\d+")
MONTH_STEMS = ("январ", "феврал", "март", "апрел", "май", "июн", "июл",
               "август", "сентябр", "октябр", "ноябр", "декабр")
# A number directly preceded by "-" with a short run of letters right
# before that (no space) is almost always part of a fixed name/code --
# COVID-19, АН-2, С-130 -- not a fact that has a "plausible alternative".
FIXED_CODE_PREFIX_RE = re.compile(r"[A-Za-zА-Яа-яӨөҮүҢң]{1,8}-$")


def load_contexts(n, min_chars=300, max_chars=1500, seed=42):
    corpus = load_dataset("the-cramer-project/Kyrgyz_News_Corpus")["train"]
    candidates = [
        row["text"] for row in corpus if row["text"] and min_chars <= len(row["text"]) <= max_chars
    ]
    random.Random(seed).shuffle(candidates)
    return candidates[:n]


def generate_grounded(model, tokenizer, text, max_new_tokens=150):
    prompt = GROUNDED_PROMPT.format(text=text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def perturb_number(n: int, rng: random.Random) -> int:
    """A different value in the same rough order of magnitude as n --
    e.g. 29 -> 8..17 or 41..70, not something wildly different in scale.

    Calendar years need special-casing: a 10-60% shift turns 2019 into
    something like 1087, which is absurd rather than subtly wrong (a
    percentage delta only makes sense for counts/quantities, not a
    number whose plausible range is a handful of years wide). Detected
    by "4 digits, looks like a real year" and perturbed by a small
    absolute delta instead.
    """
    if n == 0:
        return rng.choice([1, 2, 3, 5, 7])
    if 1900 <= n <= 2035 and len(str(n)) == 4:
        delta = rng.randint(1, 6)
        return n - delta if (rng.random() < 0.5 and n - delta >= 1900) else n + delta
    min_delta = max(1, round(n * 0.1))
    max_delta = max(min_delta + 1, round(n * 0.6))
    delta = rng.randint(min_delta, max_delta)
    if rng.random() < 0.5 and n - delta > 0:
        return n - delta
    return n + delta


def is_fixed_code(answer: str, match: re.Match) -> bool:
    """True if this number looks like part of a fixed name/code (COVID-19,
    АН-2, С-130) rather than a standalone fact -- corrupting those doesn't
    produce a subtly-wrong fact, just an obviously broken name."""
    return bool(FIXED_CODE_PREFIX_RE.search(answer[: match.start()]))


def is_day_of_month(answer: str, match: re.Match) -> bool:
    """True if this number is immediately followed by a Kyrgyz month stem
    (e.g. "31-мартта") -- such numbers must stay within 1-31, which a
    magnitude-percentage delta doesn't respect (31 -> 47 is not a real date)."""
    tail = answer[match.end():match.end() + 12].lstrip("-").lower()
    return tail.startswith(MONTH_STEMS)


def perturb_day_of_month(n: int, rng: random.Random) -> int:
    delta = rng.randint(1, 8)
    candidate = n - delta if (rng.random() < 0.5 and n - delta >= 1) else n + delta
    return max(1, min(31, candidate)) if candidate != n else n + 1


def corrupt_fact(answer: str, rng: random.Random) -> str | None:
    """Swap exactly one number in the answer for a different, plausible
    number. Returns None if the summary has no corruptible number."""
    candidates = [m for m in NUMBER_RE.finditer(answer) if not is_fixed_code(answer, m)]
    if not candidates:
        return None
    match = rng.choice(candidates)
    original = match.group()
    n = int(original)
    new_n = perturb_day_of_month(n, rng) if is_day_of_month(answer, match) else perturb_number(n, rng)
    return answer[: match.start()] + str(new_n) + answer[match.end():]


def corrupt_addition(answer: str, other_answers: list[str], rng: random.Random) -> str:
    """Append one sentence borrowed from a different article's grounded
    summary, as if it were an additional fact -- an unsupported claim this
    context says nothing about (RAGTruth's "baseless_info" hallucination
    type, as opposed to "evident_conflict" for corrupt_fact). Always
    succeeds given a non-empty pool, so it works as a fallback for
    summaries with no number to perturb."""
    donor = rng.choice(other_answers)
    first_sentence = donor.split(".")[0].strip().strip('"')
    if not first_sentence:
        first_sentence = donor.strip()
    connector = rng.choice(["Ошондой эле", "Мындан тышкары", "Дагы"])
    addition = first_sentence[0].lower() + first_sentence[1:] if first_sentence else first_sentence
    return f"{answer.rstrip('.')}. {connector} {addition}."


def make_hallucinated(answer: str, other_answers: list[str], rng: random.Random) -> str:
    """~50/50 between the two corruption types when both are possible, so
    corruption type doesn't correlate with article topic (e.g. weather
    reports always having numbers would otherwise concentrate one type in
    one topic, letting a classifier learn a topic shortcut instead of an
    actual grounded-vs-hallucinated distinction)."""
    if rng.random() < 0.5:
        result = corrupt_fact(answer, rng)
        if result is not None:
            return result
    return corrupt_addition(answer, other_answers, rng)


def main(n_articles=500, seed=42):
    rng = random.Random(seed)
    contexts = load_contexts(n_articles, seed=seed)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH, max_seq_length=2048, load_in_4bit=True
    )
    model.load_adapter(ADAPTER_PATH)
    FastLanguageModel.for_inference(model)

    print(f"Generating grounded summaries for {len(contexts)} articles...")
    grounded_answers = []
    for i, context in enumerate(contexts, start=1):
        answer = generate_grounded(model, tokenizer, context)
        grounded_answers.append(answer)
        if i % 25 == 0:
            print(f"  [{i}/{len(contexts)}]")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, (context, grounded) in enumerate(zip(contexts, grounded_answers)):
            other_answers = grounded_answers[:i] + grounded_answers[i + 1:]
            hallucinated = make_hallucinated(grounded, other_answers, rng)
            f.write(json.dumps({"context": context, "answer": grounded, "label": 1}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"context": context, "answer": hallucinated, "label": 0}, ensure_ascii=False) + "\n")

    print(f"Saved {len(contexts) * 2} examples ({len(contexts)} pairs) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
