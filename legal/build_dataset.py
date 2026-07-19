"""Юридический guard-датасет из корпуса Myizam (ru + ky).

Мотивация (замер myizam/docs/eval_report.md): ragguard-kyrgyz-v1 на юрдомене
ловит подмену ЦИФР идеально (Δ+0.99), но слеп к числам/дробям СЛОВАМИ
(«трех месяцев»→«девяти» — Δ~0) и даёт ложную тревогу на цифру в ответе при
числе-словом в контексте. Причина — синтетические негативы v1 портили только
цифры, и домен был новостной.

Контексты: статьи 8 кодексов КР из D:/claude_projects/myizam/data/chunks
(обе языковые версии). Honest-ответ = дословное предложение статьи с числом
(grounded по построению). Порчи (label=0), на пару — один тип:
  digit    — возмущение цифры правдоподобной дельтой (подход v1);
  word     — подмена дроби/числительного СЛОВАМИ по таблице (новое!);
  baseless — предложение из ДРУГОЙ статьи (coarse-тип v1).

Выход: data/legal_hallucination.jsonl (context, answer, label, kind, lang) —
формат совместим с kyrgyz/data.py (лишние поля загрузчик игнорирует).

Usage:  python -m legal.build_dataset
"""

import json
import pathlib
import random
import re

MYIZAM_CHUNKS = pathlib.Path("D:/claude_projects/myizam/data/chunks")
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "legal_hallucination.jsonl"

MAX_PAIRS_PER_LANG = 2500
MIN_CTX, MAX_CTX = 300, 3500
MIN_SENT, MAX_SENT = 40, 350

WORD_TABLES = {
    "ru": [
        ["одной четверти", "трех четвертей", "одной трети", "двух третей", "половины"],
        ["одна четверть", "три четверти", "одна треть", "две трети", "половина"],
        ["двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"],
        ["двух", "трех", "пяти", "десяти", "пятнадцати"],
        ["два", "три", "пять", "десять", "пятнадцать"],
    ],
    "kg": [
        ["төрттөн бири", "төрттөн үчү", "үчтөн бири", "үчтөн экиси", "жарымы"],
        ["жыйырма", "отуз", "кырк", "элүү", "алтымыш", "жетимиш", "сексен", "токсон"],
        ["эки", "үч", "беш", "он", "он беш"],
    ],
}

SENT_SPLIT = re.compile(r"(?<=[.;])\s+")
DIGIT = re.compile(r"(?<![\d-])(\d{1,4})(?![\d-])")


def perturb_digit(m: re.Match) -> str:
    n = int(m.group(1))
    rng = random.Random(n)
    if 1900 <= n <= 2100:                      # год: сдвиг 1-6 (подход v1)
        return str(n + rng.choice([-6, -3, -2, -1, 1, 2, 3, 6]))
    delta = max(1, round(n * rng.choice([0.3, 0.5, 1.0])))
    return str(n + delta if rng.random() < 0.5 else max(1, n - delta))


def corrupt_word(sentence: str, lang: str, rng: random.Random) -> str | None:
    for table in WORD_TABLES[lang]:
        for form in table:
            # для коротких форм — граница слова, чтобы «три» не матчился внутри «территории»
            pat = re.compile(rf"(?<![\w-]){re.escape(form)}(?![\w-])")
            if pat.search(sentence):
                other = rng.choice([f for f in table if f != form])
                return pat.sub(other, sentence, count=1)
    return None


def load_articles(lang: str) -> list[dict]:
    arts = []
    for f in sorted(MYIZAM_CHUNKS.glob(f"*_{lang}.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            if c["ArticleNumber"] and c["Part"] == 1 and MIN_CTX <= len(c["Text"]) <= MAX_CTX:
                arts.append(c)
    return arts


def main() -> None:
    rng = random.Random(42)
    rows = []
    for lang in ("ru", "kg"):
        arts = load_articles(lang)
        rng.shuffle(arts)
        made = 0
        for i, art in enumerate(arts):
            if made >= MAX_PAIRS_PER_LANG:
                break
            sentences = [s.strip() for s in SENT_SPLIT.split(art["Text"]) if MIN_SENT <= len(s.strip()) <= MAX_SENT]
            numbered = [s for s in sentences if DIGIT.search(s)]
            worded = [s for s in sentences if corrupt_word(s, lang, random.Random(0)) is not None]
            if not sentences:
                continue

            # Приоритет word-порчам — они закрывают слепую зону; baseless — добивка
            if worded:
                kind = "word"
            elif numbered and rng.random() < 0.6:
                kind = "digit"
            else:
                kind = "baseless"
            if kind == "word" and worded:
                honest = rng.choice(worded)
                corrupted = corrupt_word(honest, lang, rng)
            elif kind == "digit" and numbered:
                honest = rng.choice(numbered)
                idx = rng.randrange(len(DIGIT.findall(honest)))
                count = [0]
                def repl(m, idx=idx, count=count):
                    count[0] += 1
                    return perturb_digit(m) if count[0] - 1 == idx else m.group(1)
                corrupted = DIGIT.sub(repl, honest)
            else:
                honest = rng.choice(sentences)
                donor = arts[(i + 37) % len(arts)]
                donor_sents = [s.strip() for s in SENT_SPLIT.split(donor["Text"]) if MIN_SENT <= len(s.strip()) <= MAX_SENT]
                if not donor_sents:
                    continue
                corrupted = honest + " " + rng.choice(donor_sents)
                kind = "baseless"

            if corrupted == honest:
                continue
            ctx = art["Header"] + "\n" + art["Text"]
            rows.append({"context": ctx, "answer": honest, "label": 1, "kind": kind, "lang": lang})
            rows.append({"context": ctx, "answer": corrupted, "label": 0, "kind": kind, "lang": lang})
            made += 1
        print(f"{lang}: пар={made}")

    kinds = {}
    for r in rows:
        if r["label"] == 0:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print("негативы по типам:", kinds)

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"итого примеров: {len(rows)} → {OUT}")


if __name__ == "__main__":
    main()
