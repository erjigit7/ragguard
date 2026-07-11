"""Step 2 of 2: score the pair verify_generate.py produced with the
trained RagGuard-Kyrgyz detector.

Run this with RagGuard's own venv, from the RagGuard repo root:

    python -m kyrgyz.verify_score
"""

import json
import sys
from pathlib import Path

from kyrgyz.model import load_model, check_groundedness

sys.stdout.reconfigure(encoding="utf-8")

PAIR_PATH = Path(__file__).resolve().parent / "_verify_pair.json"


def main():
    if not PAIR_PATH.exists():
        print("Не нашёл _verify_pair.json — сначала запустите verify_generate.py (из venv KyrgyzLLM).")
        return

    pair = json.loads(PAIR_PATH.read_text(encoding="utf-8"))
    model = load_model()

    g = check_groundedness(model, pair["context"], pair["grounded"])
    h = check_groundedness(model, pair["context"], pair["hallucinated"])

    print("=== ПРАВДИВЫЙ ПЕРЕСКАЗ ===")
    print(pair["grounded"])
    print(f"-> детектор: grounded={g['grounded']}, score={g['score']}")

    print("\n=== ИСПОРЧЕННЫЙ ПЕРЕСКАЗ ===")
    print(pair["hallucinated"])
    print(f"-> детектор: grounded={h['grounded']}, score={h['score']}")

    print("\n(score ближе к 1 = детектор уверен, что это правда; ближе к 0 = уверен, что выдумка)")


if __name__ == "__main__":
    main()
