"""Train the Kyrgyz RagGuard cross-encoder.

Mirrors scripts/train.py (the English version, trained on 15k RAGTruth
pairs), but on kyrgyz_hallucination.jsonl (3000 pairs / 6000 examples,
half number-perturbation negatives, half unsupported-addition negatives).

The first dataset iteration (341 pairs, number-perturbation only) never
learned anything -- loss flat at ln(2) across 8-30 epochs and two
learning rates, while a shuffled-answer sanity task trained fine, i.e.
the negatives were too subtle for that little data, not a pipeline bug.
This config assumes the current, bigger and easier dataset.
"""

from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

from kyrgyz.data import load_kyrgyz_hallucination
from kyrgyz.model import DEFAULT_MODEL_PATH, new_model


def main():
    train_dataset, test_dataset = load_kyrgyz_hallucination()
    print(f"Train: {len(train_dataset)} examples | Test: {len(test_dataset)} examples")

    model = new_model()
    loss = BinaryCrossEntropyLoss(model)

    args = CrossEncoderTrainingArguments(
        output_dir="models/ragguard-kyrgyz-v1-checkpoints",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=25,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss=loss,
    )
    trainer.train()
    model.save_pretrained(DEFAULT_MODEL_PATH)
    print(f"Saved trained model to {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
