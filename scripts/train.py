from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

from ragguard.data import load_ragtruth
from ragguard.model import DEFAULT_MODEL_PATH, new_model


def main():
    train_dataset, test_dataset = load_ragtruth()
    model = new_model()
    loss = BinaryCrossEntropyLoss(model)

    args = CrossEncoderTrainingArguments(
        output_dir="models/ragguard-v1-checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        logging_steps=50,
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
