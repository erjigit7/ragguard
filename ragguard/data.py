from datasets import Dataset, load_dataset

DATASET_NAME = "wandb/RAGTruth-processed"


def _to_pairs(split) -> Dataset:
    """Turn a RAGTruth split into (context, answer, label) pairs.
    label = 1 means the answer is grounded in the context, 0 means it
    contains a hallucination (an evident conflict with the context, or a
    baseless claim not supported by it)."""
    hallucination_flags = split["hallucination_labels_processed"]
    labels = [
        0 if (flags["evident_conflict"] > 0 or flags["baseless_info"] > 0) else 1
        for flags in hallucination_flags
    ]
    return Dataset.from_dict({"context": split["context"], "answer": split["output"], "label": labels})


def load_ragtruth():
    """Return (train_dataset, test_dataset) of (context, answer, label) pairs."""
    raw = load_dataset(DATASET_NAME)
    return _to_pairs(raw["train"]), _to_pairs(raw["test"])
