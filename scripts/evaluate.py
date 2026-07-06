from sklearn.metrics import accuracy_score, classification_report

from ragguard.data import load_ragtruth
from ragguard.model import load_model


def main():
    _, test_dataset = load_ragtruth()
    model = load_model()

    pairs = list(zip(test_dataset["context"], test_dataset["answer"]))
    scores = model.predict(pairs)
    predictions = [1 if score >= 0.5 else 0 for score in scores]
    labels = test_dataset["label"]

    print(f"Test set size: {len(labels)}")
    print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
    print(classification_report(labels, predictions, target_names=["hallucinated", "grounded"]))


if __name__ == "__main__":
    main()
