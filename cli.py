"""CLI for Naive Bayes text classification."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from naive_bayes.model import NaiveBayesClassifier


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and test Naive Bayes classifier.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--text", required=True, help="Text to classify")
    args = parser.parse_args()

    texts, labels = load_dataset(args.data)
    model = NaiveBayesClassifier()
    model.fit(texts, labels)
    prediction = model.predict(args.text)
    print(prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
