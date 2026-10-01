"""Train the tiny sentiment model from a CSV file (columns: text,label).

Labels may be integers (1/0) or strings (positive/negative).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model import SentimentModel, sigmoid, tokenize


def parse_label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"1", "positive", "pos", "true"}:
        return 1
    if normalized in {"0", "negative", "neg", "false"}:
        return 0
    raise ValueError(f"unknown label: {value!r}")


def load_rows(path: str | Path) -> list[tuple[str, int]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [(row["text"], parse_label(row["label"])) for row in reader]


def train(
    rows: list[tuple[str, int]],
    epochs: int = 300,
    learning_rate: float = 0.5,
) -> SentimentModel:
    """Train with online logistic-regression updates."""
    weights: dict[str, float] = {}
    bias = 0.0

    for _ in range(epochs):
        for text, label in rows:
            tokens = tokenize(text)
            value = bias + sum(weights.get(token, 0.0) for token in tokens)
            error = label - sigmoid(value)
            bias += learning_rate * error
            for token in tokens:
                weights[token] = weights.get(token, 0.0) + learning_rate * error

    return SentimentModel(weights=weights, bias=bias)


def evaluate(rows: list[tuple[str, int]], model: SentimentModel) -> float:
    correct = sum(
        1 for text, label in rows if (model.score(text) >= 0.5) == bool(label)
    )
    return correct / len(rows) if rows else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sentiment.csv")
    parser.add_argument("--output", default="models/sentiment.json")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    args = parser.parse_args(argv)

    rows = load_rows(args.data)
    model = train(rows, epochs=args.epochs, learning_rate=args.learning_rate)
    model.save(args.output)

    accuracy = evaluate(rows, model)
    print(f"Trained on {len(rows)} examples — training accuracy: {accuracy:.1%}")
    print(f"Saved model to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
