"""Tiny logistic-regression sentiment model implemented in pure Python."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def sigmoid(value: float) -> float:
    """Numerically stable sigmoid."""
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


class SentimentModel:
    """Bag-of-words logistic regression with externally stored weights."""

    def __init__(self, weights: dict[str, float], bias: float = 0.0) -> None:
        self.weights = weights
        self.bias = bias

    @classmethod
    def load(cls, path: str | Path) -> "SentimentModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(weights=payload["weights"], bias=payload.get("bias", 0.0))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {"weights": self.weights, "bias": self.bias}, indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )

    def score(self, text: str) -> float:
        """Probability that the text is positive."""
        value = self.bias
        for token in tokenize(text):
            value += self.weights.get(token, 0.0)
        return sigmoid(value)

    def predict(self, text: str) -> tuple[str, float]:
        """Return ``(label, confidence)``."""
        probability = self.score(text)
        label = "positive" if probability >= 0.5 else "negative"
        confidence = probability if label == "positive" else 1 - probability
        return label, confidence
