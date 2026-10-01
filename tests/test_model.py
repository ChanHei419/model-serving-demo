import tempfile
import unittest
from pathlib import Path

from app.model import SentimentModel, sigmoid, tokenize


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_splits(self):
        self.assertEqual(tokenize("Good, isn't it?"), ["good", "isn't", "it"])


class SigmoidTests(unittest.TestCase):
    def test_zero_is_half(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_extremes(self):
        self.assertGreater(sigmoid(50), 0.999)
        self.assertLess(sigmoid(-50), 0.001)


class SentimentModelTests(unittest.TestCase):
    def test_predicts_with_known_weights(self):
        model = SentimentModel(weights={"good": 4.0, "bad": -4.0}, bias=0.0)

        label, confidence = model.predict("good")
        self.assertEqual(label, "positive")
        self.assertGreater(confidence, 0.9)

        label, _ = model.predict("bad")
        self.assertEqual(label, "negative")

    def test_neutral_text_scores_half(self):
        model = SentimentModel(weights={"good": 4.0}, bias=0.0)
        self.assertAlmostEqual(model.score("unknown words"), 0.5)

    def test_save_and_load_roundtrip(self):
        model = SentimentModel(weights={"win": 1.5}, bias=0.25)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            loaded = SentimentModel.load(path)
            self.assertEqual(loaded.weights, model.weights)
            self.assertAlmostEqual(loaded.bias, model.bias)


if __name__ == "__main__":
    unittest.main()
