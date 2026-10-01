import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.model import SentimentModel


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        model_path = Path(cls.tmp.name) / "model.json"
        SentimentModel(weights={"good": 5.0, "bad": -5.0}).save(model_path)
        cls.client = TestClient(create_app(model_path=model_path))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_ready(self):
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ready"], True)

    def test_predict_positive(self):
        response = self.client.post("/predict", json={"text": "good"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["label"], "positive")
        self.assertGreater(body["confidence"], 0.9)

    def test_predict_rejects_empty_text(self):
        response = self.client.post("/predict", json={"text": ""})
        self.assertEqual(response.status_code, 422)

    def test_metrics_exposes_counters(self):
        client = TestClient(create_app(model_path=Path(self.tmp.name) / "model.json"))
        client.get("/health")
        body = client.get("/metrics").text
        self.assertIn("http_requests_total", body)
        self.assertIn('path="/health"', body)


class MissingModelTests(unittest.TestCase):
    def test_missing_model_returns_503(self):
        client = TestClient(create_app(model_path="does-not-exist.json"))
        self.assertEqual(client.get("/ready").status_code, 503)
        self.assertEqual(
            client.post("/predict", json={"text": "hi"}).status_code, 503
        )


if __name__ == "__main__":
    unittest.main()
