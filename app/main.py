"""FastAPI application serving the sentiment model."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse

from app.metrics import Metrics
from app.model import SentimentModel
from app.schemas import PredictRequest, PredictResponse

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "models/sentiment.json")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='{"level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    )


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Application factory (use with ``uvicorn --factory``)."""
    configure_logging()

    path = Path(model_path or DEFAULT_MODEL_PATH)
    app = FastAPI(title="Model Serving Demo", version=MODEL_VERSION)
    metrics = Metrics()

    model: SentimentModel | None = None
    if path.exists():
        model = SentimentModel.load(path)
    else:
        logging.getLogger(__name__).warning(
            "model artifact not found at %s — /predict returns 503", path
        )

    @app.middleware("http")
    async def observe_requests(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        metrics.observe(
            request.method,
            request.url.path,
            response.status_code,
            time.perf_counter() - started,
        )
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": MODEL_VERSION}

    @app.get("/ready")
    def ready(response: Response) -> dict:
        if model is None:
            response.status_code = 503
            return {"ready": False, "reason": "model not loaded"}
        return {"ready": True, "version": MODEL_VERSION}

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, response: Response) -> PredictResponse:
        if model is None:
            response.status_code = 503
            return PredictResponse(
                label="unavailable", confidence=0.0, model_version=MODEL_VERSION
            )
        label, confidence = model.predict(payload.text)
        return PredictResponse(
            label=label,
            confidence=round(confidence, 4),
            model_version=MODEL_VERSION,
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint() -> str:
        return metrics.render()

    return app
