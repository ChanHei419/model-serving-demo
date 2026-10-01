# Model Serving Demo

![Tests](https://github.com/ChanHei419/model-serving-demo/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-unittest-blue)

A production-style pattern for serving a machine-learning model as an API — the **MLOps layer** around a model:

- **Model artifact as data** — weights live in `models/sentiment.json`, decoupled from code
- **Training script** — logistic regression implemented from scratch (no ML frameworks), producing the artifact
- **FastAPI service** — `/health`, `/ready`, `/predict`, `/metrics` endpoints
- **Observability** — request counters and latency in Prometheus text format, JSON logs
- **Dockerized** — multi-stage image, non-root user, healthcheck
- **Tested** — model, API, and metrics covered by stdlib `unittest`

> Small on purpose: everything here is dependency-light so the *serving patterns* are the point, not the model.

---

## Architecture

```mermaid
flowchart LR
  A[data/sentiment.csv] -->|scripts/train_sentiment.py| B[models/sentiment.json]
  B --> C[FastAPI app]
  C --> D[/predict]
  C --> E[/health · /ready]
  C --> F[/metrics]
  D --> G[Client]
```

## Quick start

```bash
pip install -r requirements.txt

# 1. (Re)train the model artifact
python scripts/train_sentiment.py --data data/sentiment.csv --output models/sentiment.json

# 2. Run the API
uvicorn "app.main:create_app" --factory --reload

# 3. Call it
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "this project is great and useful"}'
```

```json
{"label": "positive", "confidence": 0.9918, "model_version": "1.0.0"}
```

### Docker

```bash
docker build -t model-serving-demo .
docker run -p 8000:8000 model-serving-demo
```

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness — process is up |
| `GET /ready` | Readiness — model artifact is loaded (503 otherwise) |
| `POST /predict` | `{"text": "..."}` → `{label, confidence, model_version}` |
| `GET /metrics` | Prometheus-format counters and request latency |

## Project structure

```
.
├── app/
│   ├── main.py          # FastAPI app factory + middleware
│   ├── model.py         # logistic-regression model (pure Python)
│   ├── schemas.py       # pydantic request/response models
│   └── metrics.py       # in-process Prometheus-format metrics
├── scripts/
│   └── train_sentiment.py
├── data/sentiment.csv   # small labelled dataset
├── models/sentiment.json
├── tests/               # unit + API tests
└── Dockerfile
```

## Configuration

| Env var | Default | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | `models/sentiment.json` | Model artifact location |
| `MODEL_VERSION` | `1.0.0` | Version reported by the API |

## Tests

```bash
python -m unittest discover -s tests -v
```

## Author

**HeiChan (Chan Hei Lun)** — BEng in Information Engineering, CUHK
[github.com/ChanHei419](https://github.com/ChanHei419)
