"""Vertex-compatible prediction service with the same preprocessing as training."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .features import validate_instances
from .train import predict_bundle

BUNDLE = None


@asynccontextmanager
async def lifespan(app):
    global BUNDLE
    model_dir = os.environ.get("AIP_STORAGE_URI") or os.environ.get("MODEL_DIR", "artifacts")
    if model_dir.startswith("gs://"):
        from google.cloud import storage
        bucket, key = model_dir[5:].split("/", 1)
        path = Path("/tmp/readmission-model.joblib")
        storage.Client().bucket(bucket).blob(f"{key.rstrip('/')}/model.joblib").download_to_filename(str(path))
    else:
        path = Path(model_dir) / "model.joblib"
    BUNDLE = joblib.load(path)
    yield


app = FastAPI(title="Discharge follow-up research prototype", lifespan=lifespan)


class PredictionRequest(BaseModel):
    instances: list[dict]


@app.get(os.environ.get("AIP_HEALTH_ROUTE", "/health"))
def health():
    if BUNDLE is None:
        raise HTTPException(503, "Model not loaded.")
    return {"status": "ready"}


@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/predict"))
def predict(request: PredictionRequest):
    if BUNDLE is None:
        raise HTTPException(503, "Model not loaded.")
    try:
        frame = validate_instances(request.instances)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    scores = predict_bundle(BUNDLE, frame)
    return {"predictions": [{"readmission_risk": float(score), "model": BUNDLE["model_name"]} for score in scores]}
