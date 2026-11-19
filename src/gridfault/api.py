"""FastAPI service: classify a 3-phase relay window.

Run:  uvicorn gridfault.api:app   (with PYTHONPATH=src when running from the repo)
Env:  GRIDFAULT_MODEL_PATH overrides where the trained model is loaded from.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from .features import extract
from .model import MODEL_PATH, load_model
from .simulate import CHANNELS, CLASSES, WINDOW

log = logging.getLogger("gridfault.api")

MAX_ABS = 100.0  # per-unit; far above anything physical, rejects unit mix-ups
Sample = Annotated[float, Field(allow_inf_nan=False)]


class PredictRequest(BaseModel):
    window: list[list[Sample]] = Field(
        description=(
            f"{len(CHANNELS)} channels x {WINDOW} samples at 4 kHz, in per-unit. "
            f"Channel order: {', '.join(CHANNELS)}."
        )
    )

    @field_validator("window")
    @classmethod
    def check_shape_and_range(cls, w: list[list[float]]) -> list[list[float]]:
        if len(w) != len(CHANNELS):
            raise ValueError(f"expected {len(CHANNELS)} channels, got {len(w)}")
        for name, row in zip(CHANNELS, w, strict=True):
            if len(row) != WINDOW:
                raise ValueError(f"channel {name} has {len(row)} samples, expected {WINDOW}")
            if max(abs(v) for v in row) > MAX_ABS:
                raise ValueError(f"channel {name} has values beyond +/-{MAX_ABS} pu")
        return w


class PredictResponse(BaseModel):
    label: str
    fault_detected: bool
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("GRIDFAULT_MODEL_PATH", MODEL_PATH))
    app.state.model = None
    if path.exists():
        app.state.model = load_model(path)
        log.info("loaded model from %s", path)
    else:
        log.warning("no model at %s; /predict will return 503", path)
    yield


app = FastAPI(title="grid-fault-service", version="0.2.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    loaded = request.app.state.model is not None
    if not loaded:
        response.status_code = 503
    return HealthResponse(
        status="ok" if loaded else "unavailable", model_loaded=loaded, classes=CLASSES
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    clf = request.app.state.model
    if clf is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    x = np.asarray(req.window, dtype=np.float32)[None]
    proba = clf.predict_proba(extract(x))[0]
    names = [CLASSES[int(c)] for c in clf.classes_]
    best = int(np.argmax(proba))
    return PredictResponse(
        label=names[best],
        fault_detected=names[best] != "NORMAL",
        confidence=round(float(proba[best]), 4),
        probabilities={n: round(float(p), 4) for n, p in zip(names, proba, strict=True)},
    )
