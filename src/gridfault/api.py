"""FastAPI service: classify a 3-phase relay window.

Run:  uvicorn gridfault.api:app   (with PYTHONPATH=src when running from the repo)
Env:  GRIDFAULT_MODEL_PATH and GRIDFAULT_REFERENCE_PATH override where the trained model
      and the drift reference statistics are loaded from.
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

from .drift import REFERENCE_PATH, DriftMonitor, load_reference
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


class DriftedFeature(BaseModel):
    feature: str
    psi: float
    ks_p_value: float


class ClassDrift(BaseModel):
    n: int
    evaluated: bool
    drifted: bool | None = None
    n_drifted_features: int | None = None
    max_psi: float | None = None
    top_drifted_features: list[DriftedFeature] = []


class DriftReport(BaseModel):
    status: str
    drift_detected: bool
    min_samples_per_class: int
    n_buffered: int
    classes: dict[str, ClassDrift]


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("GRIDFAULT_MODEL_PATH", MODEL_PATH))
    ref_path = Path(os.environ.get("GRIDFAULT_REFERENCE_PATH", REFERENCE_PATH))
    app.state.model = None
    app.state.monitor = None
    if path.exists():
        app.state.model = load_model(path)
        log.info("loaded model from %s", path)
    else:
        log.warning("no model at %s; /predict will return 503", path)
    if ref_path.exists():
        app.state.monitor = DriftMonitor(load_reference(ref_path))
        log.info("loaded drift reference from %s", ref_path)
    else:
        log.warning("no drift reference at %s; /drift will return 503", ref_path)
    yield


app = FastAPI(title="grid-fault-service", version="0.3.0", lifespan=lifespan)


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
    feats = extract(x)
    proba = clf.predict_proba(feats)[0]
    names = [CLASSES[int(c)] for c in clf.classes_]
    best = int(np.argmax(proba))
    monitor = request.app.state.monitor
    if monitor is not None:
        monitor.add(int(clf.classes_[best]), feats[0])
    return PredictResponse(
        label=names[best],
        fault_detected=names[best] != "NORMAL",
        confidence=round(float(proba[best]), 4),
        probabilities={n: round(float(p), 4) for n, p in zip(names, proba, strict=True)},
    )


def _monitor(request: Request) -> DriftMonitor:
    monitor = request.app.state.monitor
    if monitor is None:
        raise HTTPException(status_code=503, detail="drift reference not loaded")
    return monitor


@app.get("/drift", response_model=DriftReport)
def drift(request: Request) -> dict:
    """Compare recent inputs with the training distribution, per predicted class."""
    return _monitor(request).report()


@app.post("/drift/reset")
def drift_reset(request: Request) -> dict[str, str]:
    """Clear the buffered windows (e.g. after a recalibration or between demo phases)."""
    _monitor(request).reset()
    return {"status": "reset"}
