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
from .explain import GroundingError, WindowContext, generate_report
from .features import extract
from .llm import make_llm
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


class Evidence(BaseModel):
    feature: str
    value: float
    normal_median: float


class ExplainResponse(BaseModel):
    fault_type: str
    fault_detected: bool
    confidence: float
    summary: str
    evidence: list[Evidence]
    drift_note: str
    recommended_checks: list[str]
    model: str
    grounding_checked: bool


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
    app.state.llm = make_llm()
    if app.state.llm is None:
        log.warning("no ANTHROPIC_API_KEY or GROQ_API_KEY; /explain will return 503")
    else:
        log.info("operator reports via %s (%s)", app.state.llm.provider, app.state.llm.model)
    yield


app = FastAPI(title="grid-fault-service", version="0.4.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    loaded = request.app.state.model is not None
    if not loaded:
        response.status_code = 503
    return HealthResponse(
        status="ok" if loaded else "unavailable", model_loaded=loaded, classes=CLASSES
    )


def _score(clf, req: PredictRequest) -> tuple[np.ndarray, np.ndarray]:
    feats = extract(np.asarray(req.window, dtype=np.float32)[None])
    return feats, clf.predict_proba(feats)[0]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    clf = request.app.state.model
    if clf is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    feats, proba = _score(clf, req)
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


@app.post("/explain", response_model=ExplainResponse)
def explain(req: PredictRequest, request: Request) -> dict:
    """Plain-language fault report for an operator, grounded in this service's own outputs.

    The window is scored again here rather than trusting a prediction sent by the client,
    and it is not added to the drift buffer, so calling /predict and then /explain on the
    same window does not count it twice.
    """
    clf = request.app.state.model
    monitor = request.app.state.monitor
    llm = request.app.state.llm
    if clf is None or monitor is None:
        raise HTTPException(status_code=503, detail="model or drift reference not loaded")
    if llm is None:
        raise HTTPException(
            status_code=503, detail="no LLM configured: set ANTHROPIC_API_KEY or GROQ_API_KEY"
        )
    feats, proba = _score(clf, req)
    ctx = WindowContext(
        features=feats[0],
        proba=proba,
        class_indices=[int(c) for c in clf.classes_],
        monitor=monitor,
    )
    try:
        return generate_report(llm.client, ctx, model=llm.model)
    except GroundingError as e:
        log.warning("rejected ungrounded report: %s", e)
        raise HTTPException(
            status_code=502, detail=f"report failed the grounding check: {e}"
        ) from e
