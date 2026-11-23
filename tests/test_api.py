import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gridfault.api import app
from gridfault.drift import save_reference
from gridfault.features import extract
from gridfault.model import fit_classifier, save_model
from gridfault.simulate import CLASSES, WINDOW, make_dataset, simulate_window


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("artifacts")
    x, y = make_dataset(150, seed=1)
    feats = extract(x)
    clf = fit_classifier(feats, y, max_iter=60)
    return {
        "model": save_model(clf, d / "model.joblib"),
        "reference": save_reference(d / "reference.npz", feats, y),
    }


@pytest.fixture()
def client(artifacts, monkeypatch):
    monkeypatch.setenv("GRIDFAULT_MODEL_PATH", str(artifacts["model"]))
    monkeypatch.setenv("GRIDFAULT_REFERENCE_PATH", str(artifacts["reference"]))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client_no_reference(artifacts, tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDFAULT_MODEL_PATH", str(artifacts["model"]))
    monkeypatch.setenv("GRIDFAULT_REFERENCE_PATH", str(tmp_path / "missing.npz"))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client_no_model(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDFAULT_MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("GRIDFAULT_REFERENCE_PATH", str(tmp_path / "missing.npz"))
    with TestClient(app) as c:
        yield c


def _body(label: str, seed: int = 0) -> dict:
    w = simulate_window(label, np.random.default_rng(seed))
    return {"window": w.tolist()}


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "classes": CLASSES}


def test_predict_returns_valid_distribution(client):
    r = client.post("/predict", json=_body("AG"))
    assert r.status_code == 200
    out = r.json()
    assert out["label"] in CLASSES
    assert set(out["probabilities"]) == set(CLASSES)
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-2
    assert out["confidence"] == max(out["probabilities"].values())
    assert out["fault_detected"] == (out["label"] != "NORMAL")


@pytest.mark.parametrize("label", ["NORMAL", "AG", "BC", "ABC"])
def test_predict_classifies_clean_examples(client, label):
    r = client.post("/predict", json=_body(label, seed=11))
    assert r.json()["label"] == label


def test_wrong_channel_count_is_422(client):
    body = _body("AG")
    body["window"] = body["window"][:5]
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    assert "expected 6 channels" in json.dumps(r.json())


def test_wrong_sample_count_is_422(client):
    body = _body("AG")
    body["window"][2] = body["window"][2][:-1]
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    assert f"expected {WINDOW}" in json.dumps(r.json())


def test_out_of_range_values_are_422(client):
    body = _body("AG")
    body["window"][0][10] = 1e6
    assert client.post("/predict", json=body).status_code == 422


def test_nan_is_rejected(client):
    body = json.dumps(_body("AG")).replace("0.", "NaN,0.", 1)  # inject a NaN token
    r = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_missing_field_is_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_non_numeric_is_422(client):
    body = _body("AG")
    body["window"][0][0] = "abc"
    assert client.post("/predict", json=body).status_code == 422


def test_no_model_gives_503(client_no_model):
    assert client_no_model.get("/health").status_code == 503
    assert client_no_model.get("/health").json()["model_loaded"] is False
    assert client_no_model.post("/predict", json=_body("AG")).status_code == 503


def test_drift_starts_empty(client):
    r = client.get("/drift")
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "insufficient_data"
    assert out["drift_detected"] is False
    assert out["n_buffered"] == 0


def test_predict_feeds_the_drift_buffer(client):
    for seed in range(3):
        assert client.post("/predict", json=_body("AG", seed)).status_code == 200
    out = client.get("/drift").json()
    assert out["n_buffered"] == 3
    assert out["classes"]["AG"]["n"] == 3
    assert out["classes"]["AG"]["evaluated"] is False  # far below the minimum sample size


def test_drift_reset_clears_buffer(client):
    client.post("/predict", json=_body("AG"))
    assert client.post("/drift/reset").json() == {"status": "reset"}
    assert client.get("/drift").json()["n_buffered"] == 0


def test_rejected_requests_do_not_enter_the_drift_buffer(client):
    body = _body("AG")
    body["window"] = body["window"][:5]
    assert client.post("/predict", json=body).status_code == 422
    assert client.get("/drift").json()["n_buffered"] == 0


def test_missing_reference_disables_drift_but_not_predict(client_no_reference):
    assert client_no_reference.post("/predict", json=_body("AG")).status_code == 200
    assert client_no_reference.get("/drift").status_code == 503
    assert client_no_reference.post("/drift/reset").status_code == 503
