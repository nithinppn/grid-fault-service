"""Tests for the operator report. No network: a scripted fake stands in for the Anthropic
client. It calls the tools like the real model would, builds its report from the actual tool
results in the conversation, and can be told to fabricate numbers."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gridfault.api import app
from gridfault.drift import DriftMonitor, build_reference, save_reference
from gridfault.explain import (
    TOOLS,
    WindowContext,
    check_grounding,
    describe_feature,
    generate_report,
)
from gridfault.features import FEATURE_NAMES, extract
from gridfault.llm import LLM
from gridfault.model import fit_classifier, save_model
from gridfault.simulate import make_dataset, simulate_window


def _block(**kw):
    return SimpleNamespace(**kw)


class FakeLLM:
    """Mimics client.messages.create for a tool-using model.

    `tamper(report, attempt)` may modify the report before it is submitted; `attempt`
    counts submissions so a test can fabricate first and correct on the retry.
    """

    def __init__(self, tamper=None, skip_tools=False):
        self.tamper = tamper or (lambda r, attempt: r)
        self.skip_tools = skip_tools
        self.calls = []
        self.submissions = 0
        self.messages = self  # so client.messages.create works

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["tool_choice"] == {"type": "auto"}  # forced choice 400s on 5.5 models
        results = {}
        for msg in kwargs["messages"]:
            if msg["role"] == "assistant":
                for b in msg["content"]:
                    if b["type"] == "tool_use":
                        results[b["id"]] = b["name"]
            elif isinstance(msg["content"], list):
                for b in msg["content"]:
                    if b["type"] == "tool_result" and not b.get("is_error"):
                        results[results[b["tool_use_id"]]] = json.loads(b["content"])

        if not self.skip_tools and "predict_window" not in results:
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    _block(type="text", text="Checking the window."),
                    _block(type="tool_use", id="t1", name="predict_window", input={}),
                    _block(type="tool_use", id="t2", name="explain_features", input={}),
                    _block(type="tool_use", id="t3", name="get_drift_status", input={}),
                ],
            )

        pred = results.get("predict_window", {"label": "AG", "confidence": 1.0})
        feats = results.get("explain_features", {"features": []})["features"][:2]
        drift = results.get("get_drift_status", {"drift_detected": False})
        top = feats[0] if feats else {"feature": "x", "value": 0, "normal_median": 0}
        report = {
            "fault_type": pred["label"],
            "fault_detected": pred["label"] != "NORMAL",
            "confidence": pred["confidence"],
            "summary": (
                f"{pred['label']} fault with confidence {pred['confidence']}. "
                f"The signals show {top['feature']} at {top['value']} against a normal "
                f"median of {top['normal_median']}."
            ),
            "evidence": [
                {"feature": f["feature"], "value": f["value"], "normal_median": f["normal_median"]}
                for f in feats
            ],
            "drift_note": "Drift detected." if drift["drift_detected"] else "No drift detected.",
            "recommended_checks": ["Inspect the relay record for the affected phase."],
        }
        report = self.tamper(report, self.submissions)
        self.submissions += 1
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[_block(type="tool_use", id=f"s{self.submissions}", name="submit_report",
                            input=report)],
        )  # fmt: skip


@pytest.fixture(scope="module")
def trained():
    x, y = make_dataset(150, seed=1)
    feats = extract(x)
    return fit_classifier(feats, y, max_iter=60), feats, y


@pytest.fixture()
def ctx(trained):
    clf, feats, y = trained
    f = extract(simulate_window("AG", np.random.default_rng(3))[None])
    return WindowContext(
        features=f[0],
        proba=clf.predict_proba(f)[0],
        class_indices=[int(c) for c in clf.classes_],
        monitor=DriftMonitor(build_reference(feats, y)),
    )


# ---------- grounding check ----------


def test_grounded_report_is_accepted(ctx):
    report = generate_report(FakeLLM(), ctx, model="fake")
    assert report["fault_type"] == "AG"
    assert report["grounding_checked"] is True
    assert report["evidence"], "report should cite at least one feature"


def test_every_cited_value_comes_from_a_tool_result(ctx):
    report = generate_report(FakeLLM(), ctx, model="fake")
    returned = {f["feature"]: f for f in ctx.explain_features()["features"]}
    for item in report["evidence"]:
        assert item["feature"] in returned
        assert item["value"] == returned[item["feature"]]["value"]
        assert item["normal_median"] == returned[item["feature"]]["normal_median"]
    assert report["confidence"] == ctx.predict_window()["confidence"]


def test_fabricated_evidence_value_is_rejected(ctx):
    def lie(r, attempt):
        r["evidence"][0]["value"] = 12.345
        return r

    with pytest.raises(Exception, match="evidence .* value=12.345"):
        generate_report(FakeLLM(tamper=lie), ctx, model="fake")


def test_fabricated_number_in_summary_is_rejected(ctx):
    def lie(r, attempt):
        r["summary"] += " Fault current reached 8.7 kA."
        return r

    with pytest.raises(Exception, match="summary cites 8.7"):
        generate_report(FakeLLM(tamper=lie), ctx, model="fake")


def test_wrong_fault_type_is_rejected(ctx):
    def lie(r, attempt):
        r["fault_type"] = "ABC"
        return r

    with pytest.raises(Exception, match="fault_type 'ABC'"):
        generate_report(FakeLLM(tamper=lie), ctx, model="fake")


def test_feature_not_returned_by_tool_is_rejected(ctx):
    def lie(r, attempt):
        r["evidence"].append({"feature": "Ib_rms_last", "value": 0.7, "normal_median": 0.7})
        return r

    returned = {f["feature"] for f in ctx.explain_features()["features"]}
    if "Ib_rms_last" in returned:
        pytest.skip("feature happens to be in the top list for this window")
    with pytest.raises(Exception, match="'Ib_rms_last' was not returned"):
        generate_report(FakeLLM(tamper=lie), ctx, model="fake")


def test_model_can_correct_itself_after_rejection(ctx):
    def lie_once(r, attempt):
        if attempt == 0:
            r["confidence"] = 0.5
        return r

    llm = FakeLLM(tamper=lie_once)
    report = generate_report(llm, ctx, model="fake")
    assert report["confidence"] == ctx.predict_window()["confidence"]
    assert llm.submissions == 2
    errors = [
        b
        for m in llm.calls[-1]["messages"]
        if m["role"] == "user" and isinstance(m["content"], list)
        for b in m["content"]
        if b.get("is_error")
    ]
    assert len(errors) == 1 and "confidence 0.5" in errors[0]["content"]


def test_report_without_calling_tools_is_rejected(ctx):
    with pytest.raises(Exception, match="predict_window was not called"):
        generate_report(FakeLLM(skip_tools=True), ctx, model="fake")


def test_percentages_of_returned_values_are_allowed(ctx):
    report = generate_report(FakeLLM(), ctx, model="fake")
    pct = round(report["confidence"] * 100)
    report["summary"] = f"Confidence is {pct}%."
    assert check_grounding(report, ctx) == []


# ---------- tool definitions ----------


def test_submit_report_schema_lists_every_required_field():
    submit = next(t for t in TOOLS if t["name"] == "submit_report")
    props = submit["input_schema"]["properties"]
    assert set(submit["input_schema"]["required"]) == set(props)


def test_every_feature_has_a_readable_description():
    for name in FEATURE_NAMES:
        assert describe_feature(name) != name


# ---------- endpoint ----------


@pytest.fixture()
def api(trained, tmp_path, monkeypatch):
    clf, feats, y = trained
    monkeypatch.setenv("GRIDFAULT_MODEL_PATH", str(save_model(clf, tmp_path / "m.joblib")))
    monkeypatch.setenv(
        "GRIDFAULT_REFERENCE_PATH", str(save_reference(tmp_path / "r.npz", feats, y))
    )
    for var in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "GRIDFAULT_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    with TestClient(app) as c:
        yield c


def _body(label="AG"):
    return {"window": simulate_window(label, np.random.default_rng(3)).tolist()}


def test_explain_without_api_key_is_503(api):
    r = api.post("/explain", json=_body())
    assert r.status_code == 503
    assert "GROQ_API_KEY" in r.json()["detail"]


def test_explain_returns_grounded_report(api):
    app.state.llm = LLM(FakeLLM(), "fake", "fake")
    r = api.post("/explain", json=_body())
    assert r.status_code == 200
    out = r.json()
    assert out["fault_type"] == "AG" and out["grounding_checked"] is True


def test_explain_ungrounded_report_is_502(api):
    def lie(r, attempt):
        r["summary"] += " Estimated distance 42 km."
        return r

    app.state.llm = LLM(FakeLLM(tamper=lie), "fake", "fake")
    r = api.post("/explain", json=_body())
    assert r.status_code == 502
    assert "summary cites 42" in r.json()["detail"]


def test_explain_does_not_feed_the_drift_buffer(api):
    app.state.llm = LLM(FakeLLM(), "fake", "fake")
    api.post("/explain", json=_body())
    assert api.get("/drift").json()["n_buffered"] == 0


def test_explain_validates_input_like_predict(api):
    app.state.llm = LLM(FakeLLM(), "fake", "fake")
    assert api.post("/explain", json={"window": [[1, 2, 3]]}).status_code == 422


@pytest.mark.parametrize(
    "label,expected",
    [
        ("AG", "single line-to-ground fault on phase A"),
        ("BC", "line-to-line fault between phase B and phase C"),
        ("CAG", "double line-to-ground fault between phase C and phase A"),
        ("ABC", "three-phase fault (phases A, B and C)"),
        ("NORMAL", "no fault (normal operation)"),
    ],
)
def test_fault_descriptions_never_treat_ground_as_a_phase(label, expected):
    from gridfault.explain import describe_fault

    assert describe_fault(label) == expected
    assert "phase G" not in describe_fault(label)


def test_drift_summary_says_not_evaluated_when_buffer_is_small(ctx):
    out = ctx.get_drift_status()
    assert out["predicted_class"]["evaluated"] is False
    assert out["summary"].startswith("Drift not evaluated yet: 0 of 100")
    assert "no drift detected" not in out["summary"].lower()
