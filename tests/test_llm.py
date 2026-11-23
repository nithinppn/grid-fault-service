"""The Groq adapter: OpenAI-format requests and responses, driven through the same report
loop and grounding check as the Anthropic path. No network."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from gridfault.drift import DriftMonitor, build_reference
from gridfault.explain import GroundingError, WindowContext, generate_report
from gridfault.features import extract
from gridfault.llm import GroqAdapter, make_llm
from gridfault.model import fit_classifier
from gridfault.simulate import make_dataset, simulate_window


def _call(cid, name, args):
    raw = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=raw))


def _reply(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeGroq:
    """OpenAI-style client: first asks for the three tools, then submits a report built from
    the `role: tool` messages it received."""

    def __init__(self, tamper=None, bad_json=False):
        self.tamper = tamper or (lambda r: r)
        self.bad_json = bad_json
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kw):
        self.requests.append(kw)
        tool_msgs = [m for m in kw["messages"] if m["role"] == "tool"]
        if not tool_msgs:
            return _reply(
                "Looking at the window.",
                [
                    _call("c1", "predict_window", {}),
                    _call("c2", "explain_features", {}),
                    _call("c3", "get_drift_status", {}),
                ],
            )
        if self.bad_json:
            return _reply(None, [_call("c4", "submit_report", "{not json")])
        res = {
            m["name"]: json.loads(m["content"]) for m in tool_msgs if m["name"] != "submit_report"
        }
        pred, feats = res["predict_window"], res["explain_features"]["features"][:2]
        report = {
            "fault_type": pred["label"],
            "fault_detected": pred["fault_detected"],
            "confidence": pred["confidence"],
            "summary": f"{pred['label']} fault, confidence {pred['confidence']}.",
            "evidence": [
                {"feature": f["feature"], "value": f["value"], "normal_median": f["normal_median"]}
                for f in feats
            ],
            "drift_note": "Not enough traffic yet to evaluate drift.",
            "recommended_checks": ["Compare with the remote-end relay record."],
        }
        return _reply(
            None, [_call(f"c{len(self.requests) + 3}", "submit_report", self.tamper(report))]
        )


@pytest.fixture(scope="module")
def ctx_factory():
    x, y = make_dataset(150, seed=1)
    feats = extract(x)
    clf = fit_classifier(feats, y, max_iter=60)
    ref = build_reference(feats, y)

    def make():
        f = extract(simulate_window("BC", np.random.default_rng(4))[None])
        return WindowContext(
            f[0], clf.predict_proba(f)[0], [int(c) for c in clf.classes_], DriftMonitor(ref)
        )

    return make


def test_groq_report_goes_through_the_same_grounding_check(ctx_factory):
    report = generate_report(GroqAdapter(FakeGroq()), ctx_factory(), model="groq-model")
    assert report["fault_type"] == "BC"
    assert report["grounding_checked"] is True
    assert report["model"] == "groq-model"


def test_request_is_translated_to_openai_format(ctx_factory):
    fake = FakeGroq()
    generate_report(GroqAdapter(fake), ctx_factory(), model="m")
    first, second = fake.requests[0], fake.requests[1]
    assert first["tool_choice"] == "auto"
    assert first["messages"][0]["role"] == "system"
    assert all(t["type"] == "function" for t in first["tools"])
    assert "strict" not in json.dumps(first["tools"])  # Anthropic-only field is dropped
    calls = [m for m in second["messages"] if m["role"] == "assistant"][0]["tool_calls"]
    tools = [m for m in second["messages"] if m["role"] == "tool"]
    assert [c["id"] for c in calls] == [t["tool_call_id"] for t in tools]
    assert {t["name"] for t in tools} == {"predict_window", "explain_features", "get_drift_status"}


def test_fabrication_is_rejected_on_groq_too(ctx_factory):
    def lie(r):
        r["summary"] += " Fault resistance about 2.5 ohm."
        return r

    fake = FakeGroq(tamper=lie)
    with pytest.raises(GroundingError, match="summary cites 2.5"):
        generate_report(GroqAdapter(fake), ctx_factory(), model="m")
    rejection = [m for m in fake.requests[-1]["messages"] if m["role"] == "tool"][-1]
    assert rejection["content"].startswith("ERROR: Report rejected")


def test_malformed_tool_arguments_fail_the_check_instead_of_crashing(ctx_factory):
    with pytest.raises(GroundingError):
        generate_report(GroqAdapter(FakeGroq(bad_json=True)), ctx_factory(), model="m")


def test_provider_selection(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "GRIDFAULT_LLM_PROVIDER",
        "GRIDFAULT_LLM_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    assert make_llm() is None
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    llm = make_llm()
    assert llm.provider == "groq" and llm.model == "openai/gpt-oss-120b"
    assert isinstance(llm.client, GroqAdapter)
    monkeypatch.setenv("GRIDFAULT_LLM_MODEL", "llama-3.3-70b-versatile")
    assert make_llm().model == "llama-3.3-70b-versatile"
