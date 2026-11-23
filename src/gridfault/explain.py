"""Operator fault report: an LLM writes it, but only from numbers the service computed.

The client is anything with an Anthropic-style `messages.create`; see `llm.py` for Groq.

How it works:

1. The model gets tools that run this service's own logic on the request window:
   `predict_window` (the classifier), `explain_features` (which measured features deviate
   most from healthy operation) and `get_drift_status` (the drift monitor).
2. It must finish by calling `submit_report` with a structured report.
3. Before the report is accepted, `check_grounding` verifies that the fault type and
   confidence match the prediction, every cited feature value matches a tool result, and
   every number in the free text appears in some tool result. On a violation the model gets
   one chance to correct itself; after that the request fails instead of returning a report
   with invented numbers.

`explain_features` is a deviation-from-normal measure (robust z-score against the healthy
training windows), not a model attribution. The report wording says so.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .drift import DriftMonitor
from .features import FEATURE_NAMES
from .simulate import CLASSES

MAX_TURNS = 6
MAX_CORRECTIONS = 1
TOP_K_FEATURES = 5

_NUMBER = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?")
_PHASES = {"Va": "phase A voltage", "Vb": "phase B voltage", "Vc": "phase C voltage",
           "Ia": "phase A current", "Ib": "phase B current", "Ic": "phase C current"}  # fmt: skip


def describe_fault(label: str) -> str:
    """Plain-language fault description: letters are phases, G means ground."""
    if label == "NORMAL":
        return "no fault (normal operation)"
    ground = label.endswith("G")
    phases = [p for p in label.rstrip("G")] if ground else list(label)
    names = " and ".join(f"phase {p}" for p in phases)
    if len(phases) == 3:
        return "three-phase fault (phases A, B and C)"
    if len(phases) == 1:
        return f"single line-to-ground fault on {names}"
    return f"{'double line-to-ground' if ground else 'line-to-line'} fault between {names}"


class GroundingError(RuntimeError):
    """The model did not produce a report grounded in the service's own outputs."""


def describe_feature(name: str) -> str:
    """Plain-language description of a feature name, for the operator."""
    if name.startswith(("seq0_", "seq1_", "seq2_")):
        seq = {"seq0": "zero", "seq1": "positive", "seq2": "negative"}[name[:4]]
        qty = "voltage" if name.endswith("_V") else "current"
        return f"{seq}-sequence {qty} magnitude"
    if name.endswith("_over_seq1"):
        qty = "voltage" if name.startswith("V") else "current"
        seq = "negative" if "seq2" in name else "zero"
        return f"{seq}-to-positive sequence {qty} ratio"
    ch, _, kind = name.partition("_")
    base = _PHASES.get(ch, ch)
    return {
        "rms_last": f"{base} RMS, last cycle",
        "rms_ratio": f"{base} RMS, last cycle vs first cycle",
        "absmax_ratio": f"{base} peak, last cycle vs first cycle",
        "mean_last": f"{base} DC offset, last cycle",
    }.get(kind, name)


TOOLS: list[dict[str, Any]] = [
    {
        "name": "predict_window",
        "description": (
            "Run the fault classifier on the measurement window from this request. Returns the "
            "predicted class, its confidence and the three most likely classes."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "explain_features",
        "description": (
            "List the measured features of this window that deviate most from healthy "
            "operation (robust z-score against the normal training windows). This shows what "
            "is abnormal in the signals; it is not an attribution of the model's decision."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_drift_status",
        "description": (
            "Return the drift monitor's current status: whether recent traffic differs from "
            "the training data, overall and for the predicted class."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "submit_report",
        "description": (
            "Submit the final operator report. Call this exactly once, after the other tools. "
            "Every number you write must be copied from a tool result."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "fault_type": {"type": "string", "enum": CLASSES},
                "fault_detected": {"type": "boolean"},
                "confidence": {"type": "number"},
                "summary": {
                    "type": "string",
                    "description": "Two or three sentences for a grid operator.",
                },
                "evidence": {
                    "type": "array",
                    "description": "Features from explain_features that support the call.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "feature": {"type": "string"},
                            "value": {"type": "number"},
                            "normal_median": {"type": "number"},
                        },
                        "required": ["feature", "value", "normal_median"],
                    },
                },
                "drift_note": {"type": "string"},
                "recommended_checks": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "fault_type",
                "fault_detected",
                "confidence",
                "summary",
                "evidence",
                "drift_note",
                "recommended_checks",
            ],
        },
    },
]

SYSTEM_PROMPT = """You write short fault reports for transmission-grid operators.

Call predict_window, explain_features and get_drift_status, then call submit_report once.
Rules:
- fault_type and confidence must equal what predict_window returned. Name the fault with
  predict_window's "description"; in fault codes the letters are phases and G means ground,
  so never call G a phase.
- drift_note must restate get_drift_status's "summary". If drift was not evaluated, say so;
  never write "no drift detected" in that case.
- Copy every number exactly from a tool result. Do not compute, round differently, or estimate
  new numbers. If you need a number that no tool gave you, leave it out.
- explain_features shows which signals look abnormal, not why the model decided. Phrase
  evidence as "the signals show ...", never as "the model used ...".
- If confidence is below 0.8 or drift is detected, say the call should be verified.
- recommended_checks: at most three concrete checks, e.g. inspect the relay record for the
  affected phases, compare with the remote-end relay, check CT and VT wiring.
- The data is simulated; do not name real substations or equipment."""


@dataclass
class WindowContext:
    """Everything the tools may reveal about one request, computed by the service."""

    features: np.ndarray  # shape (F,)
    proba: np.ndarray  # aligned with `class_indices`
    class_indices: list[int]
    monitor: DriftMonitor
    tool_outputs: list[dict] = field(default_factory=list)

    @property
    def predicted(self) -> tuple[str, float]:
        best = int(np.argmax(self.proba))
        return CLASSES[self.class_indices[best]], round(float(self.proba[best]), 4)

    def predict_window(self) -> dict:
        label, conf = self.predicted
        order = np.argsort(self.proba)[::-1][:3]
        return {
            "label": label,
            "fault_detected": label != "NORMAL",
            "description": describe_fault(label),
            "confidence": conf,
            "top_classes": [
                {
                    "label": CLASSES[self.class_indices[i]],
                    "probability": round(float(self.proba[i]), 4),
                }
                for i in order
            ],
        }

    def explain_features(self) -> dict:
        normal = self.monitor.reference[CLASSES.index("NORMAL")]
        med = np.median(normal, axis=0)
        q75, q25 = np.percentile(normal, [75, 25], axis=0)
        scale = np.maximum(q75 - q25, 1e-3)
        z = (self.features - med) / scale
        top = np.argsort(-np.abs(z))[:TOP_K_FEATURES]
        return {
            "method": "robust z-score against healthy training windows",
            "features": [
                {
                    "feature": FEATURE_NAMES[j],
                    "description": describe_feature(FEATURE_NAMES[j]),
                    "value": round(float(self.features[j]), 3),
                    "normal_median": round(float(med[j]), 3),
                    "deviation_z": round(float(z[j]), 1),
                }
                for j in top
            ],
        }

    def get_drift_status(self) -> dict:
        report = self.monitor.report()
        label, _ = self.predicted
        cls = report["classes"].get(label, {"n": 0, "evaluated": False})
        need = report["min_samples_per_class"]
        if not cls["evaluated"]:
            note = (
                f"Drift not evaluated yet: {cls['n']} of {need} {label} windows buffered, "
                "so no conclusion about drift is possible."
            )
        elif cls.get("drifted"):
            note = f"Drift detected: recent {label} windows differ from the training data."
        else:
            note = f"No drift detected for {label} windows."
        if report["drift_detected"] and not cls.get("drifted"):
            note += " Drift was detected in other classes."
        return {
            "summary": note,
            "status": report["status"],
            "drift_detected": report["drift_detected"],
            "min_samples_per_class": report["min_samples_per_class"],
            "predicted_class": {
                "label": label,
                "windows_buffered": cls["n"],
                "evaluated": cls["evaluated"],
                "drifted": cls.get("drifted"),
            },
        }

    def run_tool(self, name: str) -> dict:
        fn = {
            "predict_window": self.predict_window,
            "explain_features": self.explain_features,
            "get_drift_status": self.get_drift_status,
        }.get(name)
        if fn is None:
            raise KeyError(f"unknown tool {name}")
        out = fn()
        self.tool_outputs.append({"tool": name, "result": out})
        return out


def _numbers_in(obj: Any) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, int | float):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _numbers_in(v)]
    if isinstance(obj, list | tuple):
        return [n for v in obj for n in _numbers_in(v)]
    if isinstance(obj, str):
        return [float(m) for m in _NUMBER.findall(obj)]
    return []


def _matches(token: str, known: list[float]) -> bool:
    """True if a number written in text matches a known value at the precision written."""
    x = float(token)
    decimals = len(token.split(".")[1]) if "." in token else 0
    tol = 0.5 * 10**-decimals + 1e-9
    for v in known:
        if abs(x - v) <= tol or abs(x - round(v, decimals)) <= 1e-9:
            return True
        if abs(x - 100 * v) <= max(tol, 0.5):  # 0.973 written as 97%
            return True
    return False


def check_grounding(report: dict, ctx: WindowContext) -> list[str]:
    """Return a list of violations; empty means every claim traces to a tool result."""
    problems: list[str] = []
    called = {t["tool"] for t in ctx.tool_outputs}
    for needed in ("predict_window", "explain_features", "get_drift_status"):
        if needed not in called:
            problems.append(f"{needed} was not called before submit_report")

    label, conf = ctx.predicted
    if report.get("fault_type") != label:
        problems.append(f"fault_type {report.get('fault_type')!r} but prediction was {label!r}")
    if report.get("fault_detected") != (label != "NORMAL"):
        problems.append("fault_detected does not match the prediction")
    if (
        not isinstance(report.get("confidence"), int | float)
        or abs(report["confidence"] - conf) > 5e-4
    ):
        problems.append(f"confidence {report.get('confidence')} but prediction was {conf}")

    feats = {f["feature"]: f for f in ctx.explain_features()["features"]}
    for item in report.get("evidence", []):
        ref = feats.get(item.get("feature"))
        if ref is None:
            problems.append(f"evidence feature {item.get('feature')!r} was not returned")
            continue
        for key in ("value", "normal_median"):
            if abs(float(item.get(key, float("nan"))) - ref[key]) > 5e-4:
                problems.append(
                    f"evidence {ref['feature']} {key}={item.get(key)} but tool returned {ref[key]}"
                )

    known = [n for t in ctx.tool_outputs for n in _numbers_in(t["result"])]
    for field_name in ("summary", "drift_note"):
        for tok in _NUMBER.findall(report.get(field_name, "")):
            if not _matches(tok, known):
                problems.append(f"{field_name} cites {tok}, which no tool returned")
    for check in report.get("recommended_checks", []):
        for tok in _NUMBER.findall(check):
            if not _matches(tok, known):
                problems.append(f"recommended_checks cites {tok}, which no tool returned")
    return problems


def _block_to_dict(block: Any) -> dict:
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {"type": "text", "text": getattr(block, "text", "")}


def generate_report(client: Any, ctx: WindowContext, model: str) -> dict:
    """Run the tool loop and return a grounded report, or raise GroundingError."""
    messages: list[dict] = [
        {"role": "user", "content": "Write the operator report for this measurement window."}
    ]
    corrections = 0
    for _ in range(MAX_TURNS):
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice={"type": "auto"},
            messages=messages,
        )
        blocks = [_block_to_dict(b) for b in resp.content]
        messages.append({"role": "assistant", "content": blocks})
        tool_uses = [b for b in blocks if b["type"] == "tool_use"]
        if not tool_uses:
            messages.append(
                {"role": "user", "content": "Finish by calling submit_report with the report."}
            )
            continue

        results = []
        for tu in tool_uses:
            if tu["name"] == "submit_report":
                problems = check_grounding(tu["input"], ctx)
                if not problems:
                    return {**tu["input"], "model": model, "grounding_checked": True}
                if corrections >= MAX_CORRECTIONS:
                    raise GroundingError("; ".join(problems))
                corrections += 1
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "is_error": True,
                        "content": "Report rejected, fix these and resubmit: "
                        + "; ".join(problems),
                    }
                )
                continue
            try:
                out = ctx.run_tool(tu["name"])
                results.append(
                    {"type": "tool_result", "tool_use_id": tu["id"], "content": json.dumps(out)}
                )
            except KeyError as e:
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "is_error": True,
                        "content": str(e),
                    }
                )
        messages.append({"role": "user", "content": results})
    raise GroundingError(f"no accepted report after {MAX_TURNS} turns")
