# grid-fault-service

![CI](https://github.com/nithinppn/grid-fault-service/actions/workflows/ci.yml/badge.svg)

A small end-to-end ML service for power-system protection: classify the type of a three-phase fault (or no fault) from a 60 ms window of relay measurements, serve it over an API, and watch for input drift in production.

It covers the whole loop on one problem: data generation, feature engineering, experiment tracking, serving with input validation, tests, containerisation, CI, drift monitoring, and an LLM-written operator report that may only cite numbers the service computed.

## Architecture

```mermaid
flowchart LR
    A["Simulator<br/>3-phase windows"] --> B["Features<br/>phasors, sequence components"]
    B --> C["Train<br/>HistGradientBoosting + MLflow"]
    C --> D[("model.joblib<br/>reference.npz")]
    D --> E["FastAPI<br/>predict, health"]
    E --> F["Drift monitor<br/>per predicted class"]
    F --> G["GET drift"]
    E --> H["LLM agent<br/>tool calls into predict + drift"]
    F --> H
    H --> I["POST explain<br/>grounding-checked report"]
```

## Results

| Split | Accuracy | Macro F1 |
|---|---|---|
| Test (same distribution as training) | 0.999 | 0.999 |
| Shifted (higher fault resistance, farther faults, more noise) | 0.796 | 0.790 |

![results](reports/figures/results.png)

The in-distribution score is near-perfect because the simulator is clean and the features are physics-based, so it says little on its own. The shifted set is the informative number: most of the errors are high-resistance three-phase faults (ABC) read as normal load, because a balanced fault with a small voltage sag has no sequence-component signature.

## Quickstart

```bash
docker build -t grid-fault-service .        # trains the model during the build
docker run --rm -p 8000:8000 grid-fault-service
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @examples/ag_fault.json
```

```json
{"label":"AG","fault_detected":true,"confidence":1.0,"probabilities":{"NORMAL":0.0,"AG":1.0,"...":0.0}}
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | 200 when the model is loaded, 503 otherwise |
| `POST /predict` | body `{"window": [[...240 floats] x 6]}` in per-unit; channel order Va, Vb, Vc, Ia, Ib, Ic |
| `GET /drift` | compare recent inputs with the training distribution, per predicted class |
| `POST /drift/reset` | clear the drift buffer |
| `POST /explain` | same body as `/predict`; returns a plain-language operator report (needs `GROQ_API_KEY` or `ANTHROPIC_API_KEY`) |

Invalid input (wrong channel count, wrong length, NaN, values beyond +/-100 pu) returns 422 with a message saying what is wrong. Interactive docs at `/docs`.

The model is trained inside the Docker build rather than committed, so the image is reproducible and the model always matches the image's library versions. A multi-stage build keeps MLflow and matplotlib out of the runtime image.

## Drift monitoring

Every `/predict` call stores the window's feature vector (not the raw waveform) in a rolling per-class buffer. `GET /drift` compares each buffer with the training distribution of that class.

- **Per predicted class.** A healthy network sends mostly normal windows, while the training set is class-balanced. Comparing the whole stream with the training set would alarm on that change in class mix alone. Within a class the question is narrower: do the windows the model calls "AG" still look like the AG windows it learned from?
- **Two tests must agree** for a feature to count as drifted: Population Stability Index above 0.2 and a two-sample Kolmogorov-Smirnov p-value below 0.001. A class is flagged when at least 3 features drifted, and only once it has 100 buffered windows.
- **Checked against simulation** (true labels as buckets): across 8 in-distribution runs there was not a single drifted feature in any class, and all 4 shifted runs flagged every class.

Reproduce it against a running service (about a minute, it sends ~2,600 requests):

```bash
PYTHONPATH=src python -m gridfault.demo_drift
```

Output from a real run (rows with fewer than 100 windows are not evaluated and are omitted):

```text
Phase 1: in-distribution traffic
  status=ok  drift_detected=False  buffered=1320
    NORMAL: n=120  drifted_features=0   max_psi=0.0
        AG: n=120  drifted_features=0   max_psi=0.0
       ...  (all 11 classes: 0 drifted features)

Phase 2: shifted traffic (higher fault resistance, farther faults, more noise)
  status=ok  drift_detected=True  buffered=1320
    NORMAL: n=233  drifted_features=29  max_psi=2.66
        AG: n=110  drifted_features=28  max_psi=8.283
        AB: n=139  drifted_features=31  max_psi=8.283
       ...
```

In phase 2 the NORMAL bucket holds 233 windows although only 120 were truly normal, so roughly 110 shifted faults ended up there as misread normal load, the failure shown in the results above. Several fault classes (BG, CA, ABC) fell below 100 windows because their members were misclassified, so they were not evaluated. Drift detection and the accuracy drop point at the same problem.

## Operator report

`POST /explain` turns a prediction into a short report a grid operator can act on: fault type, confidence, which signals look abnormal, drift status, and what to check next. It is written by an LLM with tool calling (Groq or Anthropic), but the model never has to make up a number:

1. The model gets three tools that run this service's own code on the request window: `predict_window` (the classifier), `explain_features` (the measured features that deviate most from healthy operation) and `get_drift_status` (the drift monitor).
2. It must finish by calling `submit_report` with a structured report.
3. A grounding check runs before the report is returned. The fault type and confidence must equal the prediction, every cited feature value must match a tool result, and every number in the free text must appear in some tool result (a percentage of a returned value is allowed). If the check fails, the model is told exactly which claims failed and gets one retry; if it fails again the endpoint returns 502 instead of a report with invented numbers.

```bash
docker run --rm -p 8000:8000 -e GROQ_API_KEY grid-fault-service
curl -X POST localhost:8000/explain -H 'Content-Type: application/json' -d @examples/ag_fault.json
```

The provider is whichever key is set (`ANTHROPIC_API_KEY` first, then `GROQ_API_KEY`), or `GRIDFAULT_LLM_PROVIDER=groq|anthropic`. Defaults are `openai/gpt-oss-120b` on Groq and `claude-haiku-5-5` on Anthropic; set `GRIDFAULT_LLM_MODEL` to change the model. The report loop speaks the Anthropic Messages format, and `llm.py` translates it to Groq's OpenAI-compatible format, so both providers go through the same tools and the same grounding check. Without a key `/explain` returns 503 and everything else works as before. `/explain` does not add the window to the drift buffer, so calling `/predict` and `/explain` on the same window does not count it twice.

What the model sees for `examples/ag_fault.json` (real tool output, top 3 features shown):

```json
{"label": "AG", "fault_detected": true, "confidence": 1.0}
{"feature": "I_seq2_over_seq1", "description": "negative-to-positive sequence current ratio", "value": 0.655, "normal_median": 0.001}
{"feature": "seq0_I", "description": "zero-sequence current magnitude", "value": 0.442, "normal_median": 0.0}
{"feature": "seq2_I", "description": "negative-sequence current magnitude", "value": 0.442, "normal_median": 0.001}
```

Equal zero- and negative-sequence current is the textbook signature of a single-line-to-ground fault, so the evidence the report cites agrees with the physics, not only with the classifier.

Example report from a live run (Groq, `openai/gpt-oss-120b`, `examples/ag_fault.json`; evidence list shortened to 3 of 5):

```json
{
  "fault_type": "AG",
  "fault_detected": true,
  "confidence": 1.0,
  "summary": "The classifier identified a single line-to-ground fault on phase A with high confidence. Abnormal sequence current ratios and elevated zero-sequence currents support this diagnosis. Operators should verify relay settings and equipment on phase A.",
  "evidence": [
    {"feature": "I_seq2_over_seq1", "value": 0.655, "normal_median": 0.001},
    {"feature": "seq0_I", "value": 0.442, "normal_median": 0.0},
    {"feature": "seq2_I", "value": 0.442, "normal_median": 0.001}
  ],
  "drift_note": "Drift not evaluated yet: 0 of 100 AG windows buffered, so no conclusion about drift is possible.",
  "recommended_checks": [
    "Inspect the relay record for phase A for line-to-ground fault indications.",
    "Compare protection relay measurements with the remote-end relay for phase A.",
    "Verify CT and VT wiring on phase A for proper connections."
  ],
  "model": "openai/gpt-oss-120b",
  "grounding_checked": true
}
```

The first live run passed the grounding check but still read badly: it wrote "no drift detected" when drift had not been evaluated, and called ground "phase G". No number was wrong, so the check could not catch it. The fix was to move the wording into the tools (`predict_window` returns a fault description, `get_drift_status` a drift summary) so the model copies it instead of inventing it. The check guards numbers; the tools have to guard the words.

`explain_features` is a deviation-from-normal measure (robust z-score against healthy training windows), not an attribution of the classifier's decision, and the prompt tells the model to word it that way. The tests (`tests/test_explain.py`, `tests/test_llm.py`) use scripted fake clients for both formats, so CI needs no API key: they check that a grounded report is accepted and that a fabricated evidence value, an invented number in the summary, a wrong fault type, a feature the tool never returned, or a report written without calling the tools are all rejected, and that malformed tool arguments from the model fail the check instead of crashing the service.

## Develop

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src pytest -q        # 68 tests: simulator physics, features, API, drift, operator report
ruff check . && ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs lint, tests, then builds the image and smoke-tests the running container with real `/predict` and `/drift` calls.

Train and track experiments (needs the extra training dependencies):

```bash
pip install -r requirements-train.txt
PYTHONPATH=src python -m gridfault.train
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Layout

- `src/gridfault/simulate.py`: synthetic generator for 3-phase voltage/current windows (11 classes: normal plus 10 fault types). Varies fault resistance, distance, inception time, DC offset, harmonics and noise; includes load steps as hard negatives.
- `src/gridfault/features.py`: fundamental phasors, symmetrical components (zero/positive/negative sequence), pre/post-fault RMS ratios.
- `src/gridfault/train.py`: baseline training, MLflow tracking, saves `artifacts/model.joblib` and the drift reference.
- `src/gridfault/model.py`: fit, save and load helpers.
- `src/gridfault/api.py`: FastAPI service with Pydantic-validated input.
- `src/gridfault/drift.py`: PSI and KS drift monitor.
- `src/gridfault/demo_drift.py`: drift demo against a running service.
- `src/gridfault/explain.py`: operator report agent (tools, tool loop, grounding check).
- `src/gridfault/llm.py`: provider selection and the Groq (OpenAI-format) adapter.
- `src/gridfault/make_example.py`: regenerates `examples/*.json`.

## Limitations

- **The data is synthetic and simplified:** single source, no transformers, no CT saturation, no network topology. It is not a substitute for EMT simulation or field recordings, and the model's scores say nothing about real relays.
- **The shifted split is one hand-picked shift**, not a robustness study. The drift thresholds were checked against that same shift: its sensitivity to mild or gradual drift is not measured.
- **Drift strata use the model's own predictions**, so misclassified windows land in the wrong bucket (visible above). That still tends to show up as drift, but it is not a clean per-true-class test.
- **The drift buffer lives in process memory.** It resets on restart and is per worker. A production setup would export the statistics to a shared metrics store.
- The grounding check verifies numbers, not reasoning: a report can cite only real values and still draw a weak conclusion from them. It is a guard against invented figures, not a review.
- `/explain` makes several model calls per request (typically two to three), so it is slower and costs more than `/predict`; it is meant for an operator looking at one event, not for every window.
- No authentication or rate limiting on the API.

## License

MIT, see `LICENSE`.
