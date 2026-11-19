# grid-fault-service

End-to-end ML engineering on a power-systems problem: classify three-phase fault types from short relay measurement windows, then train, serve, test, ship through CI, and monitor.

**Status:** Weekend 2 of 3 (model, API, tests, Docker, CI). Drift monitoring and final polish are next.

## What is here

- `src/gridfault/simulate.py`: synthetic generator for 3-phase voltage/current windows (11 classes: normal + 10 fault types). Varies fault resistance, distance, inception time, DC offset, harmonics, noise. Includes load steps as hard negatives.
- `src/gridfault/features.py`: fundamental phasors, symmetrical components (zero/positive/negative sequence), pre/post-fault RMS ratios.
- `src/gridfault/train.py`: gradient-boosting baseline, tracked with MLflow; saves `artifacts/model.joblib`.
- `src/gridfault/model.py`: fit, save and load helpers shared by training, API and tests.
- `src/gridfault/api.py`: FastAPI service with Pydantic-validated input.
- `src/gridfault/make_example.py`: regenerates `examples/*.json` request bodies.

## Quickstart: run the service

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
| `POST /predict` | body `{"window": [[...240 floats] x 6]}` in per-unit, channel order Va, Vb, Vc, Ia, Ib, Ic |

Invalid input (wrong channel count, wrong length, NaN, values beyond +/-100 pu) returns 422 with a message saying what is wrong. Interactive docs at `/docs`.

The model is trained inside the Docker build rather than committed, so the image is reproducible and the model always matches the image's library versions. A multi-stage build keeps MLflow and matplotlib out of the runtime image.

## Develop

```bash
pip install -r requirements-dev.txt
pytest -q                 # 25 tests: simulator physics, features, API behaviour
ruff check . && ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs lint, tests, then builds the image and smoke-tests the running container with a real `/predict` call.

## Train and track experiments

```bash
pip install -r requirements-train.txt
PYTHONPATH=src python -m gridfault.train
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Baseline results (seed 0)

| Split | Accuracy | Macro F1 |
|---|---|---|
| Test (same distribution as training) | 0.999 | 0.999 |
| Shifted (higher fault resistance, farther faults, more noise) | 0.796 | 0.790 |

![results](reports/figures/results.png)

The in-distribution score is near-perfect because the simulator is clean and the features are physics-based, so it says little on its own. The shifted set is the informative number: most errors are high-resistance three-phase faults (ABC) being read as normal load, since a balanced fault with a small voltage sag has no sequence-component signature.

## Limitations

- Data is synthetic and simplified (single source, no transformers, no CT saturation, no network topology). It is not a substitute for EMT simulation or field recordings.
- The shifted split is one hand-picked shift, not a full robustness study.
