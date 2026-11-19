# grid-fault-service

End-to-end ML engineering on a power-systems problem: classify three-phase fault types from short relay measurement windows, then train, serve, test, ship through CI, and monitor.

**Status:** Weekend 1 of 3 (model and data). Serving, tests, CI and drift monitoring are next.

## What is here

- `src/gridfault/simulate.py`: synthetic generator for 3-phase voltage/current windows (11 classes: normal + 10 fault types). Varies fault resistance, distance, inception time, DC offset, harmonics, noise. Includes load steps as hard negatives.
- `src/gridfault/features.py`: fundamental phasors, symmetrical components (zero/positive/negative sequence), pre/post-fault RMS ratios.
- `src/gridfault/train.py`: gradient-boosting baseline, tracked with MLflow.

## Run

```bash
pip install -r requirements.txt
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
