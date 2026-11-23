"""Input-drift monitoring for the fault classifier.

Design choices, and why:

* The monitor works on the extracted feature vectors, not raw waveforms, so it is cheap and
  the buffer holds no raw measurements.
* Drift is tested **per predicted class**. A fleet that is mostly healthy sends far more NORMAL
  windows than the class-balanced training set contains. Comparing the whole stream with the
  training set would raise an alarm for that change in class mix alone. Within one predicted
  class the question is narrower: do the windows the model calls "AG" still look like the AG
  windows it was trained on?
* A feature counts as drifted only if two tests agree: Population Stability Index above 0.2
  (the usual "significant shift" level) and a two-sample Kolmogorov-Smirnov test below a
  strict p-value. A class counts as drifted if several features did. Requiring agreement and
  a minimum sample size keeps false alarms rare; the thresholds were checked against
  in-distribution and shifted simulated data (see tests/test_drift.py).

Limitation: the stratum is the model's own prediction, so a drifted window that is
misclassified lands in a different stratum. That still tends to show up, because the
misclassified window looks unlike the training windows of the class it was assigned to.
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import numpy as np
from scipy import stats

from .features import FEATURE_NAMES
from .model import ROOT
from .simulate import CLASSES

REFERENCE_PATH = ROOT / "artifacts" / "reference.npz"

N_BINS = 10
PSI_THRESHOLD = 0.2
KS_ALPHA = 1e-3
MIN_DRIFTED_FEATURES = 3
MIN_SAMPLES = 100  # per predicted class before it is evaluated
MAX_BUFFER = 500  # most recent windows kept per predicted class
_EPS = 1e-4


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = N_BINS) -> float:
    """Population Stability Index of `current` against `reference` (1-D samples).

    Bin edges are reference quantiles, so each reference bin holds ~1/n_bins of the data.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1]))
    ref_p = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=edges.size + 1)
    cur_p = np.bincount(np.searchsorted(edges, current, side="right"), minlength=edges.size + 1)
    ref_p = np.clip(ref_p / len(reference), _EPS, None)
    cur_p = np.clip(cur_p / len(current), _EPS, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def build_reference(
    features: np.ndarray, labels: np.ndarray, n_per_class: int = 300, seed: int = 0
) -> dict[int, np.ndarray]:
    """Subsample training features per class to form the reference distribution."""
    rng = np.random.default_rng(seed)
    ref = {}
    for k in range(len(CLASSES)):
        rows = np.flatnonzero(labels == k)
        if rows.size:
            pick = rng.choice(rows, size=min(n_per_class, rows.size), replace=False)
            ref[k] = features[pick].astype(np.float32)
    return ref


def save_reference(path: Path, features: np.ndarray, labels: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = build_reference(features, labels)
    np.savez_compressed(path, **{f"class_{k}": v for k, v in ref.items()})
    return path


def load_reference(path: Path = REFERENCE_PATH) -> dict[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {int(name.split("_")[1]): z[name] for name in z.files}


class DriftMonitor:
    """Rolling per-class buffer of feature vectors, compared with the training reference."""

    def __init__(
        self,
        reference: dict[int, np.ndarray],
        min_samples: int = MIN_SAMPLES,
        max_buffer: int = MAX_BUFFER,
    ) -> None:
        self.reference = reference
        self.min_samples = min_samples
        self._buffers: dict[int, deque[np.ndarray]] = {
            k: deque(maxlen=max_buffer) for k in range(len(CLASSES))
        }
        self._lock = threading.Lock()

    def add(self, class_idx: int, features: np.ndarray) -> None:
        with self._lock:
            self._buffers[class_idx].append(np.asarray(features, dtype=np.float32))

    def reset(self) -> None:
        with self._lock:
            for buf in self._buffers.values():
                buf.clear()

    def _evaluate_class(self, k: int, current: np.ndarray) -> dict:
        ref = self.reference[k]
        drifted = []
        for j, name in enumerate(FEATURE_NAMES):
            p = psi(ref[:, j], current[:, j])
            if p > PSI_THRESHOLD:
                ks_p = float(stats.ks_2samp(ref[:, j], current[:, j]).pvalue)
                if ks_p < KS_ALPHA:
                    drifted.append((name, p, ks_p))
        drifted.sort(key=lambda t: -t[1])
        return {
            "n_drifted_features": len(drifted),
            "max_psi": round(drifted[0][1], 3) if drifted else 0.0,
            "top_drifted_features": [
                {"feature": n, "psi": round(p, 3), "ks_p_value": float(f"{kp:.3g}")}
                for n, p, kp in drifted[:5]
            ],
            "drifted": len(drifted) >= MIN_DRIFTED_FEATURES,
        }

    def report(self) -> dict:
        with self._lock:
            snapshot = {k: np.stack(buf) for k, buf in self._buffers.items() if len(buf)}
        classes = {}
        for k, current in snapshot.items():
            entry = {"n": len(current), "evaluated": False}
            if len(current) >= self.min_samples and k in self.reference:
                entry["evaluated"] = True
                entry.update(self._evaluate_class(k, current))
            classes[CLASSES[k]] = entry
        evaluated = [c for c in classes.values() if c["evaluated"]]
        return {
            "status": "ok" if evaluated else "insufficient_data",
            "drift_detected": any(c["drifted"] for c in evaluated),
            "min_samples_per_class": self.min_samples,
            "n_buffered": sum(c["n"] for c in classes.values()),
            "classes": classes,
        }
