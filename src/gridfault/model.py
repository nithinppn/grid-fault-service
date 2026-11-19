"""Model fitting and (de)serialisation, shared by training, the API and the tests."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"


def fit_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 200,
    learning_rate: float = 0.1,
    seed: int = 0,
) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        early_stopping=False,
        random_state=seed,
    )
    clf.fit(features, labels)
    return clf


def save_model(clf: HistGradientBoostingClassifier, path: Path = MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    return path


def load_model(path: Path = MODEL_PATH) -> HistGradientBoostingClassifier:
    """Load a model produced by `save_model`.

    joblib uses pickle: only load files this project wrote itself, never a file
    from an untrusted source.
    """
    return joblib.load(path)
