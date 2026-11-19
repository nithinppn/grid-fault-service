"""Train the baseline fault classifier, log to MLflow, save the results figure.

Run:  python -m gridfault.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .features import extract
from .simulate import CLASSES, make_dataset

ROOT = Path(__file__).resolve().parents[2]


def plot_results(cm: np.ndarray, f1_test: np.ndarray, f1_ood: np.ndarray, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax1.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax1.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    ax1.set_yticks(range(len(CLASSES)), CLASSES)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Confusion matrix, shifted set (row-normalised)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cmn[i, j] > 0.005:
                ax1.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                         color="white" if cmn[i, j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax1, fraction=0.046)

    x = np.arange(len(CLASSES))
    ax2.bar(x - 0.2, f1_test, 0.4, label="test (in-distribution)")
    ax2.bar(x + 0.2, f1_ood, 0.4, label="shifted (higher Rf, farther, noisier)")
    ax2.set_xticks(x, CLASSES, rotation=45)
    ax2.set_ylim(0, 1.3)
    ax2.set_ylabel("F1")
    ax2.set_title("Per-class F1")
    ax2.legend(loc="upper center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=600)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_eval = max(args.n_per_class // 4, 50)
    x_tr, y_tr = make_dataset(args.n_per_class, seed=args.seed + 1)
    x_va, y_va = make_dataset(n_eval, seed=args.seed + 2)
    x_te, y_te = make_dataset(n_eval, seed=args.seed + 3)
    x_ood, y_ood = make_dataset(n_eval, seed=args.seed + 4, shifted=True)
    f_tr, f_va, f_te, f_ood = (extract(a) for a in (x_tr, x_va, x_te, x_ood))

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("grid-fault-baseline")
    with mlflow.start_run(run_name="hgb-window-features"):
        params = {
            "model": "HistGradientBoostingClassifier",
            "n_per_class": args.n_per_class,
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "n_features": f_tr.shape[1],
            "n_classes": len(CLASSES),
        }
        mlflow.log_params(params)

        clf = HistGradientBoostingClassifier(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            early_stopping=False,
            random_state=args.seed,
        )
        clf.fit(f_tr, y_tr)

        for name, f, y in (("val", f_va, y_va), ("test", f_te, y_te), ("ood", f_ood, y_ood)):
            pred = clf.predict(f)
            mlflow.log_metric(f"{name}_accuracy", accuracy_score(y, pred))
            mlflow.log_metric(f"{name}_macro_f1", f1_score(y, pred, average="macro"))
            print(f"{name:>4}: acc={accuracy_score(y, pred):.4f}  macro-F1={f1_score(y, pred, average='macro'):.4f}")

        pred_te, pred_ood = clf.predict(f_te), clf.predict(f_ood)
        labels = list(range(len(CLASSES)))
        cm = confusion_matrix(y_ood, pred_ood, labels=labels)
        f1_te = f1_score(y_te, pred_te, average=None, labels=labels)
        f1_oo = f1_score(y_ood, pred_ood, average=None, labels=labels)

        fig_path = ROOT / "reports" / "figures" / "results.png"
        plot_results(cm, f1_te, f1_oo, fig_path)
        mlflow.log_artifact(str(fig_path))
        # MLflow serialises sklearn models with skops, which flags HGB's internal tree type.
        # We trust exactly this one type because we produce and load our own artifact;
        # never load a model file from an untrusted source with this allow-list.
        mlflow.sklearn.log_model(
            clf,
            name="model",
            skops_trusted_types=[
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            ],
        )

        for c, a, b in zip(CLASSES, f1_te, f1_oo):
            print(f"  {c:>6}  test F1={a:.3f}  shifted F1={b:.3f}")


if __name__ == "__main__":
    main()
