"""Demo: feed the running service normal and shifted traffic and print what /drift says.

Run the service first, then:  PYTHONPATH=src python -m gridfault.demo_drift
Uses only the standard library plus the project's own simulator.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np

from .simulate import CLASSES, simulate_window


def _call(url: str, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _send_traffic(base: str, n_per_class: int, shifted: bool, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for label in CLASSES:
        for _ in range(n_per_class):
            window = simulate_window(label, rng, shifted=shifted)
            _call(f"{base}/predict", "POST", {"window": np.round(window, 5).tolist()})


def _summarise(title: str, report: dict) -> None:
    print(f"\n{title}")
    print(
        f"  status={report['status']}  drift_detected={report['drift_detected']}  "
        f"buffered={report['n_buffered']}"
    )
    for name, c in report["classes"].items():
        if c["evaluated"]:
            print(
                f"    {name:>6}: n={c['n']:<4} drifted_features={c['n_drifted_features']:<3} "
                f"max_psi={c['max_psi']}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n-per-class", type=int, default=120)
    args = ap.parse_args()
    base = args.url.rstrip("/")

    _call(f"{base}/drift/reset", "POST")
    _send_traffic(base, args.n_per_class, shifted=False, seed=101)
    _summarise("Phase 1: in-distribution traffic", _call(f"{base}/drift"))

    _call(f"{base}/drift/reset", "POST")
    _send_traffic(base, args.n_per_class, shifted=True, seed=202)
    _summarise(
        "Phase 2: shifted traffic (higher fault resistance, farther faults, more noise)",
        _call(f"{base}/drift"),
    )


if __name__ == "__main__":
    main()
