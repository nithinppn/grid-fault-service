"""Write example request bodies for the API into examples/.

Run:  PYTHONPATH=src python -m gridfault.make_example
"""

from __future__ import annotations

import json

import numpy as np

from .model import ROOT
from .simulate import simulate_window

EXAMPLES = {"normal": "NORMAL", "ag_fault": "AG", "abc_fault": "ABC"}


def main() -> None:
    out = ROOT / "examples"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(123)
    for fname, label in EXAMPLES.items():
        window = simulate_window(label, rng)
        body = {"window": np.round(window, 5).tolist()}
        (out / f"{fname}.json").write_text(json.dumps(body))
        print(f"wrote examples/{fname}.json ({label})")


if __name__ == "__main__":
    main()
