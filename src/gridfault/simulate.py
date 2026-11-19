"""Synthetic three-phase fault window generator.

Deliberately simplified single-source model (NOT the thesis EMT dataset):
a relay measures Va, Vb, Vc, Ia, Ib, Ic (per unit) over a short window. A fault
starts at a random sample; its severity depends on fault resistance and
distance. Normal windows include load steps and harmonics as hard negatives.

Limitations: no transformers, CT saturation, or network topology effects.
The point is a reproducible dataset for an end-to-end ML service, not a
physically complete power-system model.
"""

from __future__ import annotations

import numpy as np

FS = 4000  # sampling rate, Hz
F0 = 50.0  # nominal frequency, Hz
SPC = int(FS / F0)  # samples per cycle (80)
N_CYCLES = 3
WINDOW = SPC * N_CYCLES  # 240 samples = 60 ms
CHANNELS = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]

# class name -> faulted phase indices (0=A, 1=B, 2=C), involves ground?
FAULT_TYPES: dict[str, tuple[tuple[int, ...], bool]] = {
    "NORMAL": ((), False),
    "AG": ((0,), True),
    "BG": ((1,), True),
    "CG": ((2,), True),
    "AB": ((0, 1), False),
    "BC": ((1, 2), False),
    "CA": ((2, 0), False),
    "ABG": ((0, 1), True),
    "BCG": ((1, 2), True),
    "CAG": ((2, 0), True),
    "ABC": ((0, 1, 2), False),
}
CLASSES = list(FAULT_TYPES)

_ZS = 0.15  # source impedance (pu)
_ZL = 0.30  # line impedance to remote end (pu)
_FAULT_ANGLE = np.deg2rad(80.0)  # inductive fault loop


def _rng_params(rng: np.random.Generator, shifted: bool) -> dict:
    """Draw scenario parameters. `shifted=True` moves them out of the training range."""
    if shifted:
        rf = np.exp(rng.uniform(np.log(0.3), np.log(1.5)))  # higher resistance
        dist = rng.uniform(0.6, 1.0)  # farther faults
        noise = rng.uniform(0.01, 0.025)
    else:
        rf = np.exp(rng.uniform(np.log(0.005), np.log(0.4)))
        dist = rng.uniform(0.05, 0.7)
        noise = rng.uniform(0.002, 0.01)
    return {
        "rf": rf,
        "dist": dist,
        "noise": noise,
        "v_amp": rng.uniform(0.95, 1.05),
        "phi": rng.uniform(0, 2 * np.pi),
        "i_load": rng.uniform(0.15, 0.6),
        "pf_angle": np.deg2rad(rng.uniform(8, 35)),
        "t0": int(rng.integers(SPC // 2, WINDOW - SPC - 1)),
        "tau": rng.uniform(0.015, 0.06),
        "f": F0 + rng.normal(0, 0.1),
    }


def simulate_window(label: str, rng: np.random.Generator, shifted: bool = False) -> np.ndarray:
    """Return an array of shape (6, WINDOW): Va, Vb, Vc, Ia, Ib, Ic."""
    p = _rng_params(rng, shifted)
    t = np.arange(WINDOW) / FS
    w = 2 * np.pi * p["f"]
    ph = p["phi"] + np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3])  # abc sequence

    # pre-fault steady state
    v = np.stack([p["v_amp"] * np.sin(w * t + ph[k]) for k in range(3)])
    i = np.stack([p["i_load"] * np.sin(w * t + ph[k] - p["pf_angle"]) for k in range(3)])

    phases, has_ground = FAULT_TYPES[label]
    after = (np.arange(WINDOW) >= p["t0"]).astype(float)

    if phases:
        z_loop = _ZL * p["dist"]
        # remaining voltage fraction and fault current magnitude at the relay
        v_frac = (z_loop + p["rf"]) / (_ZS + z_loop + p["rf"])
        i_fault = p["v_amp"] / (_ZS + z_loop + p["rf"])
        if len(phases) == 2 and not has_ground:
            i_fault *= 0.87  # phase-phase fault current
        tt = np.clip(t - p["t0"] / FS, 0, None)
        dc = np.exp(-tt / p["tau"])

        for n, k in enumerate(phases):
            v_k = p["v_amp"] * np.sin(w * t + ph[k])
            sag = 1.0 - (1.0 - v_frac) * after
            if len(phases) == 2 and not has_ground:
                sag = 1.0 - (1.0 - 0.5 - 0.5 * v_frac) * after  # LL: partial collapse
            v[k] = v_k * sag
            # for phase-phase faults the two currents are in opposition
            sign = -1.0 if (len(phases) == 2 and not has_ground and n == 1) else 1.0
            # phase-phase faults circulate one current: same reference, opposite sign
            ref = ph[phases[0]] if (len(phases) == 2 and not has_ground) else ph[k]
            ang = ref - _FAULT_ANGLE
            i_post = sign * i_fault * np.sin(w * t + ang)
            i_post -= sign * i_fault * np.sin(w * p["t0"] / FS + ang) * dc  # DC offset
            i[k] = i[k] * (1 - after) + (i[k] * 0.3 + i_post) * after

        if has_ground and len(phases) < 3:
            for k in set(range(3)) - set(phases):  # healthy phase swell on ground faults
                v[k] = v[k] * (1.0 + 0.08 * (1 - v_frac) * after)

    elif rng.random() < 0.5:
        # hard negative: load step (current rises, voltage barely moves)
        step = rng.uniform(1.5, 3.5)
        i = i * (1 + (step - 1) * after)
        v = v * (1 - 0.03 * (step - 1) / 2.5 * after)

    # harmonics and noise on every window
    h3, h5 = rng.uniform(0, 0.02), rng.uniform(0, 0.015)
    for k in range(3):
        v[k] += h3 * np.sin(3 * w * t + ph[k]) + h5 * np.sin(5 * w * t + ph[k])
        i[k] += 0.5 * h5 * np.sin(5 * w * t + ph[k])

    x = np.concatenate([v, i], axis=0)
    x += rng.normal(0, p["noise"], size=x.shape)
    return x.astype(np.float32)


def make_dataset(
    n_per_class: int, seed: int, shifted: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced dataset: X shape (N, 6, WINDOW), y integer labels indexing CLASSES."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for idx, label in enumerate(CLASSES):
        for _ in range(n_per_class):
            xs.append(simulate_window(label, rng, shifted))
            ys.append(idx)
    x, y = np.stack(xs), np.array(ys, dtype=np.int64)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]
