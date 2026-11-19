"""Window-level features: phasors, symmetrical components, pre/post changes."""

from __future__ import annotations

import numpy as np

from .simulate import SPC

_A = np.exp(2j * np.pi / 3)

FEATURE_NAMES: list[str] = (
    [f"{c}_rms_last" for c in ("Va", "Vb", "Vc", "Ia", "Ib", "Ic")]
    + [f"{c}_rms_ratio" for c in ("Va", "Vb", "Vc", "Ia", "Ib", "Ic")]
    + [f"{c}_absmax_ratio" for c in ("Va", "Vb", "Vc", "Ia", "Ib", "Ic")]
    + [f"{c}_mean_last" for c in ("Ia", "Ib", "Ic")]
    + [f"{s}_{q}" for q in ("V", "I") for s in ("seq0", "seq1", "seq2")]
    + ["V_seq2_over_seq1", "I_seq2_over_seq1", "I_seq0_over_seq1", "V_seq0_over_seq1"]
)


def _phasor(x: np.ndarray) -> np.ndarray:
    """Fundamental phasor of the last cycle for each channel. x: (N, C, T)."""
    seg = x[..., -SPC:]
    k = np.arange(SPC)
    basis = np.exp(-2j * np.pi * k / SPC)
    return (seg * basis).sum(axis=-1) * (np.sqrt(2) / SPC)


def _rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt((x**2).mean(axis=-1))


def extract(x: np.ndarray) -> np.ndarray:
    """x: (N, 6, T) -> (N, F) feature matrix."""
    first = x[..., :SPC]
    last = x[..., -SPC:]
    eps = 1e-6

    rms_last = _rms(last)
    rms_ratio = rms_last / (_rms(first) + eps)
    amax_ratio = np.abs(last).max(axis=-1) / (np.abs(first).max(axis=-1) + eps)
    mean_last = last[:, 3:].mean(axis=-1)  # DC offset in currents

    ph = _phasor(x)
    seq = []
    for sl in (slice(0, 3), slice(3, 6)):
        a, b, c = ph[:, sl].T
        s0 = (a + b + c) / 3
        s1 = (a + _A * b + _A**2 * c) / 3
        s2 = (a + _A**2 * b + _A * c) / 3
        seq.append(np.stack([np.abs(s0), np.abs(s1), np.abs(s2)], axis=1))
    v_seq, i_seq = seq
    ratios = np.stack(
        [
            v_seq[:, 2] / (v_seq[:, 1] + eps),
            i_seq[:, 2] / (i_seq[:, 1] + eps),
            i_seq[:, 0] / (i_seq[:, 1] + eps),
            v_seq[:, 0] / (v_seq[:, 1] + eps),
        ],
        axis=1,
    )
    feats = np.concatenate([rms_last, rms_ratio, amax_ratio, mean_last, v_seq, i_seq, ratios], axis=1)
    assert feats.shape[1] == len(FEATURE_NAMES)
    return feats.astype(np.float32)
