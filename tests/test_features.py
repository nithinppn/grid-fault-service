import numpy as np

from gridfault.features import FEATURE_NAMES, extract
from gridfault.simulate import FS, SPC, WINDOW


def _balanced_window(amp: float = 1.0) -> np.ndarray:
    """Perfectly balanced abc-sequence 50 Hz voltages and currents, no noise."""
    t = np.arange(WINDOW) / FS
    ph = np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3])
    sig = np.stack([amp * np.sin(2 * np.pi * 50 * t + p) for p in ph])
    return np.concatenate([sig, 0.5 * sig], axis=0)[None].astype(np.float32)


def test_feature_count_matches_names():
    f = extract(_balanced_window())
    assert f.shape == (1, len(FEATURE_NAMES))
    assert np.isfinite(f).all()


def test_balanced_signal_is_pure_positive_sequence():
    f = extract(_balanced_window())[0]
    get = lambda n: f[FEATURE_NAMES.index(n)]  # noqa: E731
    assert get("seq0_V") < 1e-3
    assert get("seq2_V") < 1e-3
    assert get("seq0_I") < 1e-3
    assert get("seq2_I") < 1e-3
    # phasors are RMS-scaled, so a unit-amplitude sine has magnitude 1/sqrt(2)
    assert abs(get("seq1_V") - 1 / np.sqrt(2)) < 0.01
    assert abs(get("seq1_I") - 0.5 / np.sqrt(2)) < 0.01


def test_unchanged_signal_has_rms_ratio_one():
    f = extract(_balanced_window())[0]
    for name in ("Va_rms_ratio", "Ia_rms_ratio"):
        assert abs(f[FEATURE_NAMES.index(name)] - 1.0) < 0.02


def test_doubling_current_in_last_cycle_doubles_rms_ratio():
    x = _balanced_window()
    x[0, 3, -SPC:] *= 2.0  # phase A current, last cycle only
    f = extract(x)[0]
    assert abs(f[FEATURE_NAMES.index("Ia_rms_ratio")] - 2.0) < 0.05
    assert abs(f[FEATURE_NAMES.index("Ib_rms_ratio")] - 1.0) < 0.05
