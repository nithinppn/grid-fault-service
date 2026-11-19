import numpy as np

from gridfault.features import FEATURE_NAMES, extract
from gridfault.simulate import CLASSES, WINDOW, make_dataset, simulate_window


def test_dataset_shape_and_balance():
    x, y = make_dataset(5, seed=0)
    assert x.shape == (5 * len(CLASSES), 6, WINDOW)
    assert np.isfinite(x).all()
    assert set(np.bincount(y)) == {5}


def test_same_seed_is_deterministic():
    a, ya = make_dataset(3, seed=42)
    b, yb = make_dataset(3, seed=42)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ya, yb)


def test_different_seed_differs():
    a, _ = make_dataset(3, seed=1)
    b, _ = make_dataset(3, seed=2)
    assert not np.array_equal(a, b)


def _mean_feature(label: str, name: str, n: int = 40) -> float:
    rng = np.random.default_rng(7)
    x = np.stack([simulate_window(label, rng) for _ in range(n)])
    return float(extract(x)[:, FEATURE_NAMES.index(name)].mean())


def test_phase_phase_fault_has_no_zero_sequence_current():
    # LL faults circulate between two phases: I0 ~ 0 and I2 ~ I1
    assert _mean_feature("AB", "I_seq0_over_seq1") < 0.15
    assert _mean_feature("AB", "I_seq2_over_seq1") > 0.8


def test_single_line_ground_fault_has_zero_sequence_current():
    assert _mean_feature("AG", "I_seq0_over_seq1") > 0.5


def test_three_phase_fault_is_balanced():
    assert _mean_feature("ABC", "I_seq2_over_seq1") < 0.15
    assert _mean_feature("ABC", "I_seq0_over_seq1") < 0.15


def test_fault_raises_current_and_sags_voltage_on_faulted_phase():
    assert _mean_feature("AG", "Ia_rms_ratio") > 2.0
    assert _mean_feature("AG", "Va_rms_ratio") < 0.95
    assert _mean_feature("AG", "Ib_rms_ratio") < 1.2  # healthy phase unchanged


def test_shifted_set_is_harder_to_distinguish_from_normal():
    # shifted faults are higher-resistance and farther, so their voltage sag is milder
    rng = np.random.default_rng(3)
    base = np.stack([simulate_window("ABC", rng) for _ in range(40)])
    rng = np.random.default_rng(3)
    shifted = np.stack([simulate_window("ABC", rng, shifted=True) for _ in range(40)])
    i = FEATURE_NAMES.index("Va_rms_ratio")
    assert extract(shifted)[:, i].mean() > extract(base)[:, i].mean()
