import numpy as np
import pytest

from gridfault.drift import (
    DriftMonitor,
    build_reference,
    load_reference,
    psi,
    save_reference,
)
from gridfault.features import FEATURE_NAMES, extract
from gridfault.simulate import CLASSES, make_dataset

N_EVAL = 110  # per class, just above the monitor's minimum sample size


@pytest.fixture(scope="module")
def train():
    x, y = make_dataset(300, seed=1)
    return extract(x), y


@pytest.fixture(scope="module")
def in_dist():
    x, y = make_dataset(N_EVAL, seed=5)
    return extract(x), y


@pytest.fixture(scope="module")
def shifted():
    x, y = make_dataset(N_EVAL, seed=6, shifted=True)
    return extract(x), y


def _monitor(train, data, **kw) -> DriftMonitor:
    m = DriftMonitor(build_reference(*train), **kw)
    for row, k in zip(*data, strict=True):
        m.add(int(k), row)
    return m


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=2000), rng.normal(size=500)) < 0.05


def test_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=2000), rng.normal(loc=1.5, size=500)) > 0.5


def test_psi_of_unchanged_constant_feature_is_zero():
    assert psi(np.ones(100), np.ones(50)) == 0.0


def test_psi_flags_a_feature_that_was_constant_and_now_varies():
    assert psi(np.ones(100), np.random.default_rng(0).normal(size=50)) > 0.2


def test_too_little_data_is_not_evaluated(train, in_dist):
    m = DriftMonitor(build_reference(*train))
    for row, k in list(zip(*in_dist, strict=True))[:20]:
        m.add(int(k), row)
    r = m.report()
    assert r["status"] == "insufficient_data"
    assert r["drift_detected"] is False
    assert not any(c["evaluated"] for c in r["classes"].values())


def test_no_alarm_on_in_distribution_traffic(train, in_dist):
    r = _monitor(train, in_dist).report()
    assert r["status"] == "ok"
    assert all(r["classes"][c]["evaluated"] for c in CLASSES)
    assert r["drift_detected"] is False
    assert all(c["n_drifted_features"] == 0 for c in r["classes"].values())


def test_alarm_on_shifted_traffic(train, shifted):
    r = _monitor(train, shifted).report()
    assert r["drift_detected"] is True
    assert all(r["classes"][c]["drifted"] for c in CLASSES)
    top = r["classes"]["AG"]["top_drifted_features"]
    assert top and top[0]["feature"] in FEATURE_NAMES and top[0]["psi"] > 0.2


def test_a_single_drifted_class_is_enough_to_alarm(train, in_dist, shifted):
    m = _monitor(train, in_dist)
    ag = CLASSES.index("AG")
    m.reset()
    for row, k in zip(*in_dist, strict=True):
        m.add(int(k), row)
    # replace the AG stratum with shifted AG windows
    m._buffers[ag].clear()
    for row, k in zip(*shifted, strict=True):
        if k == ag:
            m.add(int(k), row)
    r = m.report()
    assert r["drift_detected"] is True
    assert r["classes"]["AG"]["drifted"] is True
    assert r["classes"]["BG"]["drifted"] is False


def test_reset_empties_the_buffer(train, shifted):
    m = _monitor(train, shifted)
    m.reset()
    assert m.report()["n_buffered"] == 0


def test_buffer_is_bounded(train, in_dist):
    m = _monitor(train, in_dist, max_buffer=50)
    assert all(c["n"] <= 50 for c in m.report()["classes"].values())


def test_reference_roundtrip(train, tmp_path):
    path = save_reference(tmp_path / "ref.npz", *train)
    ref = load_reference(path)
    assert sorted(ref) == list(range(len(CLASSES)))
    assert all(v.shape == (300, len(FEATURE_NAMES)) for v in ref.values())
