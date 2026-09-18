import numpy as np
import pandas as pd

from medscore.train import BRACKETED, _band_of, _xy, split_and_normalise


def _frame(subject, proto, n, offset):
    rng = np.random.default_rng(hash((subject, proto)) % 2**32)
    return pd.DataFrame({"subject": subject, "protocol": proto, "trial": np.arange(n),
                         "alpha_logpow_O1": offset + rng.standard_normal(n),
                         "slow_gamma_relpow_Fz": offset + rng.standard_normal(n)})


def _cohort():
    D = pd.concat([_frame("S1", "EO1", 100, 5.0), _frame("S1", "M1", 300, 5.0),
                   _frame("S1", "EO2", 100, 5.0), _frame("S1", "M2", 300, 5.0),
                   _frame("S2", "EO1", 100, 50.0), _frame("S2", "M1", 300, 50.0)], ignore_index=True)
    Z = split_and_normalise(D)
    Z["gid"] = Z.subject.map({"S1": 0, "S2": 1})
    Z["grp"] = Z.subject.map({"S1": "A", "S2": "C"})
    return Z


def test_split_uses_early_eo1_as_calibration_and_zscores():
    Z = _cohort()
    assert set(Z.segment) >= {"calib", "rest1", "med1"}
    for s in ("S1", "S2"):
        cal = Z[(Z.subject == s) & (Z.segment == "calib")]
        assert len(cal) == 40                                   # 40 % of 100 EO1 trials
        assert abs(cal.alpha_logpow_O1.mean()) < 1e-9          # centred on own calibration
        assert cal.trial.max() < Z[(Z.subject == s) & (Z.segment == "rest1")].trial.min()
    # subject offset removed in z-space, retained in abs__ columns
    z1, z2 = (Z[(Z.subject == s) & (Z.segment == "med1")].alpha_logpow_O1.mean() for s in ("S1", "S2"))
    assert abs(z1 - z2) < 1.0
    a1, a2 = (Z[(Z.subject == s) & (Z.segment == "med1")]["abs__alpha_logpow_O1"].mean() for s in ("S1", "S2"))
    assert abs(a1 - a2) > 40


def test_xy_selects_segments_labels_and_feature_sets():
    Z = _cohort()
    X, y, g, grp = _xy(Z, BRACKETED)
    assert len(y) == (60 + 300 + 100 + 300) + (60 + 300)
    assert y.sum() == 900 and set(g) == {0, 1} and set(grp) == {"A", "C"}
    assert not any(c.startswith("abs__") for c in X.columns)
    Xa, *_ = _xy(Z, BRACKETED, absolute=True)
    assert list(Xa.columns) == list(X.columns)
    Xi, *_ = _xy(Z, BRACKETED, invariant=True)
    assert list(Xi.columns) == ["slow_gamma_relpow_Fz"]
    Xg, yg, *_ = _xy(Z, BRACKETED, group="C")
    assert len(yg) == 360


def test_band_of():
    assert _band_of("slow_gamma_relpow_Fz") == "slow_gamma"
    assert _band_of("plv_alpha_global") == "other"
