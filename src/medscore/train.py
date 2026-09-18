"""Meditative-state scoring: evaluation suite.

    python -m medscore.train

Design
------
Per subject, EO1 (eyes-open rest, 5 min) is split by time: the first ``CALIB_FRAC`` of
good trials is the **calibration** segment; every feature of every later window is
z-scored against it. Scored segments: rest1 (rest of EO1), med1 (M1, 15 min open-eyed
meditation), rest2 (EO2, ~40 min later), med2 (M2). All splits are subject-wise.

The naive design (rest1 vs med1) is confounded by time-in-session: the two *rest* blocks
(rest1 vs rest2) are separable almost as well as rest vs meditation, i.e. the model learns
drift relative to the calibration segment (impedance, gel, arousal). The main analysis
therefore **brackets** meditation with rest recorded both before and after it
(rest1 + rest2 vs med1 + med2), so monotonic drift cannot be used as a label proxy.

Evaluations
-----------
main               bracketed rest vs meditation, baseline-relative features; per-group AUC
by_group           the same model fitted within meditators only / controls only
ablation_absolute  bracketed design, features without per-person calibration
ablation_invariant bracketed design, amplitude-invariant features only (no log-power,
                   aperiodic offset, Hjorth)
naive              rest1 vs med1 (the number you would report without the control)
null_*             rest1 vs rest2, med1 vs med2 (drift detectors; should be ~0.5 for a
                   state-only signal)
trait_check        per-subject median calibrated meditation score, meditators vs controls
shap               importance by band and by feature family for the main model
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
from eegscore.explain import shap_summary
from eegscore.features import BaselineStats
from eegscore.models import build_classical
from eegscore.scoring import Scorer
from eegscore.validation import subject_bootstrap_auc, subject_cv
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from .loader import CACHE, subjects

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports"
log = logging.getLogger("medscore")

CFG = {"models": {"classical": {"type": "lightgbm", "n_estimators": 200, "learning_rate": 0.03,
                                "num_leaves": 7, "min_child_samples": 20}}}
CALIB_FRAC = 0.4
N_SPLITS = 6
META = ["subject", "protocol", "trial"]
BRACKETED = {"rest1": 0, "rest2": 0, "med1": 1, "med2": 1}
_BAND_RE = re.compile(r"^(delta|theta|alpha|beta|slow_gamma|fast_gamma)_")
_AMP_RE = re.compile(r"logpow|aperiodic_offset|hjorth")


def load_all() -> tuple[pd.DataFrame, dict[str, str]]:
    subs = {s.name: s for s in subjects()}
    frames = [pd.read_parquet(f) for f in sorted(CACHE.glob("*.parquet")) if f.stem in subs]
    D = pd.concat(frames, ignore_index=True)
    return D, {n: s.group for n, s in subs.items()}


def split_and_normalise(D: pd.DataFrame) -> pd.DataFrame:
    """Add columns: segment (calib / rest1 / med1 / rest2 / med2), z-scored features, and
    the raw features under an ``abs__`` prefix for the ablation."""
    feats = [c for c in D.columns if c not in META]
    out = []
    for s, d in D.groupby("subject"):
        d = d.sort_values(["protocol", "trial"]).copy()
        eo1 = d[d.protocol == "EO1"]
        n_cal = round(CALIB_FRAC * len(eo1))
        seg = pd.Series("drop", index=d.index)
        seg[eo1.index[:n_cal]] = "calib"
        seg[eo1.index[n_cal:]] = "rest1"
        seg[d.index[d.protocol == "M1"]] = "med1"
        seg[d.index[d.protocol == "EO2"]] = "rest2"
        seg[d.index[d.protocol == "M2"]] = "med2"
        d["segment"] = seg
        if n_cal < 10 or (seg == "rest1").sum() < 10 or (seg == "med1").sum() < 20:
            log.warning("%s: too few trials, skipped", s)
            continue
        Z = BaselineStats.fit(d.loc[seg == "calib", feats]).transform(d[feats])
        Z.columns = feats
        out.append(pd.concat([d[META + ["segment"]], Z, d[feats].add_prefix("abs__")], axis=1))
    return pd.concat(out, ignore_index=True)


def _band_of(name: str) -> str:
    m = _BAND_RE.match(name)
    return m.group(1) if m else "other"


def _xy(D, segments: dict[str, int], absolute=False, invariant=False, group=None):
    m = D.segment.isin(segments)
    if group is not None:
        m &= D.grp == group
    if absolute:
        cols = [c for c in D.columns if c.startswith("abs__")]
    else:
        cols = [c for c in D.columns if c not in META + ["segment", "gid", "grp"] and not c.startswith("abs__")]
        if invariant:
            cols = [c for c in cols if not _AMP_RE.search(c)]
    X = D.loc[m, cols].reset_index(drop=True)
    if absolute:
        X.columns = [c[5:] for c in X.columns]
    y = D.loc[m, "segment"].map(segments).to_numpy()
    return X, y, D.loc[m, "gid"].to_numpy(), D.loc[m, "grp"].to_numpy()


def fp(Xtr, ytr, Xte):
    return build_classical(CFG, 7).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def block(name, X, y, g):
    res = subject_cv(fp, X, y, g, N_SPLITS)
    auc, lo, hi = subject_bootstrap_auc(res)
    m = {**res.metrics(), "auc_ci95": [lo, hi], **res.session_level_metrics()}
    log.info("%-32s OOF AUC %.3f [%.3f, %.3f]  bAcc %.3f  recording AUC %.3f  (%d windows, %d subjects)",
             name, auc, lo, hi, m["balanced_accuracy"], m["recording_auc"], m["n_windows"], m["n_subjects"])
    return res, m


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("eegscore.validation").setLevel(logging.WARNING)
    t0 = time.time()
    REP.mkdir(exist_ok=True)
    D, groups_of = load_all()
    D = split_and_normalise(D)
    D["gid"] = D.subject.map({s: i for i, s in enumerate(sorted(D.subject.unique()))})
    D["grp"] = D.subject.map(groups_of)
    n_med, n_ctl = (D.groupby("subject").grp.first() == "A").sum(), (D.groupby("subject").grp.first() == "C").sum()
    log.info("%d subjects (%d meditators, %d controls), %d trials", D.gid.nunique(), n_med, n_ctl, len(D))
    metrics = {"n_subjects": int(D.gid.nunique()), "n_meditators": int(n_med), "n_controls": int(n_ctl),
               "n_trials": len(D), "calib_frac": CALIB_FRAC, "design": "rest1+rest2 vs med1+med2"}

    # main: bracketed --------------------------------------------------------------
    X, y, g, grp = _xy(D, BRACKETED)
    res, metrics["main"] = block("main: bracketed rest vs meditation", X, y, g)
    metrics["main"]["auc_meditators"] = float(roc_auc_score(y[grp == "A"], res.oof_prob[grp == "A"]))
    metrics["main"]["auc_controls"] = float(roc_auc_score(y[grp == "C"], res.oof_prob[grp == "C"]))
    log.info("   pooled model: meditators AUC %.3f, controls AUC %.3f", metrics["main"]["auc_meditators"], metrics["main"]["auc_controls"])
    for k, name in (("A", "meditators"), ("C", "controls")):
        Xg, yg, gg, _ = _xy(D, BRACKETED, group=k)
        _, metrics[f"by_group_{name}"] = block(f"within {name} only", Xg, yg, gg)

    # ablations ---------------------------------------------------------------------
    Xa, ya, ga, _ = _xy(D, BRACKETED, absolute=True)
    _, metrics["ablation_absolute"] = block("ablation: no calibration", Xa, ya, ga)
    Xi, yi, gi, _ = _xy(D, BRACKETED, invariant=True)
    _, metrics["ablation_invariant"] = block("ablation: amplitude-invariant", Xi, yi, gi)

    # naive design and drift nulls ----------------------------------------------------
    Xn, yn, gn, _ = _xy(D, {"rest1": 0, "med1": 1})
    _, metrics["naive_rest1_vs_med1"] = block("naive: rest1 vs med1", Xn, yn, gn)
    Xr, yr, gr, _ = _xy(D, {"rest1": 0, "rest2": 1})
    _, metrics["null_rest1_vs_rest2"] = block("null: rest1 vs rest2", Xr, yr, gr)
    Xm, ym, gm, _ = _xy(D, {"med1": 0, "med2": 1})
    _, metrics["null_med1_vs_med2"] = block("null: med1 vs med2", Xm, ym, gm)

    # trait check from the main model --------------------------------------------------
    scorer = Scorer().fit(res.oof_prob, y)
    sub_names = D.loc[D.segment.isin(BRACKETED), "subject"].to_numpy()
    sc = pd.DataFrame({"subject": sub_names, "y": y, "score": scorer.window_scores(res.oof_prob)})
    per = sc[sc.y == 1].groupby("subject").score.median().rename("med_score").to_frame()
    per["rest_score"] = sc[sc.y == 0].groupby("subject").score.median()
    per["delta"] = per.med_score - per.rest_score
    per["group"] = per.index.map(groups_of)
    a, c = per.loc[per.group == "A"], per.loc[per.group == "C"]
    metrics["trait_check"] = {
        "meditators_median_med_score": float(a.med_score.median()), "controls_median_med_score": float(c.med_score.median()),
        "meditators_median_delta": float(a.delta.median()), "controls_median_delta": float(c.delta.median()),
        "mannwhitney_p_med_score": float(mannwhitneyu(a.med_score, c.med_score).pvalue),
        "mannwhitney_p_delta": float(mannwhitneyu(a.delta, c.delta).pvalue),
        "auc_group_from_delta": float(roc_auc_score((per.group == "A").astype(int), per.delta)),
        "n_meditators": len(a), "n_controls": len(c)}
    log.info("trait: med-rest delta meditators %.1f vs controls %.1f (p=%.3g)", a.delta.median(), c.delta.median(),
             metrics["trait_check"]["mannwhitney_p_delta"])
    per.to_csv(REP / "per_subject_scores.csv")

    # explainability --------------------------------------------------------------------
    model = build_classical(CFG, 7).fit(X, y)
    try:
        _, per_feat, fam = shap_summary(model, X, max_rows=4000)
        metrics["shap_by_family"] = fam.round(4).to_dict()
        metrics["shap_by_band"] = per_feat.groupby(per_feat.index.map(_band_of)).sum().sort_values(ascending=False).round(4).to_dict()
        metrics["shap_top15"] = per_feat.head(15).round(4).to_dict()
        per_feat.head(60).to_csv(REP / "shap_top_features.csv", header=["mean_abs_shap"])
        log.info("SHAP by band: %s", metrics["shap_by_band"])
    except Exception as e:  # noqa: BLE001
        log.warning("SHAP skipped: %s", e)

    metrics["seconds"] = round(time.time() - t0, 1)
    (REP / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    pd.DataFrame({"subject": sub_names, "y": y, "oof_prob": res.oof_prob}).to_csv(REP / "oof_predictions.csv", index=False)
    print(json.dumps({k: v for k, v in metrics.items() if k != "shap_top15"}, indent=2, default=float))


if __name__ == "__main__":
    main()
