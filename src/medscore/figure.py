"""Results figure from reports/: per-subject scores by group, AUC of every evaluation, SHAP by band."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = Path(__file__).resolve().parents[2] / "reports"


def main():
    m = json.loads((REP / "metrics.json").read_text())
    per = pd.read_csv(REP / "per_subject_scores.csv", index_col=0)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    rng = np.random.default_rng(0)

    # 1. per-subject median score, rest vs meditation, coloured by group
    for grp, col, lab, dx in (("C", "#4477aa", "controls", -0.08), ("A", "#ee6677", "meditators", 0.08)):
        d = per[per.group == grp]
        j = 0.02 * rng.standard_normal(len(d))
        ax[0].scatter(dx + j, d.rest_score, s=14, color=col, alpha=.7)
        ax[0].scatter(1 + dx + j, d.med_score, s=14, color=col, alpha=.7, label=lab)
        for (_, r), jj in zip(d.iterrows(), j):
            ax[0].plot([dx + jj, 1 + dx + jj], [r.rest_score, r.med_score], color=col, alpha=.15, lw=.8)
    ax[0].set(xticks=[0, 1], xticklabels=["rest (EO1 late + EO2)", "meditation (M1 + M2)"],
              ylabel="meditative-state score (0-100)", title="Per-subject median score, out-of-fold")
    ax[0].legend(frameon=False, loc="upper left")

    # 2. AUC of every evaluation
    rows = [("bracketed\nrest vs med", "main", "#1f3a5f"),
            ("meditators\nonly", "by_group_meditators", "#ee6677"),
            ("controls\nonly", "by_group_controls", "#4477aa"),
            ("no per-person\ncalibration", "ablation_absolute", "#8fb3e0"),
            ("amplitude-\ninvariant feats", "ablation_invariant", "#8fb3e0"),
            ("naive\nrest1 vs med1", "naive_rest1_vs_med1", "#bbbbbb"),
            ("null\nrest1 vs rest2", "null_rest1_vs_rest2", "#bbbbbb"),
            ("null\nmed1 vs med2", "null_med1_vs_med2", "#bbbbbb")]
    vals = [m[k]["auc"] for _, k, _ in rows]
    lo = [m[k]["auc"] - m[k]["auc_ci95"][0] for _, k, _ in rows]
    hi = [m[k]["auc_ci95"][1] - m[k]["auc"] for _, k, _ in rows]
    ax[1].bar(range(len(rows)), vals, yerr=[lo, hi], color=[c for _, _, c in rows], capsize=3)
    ax[1].axhline(0.5, color="k", ls=":", lw=1)
    ax[1].set(xticks=range(len(rows)), ylim=(0.4, 1.0), ylabel="subject-wise OOF AUC (window level)",
              title="Evaluations, ablations, and drift nulls")
    ax[1].set_xticklabels([r[0] for r in rows], fontsize=6.5, rotation=35, ha="right")

    # 3. SHAP by band
    band = pd.Series(m.get("shap_by_band", {})).sort_values()
    ax[2].barh(band.index, band.values, color="#1f3a5f")
    ax[2].set(xlabel="sum of mean |SHAP| over features", title="What the main model uses, by band")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REP / "overview.png", dpi=150)
    print("saved", REP / "overview.png")


if __name__ == "__main__":
    main()
