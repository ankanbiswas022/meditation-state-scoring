"""Load the BK1 meditation cohort (IISc) segmented EEG into feature tables.

Data are *not* part of this repository (participant EEG under IISc ethics approval).
Point ``MEDSCORE_DATA`` at the study root that contains ``data/segmentedData`` and the
``BK1AllSubjectList.mat`` / ``BK1ProjectDetails.mat`` information files.

Layout (per subject, per protocol)::

    data/segmentedData/<subj>/EEG/<date>/<protocol>/segmentedData/LFP/elec{1..64}.mat
        analogData: (n_trials, 2500) at 1000 Hz, trial = -1.25 .. +1.25 s (2.5 s)
    .../segmentedData/badTrials_wo_v8.mat  -> badTrials, badElecs (struct), eegElectrodeLabels

Protocols used: EO1 (eyes-open rest, 120 trials = 5 min), M1 (open-eyed meditation,
360 trials = 15 min), EO2, M2 (same, later in the session).

Each 2.5-s trial becomes one window. Bad trials are dropped; bad electrodes are set to
NaN so the feature table carries missing values for them (imputed inside the model
pipeline) rather than silently using noisy channels. Features are computed with the
``eegscore`` extractor from the public cognitive-scoring repo, so the two projects share
one feature contract.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from eegscore.features import FeatureExtractor
from eegscore.preprocess import PreprocessParams, preprocess_array

log = logging.getLogger(__name__)

ROOT = Path(os.environ.get("MEDSCORE_DATA", r"D:\Projects\ProjectDhyaan\BK1"))
INFO = ROOT / "ProjectDhyaanBK1Programs" / "commonAnalysisCodes" / "informationFiles"
CACHE = Path(__file__).resolve().parents[2] / "cache"
PROTOCOLS = ("EO1", "M1", "EO2", "M2")
DECLARED_BAD = {"004P", "081SN", "069MG", "092KB", "063VK"}

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 20),
         "slow_gamma": (20, 35), "fast_gamma": (36, 66)}


@dataclass
class Subject:
    name: str
    date: str
    group: str        # 'A' advanced meditator / 'C' control
    age: float
    gender: str


def subjects() -> list[Subject]:
    d = sio.loadmat(INFO / "BK1ProjectDetails.mat")["demographicDetails"]
    hdr = [str(x[0]) if x.size else "" for x in d[0]]
    col = {h: i for i, h in enumerate(hdr)}
    good = sio.loadmat(INFO / "BK1AllSubjectList.mat")
    keep = {str(x[0]) for x in good["allSubjectList"].ravel()} - DECLARED_BAD
    out = []
    for r in d[1:]:
        name = str(r[col["SubjectName"]][0])
        if name not in keep:
            continue
        out.append(Subject(name, str(r[col["ExpDate"]][0]), str(r[col["Label"]][0]),
                           float(r[col["Age"]].ravel()[0]), str(r[col["Gender"]][0])))
    return out


def _proto_dir(s: Subject, proto: str) -> Path:
    return ROOT / "data" / "segmentedData" / s.name / "EEG" / s.date / proto / "segmentedData"


def load_protocol(s: Subject, proto: str, n_elec: int = 64) -> tuple[np.ndarray, list[str], np.ndarray, float]:
    """Return (X (n_good_trials, n_elec, n_samples) Volts with bad electrodes NaN, labels,
    bad_elec_mask, sfreq)."""
    d = _proto_dir(s, proto)
    bt = sio.loadmat(d / "badTrials_wo_v8.mat")
    bad_trials = set(bt["badTrials"].ravel().astype(int) - 1)             # MATLAB 1-based
    be = bt["badElecs"]
    bad_elecs = set()
    for f in ("badImpedanceElecs", "noisyElecs", "flatPSDElecs", "declaredBadElectrodes"):
        if f in (be.dtype.names or ()):
            bad_elecs |= set(be[f][0, 0].ravel().astype(int) - 1)
    labels = [str(x[0]) for x in bt["eegElectrodeLabels"].ravel()] if "eegElectrodeLabels" in bt \
        else [f"E{i+1}" for i in range(n_elec)]
    tv = sio.loadmat(d / "LFP" / "lfpInfo.mat")["timeVals"].ravel()
    sfreq = float(round(1 / (tv[1] - tv[0])))
    first = sio.loadmat(d / "LFP" / "elec1.mat")["analogData"]
    good = np.array(sorted(set(range(first.shape[0])) - bad_trials))
    X = np.full((len(good), n_elec, first.shape[1]), np.nan, dtype=np.float32)
    for i in range(n_elec):
        if i in bad_elecs:
            continue
        a = first if i == 0 else sio.loadmat(d / "LFP" / f"elec{i+1}.mat")["analogData"]
        X[:, i, :] = a[good] * 1e-6                                       # uV -> V
    mask = np.zeros(n_elec, bool); mask[list(bad_elecs)] = True
    return X, labels[:n_elec], mask, sfreq


def features_for_subject(s: Subject, pp: PreprocessParams, fx_kwargs: dict,
                         protocols=PROTOCOLS) -> pd.DataFrame:
    """Per-trial feature table for one subject across protocols (cached to parquet)."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{s.name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    frames = []
    for proto in protocols:
        try:
            X, labels, bad, sfreq = load_protocol(s, proto)
        except FileNotFoundError as e:
            log.warning("%s %s missing (%s)", s.name, proto, e); continue
        # preprocess trial by trial with bad electrodes zero-filled (they stay NaN in features)
        Xp = []
        for tr in X:
            x = np.nan_to_num(tr, nan=0.0)
            Xp.append(preprocess_array(x, sfreq, pp))
        Xp = np.stack(Xp)
        fx = FeatureExtractor(sfreq=pp.resample_to or sfreq, ch_names=labels, **fx_kwargs)
        F = fx.transform(Xp)
        # mark features of bad electrodes as missing
        for i, ch in enumerate(labels):
            if bad[i]:
                F.loc[:, [c for c in F.columns if c.endswith(f"_{ch}")]] = np.nan
        F.insert(0, "trial", np.arange(len(F)))
        F.insert(0, "protocol", proto)
        F.insert(0, "subject", s.name)
        frames.append(F)
        log.info("%s %s: %d trials, %d bad electrodes, %d features", s.name, proto, len(F), bad.sum(), F.shape[1] - 3)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(f)
    return out
