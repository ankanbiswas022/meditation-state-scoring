"""Extract per-trial features for every subject (parallel, cached to cache/<subject>.parquet).

    python -m medscore.extract [--workers 4]
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from eegscore.preprocess import PreprocessParams

from .loader import BANDS, features_for_subject, subjects

log = logging.getLogger("medscore.extract")

PP = PreprocessParams(l_freq=1.0, h_freq=80.0, notch=50.0, reference="average", resample_to=250.0,
                      amplitude_reject_uv=1e9, flat_threshold_uv=0.0)     # trials already QC'd
FX = {"bands": BANDS, "multitaper_bandwidth": 2.0, "aperiodic_fit_range": (2.0, 70.0),
      "connectivity_band": "alpha", "fmax": 80.0}


def _one(s):
    logging.basicConfig(level=logging.WARNING)
    F = features_for_subject(s, PP, FX)
    return s.name, len(F)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    subs = subjects()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_one, s): s.name for s in subs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                name, n = f.result()
                log.info("[%d/%d] %s: %d trials  (%.0f s elapsed)", i, len(subs), name, n, time.time() - t0)
            except Exception as e:  # noqa: BLE001
                log.error("%s failed: %s", futs[f], e)


if __name__ == "__main__":
    main()
