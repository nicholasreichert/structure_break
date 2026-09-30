"""Run main.infer end-to-end on the reduced test set (100 series) with a model directory.
Checks the production path loads and streams; the TS-AUC on 100 series is noisy.

usage (repo root): python research/smoke_infer.py [model_dir]
"""

import sys

import numpy as np

from common import load_test_reduced, main, ts_auc


def run(model_dir):
    ds = load_test_reduced()
    gen = main.infer(((ds.segments(k)[0], ds.segments(k)[1]) for k in range(len(ds))), model_dir)
    next(gen)  # ready
    s = np.array(list(gen))
    y = np.concatenate([ds.labels(k) for k in range(len(ds))])
    step = np.concatenate([np.arange(ds.online_len[k]) for k in range(len(ds))])
    assert len(s) == len(y) and np.isfinite(s).all()
    print(f"{model_dir}: {len(s)} predictions, reduced-test TS-AUC {ts_auc(y, s, step):.4f}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "resources")
