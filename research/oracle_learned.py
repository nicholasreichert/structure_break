"""Oracle screen for the learned (nonlinear) predictor: does its error see breaks the AR misses?

With the TRUE tau, per series: standardised post-break mean of each stream (null: random cut).
Reports single-stream AUCs, then the oracle_combo ceiling (CV LightGBM over all series-level
stats) with and without the learned streams.

usage: python research/oracle_learned.py [--p 10] [--D 100] [--bw 3.0] [--every 1] [--no-combo]
"""
import argparse
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from common import load_train, main
from learned import learned_streams
from oracle_combo import feats as combo_feats

ARGS = None


def one(job):
    h, o, t = job
    x = np.concatenate([h, o]).astype(np.float64)
    nh = len(h)
    S, info = learned_streams(x, nh, p=ARGS.p, D=ARGS.D, bandwidth=ARGS.bw)
    out = []
    for s in S.values():
        sh, post = s[ARGS.p : nh], s[nh + t :]
        out.append((post.mean() - sh.mean()) / main._long_run_sd(sh) * np.sqrt(len(post)))
    hist_gain = S["nl_gain"][ARGS.p : nh].mean() / S["ar_e2"][ARGS.p : nh].mean()
    base = combo_feats(job) if ARGS.combo else []
    return out, base, [hist_gain, info["lam"]], list(S)


def init(a):
    global ARGS
    ARGS = a


def cv_auc(X, y, seed=0):
    fold = np.random.default_rng(seed).permutation(len(y)) % 5
    oof = np.zeros(len(y))
    for f in range(5):
        b = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50,
                           feature_fraction=0.8, verbose=-1, seed=0), lgb.Dataset(X[fold != f], y[fold != f]), 400)
        oof[fold == f] = b.predict(X[fold == f])
    return roc_auc_score(y, oof)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--D", type=int, default=100)
    ap.add_argument("--bw", type=float, default=3.0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--no-combo", dest="combo", action="store_false")
    a = ap.parse_args()
    init(a)

    ds = load_train()
    rng = np.random.default_rng(0)
    u_emp = (ds.tau[ds.has_break] + 0.5) / ds.online_len[ds.has_break]
    jobs, lab = [], []
    for k in range(len(ds)):
        h, o = ds.segments(k)
        t = int(ds.tau[k]) if ds.tau[k] >= 0 else int(rng.choice(u_emp) * len(o))
        if len(o) - t < 20 or k % a.every:
            continue
        jobs.append((h, o, t))
        lab.append(int(ds.tau[k] >= 0))
    t0 = time.time()
    with Pool(20, initializer=init, initargs=(a,)) as pool:
        res = pool.map(one, jobs, chunksize=10)
    print(f"{len(jobs)} series in {time.time() - t0:.0f}s  (p={a.p} D={a.D} bw={a.bw})")
    y = np.array(lab)
    N = np.array([r[0] for r in res])
    info = np.array([r[2] for r in res])
    names = res[0][3]
    print(f"history: NL beats AR by median {np.median(info[:, 0]):.4f} of AR mse "
          f"(q90 {np.quantile(info[:, 0], 0.9):.4f}); median lambda/n {np.median(info[:, 1]):.3g}")
    for i, n_ in enumerate(names):
        print(f"  {n_:>9}  |stat| AUC {roc_auc_score(y, np.abs(N[:, i])):.3f}   signed AUC {roc_auc_score(y, N[:, i]):.3f}")
    if a.combo:
        B = np.array([r[1] for r in res])
        print(f"combo ceiling, existing stats: {cv_auc(B, y):.4f}")
        print(f"combo ceiling, + learned     : {cv_auc(np.hstack([B, N]), y):.4f}")


if __name__ == "__main__":
    run()
