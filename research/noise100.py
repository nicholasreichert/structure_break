"""Spread of TS-AUC on random 100-series subsets of a held-out fold (how noisy is the reduced test?)."""
import lightgbm as lgb
import numpy as np

from common import cached, folds, load_train, main, ts_auc


def run():
    tr = cached("train", load_train)
    X, y, series, step = tr["X"], tr["y"], tr["series"], tr["step"]
    fold_of = folds(series.max() + 1)[series]
    trn = np.flatnonzero((fold_of != 0) & (step % 2 == 0))
    val = np.flatnonzero(fold_of == 0)
    b = lgb.train(dict(main.LGB_PARAMS, num_threads=8), lgb.Dataset(X[trn], y[trn]), 600)
    p = b.predict(X[val])
    np.save("cache/oof_fold0.npy", p)
    vs = series[val]
    uniq = np.unique(vs)
    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(200):
        pick = np.isin(vs, rng.choice(uniq, 100, replace=False))
        aucs.append(ts_auc(y[val][pick], p[pick], step[val][pick]))
    aucs = np.array(aucs)
    print(f"full fold {ts_auc(y[val], p, step[val]):.4f}; 100-series subsets: mean {aucs.mean():.4f} "
          f"sd {aucs.std():.4f}, 5-95% [{np.quantile(aucs, .05):.4f}, {np.quantile(aucs, .95):.4f}]")


if __name__ == "__main__":
    run()
