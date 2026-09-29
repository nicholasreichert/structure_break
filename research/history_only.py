"""Can the historical segment alone predict whether a series will break? Series-level CV AUC."""
import lightgbm as lgb
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from common import load_train


def acf(a, L):
    a = a - a.mean()
    return float(a[L:] @ a[:-L] / (a @ a))


def hist_feats(h):
    h = h.astype(np.float64)
    n = len(h)
    a = np.abs(h - h.mean())
    q = n // 4
    f = dict(
        log_n=np.log(n), mean=h.mean(), sd=h.std(), skew=stats.skew(h), kurt=stats.kurtosis(h),
        uniq=len(np.unique(h)) / n, mn=h.min(), mx=h.max(),
        mean_last_q=h[-q:].mean() - h.mean(), sd_last_q=h[-q:].std() / h.std(),
        mean_first_q=h[:q].mean() - h.mean(), sd_first_q=h[:q].std() / h.std(),
    )
    for L in (1, 2, 3, 5, 10):
        f[f"acf{L}"] = acf(h, L)
        f[f"acfabs{L}"] = acf(a, L)
    return f


def run():
    ds = load_train()
    F = [hist_feats(ds.segments(k)[0]) for k in range(len(ds))]
    names = list(F[0])
    X = np.array([[f[n] for n in names] for f in F])
    y = ds.has_break.astype(int)
    print(f"{len(y)} series, break rate {y.mean():.3f}")
    for i, n in enumerate(names):
        auc = roc_auc_score(y, X[:, i])
        print(f"  {n:>12}: AUC {max(auc, 1 - auc):.3f}   median {np.median(X[:, i]):.4g}")
    fold = np.random.default_rng(0).permutation(len(y)) % 5
    oof = np.zeros(len(y))
    for f in range(5):
        b = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50,
                           feature_fraction=0.8, verbose=-1, seed=0), lgb.Dataset(X[fold != f], y[fold != f]), 300)
        oof[fold == f] = b.predict(X[fold == f])
    print(f"history-only CV AUC: {roc_auc_score(y, oof):.4f}")


if __name__ == "__main__":
    run()
