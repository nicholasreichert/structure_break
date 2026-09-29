"""Ceiling of our statistic family: CV LightGBM on oracle post-break stats (true tau), series level."""
import lightgbm as lgb
import numpy as np
from multiprocessing import Pool
from scipy import stats
from sklearn.metrics import roc_auc_score

from common import load_train, main
from oracle_streams import candidate_streams


def feats(args):
    h, o, t = args
    x = np.concatenate([h, o]).astype(np.float64)
    nh = len(h)
    S = candidate_streams(x, nh)
    st = main.FeatureState(h)
    E = np.array([st._raw(float(v)) for v in o])  # production streams, raw
    H = np.array([main.FeatureState(h)._raw(float(v)) for v in h[-1:]])  # dummy to keep API
    out = []
    for name, s in S.items():
        sh, post = s[1:nh], s[nh + t:]
        out.append((post.mean() - sh.mean()) / main._long_run_sd(sh) * np.sqrt(len(post)))
    e = (E[t:] - np.array(st.center)) * np.array(st.scale)
    out += list(e.sum(0) / np.sqrt(len(e)))
    post = o[t:].astype(np.float64)
    out += [stats.ks_2samp(h, post).statistic * np.sqrt(len(post)), stats.levene(h, post).statistic,
            np.log(len(post)), np.log(nh)]
    return out


def run():
    ds = load_train()
    rng = np.random.default_rng(0)
    u_emp = (ds.tau[ds.has_break] + 0.5) / ds.online_len[ds.has_break]
    jobs, lab = [], []
    for k in range(len(ds)):
        h, o = ds.segments(k)
        t = int(ds.tau[k]) if ds.tau[k] >= 0 else int(rng.choice(u_emp) * len(o))
        if len(o) - t < 20:
            continue
        jobs.append((h, o, t)); lab.append(int(ds.tau[k] >= 0))
    with Pool(20) as p:
        X = np.array(p.map(feats, jobs, chunksize=20))
    y = np.array(lab)
    import sys
    if "--abs" in sys.argv:
        X = np.abs(X)
        print("USING ABS FEATURES")
    print(X.shape, "series; single best |stat| AUC:",
          round(max(max(roc_auc_score(y, np.abs(X[:, j])), roc_auc_score(y, -np.abs(X[:, j]))) for j in range(X.shape[1])), 3))
    fold = rng.permutation(len(y)) % 5
    oof = np.zeros(len(y))
    for f in range(5):
        b = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50,
                           feature_fraction=0.8, verbose=-1, seed=0), lgb.Dataset(X[fold != f], y[fold != f]), 400)
        oof[fold == f] = b.predict(X[fold == f])
    print("combined oracle CV AUC:", round(roc_auc_score(y, oof), 4))
    names = list(candidate_streams(np.random.default_rng(0).normal(size=300), 200)) + [f"prod_{s}" for s in main.STREAMS] + ["ks", "levene", "log_post", "log_hist"]
    imp = b.feature_importance("gain")
    order = np.argsort(imp)[::-1]
    print("gain:", ", ".join(f"{names[j]}={imp[j] / imp.sum():.3f}" for j in order[:25]))
    big = X[:, -2] >= np.log(200)
    print("  with >=200 post points:", round(roc_auc_score(y[big], oof[big]), 4), f"(n={big.sum()})")


if __name__ == "__main__":
    run()
