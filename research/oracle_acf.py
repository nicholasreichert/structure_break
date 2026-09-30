"""Oracle AUC of PURE dependence change: post-segment acf (self-centred) vs history acf."""
import numpy as np
from scipy.special import ndtri
from sklearn.metrics import roc_auc_score
from common import load_train

LAGS = [1, 2, 3, 5, 10, 20, 50]


def acf(v, L):
    v = v - v.mean()
    return (v[L:] * v[:-L]).mean() / (v * v).mean()


def run():
    ds = load_train()
    rng = np.random.default_rng(0)
    u_emp = (ds.tau[ds.has_break] + 0.5) / ds.online_len[ds.has_break]
    R, lab = [], []
    for k in range(0, len(ds), 2):
        h, o = ds.segments(k)
        nh = len(h)
        t = int(ds.tau[k]) if ds.tau[k] >= 0 else int(rng.choice(u_emp) * len(o))
        n = len(o) - t
        if n < 100:
            continue
        sh = np.sort(h)
        def gs(x):
            u = ((np.searchsorted(sh, x, "left") + np.searchsorted(sh, x, "right")) / 2 + 0.5) / (nh + 1)
            return ndtri(u)
        gh, gp = gs(h), gs(o[t:])
        ah, ap = np.abs(gh), np.abs(gp)
        row = []
        for L in LAGS:
            row.append((acf(gp, L) - acf(gh, L)) * np.sqrt(n))
            row.append((acf(ap, L) - acf(ah, L)) * np.sqrt(n))
        row.append(sum(((acf(gp, L) - acf(gh, L)) ** 2) for L in range(1, 11)) * n)
        row.append(sum(((acf(ap, L) - acf(ah, L)) ** 2) for L in range(1, 11)) * n)
        # marginal only: two-sample KS-like on g (post vs uniform null)
        row.append(abs(gp.mean()) * np.sqrt(n)); row.append(abs((gp ** 2).mean() - 1) * np.sqrt(n / 2))
        R.append(row); lab.append(ds.tau[k] >= 0)
    R, lab = np.array(R), np.array(lab)
    print(f"{len(lab)} series with >=100 post points")
    for i, L in enumerate(LAGS):
        print(f"lag {L:>3}: acf(g) AUC {roc_auc_score(lab, np.abs(R[:, 2*i])):.3f}   acf(|g|) AUC {roc_auc_score(lab, np.abs(R[:, 2*i+1])):.3f}")
    print(f"sumsq acf(g) 1..10 {roc_auc_score(lab, R[:, -4]):.3f}  sumsq acf(|g|) 1..10 {roc_auc_score(lab, R[:, -3]):.3f}")
    print(f"mean {roc_auc_score(lab, R[:, -2]):.3f}  var {roc_auc_score(lab, R[:, -1]):.3f}")
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200), np.abs(R), lab, cv=5, method="predict_proba")[:, 1]
    print(f"GBDT on all oracle stats: {roc_auc_score(lab, p):.3f}")


if __name__ == "__main__":
    run()
