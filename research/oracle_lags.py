"""Oracle (true tau) AUC of post-break lag-product statistics at many lags, on normal scores."""
import numpy as np
from scipy.special import ndtri
from sklearn.metrics import roc_auc_score
from common import load_train

LAGS = list(range(1, 11)) + [12, 15, 20, 25, 30, 40, 50]


def run():
    ds = load_train()
    rng = np.random.default_rng(0)
    u_emp = (ds.tau[ds.has_break] + 0.5) / ds.online_len[ds.has_break]
    R, lab = [], []
    for k in range(0, len(ds), 2):
        h, o = ds.segments(k)
        nh = len(h)
        t = int(ds.tau[k]) if ds.tau[k] >= 0 else int(rng.choice(u_emp) * len(o))
        if len(o) - t < 100:
            continue
        x = np.concatenate([h, o]).astype(np.float64)
        sh = np.sort(h)
        u = ((np.searchsorted(sh, x, "left") + np.searchsorted(sh, x, "right")) / 2 + 0.5) / (nh + 1)
        g = ndtri(u)
        a = np.abs(g) - np.abs(g[:nh]).mean()
        post = slice(nh + t, None)
        n = len(o) - t
        row = []
        for L in LAGS:
            pg = g[L:] * g[:-L]
            pa = a[L:] * a[:-L]
            row.append(pg[nh + t - L:].sum() / np.sqrt(n) - pg[: nh - L].mean() * np.sqrt(n))  # signed
            row.append(pa[nh + t - L:].sum() / np.sqrt(n) - pa[: nh - L].mean() * np.sqrt(n))
        # portmanteau over lags 1..10 and 1..30 on the post segment
        gp = g[post]
        ac = np.array([np.mean(gp[L:] * gp[:-L]) for L in range(1, 31)]) * np.sqrt(n)
        row += [np.sum(ac[:10] ** 2), np.sum(ac ** 2)]
        R.append(row)
        lab.append(ds.tau[k] >= 0)
    R, lab = np.array(R), np.array(lab)
    print(f"{len(lab)} series with >=100 post points")
    for i, L in enumerate(LAGS):
        print(f"lag {L:>3}: |gg| AUC {roc_auc_score(lab, np.abs(R[:, 2*i])):.3f}   |aa| AUC {roc_auc_score(lab, np.abs(R[:, 2*i+1])):.3f}")
    print(f"LB10 AUC {roc_auc_score(lab, R[:, -2]):.3f}  LB30 AUC {roc_auc_score(lab, R[:, -1]):.3f}")


if __name__ == "__main__":
    run()
