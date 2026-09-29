"""Screen candidate streams: oracle AUC of the post-break standardised mean (true tau).

Vectorised numpy only -- this is a screening tool, not the production feature code.
"""

import numpy as np
from scipy.special import ndtri
from sklearn.metrics import roc_auc_score

from common import load_train, main


def lrsd(s):
    return main._long_run_sd(s)


def candidate_streams(x, nh):
    """x = hist+online concatenated. Returns dict name -> stream over full x (same length)."""
    h = x[:nh]
    z = (x - h.mean()) / h.std()
    sh = np.sort(h)
    rank = (np.searchsorted(sh, x, "left") + np.searchsorted(sh, x, "right")) / 2.0
    u = (rank + 0.5) / (nh + 1.0)
    g = ndtri(u)

    def lag(a, k):
        out = np.empty_like(a)
        out[:k] = a[0]
        out[k:] = a[:-k]
        return out

    ag = np.abs(g)
    # EWMA volatility filter (RiskMetrics lambda) on z
    lam = 0.94
    v = np.empty_like(z)
    v[0] = 1.0
    for i in range(1, len(z)):
        v[i] = lam * v[i - 1] + (1 - lam) * z[i - 1] ** 2
    zf = z / np.sqrt(v)
    S = {
        "g2": g * g,
        "absg": ag,
        "g3": g ** 3,
        "g4": np.minimum(g ** 4, 50),
        "logabs": np.log(np.abs(z) + 1e-3),
        "tail05": (u < 0.05) | (u > 0.95),
        "tail01": (u < 0.01) | (u > 0.99),
        "center": np.abs(u - 0.5) < 0.1,
        "upper": u > 0.5,
        "gg1": g * lag(g, 1),
        "gg2": g * lag(g, 2),
        "gg3": g * lag(g, 3),
        "gg5": g * lag(g, 5),
        "gg10": g * lag(g, 10),
        "sgn1": np.sign(g) * np.sign(lag(g, 1)),
        "vol1": ag * lag(ag, 1),
        "vol5": ag * lag(ag, 5),
        "dg2": (g - lag(g, 1)) ** 2,
        "zf2": zf ** 2,
        "absdz": np.abs(z - lag(z, 1)),
    }
    return {k: v.astype(np.float64) for k, v in S.items()}


def run():
    ds = load_train()
    rng = np.random.default_rng(0)
    u_emp = (ds.tau[ds.has_break] + 0.5) / ds.online_len[ds.has_break]
    rows, lab = [], []
    names = None
    for k in range(0, len(ds), 3):
        h, o = ds.segments(k)
        x = np.concatenate([h, o]).astype(np.float64)
        nh = len(h)
        t = int(ds.tau[k]) if ds.tau[k] >= 0 else int(rng.choice(u_emp) * len(o))
        if len(o) - t < 5:
            continue
        S = candidate_streams(x, nh)
        names = list(S)
        r = []
        for name in names:
            s = S[name]
            sh_ = s[1:nh]
            post = s[nh + t:]
            r.append(abs(post.mean() - sh_.mean()) / lrsd(sh_) * np.sqrt(len(post)))
        rows.append(r)
        lab.append(ds.tau[k] >= 0)
    R, lab = np.array(rows), np.array(lab)
    print(f"{len(lab)} series")
    res = sorted(((roc_auc_score(lab, R[:, i]), n) for i, n in enumerate(names)), reverse=True)
    for auc, n in res:
        print(f"{n:>8} {auc:.3f}")


if __name__ == "__main__":
    run()
