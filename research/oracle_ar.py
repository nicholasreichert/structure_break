"""Oracle AUC of residual-based streams for richer null models fitted on history."""
import numpy as np
from scipy.special import ndtri
from sklearn.metrics import roc_auc_score
from common import load_train, main


def ar_resid(z, nh, p):
    X = np.column_stack([np.r_[np.zeros(i + 1), z[: -i - 1]] for i in range(p)])
    phi, *_ = np.linalg.lstsq(X[p:nh], z[p:nh], rcond=None)
    r = z - X @ phi
    return r / r[p:nh].std()


def ewma_vol(r, lam):
    v = np.empty_like(r)
    v[0] = 1.0
    r2 = r * r
    for i in range(1, len(r)):
        v[i] = lam * v[i - 1] + (1 - lam) * r2[i - 1]
    return r / np.sqrt(v)


def run():
    ds = load_train()
    rng = np.random.default_rng(0)
    u_emp = (ds.tau[ds.has_break] + 0.5) / ds.online_len[ds.has_break]
    rows, lab, names = [], [], None
    for k in range(0, len(ds), 3):
        h, o = ds.segments(k)
        x = np.concatenate([h, o]).astype(np.float64)
        nh = len(h)
        t = int(ds.tau[k]) if ds.tau[k] >= 0 else int(rng.choice(u_emp) * len(o))
        if len(o) - t < 5:
            continue
        z = (x - h.mean()) / h.std()
        S = {}
        for p in (1, 3, 5, 10, 20):
            r = ar_resid(z, nh, p)
            S[f"r2_p{p}"] = r * r
            S[f"absr_p{p}"] = np.abs(r)
        r5 = ar_resid(z, nh, 5)
        for lam in (0.9, 0.97):
            f = ewma_vol(r5, lam)
            S[f"f2_p5_l{lam}"] = f * f
        rh = np.sort(r5[5:nh])
        u = ((np.searchsorted(rh, r5, "left") + np.searchsorted(rh, r5, "right")) / 2 + 0.5) / (len(rh) + 1)
        rg = ndtri(u)
        S["rg2_p5"] = rg * rg
        S["rgabs_p5"] = np.abs(rg)
        S["rg_lag1_p5"] = rg * np.r_[0, rg[:-1]]
        names = list(S)
        row = []
        for n_ in names:
            s = S[n_]
            sh_ = s[25:nh]
            post = s[nh + t:]
            row.append(abs(post.mean() - sh_.mean()) / main._long_run_sd(sh_) * np.sqrt(len(post)))
        rows.append(row)
        lab.append(ds.tau[k] >= 0)
    R, lab = np.array(rows), np.array(lab)
    for auc, n_ in sorted(((roc_auc_score(lab, R[:, i]), n_) for i, n_ in enumerate(names)), reverse=True):
        print(f"{n_:>12} {auc:.3f}")


if __name__ == "__main__":
    run()
