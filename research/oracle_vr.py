"""Oracle AUC for multi-scale variance-ratio / spectral streams (true tau)."""
import numpy as np
from sklearn.metrics import roc_auc_score

from common import load_train


def batch_sd(s, L):
    m = len(s) // L
    return s[: m * L].reshape(m, L).mean(1).std(ddof=1) * np.sqrt(L)


def run():
    ds = load_train()
    rng = np.random.default_rng(0)
    u_emp = (ds.tau[ds.has_break] + 0.5) / ds.online_len[ds.has_break]
    scales = (2, 4, 8, 16, 32, 64, 128)
    rows, lab = [], []
    for k in range(0, len(ds), 3):
        h, o = ds.segments(k)
        x = np.concatenate([h, o]).astype(np.float64)
        nh = len(h)
        t = int(ds.tau[k]) if ds.tau[k] >= 0 else int(rng.choice(u_emp) * len(o))
        if len(o) - t < 30:
            continue
        z = (x - h.mean()) / h.std()
        c = np.concatenate([[0.0], np.cumsum(z)])
        r = []
        for L in scales:
            s = np.full(len(z), np.nan)
            s[L - 1:] = (c[L:] - c[:-L]) ** 2 / L  # squared block sum ending at i
            sh = s[L - 1:nh]
            post = s[nh + t:]
            # standardise with batch means at a block size well above L (overlap makes s persistent)
            sd = batch_sd(sh - sh.mean(), 4 * L)
            r.append(abs(post.mean() - sh.mean()) / sd * np.sqrt(len(post)))
            # log variance ratio on the post segment vs history (not standardised): direct VR change
            r.append(abs(np.log(post.mean() + 1e-9) - np.log(sh.mean() + 1e-9)))
        rows.append(r)
        lab.append(ds.tau[k] >= 0)
    R, lab = np.array(rows), np.array(lab)
    print(f"{len(lab)} series")
    for i, L in enumerate(scales):
        print(f"L={L:4d}: z-stat AUC {roc_auc_score(lab, R[:, 2*i]):.3f}   |dlogVR| AUC {roc_auc_score(lab, R[:, 2*i+1]):.3f}")


if __name__ == "__main__":
    run()
