"""Oracle AUC (true tau): iid-null stats vs per-series-calibrated vs vol-whitened stats."""
import numpy as np
from scipy.special import ndtri
from sklearn.metrics import roc_auc_score
from common import load_train


def ewma_filter(g, lam, v0):
    v = np.empty_like(g)
    prev = v0
    for i in range(len(g)):
        v[i] = prev
        prev = lam * prev + (1 - lam) * g[i] * g[i]
    return g / np.sqrt(v)


def stats(w):
    n = len(w)
    return np.array([
        w.mean() * np.sqrt(n),
        ((w * w).mean() - 1) * np.sqrt(n / 2),
        (np.abs(w).mean() - np.sqrt(2 / np.pi)) * np.sqrt(n),
        (w[1:] * w[:-1]).mean() * np.sqrt(n),
        ((np.abs(w[1:]) * np.abs(w[:-1])).mean() - 2 / np.pi) * np.sqrt(n),
    ])


NAMES = ["mean", "var", "abs", "acf1", "vol1"]


def run():
    ds = load_train()
    rng = np.random.default_rng(0)
    u_emp = (ds.tau[ds.has_break] + 0.5) / ds.online_len[ds.has_break]
    cols = {k: [] for k in ["iid", "cal", "wh97", "wh97cal", "wh99", "wh99cal"]}
    lab = []
    for k in range(0, len(ds), 2):
        h, o = ds.segments(k)
        nh = len(h)
        t = int(ds.tau[k]) if ds.tau[k] >= 0 else int(rng.choice(u_emp) * len(o))
        n = len(o) - t
        if n < 50:
            continue
        x = np.concatenate([h, o]).astype(np.float64)
        sh = np.sort(h)
        u = ((np.searchsorted(sh, x, "left") + np.searchsorted(sh, x, "right")) / 2 + 0.5) / (nh + 1)
        g = ndtri(u)
        n_ = min(n, nh // 4)
        starts = rng.integers(200, nh - n_, size=40)
        for tag, s in [("iid", g), ("wh97", ewma_filter(g, 0.97, 1.0)), ("wh99", ewma_filter(g, 0.99, 1.0))]:
            if tag != "iid":
                # re-standardise whitened series on history (post-burn-in)
                hh = s[200:nh]
                s = (s - hh.mean()) / hh.std()
            post = stats(s[nh + t: nh + t + n_])
            cols[tag].append(post)
            null = np.array([stats(s[a:a + n_]) for a in starts])
            cols[tag + "cal" if tag != "iid" else "cal"].append((post - null.mean(0)) / null.std(0))
        lab.append(ds.tau[k] >= 0)
    lab = np.array(lab)
    print(f"{len(lab)} series (>=50 post points, window capped at hist/4)")
    for tag, v in cols.items():
        v = np.abs(np.array(v))
        print(f"{tag:>8}: " + "  ".join(f"{nm} {roc_auc_score(lab, v[:, i]):.3f}" for i, nm in enumerate(NAMES)))
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import HistGradientBoostingClassifier
    for combo in (["iid"], ["iid", "cal"], ["iid", "cal", "wh97", "wh97cal", "wh99", "wh99cal"]):
        X = np.abs(np.hstack([np.array(cols[c]) for c in combo]))
        p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200), X, lab, cv=5, method="predict_proba")[:, 1]
        print(f"GBDT {'+'.join(combo)}: {roc_auc_score(lab, p):.3f}")


if __name__ == "__main__":
    run()
