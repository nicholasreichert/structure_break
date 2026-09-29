"""Oracle AUC of two-sample tests: hist vs x_online[tau:] (null: random cut). Uses true tau."""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from common import load_train


def run():
    ds = load_train()
    rng = np.random.default_rng(0)
    u_emp = (ds.tau[ds.has_break] + 0.5) / ds.online_len[ds.has_break]
    R, lab, npost = [], [], []
    for k in range(0, len(ds), 4):
        h, o = ds.segments(k)
        h = h.astype(np.float64); o = o.astype(np.float64)
        t = int(ds.tau[k]) if ds.tau[k] >= 0 else int(rng.choice(u_emp) * len(o))
        post = o[t:]
        if len(post) < 30:
            continue
        ks = stats.ks_2samp(h, post).statistic * np.sqrt(len(post))
        mw = abs(stats.mannwhitneyu(h, post).statistic / (len(h) * len(post)) - 0.5) * np.sqrt(len(post))
        lev = stats.levene(h, post).statistic
        ad = stats.anderson_ksamp([h, post]).statistic
        ac = lambda a, L: np.corrcoef(a[L:], a[:-L])[0, 1]
        dac1 = abs(ac(post, 1) - ac(h, 1)) * np.sqrt(len(post))
        dac1a = abs(ac(np.abs(post), 1) - ac(np.abs(h), 1)) * np.sqrt(len(post))
        R.append([ks, mw, lev, ad, dac1, dac1a]); lab.append(ds.tau[k] >= 0); npost.append(len(post))
    R, lab, npost = np.array(R), np.array(lab), np.array(npost)
    names = ["KS", "MannWhitney", "Levene", "AndersonDarling", "d_acf1", "d_acf1_abs"]
    print(f"{len(lab)} series")
    for i, nme in enumerate(names):
        print(f"{nme:>16}: AUC {roc_auc_score(lab, R[:, i]):.3f}")
    big = npost >= 200
    print("--- post >= 200 points only:", big.sum())
    for i, nme in enumerate(names):
        print(f"{nme:>16}: AUC {roc_auc_score(lab[big], R[big, i]):.3f}")


if __name__ == "__main__":
    run()
