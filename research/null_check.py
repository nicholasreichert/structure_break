"""Are no-break online segments distributed like their own history? KS p-values by group."""
import numpy as np
from scipy import stats

from common import load_train


def run():
    ds = load_train()
    out = {"null: hist vs online": [], "break: hist vs pre": [], "break: hist vs post": [], "break: pre vs post": []}
    for k in range(0, len(ds), 3):
        h, o = ds.segments(k)
        t = int(ds.tau[k])
        if t < 0:
            if len(o) >= 60:
                out["null: hist vs online"].append(stats.ks_2samp(h, o).pvalue)
        elif t >= 30 and len(o) - t >= 30:
            out["break: hist vs pre"].append(stats.ks_2samp(h, o[:t]).pvalue)
            out["break: hist vs post"].append(stats.ks_2samp(h, o[t:]).pvalue)
            out["break: pre vs post"].append(stats.ks_2samp(o[:t], o[t:]).pvalue)
    for kname, p in out.items():
        p = np.array(p)
        print(f"{kname:<22} n={len(p):5d}  P(p<0.01)={np.mean(p < 0.01):.3f}  P(p<1e-4)={np.mean(p < 1e-4):.3f}  median p={np.median(p):.3f}")


if __name__ == "__main__":
    run()
