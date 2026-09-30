"""Is the 'history = null' assumption calibrated? Compare var z-stats of online windows
for no-break series with the same stat on windows *inside* the history."""
import numpy as np
from scipy.special import ndtri
from common import load_train


def run():
    ds = load_train()
    rng = np.random.default_rng(0)
    out = {"nobreak_online": [], "break_pre": [], "break_post": [], "hist_internal": []}
    for k in range(len(ds)):
        h, o = ds.segments(k)
        nh = len(h)
        sh = np.sort(h)
        def gs(x):
            u = ((np.searchsorted(sh, x, "left") + np.searchsorted(sh, x, "right")) / 2 + 0.5) / (nh + 1)
            return ndtri(u)
        def vstat(gw):
            return ((gw ** 2).mean() - 1) * np.sqrt(len(gw) / 2)
        t = int(ds.tau[k])
        if t < 0 and len(o) >= 100:
            out["nobreak_online"].append(vstat(gs(o[:100])))
        if t >= 100:
            out["break_pre"].append(vstat(gs(o[t - 100:t])))
        if t >= 0 and len(o) - t >= 100:
            out["break_post"].append(vstat(gs(o[t:t + 100])))
        s = rng.integers(0, nh - 100)
        out["hist_internal"].append(vstat(gs(h[s:s + 100])))
    for kname, v in out.items():
        v = np.array(v)
        print(f"{kname:>15}: n={len(v):5d}  mean {v.mean():+.3f}  sd {v.std():.3f}  P(|z|>1.96) {np.mean(np.abs(v) > 1.96):.3f}  P(|z|>4) {np.mean(np.abs(v) > 4):.3f}")


if __name__ == "__main__":
    run()
