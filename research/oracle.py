"""How detectable are the breaks at all? Oracle z-stats using the TRUE tau.

For break series: standardised mean of each stream over x_online[tau:].
For no-break series: same over x_online[u*T:] with u drawn from the tau/T distribution.
AUC of |stat| between the two groups = how separable each break type is with
perfect knowledge of where to look.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from common import load_train, main


class Probe(main.FeatureState):
    def streams(self, x_online):
        out = []
        for x in x_online:
            self.update(float(x))
            out.append(self._last_e)
        return np.array(out)


# capture the standardised stream vector of each step
_orig = main.FeatureState.update


def _update(self, x):
    zt = (x - self.mu) / self.sd
    feats = _orig(self, x)
    # re-derive e from the cumulative sums (sums[i] after - before)
    s = np.array(self.sums)
    self._last_e = s - getattr(self, "_prev_sums", 0.0)
    self._prev_sums = s
    return feats


main.FeatureState.update = _update


def run():
    ds = load_train()
    rng = np.random.default_rng(0)
    u_emp = (ds.tau[ds.has_break] + 0.5) / ds.online_len[ds.has_break]
    stats, lab, npost = [], [], []
    for k in range(0, len(ds), 2):
        h, o = ds.segments(k)
        E = Probe(h).streams(o)
        t = int(ds.tau[k])
        if t < 0:
            t = int(rng.choice(u_emp) * len(o))
        post = E[t:]
        if len(post) < 5:
            continue
        stats.append(np.abs(post.sum(0)) / np.sqrt(len(post)))
        lab.append(ds.tau[k] >= 0)
        npost.append(len(post))
    S, lab, npost = np.array(stats), np.array(lab), np.array(npost)
    print(f"{len(lab)} series")
    for i, name in enumerate(main.STREAMS):
        print(f"{name:>5}  AUC {roc_auc_score(lab, S[:, i]):.3f}   "
              f"frac breaks |z|>4: {np.mean(S[lab, i] > 4):.3f}  (null {np.mean(S[~lab, i] > 4):.3f})")
    best = S.max(1)
    print(f"max-over-streams AUC {roc_auc_score(lab, best):.3f}; "
          f"breaks with any |z|>5: {np.mean(best[lab] > 5):.3f} (null {np.mean(best[~lab] > 5):.3f})")


if __name__ == "__main__":
    run()
