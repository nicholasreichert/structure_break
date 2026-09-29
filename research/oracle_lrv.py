"""Compare long-run-sd estimators for normalising streams (oracle AUC, true tau)."""
import numpy as np
from sklearn.metrics import roc_auc_score
from common import load_train, main
from oracle_streams import candidate_streams

KEEP = ["g2", "absg", "absdz", "vol1", "gg1", "gg3", "tail05", "g4"]


def batch_sd(s, L):
    m = len(s) // L
    b = s[: m * L].reshape(m, L).mean(1)
    return b.std(ddof=1) * np.sqrt(L)


EST = {
    "bartlett": main._long_run_sd,
    "iid": lambda s: s.std(),
    "bm25": lambda s: batch_sd(s, 25),
    "bm50": lambda s: batch_sd(s, 50),
    "bm100": lambda s: batch_sd(s, 100),
    "max_b_bm50": lambda s: max(main._long_run_sd(s), batch_sd(s, 50)),
}


def run():
    ds = load_train()
    rng = np.random.default_rng(0)
    u_emp = (ds.tau[ds.has_break] + 0.5) / ds.online_len[ds.has_break]
    R = {e: [] for e in EST}
    lab = []
    for k in range(0, len(ds), 3):
        h, o = ds.segments(k)
        x = np.concatenate([h, o]).astype(np.float64)
        nh = len(h)
        t = int(ds.tau[k]) if ds.tau[k] >= 0 else int(rng.choice(u_emp) * len(o))
        if len(o) - t < 5:
            continue
        S = candidate_streams(x, nh)
        for e, f in EST.items():
            row = []
            for name in KEEP:
                s = S[name]
                sh_ = s[1:nh]
                post = s[nh + t:]
                row.append(abs(post.mean() - sh_.mean()) / f(sh_) * np.sqrt(len(post)))
            R[e].append(row)
        lab.append(ds.tau[k] >= 0)
    lab = np.array(lab)
    print("est        " + " ".join(f"{n:>7}" for n in KEEP))
    for e in EST:
        A = np.array(R[e])
        print(f"{e:<10} " + " ".join(f"{roc_auc_score(lab, A[:, i]):7.3f}" for i in range(len(KEEP))))


if __name__ == "__main__":
    run()
