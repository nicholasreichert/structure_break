"""TS-AUC of every cached feature used alone as the score (on a random half of train series)."""

import numpy as np

from common import cached, load_train, ts_auc


def run():
    tr = cached("train", load_train)
    keep = np.random.default_rng(0).random(tr["series"].max() + 1) < 0.5
    m = keep[tr["series"]]
    y, step = tr["y"][m], tr["step"][m]
    rows = []
    for j, name in enumerate(tr["names"]):
        rows.append((ts_auc(y, tr["X"][m, j], step), name))
    for auc, name in sorted(rows, reverse=True):
        print(f"{auc:.4f}  {name}")


if __name__ == "__main__":
    run()
