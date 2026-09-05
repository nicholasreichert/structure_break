
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PERIOD_HISTORICAL = 1
PERIOD_ONLINE = 2
NO_BREAK = -1  # sentinel used in y_*_index.parquet


@dataclass
class Dataset:
    ids: np.ndarray       # (n_series,)  int64, sorted ascending
    value: np.ndarray     # (n_rows,)    float32, every observation concatenated
    starts: np.ndarray    # (n_series,)  first row of each series
    ends: np.ndarray      # (n_series,)  one past the last row
    hist_len: np.ndarray  # (n_series,)  length of the historical segment
    tau: np.ndarray       # (n_series,)  int64, -1 = no break (all -1 if unlabelled)

    # positional access (k = 0..n-1, not dataset id)

    def __len__(self) -> int:
        return len(self.ids)

    def segments(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        s, h, e = self.starts[k], self.hist_len[k], self.ends[k]
        return self.value[s : s + h], self.value[s + h : e]
 
    def tuple_at(self, k: int):
        x_hist, x_online = self.segments(k)
        t = int(self.tau[k])
        return int(self.ids[k]), x_hist, x_online, (None if t == NO_BREAK else t)

    def labels(self, k: int) -> np.ndarray:
        _, x_online = self.segments(k)
        y = np.zeros(len(x_online), dtype=np.int64)
        t = int(self.tau[k])
        if t != NO_BREAK:
            y[t:] = 1
        return y


    def index_of(self, series_id: int) -> int:
        k = int(np.searchsorted(self.ids, series_id))
        if k >= len(self.ids) or self.ids[k] != series_id:
            raise KeyError(f"no series with id {series_id}")
        return k

    def get(self, series_id: int):
        return self.tuple_at(self.index_of(series_id))


    def __iter__(self) -> Iterator:
        for k in range(len(self)):
            yield self.tuple_at(k)


    @property
    def online_len(self) -> np.ndarray:
        return self.ends - self.starts - self.hist_len

    @property
    def has_break(self) -> np.ndarray:
        return self.tau != NO_BREAK


def load(x_path: str, index_path: Optional[str] = None) -> Dataset:
    tbl = pq.read_table(x_path, columns=["id", "period", "value"])
    ids_col = tbl["id"].to_numpy()
    period = tbl["period"].to_numpy().astype(np.int8)
    value = tbl["value"].to_numpy().astype(np.float32)

    assert np.all(np.diff(ids_col) >= 0), "X parquet is not sorted by id"

    ids = np.unique(ids_col)
    starts = np.searchsorted(ids_col, ids, side="left")
    ends = np.searchsorted(ids_col, ids, side="right")

    hist_len = np.add.reduceat((period == PERIOD_HISTORICAL).astype(np.int64), starts)

    assert np.array_equal(
        np.add.reduceat((period == PERIOD_ONLINE).astype(np.int64), starts),
        ends - starts - hist_len,
    ), "historical and online rows are interleaved"

    if index_path is None:
        tau = np.full(len(ids), NO_BREAK, dtype=np.int64)
    else:
        yi = pd.read_parquet(index_path)
        aligned = yi["tau_index"].reindex(ids)
        assert not aligned.isna().any(), "some ids have no row in the index file"
        tau = aligned.to_numpy().astype(np.int64)

    return Dataset(ids=ids, value=value, starts=starts, ends=ends,
                   hist_len=hist_len, tau=tau)


def load_train() -> Dataset:
    return load("data/X_train.parquet", "data/y_train_index.parquet")


def load_test_reduced() -> Dataset:
    return load("data/X_test.reduced.parquet", "data/y_test_index.reduced.parquet")


def self_test(ds: Dataset, y_path: str, n_check: int = 300, seed: int = 0) -> None:
    y = pd.read_parquet(y_path)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(ds), size=min(n_check, len(ds)), replace=False)

    n_break = 0
    for k in picks:
        sid = int(ds.ids[k])
        provided = y.loc[sid]["target"].to_numpy()
        expected = ds.labels(k)

        assert len(expected) == len(provided), (
            f"id {sid}: online length {len(expected)} != {len(provided)} label rows"
        )
        assert np.array_equal(expected, provided), f"id {sid}: labels disagree"
        n_break += int(ds.tau[k] != NO_BREAK)

    assert n_break > 0, "sample contained no break series; test was vacuous"
    print(f"self_test OK: {len(picks)} series ({n_break} with breaks) match y_train")


if __name__ == "__main__":
    import time

    t0 = time.time()
    ds = load_train()
    print(f"loaded {len(ds)} series in {time.time() - t0:.2f}s")
    print(f"  with break : {ds.has_break.sum()} ({100 * ds.has_break.mean():.1f}%)")
    print(f"  hist len   : {ds.hist_len.min()}–{ds.hist_len.max()}")
    print(f"  online len : {ds.online_len.min()}–{ds.online_len.max()}")

    sid, x_hist, x_online, tau = ds.get(0)
    print(f"  id {sid}: hist={len(x_hist)} online={len(x_online)} tau={tau}")

    self_test(ds, "data/y_train.parquet")
