from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from data_loader.load import NO_BREAK, Dataset, load_train

MIN_SIDE = 50   # min obs on each side of tau before estimation
THRES_Q = 0.95  # null-control quantile
FIGDIR = "figures"
RNG_SEED = 0

# vectorized segment stats

# mean/sd/acf1 over some half-open int [lo, hi)

class Prefix:
    def __init__(self, value: np.ndarray) -> None:
        v = value.astype(np.float64, copy=False)
        self.v = v
        z = np.zeros(1, dtype=np.float64)
        self.c1 = np.concatenate([z, np.cumsum(v)])
        self.c2 = np.concatenate([z, np.cumsum(v * v)])

        self.cp = np.concatenate([z, np.cumsum(v[:-1] * v[1:])])

    def stats(self, lo: np.ndarray, hi: np.ndarray) -> "SegStats":
        lo = np.asarray(lo, dtype=np.int64)
        hi = np.asarray(hi, dtype=np.int64)
        n = hi - lo

        ok = n >= 2

        lo_s = np.where(ok, lo, 0)
        hi_s = np.where(ok, hi, 2)
        n_s = (hi_s - lo_s).astype(np.float64)

        s = self.c1[hi_s] - self.c1[lo_s]
        q = self.c2[hi_s] - self.c2[lo_s]
        mean = s / n_s

        css = np.maximum(q - n_s * mean * mean, 0.0)
        var = css / (n_s - 1.0)
        sd = np.sqrt(var)

        #acf1 = sum_{i=lo}^{hi-2} (x_i - m)(x_{i+1} - m ) / sum_{i=lo}^{hi-1} (x_i - m)^2
        s_lag = self.cp[hi_s - 1] - self.cp[lo_s]
        s_head = s - self.v[hi_s - 1]
        s_tail = s - self.v[lo_s]
        num = s_lag - mean * s_head - mean * s_tail + (n_s - 1.0) * mean * mean

        with np.errstate(divide="ignore", invalid="ignore"):
            acf1 = np.where(css > 0, num / css, np.nan)

        nan = np.full(n.shape, np.nan)
        return SegStats(
            n=n,
            mean=np.where(ok, mean, nan),
            sd=np.where(ok, sd, nan),
            acf1=np.where(ok, acf1, nan),
        )

@dataclass
class SegStats:
    n: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    acf1: np.ndarray

# segment bounds
def hist_bounds(ds: Dataset):
    return ds.starts, ds.starts + ds.hist_len

def online_bounds(ds: Dataset):
    return ds.starts + ds.hist_len, ds.ends

def full_bounds(ds: Dataset):
    return ds.starts, ds.ends


def break_position(ds: Dataset) -> np.ndarray:
    m = ds.has_break
    tau = ds.tau[m]
    olen = ds.online_len[m]

    u = (tau + 0.5) / olen
    n = len(ds)
    n_break = int(m.sum())
    p_rate = stats.binomtest(n_break, n, 0.5).pvalue
    print(f"break rate {m.mean():4.f} ({n_break}/{n}) vs 0.5: p = {p_rate:.3f}")
    print(f"tau min={tau.min()} max={tau.max()} median={np.median(olen):.0f}")
    print(f"online_len min={olen.min()} max={olen.max()} median={np.median(olen):.0f}")

    ks = stats.kstest(u, "uniform")
    print(f"u ~ uniform? KS D={ks.statistic:.4f} p={ks.pvalue:.3f}")

    counts = np.histogram(u, bins=10, range=(0,1))[0]
    chi = stats.chisquare(counts)
    print(f"ch2 (10bins) p={chi.pvalue:.3f}")
    print(f"mean={u.mean():.4f} (should be 0.5 +/- {np.sqrt(1/12/len(u)):.4f})")

    fig,ax = plt.subplots(1, 2, figsize=(11,4))
    ax[0].hist(tau,bins=60)
    ax[0].set_title("tau (index within online segment)")
    ax[1].hist(u, bins=40, range=(0,1))
    ax[1].axhline(len(u) / 40, ls="--", c="k", lw=1, label="uniform")
    ax[1].set_title("tau / online_len")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/q3_break_position.png", dpi=110)
    plt.close(fig)

    return u

def standardization(ds: Dataset, pre: Prefix) -> None:
    seg = {
        "x_hist": (pre.stats(*hist_bounds(ds)), ds.hist_len),
        "x_online": (pre.stats(*online_bounds(ds)), ds.online_len),
        "full series": (pre.stats(*full_bounds(ds)), ds.ends - ds.starts),
    }

    forced = {}
    for name, (s, seglen) in seg.items():
        n = seglen.astype(np.float64)
        sd1 = s.sd
        sd0 = sd1 * np.sqrt((n-1.0) / n)

        sp_mean = float(np.nanstd(s.mean * np.sqrt(n)))
        sp_sd1 = float(np.nanstd((sd1-1.0) * np.sqrt(2.0*n)))
        sp_sd0 = float(np.nanstd((sd0-1.0) * np.sqrt(2.0*n)))

        ddof = 1 if sp_sd1 <= sp_sd0 else 0
        sp_sd = min(sp_sd1, sp_sd0)

        mean_forced = sp_mean < 0.05
        sd_forced = sp_sd < 0.05
        forced[name] = mean_forced and sd_forced

        tag = f"ddof={ddof}" if sd_forced else "-"
        verdict = "STANDARDIZED" if forced[name] else "sample"

        print(f"{name:<12} {sp_mean:>13.4f} {sp_sd:>11.4f} {tag:>7} {verdict}")

        h = seg["x_hist"][0]

        fig, ax = plt.subplots(1, 2, figsize=(11,4))
        ax[0].hist(h.mean, bins=80)
        ax[0].set_title("mean(x_hist)")
        ax[1].hist(h.sd, bins=80)
        ax[1].set_title("std(x_hist)")
        fig.tight_layout()
        fig.savefig(f"{FIGDIR}/q4_standardization.png", dpi=110)
        plt.close(fig)

def contrast(pre: Prefix, lo, cut, hi):
    before = pre.stats(lo, cut)
    after = pre.stats(cut, hi)

    d_mean = (after.mean - before.mean) / before.sd

    with np.errstate(divide="ignore", invalid="ignore"):
        d_logsd = np.log(after.sd / before.sd)

    d_acf1 = after.acf1 - before.acf1

    return before, after, d_mean, d_logsd, d_acf1

def break_census(ds: Dataset, pre: Prefix, u_empirical: np.ndarray) -> None:
    onl_lo, onl_hi = online_bounds(ds)

    # treatment group, real breaks
    m = ds.has_break
    b_lo, b_hi = onl_lo[m], onl_hi[m]
    b_cut = b_lo + ds.tau[m]

    keep_b = ((b_cut - b_lo) >= MIN_SIDE) & ((b_hi - b_cut)  MIN_SIDE)
    print(f"\nbreak_series {m.sum()}")
    print(f" with >= {MIN_SIDE} points a side {keep_b.sum()} "
          f"({100 * keep_b.mean():.1f}%; dropped {100 * (1 - keep_b.mean()):1.f}%)")

    _, _, dm, ds_, da = contrast(pre, b_lo[keep_b], b_cut[keep_b], b_hi[keep_b])

    #control grp
    c = -m
    c_lo, c_hi = onl_lo[c], onl_hi[c]
    c_olen = (c_hi - c_lo).astype(np.float64)

    rng = np.random.default_rng(RNG_SEED)
    u_fake = rng.choice(u_empirical, size=len(c_lo), replace=True)
    c_off = np.clip(np.floor(u_fake * c_olen).astype(np))



