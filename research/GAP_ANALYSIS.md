# Gap analysis, 29 Sep 2026: why we sit at 0.614 while the top is 0.687

All numbers are 2-fold grouped CV TS-AUC on train unless marked "oracle". An oracle score uses
the true break time, compares data after it with the history, and is a plain AUC over series.

## What the leaders are *not* doing (ruled out)

| Hypothesis | Test | Result |
|---|---|---|
| Know the online length T | `(t+1)/T` alone scores 0.631 | T is never sent (socket runner, one point at a time); staff: "not accessible by design", exploiting it = cheating |
| Series id / order leak | corr(id, T), corr(id, break) | 0.00 |
| Float-precision or splice artefact | mantissa bits, tie rate, jump at tau | identical pre/post, no ties, jump ratio 1.07 |
| Discrete price-lattice in real series | tie fraction of histories | ~0: no lattice |
| Scoring stratum differs from ours | read `scoring.py` | same `cumcount` online step as `research/common.ts_auc` |

## What the data actually looks like

- **Breaks are subtle in marginals.** A KS test of post-break data against history rejects at 1%
  for 15.6% of break series vs 13.2% of no-break online segments.
- **The iid null is badly wrong: volatility clustering.** Under iid, a 100-point variance
  z-stat has sd 1. We measure sd 1.9 on windows inside the history and 2.4 on no-break online
  segments. The miscalibration also differs by series, which directly costs cross-sectional AUC.
- Dependence signal is spread thinly over many lags (oracle AUC 0.52-0.56 per lag, 1-50).
- Forum (t/1206) reports the break mix as dependence 54%, volatility 24%, tails 11%, mean 8%,
  variance 3%. Two teams there measured ceilings of ~0.63-0.64 for moment/whitening pipelines,
  with model class (GRU/TCN/transformer) adding <= 0.001.

## What moved the needle

| Change | CV |
|---|---|
| v6 (baseline, same folds) | 0.6140 |
| + Dirichlet/BOCPD Bayes-factor features | 0.6170 with cal (hurt) |
| **+ calibrated block (v7)**: EWMA-vol whitening + per-series horizon calibration | **0.6194** |
| + second calibrated block (AR residual, lag 2/5, tail/centre) | 0.6189 (no gain) |

### Follow-ups, 29-30 Sep (`research/cv2.py`, v7 features, same 2 folds)

Seed noise alone is about +-0.0015 on this OOF score: v7 gave 0.6194 and 0.6178 under
different bagging/feature seeds. Single-run differences below ~0.002 are not results.

| Variant | OOF | Verdict |
|---|---|---|
| v7 reference (seed 0) | 0.6178 | |
| per-step TS-AUC row weights | 0.6172 | no |
| `rank_xendcg`, online step as query | 0.6169 (0.6193 blended with v7) | no alone; blend gain is variance reduction |
| + per-series GARCH(1,1)-whitened block | 0.6196 single seed; **3 seeds: 0.6204 vs 0.6201 without** | noise, dropped |
| + order-3 ordinal-pattern block | 0.6148 | worse |
| + horizon-recalibrated v6 sums/windows (`rcal.py`) | 0.6168 | no |
| num_leaves 31 | 0.6185 | noise |
| lr 0.025, 1200 rounds | 0.6198 | yes |
| **lr 0.025, 1200 rounds, 3-seed average (v8)** | **0.6201** | shipped |

Oracle check of the same idea on 5 simple statistics: GBDT 0.547 (iid null) -> 0.563
(+calibration) -> 0.607 (+whitened).

## Honest read on the remaining gap

Every legitimate family we and the forum have measured tops out around 0.62-0.64. The ~0.05
beyond that is not explained by any detector or model class anyone has published. The
remaining candidates are all things we can't or shouldn't do: external data matching the
real-world half of the series, or information the runner is designed to hide.
v8 (v7 features, slower learning, 3-seed average) is the end of the cheap gains.

Scripts: `null_calib.py`, `null_check.py`, `oracle_acf.py`, `oracle_lags.py`, `oracle_whiten.py`,
`bayes_cp.py`, `calfeat.py`, `calfeat2.py`, `cv_extra.py`, `blocks2.py`, `rcal.py`, `cv2.py`.

## 30 Sep afternoon: where the gap actually is, and the one lever left (v9)

Fast settings (lr 0.05, 600 rounds). NB `common.folds()` makes **5** folds; "2-fold CV" trains
on 4/5 and validates folds 0 and 1 only, so OOF arrays must be masked to folds < 2.

| Probe (`diag_gap.py`) | Fold 0 TS-AUC |
|---|---|
| v8 features, 4/5 of series | 0.6310 |
| v8 features, 2/5 of series | 0.6203 |
| v8 features + hidden T (`log T`, `(t+1)/T`), **diagnostic only** | **0.6874** |
| T features alone | 0.6023 |

- **The leaderboard gap is information about T.** Our detector plus the online length lands
  on the top score (0.687) almost exactly. T is hidden by the runner and using it is against
  the rules, so the realistic ceiling for honest pipelines is the ~0.62-0.64 band.
- **The model is data-limited**: +0.011 per doubling of training series. Break examples are
  the scarce part.

| Variant (2-fold OOF) | OOF |
|---|---|
| baseline, seeds 1-2 | 0.6195 |
| model-level pseudo-null: last m history points scored as a known no-break run, stage 2 on the gap (`pseudo_null.py`) | 0.571 (stage 2 without it 0.607): no |
| history tails as extra negative series (`pseudo_aug.py`) | 0.6178: no |
| + 1 shifted-start copy (`shift_aug.py`) | 0.6236 |
| **+ 2 shifted-start copies** | **0.6271** (both folds, both seeds) |
| + 4 copies at stride 4 (same row count) | 0.6269 |

A shifted-start copy moves the first k < tau online points into the history (tau' = tau - k).
The post-break data is the same, but the detector sees it after a different history
calibration and different accumulated sums. That is enough to regularise the trees. v9 ships
2 copies. At production settings (lr 0.025, 1200 rounds, seed 0): **0.6265 vs v8 0.6198**.
