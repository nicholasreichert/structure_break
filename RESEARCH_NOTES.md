# ADIA Lab Structural Break Challenge: Real-Time Edition — research notes

Deadline: **17 Sep 2026, 16:00 UTC**. Quota refresh Wednesdays 16:00 UTC (so ~2 windows left:
now → 9 Sep, 9 Sep → 16 Sep). 15 compute-hours/week. Prize pool $100k.

---

## 0. Facts about the task (from the official docs, not the notebook)

- 10,000 train series (labelled), 100-series reduced local test, 10,000 public + 10,000 private test.
- Historical segment 1,000–5,000 obs (guaranteed break-free); online segment 10–1,000 obs.
- **50% of series contain exactly one break**, at an unknown position in the online segment; 50% have none.
- Breaks are in **mean, variance, distributional shape, or dependence structure**.
- **All series are z-scored using historical-segment statistics only.** So `mu_h ≈ 0`, `sd_h ≈ 1`
  by construction — the baseline's standardisation step is close to a no-op, and location/scale
  heterogeneity is already removed. What is *not* removed: tail shape, kurtosis, serial dependence,
  historical length. Those are the remaining nuisance dimensions.
- Determinism enforced (1e-8 tolerance on a 10% retest).
- Known data fix: 29 series had a break at the very first online step mislabelled (fixed 2026-W23).

---

## 1. What the metric actually rewards (this is where the edge is)

`TS-AUC = Σ_t w(t)·AUC(t) / Σ_t w(t)`, `w(t) = n_pos(t)·n_neg(t)`, AUC computed **cross-sectionally
across series at each fixed online step t**.

### 1.1 Only the cross-sectional ranking at each t matters

AUC is invariant to any strictly increasing transform applied uniformly to all series at that step.
Therefore:

- `KAPPA` has **zero** effect on the score.
- The `tanh` squash has **zero** effect (except through float saturation, see §2.3).
- The `n_eff` recursion has **zero** effect: `n_eff` is a deterministic function of `t` alone, identical
  for every series, so dividing by it is a per-t monotone rescale.
- Any t-only calibration curve you bolt on afterwards has **zero** effect.

The *only* thing in the baseline that can move the leaderboard is `ALPHA`, because it changes the
relative ordering of series. Do not waste time tuning the rest.

### 1.2 The AUC-optimal score is the Shiryaev posterior

At a fixed t, AUC is maximised by ranking on `P(label_t = 1 | F_t)` = `P(τ ≤ t | data so far)`. That is
*exactly* the Shiryaev statistic from Bayesian quickest change detection (Shiryaev 1963). The whole
competition is "estimate the run-length / change posterior as well as you can."

Better: the prior `P(τ ≤ t | still alive at t)` is **common to all series** at step t, so it cancels
out of the ranking. What survives is the **mixture likelihood ratio**

    Λ_t = Σ_{k ≤ t} π(k) · Π_{i=k}^{t} [ f_post(x_i) / f_pre(x_i) ]

i.e. the **Shiryaev–Roberts** statistic, updated in O(1) as `R_t = (1 + R_{t-1}) · LR_t`.

**This is the single most important structural insight.** SR (average-LR) is the right object here,
*not* CUSUM (max-LR), because the target is a posterior over an unknown random change time. The
notebook's "Ideas" section suggests CUSUM; SR is the theoretically matched choice, and SR / SR-r
(Pollak 1985) is known to dominate CUSUM for randomly-timed changes.

### 1.3 Cross-series comparability is the second lever

AUC at step t compares a series with 1,000 historical points and heavy tails against one with 5,000
points and Gaussian tails. If your statistic's **null distribution differs across series**, you lose
ranking power for free. You want a **pivotal / distribution-free** statistic: same H0 law for every
series. Two routes, both cheap here:

1. **Rank / PIT transform** against the historical ECDF: `u_t = F_H(x_t)` is Uniform(0,1) under H0 for
   *any* marginal. Run everything on `u_t` (or normal scores `Φ⁻¹(u_t)`).
2. **Per-series self-calibration**: the historical segment is given in full up front and is guaranteed
   break-free. Replay each detector over it to get that series' own null quantiles as a function of
   run length, then emit the standardised value / empirical p-value. This is the classic
   "in-control reference sample" / self-starting distribution-free control chart setup.

### 1.4 Know your weight profile

`w(t) = n_pos(t)·n_neg(t)`. With online lengths spread over 10–1,000, the number of *alive* series
falls fast with t while `n_pos` rises. Measure the actual `w(t)` curve on the training set before
optimising anything — you may find most of the weight sits at small `t`, in which case detection
*speed* matters far more than late-stage precision.

---

## 2. Review of the baseline as written

### 2.1 Submission-breaking bug in your local copy

Lines 317–320 of your `baseline.py` have both the model load and the readiness `yield` commented out:

```python
# model = joblib.load(os.path.join(model_directory_path, "model.joblib"))
# yield  # Signal readiness to the runner.
```

In the **official** quickstarter notebook both lines are **active**. The bare `yield` is the handshake
that tells the runner the generator is ready; without it the runner's first `next()` consumes what
should be the first score, and every subsequent score is off by one — silently, producing garbage.
Uncomment the `yield` before you submit anything.

### 2.2 Structural mismatch: a tracking filter where you need an accumulator

The EWMA is a *forgetting* filter. Its variance floors at `σ²·α/(2−α)` and does **not** shrink as more
post-break data arrives. So for a mean shift δ, the baseline's SNR saturates at roughly
`δ·sqrt((2−α)/α)` and stays there forever. The optimal statistic (SR / CUSUM / GLR) instead grows like
`(t−τ)·δ²/2`. Since the label stays 1 forever after τ, and the metric keeps re-scoring you at every
later step, the baseline **systematically under-scores long-established breaks** — which is exactly
where `w(t)` is large in the middle of the time axis. This is the biggest single loss.

### 2.3 Float saturation creates ties

The notebook claims the score "does not saturate". In float64, `tanh(x) == 1.0` exactly for
`x ≳ 19.1`, so `|z| ≳ 57` collapses to a hard 1.0. Ties between a positive and a negative score 0.5
in AUC instead of 1.0. Rare but free to fix: use a map that never reaches 1.0 in floating point,
e.g. `s = x/(1+x)` on a non-negative statistic, or `s = 0.5 + atan(z)/π`.

### 2.4 Minor: the ESS constant is wrong

The variance-correct effective sample size for an EWMA is `(2−α)/α ≈ 39` at α = 0.05, not `1/α = 20`.
Harmless for TS-AUC (§1.1), but it will bite you the moment you try to build a calibrated p-value.

### 2.5 Blind spots

Only mean shifts. Variance, shape, and dependence changes — all explicitly present in the data — are
picked up only incidentally (a variance increase makes the EWMA noisier, which slightly raises `|z|`).
Given the data is pre-standardised on history, a pure variance break leaves the mean at 0 and the
baseline is nearly blind to it.

### 2.6 Compute budget (good news)

10k series × ≤1,000 steps ≈ up to 10M scores. At `INFER_PARALLELISM=4`, a 1-hour inference run gives
you ~1.4 ms of core time **per step**. That is far more than the notebook's "a few microseconds"
warning implies — enough for a ~20-feature streaming state update *and* a single-row GBDT predict.
The real risk is Python / pandas call overhead, not FLOPs. Profile `Booster.predict` on a preallocated
`np.empty((1, n_feat))` early; if it is >200 µs, compile with `lleaves` / `treelite` or distil to a
small numpy-evaluated MLP.

---

## 3. The field: what the literature offers

### 3.1 Classical sequential (quickest) change detection

- **CUSUM** — Page (1954); minimax-optimal under Lorden's criterion (Moustakides 1986). Max-LR.
- **Shiryaev (1963)** — Bayesian formulation; optimal rule thresholds the posterior that the change
  has already happened. *This is our objective.*
- **Shiryaev–Roberts** (Roberts 1966; Pollak 1985) — `R_t = (1 + R_{t−1})·LR_t`, O(1), average-LR.
- **Unknown post-change parameter** — window-limited GLR (Lai 1995/1998), mixture likelihood ratio
  (Pollak & Siegmund), adaptive CUSUM. Practical version: a small **grid of hypothesised shift sizes**
  with one SR recursion each, combined by log-sum-exp.
- Surveys: Veeravalli & Banerjee, *Quickest Change Detection* (arXiv:1210.5552); Xie, Zou, Xie,
  Veeravalli, *Sequential (Quickest) Change Detection: Classical Results and New Directions*
  (arXiv:2104.04186); Xie, Xie, Moustakides, *Computation vs statistical performance* (arXiv:2210.05181).

### 3.2 Bayesian online CPD

- **Adams & MacKay (2007)**, BOCPD — maintains the full **run-length posterior** by HMM-style message
  passing. Fearnhead & Liu (2007) independently. `P(break already happened) = 1 − P(r_t = t)` falls
  straight out, which is literally the quantity the metric wants. Naive cost O(t)/step; prune to
  top-K hypotheses for O(K). Nonparametric KDE variants exist (Stat. & Comp. 2024).

### 3.3 Nonparametric / model-free streaming detectors

- **NEWMA** (Keriven, Garreau, Poli 2020, arXiv:1805.08061) — two EWMAs at different forgetting rates
  `Λ > λ` over a feature map Ψ: `z_t = (1−Λ)z_{t−1} + ΛΨ(x_t)`, `z'_t = (1−λ)z'_{t−1} + λΨ(x_t)`,
  statistic `‖z_t − z'_t‖`. With random Fourier features this approximates an MMD between "recent" and
  "older" data **without storing any data**. O(m) time and memory. Purpose-built for exactly this
  constraint set, and catches arbitrary distributional change, not just mean.
- **Online RFF-MMD** (Kalinke et al. 2025, arXiv:2505.17789) — constant-time update, minimax-optimal
  detection delay. A cleaner, more recent version of the same idea.
- **scan-B** (Li, Xie, Dai, Song) — kernel MMD comparing a recent block against past blocks. Stronger
  but O(B²)/step and stores raw data; probably too heavy here.
- **Distribution-free rank CUSUM charts** — Mann–Whitney CUSUM (arXiv:1305.4318), signed sequential
  rank CUSUMs (arXiv:1706.03901), self-starting adaptive rank CUSUM. In-control behaviour is exactly
  distribution-free, i.e. **pivotal across series** — precisely the §1.3 property.
- **e-detectors / testing by betting** (Shin & Ramdas, arXiv:2203.03532; backward confidence
  sequences, arXiv:2302.02544) — anytime-valid nonparametric sequential change detection with
  composite pre- and post-change classes. An e-process is a nonnegative supermartingale under H0, so
  `log E_t` is an **evidence accumulator that is automatically comparable across series**. Conceptually
  the cleanest answer to §1.2 + §1.3 simultaneously.

### 3.4 Learned / classifier-based CPD (why supervised ML is principled here, not a hack)

- **Li, Fearnhead, Fryzlewicz, Wang, JRSSB 2024** (arXiv:2211.03860), *Automatic change-point detection
  via deep learning* — train a network on **simulated** labelled series to classify change / no-change.
  They show standard statistics like CUSUM are representable by simple networks, so the learned
  detector strictly generalises them. With 10,000 labelled series this competition hands you the
  training set for free; this is the amortised-inference argument for a supervised combiner.
- **changeAUC** (arXiv:2404.06995) — use a classifier's *ranking* as the CPD statistic; the null limit
  is pivotal and independent of the classifier, and only rank-monotonicity matters, so a weakly
  trained classifier still works. Reassuring for a competition setting.
- **Londschien et al.** — random-forest classifier-based nonparametric likelihood ratio for CPD.

### 3.5 Prior art from the 2025 (offline) edition of this exact competition

Break location was *known* and one score per series; the winners were all heavy feature engineering
plus gradient boosting stacks. Their feature vocabulary transfers; their computation does not (you
must re-derive everything as an O(1) recursion).

- **1st, Alphabot** — stacking, 8 first-level XGBoost / RandomForest models + meta-model.
  ~0.9065 CV AUC, 0.9014 private.
- **2nd** — 2,408 features from 6 transforms (z-score, cumsum, dense rank, abs, moving average,
  moving std) × descriptive stats + **F-test, Levene, Kolmogorov–Smirnov**; SHAP + LightGBM-gain
  feature selection; LightGBM; plus **TabPFN**-generated features.
- **Public solution (gsoisson)** — 200+ features: quantile ratios, threshold-crossing rates, ACF/PACF
  deltas, FFT bandpower, first/second differences, AR coefficients; robust median/MAD standardisation
  on the reference segment, winsorisation, detrending; XGB + LGBM + CatBoost with an XGB meta-learner,
  nested CV, Optuna.

Takeaway: a GBDT over a rich, well-calibrated feature bank is the proven winning shape for this data.
The real-time constraint changes *how you compute the features*, not *whether* to use this recipe.

---

## 4. Recommended plan (13 days)

### Day 1 — instrumentation, before any modelling

1. Uncomment the readiness `yield`. Confirm the local tester reproduces the baseline TS-AUC.
2. **Decouple feature extraction from the crunch runner.** Write a plain script that replays the 10k
   training series and dumps a `(series_id, t, features..., label_t)` parquet. Then iterate on models
   in seconds instead of minutes. This is the single highest-leverage engineering decision.
3. Measure: the `w(t)` weight profile; per-step AUC(t) of the baseline; distribution of online lengths
   and of τ; verify `mean(x_hist) ≈ 0, sd(x_hist) ≈ 1`.
4. Diagnose *which break types* the baseline misses by clustering the training series on
   (Δmean, Δvar, Δacf1, ΔKS) across τ. That tells you which detectors to build first.

### Days 2–4 — a bank of O(1) streaming statistics

All two-sided, all constant memory, all at 2–3 forgetting rates (α ∈ {0.01, 0.05, 0.2}) so you cover
both fast detection and long-established breaks:

- **SR mean detector**: `R ← (1+R)·LR` over a grid δ ∈ {±0.25, ±0.5, ±1, ±2}; report `logsumexp`.
- **SR variance detector**: same, scale ratios {0.5, 0.7, 1.5, 2, 3}.
- **Rank / PIT channel** (the pivotality lever): precompute a 1,024-bin quantile lookup of the
  historical segment; map each `x_t → u_t = F_H(x_t)`; run SR / CUSUM on `Φ⁻¹(u_t)` and on `u_t − 0.5`.
  Distribution-free by construction.
- **Tail mass**: SR on `1{|x|>2}`, `1{|x|>3}` against historical rates.
- **Dependence**: EWMA of `x_t·x_{t−k}` for k = 1,2,3 vs historical ACF; and the von Neumann ratio
  `EWMA(|x_t − x_{t−1}|) / EWMA(|x_t|)`, which moves under a dependence change even at constant
  variance.
- **NEWMA**: two EWMAs of an m ≈ 16–32 random-Fourier map of the lag vector `(x_t, x_{t−1}, x_{t−2})`;
  statistic `‖z − z'‖`. Catches shape + dependence changes the moment-based stats miss.
- **e-detector**: a mixture-of-bets supermartingale on `u_t`; emit `log E_t`.
- **Pruned BOCPD** run-length posterior with a Student-t (unknown variance) observation model, top-K
  hypotheses — output `P(r_t < t)` directly. Feed it in as a feature rather than using it alone.

### Days 4–8 — per-series self-calibration

For each statistic, replay it over the break-free historical segment (vectorised numpy — you are
allowed to scan it) with ~20 random restarts to get that series' own null mean / sd / quantiles as a
function of run length. Emit the standardised value or empirical p-value. Budget: ~1–2 ms per stat
per series offline, i.e. a few minutes total for 10k series. **This is where the cross-series AUC
comes from.**

### Days 6–11 — supervised combiner

- Stack calibrated stats + `t` + `len(hist)` + a few historical summaries (kurtosis, ACF1, tail rate)
  into a per-step feature vector. Train LightGBM on `(features, label_t)`.
- ~5M candidate rows; subsample time steps stratified by `t`, and **weight rows by `w(t)/n(t)`** so
  training weight matches the metric.
- **Group-split CV by series id.** Never split within a series.
- **Metric-aligned trick worth trying:** train with LightGBM's ranking objective (`lambdarank` /
  `rank_xendcg`) using **the time step `t` as the query group**. That optimises cross-sectional
  ranking within each step, which is literally TS-AUC, instead of a proxy log-loss.
- Include `t` as a feature — it cannot shift the ranking on its own, but it lets the model learn
  interactions ("at t = 20 trust the fast EWMA, at t = 500 trust the accumulator").
- Cheap experiment: a **ratchet** `s_t ← max(s_t, γ·s_{t−1})`, γ ≈ 0.99. The label is monotone in t, so
  this sometimes helps; but SR already accumulates, so A/B it rather than assuming.

### Days 11–13 — hardening

- Profile inference end-to-end; make sure a full run fits comfortably inside the weekly quota, with
  room for the 10% determinism retest.
- Check determinism: no unseeded RNG (the random Fourier features must use a **fixed seed** saved by
  `train()`), no dict-ordering dependence, no float non-associativity from threading.
- Do model selection on grouped CV over the 10k training series. The 100-series local test set is a
  smoke test only — TS-AUC on 100 series is far too noisy to select on.

### Explicitly not worth your time

- Tuning `KAPPA`, the `tanh`, or the `n_eff` constant — provably zero effect (§1.1).
- Any cross-series normalisation at inference time — impossible in the streaming protocol, and
  useless even if it were possible.
- Chasing the 2025 winners' 2,400-feature counts. Under a streaming constraint, 20–40 well-calibrated
  recursive statistics will beat 2,000 badly-calibrated ones.

---

## 5. Sources

- Competition docs: https://docs.crunchdao.com/competitions/competitions/structural-break-real-time
- Changelog (2026-W23 fixes): https://forum.crunchdao.com/t/2026-w23-structural-break-fixes/1156
- Official quickstarter notebook: https://github.com/crunchdao/quickstarters (structural-break-real-time)
- Veeravalli & Banerjee, Quickest Change Detection — arXiv:1210.5552
- Xie, Zou, Xie, Veeravalli, Sequential (Quickest) Change Detection — arXiv:2104.04186
- Xie, Xie, Moustakides, Computation vs statistical performance — arXiv:2210.05181
- Adams & MacKay, Bayesian Online Changepoint Detection — arXiv:0710.3742
- Keriven, Garreau, Poli, NEWMA — arXiv:1805.08061
- Kalinke et al., Optimal Online Change Detection via Random Fourier Features — arXiv:2505.17789
- Shin & Ramdas, E-detectors — arXiv:2203.03532
- Shekhar & Ramdas, Sequential change detection via backward confidence sequences — arXiv:2302.02544
- Li, Fearnhead, Fryzlewicz, Wang, Automatic change-point detection via deep learning — arXiv:2211.03860
- changeAUC, Model-free Change-Point Detection Using AUC of a Classifier — arXiv:2404.06995
- Mann–Whitney nonparametric CUSUM chart — arXiv:1305.4318
- Signed sequential rank CUSUMs — arXiv:1706.03901
- 2025 2nd place: https://github.com/aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution
- 2025 public solution: https://github.com/gsoisson/adia-structural-break
