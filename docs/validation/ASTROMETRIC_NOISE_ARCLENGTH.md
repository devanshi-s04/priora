# At what arc length does v0's delta-function attributable break?

*The open question `validation/ASTROMETRIC_NOISE.md` leaves and DESIGN §2.7 names: the 45-minute verdict (harmless) is scoped by the arc, not by the astrometry, because the rate error is `sqrt(2) sigma / dt`.  This sweep re-runs the whole WP8 experiment at dt = 60, 45, 30, 15, 10, 5 min and applies a criterion fixed in advance.  Harness: `validation/astrometric_noise.py sweep`.  Noiseless reference: the stored `gate_v0` scores — nothing was re-scored on that side.*

## The criterion (declared before the sweep was run)

v0's delta-function attributable **breaks** at arc length dt if any of three arms trips.  Two further tripwires are reported as **warnings** — they flag a cost the reader should see, not a failure.

- **C1 (calibration)** — the noisy scores FAIL the WP6 gate's own reliability verdict against the true labels: either the global count arm (|Z_tot|, Poisson-binomial) or the binned arm (max|z| over equal-expected-NEO-count bins) has a parametric-bootstrap p-value <= alpha = 0.01. This reuses the gate's own thresholds; nothing new is invented.
- **C2 (triage churn)** — more than 1% of tracklets change side of a triage threshold (p = 0.1/0.3/0.5/0.7) per noise realization. Estimated as the MC mean crossing count over the realizations of the stratified NEO-enriched subsample, divided by its size — an UPPER bound on the full population's rate, since the subsample holds every true NEO and hence essentially every mid-probability tracklet in the catalogue. Cross-checked against the full-catalogue single realization.
- **C3 (lost tracklets)** — more than 1% of tracklets that scored finitely noiseless get an UNDEFINED score under noise (the noisy attributable admits no bound orbit the classifier can integrate). Such a tracklet is not merely mis-scored, it is unscored.
- **W1 (warning, ranking)** — the paired-bootstrap 95% CI on Delta-AUC excludes zero AND |Delta AUC| > 0.0126, the priora-over-digest2 method gap D-14 measured.
- **W2 (warning, operating point)** — completeness at the FIXED WP7 matched-completeness cut (chosen once on noiseless scores) drops by more than 1 percentage point.

C1 is the gate's own reliability test, not a new one: the same equal-expected-NEO-count bins, the same Poisson-binomial z, the same parametric-bootstrap thresholds that `validation/GATE_RESULTS.md` runs, applied to the *noisy* scores against the *true* labels.  A failure there is a property of the data, not a bug.

## Bottom line

**The assumption breaks at dt = 5 min.** Every arc length at or below 5 min trips at least one break arm; the shortest arc that does not is 10 min.

That clean verdict at 10 min has little margin and should not be read as a recommendation: its worst arm's p-value is 0.0130 against alpha = 0.01, i.e. only 1.30x clear of failing (max\|z\| = 3.05 against a bootstrap threshold of 3.05). Read the transition as **somewhere between 10 and 5 minutes**, with 10 min already marginal.

The shortest arc length with comfortable margin (worst arm's p > 0.1) is **15 min**. That, not 10 min, is the number to design a cadence against.

The second result is about the *shape* of the degradation rather than its size.  The rate error grows exactly as 1/dt — that is arithmetic, and the tests assert it — but the score's response to it does **not**: `std(P)` scales as dt^-0.84 at the 90th percentile against the predicted −1, and the per-tracklet 1/dt law departs at dt = 45 min.  The shortfall is traced below to v0's piecewise-constant prior, not to the integration grid, and it means the degradation grows **more slowly** than the naive scaling predicts — the answer is more robust than the algebra, not less.

| dt [min] | rate err / own motion (p50 / p90) | Z_tot (thr) | max\|z\| (thr) | C1 verdict | ΔAUC [95% CI] | churn / realization | lost | completeness @ fixed cut | contamination @ fixed cut | breaks? |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 0.14% / 0.49% | -0.42 (2.56) | 0.71 (2.62) | PASS | +0.00057 [-0.0002, +0.0019] | 0.008% | 0.000% | 87.6% | 5.81% | no |
| 45 | 0.19% / 0.65% | -0.42 (2.56) | 0.72 (2.62) | PASS | +0.00097 [-0.0003, +0.0032] | 0.024% | 0.000% | 87.6% | 5.26% | no |
| 30 | 0.28% / 0.98% | -0.43 (2.57) | 0.72 (2.62) | PASS | +0.00151 [-0.0004, +0.0049] | 0.064% | 0.000% | 87.6% | 5.26% | no |
| 15 | 0.56% / 1.96% | -0.74 (2.49) | 1.56 (3.12) | PASS | +0.00080 [-0.0010, +0.0036] | 0.144% | 0.000% | 87.6% | 5.81% | no |
| 10 | 0.85% / 2.94% | -1.53 (2.40) | 3.05 (3.05) | PASS | +0.00111 [-0.0018, +0.0056] | 0.216% | 0.000% | 87.6% | 8.99% | no |
| 5 | 1.69% / 5.89% | -5.62 (2.47) | 4767.31 (8.52) | **FAIL** | +0.00160 [-0.0034, +0.0084] | 0.400% | 0.000% | 88.1% | 23.83% | **YES** |

*Churn* is the MC mean over 25 realizations of the 500-tracklet NEO-enriched subsample, at its worst threshold, divided by the subsample size — an upper bound on the full population's rate.  *Lost* is the fraction of tracklets that scored finitely noiseless and become undefined under noise.  *Completeness* is at the fixed WP7 cut p >= 0.254 (noiseless: 88.1%).

**dt = 5 min**
- BREAK: C1 calibration: reliability verdict FAIL (p_global = 9.74e-10, p_binned = 0.0005, alpha = 0.01)

### The declared criterion watched the wrong half of the operating point

W2, as declared above, tests the **completeness** at the fixed WP7 cut.  That is the half that does not move: completeness is 87.6% at 60 min and 88.1% at 5 min, because the cut pins it.  The cost lands entirely on **contamination**, which goes from 6.32% noiseless to 23.83% at 5 min — a factor 3.8.  In operational terms the same cut flags 172 tracklets at 60 min and 214 at 5 min for the same number of real NEOs — the extra flags are all false positives arriving at the triage desk.

This is reported as **post-hoc** and enters no verdict: a tripwire invented after seeing the data is not evidence, and the break at 5 min is called by C1 on its own.  But the criterion would have been better written with a contamination arm, and that is a finding about the criterion, so it is recorded here rather than quietly fixed.  Arc lengths whose contamination more than 2x-es: 5 min.

![calibration and triage churn vs arc length](arclength_headline.png)

## What is varied, and what is held fixed

One thing changes across the rows above: the arc length.  The per-detection sigma (Sorcha's model, m5 = 24, FWHM = 700 mas), the catalogue, the labels, the classifier, the scoring grid **and the standard-normal draws themselves** are identical at every arc length.  That last point is what makes the sweep sharp: `noisy_attributable` draws `e0, e1` from `default_rng([seed, stream])` and dt enters only as the divisor in `(d_1 - d_0) / dt`, so every arc length perturbs *the same endpoints by the same offsets* and differs only in the 1/dt amplification of the rates.  The dt-to-dt comparison is therefore paired per tracklet and carries no Monte-Carlo noise of its own.

dt = 45 min is **adopted** from `/home/mjuric/priora/validation/output/astrometric_noise` rather than re-scored: the injected catalogue is verified identical array-by-array and the stored scores are reused, so that row is literally the number published in `validation/ASTROMETRIC_NOISE.md`, not a reproduction of it.

That makes the 45-minute row a cross-check on this harness, and it passes on every point estimate: Z_tot = −0.42, ĉ = 0.925, reliability max\|z\| = 0.72 against a bootstrap threshold of 2.62, AUC 0.98125 → 0.98222, ΔAUC = +0.00097, \|ΔP\| percentiles 5.06e-06 / 7.03e-05 / 7.20e-04, 87.6% completeness at 5.3% contamination, MC std(P) 4.55e-06 / 1.71e-04 / 8.17e-03 — all identical to the published values.

**One published number does not reproduce, and it is not this sweep's doing.** `ASTROMETRIC_NOISE.md` quotes the ΔAUC 95% CI as [−0.00014, +0.00261]; every ΔAUC CI in this report is wider (at 45 min, [−0.00029, +0.00321]). The point estimate is bit-identical, and the CI is produced by `validation.digest2_comparison.auc_bootstrap`, which is outside this harness and was modified while this sweep was running. The CIs here are all from one analysis pass against one version of that function, so they are mutually comparable and the ΔAUC column's conclusion (consistent with zero at every arc length, two orders of magnitude below the 0.0126 method gap) is unaffected — but a reader diffing this row against the published CI should know why it moved. Nothing else in the comparison changed.

| V | SNR | sigma [mas] | rate err @ 60 min | rate err @ 45 min | rate err @ 30 min | rate err @ 15 min | rate err @ 10 min | rate err @ 5 min |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 466.29 | 10.04 | 0.34″/d | 0.45″/d | 0.68″/d | 1.36″/d | 2.04″/d | 4.09″/d |
| 21 | 67.67 | 11.77 | 0.40″/d | 0.53″/d | 0.80″/d | 1.60″/d | 2.40″/d | 4.79″/d |
| 23 | 12.33 | 35.50 | 1.21″/d | 1.61″/d | 2.41″/d | 4.82″/d | 7.23″/d | 14.46″/d |
| 24 | 5.00 | 84.59 | 2.87″/d | 3.83″/d | 5.74″/d | 11.48″/d | 17.23″/d | 34.45″/d |

The right-hand columns are the whole experiment in one table.  The endpoint error is a fixed property of the survey; the arc length alone decides whether the faintest tracklet's rate is uncertain by 2.9 arcsec/day (60 min) or by 34.5 arcsec/day (5 min) per coordinate — the latter being a materially different orbit.

## The rate error against the tracklets' own motion

The classifier does not respond to an absolute rate error; it responds to a rate error *relative to the motion that identifies the object*.  The median `gate_v0` tracklet moves 851 arcsec/day.

| dt [min] | rate error / own motion (p50) | (p90) | (p99) | > 10% of own motion | > 50% | > 100% |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 0.14% | 0.49% | 2.49% | 0.07% | 0.01% | 0.00% |
| 45 | 0.19% | 0.65% | 3.32% | 0.14% | 0.01% | 0.00% |
| 30 | 0.28% | 0.98% | 4.98% | 0.34% | 0.03% | 0.01% |
| 15 | 0.56% | 1.96% | 9.96% | 1.00% | 0.06% | 0.03% |
| 10 | 0.85% | 2.94% | 14.95% | 1.73% | 0.11% | 0.06% |
| 5 | 1.69% | 5.89% | 29.89% | 4.45% | 0.43% | 0.11% |

![rate error vs arc length](rate_error.png)

## Per-tracklet score change

| dt [min] | median \|ΔP\| | p90 | p99 | max | median \|ΔP\| (true NEO) | p99 (true NEO) |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 4.09e-06 | 5.75e-05 | 6.36e-04 | 2.30e-02 | 0.00e+00 | 2.00e-02 |
| 45 | 5.06e-06 | 7.03e-05 | 7.20e-04 | 3.54e-02 | 0.00e+00 | 2.54e-02 |
| 30 | 6.96e-06 | 9.17e-05 | 9.78e-04 | 7.48e-02 | 0.00e+00 | 2.03e-02 |
| 15 | 1.20e-05 | 1.57e-04 | 1.87e-03 | 9.32e-01 | 0.00e+00 | 4.32e-02 |
| 10 | 1.63e-05 | 2.09e-04 | 3.00e-03 | 9.96e-01 | 0.00e+00 | 5.12e-02 |
| 5 | 2.86e-05 | 3.79e-04 | 7.69e-03 | 1.00e+00 | 0.00e+00 | 8.67e-02 |

## Variance scaling: is the response still linear?

DESIGN §2.7's expectation is a locally linear classifier response, in which a rate error proportional to 1/dt gives `std(P) ∝ 1/dt` and therefore `var(P) ∝ 1/dt²`.  That is a *testable* claim here rather than an assumption, because the draws are common across dt.  Two independent forms of the test:

1. **Per-tracklet log-log slopes** of `std(P)` on dt, over the 62 of 500 subsample tracklets whose scatter is non-zero at every arc length: median -0.826 (10th–90th pct -1.125 to -0.272), against the predicted −1.  In variance terms: median -1.651 against the predicted −2.
2. **Aggregate slopes** of the population percentile curves: median -0.824 (r² 0.9952), 90th percentile -0.843 (r² 0.9813), 99th percentile -1.008, max -0.978.

| dt [min] | median std(P) | p90 | p99 | max | frac std > 0.01 | linear-law ratio (p50) | (p10–p90) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 3.83e-06 | 1.42e-04 | 7.13e-03 | 2.05e-02 | 0.8% | 1.000 | 1.000–1.000 |
| 45 | 4.55e-06 | 1.71e-04 | 8.17e-03 | 2.41e-02 | 1.0% | 0.888 | 0.752–1.096 |
| 30 | 6.10e-06 | 2.43e-04 | 1.26e-02 | 3.38e-02 | 1.6% | 0.785 | 0.550–1.483 |
| 15 | 1.06e-05 | 3.70e-04 | 2.41e-02 | 6.95e-02 | 2.2% | 0.666 | 0.334–1.432 |
| 10 | 1.53e-05 | 5.39e-04 | 3.99e-02 | 1.12e-01 | 3.4% | 0.639 | 0.273–1.467 |
| 5 | 2.97e-05 | 1.25e-03 | 8.29e-02 | 2.17e-01 | 4.6% | 0.617 | 0.185–1.418 |

The *linear-law ratio* is the measured `std(P)` divided by the pure-1/dt extrapolation from the 60-minute anchor, per tracklet, over the 62 tracklets whose anchor `std(P)` exceeds 1e-04.  1.000 means the response is still exactly linear; below 1 means it has started to saturate (the posterior is being pushed against 0 or 1, or off the local gradient entirely).

**Is the shortfall real, or is it the measurement?**  Two things could manufacture a ratio below 1 without the classifier being sublinear at all, and both are testable here rather than arguable.

*Candidate 1 — the integration lattice.*  v0 rebuilds the (ρ, ρ̇) admissible region from each attributable and integrates it on a fixed `(1200, 50)` grid, so a perturbation can add or drop whole rings of boundary nodes and move `P` by a discrete jump that owes nothing to the posterior.  That does happen, and at a rate that grows sharply as the arc shortens:

| dt [min] | tracklets whose node count changes | p90 \|Δ nodes\| | p90 relative | median \|ΔP\| |
|---:|---:|---:|---:|---:|
| 60 | 17.9% | 50 of 39300 | 0.12% | 4.09e-06 |
| 45 | 22.3% | 50 of 39300 | 0.13% | 5.06e-06 |
| 30 | 30.0% | 50 of 39300 | 0.13% | 6.96e-06 |
| 15 | 46.1% | 100 of 39300 | 0.25% | 1.20e-05 |
| 10 | 56.3% | 150 of 39300 | 0.36% | 1.63e-05 |
| 5 | 72.5% | 300 of 39300 | 0.69% | 2.86e-05 |

It is nevertheless **not** the cause.  Restricting the whole test to the 5482 of 20000 tracklets (27%) whose node count is identical to the noiseless run at *every* arc length — switching the lattice mechanism off entirely — leaves the step ratios where they were (last column of the step table below).

*Candidate 2 — small-amplitude roughness.*  A tracklet whose real response is at the roughness scale of the model measures the roughness, not the posterior, and its ratio is biased toward `dt/dt_ref`.  If that were driving the shortfall, excluding such tracklets would move the answer, so the identical test is re-run at a ladder of anchor floors:

| anchor floor on std(P) | tracklets | 60 min | 45 min | 30 min | 15 min | 10 min | 5 min |
|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 383 | 1.000 | 0.929 | 0.879 | 0.743 | 0.703 | 0.675 |
| 1e-06 | 328 | 1.000 | 0.923 | 0.866 | 0.704 | 0.651 | 0.625 |
| 1e-05 | 185 | 1.000 | 0.905 | 0.802 | 0.628 | 0.570 | 0.515 |
| 1e-04 | 62 | 1.000 | 0.888 | 0.785 | 0.666 | 0.639 | 0.617 |
| 1e-03 | 24 | 1.000 | 0.932 | 0.882 | 0.766 | 0.711 | 0.721 |
| 1e-02 | 4 | 1.000 | 0.899 | 0.790 | 0.693 | 0.596 | 0.464 |

Median linear-law ratio; 1.000 is exact linearity.

Across the ladder rows holding at least 20 tracklets (floors none, 1e-06, 1e-05, 1e-04, 1e-03), the median ratio moves by at most **0.205** at any arc length: 60 min 1.00–1.00, 45 min 0.89–0.93, 30 min 0.78–0.88, 15 min 0.63–0.77, 10 min 0.57–0.71, 5 min 0.52–0.72.

Both halves of that answer matter and they point different ways. The ratio **does** drift with the floor (up to 0.205), so part of the measured shortfall is an amplitude-dependent artefact and the low-floor rows overstate it. But it does not vanish: taking the single most favourable floor at each arc length, the 5-minute ratio is still 0.721 — far outside 1 ± 0.1. A real sublinear component survives every control applied here, and it is that component the rest of this section is about.

An anchored ratio cannot say *where* a departure happened, because every row is measured against the same longest arc.  The anchor-free version compares only adjacent arc lengths, each pair anchored on its own longer member.  A **constant** step ratio below 1 means the response left the power law once and has been self-similar since; step ratios that **march downward** mean progressive saturation.

| arc pair | factor | median step ratio, std(P) (n) | median step ratio, paired \|ΔP\| (n) | ...restricted to fixed-node tracklets (n) |
|---|---:|---:|---:|---:|
| 60 → 45 min | 1.33x | 0.888 (62) | 0.844 (1231) | 0.829 (245) |
| 45 → 30 min | 1.50x | 0.869 (70) | 0.798 (1444) | 0.792 (295) |
| 30 → 15 min | 2.00x | 0.840 (80) | 0.724 (1882) | 0.715 (399) |
| 15 → 10 min | 1.50x | 0.933 (104) | 0.843 (2779) | 0.801 (621) |
| 10 → 5 min | 2.00x | 0.916 (129) | 0.761 (3523) | 0.669 (779) |

The last column is the control described above: the lattice mechanism is switched off and the answer does not move.

One failure mode is invisible to every ratio and slope above and is therefore counted separately.  117 of 500 subsample tracklets have `std(P)` numerically **zero** at the 60-minute anchor — the classifier is completely certain about them, and a ratio to zero has no value.  Of those, 0 move at all at 45 min (0 by more than 0.01), 0 move at all at 30 min (0 by more than 0.01), 1 move at all at 15 min (0 by more than 0.01), 1 move at all at 10 min (0 by more than 0.01), 1 move at all at 5 min (0 by more than 0.01).

The verdict below is read off the **cleanest adequately-populated** row of that ladder — anchor floor 1e-04, 62 tracklets (the largest floor still holding 30) — rather than off a floor chosen to suit the answer.

**Verdict: the 1/dt law departs at dt = 45 min** (median ratio 0.888 on that subset, outside 1 ± 0.1). Above that arc length the response is linear and the perturbation is a zero-mean, correctly-modelled Gaussian; at and below it the response has left the linear regime, which is exactly where an outer marginalization stops being a refinement and starts being a correction.

**What the surviving sublinearity is, and why it matters less than it looks.** The lattice control does not move it and no anchor floor removes it, so a real component is a property of v0's response. The mechanism is visible in *which* tracklets are sublinear: the aggregate slopes above are -0.82 at the median but -1.01 at the 99th percentile and -0.98 at the maximum — the tracklets that actually move scale essentially as 1/dt exactly, and it is the typical, barely-responding tracklet that is sublinear. That is the signature of a **piecewise-constant** prior: `Digest2Histogram` bins (q, e, i, H) into 45,936 cells (DESIGN §3), so the likelihood is a step function of the orbit and the score's response to an attributable perturbation is a staircase, not a gradient. Marginalizing over the ~39300 quadrature nodes in a typical admissible region averages that many staircases — enough to look smooth for a tracklet whose perturbation crosses many bin walls, not enough for one that crosses few. Two observations fit only this explanation: the median |ΔP| for true NEOs at 60 min is 0.00e+00 — **exactly zero**, which a smooth response can never produce and a step function produces whenever the perturbation stays inside its bins; and 117 of 500 subsample tracklets have identically zero MC scatter at the anchor arc.

Three consequences worth stating plainly. **(i)** The degradation grows more slowly than the rate error, so the short-arc answer is *better* than the 1/dt algebra predicts — the extrapolation in `ASTROMETRIC_NOISE.md`'s "where this could break" was conservative. **(ii)** The tail is nevertheless linear, so the tracklets that can actually cross a triage threshold do degrade at the full 1/dt rate, and the C2 arm above is the number to trust rather than the median score change. **(iii)** "The linear regime", which is the regime a Gaussian covariance marginalization models correctly, is a statement about the responsive tail and about population aggregates — not a per-tracklet fact. On this axis v1 owes a smoother prior at least as much as it owes an outer marginalization; marginalizing a staircase with a Gaussian is a well-posed operation but it is not the operation the roughness calls for.

The same test on the full catalogue's single paired realization — sharper still, because `|ΔP| · dt` is a *deterministic* per-tracklet quantity there rather than an estimated scatter (n = 1231 tracklets above the 1e-04 anchor floor):

| dt [min] | 60 | 45 | 30 | 15 | 10 | 5 |
|---|---|---|---|---|---|---|
| median \|ΔP\|·dt / anchor | 1.000 | 0.844 | 0.670 | 0.456 | 0.384 | 0.275 |
| 10th pct | 1.000 | 0.428 | 0.187 | 0.087 | 0.076 | 0.042 |
| 90th pct | 1.000 | 1.255 | 1.322 | 1.286 | 1.262 | 1.282 |

...and its own floor ladder, for the same reason:

| anchor floor on \|ΔP\| | tracklets | 60 min | 45 min | 30 min | 15 min | 10 min | 5 min |
|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 19882 | 1.000 | 0.928 | 0.873 | 0.776 | 0.718 | 0.636 |
| 1e-06 | 14857 | 1.000 | 0.900 | 0.812 | 0.673 | 0.604 | 0.499 |
| 1e-05 | 6668 | 1.000 | 0.873 | 0.733 | 0.539 | 0.447 | 0.351 |
| 1e-04 | 1231 | 1.000 | 0.844 | 0.670 | 0.456 | 0.384 | 0.275 |
| 1e-03 | 118 | 1.000 | 0.936 | 0.783 | 0.495 | 0.434 | 0.378 |
| 1e-02 | 7 | 1.000 | 0.984 | 0.824 | 0.723 | 0.709 | 0.569 |

![std(P) scaling](std_scaling.png)

## Which tracklets move, at the shortest arc (5 min)

Only a tracklet near a triage threshold can be moved across one, so the triage-relevant slice of the sensitivity map is `std(P)` against the *noiseless* score.  Compared side by side with the longest arc tested, so the growth is visible per score bin rather than only in the population aggregate.

| noiseless P(NEO) | n | median std @ 60 min | median std @ 5 min | p90 std @ 5 min | growth (median) |
|---|---:|---:|---:|---:|---:|
| 0–0.0001 | 69 | 6.52e-07 | 7.65e-06 | 5.31e-05 | 11.7x |
| 0.0001–0.01 | 249 | 1.01e-05 | 7.05e-05 | 4.76e-04 | 7.0x |
| 0.01–0.1 | 12 | 1.73e-04 | 1.24e-03 | 3.21e-03 | 7.2x |
| 0.1–0.3 | 9 | 5.81e-03 | 7.36e-02 | 8.39e-02 | 12.7x |
| 0.3–0.5 | 4 | 6.48e-03 | 4.33e-02 | 9.87e-02 | 6.7x |
| 0.5–0.7 | 6 | 2.14e-03 | 1.46e-02 | 2.80e-02 | 6.8x |
| 0.7–0.9 | 7 | 1.02e-03 | 1.16e-02 | 1.09e-01 | 11.4x |
| 0.9–1 | 144 | 0.00e+00 | 0.00e+00 | 2.59e-04 | n/a |

Fraction of subsample tracklets with std(P) > 0.01: 0.8% at 60 min, 4.6% at 5 min; > 0.10: 0.0% and 0.8%.  Mean over realizations minus the noiseless score (a nonlinear response shows up here as a systematic offset): +2.76e-04 at 5 min, against -2.54e-05 at 60 min.

`ASTROMETRIC_NOISE.md`'s second caveat — "the fractional rate error is worst for the slowest objects" — is confirmed directly here, and in the worst possible way.  2 of the five most sensitive tracklets at 5 min are near-stationary non-NEOs (rate ~0.028 deg/day) whose noiseless score is 0.0000 — the classifier is *certain* they are not NEOs — and whose noisy score has mean 0.0700 and scatter 0.1826.  A 14 arcsec/day rate error on a 100 arcsec/day object is not a perturbation of its orbit, it is a different orbit, and the manufactured NEO probability is where the C1 failure comes from: these tracklets are what inflates the expected NEO count from 187 to 221 against 185 real ones.

The most sensitive tracklets at 5 min:

| gen index | NEO? | V | rate [deg/day] | sigma [mas] | P noiseless | P mean | std(P) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8065 | yes | 23.57 | 0.109 | 58.4 | 0.7864 | 0.7371 | 0.2167 |
| 6325 | no | 23.11 | 0.028 | 38.9 | 0.0000 | 0.0700 | 0.1826 |
| 747 | no | 22.18 | 0.028 | 19.3 | 0.0000 | 0.0361 | 0.1296 |
| 1910 | yes | 23.91 | 0.745 | 96.0 | 0.3865 | 0.3793 | 0.1123 |
| 1457 | yes | 23.86 | 0.708 | 90.9 | 0.2536 | 0.1818 | 0.0882 |
| 14718 | no | 23.36 | 0.268 | 50.7 | 0.1039 | 0.1361 | 0.0828 |
| 4309 | yes | 23.91 | 0.648 | 92.8 | 0.1952 | 0.2448 | 0.0808 |
| 10731 | yes | 23.94 | 0.748 | 98.4 | 0.1486 | 0.1406 | 0.0770 |
| 7115 | yes | 23.61 | 0.710 | 72.3 | 0.2858 | 0.2977 | 0.0736 |
| 14521 | yes | 23.91 | 0.338 | 83.8 | 0.4182 | 0.4814 | 0.0668 |

## Positions vs. rates, arc length by arc length

The position error does not depend on dt at all; only the rate error does.  The share of the induced score variance carried by the rate term is therefore a direct read on whether shortening the arc is doing what the algebra says.

| dt [min] | rate share of var(ΔP) | position share | corr(both, rate) | linearity residual RMS | p90 \|ΔP\| rate-only | p90 \|ΔP\| pos-only |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 100.0% | 0.0% | 1.000 | 1.87e-06 | 1.16e-04 | 1.22e-08 |
| 45 | 100.0% | 0.0% | 1.000 | 1.95e-06 | 1.59e-04 | 1.22e-08 |
| 30 | 100.0% | 0.0% | 1.000 | 1.88e-06 | 1.56e-04 | 1.22e-08 |
| 15 | 100.0% | 0.0% | 1.000 | 2.15e-06 | 2.92e-04 | 1.22e-08 |
| 10 | 99.8% | 0.0% | 1.000 | 1.46e-05 | 3.68e-04 | 1.22e-08 |
| 5 | 100.0% | 0.0% | 1.000 | 1.87e-06 | 5.76e-04 | 1.22e-08 |

The position-only column is the control: it is (up to the shared draws) the same number at every arc length, because a rigid translation of the tracklet is dt-independent by construction.

## Triage thresholds

| dt [min] | worst threshold | MC mean crossings / realization | worst realization | mean NEOs lost / realization | full-catalogue crossings (20000) |
|---:|---:|---:|---:|---:|---:|
| 60 | p >= 0.1 | 0.04 | 1 | 0.00 | 0 (at p >= 0.1) |
| 45 | p >= 0.1 | 0.12 | 1 | 0.00 | 0 (at p >= 0.1) |
| 30 | p >= 0.1 | 0.32 | 2 | 0.00 | 0 (at p >= 0.1) |
| 15 | p >= 0.1 | 0.72 | 2 | 0.04 | 5 (at p >= 0.1) |
| 10 | p >= 0.1 | 1.08 | 3 | 0.32 | 15 (at p >= 0.1) |
| 5 | p >= 0.1 | 2.00 | 4 | 0.88 | 59 (at p >= 0.1) |

## Calibration and the operating point

| dt [min] | ĉ [95%] | Z_tot (p) | reliability max\|z\| (thr) | verdict | Brier | ECE | completeness | contamination |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 0.925 [0.647, 1.324] | -0.42 (0.751) | 0.71 (2.62) | PASS | 0.00140 | 1.19e-04 | 87.6% | 5.8% |
| 45 | 0.925 [0.647, 1.324] | -0.42 (0.752) | 0.72 (2.62) | PASS | 0.00140 | 1.19e-04 | 87.6% | 5.3% |
| 30 | 0.924 [0.646, 1.322] | -0.43 (0.744) | 0.72 (2.62) | PASS | 0.00140 | 1.22e-04 | 87.6% | 5.3% |
| 15 | 0.873 [0.609, 1.253] | -0.74 (0.525) | 1.56 (3.12) | PASS | 0.00145 | 2.10e-04 | 87.6% | 5.8% |
| 10 | 0.749 [0.517, 1.084] | -1.53 (0.141) | 3.05 (3.05) | PASS | 0.00160 | 4.49e-04 | 87.6% | 9.0% |
| 5 | 0.317 [0.210, 0.479] | -5.62 (0.000) | 4767.31 (8.52) | **FAIL** | 0.00260 | 1.81e-03 | 88.1% | 23.8% |

![ranking and operating point](ranking_operating_point.png)

At N = 2×10⁴ (≈185 NEOs) the fitted odds scale is only constrained to roughly ±35%, so the ĉ column says "no detectable miscalibration at the ±35% level", not "identical".  The paired per-tracklet ΔP columns above are the sharper statement, and the C1 arm is deliberately the gate's own bootstrap-thresholded test rather than an eyeball comparison of ĉ.

## Assumptions

- Photometric noise is NOT injected: V is carried through unchanged. The measured degradation is therefore the astrometric contribution alone; photometric scatter would additionally smear H(V, rho).
- Two detections per tracklet at each arc length, exactly linear motion. The arc length is the ONLY thing varied: the endpoint sigma, the standard-normal draws, the catalogue, the labels and the classifier are all held fixed, so every dt-to-dt difference below is caused by the 1/dt rate-error amplification and by nothing else.
- Common random numbers across dt: at fixed (seed, stream, tracklet) every arc length perturbs the same endpoints by the same offsets, so |Delta P| at two arc lengths is a PAIRED comparison and the dt-to-dt curve carries no Monte-Carlo noise of its own. It also means the sweep measures the response to one particular noise realization per arc length; the MC block supplies the ensemble spread.
- A shorter arc is modelled as a shorter arc, not as a worse survey: sigma per detection is unchanged (Sorcha's model at m5 = 24, FWHM = 700 mas). Real short pairs inside one visit pair would carry the same per-detection sigma, so this is the right comparison; a pair of consecutive 30 s exposures would additionally differ in nothing but dt.
- dt = 45 min is ADOPTED from the WP8 headline run (identical injected catalogue, verified array-by-array) rather than re-scored, so that row is literally the published number and not a reproduction of it.
- The attributable is rebuilt by finite differencing the two noisy endpoints, which is the maximum-likelihood 2-point estimate. A least-squares fit to >2 detections is the same estimator with a smaller variance.
- One seeing/depth pair for the whole catalogue (0.7 arcsec, m5 = 24.0); Sorcha draws both per visit from an opsim database. Real Rubin visits spread around these values, which would widen the sigma distribution without moving its centre much.
- sigma is applied in the local sky metric with cos(delta) evaluated once at the attributable's declination; a <1% metric error for all but a handful of >5 deg/day tracklets, and the price of an exactly additive position/rate decomposition.
- No light-time, aberration or field-distortion terms: the same geometric treatment v0's classifier itself uses.
- The population is the WP6 self-consistency catalogue (digest2's own model, All variant, 14 <= V < 24, geocentre). Calibration statements are therefore about the prior-matched case; DESIGN 4.3's prior-mismatch stress test is a separate and larger effect.

## Reproduce

```
OMP_NUM_THREADS=1 PRIORA_BACKEND=python \
  python3 -m validation.astrometric_noise sweep --workers 84
```

Seed 20260902; source catalogue `/home/mjuric/priora/validation/output/gate_v0` (read-only); 6 arc lengths x (1 x 20k full-catalogue realization + 25 x 500 MC + 2 x 500 decomposition). Every stage is resumable and a pure function of its inputs.
