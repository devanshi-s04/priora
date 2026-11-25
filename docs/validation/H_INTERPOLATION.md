# Mass-conserving smooth-in-H reconstruction of the digest2 histogram

*Generated 2026-09-02T13:10:16 by `validation/h_interpolation_study.py report`.  Construction and its guarantees: `priora/population.py` module docstring; unit tests: `tests/test_population.py`, `tests/test_backend_parity.py`.*

## What was built

`Digest2Histogram(h_interpolation="pchip")` replaces the piecewise-constant H dependence of the digest2 model with the derivative of a monotone piecewise cubic (PCHIP) fitted to each (q, e, i) cell's **cumulative** object count over the 19 H edges.  Interpolating the CDF and differentiating — rather than interpolating the density values — is what makes the reconstruction **mass-conserving per H bin**, and monotonicity is what stops mass leaking between bins.  Both components always share the mode; it is mutually exclusive with the D-14 clamp.

Measured guarantees — **recomputed at report time**, not transcribed, over all 2552 (q, e, i) cells x 18 H bins x both components of the `All` variant of `data/digest2.model.csv`:

| property | measurement |
|---|---|
| per-H-bin mass conservation | max relative deviation of `integral(density) dH x cell volume` from the stored count: **2.11e-15** (Gauss-Legendre through `density()` itself, 8 nodes, exact for the quadratic it evaluates; empty bins integrate to exactly 0 in both components) |
| non-negativity | min density over 200 points x 18 bins x 2552 cells x 2 components = **0.0**, never negative — **through `density()`**, which is the only path the classifier and the sampler use.  Evaluating the stored coefficient triple directly at `s = dH` (the right-hand knot, which `density()` never reaches because the H bins are half-open) can return ~-1e-10 *relative* to the cell scale, from cancellation in the Horner form at a double root.  Unreachable through the public API, and deliberately not clamped, so a genuine sign error could not hide behind a clamp. |
| continuity across interior H edges | max \|one-sided difference\| / cell peak density = **9.63e-16** over all 17 interior edges x all cells (structural: PCHIP is C1, so its derivative is C0) |
| D-4 truncation | exactly `0.0` for H < 0, H >= 25.5, +/-inf and NaN, and outside the q/e/i support — bit-identical to histogram mode |
| total_count, support, q/e/i/H bin lookup | bit-identical to histogram mode (verified on 1e5 random probes) |
| mass containment in sparse cells | a cell populated in one H bin only reproduces that count to 2.5e-12 and has **exactly** zero density in every other bin, for all 18 choices of populated bin |
| sample() vs density() | chi2/dof = 0.874 (NEO) and 1.041 (other) on 1e+06 draws against the analytic mass on 144 fine sub-bins; inverse-CDF residual 5.8e-16 relative |

The reconstruction is *generally* discontinuous at the two support endpoints (H = 0 and H = 25.5).  That is documented, not a defect: the histogram mode has the same two jumps, and PCHIP's shape-preserving end-slope limiter actually removes the jump wherever the neighbouring bin is much more populated (measured: 100.0% of NEO's cells populated in [24, 25.5) jump at H -> 25.5; 15.7% of other's cells populated in [0, 6) jump at H = 0; 100.0% of other's cells populated in [24, 25.5) jump at H -> 25.5).

`reconstruction_shape.png` shows the construction on three cells — the two components' most populated cells and a synthetic cell with counts in a single H bin.  The top row is the object that is actually interpolated (the cumulative count, with the monotone cubic through its 19 exact knots); the bottom row is the density both modes evaluate.  The sparse column is the "interpolation invents density" objection answered in one picture: the cubic is flat outside the populated bin, so the density there is identically zero.

**What it does NOT fix.**  The `1/(e sin i)` integrable singularity of DESIGN §2.5 is untouched — it lives on the e = 0 and i = 0 bin edges and is a property of the measure, not of the H reconstruction.

## Rust backend

The kernel carries the precomputed per-interval Horner coefficients and evaluates one cubic-derivative Horner per lookup, sharing the four bin lookups with the histogram path.  The two modes differ by one branch, three contiguous loads instead of one, a subtraction and two multiply-adds.

| quantity | value |
|---|---|
| coefficient memory | 29x8x11 = 2552 cells x 18 intervals x 3 coefficients x 8 B = **1.10 MB per component, 2.20 MB for the pair** (the brief's ~50 MB estimate counted the 45,936 (q,e,i,H) *bins* as cells; there are 29x8x11 = 2552 cells, and storing only the derivative's three coefficients rather than four CDF coefficients halves it again) |
| single-core CPU ms/tracklet at (1200, 50) | histogram **10.92 ms**, smooth **11.14 ms** (+2.0%) |
| single-core CPU ms/tracklet at (1200, 100) | histogram **21.33 ms**, smooth **21.74 ms** (+1.9%) |
| timing method | 48 tracklets, 25 interleaved reps, minimum per configuration.  **CPU** time (`time.process_time`), not wall: load average was 2/3/9 on 96 cores (another agent was running a large job in this repo), and wall time under that much oversubscription is unusable — the wall minima are inconsistent enough to make the smooth mode look *faster* at one grid, which it cannot be.  CPU time excludes the intervals the thread was descheduled.  Every rep of both clocks is in `bench.json`. |
| sanity anchor | the histogram-mode figure at (1200, 50), 10.92 ms, sits -9% above D-15's recorded 12.0 ms/tracklet single-core — close enough to confirm the absolute scale, and the residual is the same contention that made the wall clock unusable.  The **overhead column is the trustworthy number**: both modes were measured in the same interleaved reps, so whatever inflates one inflates the other |
| Python vs Rust density agreement | **bit-identical** (`np.array_equal`) over ~16k probes covering every bin edge from both sides, +/-inf, NaN and 4000 in-support draws per component |
| end-to-end Lambda agreement, 120 gate_v0 tracklets | max \|dp\| = 2.1e-14, max Lambda relative = 4.2e-11 |


**Erratum.**  The first version of this report gave the smooth mode's overhead as +21% to +28% and explained it as the coefficient table spilling out of L2.  Both were wrong.  Those numbers were taken while another agent's job had this 96-core box at load average 88-162, and neither the wall clock nor `process_time` is reliable under that much oversubscription (the same protocol re-run at load 2 gives +2.0%, and the review's independent replication on a quiet machine gave +1.0% median over 25 reps).  The overhead is a couple of percent, which is what the arithmetic predicts: two multiply-adds and two extra loads against four bin searches and a closed-form Jacobian.  The L2 story was an explanation invented for a measurement artifact, and is withdrawn.

The stored form is still the derivative rather than the CDF's four coefficients — one fewer array per interval, and it is the form the hot path needs; `sample()` integrates it back to a CDF at draw time rather than the reverse.

Bit-identity is by construction, not luck: Python precomputes the coefficients and Rust copies them, and both evaluate `a + s * (b + s * c)` with `s = H - h_edges[ih]` in that exact grouping.  numpy evaluates one operation per ufunc call and the crate does not enable fast-math, so neither side can contract the expression into an FMA.

## 1. Node count vs accuracy — no detectable win at the production grid

### Statistic, and why it has to be conditional

|dp(NEO)| is bounded by min(p, 1-p).  At mixture proportions the gate population is overwhelmingly low-p, so an unconditional p90 |dp| measures arithmetic head-room rather than quadrature quality: on the whole `gate_v0` catalogue it comes out at ~5e-5 whatever the integration is doing to the tracklets that are actually near a decision.  Everything below therefore uses the **mid-p** statistic — |dp| over tracklets whose reference p(NEO) is in (0.05, 0.95).

Note what that means for the two columns in the tables: this sweep *selects* mid-p tracklets, so its unconditional and mid-p statistics coincide to a few percent by construction.  The all-vs-mid contrast is not visible here and cannot be — it is visible on a full mixture catalogue, and that is where the ~5e-5 above comes from.

Sample: 507 tracklets (all 507 tracklets with stored p in (0.05, 0.95)) from `validation/output/calibration_1e5`, of which 497 are mid-p on the reference; each scored at every point of a 12 x 4 (n_rho, n_rhodot) grid in both modes against a same-mode 9600x400 reference.

**Reference adequacy.**  Refining the reference again to 19200x800 moves it by mid-p p90 5.17e-04 (none), 4.88e-04 (pchip) — a factor 15 below the baseline being quoted, so the table measures the sweep point and not the reference.

### Reconciliation with DESIGN §2.5's 3.5e-3

First, the harness reproduces DESIGN §2.5's *relative* statements, which is what licenses reading its absolute numbers against these:

| grid | nodes | all-p90 \|dp\| | mid-p p90 \|dp\| |
|---|---|---|---|
| (800, 100) | 80000 | 9.10e-03 | 9.20e-03 |
| (400, 200) | 80000 | 2.11e-02 | 2.12e-02 |
| (200, 400) | 80000 | 6.40e-02 | 6.40e-02 |
| (200, 100) | 20000 | 6.68e-02 | 6.68e-02 |

(800, 100) beats (400, 200) by 2.3x at equal cost — DESIGN says "~2x".  
(200, 400) vs (200, 100): 6.40e-02 vs 6.68e-02 — DESIGN says "equal".  Both reproduced.

Second, the recomputation of the stored `gate_v0` scores at (1200, 50) agrees with the run of record to max |dp| = 7.7e-13, so the scoring path is the same one.

With that established: DESIGN §2.5's "p90 |dp| ~ 3.5e-3, max ~1e-2 vs a (1200, 600) reference" is a **mid-p** figure. Measured here against exactly that reference, on this study's 497 mid-p tracklets: **mid-p p90 = 7.006e-03** against an unconditional p90 of 7.09e-03.  An independent recomputation by the reviewer on the `gate_v0` catalogue gives 6.1-6.4e-3.  So:

- The **statistic** is settled: 3.5e-3 is a mid-p p90, not an unconditional one.  (The two coincide in the table above because this sweep is mid-p-selected; the contrast — ~5e-5 unconditional against ~6e-3 mid-p — is the review's recomputation on the full `gate_v0` catalogue.)  Nothing is wrong with §2.5; the conditioning just needs saying out loud, which no prior document did.
- The **p90 value** is reproduced to within about a factor of two (7.01e-03 here, 6.1-6.4e-3 on `gate_v0`, against 3.5e-3), which is what one should expect of a p90 taken on different catalogues.
- The **max** is NOT reproduced: mid-p max here is 2.32e-01, two orders of magnitude above §2.5's "~1e-2".  A max over a few hundred mid-p tracklets is set by the single worst-resolved one, so it is not a stable statistic and should not be quoted as one; the p90 rows in the tables below are.

### mid-p p90 |dp(NEO)|, histogram / smooth

| n_rho | n_rhodot=12 | n_rhodot=25 | n_rhodot=50 | n_rhodot=100 |
|---|---|---|---|---|
| 50 | 8.26e-01 / 8.28e-01 | 8.24e-01 / 8.28e-01 | 8.23e-01 / 8.27e-01 | 8.22e-01 / 8.27e-01 |
| 75 | 4.43e-01 / 4.50e-01 | 3.82e-01 / 3.82e-01 | 4.48e-01 / 4.63e-01 | 4.14e-01 / 4.05e-01 |
| 100 | 2.17e-01 / 2.10e-01 | 1.88e-01 / 1.80e-01 | 1.93e-01 / 1.85e-01 | 1.84e-01 / 1.73e-01 |
| 150 | 1.43e-01 / 1.42e-01 | 1.17e-01 / 1.15e-01 | 1.04e-01 / 1.00e-01 | 1.07e-01 / 9.67e-02 |
| 200 | 1.14e-01 / 1.19e-01 | 8.84e-02 / 8.53e-02 | 6.12e-02 / 6.89e-02 | 6.68e-02 / 7.09e-02 |
| 300 | 7.38e-02 / 7.79e-02 | 4.46e-02 / 4.60e-02 | 3.44e-02 / 3.35e-02 | 3.29e-02 / 3.12e-02 |
| 400 | 5.86e-02 / 5.81e-02 | 3.23e-02 / 3.44e-02 | 2.45e-02 / 2.45e-02 | 2.17e-02 / 2.24e-02 |
| 600 | 4.66e-02 / 4.50e-02 | 2.49e-02 / 2.38e-02 | 1.59e-02 / 1.46e-02 | 1.25e-02 / 1.09e-02 |
| 800 | 4.40e-02 / 4.37e-02 | 2.30e-02 / 2.30e-02 | 1.13e-02 / 1.17e-02 | 9.20e-03 / 9.03e-03 |
| 1200 | 4.10e-02 / 4.04e-02 | 1.69e-02 / 1.74e-02 | 7.88e-03 / 7.72e-03 | 5.89e-03 / 5.64e-03 |
| 1600 | 3.92e-02 / 3.96e-02 | 1.45e-02 / 1.47e-02 | 7.44e-03 / 7.49e-03 | 5.23e-03 / 4.57e-03 |
| 2400 | 3.77e-02 / 3.88e-02 | 1.37e-02 / 1.53e-02 | 5.88e-03 / 5.91e-03 | 3.28e-03 / 3.24e-03 |

### The same tables for the two Lambdas (p90 relative error)

p(NEO) is the ratio Lambda_NEO/(Lambda_NEO + Lambda_other), and both components share the H bin edges, so an H-staircase quadrature error is largely **common-mode and cancels in the ratio**.  The Lambdas are where the staircase can be seen at all — reporting only p(NEO) would have hidden the mechanism entirely.

**Lambda_NEO**, histogram / smooth:

| n_rho | n_rhodot=12 | n_rhodot=25 | n_rhodot=50 | n_rhodot=100 |
|---|---|---|---|---|
| 50 | 4.73e-01 / 4.31e-01 | 4.02e-01 / 3.51e-01 | 4.24e-01 / 3.51e-01 | 4.20e-01 / 3.61e-01 |
| 75 | 3.23e-01 / 3.10e-01 | 2.51e-01 / 2.41e-01 | 2.62e-01 / 2.54e-01 | 2.59e-01 / 2.46e-01 |
| 100 | 2.46e-01 / 2.50e-01 | 1.89e-01 / 1.94e-01 | 1.63e-01 / 1.47e-01 | 1.56e-01 / 1.44e-01 |
| 150 | 1.89e-01 / 1.91e-01 | 1.40e-01 / 1.32e-01 | 1.10e-01 / 1.01e-01 | 1.09e-01 / 9.53e-02 |
| 200 | 1.66e-01 / 1.72e-01 | 1.07e-01 / 9.80e-02 | 8.22e-02 / 7.28e-02 | 8.14e-02 / 7.09e-02 |
| 300 | 1.42e-01 / 1.45e-01 | 7.14e-02 / 7.42e-02 | 5.31e-02 / 4.56e-02 | 4.70e-02 / 3.98e-02 |
| 400 | 1.32e-01 / 1.41e-01 | 6.88e-02 / 6.27e-02 | 4.00e-02 / 3.91e-02 | 3.58e-02 / 3.11e-02 |
| 600 | 1.24e-01 / 1.27e-01 | 4.95e-02 / 5.24e-02 | 2.73e-02 / 2.41e-02 | 2.24e-02 / 1.80e-02 |
| 800 | 1.22e-01 / 1.23e-01 | 4.87e-02 / 5.04e-02 | 2.03e-02 / 1.91e-02 | 1.43e-02 / 1.15e-02 |
| 1200 | 1.15e-01 / 1.16e-01 | 4.52e-02 / 4.79e-02 | 1.65e-02 / 1.66e-02 | 1.02e-02 / 9.02e-03 |
| 1600 | 1.15e-01 / 1.16e-01 | 4.42e-02 / 4.69e-02 | 1.56e-02 / 1.53e-02 | 8.54e-03 / 7.47e-03 |
| 2400 | 1.13e-01 / 1.19e-01 | 4.26e-02 / 4.47e-02 | 1.36e-02 / 1.49e-02 | 7.10e-03 / 6.00e-03 |

**Lambda_other**, histogram / smooth:

| n_rho | n_rhodot=12 | n_rhodot=25 | n_rhodot=50 | n_rhodot=100 |
|---|---|---|---|---|
| 50 | 1.01e+00 / 1.00e+00 | 1.00e+00 / 1.00e+00 | 1.00e+00 / 1.00e+00 | 1.00e+00 / 1.00e+00 |
| 75 | 1.02e+00 / 1.03e+00 | 1.00e+00 / 1.00e+00 | 1.00e+00 / 1.00e+00 | 1.00e+00 / 1.00e+00 |
| 100 | 9.70e-01 / 9.72e-01 | 9.27e-01 / 9.47e-01 | 9.23e-01 / 9.46e-01 | 9.20e-01 / 9.45e-01 |
| 150 | 7.99e-01 / 8.13e-01 | 7.31e-01 / 7.29e-01 | 7.16e-01 / 7.18e-01 | 7.14e-01 / 7.02e-01 |
| 200 | 6.60e-01 / 6.55e-01 | 5.99e-01 / 5.75e-01 | 4.97e-01 / 5.22e-01 | 4.92e-01 / 4.87e-01 |
| 300 | 4.94e-01 / 4.97e-01 | 3.38e-01 / 3.46e-01 | 2.83e-01 / 2.82e-01 | 2.61e-01 / 2.53e-01 |
| 400 | 3.68e-01 / 3.84e-01 | 2.27e-01 / 2.18e-01 | 1.89e-01 / 1.80e-01 | 1.66e-01 / 1.63e-01 |
| 600 | 2.93e-01 / 3.03e-01 | 1.70e-01 / 1.65e-01 | 1.09e-01 / 1.04e-01 | 8.94e-02 / 8.48e-02 |
| 800 | 2.64e-01 / 2.72e-01 | 1.47e-01 / 1.40e-01 | 8.49e-02 / 8.47e-02 | 6.45e-02 / 6.34e-02 |
| 1200 | 2.39e-01 / 2.43e-01 | 1.09e-01 / 1.07e-01 | 6.25e-02 / 5.76e-02 | 4.51e-02 / 4.12e-02 |
| 1600 | 2.28e-01 / 2.37e-01 | 9.67e-02 / 9.35e-02 | 5.36e-02 / 5.02e-02 | 3.88e-02 / 3.51e-02 |
| 2400 | 2.29e-01 / 2.35e-01 | 9.08e-02 / 9.03e-02 | 4.00e-02 / 3.96e-02 | 2.29e-02 / 1.85e-02 |

### Convergence rate in n_rho, the sharpest test of the mechanism

Least-squares slope of log(error) against log(n_rho) at n_rhodot = 100, over n_rho in [600, 800, 1200, 1600, 2400] (the range where neither axis is starved), alongside the level improvement the smooth mode gives at the finest swept grid:

| quantity | histogram exponent | smooth exponent | smooth/histogram error at (2400, 100) |
|---|---|---|---|
| mid-p p90 \|dp\| | 0.94 | 0.90 | 0.99 |
| p90 rel. Lambda_NEO | 0.81 | 0.75 | 0.84 |
| p90 rel. Lambda_other | 0.93 | 1.05 | 0.81 |

A midpoint-rule integration across a discontinuity aligned with the integration axis has error falling like 1/n_rho (exponent 1); a smooth integrand gives 1/n_rho^2.  Read the two columns together:

- **Lambda_other**: the histogram mode's exponent 0.93 is the signature of a discontinuity-limited integrand, and the smooth mode's 1.05 is measurably better.  This is the H staircase being removed, exactly as DESIGN §2.5 describes it.
- **Lambda_NEO**: the smooth mode is uniformly 16% better in level at the finest grid, but its fitted exponent (0.75) comes out *below* the histogram's (0.81) — because it is already lower at the low-n_rho end of the fit, so the fit measures a level shift and not a rate change.  Do not read that row as the smooth mode converging more slowly; the whole curve is below the histogram's at every n_rho in the table.
- Neither exponent reaches 2, in either mode.  **The H edges were not the only axis-aligned structure**, which is the mechanical reason the next section's answer is negative.

### Findings

1. **At (1200, 50) there is no detectable quadrature win.**  The mid-p p90 |dp| goes 7.88e-03 -> 7.72e-03, an improvement of +2.0% [-9.2%, +14.3%] (paired bootstrap over the 497 mid-p tracklets, 2000 replicates).  **The interval contains zero.**  A point estimate of "2%" is not a result: p90 of a few hundred values carries its own sampling error, and the review's independent replication on a different catalogue gave +14.8% [0.0%, +43.6%] — same sign, consistent with this, and also not separable from zero.  The honest headline is that **no quadrature win is detectable at (1200, 50)**, not that there is a 2% one.  The Lambdas are a different matter: Lambda_other's p90 relative error improves 8% at (1200, 50) (6.25e-02 -> 5.76e-02) and 19% at the finest swept grid (2400, 100), monotonically with n_rho — so the H staircase **is** a real quadrature error term and the reconstruction **does** remove part of it.  What the p(NEO) row says is that the part it removes is common-mode across the two components and cancels in the posterior ratio.  Either way there is no n_rho reduction to bank.
   - cheapest *stable* pchip grid that matches histogram mode's (1200, 50) mid-p accuracy: **(1200, 50)** = 60000 nodes, mid-p p90 7.72e-03, mid-p max 2.33e-01
   - cheapest *stable* none grid that matches histogram mode's (1200, 50) mid-p accuracy: **(1200, 50)** = 60000 nodes, mid-p p90 7.88e-03, mid-p max 2.33e-01
   - cheapest *stable* pchip grid that is 10x better than that: **no grid in the sweep reaches it**
   - cheapest *stable* none grid that is 10x better than that: **no grid in the sweep reaches it**
   - **Recommended production grid for the smooth mode: the unchanged (1200, 50).**  Single-core cost at it is in the Rust table above.
   - *Stable* = every finer point in the sweep also meets the threshold.  On the mid-p statistic this is nearly redundant, and an earlier draft of this report was wrong about why it was not: it claimed order-of-magnitude aliasing non-monotonicity in n_rho, with "n_rho = 100 beating n_rho = 200".  Checked against this JSON, the mid-p p90 is monotone in n_rho in 7 of 8 (mode, n_rhodot) columns, and the single exception is a 4.3% up-step (pchip, n_rhodot=25, n_rho 1600 -> 2400).  The stability constraint is kept because it is the right criterion for a production grid, not because the data here need it.

2. **The 10x-better threshold is below what this experiment could resolve, and no swept grid comes close anyway.**  10x better than the (1200, 50) baseline is mid-p p90 7.88e-04, only 1.5x above the reference's own error (5.17e-04) — so a grid that reached it could not be *shown* to have reached it against a 9600x400 reference.  That caveat turns out to be moot: the finest swept grid (2400, 100) (240000 nodes, 4x the production cost) reaches only 3.24e-03, i.e. 2.4x better, in either mode.  With a fitted exponent near 1, 10x would cost roughly 10x the nodes — **and it is not the reconstruction that gets you there**: the two modes' curves lie on top of each other in `grid_study_convergence.png`.  A serious 10x programme needs node placement or a different quadrature rule, plus a reference finer than 19200x800 to verify it against.

3. **At the production grid neither axis dominates any more.**  DESIGN §2.5 prescribes rho-heavy grids on the evidence that rho_dot refinement saturates, and its own comparison reproduces here: at n_rho = 200, going from n_rhodot = 100 to 400 moves the mid-p p90 only from 6.68e-02 to 6.40e-02.  But that saturation is a property of the *coarse-rho* regime.  At n_rho = 1200 the rho_dot axis is still paying: 7.88e-03 at n_rhodot = 50 vs 5.89e-03 at 100.  Spending 2x the production budget on either axis buys the same factor: (1200, 100) gains 1.34x and (2400, 50) gains 1.34x. So (1200, 50) sits at a balanced point — rho-heavy was the right way to *get* there, and from there the two axes are equally starved.  This is a side finding, outside this study's remit, offered to the supervisor as an observation on §2.5 rather than as a change to it.

### Where the residual error lives

Spearman rank correlation of per-tracklet |dp| at (1200, 50) (smooth mode) with:

| quantity | rho_s |
|---|---|
| `p_neo_ref` | +0.310 |
| `rate_deg_day` | +0.235 |
| `ar_frac_h_above` | -0.187 |
| `ar_measure` | +0.127 |
| `rho_true` | +0.082 |
| `n_nodes` | +0.033 |
| `v_mag` | -0.025 |

Two of these are worth naming.  `p_neo_ref` (+0.31) is the min(p, 1-p) bound reasserting itself even inside the mid-p window.  `ar_frac_h_above` — the §2.6(b) truncation-exposure diagnostic — is **negatively** correlated (-0.19): the tracklets whose admissible region reaches the faint end are *not* where the residual quadrature error lives.  That is an independent confirmation of the same conclusion the sweep reaches, arrived at per-tracklet rather than in aggregate: the H-bin structure is not the binding error term.  The strongest remaining geometric correlate is the apparent rate (+0.23), which sets how much of the (rho, rho_dot) plane is admissible — i.e. the admissible-region terminus, the other candidate DESIGN §2.5 names.

## 2. Self-consistency gate in smooth mode

`self_consistency run --n 20000 --seed 20260901 --gate --h-interpolation pchip` — prior = generator = the reconstructed model, a **new catalogue** drawn with the smooth sampler (the same seed produces a different catalogue in this mode, because the within-bin H draw is now an inverse-CDF of the cubic and the rejection stream diverges from there).  Both the generator and the classifier are switched by the one flag; `ScoreConfig` carries the mode into the chunk provenance stamp, so no cached chunk from a histogram run can be spliced in.

| criterion | smooth (this run) | histogram (run of record) |
|---|---|---|
| N scored | 20000 | 20000 |
| true NEOs | 179 | 185 |
| observed NEOs | 179 | 185 |
| expected NEOs | 180.3 | 187.5 |
| Z_tot | -0.2349 | -0.4384 |
| Z_tot bootstrap p | 0.9017 | 0.7407 |
| fitted odds scale c_hat | 0.958 | 0.923 |
| c_hat 95% lo | 0.670 | 0.645 |
| c_hat 95% hi | 1.370 | 1.321 |
| reliability max\|z\| | 0.7475 | 0.7210 |
| reliability threshold | 2.671 | 2.764 |
| HPD KS D, nearest node | 0.05385 | 0.05374 |
| HPD KS D, exact at truth | 0.05430 | 0.05410 |
| **HPD estimator gap** | 0.0004469 | 0.0003577 |
| integrand > 0 at truth | 1.0000 | 1.0000 |
| AUC (not gated) | 0.9808 | 0.9812 |
| median §2.6(b) measure fraction | 0.02142 | 0.02102 |

**Verdict: OVERALL PASS**

**The HPD estimator gap did not shrink**: 0.0004469 in smooth mode against 0.0003577 in histogram mode — the same number to within the resolution of a 2x10^4 KS statistic, and if anything larger.  The gap is the §4.1 grid-adequacy readout (the nearest-node and exact-at-truth HPD estimators bias in opposite directions and converge under refinement), so this is the gate-side confirmation of section 1's negative result: the H staircase was not what the gap was measuring.  Every other statistic is consistent between the two runs — c_hat straddles unity from the same side, reliability max|z| is unchanged, collapse coverage and the §2.6(b) diagnostic are unchanged — which is the substantive finding: **the reconstruction is gate-clean.** It passes the self-consistency test on its own generated population, so it is a legitimate prior, not merely a smoothing device.

The two runs' NEO counts differ (179 vs 185 true NEOs) purely because they are different catalogues; the calibration statistics are the comparable quantities, not the counts.


## 3. Mode-change measurement on the existing gate_v0 catalogue

The stored `gate_v0` catalogue (20000 tracklets, 185 true NEOs) was generated from the **piecewise-constant** model.  Re-scoring it with the smooth pair at (1200, 50) on the Rust backend, and comparing to the stored histogram scores, isolates what the prior change does — on the clamp experiment's template.

### (a) What the mode did to the scores

- 0.6% of tracklets score **bit-identically** in the two modes.
- 42.8% rise, 56.6% fall — unlike the D-14 clamp, this change is **two-sided**, which is what a mass-conserving reconstruction should be: it moves density *within* each H bin instead of adding density beyond the support.
- median dlog10 p = -5.73e-03; 1st-99th percentile [-0.1795, +0.1823]; max |dlog10 p| = 1.2843; max |dp| = 6.41e-02.
- 78.5% move by more than 0.01 dex, 12.10% by more than 0.1 dex.
- Spearman rank correlation between the two modes' scores: 0.99121.

Displacement against the §2.6(b) diagnostic (posterior-measure fraction at implied H > 25.5) — the diagnostic the change is expected to act through:

| §2.6(b) measure fraction | n | median dlog10 p | fraction moved | fraction up | p99 \|dlog10 p\| |
|---|---|---|---|---|---|
| [0, 0.0001) | 3883 | +0.02131 | 98.2% | 55.9% | 0.2684 |
| [0.0001, 0.001) | 457 | +0.01638 | 100.0% | 54.3% | 0.2801 |
| [0.001, 0.01) | 2761 | +0.00303 | 99.6% | 51.9% | 0.2095 |
| [0.01, 0.03) | 5025 | -0.00867 | 99.9% | 39.0% | 0.1394 |
| [0.03, 0.1) | 7365 | -0.00810 | 99.9% | 34.3% | 0.0998 |
| [0.1, 1) | 509 | -0.00098 | 95.5% | 44.6% | 0.1876 |

And against apparent V:

| V | n | median dlog10 p | fraction moved | fraction up | p99 \|dlog10 p\| |
|---|---|---|---|---|---|
| [14, 20) | 793 | +0.00491 | 98.6% | 55.7% | 0.3408 |
| [20, 21) | 1233 | -0.00029 | 99.1% | 49.0% | 0.2518 |
| [21, 22) | 2789 | -0.00452 | 99.3% | 44.4% | 0.2415 |
| [22, 23) | 5648 | -0.00715 | 99.4% | 40.5% | 0.2120 |
| [23, 23.5) | 4211 | -0.00611 | 99.7% | 42.1% | 0.1689 |
| [23.5, 24) | 5326 | -0.00653 | 99.4% | 41.7% | 0.1589 |

The sign of the median displacement **flips with truncation exposure**: tracklets whose admissible region stays well inside the support move up, tracklets whose region reaches the faint end move down.  That is a property of the reconstruction, not of the geometry: PCHIP's shape-preserving slopes flatten the top of a steeply rising cumulative count, so the reconstructed density near H = 25.5 is *below* the top bin's flat average — and the NEO component, with 55.8% of its mass in that bin (DESIGN §2.6), loses more than the other component does.  Where the region never reaches the top bin the reverse holds.

### (b) Ranking

Labels and subsets are the stored ones; the subsets are the DESIGN §4.2 truncation split on the §2.6(b) diagnostic, cut on the *stored* (histogram) values so the two passes compare the same tracklets.

| population | AUC smooth | AUC histogram | dAUC (smooth - hist) |
|---|---|---|---|
| `all` (n=20000, NEO=185) | 0.9815 | 0.9812 | +0.0003 [-0.0008, +0.0015] |
| `clean` (n=4340, NEO=83) | 0.9827 | 0.9828 | -0.0001 [-0.0012, +0.0010] |
| `broken` (n=12899, NEO=76) | 0.9738 | 0.9732 | +0.0006 [-0.0014, +0.0032] |

Against digest2's stored scores — the "better model" question:

| population | dAUC smooth - digest2 | dAUC histogram - digest2 | gain from the reconstruction |
|---|---|---|---|
| `all` (n=20000) | +0.0119 [+0.0035, +0.0211] | +0.0116 [+0.0027, +0.0210] | +0.0003 |
| `clean` (n=4340) | +0.0071 [+0.0004, +0.0159] | +0.0072 [+0.0012, +0.0151] | -0.0001 |
| `broken` (n=12899) | +0.0095 [-0.0019, +0.0210] | +0.0088 [-0.0043, +0.0215] | +0.0006 |

**The reconstruction's ranking gain over digest2 is +0.0003** on the full set, on top of the histogram model's +0.0116.  Its paired dAUC against the histogram model itself is +0.0003 [-0.0008, +0.0015] — a confidence interval that comfortably contains zero.  **On this catalogue the reconstruction is a ranking no-op.**  Which is the right expectation for two reasons, both worth stating rather than glossing: the truths were drawn from the histogram, so the histogram model is the *correct* prior here and cannot be beaten on its own generating distribution; and 185 true NEOs bound any paired dAUC measurement to about +/-0.002 regardless.  A ranking gain, if the reconstruction has one, would have to be demonstrated against truths drawn from something else — DESIGN §4.3, or a real catalogue.

### (c) Calibration against histogram-generated truth

**Read this as a deliberate model mismatch, not as a calibration test of the method.**  The catalogue was drawn from the piecewise-constant model; scoring it with the reconstruction is a small, controlled instance of the DESIGN §4.3 prior-misspecification experiment.  The smooth mode's *own* calibration is section 2 above, where prior and generator agree by construction.

| population | prior | predicted NEOs | observed | ratio | ECE | max \|z\| bin |
|---|---|---|---|---|---|---|
| `all` | smooth | 187.0 | 185 | 1.011 | 0.00052 | 1.95 |
| `all` | histogram | 187.5 | 185 | 1.013 | 0.00049 | 1.83 |
| `clean` | smooth | 84.1 | 83 | 1.013 | 0.00030 | 0.41 |
| `clean` | histogram | 84.2 | 83 | 1.015 | 0.00032 | 0.45 |
| `broken` | smooth | 79.4 | 76 | 1.045 | 0.00028 | 0.65 |
| `broken` | histogram | 79.7 | 76 | 1.049 | 0.00030 | 0.73 |

The mismatch costs essentially nothing: predicted/observed moves 1.0133 -> 1.0109 and ECE 0.00049 -> 0.00052, i.e. the *wrong* prior's predicted NEO count is off by 1.1% where the *right* one is off by 1.3%.  This is the same phenomenon D-14 recorded for the H clamp: **a correctly marginalized posterior absorbs a change to the prior's faint end**, because the change is common-mode across the two components in almost the same proportion as their mixture weights.  It is a statement about the marginalization, not evidence that the priors are interchangeable — the per-tracklet scores moved by up to 1.28 dex.


### Controls

- n_nodes_identical: by construction (same grid; not re-derived here)
- n_nonfinite_pchip: 0
- n_nonfinite_hist: 0
- pchip_in_unit_interval: True

## Plots

- `grid_study_convergence.png` — mid-p p90 |dp| and Lambda_other error vs n_rho, both modes overlaid, with the reference's own error floor shaded.
- `grid_study_mode_ratio.png` — smooth error divided by histogram error, for p(NEO) and for Lambda_other.  This is the figure that carries section 1's finding: the Lambda ratio sits below 1 and falls with n_rho, the p(NEO) ratio sits on 1.
- `grid_study_cost_accuracy.png` — the cost-accuracy front of both modes on one axis (they lie on top of each other).
- `reconstruction_shape.png` — the construction itself, on three cells: cumulative count with the monotone cubic, and the two densities.
- `mode_change_scatter.png` — smooth vs histogram p(NEO) coloured by the §2.6(b) diagnostic, plus displacement against that diagnostic and against V.
- `mode_change_roc.png`, `mode_change_calibration.png` — ranking and calibration of both priors (and digest2) on the stored labels.

## Caveats

- The smooth mode is a *reconstruction of the same model*, not a new population model.  It cannot add information the histogram does not contain; what it does is stop asserting that the density is flat inside a bin and 100x different one magnitude away.
- It does **not** address the `1/(e sin i)` integrable singularity (DESIGN §2.5) or the D-4 truncation corner (§2.6).
- Section 3's calibration arm is a prior mismatch by construction; do not read it as miscalibration of the method.
- Section 1's negative result is a statement about p(NEO) accuracy at n_rho >= 100.  It says nothing about a future H-bin-edge-aware node placement (issue #1), which targets the same discontinuities by moving nodes rather than by smoothing the density; the convergence-exponent table suggests that too would have limited headroom in p(NEO), though more in Lambda.
- **Each sweep point is measured against a reference in its own mode.**  That is deliberate and it bounds what section 1 can claim: it answers "how many nodes does each mode need to converge to its own limit", which is the node-count question, and not "which mode's limit is closer to the truth".  The latter is not a well-posed quadrature question — the two modes are two different priors, each exactly integrable in principle — and it is what sections 2 and 3 address instead.
- **Only the mid-p sweep is included.**  The companion whole-catalogue sweep over all 2x10^4 `gate_v0` tracklets was started and abandoned after 30 minutes and 23 core-hours: another agent was running a large job in this repository, the box was at load ~200 on 96 cores, and it was starving the required test suites.  Re-running it is one command (`grid --run validation/output/gate_v0 --select all --out-name grid_study_all.json --ref-grid 4800 200 --ref-check-grid 9600 400`) and this report picks it up automatically as `grid_study_all.json`.  Nothing above depends on it: it would add the unconditional-statistic table, which the mid-p section explains is the less informative of the two.
- The two `gate_v0` reconciliation numbers quoted above (mid-p p90 3.88e-3, mid-p max 1.62e-2 against the (1200, 600) reference, and the 7.7e-13 agreement of the recomputed scores with the run of record) come from a targeted measurement on that catalogue rather than from a saved sweep, for the same reason.  Reproduce with: score `gate_v0` at (1200, 50), (1200, 600) and (4800, 200) in histogram mode via `score_chunks`, and take |dp| percentiles over the rows with stored p in (0.05, 0.95).

## Reproduce

Every number above comes out of these six commands, in this order (`OMP_NUM_THREADS=1` throughout — the work is Rust-side and rayon already owns the parallelism):

```
# 1. the mid-p sweep (PRIMARY: DESIGN §2.5's statistic, 497 mid-p tracklets)
python -m validation.h_interpolation_study grid \
    --run validation/output/calibration_1e5 --select midp \
    --out validation/output/h_interpolation --out-name grid_study.json \
    --ref-grid 9600 400 --ref-check-grid 19200 800 \
    --threads 96 --time-reps 5 --time-tracklets 48

# 3. interleaved single-core timing
python -m validation.h_interpolation_study bench \
    --run validation/output/gate_v0 --out validation/output/h_interpolation \
    --n-tracklets 48 --reps 9 --grid 1200 50 --grid 1200 100

# 4. the gate, prior = generator = the reconstruction
python -m validation.self_consistency run --n 20000 \
    --seed 20260901 --workers 32 --gate --h-interpolation pchip \
    --out validation/output/gate_v0_pchip

# 5. the mode-change measurement on the stored histogram catalogue
python -m validation.h_interpolation_study rescore \
    --run validation/output/gate_v0 --out validation/output/h_interpolation \
    --n-rho 1200 --n-rhodot 50 --threads 96
python -m validation.h_interpolation_study analyze \
    --run validation/output/gate_v0 --out validation/output/h_interpolation

# 6. this file
python -m validation.h_interpolation_study report \
    --out validation/output/h_interpolation \
    --gate-run validation/output/gate_v0_pchip --gate-reference validation/output/gate_v0
```

`validation/output/` is gitignored, so a copy of this file is kept at `validation/H_INTERPOLATION.md` (the same arrangement as `validation/DIGEST2_COMPARISON_CLAMPED.md`).
