# Astrometric noise vs. priora v0's delta-function attributable

*Does D-6 (the exact-attributable assumption) cost anything at Rubin-level astrometric noise, or is v1's covariance marginalization urgent?  Harness: `validation/astrometric_noise.py`.  Noiseless reference: the stored `gate_v0` scores — nothing was re-scored on that side.*

## Bottom line

At the adopted Rubin-like noise level (Sorcha's model, m5 = 24, FWHM = 700 mas; median realised sigma 35.6 mas per detection per coordinate, 76.7 mas at the 90th percentile), on 20000 tracklets containing 185 true NEOs:

- **Calibration is untouched.** The global odds scale moves from ĉ = 0.923 (noiseless) to ĉ = 0.925 (noisy); the global count statistic from Z = -0.44 to Z = -0.42. Both stay inside the run's own ±35% odds resolution at N = 2×10⁴.
- **Ranking is untouched.** ΔAUC = +0.00097 [-0.00014, +0.00261] (paired bootstrap) — consistent with zero, and an order of magnitude below the +0.0126 method gap over digest2 that D-14 measured.
- **The operating point is untouched.** Deploying WP7's matched-completeness cut (p ≥ 0.254, chosen on noiseless scores) against noisy scores gives 87.6% completeness at 5.3% contamination, versus 88.1% / 6.3% noiseless.
- **Triage thresholds are stable.** Zero tracklets crossed p = 0.1 / 0.3 / 0.5 / 0.7 in the headline realization; averaged over 25 realizations of an NEO-enriched subsample the worst threshold churns 0.12 tracklets per realization, and not one true NEO fell below a cut in any of them.
- **Per-tracklet scores barely move.** Median |ΔP(NEO)| = 5.06e-06, 90th percentile 7.03e-05, 99th 7.20e-04.

**Verdict: v0's delta-function attributable is acceptable for Rubin triage, and v1's covariance marginalization is not urgent on this axis.** The reason is arithmetic rather than luck: a 45-minute arc converts an 85 mas endpoint error into a rate error of ~3.8 arcsec/day, which is 0.20% of the median tracklet's own motion and 0.86% at the 90th percentile. The classifier's response to the attributable is smooth on that scale. The caveats that could change this answer are in *Where this could break* below; the largest is arc length, not astrometric precision.

## The error model, and where it comes from

Transcribed from **sorcha src/sorcha/modules/PPAddUncertainties.py :: calcAstrometricUncertainty + calcRandomAstrometricErrorPerCoord; src/sorcha/modules/PPTrailingLoss.py :: calcTrailingLoss**, ref `v1.2.1 (4e37cfe974a04a2197c64ff5074dc68608fe1893)`. Sorcha's own citations:

- Ivezic et al. 2019, ApJ 873, 111 (LSST overview paper; arXiv:0805.2366, cited in Sorcha's docstring)
- LSST Science Requirements Document (ls.st/srd) for the 10 mas systematic floor
- Veres & Chesley 2017, AJ 154, 12 (trailing losses)

The chain, in Sorcha's order:

```
dmag  = 1.25 log10(1 + a x^2/(1 + b x)),  x = v texp/seeing,  a=0.761 b=1.162
x_snr = 10^(0.4 (V + dmag - m5))
SNR   = 1 / sqrt((0.04 - gamma) x_snr + gamma x_snr^2),        gamma = 0.039
sigma = sqrt(error_sys^2 + (astErrCoeff * FWHMeff / SNR)^2)
        error_sys = 10 mas,  astErrCoeff = 0.60,  FWHMeff = 700 mas
```

Adopted parameters: **m5 = 24** — Sorcha hardcodes no default depth (it reads `fiveSigmaDepth` per visit from an opsim database), so we take the value internally consistent with the catalogue being perturbed: the WP6 generator's faint cut is V < 24 and is documented as "the Rubin-like detection limit", so setting m5 = V_max makes this catalogue exactly a 5σ-limited sample and puts its faintest tracklets at SNR = 5. It is also the conservative choice; a half-magnitude-deeper survey is run below as a sensitivity case. **FWHMeff = 700 mas** is Sorcha's own default argument value; **error_sys = 10 mas**, **astErrCoeff = 0.60**, **gamma = 0.039**, **nvisit = 1**, **texp = 30 s**, trailing losses **on** are all Sorcha defaults.

σ for one detection, one coordinate (static source, no trailing):

| V | SNR | σ_rand [mas] | σ [mas] | σ [arcsec] |
|---:|---:|---:|---:|---:|
| 18 | 466.29 | 0.90 | 10.04 | 0.0100 |
| 21 | 67.67 | 6.21 | 11.77 | 0.0118 |
| 23 | 12.33 | 34.07 | 35.50 | 0.0355 |
| 24 | 5.00 | 84.00 | 84.59 | 0.0846 |

Realised on the catalogue: median 35.6 mas (NEOs 38.9 mas — they are both fainter on average and trailed), p90 76.7 mas, max 156.0 mas. Median trailing loss 0.047 mag overall, 0.362 mag for NEOs. See `error_model.png`.

## What the noise does to the scores

One realization on all 20000 tracklets, dt = 45 min, position from detection 0 and rates from the two-endpoint finite difference. V is carried through unchanged (photometric noise out of scope).

| quantity | subset | median \|Δ\| | p90 \|Δ\| | p99 \|Δ\| | max \|Δ\| | mean Δ |
|---|---|---:|---:|---:|---:|---:|
| ΔP(NEO) | all | 5.06e-06 | 7.03e-05 | 7.20e-04 | 3.54e-02 | -4.20e-06 |
| ΔP(NEO) | true NEO | 0.00e+00 | 7.01e-04 | 2.54e-02 | 3.54e-02 | -1.31e-04 |
| ΔP(NEO) | true other | 5.09e-06 | 6.90e-05 | 6.75e-04 | 2.79e-02 | -3.01e-06 |
| Δ ln LR | all | 1.75e-02 | 9.86e-02 | 5.74e-01 | 7.42e+00 | +1.25e-02 |
| Δ ln LR | true NEO | 1.28e-02 | 1.07e-01 | 3.14e-01 | 5.43e-01 | +4.06e-03 |
| Δ ln LR | true other | 1.76e-02 | 9.86e-02 | 5.77e-01 | 7.42e+00 | +1.25e-02 |

Magnitude-resolved (`delta_vs_observables.png`) — the error model is magnitude-dependent, so this is where it has to show:

| V bin | n | median \|ΔP\| | p90 \|ΔP\| | p99 \|ΔP\| |
|---|---:|---:|---:|---:|
| 14.0–20.0 | 793 | 1.46e-06 | 2.82e-05 | 4.91e-04 |
| 20.0–21.0 | 1233 | 1.82e-06 | 2.49e-05 | 2.63e-04 |
| 21.0–22.0 | 2789 | 1.98e-06 | 2.88e-05 | 3.40e-04 |
| 22.0–22.5 | 2333 | 2.96e-06 | 3.86e-05 | 3.55e-04 |
| 22.5–23.0 | 3315 | 4.45e-06 | 5.84e-05 | 5.11e-04 |
| 23.0–23.5 | 4211 | 7.54e-06 | 8.74e-05 | 7.17e-04 |
| 23.5–24.0 | 5326 | 1.12e-05 | 1.16e-04 | 1.55e-03 |

| rate bin [deg/day] | n | median \|ΔP\| | p90 \|ΔP\| | p99 \|ΔP\| |
|---|---:|---:|---:|---:|
| 0.003–0.03 | 517 | 1.75e-06 | 4.11e-05 | 3.53e-04 |
| 0.03–0.1 | 2252 | 2.21e-06 | 2.65e-05 | 3.16e-04 |
| 0.1–0.2 | 5002 | 2.45e-06 | 3.28e-05 | 3.99e-04 |
| 0.2–0.35 | 6145 | 5.36e-06 | 6.31e-05 | 5.62e-04 |
| 0.35–0.6 | 5731 | 1.23e-05 | 1.06e-04 | 7.61e-04 |
| 0.6–1 | 235 | 2.53e-04 | 2.72e-03 | 2.12e-02 |
| 1–2 | 66 | 0.00e+00 | 0.00e+00 | 1.68e-06 |
| 2–10 | 41 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

Two features of that table are worth naming rather than leaving to be puzzled over. The peak at 0.6–1 deg/day is the NEO band: those tracklets are the faint, trailed, mid-probability ones where the posterior actually has a gradient. Above 1 deg/day the deltas collapse to exactly zero because essentially every such tracklet already scores P(NEO) = 1 to machine precision — nothing in the model can move them. The rise with V is the error model itself: σ grows ~8× from V = 18 to V = 24 while the median |ΔP| grows ~8×, i.e. the response is close to linear in σ.

## Calibration under noise

The classifier now assumes an exact attributable that is *wrong*. Running the WP6 gate's own calibration machinery (equal-expected-NEO-count bins, Poisson-binomial z, parametric-bootstrap thresholds, fitted global odds scale) against the true labels measures the real miscalibration Rubin would see from v0. This is no longer a self-consistency test — a failure here would be a *property of the data*, not a bug.

| statistic | noiseless | with noise |
|---|---:|---:|
| observed / expected NEO count | 185.0 / 187.5 | 185.0 / 187.4 |
| global Z_tot (p) | -0.44 (0.741) | -0.42 (0.752) |
| fitted odds scale ĉ [95%] | 0.923 [0.645, 1.321] | 0.925 [0.647, 1.324] |
| reliability max\|z\| (bootstrap 99th) | 0.72 (2.66) | 0.72 (2.62) |
| reliability verdict | PASS | PASS |
| Brier score | 0.00140 | 0.00140 |
| expected calibration error | 1.23e-04 | 1.19e-04 |

See `calibration.png`. Caveat carried over from DESIGN §4.1: at N = 2×10⁴ (≈185 NEOs) the odds scale is only constrained to roughly ±35%, so this says "no detectable miscalibration at the ±35% level", not "identical to five decimal places". The paired per-tracklet deltas above are the sharper statement.

## Ranking and threshold stability

AUC 0.98125 (noiseless) → 0.98222 (noisy); ΔAUC = +0.00097 [-0.00014, +0.00261] from a paired, label-stratified bootstrap (2000 replicates). See `ranking.png`.

| threshold | flagged (noiseless → noisy) | crossed up | crossed down | NEOs lost | tracklets at risk |
|---|---:|---:|---:|---:|---:|
| p ≥ 0.1 | 205 → 205 | 0 | 0 | 0 | 1 |
| p ≥ 0.3 | 169 → 169 | 0 | 0 | 0 | 0 |
| p ≥ 0.5 | 161 → 161 | 0 | 0 | 0 | 0 |
| p ≥ 0.7 | 153 → 153 | 0 | 0 | 0 | 0 |

"At risk" counts tracklets whose noiseless score sits within 7.20e-04 (the 99th percentile of |ΔP|) of the cut. The zeros above are therefore two facts at once: the perturbation is small **and** almost nothing sits near a triage threshold at this base rate — 102 of 20,000 tracklets score anywhere in (0.05, 0.95). A single realization cannot separate "impossible" from "unlikely this time", so the MC block below reports the expected churn per realization instead.

Averaged over the MC realizations on the 500-tracklet subsample (which is *enriched* in the interesting tracklets — it holds all 185 NEOs, so its churn rate is an upper bound on the full population's):

| threshold | flagged (noiseless) | mean crossings / realization | worst realization | realizations with any | mean NEOs lost |
|---|---:|---:|---:|---:|---:|
| p ≥ 0.1 | 170 | 0.12 | 1 | 3/25 | 0.00 |
| p ≥ 0.3 | 161 | 0.12 | 1 | 3/25 | 0.00 |
| p ≥ 0.5 | 157 | 0.00 | 0 | 0/25 | 0.00 |
| p ≥ 0.7 | 151 | 0.00 | 0 | 0/25 | 0.00 |

## The operating point (WP7's matched-completeness cut)

`validation/DIGEST2_COMPARISON.md` sets the reference: digest2's own NEOCP cut (raw ≥ 65) recovers 88.1% of the true NEOs on `gate_v0`, and priora matches that completeness at 6.3% contamination. Three readings, of which the third is the operationally honest one — a threshold is chosen once, on a study like WP7's, and then deployed against noisy data:

| reading | threshold | flagged | completeness | contamination |
|---|---:|---:|---:|---:|
| noiseless p >= 0.254 | 0.254 | 174 | 88.1% | 6.3% |
| noisy p >= 0.239 | 0.239 | 175 | 88.1% | 6.9% |
| noisy p >= 0.254 (noiseless cut) | 0.254 | 171 | 87.6% | 5.3% |
| noiseless p >= 0.5 | 0.5 | 161 | 84.9% | 2.5% |
| noisy p >= 0.5 | 0.5 | 161 | 84.9% | 2.5% |

## Per-tracklet noise sensitivity map

25 independent noise realizations of a fixed 500-tracklet stratified subsample (**all 185 true NEOs** plus a V-stratified draw of the rest). The spread of P(NEO) at fixed object and fixed error model is the per-tracklet cost of D-6, free of any population averaging. See `mc_sensitivity.png`.

- std(P) overall: median 4.55e-06, p90 1.71e-04, p99 8.17e-03, max 0.024.
- std(P) for true NEOs: median 0.00e+00, p90 1.77e-03, max 0.024.
- 1.0% of tracklets have std(P) > 0.01, 0.0% > 0.05, 0.0% > 0.10.
- Mean over realizations minus the noiseless score: median +0.00e+00, mean -3.07e-05 — the noise is essentially unbiased at this amplitude (a nonlinear response would show up here as a systematic offset, and is what an outer marginalization would actually correct).

std(P) against the noiseless score — the triage-relevant slice, because only a tracklet near a threshold can be moved across it:

| noiseless P(NEO) | n | median std | p90 std | p99 std |
|---|---:|---:|---:|---:|
| 0–0.0001 | 69 | 9.54e-07 | 4.46e-06 | 1.68e-05 |
| 0.0001–0.01 | 249 | 1.31e-05 | 8.10e-05 | 2.99e-04 |
| 0.01–0.1 | 12 | 2.19e-04 | 1.09e-03 | 2.55e-03 |
| 0.1–0.3 | 9 | 7.33e-03 | 1.30e-02 | 2.00e-02 |
| 0.3–0.5 | 4 | 7.06e-03 | 2.05e-02 | 2.37e-02 |
| 0.5–0.7 | 6 | 2.18e-03 | 4.08e-03 | 5.10e-03 |
| 0.7–0.9 | 7 | 1.33e-03 | 1.26e-02 | 2.12e-02 |
| 0.9–1 | 144 | 0.00e+00 | 6.17e-05 | 1.05e-03 |

| V bin | n | median std | p90 std | p99 std |
|---|---:|---:|---:|---:|
| 14.0–20.0 | 94 | 3.06e-06 | 1.60e-04 | 7.95e-04 |
| 20.0–21.0 | 44 | 2.31e-06 | 7.59e-05 | 2.74e-03 |
| 21.0–22.0 | 79 | 1.68e-06 | 1.15e-04 | 1.31e-03 |
| 22.0–22.5 | 50 | 3.49e-06 | 4.88e-05 | 3.81e-03 |
| 22.5–23.0 | 50 | 4.13e-06 | 2.70e-04 | 5.07e-03 |
| 23.0–23.5 | 74 | 1.09e-05 | 1.40e-04 | 2.24e-03 |
| 23.5–24.0 | 109 | 2.03e-05 | 1.14e-03 | 2.21e-02 |

The most sensitive tracklets in the subsample:

| gen index | NEO? | V | rate [deg/day] | σ [mas] | P noiseless | P mean | std(P) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1910 | yes | 23.91 | 0.745 | 96.0 | 0.3865 | 0.3791 | 0.0241 |
| 8065 | yes | 23.57 | 0.109 | 58.4 | 0.7864 | 0.7890 | 0.0222 |
| 1457 | yes | 23.86 | 0.708 | 90.9 | 0.2536 | 0.2445 | 0.0208 |
| 14521 | yes | 23.91 | 0.338 | 83.8 | 0.4182 | 0.4128 | 0.0120 |
| 10731 | yes | 23.94 | 0.748 | 98.4 | 0.1486 | 0.1468 | 0.0110 |
| 4309 | yes | 23.91 | 0.648 | 92.8 | 0.1952 | 0.1999 | 0.0081 |
| 7115 | yes | 23.61 | 0.710 | 72.3 | 0.2858 | 0.2878 | 0.0079 |
| 19135 | yes | 22.91 | 0.602 | 38.2 | 0.1142 | 0.1202 | 0.0073 |
| 1958 | yes | 22.05 | 0.138 | 18.1 | 0.1597 | 0.1592 | 0.0068 |
| 8117 | yes | 23.77 | 0.762 | 85.3 | 0.8284 | 0.8234 | 0.0061 |

## Positions vs. rates

DESIGN §2.7 expects the *rate* uncertainty to dominate on short arcs. The three runs share their standard-normal draws, so the split is exact rather than a comparison of two unrelated experiments:

- `pos`: the same offset applied to **both** detections — a rigid translation of the tracklet. Position error σ, rate error identically zero.
- `rate`: detection 0 left exactly alone, detection 1 offset by the difference of the two draws. Position exact, rate error identical draw-for-draw to the physical case.
- `both`: the physical case; by construction `ΔA(both) = ΔA(pos) + ΔA(rate)` to the last bit.

| noise applied to | median \|ΔP\| | p90 \|ΔP\| | p99 \|ΔP\| | RMS ΔP | max \|ΔP\| |
|---|---:|---:|---:|---:|---:|
| positions only | 1.29e-10 | 1.22e-08 | 7.77e-07 | 1.86e-06 | 4.12e-05 |
| rates only | 2.19e-06 | 1.59e-04 | 6.88e-03 | 1.30e-03 | 1.58e-02 |
| both (physical) | 2.19e-06 | 1.59e-04 | 6.88e-03 | 1.30e-03 | 1.58e-02 |

Variance share of ΔP: rates 100.0%, positions 0.0% (corr(both, rate) = 1.000, corr(both, pos) = -0.049). The two one-at-a-time runs reconstruct the joint one with an RMS residual of 1.95e-06 in ΔP — i.e. the classifier responds essentially linearly to the perturbation at this amplitude, which is precisely the regime a Gaussian covariance marginalization would model correctly (and therefore also the regime in which it has least to add). See `decomposition.png`.

**Depth sensitivity.** The same subsample and the same draws at m5 = 24.5 (half a magnitude deeper) gives p90 |ΔP| = 1.15e-04, a factor 0.72 of the baseline — the answer is not sitting on a knife edge of the adopted depth.

## Admissible-region geometry under noise

A bound orbit's true (ρ, ρ̇) must lie inside the admissible region of its *exact* attributable. Under noise the region moves and need not contain the truth any more — the geometric footprint of the delta-function assumption.

- Truth inside the admissible region: 100.00% noiseless → 100.00% noisy.
- Median posterior mass within a factor 2 of the true ρ: 0.9999 → 0.9999.

## Where this could break

The conclusion is scoped by the arc, not by the astrometry. The rate error scales as σ√2/dt, so:

- **Shorter arcs.** A 15-minute pair triples the rate error; a 5-minute pair multiplies it by 9. That is a genuinely different regime and this experiment does not cover it. If Rubin's tracklet cadence tightens, re-run with `--dt-min`.
- **Slow movers.** The fractional rate error is worst for the slowest objects (distant, near-stationary), which is visible in the rate-binned table above. They are also the ones the classifier is most confident about, which is why the score still barely moves — but a population with more TNOs would weight this differently.
- **Photometric noise**, deliberately excluded here, enters through H(V, ρ) with unit Jacobian (DESIGN §2.2) and is a separate, additive study.
- **Prior mismatch** (DESIGN §4.3) is a far larger effect than anything measured here and is the thing that should actually be prioritised over covariance marginalization.

## Assumptions

- Photometric noise is NOT injected: V is carried through unchanged. The measured degradation is therefore the astrometric contribution alone; photometric scatter would additionally smear H(V, rho).
- Two detections per tracklet, dt = 45 min, exactly linear motion — the WP7 emission convention. Real Rubin tracklets carry >= 2 detections; more detections would shrink the rate error roughly as 1/sqrt(n_det) (and the arc length, not the detection count, sets the leading 1/dt factor).
- The attributable is rebuilt by finite differencing the two noisy endpoints, which is the maximum-likelihood 2-point estimate. A least-squares fit to >2 detections is the same estimator with a smaller variance.
- One seeing/depth pair for the whole catalogue (0.7 arcsec, m5 = 24.0); Sorcha draws both per visit from an opsim database. Real Rubin visits spread around these values, which would widen the sigma distribution without moving its centre much.
- sigma is applied in the local sky metric with cos(delta) evaluated once at the attributable's declination; a <1% metric error for all but a handful of >5 deg/day tracklets, and the price of an exactly additive position/rate decomposition.
- No light-time, aberration or field-distortion terms: the same geometric treatment v0's classifier itself uses.
- The population is the WP6 self-consistency catalogue (digest2's own model, All variant, 14 <= V < 24, geocentre). Calibration statements are therefore about the prior-matched case; DESIGN 4.3's prior-mismatch stress test is a separate and larger effect.

## Reproduce

```
OMP_NUM_THREADS=1 python3 -m validation.astrometric_noise run --workers 32
```

Seed 20260902; source catalogue `/home/mjuric/priora/validation/output/gate_v0` (read-only). Every stage is resumable and a pure function of its inputs.
