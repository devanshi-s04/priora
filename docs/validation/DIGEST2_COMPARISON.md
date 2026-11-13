# WP7 — digest2 head-to-head comparison (DESIGN §4.2)

*Produced by `validation/digest2_comparison.py`. Raw outputs (plots, `summary.json`, `summary.txt`) live under `validation/output/digest2_comparison/` (gitignored); this file is the durable record.*

## What was compared, and how the premise was enforced

| | |
|---|---|
| digest2 | `2.8.2` from `github.com/Smithsonian/mpc-public@c9e3f89a26ec`, subdirectory `digest2`, built from sdist |
| install | `CFLAGS="-D_GNU_SOURCE" pip install --user "git+https://github.com/Smithsonian/mpc-public.git@c9e3f89a26ec#subdirectory=digest2"` (the flag is required: the sdist compiles `-std=c99`, under which `rand_r` is undeclared and the build fails) |
| population model | `data/digest2.model.csv`, sha256 `2b9d9d2a6d8ab16d…`, passed via `Digest2(model_path=…)` |
| same file? | the pip package ships `digest2/data/digest2.model.csv` as an unresolved git-LFS pointer whose `oid` is `2b9d9d2a6d8ab16d…` — **identical** to ours, so the shared-model premise is exact, not approximate |
| observatory codes | `digest2.obscodes` (auto-downloaded from the MPC); only code 500 used |
| input format | ADES PSV (`parse_ades_psv`), 20000 tracklets x 2 detections, dt = 45 min |
| parse rate | **100.00%** (20000/20000 tracklets), 100.00% scored |
| throughput | 10.1 ms/tracklet at 12 threads |
| score compared | digest2 `raw.NEO` (All population) against ds42 `p_neo` (All variant). digest2's default `noid.NEO` uses the `Unk` histogram and is a different quantity; it is reported alongside. |

Each WP6 attributable becomes a two-detection tracklet with detection 0 **at** the attributable epoch (digest2 anchors its ranging on `obsPair[0]`, so its reference position and observer state are exactly the attributable's) and detection 1 at `epoch + dt`, propagated linearly in (alpha, delta). Since the WP6 attributable is itself a first-order motion model, the pair reproduces it by construction.

Rows with a non-finite ds42 `p_neo` are excluded from both sides (they cannot enter a ROC on either): `enriched_f01` 1. These are tracklets where both component likelihoods underflow to zero, so the posterior is 0/0; excluding them from both codes is the only symmetric option.

## Headline: the ranking gap on the clean subset

| run | subset | n | NEOs | AUC ds42 | AUC digest2 | ΔAUC (95% CI) |
|---|---|---:|---:|---|---|---|
| gate_v0 | all | 20000 | 185 | 0.9812 [0.9666, 0.9933] | 0.9696 [0.9476, 0.9879] | **+0.0116** [-0.0004, +0.0248] |
| gate_v0 | **clean** | 4340 | 83 | 0.9828 [0.9622, 0.9986] | 0.9756 [0.9468, 0.9984] | **+0.0072** [-0.0001, +0.0188] |
| gate_v0 | broken | 12899 | 76 | 0.9732 [0.9455, 0.9936] | 0.9643 [0.9295, 0.9904] | **+0.0088** [-0.0101, +0.0269] |
| enriched_f01 | all | 9999 | 423 | 0.9816 [0.9736, 0.9889] | 0.9676 [0.9554, 0.9782] | **+0.0140** [+0.0066, +0.0222] |
| enriched_f01 | **clean** | 2354 | 226 | 0.9797 [0.9647, 0.9915] | 0.9750 [0.9583, 0.9872] | **+0.0047** [-0.0065, +0.0151] |
| enriched_f01 | broken | 6322 | 164 | 0.9816 [0.9696, 0.9917] | 0.9696 [0.9518, 0.9842] | **+0.0120** [+0.0049, +0.0204] |

On the clean subset of `gate_v0` — the DESIGN §4.2 headline — ds42 ranks better by **ΔAUC = +0.0072** (95% CI [-0.0001, +0.0188], paired label-stratified bootstrap, B = 2000; bootstrap P(Δ ≤ 0) = 0.0295). Both codes read the same population file and the test population is drawn from it, so this gap is attributable to the method — the Jacobian, the marginalisation over the admissible region, and the posterior — and not to a better prior.

Not every subset resolves the gap, and the table should be read that way: `gate_v0/all` includes zero (P(Δ ≤ 0) = 0.029); `gate_v0/clean` includes zero (P(Δ ≤ 0) = 0.029); `gate_v0/broken` includes zero (P(Δ ≤ 0) = 0.166); `enriched_f01/clean` includes zero (P(Δ ≤ 0) = 0.190). With ~10^2 NEOs per subset the paired bootstrap resolves a difference of order 10^-2 in AUC only marginally; the sign is consistent across every subset of both runs, but a single subset's interval is not on its own decisive. The N = 10^5 calibration run is what tightens this.

## Calibration contrast

digest2's score is a *fraction of tagged model population*, not a posterior probability, and it does not claim to be one. Treating `raw/100` as P(NEO) and asking the calibration question WP6 asks of ds42 makes the difference concrete rather than asserted:

| run | subset | code | predicted NEOs | observed | pred/obs | Brier | ECE |
|---|---|---|---:|---:|---:|---:|---:|
| gate_v0 | all | ds42 `p_neo` | 187.5 | 185.0 | 1.01 | 1.400e-03 | 4.892e-04 |
| gate_v0 | all | digest2 `raw`/100 | 1125.9 | 185.0 | 6.09 | 1.592e-02 | 4.705e-02 |
| gate_v0 | all | digest2 `noid`/100 | 1637.4 | 185.0 | 8.85 | 2.684e-02 | 7.262e-02 |
| gate_v0 | clean | ds42 `p_neo` | 84.2 | 83.0 | 1.01 | 1.465e-03 | 3.230e-04 |
| gate_v0 | clean | digest2 `raw`/100 | 210.4 | 83.0 | 2.53 | 1.122e-02 | 2.935e-02 |
| gate_v0 | clean | digest2 `noid`/100 | 356.8 | 83.0 | 4.30 | 2.307e-02 | 6.309e-02 |
| enriched_f01 | all | ds42 `p_neo` | 77.2 | 80.7 | 0.96 | 1.562e-03 | 3.326e-04 |
| enriched_f01 | all | digest2 `raw`/100 | 563.4 | 80.7 | 6.98 | 1.568e-02 | 4.589e-02 |
| enriched_f01 | all | digest2 `noid`/100 | 832.0 | 80.7 | 10.31 | 2.603e-02 | 7.144e-02 |
| enriched_f01 | clean | ds42 `p_neo` | 41.3 | 43.1 | 0.96 | 2.315e-03 | 7.620e-04 |
| enriched_f01 | clean | digest2 `raw`/100 | 116.6 | 43.1 | 2.70 | 1.394e-02 | 3.111e-02 |
| enriched_f01 | clean | digest2 `noid`/100 | 200.9 | 43.1 | 4.66 | 2.511e-02 | 6.677e-02 |

Over the whole `gate_v0` run ds42's posterior predicts 187.5 NEOs against 185 observed (ratio 1.01); digest2's `raw`/100 predicts 1125.9, over by a factor of 6. The reliability curves (`calibration.png`) show the same thing bin by bin: ds42 tracks the diagonal, digest2 sits an order of magnitude above it across the whole score range. This is not a defect of digest2 — it is the quantity it was designed to emit — but it is the reason a digest2 score cannot be thresholded against a stated contamination budget.

## What the threshold buys (operational reading of the calibration gap)

digest2 posts a tracklet to the NEOCP at a score of 65. At equal completeness, what does each code's cut cost in contamination?

| run | subset | operating point | flagged | completeness | contamination |
|---|---|---|---:|---:|---:|
| gate_v0 | all | digest2 raw >= 65 | 399 | 88.1% | 59.1% |
| gate_v0 | all | ds42 p >= 0.254 | 174 | 88.1% | 6.3% |
| gate_v0 | all | ds42 p >= 0.5 | 161 | 84.9% | 2.5% |
| gate_v0 | clean | digest2 raw >= 65 | 107 | 91.6% | 29.0% |
| gate_v0 | clean | ds42 p >= 0.609 | 76 | 91.6% | 0.0% |
| gate_v0 | clean | ds42 p >= 0.5 | 76 | 91.6% | 0.0% |
| enriched_f01 | all | digest2 raw >= 65 | 466 | 83.0% | 65.2% |
| enriched_f01 | all | ds42 p >= 0.233 | 355 | 83.0% | 6.1% |
| enriched_f01 | all | ds42 p >= 0.5 | 333 | 78.5% | 1.7% |
| enriched_f01 | clean | digest2 raw >= 65 | 218 | 87.6% | 36.6% |
| enriched_f01 | clean | ds42 p >= 0.461 | 198 | 87.6% | 0.0% |
| enriched_f01 | clean | ds42 p >= 0.5 | 196 | 86.7% | 0.0% |

On the full `gate_v0` run digest2's own NEOCP cut flags 399 tracklets to recover 88% of the NEOs, at 59% contamination. ds42 reaches the same completeness at p ≥ 0.254, flagging 174 tracklets at 6% contamination. That factor is the practical content of the calibration table above: a calibrated posterior can be thresholded against a contamination budget, a population fraction cannot.

## The truncation subset (the "same inputs" caveat)

`d2model.c:hToBin` runs `for (ih = 0; h >= hpart[ih] && ih < HX-1; ++ih)` with `hpart[17] = 25.5`, so every trial orbit with H ≥ 24 falls in the last bin: digest2's top H bin is effectively [24, ∞). ds42 truncates the model at H = 25.5 (D-4). Where a tracklet carries appreciable admissible-region measure above H = 25.5, the two codes are not reading the same effective model and the head-to-head premise fails.

| run | median frac at H>25.5 | >1e-2 | >1e-1 | ΔAUC clean | ΔAUC broken |
|---|---:|---:|---:|---|---|
| gate_v0 | 0.0210 | 64.5% | 2.5% | +0.0072 | +0.0088 |
| enriched_f01 | 0.0206 | 63.2% | 2.9% | +0.0047 | +0.0120 |

## Disagreement census

On `gate_v0`'s clean subset the two rank orders correlate at 0.558; 30.8% of tracklets differ by more than a quarter of the population in percentile rank.

The two tails are not symmetric. The 43 tracklets digest2 ranks furthest above ds42 have median true ρ = 39.9 AU and median true q = 33.9 AU (100% with q > 5 AU, 100% beyond 10 AU) and are 0% true NEOs — distant objects on which digest2 saturates near 100 while ds42 puts them at 1e-7. The 43 tracklets ds42 ranks furthest above digest2 sit at median ρ = 3.5 AU, q = 2.3 AU — ordinary main-belt tracklets, and both codes score them low in absolute terms, so the rank gap there costs little.

That distant-object failure is real but it is not the whole story: dropping the 69 tracklets with true ρ > 10 AU from the clean subset moves ΔAUC only from +0.0072 to +0.0062. The remaining ~85% of the gap is diffuse — it comes from ds42 resolving a continuous posterior where digest2 can only tag bins as reachable or not.

The extremes:

**ds42 ranks far above digest2**

| gen index | true NEO | p_neo (42) | raw NEO (d2) | Δrank | V | H | q | e | ρ [AU] |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1962 | no | 1.08e-03 | 0.10 | +0.733 | 23.59 | 17.05 | 2.635 | 0.170 | 4.648 |
| 15207 | no | 1.70e-03 | 0.19 | +0.724 | 22.48 | 15.91 | 2.727 | 0.139 | 4.386 |
| 17596 | no | 8.92e-04 | 0.07 | +0.702 | 21.97 | 15.92 | 2.441 | 0.193 | 4.114 |
| 17836 | no | 1.13e-03 | 0.15 | +0.699 | 19.65 | 14.08 | 2.755 | 0.104 | 3.781 |
| 19564 | no | 1.10e-03 | 0.15 | +0.696 | 23.78 | 16.95 | 2.985 | 0.255 | 4.994 |
| 7632 | no | 1.14e-03 | 0.16 | +0.693 | 22.44 | 17.99 | 1.967 | 0.111 | 2.878 |
| 5114 | no | 9.45e-04 | 0.13 | +0.679 | 23.32 | 16.84 | 2.907 | 0.095 | 4.232 |
| 2455 | no | 9.63e-04 | 0.13 | +0.678 | 23.20 | 16.89 | 2.444 | 0.176 | 4.261 |
| 10244 | no | 1.33e-03 | 0.21 | +0.665 | 21.45 | 16.92 | 2.123 | 0.034 | 3.108 |
| 1283 | no | 7.86e-04 | 0.09 | +0.650 | 23.33 | 16.80 | 3.330 | 0.183 | 4.362 |

**digest2 ranks far above ds42**

| gen index | true NEO | p_neo (42) | raw NEO (d2) | Δrank | V | H | q | e | ρ [AU] |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1701 | no | 1.48e-07 | 87.30 | -0.978 | 17.92 | 1.56 | 33.931 | 0.162 | 41.428 |
| 11337 | no | 2.55e-07 | 87.83 | -0.977 | 21.01 | 4.90 | 38.609 | 0.021 | 39.274 |
| 11496 | no | 5.80e-07 | 83.16 | -0.975 | 22.49 | 6.44 | 36.562 | 0.026 | 39.275 |
| 4279 | no | 1.82e-07 | 68.23 | -0.974 | 22.80 | 6.43 | 37.028 | 0.193 | 42.285 |
| 14623 | no | 2.26e-06 | 95.72 | -0.972 | 22.67 | 6.42 | 39.996 | 0.043 | 41.124 |
| 18185 | no | 3.26e-08 | 56.18 | -0.970 | 22.58 | 6.56 | 33.767 | 0.198 | 38.831 |
| 6325 | no | 7.15e-07 | 67.86 | -0.970 | 23.11 | 7.62 | 33.239 | 0.050 | 34.535 |
| 33 | no | 1.21e-05 | 98.45 | -0.970 | 22.74 | 4.62 | 36.536 | 0.267 | 63.586 |
| 7015 | no | 2.74e-06 | 89.88 | -0.970 | 23.43 | 6.29 | 39.596 | 0.116 | 50.822 |
| 14510 | no | 8.28e-06 | 94.20 | -0.970 | 23.59 | 8.34 | 32.008 | 0.068 | 32.907 |

## Caveats and options chosen

- Single geocentric observer (obscode 500) at one epoch per tracklet: no topocentric parallax, no site-error diversity, no cadence structure. digest2's per-site error table is therefore never exercised.
- No astrometric noise. digest2 sees exact positions but still assumes its default 1.0 arcsec observational error (d2math.c obsErr; obscode 500 has no entry in MPC.config and ADES rmsRA/rmsDec are left blank, which updateRMSValues floors to that same 1 arcsec). Its motion-vector perturbation therefore explores a rate box that is wider than the truth warrants -- a fixed, favourable-to-neither handicap that widens both codes' effective admissible regions differently. ds42 v0 uses a delta-function attributable (D-6), so it explores no rate box at all.
- digest2 options: repeatable=True (deterministic MC seeding), no_threshold=False (default RMS clamping), bundled MPC.config for per-site errors, all 15 classes computed. Score compared is 'raw' (All population) to match ds42's All variant; 'noid' (Unk) is reported alongside but is a different quantity.
- The two codes use different Earth ephemerides: ds42 uses astropy's, digest2 its internal se2000 series (~1e-4 AU). Tracklet times are written as true UTC; the TDB-UTC offset shifts the observer by ~1e-5 AU, well inside that.
- Two detections per tracklet with an exactly linear arc: digest2's great-circle RMS is 0 by construction, so any RMS-driven behaviour in digest2 is at its floor. Real tracklets have >= 3 detections and non-zero residuals.
- Side effect: on first load digest2 writes a binary render of the CSV as data/digest2.model (11.7 MB, gitignored by data/*). It is derived from our file and is not read in preference to it -- an explicit model_path always wins -- but it is an artifact this harness creates outside its own output directory.
- The test population is drawn from the same model file both codes read, so this measures method under a correct prior. DESIGN §4.3 (v1+) is the prior-misspecification test; nothing here speaks to that.

## Reproduce

```
python -m validation.digest2_comparison run --run validation/output/gate_v0 --out validation/output/digest2_comparison/gate_v0
python -m validation.digest2_comparison run --run validation/output/enriched_f01 --out validation/output/digest2_comparison/enriched_f01
python -m validation.digest2_comparison report --out validation/output/digest2_comparison --case gate_v0 --case enriched_f01
```

---

**Addendum:** the H-clamp emulation experiment is reported in [DIGEST2_COMPARISON_CLAMPED.md](DIGEST2_COMPARISON_CLAMPED.md).

**Erratum (2026-09-02):** all bootstrap CIs in this report were regenerated after the NEOCP pilot found a stratified-resampling bug in auc_bootstrap (negative-stratum indices addressed the positive block). Point estimates were unaffected; CIs widened. See tests/test_digest2_comparison.py for the regression test.
