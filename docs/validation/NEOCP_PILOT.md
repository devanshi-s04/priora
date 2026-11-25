# WP8 — real-data NEOCP pilot (DESIGN §4.3)

*Produced by `validation/neocp_pilot.py`. This is a **prior-misspecification stress test**, not a correctness test: v0's prior is digest2's 2011 S3M-derived histogram [D-10], the real sky is not, and the astrometry is real. Read every calibration number below as a measurement of (prior mismatch + selection + method), in that expected order of size.*

## Headline

- **499 real NEOCP discovery tracklets** rebuilt from MPC astrometry and labelled by mature orbits: **305 NEO / 194 non-NEO** (q < 1.3 AU, D-2). 499 of 500 drawn objects survived the pipeline.
- **v0 did not crash, hang, or fabricate — but it does refuse.** 11 tracklets (2.2%), **100% of them true NEOs**, return `p_neo = nan`: their entire admissible region implies H > 25.5, so both hard-truncated components are zero everywhere and the posterior is 0/0 (DESIGN §2.6, D-4). These are the close, fast, small objects the NEOCP exists for. Scoring with digest2's own [24, inf) top H bin recovers 11 of them and changes nothing else (median |Δp| 2e-06).
- **Calibration survived the prior mismatch**, which was the surprise: predicted/observed NEOs 1.01 (ECE 2.2e-02) against digest2 `raw`/100 at 1.38 (ECE 2.3e-01). The NEOCP cut is a function of the observables alone, so it is exactly the ε(A, V) that DESIGN §2.4 shows cancels in the posterior.
- **Ranking**: AUC 0.9756 (priora All) vs 0.9724 (digest2 raw), ΔAUC +0.0042 [-0.0011, +0.0107], P(Δ ≤ 0) = 0.070 — the same sign and magnitude as WP7's synthetic +0.007, not resolved at this sample size, **and censored against digest2** (the sample is what digest2 already accepted).
- **D-10, All vs Unk: keep `All`.** ΔAUC(All - Unk) = +0.0104 [+0.0039, +0.0187], P(Δ ≤ 0) = 0.000; digest2's own `raw` beats its `noid` on the same tracklets. The `Unk` tables subtract a known-object catalogue frozen at the model's 2011 vintage, which inflates P(NEO) for main-belt-like tracklets rather than deflating it.
- **Operationally**, inside this already-filtered sample: digest2's own cut (`raw >= 65`) flags 402 tracklets at 98.3% completeness and 28.1% contamination; priora matches that completeness at 21.7% contamination, and at its natural p >= 0.5 gives 93.5% completeness at 5.8%.
- **Two data-handling landmines** that no synthetic test could have found: the MPC's ADES export uses **two-character band codes** (`Pw`, `Ao`, `Lr`, `Vj`) that digest2's offset table does not know (593 affected detections, ~0.7 mag each if missed), and **JPL SBDB returns display-rounded elements**, so boundary Amors arrive as exactly `q = 1.3` and get mislabelled. Both are handled and both are documented below.
- **One bug in a shared validation helper** (`digest2_comparison.auc_bootstrap`) that this sample's label balance exposed; reported, not patched — see the Ranking section.

## The sample, and what it is conditioned on

| | |
|---|---|
| label source | MPC `iau/NEO/elsewhere.csv` — every NEOCP posting 2019-03-21 to 2021-12-21 (17901 rows) |
| astrometry | MPC observations API (`data.minorplanetcenter.net/api/get-obs`, ADES dataframe), one object per request |
| orbits / truth | JPL SBDB per object; NEO iff q < 1.3 AU [D-2], cross-checked against MPC `nea_extended.json.gz` |
| eligible pool | 13471 objects (published + designated + non-comet + ground-based discovery station, deduplicated); NEO fraction 0.631 |
| drawn | 500 uniform random (seed in `sample.json`), label-blind |
| built | **499** tracklets: **305 NEO / 194 non-NEO** (observed NEO fraction 0.611) |

**Selection caveats. These are load-bearing, not boilerplate.**

1. **The sample is conditioned on digest2 saying yes.** The MPC posts a tracklet to the NEOCP only when digest2's NEO score clears its policy threshold (score >= 65) and an operator agrees. Every object here has already passed a digest2 cut, so digest2's false negatives are absent *by construction*. Any head-to-head ROC below is **censored against digest2** and must not be read as "which code triages better": it measures only the discrimination each code retains *inside* the set digest2 accepted — i.e. how much of the residual non-NEO contamination each could still remove. Note the asymmetry with calibration: because the cut is a deterministic function of the *observables* (digest2's score is a function of the tracklet, not of the object's class), it is exactly the ε(A, V) that DESIGN §2.4 shows cancels in the posterior — so the calibration question survives this selection even though the ranking comparison does not. That argument is worked through, and confirmed against the data, in the calibration section.
2. **Only designated objects carry a label.** 2594 archive rows are undesignated ("was not confirmed", "does not exist", ITF) and are dropped: their truth is genuinely unknown, and they are plausibly enriched in *both* hard NEOs (fast, lost) and noise. The measured NEO fraction is therefore the fraction *among objects that got an orbit*, not among NEOCP postings.
3. **Comets excluded** (351 rows): the digest2 model has no cometary component, so a comet is neither the NEO component nor the "other" component the two densities partition (DESIGN §5).
4. **Space-based / roving discovery stations excluded** (110 rows: NEOWISE C51, Gaia 258, ...) — no MPC parallax constants, and v0's observer model has no spacecraft ephemeris.
5. **The base rate is real, not balanced.** The draw is label-blind on purpose: a 50/50 sample would have made the calibration question ("does mean p match the observed NEO fraction?") unanswerable.

### Selection accounting

| stage | rows |
|---|---:|
| rows | 17901 |
| not published | 2582 |
| no designation | 12 |
| comet | 351 |
| no obscode | 5 |
| obscode not ground based | 110 |
| duplicate designation | 1370 |
| eligible | 13471 |

## Fit / parse failure census (real data breaks things)

| outcome | n |
|---|---:|
| n sampled | 500 |
| obs file missing | 0 |
| obs empty or error | 0 |
| no obs at discovery obscode | 0 |
| discovery night single detection | 0 |
| orbit file missing | 0 |
| orbit unavailable | 0 |
| orbit q nonfinite | 0 |
| cometary by sbdb kind | 0 |
| no usable magnitude | 1 |
| band aliased | 593 |
| band unknown to digest2 | 0 |
| fit failed | 0 |
| obscode unresolvable | 0 |
| time parse failed | 0 |
| built | 499 |
| window fallback | 0 |
| truncated span | 28 |
| precovery exists | 252 |
| label source disagree | 3 |

Of the 500 drawn objects, **499** yielded a usable discovery tracklet (99.8%); the losses, largest first: `no_usable_magnitude` (1). The failure modes the census watches for, and did **not** see at any scale here, are the ones that would have been most awkward: no parse failure, no time-format failure, no fit failure, no obscode we could not resolve, and no case where the MPC observations database failed to retain two or more published observations at the archive's `discovery_obscode` (the station whose submission triggered the posting). Whatever else this sample stresses, the ingest path is not the fragile part.

**Tracklet geometry actually seen** (medians unless noted):

| quantity | p10 | median | p90 |
|---|---:|---:|---:|
| detections per tracklet | 3.000 | 4.000 | 4.000 |
| arc length [h] | 0.341 | 0.588 | 0.924 |
| linear-fit residual RMS [arcsec] | 0.032 | 0.156 | 0.532 |
| total sky rate [deg/day] | 0.247 | 0.781 | 6.760 |
| V (median over tracklet) | 19.528 | 21.140 | 22.110 |
| true q [AU] | 0.703 | 1.088 | 1.730 |
| digest2 great-circle RMS [arcsec] | 0.029 | 0.151 | 0.512 |

Stations: F51 (211), G96 (152), F52 (53), 703 (18), I41 (16), T05 (12), V00 (11), T08 (8). Native bands before conversion: G (251), Pw (138), w (69), Ao (13), r (11), R (10), Pi (9), g (9), Ac (7), i (4), c (1).

### Two things real data broke that synthetic data could not

**1. Two-character band codes.** The MPC observations API's ADES dataframe reports the photometric band as a *two-character* code for every modern survey — `Pw` (Pan-STARRS w), `Ao`/`Ac` (ATLAS orange/cyan), `Lr`/`Lz` (Rubin), `Vj` (Johnson V) — while MPC 80-column, ADES PSV and ADES XML all carry a single character, which is what digest2's offset table is keyed on. A naive lookup therefore sends those codes to the -0.8 "unknown band" fallback: **593 of the sample's detections** would have taken a 0.6-1.0 mag photometric error, silently. The alias table in `neocp_pilot.MPC_BAND_ALIASES` was verified against the MPC itself by requesting `ADES_DF` and `OBS80` for the same objects and matching the band histograms (`Pw`->`w` 105/105 on 2017 UQ124; `Ac`->`c` 38/38 and `Ao`->`o` 32/32 on 2006 UW157). The order is not uniform — the survey codes are (system, band), the Johnson/Cousins codes are (band, system), so `Vj` is V and not J — which is why the table is explicit rather than a string slice. After aliasing, 0 detections still carry a band digest2 does not know (mostly `UNK`) and keep its -0.8 fallback, as digest2 itself would.

**2. Rounded truth.** JPL's SBDB returns orbital elements as display-rounded strings, so a boundary Amor comes back as exactly `q = "1.3"` and a naive `q < 1.3` test calls it a non-NEO. 3 objects in this sample sit on that boundary. The label therefore uses the MPC's `nea_extended` perihelion distance (7 digits, and its membership *is* the MPC's q < 1.3 definition) where the object appears there, with SBDB as the cross-check — and the disagreement between the two sources is counted rather than hidden.

Provenance flags: 252 tracklets have an *earlier* observation at some other station (a later-linked precovery — excluded from the fit on purpose, since it was not available at triage time), 0 fell back outside the NEOCP-exit window, 28 were truncated to the first 2 h of a longer night, and 3 disagree between the two label sources (SBDB q vs MPC NEA list).

## Did anything in v0 break?

| check | value |
|---|---:|
| non-finite `p_neo` (All / Unk) | 11 / 11 |
| empty admissible region | 0 |
| nodes zeroed by the Jacobian guard, total | 10521 |
| median measure fraction at H > 25.5 | 0.0784 |
| fraction with that > 1e-2 ("broken" premise) | 0.972 |
| fraction with that > 1e-1 | 0.443 |
| clean / broken subset sizes | 5 / 485 |
| `p_neo` exactly 1.0 / exactly 0.0 | 170 / 0 |
| median priora CPU per tracklet [s] | 2.62 |
| digest2 tracklets it declined to score | 0 |

Nothing raised, nothing hung, nothing produced a wrong-looking number silently — but v0 does have one **hard refusal mode**, and real NEOCP data walks straight into it.

### The H-support wall (DESIGN §2.6, D-4) — measured

**11 of 499 tracklets (2.2%) come back with `p_neo = nan`, and 100% of them are true NEOs.** They are not degenerate geometry: the admissible region is non-empty and the fit is clean. They are tracklets whose *entire* admissible region implies H > 25.5 — the minimum measure fraction above the support among them is 0.9856. With both components hard-truncated at the same H (D-4), every node evaluates both densities to zero, Λ_NEO = Λ_other = 0, and the posterior is 0/0. `classify` does the right thing — NaN plus a warning, not a fabricated number — but the tracklet is unscored.

Their signature is exactly the DESIGN §2.6 corner made physical: median sky rate 6.8 deg/day, median V 20.74, median true q 0.938 AU. Close, fast, small — i.e. the objects a NEOCP exists for. digest2 scores every one of them 100, because its top H bin is [24, inf) and never runs out of model. Affected objects: 2019 TK, 2019 UM2, 2019 VH, 2020 GO, 2020 TR6, 2020 UT, 2021 BO2, 2021 JX5, 2021 KQ1, 2021 TG10, 2021 UY.

**The fix already exists in the codebase.** Re-scoring the same tracklets with `Digest2Histogram(h_top_bin_infinite=True)` — digest2's own [24, inf) top H bin, the WP7 clamped control — recovers **11 of the 11** and leaves 0 non-finite, while changing the other tracklets by a median |Δp| of 2.26e-06. The recommendation this pilot makes is therefore concrete: **for NEOCP-like operation on a digest2-derived model, use the clamped pair.** D-4's hard truncation is right in principle (no extrapolation beyond declared support) and right in practice on the synthetic gate, where the corner was a 2.5%-of-measure curiosity; on real discovery tracklets it is a refusal to score the highest-value 2% of candidates. The honest longer-term answer is a population model whose H support reaches the objects surveys actually find (DESIGN §5, v1's NEOMOD3 side), not a clamp.

**Does the locally re-run digest2 reproduce the NEOCP cut?** Of the 499 tracklets it scored, 82.8% reach `raw >= 65` and 88.2% reach `noid >= 65` — 98.4% of the true NEOs and 58.2% of the true non-NEOs. Every one of these objects *was* posted to the NEOCP, so a fraction well below 100% is not a digest2 failure: it measures how far our reconstructed discovery tracklet (and this model file / this digest2 build) is from whatever the MPC scored at the time, plus the operator discretion and the `noid`-vs-`raw` choice in the real posting policy. Treat it as the honest upper bound on how literally the "already digest2-filtered" caveat applies to *this* reconstruction.

## Score distributions by true label

| score | label | median | mean | p10 | p90 |
|---|---|---:|---:|---:|---:|
| priora All | neo | 1 | 0.9245 | 0.751 | 1 |
| priora All | non_neo | 0.0248 | 0.1227 | 0.001628 | 0.4629 |
| priora Unk | neo | 1 | 0.9235 | 0.6838 | 1 |
| priora Unk | non_neo | 0.01913 | 0.1519 | 0.001614 | 0.6405 |
| priora All H-clamped | neo | 1 | 0.9275 | 0.7649 | 1 |
| priora All H-clamped | non_neo | 0.02481 | 0.123 | 0.00163 | 0.4632 |
| digest2 raw (All) | neo | 100 | 97.72 | 97.38 | 100 |
| digest2 raw (All) | non_neo | 68.58 | 63.28 | 25.76 | 92.16 |
| digest2 noid (Unk) | neo | 100 | 97.67 | 97.51 | 100 |
| digest2 noid (Unk) | non_neo | 78.65 | 71.44 | 28.17 | 96.5 |

![score distributions](score_hist.png)

## Ranking (ROC/AUC) — with the censoring caveat above

| subset | n | NEOs | priora All | priora Unk | priora All H-clamped | digest2 raw (All) | digest2 noid (Unk) |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 499 | 305 | 0.9756 | 0.9652 | 0.9765 | 0.9724 | 0.9539 |
| broken | 485 | 298 | 0.9774 | 0.9662 | 0.9782 | 0.9749 | 0.9563 |

Paired, label-stratified bootstrap on the differences (`neocp_pilot.paired_auc_bootstrap`). Note that a *paired* comparison must drop any tracklet either score cannot rank, so the `All-clamped vs All` row is computed on the 488 tracklets where both are finite — on which the two are all but identical. The 0.0009 gap between their single-score AUCs in the table above is entirely the 11 H-wall NEOs that only the clamped pair scores at all.

| subset | comparison | ΔAUC [95% CI] | P(Δ ≤ 0) |
|---|---|---|---:|
| all | priora All vs priora Unk | +0.0104 [+0.0039, +0.0187] | 0.000 |
| all | priora All clamped vs digest2 raw | +0.0041 [-0.0013, +0.0104] | 0.069 |
| all | priora All clamped vs All | +0.0000 [-0.0001, +0.0001] | 0.606 |
| all | priora All vs digest2 raw | +0.0042 [-0.0011, +0.0107] | 0.070 |
| all | priora Unk vs digest2 noid | +0.0128 [+0.0052, +0.0217] | 0.000 |
| broken | priora All vs priora Unk | +0.0112 [+0.0044, +0.0200] | 0.000 |
| broken | priora All clamped vs digest2 raw | +0.0033 [-0.0016, +0.0094] | 0.113 |
| broken | priora All clamped vs All | +0.0000 [-0.0001, +0.0001] | 0.597 |
| broken | priora All vs digest2 raw | +0.0035 [-0.0016, +0.0096] | 0.104 |
| broken | priora Unk vs digest2 noid | +0.0115 [+0.0039, +0.0202] | 0.000 |

> **Bug found in `validation/digest2_comparison.py:auc_bootstrap`, reported not patched** (that file is outside this task's ownership). Its resample builds `take = concatenate((integers(0, n_pos, n_pos), integers(0, n_neg, n_neg)))` and applies it to `concatenate((pos, neg))` **without offsetting the negative half by `n_pos`**, so the negative indices address the positive block. When `n_pos >= n_neg` — which is this sample, 305 NEOs against 194 non-NEOs, because the NEOCP is already digest2-filtered — every resample is 100% positives, every bootstrap AUC is NaN, the CI comes back `[nan, nan]` and `p_delta_le_0` silently reports `0.000` (because `NaN <= 0` is `False`). In WP6/WP7's regime (185 NEOs in 2x10^4) the sign is the other way and the effect is small but nonzero: the negative half is drawn from a block that is `n_pos/n_neg` ~ 0.9% positives, so those published intervals come from a slightly wrong resampling distribution. The numbers above use this module's own corrected implementation; `DIGEST2_COMPARISON.md`'s CIs should be regenerated once the shared helper is fixed.

![roc](roc.png)

![scatter](scatter.png)

## Calibration — the prior-mismatch measurement

| subset | score | predicted NEOs | observed | pred/obs | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| all | priora All | 295.6 | 294.0 | 1.01 | 5.365e-02 | 2.232e-02 |
| all | priora Unk | 301.0 | 294.0 | 1.02 | 6.751e-02 | 3.808e-02 |
| all | priora All H-clamped | 306.7 | 305.0 | 1.01 | 5.244e-02 | 2.205e-02 |
| all | digest2 raw/100 | 420.8 | 305.0 | 1.38 | 1.854e-01 | 2.321e-01 |
| all | digest2 noid/100 | 436.5 | 305.0 | 1.43 | 2.309e-01 | 2.635e-01 |

![calibration](calibration.png)

**How to read this — and the pilot's most interesting result.** The expectation going in was that calibration *had* to degrade here: the sample is a digest2-selected slice of the sky, and v0's prior is a 2011 population model. It did not. priora's posterior predicts 295.6 NEOs against 294.0 observed (ratio 1.01, ECE 2.23e-02), while digest2's `raw`/100 predicts 420.8 against the same truth (ratio 1.38, ECE 2.32e-01).

The reason the posterior survives the selection is worth stating precisely, because it is the DESIGN §2.4 argument doing real work for the first time. §2.4 requires the selection to be a function ε(A, V) of the **observables** — then it multiplies both Λ_NEO and Λ_other and cancels in the ratio. The NEOCP cut is `digest2_score(A, V) >= 65`: digest2 is a deterministic function of the tracklet, so the cut is *exactly* of that form. Conditioning on "digest2 said yes" therefore changes **which** (A, V) we see, not P(NEO | A, V) — and a calibrated posterior stays calibrated on the selected subsample. Ranking is a different matter: censoring removes digest2's own false negatives from the sample, and no cancellation argument repairs a ROC computed on a set chosen by one of the two contestants (caveat 1).

What the selection does *not* excuse, and what this table therefore still measures, is prior mismatch — and its verdict is that in the regime this sample occupies (median V 21.1, median arc 0.59 h, true q 0.15 to 4.99 AU) the mismatch between digest2's 2011 S3M populations and the real 2019-2021 sky is **mild enough not to break the posterior**. Two honest asterisks: the label censoring (undesignated postings, caveat 2) is *not* a function of (A, V) and does not cancel; and the H-support wall above removes the 11 true NEOs it cannot score from the All row's denominator, which is why the H-clamped row's observed count is 305 and the All row's is 294.

## All vs Unk: which variant belongs in NEOCP-like use? (D-10)

NEOCP candidates are by construction *undiscovered* objects, which is exactly the population the model's `Unk` (total minus known) tables describe; digest2's own default score (`noid`) uses `Unk` for the same reason, while its `raw` score uses `All`. v0 defaults to `All`. Both were scored here on identical inputs.

| variant | pred/obs | ECE | Brier | AUC (all) |
|---|---:|---:|---:|---:|
| priora All | 1.01 | 2.232e-02 | 5.365e-02 | 0.9756 |
| priora Unk | 1.02 | 3.808e-02 | 6.751e-02 | 0.9652 |
| priora All H-clamped | 1.01 | 2.205e-02 | 5.244e-02 | 0.9765 |

**The answer is All, and it is not close.** ΔAUC(All - Unk) = +0.0104 [+0.0039, +0.0187], bootstrap P(Δ ≤ 0) = 0.000 — and digest2 shows the same ordering on its own two scores (`raw` 0.9724 vs `noid` 0.9539), so this is a property of the model file rather than of either method. That contradicts the a-priori argument, and the reason is instructive: the `Unk` tables are *total minus known*, where "known" is the discovered catalogue **frozen at the model's 2011 vintage**. Subtracting it removes a large share of the bright main belt and almost none of the NEOs, which inflates P(NEO) for exactly the main-belt-like tracklets that make up this sample's negatives. The premise — "a NEOCP candidate is undiscovered, so use the undiscovered population" — needs the *current* known catalogue to hold; with a 15-year-stale one it does the opposite of what it intends. **Recommendation for D-10: keep `All` as the v0 default, and treat `Unk` as usable only with a contemporaneous known-object subtraction.**

![variant](variant_scatter.png)

## Operating points at digest2's own cut

*Within the already-digest2-selected sample, so "completeness" here is completeness relative to the NEOs that reached the NEOCP, not to the sky.*

| subset | pair | operating point | flagged | completeness | contamination |
|---|---|---|---:|---:|---:|
| all | All/raw | digest2 raw >= 65 | 402 | 98.3% | 28.1% |
| all | All/raw | priora p >= 0.5 | 292 | 93.5% | 5.8% |
| all | All/raw | priora p >= 0.0403 | 369 | 98.3% | 21.7% |
| all | Unk/noid | digest2 raw >= 65 | 429 | 98.3% | 32.6% |
| all | Unk/noid | priora p >= 0.5 | 297 | 93.5% | 7.4% |
| all | Unk/noid | priora p >= 0.03 | 374 | 98.3% | 22.7% |
| all | All-clamped/raw | digest2 raw >= 65 | 413 | 98.4% | 27.4% |
| all | All-clamped/raw | priora p >= 0.5 | 303 | 93.8% | 5.6% |
| all | All-clamped/raw | priora p >= 0.0403 | 380 | 98.4% | 21.1% |

The WP7 synthetic result was a ~10x contamination reduction at matched completeness. Here it is much smaller — 28.1% -> 21.7% at matched 98.3% completeness — and that shrinkage is the censoring at work rather than a weaker method: the easy non-NEOs that made up most of WP7's false-positive budget were never posted to the NEOCP, so they are not in this sample to remove. The operationally useful number is the other row: priora at its own p >= 0.5 flags 292 tracklets, 93.5% complete at 5.8% contamination — a ~5x contamination reduction for 4.8% of completeness, and unlike a digest2 score that threshold can be *chosen* against a stated contamination budget.

## Behaviour by true perihelion distance

| q bin | n | mean p (All) | mean p (Unk) | frac p≥0.5 | mean d2 raw | frac d2≥65 |
|---|---:|---:|---:|---:|---:|---:|
| NEO (q<1.3) | 305 | 0.924 | 0.923 | 0.90 | 97.7 | 0.98 |
| Mars-crosser (1.3-1.67) | 114 | 0.184 | 0.232 | 0.13 | 70.9 | 0.70 |
| inner belt (1.67-2.5) | 71 | 0.0378 | 0.0374 | 0.03 | 52.5 | 0.42 |
| outer belt+ (>=2.5) | 9 | 0.0208 | 0.0351 | 0.00 | 52.2 | 0.33 |

## Disagreement census

Percentile-rank correlation between priora (All) and digest2 (raw): **0.973**; 0.6% of tracklets differ by more than a quarter of the sample in rank. That is far tighter than WP7's 0.558 on synthetic tracklets, for a mechanical reason: the NEOCP cut has already removed the population where the two codes diverge most (WP7's distant-object tail), so what is left is mostly the region where both agree.

The residual tails still carry WP7's signature. The tracklets digest2 ranks furthest above priora are dominated by **slow, distant** objects — Hilda/Trojan/Centaur-like, q ~ 4.6-5.0 AU, rates ~0.1 deg/day — which digest2 saturates near 75-80 and priora puts at 1e-4, exactly the failure mode WP7 identified. In the other tail priora sits above digest2 on Mars-crossers and high-e inner-belt objects, where both codes are low in absolute terms and the rank gap costs little. Note also the tracklets where both saturate (p = 1.0 and raw = 100): they appear in these tables only because saturation ties are broken by rank, and carry no information.

**digest2 ranks far above priora**

| designation | true NEO | p_neo (42) | raw (d2) | Δrank | V | q | e | i | rate [°/d] | n | stn |
|---|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2008 KC21 | no | 3.779e-04 | 75.6 | -0.264 | 21.14 | 4.980 | 0.051 | 30.1 | 0.14 | 4 | F51 |
| 2010 GY177 | no | 1.386e-03 | 77.6 | -0.264 | 21.45 | 4.620 | 0.103 | 17.8 | 0.08 | 4 | F51 |
| 2021 UK13 | no | 8.913e-05 | 69.8 | -0.225 | 22.18 | 4.800 | 0.085 | 27.4 | 0.14 | 3 | F51 |
| 2019 OE1 | no | 2.339e-03 | 76.0 | -0.223 | 20.54 | 1.620 | 0.118 | 25.1 | 0.53 | 3 | F51 |
| 2020 HN6 | yes | 1.000e+00 | 100.0 | -0.178 | 19.84 | 1.146 | 0.104 | 22.0 | 2.46 | 4 | G96 |
| 2021 PP4 | no | 4.059e-03 | 74.8 | -0.172 | 21.64 | 1.810 | 0.060 | 24.8 | 0.45 | 3 | F51 |
| 2021 CH5 | yes | 1.000e+00 | 100.0 | -0.163 | 21.64 | 0.950 | 0.482 | 1.1 | 2.92 | 2 | G96 |
| 2020 PO1 | yes | 1.000e+00 | 100.0 | -0.163 | 21.23 | 1.108 | 0.545 | 2.7 | 0.91 | 4 | F51 |
| 2020 SQ3 | yes | 1.000e+00 | 100.0 | -0.163 | 20.54 | 0.990 | 0.076 | 18.3 | 6.92 | 4 | G96 |
| 2019 TM6 | yes | 1.000e+00 | 100.0 | -0.163 | 21.74 | 0.875 | 0.611 | 7.9 | 1.23 | 7 | G96 |

**priora ranks far above digest2**

| designation | true NEO | p_neo (42) | raw (d2) | Δrank | V | q | e | i | rate [°/d] | n | stn |
|---|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20386 | no | 1.669e-01 | 36.0 | +0.256 | 18.29 | 2.630 | 0.075 | 14.6 | 0.26 | 6 | 703 |
| 2015 PK57 | yes | 8.283e-02 | 27.1 | +0.242 | 23.04 | 0.961 | 0.619 | 4.7 | 0.19 | 3 | F51 |
| 2020 HX6 | no | 1.916e-01 | 52.1 | +0.217 | 21.49 | 1.370 | 0.267 | 10.6 | 0.47 | 4 | F51 |
| 2021 YG | yes | 1.000e+00 | 100.0 | +0.204 | 22.57 | 1.200 | 0.479 | 20.8 | 1.01 | 4 | F52 |
| 2020 PS7 | no | 5.319e-02 | 33.9 | +0.191 | 21.65 | 1.550 | 0.414 | 6.4 | 0.49 | 3 | F52 |
| 2020 KD6 | no | 2.781e-02 | 26.7 | +0.178 | 22.74 | 1.680 | 0.870 | 38.7 | 0.55 | 3 | F51 |
| 2020 OG8 | no | 8.637e-02 | 51.7 | +0.178 | 22.56 | 1.470 | 0.470 | 10.9 | 0.10 | 4 | F51 |
| 2020 FZ2 | yes | 1.000e+00 | 100.0 | +0.166 | 20.74 | 0.547 | 0.713 | 2.1 | 4.38 | 3 | F51 |
| 2021 WQ4 | yes | 1.000e+00 | 100.0 | +0.166 | 21.04 | 1.038 | 0.582 | 2.3 | 2.80 | 4 | G96 |
| 2021 GE3 | yes | 1.000e+00 | 100.0 | +0.166 | 17.61 | 0.691 | 0.579 | 34.4 | 17.78 | 4 | I41 |

## Truncation diagnostic (DESIGN §2.6(b))

Median admissible-region measure fraction with implied H > 25.5: **0.0784**; 97.2% of tracklets exceed 1e-2 and 44.3% exceed 1e-1. The WP7 "clean" subset (< 1e-3, where digest2's [24, inf) top H bin and priora's hard truncation at 25.5 really do read the same effective model) holds **5** of 499 tracklets here, against 22% of the synthetic gate catalogue. That is a finding rather than an inconvenience: real NEOCP discovery tracklets are faint and fast, so the implied-H range they sweep routinely runs past the model's support, and the DESIGN §4.2 "same inputs" premise is broken for nearly all of them. The WP7 clamped control (`DIGEST2_COMPARISON_CLAMPED.md`) is the mechanism that repairs the premise, and it is run here as a third priora pass: with digest2's top H bin on both sides no tracklet is scored against a different effective model, and the ranking barely moves (ΔAUC(clamped - All) = +0.0000 [-0.0001, +0.0001]) — the clamp's value here is that it *scores at all* the 2% of candidates the truncated pair refuses, not that it re-ranks anything.

![truncation](truncation.png)

## Assumptions and known limitations

- **Delta-function attributable** [D-6]: v0 ignores the fitted attributable's covariance. Real discovery tracklets here are much shorter than the 45-minute arcs of the ASTROMETRIC_NOISE study (median arc 0.59 h), and rate error scales as 1/dt, so the "covariance marginalisation is not urgent" conclusion of that study does **not** transfer to this regime. This is the single largest untested assumption in the pilot.
- **Unweighted linear fit** of alpha(t), delta(t): ADES per-observation uncertainties are reported inconsistently across this archive, so weighting a 3-point fit by sporadic sigmas would make the estimator depend on which observer happened to report them.
- **Photometry**: median of the band->V converted magnitudes over the tracklet, using digest2's own offset table verbatim (`digest2/observation.py:_BAND_CORRECTIONS`). No colour or taxonomy coupling [D-5]; the PSV handed to digest2 carries the already-converted V with band `V`, so digest2's conversion is a no-op and both codes consume the identical magnitude.
- **Observatory geolocation**: MPC parallax constants -> ITRS -> WGS84 geodetic, then passed to `classify(lon_deg=..., lat_deg=..., height_m=...)`. The unit tests check this reproduces every entry of `priora.attributable.MPC_OBSCODES` to sub-metre.
- **Times** are UTC as published; priora converts to TDB internally.
- **No light-time correction** anywhere, matching v0's convention.
- **The discovery tracklet is a reconstruction**, not a record of what digest2 saw at NEOCP time: the MPC back-fills the NEOCP `trksub` onto every observation once linked, so the tracklet is identified by (discovery station, earliest night inside the NEOCP-exit window). Astrometry may also have been re-reduced since.
- **The labels are today's orbits.** An object designated as a main-belter on a 3-night arc in 2019 has had 7 years of follow-up; its q is reliable, but it is not the q anybody knew at triage time.

## Reproduce

```
PRIORA_BACKEND=python python -m validation.neocp_pilot all --n 500 --workers 90
```

`fetch` is the only networked stage and it caches every byte under `raw/`; `build`, `score` and `analyze` re-run offline.


---

**Superseded behavior note (2026-09-02, D-19):** the "H-support wall" section above describes v0 behavior at the time of the pilot: 11 fully-truncated tracklets returned p_neo = nan. Under D-19 the classifier now (a) supports a finite H extension (`h_extend_to=30`, the operational default, which scores all 11 as genuine posteriors) and (b) assigns p_neo = 1.0 by explicit, status-flagged triage policy to any tracklet whose entire admissible region still lies above the support end — a policy whose empirical basis is exactly this section's 11/11 measurement. The pilot driver's detector was updated accordingly (the both-Lambda-zero triple AND the >=0.95 finite-H truncation-measure criterion, imported from the classifier).
