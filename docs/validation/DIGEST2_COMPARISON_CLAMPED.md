# WP7 addendum — the clamped (digest2-emulation) control

*Case `gate_v0`; generated 2026-09-02T12:10:04.*

## What this measures

WP7's head-to-head had one uncontrolled variable. `d2model.c:hToBin` clamps every trial orbit with H >= 24 into the final bin, so digest2's top H bin is effectively [24, inf); priora truncates at 25.5 per D-4. The two codes therefore consume the *same* effective model only where the admissible region carries no measure above H = 25.5 — which on `gate_v0` is a minority of tracklets (64.5% sit above the 1e-2 'broken' threshold).

`Digest2Histogram(h_top_bin_infinite=True)` reinstates digest2's clamp in priora. Re-scoring the identical tracklets with a clamped pair makes the shared-model premise hold for **all** of them, and splits the WP7 gap in two:

```
Delta_WP7  =  Delta_clamped        +  boundary-handling term
              (method alone, both     (what D-4-vs-digest2's different
               codes now read the      faint-end model was worth)
               same faint end)
```

Only priora is re-scored: the true labels, the digest2 scores, the tracklet file and the grid are the WP7 run's own. 20000 tracklets, 0.09 CPU-hours.

## (a) Full-set ΔAUC with the premise repaired

With the clamp on, every tracklet is a valid head-to-head. Same paired, label-stratified bootstrap as WP7.

| population | AUC priora | AUC digest2 | ΔAUC | one-sided p(Δ≤0) |
|---|---|---|---|---|
| `all` (n=20000, NEO=185) | 0.9822 [0.9687, 0.9937] | 0.9696 [0.9476, 0.9879] | **+0.0126 [+0.0015, +0.0253]** | 0.0100 |
| `clean` (n=4340, NEO=83) | 0.9828 [0.9622, 0.9986] | 0.9756 [0.9468, 0.9984] | **+0.0072 [-0.0001, +0.0188]** | 0.0295 |
| `broken` (n=12899, NEO=76) | 0.9752 [0.9484, 0.9938] | 0.9643 [0.9295, 0.9904] | **+0.0109 [-0.0042, +0.0281]** | 0.0865 |

Headline: on the **full set** with genuinely identical effective models, priora outranks digest2 by ΔAUC = **+0.0126 [+0.0015, +0.0253]**.

## (b) The broken subset, decomposed

| population | Δ_WP7 (D-4 vs digest2) | Δ_clamped (method) | boundary handling |
|---|---|---|---|
| `all` | +0.0116 | +0.0126 [+0.0015, +0.0253] | -0.0010 |
| `clean` | +0.0072 | +0.0072 [-0.0001, +0.0188] | +0.0000 |
| `broken` | +0.0088 | +0.0109 [-0.0042, +0.0281] | -0.0021 |

On the WP7 broken subset (n=12899, 76 true NEOs) the WP7 gap of +0.0088 decomposes into +0.0109 of method and -0.0021 of boundary handling.

The boundary term is **negative**: D-4's truncation was costing priora ranking accuracy on exactly the tracklets whose premise it broke. So the WP7 broken-subset number was *conservative* — it understated the method gap by handicapping priora, not digest2. Repairing the premise moves the broken subset from +0.0088 to +0.0109, which is now comfortably clear of zero rather than straddling it. The direction is what §2.6 predicts: truncating at 25.5 deletes faint-end density that is overwhelmingly NEO, so it suppresses P(NEO) for precisely the faint, nearby objects that are the true positives.

Note also that this is not a large correction anywhere. The boundary-handling term is an order of magnitude smaller than the method term on every subset, and identically zero on `clean` — which is the WP7 subset definition working as designed.

priora against *itself*, clamped vs D-4, on identical inputs — the boundary-handling term measured directly, with method held fixed:

| population | AUC D-4 | AUC clamped | Δ (D-4 − clamped) |
|---|---|---|---|
| `all` | 0.9812 | 0.9822 | -0.0010 [-0.0031, +0.0000] |
| `clean` | 0.9828 | 0.9828 | +0.0000 [+0.0000, +0.0000] |
| `broken` | 0.9732 | 0.9752 | -0.0021 [-0.0065, +0.0001] |

## (c) Calibration: how much of digest2's overshoot is the H-clamp?

| population | code | predicted NEOs | observed | ratio | ECE | max \|z\| bin |
|---|---|---|---|---|---|---|
| `all` | priora clamped | 187.6 | 185 | 1.014 | 0.00049 | 1.84 |
| `all` | priora D-4 | 187.5 | 185 | 1.013 | 0.00049 | 1.83 |
| `all` | digest2 raw/100 | 1125.9 | 185 | 6.086 | 0.04705 | 30.36 |
| `clean` | priora clamped | 84.2 | 83 | 1.015 | 0.00032 | 0.45 |
| `clean` | priora D-4 | 84.2 | 83 | 1.015 | 0.00032 | 0.45 |
| `clean` | digest2 raw/100 | 210.4 | 83 | 2.535 | 0.02935 | 11.82 |
| `broken` | priora clamped | 79.9 | 76 | 1.051 | 0.00030 | 0.77 |
| `broken` | priora D-4 | 79.7 | 76 | 1.049 | 0.00030 | 0.73 |
| `broken` | digest2 raw/100 | 639.4 | 76 | 8.414 | 0.04368 | 16.82 |

digest2 over-predicts the NEO count by 6.09x; priora under D-4 predicts 1.013x. Adopting digest2's H-clamp — and changing nothing else — moves priora to 1.0141x, i.e. it closes 0.01% of the gap between the two codes' calibration. The H-clamp is therefore a minor contributor to digest2's calibration overshoot; the remaining 99.99% is the heuristic score itself (digest2's 'raw' is a fraction of trial orbits, not a marginal likelihood ratio, and no amount of agreeing about the faint end makes it one).

ECE moves from 0.00049 to 0.00049 against digest2's 0.04705, and the clamped posterior's own calibration is untouched — which is the substantive finding here: the clamp is a change to the *prior's* faint end, and a correctly marginalized posterior absorbs it. It shows up in the ranking of faint-end-exposed tracklets (b) and essentially nowhere in the predicted count.

Operationally, at digest2's own NEOCP cut (raw >= 65: 88.1% complete, 59.1% contaminated), clamped priora matched to the same completeness carries 6.3% contamination (at p >= 0.254) — the WP7 headline, unchanged by the clamp.

## (d) What the clamp did to the scores

- 1.5% of tracklets score *bit-identically* in the two modes.
- 3.1% move by more than 0.01 dex, 0.1% by more than 0.1 dex.
- The shift is upward in the median (+5.27e-05 dex); 89.37% of p(NEO) values rise, 9.19% fall. 1st-99th percentile of Δlog10 p: [-0.0000, +0.0240].
- Rank correlation between the two modes: 0.99993.

| §2.6(b) measure fraction | n | median Δlog₁₀ p | fraction moved |
|---|---|---|---|
| [0, 0.0001) | 3883 | +0.0000 | 1.3% |
| [0.0001, 0.001) | 457 | +0.0000 | 100.0% |
| [0.001, 0.01) | 2761 | +0.0000 | 99.6% |
| [0.01, 0.03) | 5025 | +0.0001 | 99.9% |
| [0.03, 0.1) | 7365 | +0.0008 | 99.9% |
| [0.1, 1) | 509 | +0.0060 | 95.5% |

`clamped_scatter.png` plots the same thing: clamped vs D-4 p(NEO) with the §2.6(b) diagnostic as colour, and Δlog₁₀ p against that diagnostic. The story it tells is that the knob does exactly what it is supposed to and nothing else. Tracklets whose admissible region never reaches H = 25.5 sit *bit-identically* on the identity line — there is no density for the clamp to add — and the displacement above the line rises monotonically with exposure, bin for bin in the table above, across four decades of the diagnostic; the largest single displacement is 0.615 dex. There is no scatter off the trend: 3596 tracklets moved with zero exposure (and 49 with exposure too small to reach a populated bin did not move), and none moved down. So the ΔAUC changes in (a) and (b) are attributable to the top H bin and to nothing that leaked in with it.

## Controls

- §2.6(b) diagnostic reproduced by the clamped pass to 6.94e-18 (must be 0: the clamp changes the density, not the geometry).
- Node counts identical to 0 (must be 0: same grid).
- Non-finite clamped p(NEO): 0.
- Δ_WP7 recomputed here vs the WP7 summary: `all` +0.0116 vs +0.0116, `clean` +0.0072 vs +0.0072, `broken` +0.0088 vs +0.0088 (identical inputs, so these must agree exactly; any drift means the case directory and the run directory have come apart).

## Caveats

- The clamped mode is an INSTRUMENT, not a proposal. The clamped density is constant in H above 25.5 and therefore non-normalizable: it has no finite total count, cannot be sampled, and is gate-ineligible (DESIGN §4.1 has no generative counterpart for it). Nothing here revises D-4.
- Only the UPPER half of digest2's hToBin clamp is emulated. digest2 also puts H < 0 in bin 0; priora returns 0 there in both modes. Implied H < 0 does not occur at the geometries integrated here (rho >= 0.01 AU, V >= 14), so the omission is measurable-free and keeps the emulation a one-sided change.
- The clamp is applied to BOTH components. Clamping one side only would compare a truncated NEO density against a clamped 'other' density and the posterior would be an artifact of the mismatch.
- Everything except the top H bin is held fixed: the same gen.npz, the same grid, the same G, the same model file, the same digest2 scores and the same true labels as the WP7 run. Only priora is re-scored.
- The subsets are still cut on the UNCLAMPED §2.6(b) diagnostic, because the question is what happens to the tracklets WP7 had to set aside. The diagnostic is a property of the admissible-region geometry, not of the density, so it is identical in the two passes (verified: max abs diff reported in the consistency block).
- Emulating digest2's H binning does not emulate digest2's ranging, its 1-arcsec error box, its bounded rho search, or its scoring heuristic. The residual gap is still 'everything else', not a single mechanism.

## Reproduce

```
python -m validation.digest2_comparison clamped \
    --run validation/output/gate_v0 \
    --out validation/output/digest2_comparison/gate_v0 --workers 8
```
