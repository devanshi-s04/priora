# v0 Self-Consistency Gate — Results of Record

*DESIGN §4.1 gate (IMPLEMENTATION §6 item 2). Harness: `validation/self_consistency.py`
at commit `4cc81a6` (post-adversarial-review). Classifier grid (1200, 50). Raw outputs
live under `validation/output/` (gitignored); this file is the durable record.*

## Gate run (of record): **OVERALL PASS**

`run --n 20000 --seed 20260901 --workers 8 --gate --out validation/output/gate_v0`
(2026-09-01, code `4cc81a6`, model `data/digest2.model.csv` sha256 `2b9d9d2a…`, All variant)

| Criterion | Value | Verdict |
|---|---|---|
| Global NEO count | 185 obs vs 187.5 exp, Z = −0.44 (p = 0.74) | PASS |
| Fitted odds scale | ĉ = 0.923, 95% CI [0.645, 1.321] | consistent with 1 |
| Reliability (bootstrap, 5 equal-expected-NEO bins) | max\|z\| = 0.72 vs threshold 2.76 (p = 0.68) | PASS |
| HPD rank uniformity (both estimators, D ≤ 0.075) | D = 0.0537 / 0.0541, gap 0.0004 | PASS |
| Collapse coverage | truth in admissible region: 100%; median mass within 2× of true ρ: 0.9999 | PASS |
| Domain / defined / truth-density | 0 out-of-domain, 0 undefined, integrand > 0 at truth: 100% | PASS |
| Ranking (not gated) | AUC 0.9812, 0 impossible events | — |

Generation bookkeeping matched predictions: unbound-draw discard 2.14% (NEO) /
8×10⁻⁶ (other) vs the WP4-predicted 1.92% / ~10⁻⁵; V-cut acceptance 7.9% / 16.7%;
`drawn == accepted + rejected` exact.

## Enriched low-p diagnostic (not gate-eligible): PASS (indicative)

`run --n 10000 --seed 20260902 --neo-fraction 0.1` — 424 true NEOs, corrected
product-of-weights variance. Z_tot = +1.54 (p = 0.12); ĉ = 1.174 [0.930, 1.483];
reliability max|z| = 1.53 vs threshold 7.40; HPD/collapse/coverage all PASS.

The two runs' ĉ intervals straddle unity from opposite sides (0.92 vs 1.17) —
no evidence of a global odds bias. The −2.1σ NEO-deficit hint from the WP6
review's exploratory runs did not persist under the corrected statistics.

## Standing caveats (DESIGN §4.1 "what the gate validates")

1. P(NEO) is invariant under global Jacobian rescaling with both components in one
   parameterization — the gate validates the measure's local ρ-shape (near-delta
   posteriors, HPD uniformity), not its absolute normalization (that is pinned by the
   WP2 Delaunay closed-form tests).
2. At N = 2×10⁴ (≈185 NEOs) the odds scale is constrained to roughly ±35%; the
   N = 10⁵ calibration run tightens this to ~±15%.
3. §2.6(b) truncation diagnostic on the gate population: median 2.1% (mean 2.8%) of
   admissible-region measure at implied H > 25.5; 64.5% of tracklets above 1%,
   2.5% above 10%. The Mars-crosser corner {q > 1.3, H > 25.5} carries zero posterior
   mass by construction in v0. This is the population WP7 must handle (digest2 clamps
   H ≥ 24; ds42 truncates at 25.5).

## N = 10⁵ calibration run: **OVERALL PASS**

`run --n 100000 --seed 20260903 --workers 8 --gate --gate-min-n 100000`
(2026-09-01, out `validation/output/calibration_1e5`; 800 true NEOs, 1 undefined score)

- Global NEO count: 799 obs vs 794.1 exp, Z = +0.39 (p = 0.72) — PASS.
- **Fitted odds scale ĉ = 1.031, 95% CI [0.884, 1.203]** — the calibration claim at the
  ±15% level DESIGN §4.1 said requires N ~ 10⁵. Together with the gate run (0.92) and
  enriched run (1.17), three independent estimates bracket unity.
- Reliability, HPD (D = 0.0468/0.0564, gap 0.0097), coverage, domain, truth-density:
  all PASS.
- Unbound-discard 1.94% (NEO) — third independent confirmation of the WP4 figure.

## Rust backend acceptance (2026-09-02)

The Rust core kernel (D-15, issue #1) re-ran the gate with the same seed
(`validation/output/gate_v0_rust`): **OVERALL PASS with statistics identical to the run
of record** (Z_tot = −0.4384, ĉ = 0.923 [0.645, 1.321], max|z| = 0.721,
HPD D = 0.05374/0.0541, gap 0.0004; AUC equal to 15 decimal places). Golden A/B over all
20,000 stored tracklets: max |Δp_neo| = 7.7×10⁻¹³, Λ relative error ≤ 1.7×10⁻⁹,
n_masked_det identical (397,900) with zero disagreements across 8.0×10⁸ nodes
(`validation/output/backend_parity/`). Performance: 12.0 ms/tracklet single-core
(Python: 482 ms), 0.40 ms/tracklet batched at 96 threads.
