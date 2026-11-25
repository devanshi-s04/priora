# Extended-model re-run of the paper's results (h_extend_to = 30)

*2026-10-14. The paper's Methods (§2.8) says all results use the D-19 extended model. Before this re-run, the N = 10⁵
calibration run, the digest2 comparison and the astrometric-noise studies existed only for the unextended model. This
file is the record of the extended versions. Code at `a4bc848` (clean except paper figures), Rust backend,
grid (1200, 50), ρ ∈ [0.01, 100] AU, model sha256 `2b9d9d2a…`, variant All. Outputs in `validation/output/*_hext30*`
(gitignored).*

## §5 Self-consistency

| run | dir | N | NEOs obs / exp | Z_tot | ĉ [95%] | max\|z\| (99% thr) | verdict |
|---|---|---|---|---|---|---|---|
| gate (seed 20260901) | `gate_v0_hext30` (2026-09-02, pre-existing) | 2×10⁴ | 170 / 166.1 | +0.71 | 1.134 [0.803, 1.601] | 0.97 (2.64) | PASS |
| calibration A (seed 20260903) | `calibration_1e5_hext30` | 10⁵ | 851 / 875.7 | −2.00 (p = 0.047) | 0.842 [0.712, 0.997] | 1.81 (2.81) | PASS |
| calibration B (seed 20260904) | `calibration_1e5_hext30_seed2` | 10⁵ | 868 / 868.4 | −0.03 | 0.997 [0.851, 1.169] | 0.40 (2.78) | PASS |
| **A + B pooled** | — | 2×10⁵ | 1719 / 1744.1 | −1.44 | **0.919 [0.819, 1.031]** | — | — |
| enriched f = 0.1 (seed 20260902) | `enriched_f01_hext30` | 10⁴ | 80.3 / 79.3 (weighted) | +0.38 | 1.039 [0.722, 1.494] | 0.35 (3.86) | PASS (indicative) |

- **Run A's ĉ interval just excludes 1. It is a fluctuation, not a bias:**
  1. The independent run B is centred on 1.
  2. The deficit is spread over every P range, not localised: [0, 1e-3) z = −1.41, [1e-3, 1e-2) −0.48,
     [1e-2, 0.1) −1.44, [0.1, 0.5) −0.25, [0.5, 0.9999) −1.02.
  3. Re-scoring A's 17,369 tracklets with 1e-3 ≤ P < 0.9999 at (4800, 200) changes Σ P from 278.67 to 278.05 (−0.2%),
     against a deficit of ~25 NEOs, so quadrature is excluded.
- Pooled ĉ uses `validation.self_consistency.fit_odds_scale` on A and B concatenated.
- **Collapse (all runs):**
  - truth inside the admissible region: 100%;
  - integrand > 0 at the truth: 100%;
  - median posterior mass within ×2 of the true ρ: 0.9999;
  - HPD KS D 0.046–0.063 (limit 0.075).
- 0 policy assignments. Undefined scores: 3 in A (truth ρ > 100 AU), 0 elsewhere.
- Unbound-draw discard: 1.78–1.90% of NEO draws.
- AUC 0.983–0.988.

## §6 digest2 head-to-head (`digest2_comparison_hext30/`)

- digest2 2.8.2 @ c9e3f89, rebuilt locally with `CFLAGS=-D_POSIX_C_SOURCE=200809L`, because `-std=c99` hides
  `rand_r`. The build reproduces the archived digest2 scores on 2000 `gate_v0` tracklets with a max difference of
  exactly 0.

| catalogue | priora AUC | digest2 raw AUC | ΔAUC [95%] | P(Δ ≤ 0) | pred/obs priora | pred/obs digest2 raw/100 |
|---|---|---|---|---|---|---|
| gate_v0_hext30 (all, 170 NEOs) | 0.9877 | 0.9713 | **+0.0164 [+0.0047, +0.0316]** | 0.001 | 0.98 | 6.36 |
| enriched_f01_hext30 (all) | 0.9859 | 0.9646 | +0.0214 [+0.0100, +0.0349] | 0.000 | 0.99 | 7.10 |

- **Operating point** on gate_v0_hext30:
  - digest2 raw ≥ 65 flags 369, with completeness 87.1% and contamination 59.9%;
  - priora at the same completeness (p ≥ 0.214) flags 158, with 6.3% contamination;
  - priora at p ≥ 0.5 flags 145, with 82.9% / 2.8%.
- **Clamped (D-14) control:** not run. The harness refuses it for extended runs (`clamped_score_config`), and
  correctly: with the extension the two codes' effective models differ only at H ≥ 30, where 1.0% of tracklets have
  more than 1e-2 of their measure.
- **Stale labels:** in the `summary.txt` truncation block, the "H > 25.5" / "cuts 25.5" text is a fixed string. The
  numbers in that block are measured above the run's own support end (30).

## §7 Astrometric noise (`astrometric_noise_hext30/`, `astrometric_noise_arclength_hext30/`)

**45-minute headline:**
- **Calibration:**
  - ĉ moves from 1.134 to 1.130;
  - Z_tot moves from +0.71 to +0.69.
- **Ranking:**
  - AUC moves from 0.98767 to 0.98727;
  - ΔAUC = −0.00040 [−0.00075, −0.00010]. This CI *excludes* zero (the report's "consistent with zero" is wrong
    here), but the shift is 40× smaller than the priora-over-digest2 gap.
- **Per-tracklet scores:** |ΔP| p50 / p90 / p99 = 5.1e-6 / 6.9e-5 / 6.8e-4.
- **Thresholds:** 1 crossing in the full catalogue: one NEO at 0.7 (P 0.703 → 0.698); none at 0.1 / 0.3 / 0.5.
  The report's bottom line says "zero"; its own threshold table shows the one.
- **Operating point at §6's cut:** the harness's "fixed cut" is p ≥ 0.122, matched to the old 0.881 completeness target,
  not §6's p ≥ 0.214. Recomputed at 0.214 from the stored scores (2026-10-15):

  | dt | flagged | completeness | contamination | crossings (0.1/0.3/0.5/0.7, summed) | Σp |
  |---|---|---|---|---|---|
  | noiseless | 158 | 87.1% | 6.3% | — | 166.1 |
  | 60 | 157 | 86.5% | 6.4% | 1 | 166.2 |
  | 45 | 157 | 86.5% | 6.4% | 1 | 166.2 |
  | 30 | 157 | 86.5% | 6.4% | 2 | 166.3 |
  | 15 | 161 | 86.5% | 8.7% | 18 | 170.2 |
  | 10 | 168 | 86.5% | 12.5% | 44 | 176.0 |
  | 5 | 199 | 87.1% | 25.6% | 148 | 197.9 |

  At 5 min, 41 non-NEOs newly cross 0.214; 36 are TNOs (q 27–54 AU, true ρ 31–58 AU, median rate 0.025°/day vs 0.24°/day overall,
  noise ≈ 40% of their own motion). The Σp excess (+31.8) is +31.4 from non-NEOs.

**Arc-length sweep:**

| dt | Z_tot | max\|z\| (thr) | C1 | ΔAUC | contamination @ fixed cut |
|---|---|---|---|---|---|
| 60 | +0.69 | 0.96 (2.74) | PASS | −0.00020 | 10.2% |
| 45 | +0.69 | 0.96 (2.75) | PASS | −0.00040 | 9.7% |
| 30 | +0.67 | 0.96 (2.67) | PASS | −0.00050 | 9.7% |
| 15 | −0.04 | 0.58 (2.93) | PASS | −0.00091 | 12.3% |
| 10 | −1.06 | 2.70 (2.82) | PASS (p = 0.0245, marginal) | −0.00148 | 16.7% |
| 5 | −4.49 | 24.98 (3.96) | **FAIL** | −0.00318 | 30.2% |

- Same conclusion as the unextended study: breaks at 5 min, marginal at 10, comfortable from 15.
- **Stale text in the sweep's REPORT.md:** the paragraph "That makes the 45-minute row a cross-check … identical to the
  published values" quotes the *unextended* numbers (ĉ 0.925, AUC 0.98125). Use the table, not that paragraph.

## §8 NEOCP pilot (from `h_extension/neocp_delta_all.json` + `neocp_pilot/scores.npz`, no new scoring)

- **Extended, all 499 scored (0 non-finite, 0 policy):**
  - predicted 306.7 vs observed 305, pred/obs 1.006;
  - digest2 raw/100 predicts 420.8, a ratio of 1.38.
- **ΔAUC (priora extended − digest2 raw)** = +0.0041 [−0.0010, +0.0104], P(Δ ≤ 0) = 0.066
  (`neocp_pilot.paired_auc_bootstrap`, n_boot 4000).
- **Operating points, extended, all 499** (recomputed 2026-10-15 from `neocp_delta_all.json` + `scores.npz`; NEOCP_PILOT.md's
  "all" rows dropped the 11 unscored NEOs):
  - digest2 raw ≥ 65: 413 flagged, 98.4% complete, 27.4% contamination (113 FA: 80 with q < 1.67, 30 at 1.67–4.5, 3 ≥ 4.5 AU);
  - priora p ≥ 0.0403 (matched): 380 flagged, 98.4%, 21.1%;
  - priora p ≥ 0.5: 303 flagged, 93.8%, 5.6% (17 FA, all q 1.30–2.22; 15 below 1.55 AU).
  - AUC priora 0.9765, digest2 0.9724. The 11 formerly unscored tracklets (all NEOs) score P = 1 exactly.
- On real data, extended ≈ clamped: median |Δp| 0 and max 8.6e-6 (H_EXTENSION.md).

## §6 check: is digest2's outer-Solar-System false-alarm excess an artefact of its assumed astrometric error?

**Why this was a candidate cause:**
- The synthetic tracklets carry exact positions with blank ADES rms.
- For obscode 500, digest2 then uses its default 1.0″ error. `d2math.c:updateRMSValues` floors every per-detection rms
  at the configured site error, so a smaller ADES rms would have no effect.
- Over 45 min, a 1.0″ error blurs a distant object's rate by roughly its whole motion.

**The check:**
- Re-ran digest2 with `obserr500=0.1`, the value MPC.config assigns Rubin (X05).
- Mechanism: a copied MPC.config passed via `$DIGEST2_CONFIG`, so no code change.
- Output: `validation/output/digest2_comparison_hext30_obserr0p1/`.

| digest2 error | flagged (raw ≥ 65) | NEOs | false alarms | FA with q < 1.67 | 1.67–4.5 | 4.5–10 | ≥ 10 AU | ΔAUC (priora − d2) | pred/obs d2 |
|---|---|---|---|---|---|---|---|---|---|
| 1.0″ (default) | 369 | 148 | 221 | 17 | 31 | 50 | 123 | +0.0164 [+0.0047, +0.0316] | 6.36 |
| 0.1″ (Rubin) | 427 | 151 | 276 | 25 | 50 | 117 | 84 | +0.0140 [+0.0028, +0.0288] | 8.29 |

- priora at digest2's completeness: 10 false alarms at 1.0″ (9 of them Mars-crossers just outside q = 1.3). At 0.1″
  the matched cut is lower, since digest2 catches 88.8%: p ≥ 0.084 gives 26 false alarms, 14.7% contamination.
- **Verdict:** tighter errors make digest2's false alarms *more* numerous, and q ≥ 4.5 AU objects are still 73% of
  them (78% at 1.0″). The excess is a property of digest2's scoring, not of the error setting.
- The specific mechanism (e.g. no ρ⁴ change of measure, so nearby solutions are over-weighted) is **not tested**:
  priora has no switch to turn its Jacobian off. Do not state it as established.
