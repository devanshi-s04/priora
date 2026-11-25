# priora paper: provenance of every figure and number

Audit of `main.tex` / `priora_4.pdf` as of 2026-11-12. Each figure and number was traced to the script that made it and the
stored output it comes from, then re-checked against that output without re-scoring anything.

| | Checked | Matches | Problems |
|---|---|---|---|
| Figures | 11 | 10 regenerate pixel-identical from the scripts in `figures/`; 1 has the same content, different tick spacing | 0 |
| Simulation sections (2, 3, 4, 5, 6, 7, 8, 12, App. B–C) | 116 | 110 | 6 (listed below) |
| Rubin alert section (9) | 128 | 122 | 3 wrong, 3 with no stored source |
| Data Preview 2 section (10) | 31 | 22 | 4 wrong, 4 wording |

## Corrections for the paper

Numbers that are wrong:

1. **"7–16 days" (Sec. 10, Sec. 11, Summary item 3) should be 11–16 days.** The gaps from the DP2 detection to the first MPC observation are 2025 OZ 15, OT12 11, QO13 16, OL4 13 days (`~/lsst_alerts_test/dp2/precovery_fig/precovery_summary.csv`). That is the first *reported observation*, and for OT12 and QO13 it is Rubin's own; measured to the credited discovery the gaps are 15–53 days. Use "first reported observation" with 11–16 d.
2. **"All 174 tracklets with P ≥ 0.9 are NEOs" (Sec. 10)** is the count over all 285,126 pure tracklets. For the 3+ sample the sentence is about, it is **78, all NEOs** (`dp2/page/numbers_dp2.json` known.3plus.calib_bins).
3. **"2″ pairing radius" (Sec. 9.2)** does not exist. `-imrad 2.0` is make_tracklets' *image radius of 2 degrees*; no arcsecond pairing radius was set (`tracklets/mt_v5.log`).
4. **"Among the 208 objects designated after the alert" (Sec. 9.6)**: 208 tracklets of **165 objects** (`full_nd2/score/analysis.json` designated_after).
5. **"fitted perihelia agree to within 0.2%" (Sec. 9.8, multi-night linking)**: 0.2% is the median; the 90th percentile is 1.6–1.9% (`link/eval_known.json`).
6. **Unbound NEO draws in the Unk test population (Sec. 5.2): 2.2%, not 2.1%** (2.23% from the bin counts; 1.93% is the All value used in Sec. 3.4).
7. **Triage-policy NEOs "q < 0.002 AU" (Sec. 2.8 and 5.4): the largest is 0.00232 AU**; write q < 0.003 AU.
8. **"KS D ≤ 0.063" (Sec. 5.4): the largest is 0.0631**; write D < 0.064.
9. **"heliocentric distances of 13–91 AU" (Sec. 6.5): 13–89 AU** (topocentric 13–90).
10. **NEOCP digest2 NoID/100 total "437" (Sec. 8): 436.5 (436.49)**; write 436.
11. **NEOCP ΔAUC interval [+0.0049, +0.0207] (Sec. 8)** is not produced by any stored script. The reproducible value (paired, label-stratified bootstrap, 2,000 draws, seed 7, as in Sec. 6) is **[+0.0050, +0.0211]**.
12. **Sec. 9.5, "most of the deficit comes from distant Amors with H = 16–19"**: of the 21 NEO tracklets with P < 0.01, 13 are Amors and 8 Apollos; Amor H spans 16.6–21.3 (8 of 13 in 16–19); max a is 3.10 AU. Reword to "most are distant Amors (13 of 21), with a up to 3.1 AU".
13. **Sec. 9.2 stationary-source numbers are from the first pull.** On the rerun the paper describes, real slow movers have median motion 0.65″ (not 0.46″) and a 2″ cut would remove 35 distant asteroids (not 37).
14. **Sec. 9.8 candidate count: 9 + 22 + 12 = 43, not 44.** One candidate was rejected in the image review (rerun N0002437 = first-pull N0004100).
15. **Appendix A "Verification": 7,964 combinations, largest difference 4 × 10⁻¹³** is not reproduced by any stored output and conflicts with Sec. 4 (17,522 nodes, max 1.2 × 10⁻¹¹, reproduced by `numbers/jacobian_closed_form_check.py`). Replace it with a pointer to Sec. 4.

Wording that does not match what was done:

- **Sec. 10, "nine nights of 2025 July 3–11"**: by the scripts' own night convention these are dayObs July 2–10; July 3–11 are the UTC morning dates. Either is defensible; say which.
- **Sec. 10, "DP2 detections are labelled using Rubin's own MPC submissions"**: true for the unknown set only. Known tracklets are labelled by SSSource designation plus MPCORB q.
- **Sec. 10, 2025 NA "three later DP2 detections"**: three later tracklets (10 detections) on three nights.
- **Summary item 3, "not reported at the time"**: Rubin did report other 2025 ML93 detections (two, on 07-08); only our 07-10 tracklet went unreported.
- **Sec. 9.3, propagated positions "agree with the observations to 0.2–1.3″"**: stated in the matching script's docstring only; no stored test output.
- **Sec. 9.7 "Removing static sources"**: 25.2% is the share among the 291,177 quality-cut detections (17.6% of all 3,498,044 unassociated alerts); the 6,793 covers only detections inside first-pull tracklets.
- **Sec. 9.8 "This test does not flag any known asteroid"**: 2,261 of 68,287 real-asteroid tracklets have a testable pair (minimum motion ratio 0.63 against the 0.5 threshold).
- **Sec. 9.8 image review**: the 15 reviewed are from the *initial* 73-candidate list (all 9 first-pull tier A and 6 of 8 tier B); 10 of them survive into the rerun's 44.
- **Sec. 9.1**: there is no data on July 7, so "July 5–13" is eight nights.
- **Sec. 9.3 and 9.8**: the labels use the 2026 Nov 2 `mpcorb_extended` (1,578,692 orbits); the ASSIST known-object check uses `MPCORB.DAT` of Nov 4 (1,578,741). "1.58 million" fits both; "November 2" applies only to the labels.
- **Sec. 7, "only one of the 20,000 crosses the triage threshold"**: it crosses P = 0.7, one of the thresholds the noise test tracks.

`\todo`s that can now be removed:

- *confirm priora uses digest2's band-to-V table*: yes; `ingest.BAND_TO_V` equals digest2 2.8.2's table for every band.
- *confirm timing hardware* (Sec. 9.4): AMD EPYC 7401, 90 worker processes (this machine; `score/run.log` 325 s, 312 s, 171 s).
- *confirm designated-after-alert count on rerun*: 208 tracklets, 165 objects, 12 NEO tracklets, ΣP 11.33.
- *confirm exact reviewed candidate set*: the 15 rows of `full_nd2/score/image_review.csv` (see above).
- *resolve 115,906 versus 115,949*: **115,906 is correct** (pure 3+ tracklets with a catalogued q; 170 NEOs, 166.0, 268/59, 81/1 all go with it). 115,949 adds 43 tracklets without q (mostly comets and 2025 designations missing from `mpcorb_extended`) and is only the denominator of the template/trail false-flag rates.

Figure to replace: **Fig. 1 (`hypothesis_space_ranging_figs.png`)**. The paper's copy came from an earlier, uncommitted run of
`make_hypothesis_space_pair.py`; the regenerated version (same objects, posteriors, P values and axis ranges; only the x-axis
tick spacing differs) is `figures/hypothesis_space_ranging_figs.png`. Uploading it makes every figure exactly reproducible.

## Figures

All regenerate with `bash reproduce/figures/run_all.sh` into `reproduce/figures/` (nothing in `paper/figures` is touched).
"Pixel check" compares the regenerated PNG with the image embedded in `priora_4.pdf`.

| Fig. | File in paper | Script (in `figures/`) | Settings | Inputs | Pixel check |
|---|---|---|---|---|---|
| 1 | hypothesis_space_ranging_figs.png | make_hypothesis_space_pair.py (+ priora_example_figures.ipynb) | `NO_CROP=1 OUT_STEM=hypothesis_space_ranging_figs` | gate_v0_hext30 objects re-observed from X05; data/digest2.model.csv | same content, tick spacing differs |
| 2 | population_all_unk_lines.png | make_population_options.py (option b → popopt_b.png) | | data/digest2.model.csv | identical |
| 3 | calibration_hext30_reliability_linear_noid_unk800k_indomain.png | make_linear_hext30.py | `CAL_EXCLUDE=1 CAL_RUNS=<4 Unk runs> R_RUN=gate_v0unk_hext30 R_CMP=digest2_comparison_unk D2_SCORE=noid OUT_SUFFIX=_unk800k_indomain` | validation/output/gate_v0unk200k{,_s3,_s4,_s5y27}_hext30 | identical |
| 4 | d2cmp_unk_counts_p01.png | make_d2cmp_unk_counts_p01.py | | gate_v0unk_hext30, digest2_comparison_unk | identical |
| 5 | d2cmp_unk_pr.png | make_d2cmp_unk_logroc_pr.py | | same (via validation/unk_noid/analyze.load) | identical |
| 6 | digest2_vs_priora_20k_unk.png = d2cmp_scatter_unk20k.png | make_scatter_unk20k.py | | same | identical |
| 7 | neocp_bright_scatter.png | make_neocp_bright_figs.py | | gate_v0unk_bright212_hext30, digest2_comparison_unk_bright212; H peaks from validation/d2_far_known | identical |
| 8 | neocp_bright_hdist.png | make_neocp_bright_figs.py | | same + MPCORB (2026-11-04) via bright_neocp_match.py | identical |
| 9 | lsst_mover_cutouts.png | make_lsst_mover_cutouts.py | | ~/lsst_alerts_test/full_nd2/score/{image_review.csv, scores_joined.parquet}, handoff/…_detections.csv, public ALeRCE stamps cutouts/alerce_nd2/ | identical |
| 10 | lsst_unknown_scatter_movers.png | make_lsst_unknown_scatter_movers.py | | full_nd2/score/{scores_joined.parquet, image_review.csv, candidates_annot.csv} | identical |
| 11 | dp2_precovery_timeline.png | make_dp2_precovery_timeline.py | | dp2/run/score/{candidate_verdict_final.parquet, unknown_3plus_dets.parquet}, dp2/precovery_fig/mpc_obs_*.csv | identical |

## What is in this folder

| Folder | Contents |
|---|---|
| `figures/` | The 10 figure scripts (copies; each writes next to itself), the notebook Fig. 1 executes, `run_all.sh`, and the regenerated figures. |
| `numbers/` | `check_sim_numbers.py` (recomputes and checks every simulation-section number; results in `check_sim_numbers.md/json`), `cadence_arcs.py` (Sec. 7 LSST arcs), `jacobian_closed_form_check.py` (Sec. 4), `bench_kernel_unk_hext30.py`, `parity_python_rust.py`, `profile_python_reference.py` (Sec. 4 timing, parity, 88%) and their stored results. |
| `pipelines/` | Snapshot copies of the pipelines that produced the stored outputs (`validation/` self-consistency, digest2 comparison, astrometric noise, NEOCP pilot and rescore, quadrature study, bright-sample test). Run the originals in `validation/`; the copies record the versions used. |
| `alerts/` | Snapshot copies of the Rubin alert pipeline from `~/lsst_alerts_test` (same subfolders). |
| `dp2/` | Snapshot copies of the DP2 pipeline from `~/lsst_alerts_test/dp2`. DP2 data are for Rubin data-rights holders; keep private. |

### How the stored simulation outputs were made

All from the repository root with the Rubin-stack Python, `OMP_NUM_THREADS=1 PYTHONPATH=. python -m validation.unk_noid.run_sc run --out validation/output/<run> --n N --seed S --variant V --v-min 14 --v-max VMAX --h-extend-to 30 --workers 90 --no-plots`
(`run_sc` pins the Earth ephemeris to astropy `builtin`), then `python -m validation.digest2_comparison run --run validation/output/<run> --out validation/output/<d2dir>` where a digest2 comparison is used.

| Run | Used in | n | seed | variant | V range | epochs |
|---|---|---|---|---|---|---|
| gate_v0unk200k_hext30 | Sec. 5, Fig. 3, Sec. 12.2 | 200,000 | 20260902 | Unk | 14–24 | 2026 |
| gate_v0unk200k_s3_hext30 | same | 200,000 | 20260903 | Unk | 14–24 | 2026 |
| gate_v0unk200k_s4_hext30 | same | 200,000 | 20260904 | Unk | 14–24 | 2026 |
| gate_v0unk200k_s5y27_hext30 | same | 200,000 | 20260905 | Unk | 14–24 | 2027 |
| calibration_1e5_hext30, _seed2 | Sec. 5 (All A+B) | 2 × 100,000 | 20260903, 20260904 | All | 14–24 | 2026 |
| gate_v0unk_hext30 (+ digest2_comparison_unk) | Sec. 4 parity, Sec. 6, Figs. 4–6 | 20,000 | 20260901 | Unk | 14–24 | 2026 |
| gate_v0_hext30 (+ digest2_comparison_hext30{,_obserr0p1}) | Sec. 6.5 (All/raw), Sec. 7, Fig. 1 | 20,000 | 20260901 | All | 14–24 | 2026 |
| gate_v0unk_bright212_hext30 (+ digest2_comparison_unk_bright212) | Sec. 8, Figs. 7–8 | 3,500 | 20261106 | Unk | 14–21.2 | 2026 |

Other stored outputs: `astrometric_noise_hext30/` and `astrometric_noise_arclength_hext30/` (`python -m validation.astrometric_noise run` and `sweep`, Sec. 7);
`neocp_pilot/` (`python -m validation.neocp_pilot all --n 500`, Sec. 8 sample); `neocp_unk_ext/` (`rescore_neocp_unk.py`, Sec. 8 scores);
`h_extension/neocp_delta_all.json` (`rescore_neocp.py`, Sec. 2.8 and App. C); `h_interpolation/grid_study.json` (`python -m validation.h_interpolation_study`, App. B);
`validation/d2_far_known/` (`run_bright.sh`, `bright_neocp_match.py`, `bright_test.py`, Sec. 8).

## Gaps in reproducibility (updated 2026-11-13)

- The one-off steps found by the audit are now saved as scripts and verified: `alerts/recovered_steps/01-05` rebuild the known-object
  tables, `numbers_nd2.json` and `link_nd2/summary.csv`; `dp2/run/recovered_steps/09-16` rebuild the DP2 candidate chain
  (rerun end to end, every output identical: 458 rejected, 50 mixed, 23 strong, 9 known-NEO tracklets). Steps 18a/b refetch MPC
  histories over the network; the fetched files are kept, and refetching would change the figure annotations as new observations arrive.
- Network-only inputs (Fink transfers, RSP TAP queries for DP2, ALeRCE stamps, SkyBoT, MPC histories) are not re-downloaded;
  their outputs are stored and everything downstream reruns offline.
- Sec. 9.3: the 0.2–1.3″ propagation accuracy still has no stored test.
- Sec. 8: the NEOCP ΔAUC interval must read [+0.0050, +0.0211] (the reproducible value); draft 5 has [+0.0050, +0.0207].

## Simulation sections: every number

Generated by `numbers/check_sim_numbers.py` (run from the repo root: `PYTHONPATH=src python reproduce/numbers/check_sim_numbers.py`). Paths are relative to `validation/output/` unless given in full. "OK" means equal after rounding to the digits printed in the paper.

| Section | Quantity | Paper | Computed | Status | Source |
|---|---|---|---|---|---|
| 3 / Table 1 | total objects | 14542147 | 1.454e+07 | OK | `data/digest2.model.csv via population.read_digest2_model` |
| 3 / Table 1 | NEO objects | 277469 | 2.775e+05 | OK | `same` |
| 3 / Table 1 | other objects | 14264678 | 1.426e+07 | OK | `same` |
| 3 / Table 1 | NEO fraction (%) | 1.9 | 1.908 | OK | `same` |
| 3 / Table 1 | cells (q x e x i x H) | 45936 | 4.594e+04 | OK | `same` |
| 3 / Table 1 | extended NEO / other | [742145, 14868899] | ['7.421e+05', '1.487e+07'] | OK | `Digest2Histogram(h_extend_to=30).total_count` |
| 3 / Table 1 | extended NEO fraction (%) | 4.8 | 4.754 | OK | `same` |
| 3.4 | NEO fraction in highest-H bin (%) | 55.8 | 55.82 | OK | `read_digest2_model` |
| 3.4 | NEO draws unbound, All (%) | 1.93 | 1.915 | OK | `bin counts (DESIGN.md 4.1: 1.92% exact, 1.93% MC)` |
| 5.2 | NEO draws unbound, Unk (%) [paper says 2.1] | 2.1 | 2.228 | MISMATCH | `bin counts (DESIGN.md 4.1: 2.23% exact, 2.24% MC)` |
| 5.4 / Table 2 | Unk 1 obs, exp, Z_tot, c, lo, hi | [2591, 2599.2, -0.55, 0.96, 0.84, 1.1] | ['2591.0000', '2599.1649', '-0.5535', '0.9627', '0.8416', '1.1013'] | OK | `validation/output/gate_v0unk200k_hext30/scores.npz, gen.npz; self_consistency.fit_odds_scale` |
| 5.4 / Table 2 | Unk 2 obs, exp, Z_tot, c, lo, hi | [2675, 2654.6, 1.4, 1.1, 0.96, 1.25] | ['2675.0000', '2654.5695', '1.3966', '1.0971', '0.9633', '1.2494'] | OK | `validation/output/gate_v0unk200k_s3_hext30/scores.npz, gen.npz; self_consistency.fit_odds_scale` |
| 5.4 / Table 2 | Unk 3 obs, exp, Z_tot, c, lo, hi | [2642, 2650.3, -0.57, 0.96, 0.84, 1.1] | ['2642.0000', '2650.2504', '-0.5652', '0.9616', '0.8393', '1.1016'] | OK | `validation/output/gate_v0unk200k_s4_hext30/scores.npz, gen.npz; self_consistency.fit_odds_scale` |
| 5.4 / Table 2 | Unk 4 obs, exp, Z_tot, c, lo, hi | [2642, 2638.6, 0.23, 1.02, 0.89, 1.16] | ['2642.0000', '2638.5687', '0.2282', '1.0152', '0.8917', '1.1558'] | OK | `validation/output/gate_v0unk200k_s5y27_hext30/scores.npz, gen.npz; self_consistency.fit_odds_scale` |
| 5.4 / Table 2 | Unk pooled obs, exp, Z_tot, c, lo, hi | [10550, 10542.6, 0.25, 1.01, 0.94, 1.08] | ['10550.0000', '10542.5535', '0.2523', '1.0086', '0.9439', '1.0776'] | OK | `four Unk runs pooled` |
| 5.4 | pooled calibration sample size | 799906 | 7.999e+05 | OK | `four Unk runs` |
| 5.2 | NEO fraction of accepted tracklets (%) | 1.3 | 1.319 | OK | `four Unk runs` |
| 5.4 | excluded beyond 100 AU / no posterior / triage (Unk) | [52, 37, 5] | ['52', '37', '5'] | OK | `scores.npz truth_rho_in_range, p_neo, lambdas` |
| 5.4 | non-NEOs at P=1 beyond 100 AU | 20 | 20 | OK | `same` |
| 5.4 | those 20: max H | < 4 | 3.59 | OK | `gen.npz H` |
| 5.4 | those 20: min q (AU) | > 30 | 31.02 | OK | `gen.npz elements` |
| 5.4 | tracklets at P=1 within domain, all NEOs | [8828, 8828] | ['8828', '8828'] | OK | `pooled sample` |
| 5.4 / 2.8 | no-posterior NEOs: max q (AU) | < 0.004 | 0.00350 | OK | `gen.npz elements` |
| 2.8 / 5.4 | triage NEOs: max q (AU) | < 0.002 | 0.00232 | MISMATCH | `gen.npz elements` |
| 5.4 | max KS D over Unk runs | <= 0.063 | 0.06309 | MISMATCH | `summary.json collapse.hpd_ks_D / hpd_exact_ks_D` |
| 5.4 | median posterior mass within x2 of true rho (%) | 99.98 | 99.98 | OK | `summary.json collapse` |
| 5.3 | c_hat 95% half-width, one run / pooled (%) | [14, 7] | ['13.28', '6.848'] | OK | `fit_odds_scale log_c_se (per-run summary.json)` |
| 5.4 / Table 2 | All A+B obs, exp, Z_tot, c, lo, hi | [1719, 1743.8, -1.42, 0.92, 0.82, 1.03] | ['1719.0000', '1743.8126', '-1.4193', '0.9198', '0.8195', '1.0324'] | OK | `calibration_1e5_hext30{,_seed2}` |
| 12.2 | NEOs at P<0.1: observed / predicted | [540, 535.7] | ['540.0', '535.7'] | OK | `pooled Unk sample` |
| 12.2 | median sky rate low-P NEOs / all NEOs (deg/day) | [0.41, 1.5] | ['0.410', '1.498'] | OK | `gen.npz ra_rate, dec_rate` |
| 12.2 | median distance low-P NEOs / all NEOs (AU) | [1.27, 1.04] | ['1.266', '1.039'] | OK | `gen.npz rho_true` |
| 12.2 | fraction of 1.2<=q<1.3 NEOs with P<0.1 (quarter) | 0.25 | 0.259 | OK | `pooled Unk` |
| 12.2 | NEOs faster than 1 deg/day with P<0.1 | 0 | 0 | OK | `pooled Unk` |
| 6.1 | tracklets / NEOs | [19999, 287] | ['2e+04', '287'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.1 | tracklets with >1% of measure beyond H=30 (%) | 2 | 2.04 | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M) ar_frac_h_above` |
| 6.2 | priora predicted / digest2 predicted (all) | [284.0, 1349] | ['284.0', '1348.9'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.2 | digest2 / observed ratio | 4.7 | 4.70 | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.2 | digest2 <10: n, predicted, observed | [17263, 290.5, 9] | ['17263.0', '290.5', '9.0'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.2 | digest2 >=10: n, predicted, observed | [2736, 1058, 278] | ['2736.0', '1058.4', '278.0'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.2 | priora <0.1 / >=0.1: predicted, observed | [14.1, 17, 269.9, 270] | ['14.1', '17.0', '269.9', '270.0'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.2 | priora P=1 tracklets, all NEOs | [237, 237] | ['237', '237'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.3 | AUC priora / digest2 | [0.9945, 0.9846] | ['0.99454', '0.98460'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.3 | Delta AUC and 95% CI | [0.0099, 0.004, 0.0176] | ['0.00994', '0.00396', '0.01757'] | OK | `validation/unk_noid/results.json M` |
| 6.3 | non-NEO : NEO ratio (69 to 1) | 69 | 68.7 | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.4 / Table 3 | digest2>=65: NEO, non-NEO, compl %, contam % | [269, 287, 93.7, 51.6] | ['269.00', '287.00', '93.73', '51.62'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.4 / Table 3 | priora matched cut value | 0.3 | 0.3003 | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.4 / Table 3 | priora P>=0.30: NEO, non-NEO, contam % | [269, 11, 3.9] | ['269.00', '11.00', '3.93'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.4 / Table 3 | priora P>=0.5: NEO, non-NEO, compl %, contam % | [264, 3, 92.0, 1.1] | ['264.00', '3.00', '91.99', '1.12'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.3 | purity digest2>=65 / priora matched / priora P>=0.5 (%) | [48.4, 96.1, 98.9] | ['48.38', '96.07', '98.88'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.4 | NEOs selected by both / each only | [267, 2, 2] | ['267', '2', '2'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.5 | digest2 FPs with q>=4.5 / q>=10 | [213, 145] | ['213', '145'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 12.1 | digest2 FPs with 1.67<=q<4.5 | 58 | 58 | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.5 | heliocentric distance range of q>=10 FPs (AU) | [13, 91] | ['13.0', '89.0'] | MISMATCH | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M) (R_obs + rho u)` |
| 6.5 | max priora P among digest2 q>=4.5 FPs | < 0.1 | 0.094 | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.5 / 12.1 | priora FPs: Mars-crossers / main-belt range / q>=4.5 | [10, 1, 0] | ['10', '1', '0'] | OK | `validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)` |
| 6.5 | All/raw digest2 FPs at 1" -> 0.1", fraction q>=4.5 at 0.1" (%) | [221, 276, 73] | ['221.0', '276.0', '72.8'] | OK | `digest2_comparison_hext30{,_obserr0p1}/gate_v0_hext30/digest2_scores.npz` |
| 7 | All sample: tracklets / NEOs | [20000, 170] | ['2e+04', '170'] | OK | `astrometric_noise_hext30/summary.json` |
| 7 | median per-coordinate sigma (mas) | 35 | 35.32 | OK | `summary.json sigma_population` |
| 2.9 | 45-min rate error / own motion, median (%) | 0.2 | 0.2042 | OK | `summary.json` |
| 7 | 99th pct |dP| at 45 min (< 7e-4) | 0.0007 | 6.76e-04 | OK | `summary.json deltas.p_neo.all.abs_p99` |
| 7 | tracklets crossing any threshold at 45 min (one, at P=0.7) | [1, 0.7] | ['1', '0.7'] | OK | `summary.json threshold_stability` |
| 7 / Table 4 | matched cut (digest2 raw>=65 completeness on All: 148/170) ~ 0.214 | [148, 0.214] | ['148.0000', '0.2139'] | OK | `gate_v0_hext30 + digest2_comparison_hext30` |
| 7 / Table 4 | arc none: sum P, Z_tot, contam % at the matched cut | [166.1, 0.71, 6.3] | ['166.13', '0.71', '6.33'] | OK | `gate_v0_hext30/scores.npz` |
| 7 / Table 4 | arc none: completeness >= 86.5 % | >= 86.45 | 87.06 | OK | `gate_v0_hext30/scores.npz` |
| 7 / Table 4 | arc none: completeness <= 87.1 % | <= 87.14999999999999 | 87.06 | OK | `gate_v0_hext30/scores.npz` |
| 7 / Table 4 | arc 60: sum P, Z_tot, contam % at the matched cut | [166.2, 0.69, 6.4] | ['166.24', '0.69', '6.37'] | OK | `astrometric_noise_arclength_hext30/dt_60min/noisy_scores.npz` |
| 7 / Table 4 | arc 60: completeness >= 86.5 % | >= 86.45 | 86.47 | OK | `astrometric_noise_arclength_hext30/dt_60min/noisy_scores.npz` |
| 7 / Table 4 | arc 60: completeness <= 87.1 % | <= 87.14999999999999 | 86.47 | OK | `astrometric_noise_arclength_hext30/dt_60min/noisy_scores.npz` |
| 7 / Table 4 | arc 60: median rate error / motion (%) | 0.14 | 0.142 | OK | `arclength summary.json per_dt.rate_error` |
| 7 / Table 4 | arc 30: sum P, Z_tot, contam % at the matched cut | [166.3, 0.67, 6.4] | ['166.34', '0.67', '6.37'] | OK | `astrometric_noise_arclength_hext30/dt_30min/noisy_scores.npz` |
| 7 / Table 4 | arc 30: completeness >= 86.5 % | >= 86.45 | 86.47 | OK | `astrometric_noise_arclength_hext30/dt_30min/noisy_scores.npz` |
| 7 / Table 4 | arc 30: completeness <= 87.1 % | <= 87.14999999999999 | 86.47 | OK | `astrometric_noise_arclength_hext30/dt_30min/noisy_scores.npz` |
| 7 / Table 4 | arc 30: median rate error / motion (%) | 0.28 | 0.284 | OK | `arclength summary.json per_dt.rate_error` |
| 7 / Table 4 | arc 15: sum P, Z_tot, contam % at the matched cut | [170.2, -0.04, 8.7] | ['170.24', '-0.04', '8.70'] | OK | `astrometric_noise_arclength_hext30/dt_15min/noisy_scores.npz` |
| 7 / Table 4 | arc 15: completeness >= 86.5 % | >= 86.45 | 86.47 | OK | `astrometric_noise_arclength_hext30/dt_15min/noisy_scores.npz` |
| 7 / Table 4 | arc 15: completeness <= 87.1 % | <= 87.14999999999999 | 86.47 | OK | `astrometric_noise_arclength_hext30/dt_15min/noisy_scores.npz` |
| 7 / Table 4 | arc 15: median rate error / motion (%) | 0.57 | 0.568 | OK | `arclength summary.json per_dt.rate_error` |
| 7 / Table 4 | arc 10: sum P, Z_tot, contam % at the matched cut | [176.0, -1.06, 12.5] | ['176.04', '-1.06', '12.50'] | OK | `astrometric_noise_arclength_hext30/dt_10min/noisy_scores.npz` |
| 7 / Table 4 | arc 10: completeness >= 86.5 % | >= 86.45 | 86.47 | OK | `astrometric_noise_arclength_hext30/dt_10min/noisy_scores.npz` |
| 7 / Table 4 | arc 10: completeness <= 87.1 % | <= 87.14999999999999 | 86.47 | OK | `astrometric_noise_arclength_hext30/dt_10min/noisy_scores.npz` |
| 7 / Table 4 | arc 10: median rate error / motion (%) | 0.85 | 0.851 | OK | `arclength summary.json per_dt.rate_error` |
| 7 / Table 4 | arc 5: sum P, Z_tot, contam % at the matched cut | [197.9, -4.49, 25.6] | ['197.85', '-4.49', '25.63'] | OK | `astrometric_noise_arclength_hext30/dt_5min/noisy_scores.npz` |
| 7 / Table 4 | arc 5: completeness >= 86.5 % | >= 86.45 | 87.06 | OK | `astrometric_noise_arclength_hext30/dt_5min/noisy_scores.npz` |
| 7 / Table 4 | arc 5: completeness <= 87.1 % | <= 87.14999999999999 | 87.06 | OK | `astrometric_noise_arclength_hext30/dt_5min/noisy_scores.npz` |
| 7 / Table 4 | arc 5: median rate error / motion (%) | 1.7 | 1.703 | OK | `arclength summary.json per_dt.rate_error` |
| 7 | 5-min new FPs / of which TNOs / their distance range (AU) | [41, 36, 31, 58] | ['41.0', '36.0', '30.9', '58.1'] | OK | `dt_5min noisy vs noiseless` |
| 7 | LSST v5.3.7 yr-1 arcs: median (min), frac >=15 min, frac <10 min | [23, 0.7, 0.15] | ['23.16', '0.70', '0.15'] | OK | `reproduce/numbers/cadence_arcs.py` |
| 8 | tracklets / NEOs / non-NEOs | [499, 305, 194] | ['499', '305', '194'] | OK | `validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py)` |
| 8 | median detections / median arc (min) | [4, 35] | ['4.0', '35.3'] | OK | `validation/output/neocp_pilot/tracklets.json` |
| 8 | priora predicted / digest2 NoID/100 predicted | [312.0, 437] | ['312.0', '436.5'] | MISMATCH | `validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py)` |
| 8 / Table 5 | digest2>=65 NEO, non-NEO, compl, contam | [300, 140, 98.4, 31.8] | ['300.00', '140.00', '98.36', '31.82'] | OK | `validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py)` |
| 8 / Table 5 | priora matched cut | 0.03 | 0.0300 | OK | `validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py)` |
| 8 / Table 5 | priora matched NEO, non-NEO, contam | [300, 85, 22.1] | ['300.00', '85.00', '22.08'] | OK | `validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py)` |
| 8 / Table 5 | priora P>=0.5 NEO, non-NEO, compl, contam | [286, 22, 93.8, 7.1] | ['286.00', '22.00', '93.77', '7.14'] | OK | `validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py)` |
| 8 / Table 5 | fraction of sample below digest2 65 (%) | 11.8 | 11.82 | OK | `validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py)` |
| 8 | selected tracklets: digest2 - priora matched | 55 | 55 | OK | `validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py)` |
| 8 | AUC priora Unk / digest2 NoID | [0.9663, 0.9539] | ['0.96628', '0.95395'] | OK | `validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py)` |
| 8 | Delta AUC and 95% CI (paired, stratified bootstrap, seed 7, 2000) | [0.0123, 0.0049, 0.0207] | ['0.0123', '0.0050', '0.0211'] | MISMATCH | `validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py); digest2_comparison.auc_bootstrap` |
| 8 | priora P>=0.5 non-NEOs: q<2.3 / q<1.55 | [22, 17] | ['22', '17'] | OK | `validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py)` |
| 8 | digest2 FPs: q<1.67 of total / q>=4.5 | [91, 140, 4] | ['91', '140', '4'] | OK | `validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py)` |
| 8 | NEOCP median V (paper: 21.1, cut = +0.1 = 21.2) | 21.14 | 21.140 | OK | `validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py)` |
| 2.8 | NEOCP tracklets with all measure at H>25.5 (both Lambda 0), all NEOs | [11, 11] | ['11', '11'] | OK | `validation/output/h_extension/neocp_delta_all.json (truncated)` |
| 2.8 | those 11: median rate (deg/day), median V | [6.8, 20.7] | ['6.82', '20.74'] | OK | `neocp_pilot/tracklets.json` |
| 2.8 | with H to 30 the 11 get P=1 | 11 | 11 | OK | `neocp_delta_all.json (extended)` |
| App C | boundary tracklets keeping 7-10% of measure above H=30 | [4, 0.07, 0.1] | ['4.000', '0.068', '0.102'] | OK | `neocp_delta_all.json ar_frac_H_above_30` |
| App C | min fraction of measure at H>=25.5 among the 11 (>= 0.986) | 0.986 | 0.9856 | OK | `neocp_delta_all.json` |
| 8 | bright sample: tracklets / NEOs / max V | [3498, 215, 21.2] | ['3498.00', '215.00', '21.20'] | OK | `validation/output/gate_v0unk_bright212_hext30 + digest2_comparison_unk_bright212` |
| 8 / Table 6 | V<21.2: compl, contam, distant FPs, priora contam | [98.1, 54.2, 143, 17.6] | ['98.14', '54.23', '143.00', '17.58'] | OK | `validation/output/gate_v0unk_bright212_hext30 + digest2_comparison_unk_bright212` |
| 8 / Table 6 | known distant removed: compl, contam, distant FPs, priora contam | [98.1, 33.9, 1, 17.3] | ['98.14', '33.86', '1.00', '17.25'] | OK | `validation/output/gate_v0unk_bright212_hext30 + digest2_comparison_unk_bright212; distant-only H-peak removal` |
| 8 / Fig 9 | distant FPs brighter than the known peak (142 of 143) | [142, 143] | ['142', '143'] | OK | `validation/output/gate_v0unk_bright212_hext30 + digest2_comparison_unk_bright212` |
| 8 / Fig 9 | median H of distant FPs: 4.5-10 AU / >=10 AU | [11.5, 2.8] | ['11.47', '2.76'] | OK | `validation/output/gate_v0unk_bright212_hext30 + digest2_comparison_unk_bright212` |
| 8 | known/model in brightest complete bins: min, max (14-30%) | [0.14, 0.3] | ['0.143', '0.302'] | OK | `validation/d2_far_known/bright_numbers.json (bright_test.py): TNO H 0-8, Trojan/Centaur H 8-10 and 11-14` |
| 8 | known-object H peaks: 4.5-10 AU / >=10 AU | [14.75, 7.25] | ['14.75', '7.25'] | OK | `validation/d2_far_known/bright_match.json (MPCORB 2026-11-04)` |
| App C | H_max(r) for V=24 | [26.0, 24.9, 24.3, 23.8, 23.2, 22.5] | ['26.04', '24.93', '24.35', '23.76', '23.21', '22.49'] | OK | `Equation (hmax)` |
| App B | quadrature: mid-P p90 / max error vs 9600x400 | [0.008, 0.23] | ['0.0079', '0.2331'] | OK | `validation/output/h_interpolation/grid_study.json` |
| App B | mid-P tracklets | 497 | 497 | OK | `grid_study.json` |

## Section 9: Rubin alert stream

`L` = `~/lsst_alerts_test`. "First pull" = `L/full` (no nDiaSources cut); "rerun" = `L/full_nd2` (the sample the paper describes).
Status: OK = matches the stored output; WRONG / NO SOURCE as listed in the corrections above.

| Where | Quantity | Paper | Script | Stored output | Status |
|---|---|---|---|---|---|
| 9.1 | nights | Jul 5–13, Oct 29–30 | L/full/flatten.py | L/data_all/ftransfer_913305.parquet (no Jul 7) | OK |
| 9.1 | alerts / known / new / repeat diaObjects | 3,789,349 / 291,305 / 2,141,791 / 1,356,253 | L/full/flatten.py | L/data_all/transfer_913305.log; parquet | OK |
| 9.1 | known objects; detections after flux cut | 113,687; 17 removed, 288,728 | inline; L/tracklets/build_inputs.py | L/data/ftransfer_2026-11-02_all.parquet; dets_julyoct.csv | OK (no script for the parquet) |
| 9.1 | m = 31.4 − 2.5 log F; V via digest2 table | | build_inputs.py; src/priora/ingest.py BAND_TO_V | | OK |
| 9.1 | after quality cuts; new / repeat | 291,177; 190,192 / 100,985 | L/full/tracklets/build_inputs.py | numbers_nd2.json dets_before_nd | OK |
| 9.1 | reliability keeps known detections | 95.4% | L/unk/tracklets/build_inputs.py | recomputed 0.9544 | OK |
| 9.1 | after nDiaSources ≤ 2 | 217,663 | L/full_nd2/tracklets/build_inputs.py | numbers_nd2.json dets; full_nd2/run_all.log | OK |
| 9.2 | make_tracklets settings | 2″ pairing radius, 1.5 h, GCR 0.5″, 5°/day | heliolinx make_tracklets | L/tracklets/mt_v5.log | **WRONG: 2° image radius** |
| 9.2 | 10°/day adds no correct tracklets | | L/tracklets/purity.py | v5 vs v10 pure sets identical | OK |
| 9.2 | object-nights recovered; NEOs | 78,657 / 79,048; 111 / 112; 2.3 h | purity.py 5 | printed only; recomputed | OK |
| 9.2 | stationary cut; median motions; 2″ cut loses | 0.5″; 0.07″ / 0.46″; 37 | full*/score/drop_stationary.py | recomputed | **first-pull values; rerun 0.65″, 35** |
| 9.3 | MPC match; NEOs | all but one of 113,687; 213 | inline | L/data/labels_ftransfer_2026-11-02.parquet | OK (no script) |
| 9.3 | orbits; catalogue date; 10″ | 1,578,692; 2026 Nov 2 | L/full_nd2/match/{match_mpcorb,label_tracklets}.py | full_nd2/run_all.log | OK |
| 9.3 | propagation accuracy | 0.2–1.3″ | docstring | none | **NO SOURCE** |
| 9.4 | X05, TAI→UTC; digest2 2.8.2; H to 30, 1200×50; all four scores | | L/score/{build_psv.py, run_scores.sh} | score/*.log | OK |
| 9.4 | timing | ~5 min/prior, ~3 min | run_scores.sh (90 workers) | L/score/run.log 325 / 312 / 171 s | OK (machine: AMD EPYC 7401) |
| 9.5 | 273,406 / 64,938 / 81 / 208,468 | | L/score/analyze.py, L/summary_table.py | summary_table.json known_2 | OK |
| Table 7 | every cell (2+ and 3+) | | L/summary_table.py | summary_table.json known_2, known_3 | OK |
| 9.5 | ΔAUC +0.037 [0.013, 0.065] | | paper/figures/make_d2cmp_hext30.py (`R_RUN=lsst_alerts_All OUT_SUFFIX=_lsstAll`) | figures/d2cmp_hext30_provenance_lsstAll.json | OK |
| 9.5 | 21 NEO tracklets P<0.01 | 21 | | recomputed | OK |
| 9.5 | "distant Amors, H 16–19, a ≤ 3.1 AU" | | none | recomputed: 13 Amors + 8 Apollos, H 16.6–21.3, a ≤ 3.10 | **NO SOURCE (reword)** |
| 9.5 | mixed: 99.5% / 98.7%; 3.5 vs 0.13 °/day; 12 with 3+ | | L/score/analyze.py | score/analysis.json | OK |
| 9.6 | 171,859 / 3,345 / 22; 208 / 12 | | L/full_nd2/score/analyze.py | full_nd2/score/analysis.json | OK |
| 9.6 | 22.8 vs 22; 14 / 0; 29 / 17; all 17 P<0.01; 11.3 vs 12 | | same | same | OK |
| 9.6 | "208 objects designated after" | | same | designated_after.n_objects = 165 | **WRONG: 208 tracklets, 165 objects** |
| Table 8 | every cell (2+ and 3+) | | same | analysis.json | OK |
| 9.7 | 6,793; four (0.06%); 25.2% | | paper/figures/make_lsst_ndiasources_cut.py | full/match/match.parquet, dets_full_r90.parquet | OK (see wording note) |
| 9.7 | 33 of 73; 30; all 9 movers kept; 5 of 6 | | same | full_nd2/score/{firstpull_candidates_vs_cut.csv, image_review.csv} | OK |
| 9.8 | 3,213 / 2,666 / 44 / 31 | | full_nd2/score/candidates.py | numbers_nd2.json; scores_joined.parquet | OK |
| 9.8 | pair test; template test 1 of 60; 22 = 16 + 6 | | full_nd2/followup.py; link/static_flags.py; cutouts/validate_template.py | candidates_annot.csv; link/static_flags.csv; cutouts/validate/real_template_test.csv | OK |
| 9.8 | 1.58 M + 966 orbits; 200/200, 185 within 5″; 2025 NB54 at 46″, P = 0.22 | | dp2/run/mpcorb_check_alerts.py | full_nd2/score/{mpcorb_check.log, mpcorb_matches.parquet} | OK |
| 9.8 | linkages pure; 59 of 73 one-night; no candidate links; NEO-like linkages unbound | | link/{prep.py, eval_links.py}, link_nd2/run.sh | link/eval_known.json; link_nd2/summary.csv | OK |
| 9.8 | perihelia within 0.2% | | link/eval_links.py | eval_known.json dq_frac_median 0.18–0.21%, p90 1.6–1.9% | **median only** |
| 9.8 | 15 reviewed, 9 movers, 6 artefacts | | manual | full_nd2/score/image_review.csv | OK |
| 9.8 | Result: P and digest2 ranges of the 9; 22 static; 12 unreviewed | | | candidates_annot.csv, image_review.csv | OK (9 + 22 + 12 = 43 of 44) |
| 11 | 68.9 vs 81; 22.8 vs 22 | | as above | | OK |

## Section 10: Data Preview 2

`D` = `~/lsst_alerts_test/dp2`; `S` = `D/run/score`. Scripts run with the Rubin-stack Python from `D/run`.

| Where | Quantity | Paper | Script | Stored output | Status |
|---|---|---|---|---|---|
| 10 | nights; with data | nine, July 3–11; eight | D/nights.py, D/pull_dp2.py, D/run/build_inputs.py | D/nights.csv; run_tracklets.log | OK (dayObs July 2–10) |
| 10 | known-set labels | "Rubin's MPC submissions" | D/run/analyze_known.py | | **wording: SSSource designation + MPCORB q** |
| 10 | pure known tracklets; NEOs; predicted | 285,126; 393; 399.3 | D/run/analyze_known.py | D/page/numbers_dp2.json known.2plus | OK |
| 10 | 3+ tracklets; NEOs; predicted | 115,906; 170; 166.0 | same | known.3plus | OK (115,906 is the right denominator) |
| 10 | digest2 ≥ 65 non-NEOs / distant; priora P≥0.1 non-NEOs / distant | 268 / 59; 81 / 1 | same | known.3plus | OK |
| 10 | P ≥ 0.9 all NEOs | 174 | same | known.2plus.calib_bins | **WRONG for 3+: 78 (all NEOs)** |
| 10 | unassociated 3+; unmatched with P≥0.1 | 8,234; 655 | D/run/{build_psv.py, analyze_unknown3.py} | S/unknown3_analysis.json | OK |
| 10 | static rejects; mixed; strong | 458; 50; 23 | inline steps + D/run/final_verdict.py | S/candidate_verdict_final.parquet | OK (inline steps not saved) |
| 10 | known objects; NEO tracklets / objects; P range | 16; 9 / 7; 0.81–1.00 | D/run/mpcorb_check.py, final_verdict.py | candidate_verdict_final.parquet | OK |
| 10 | none submitted | | analyze_unknown3.py | S/unknown3_joined.parquet; precovery_fig/mpc_obs_*.csv | OK |
| 10 | four precovery objects | OZ, OT12, QO13, OL4 | paper/figures/make_dp2_precovery_timeline.py | D/precovery_fig/precovery_summary.csv | OK |
| 10, 11, Summary | days before first MPC observation | 7–16 | same | precovery_summary.csv: 15, 11, 16, 13 | **WRONG: 11–16** |
| 10 | 2025 NA "three later DP2 detections" | | same | mpc_obs_2025_NA.csv | wording: three tracklets |
| 11 | 399.3 vs 393; few distant contaminants (1 vs 59) | | analyze_known.py | numbers_dp2.json | OK |
| Summary | ML93 "not reported at the time" | | | mpc_obs_2025_ML93.csv | wording: other ML93 detections were reported |

