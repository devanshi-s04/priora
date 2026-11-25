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
