#!/usr/bin/env bash
# Regenerate every figure in the paper into analysis/figures/ and copy each to paper/figures/ under the name main.tex uses.
# Run through `make figures` (or `source env.sh` first). The alert and DP2 figures are made only when their data are present
# (alerts: the Fink transfers processed by `make alerts`; DP2: Rubin data rights, see alerts/dp2/README.md).
set -euo pipefail
cd "$(dirname "$0")"
P="$PRIORA_REPO/paper/figures"; mkdir -p "$P"
put() { cp "$1" "$P/$2"; echo "  paper/figures/$2"; }

# Fig. 1  posterior over (rho, rho_dot) for one NEO and one main-belt asteroid
NO_CROP=1 OUT_STEM=hypothesis_space_ranging_figs $PY make_hypothesis_space_pair.py >/dev/null
put hypothesis_space_ranging_figs.png hypothesis_space_ranging_figs.png
# Fig. 2  All vs Unk population
$PY make_population_options.py >/dev/null;                       put popopt_b.png population_all_unk_lines.png
# Fig. 3  reliability of the pooled Unk self-consistency samples
CAL_EXCLUDE=1 CAL_RUNS=gate_v0unk200k_hext30,gate_v0unk200k_s3_hext30,gate_v0unk200k_s4_hext30,gate_v0unk200k_s5y27_hext30 \
  R_RUN=gate_v0unk_hext30 R_CMP=digest2_comparison_unk D2_SCORE=noid OUT_SUFFIX=_unk800k_indomain $PY make_linear_hext30.py >/dev/null
put calibration_hext30_reliability_linear_noid_unk800k_indomain.png calibration_hext30_reliability_linear_noid_unk800k_indomain.png
# Figs. 4-6  digest2 comparison
$PY make_d2cmp_unk_counts_p01.py >/dev/null;                     put d2cmp_unk_counts_p01.png d2cmp_unk_counts_p01.png
$PY make_d2cmp_unk_logroc_pr.py >/dev/null;                      put d2cmp_unk_pr.png d2cmp_unk_pr.png
$PY make_scatter_unk20k.py >/dev/null;                           put d2cmp_scatter_unk20k.png digest2_vs_priora_20k_unk.png
# Figs. 7-8  NEOCP bright-sample test
$PY make_neocp_bright_figs.py >/dev/null
put neocp_bright_scatter.png neocp_bright_scatter.png;           put neocp_bright_hdist.png neocp_bright_hdist.png
# Figs. 9-10  Rubin alerts
if [ -f "$PRIORA_ALERTS/full_nd2/score/scores_joined.parquet" ]; then
  $PY make_lsst_mover_cutouts.py >/dev/null;                     put lsst_mover_cutouts.png lsst_mover_cutouts.png
  $PY make_lsst_unknown_scatter_movers.py >/dev/null;            put lsst_unknown_scatter_movers.png lsst_unknown_scatter_movers.png
else
  echo "  skip Figs. 9-10: no alert data in \$PRIORA_ALERTS (run 'make alerts' after downloading the Fink transfers)"
fi
# Known objects in the alerts: main-belt flow and flagged tracklets (Sec. 6.2.2)
if [ -f "$PRIORA_ALERTS/score/scores_joined.parquet" ] && [ -f "$PRIORA_ALERTS/tracklets/pairdets_v5.csv" ]; then
  $PY make_lsst_known_flow.py >/dev/null;                        put rubin_known_flow.pdf rubin_known_flow.pdf
  put rubin_known_flow.png rubin_known_flow.png
  # Appendix D: field overlaps and tracklets with 3+ detections, one night
  $PY make_lsst_overlap_night.py >/dev/null;                     put rubin_overlap_night.pdf rubin_overlap_night.pdf
  put rubin_overlap_night.png rubin_overlap_night.png
else
  echo "  skip rubin_known_flow: known-object alert data (rubin_known_flow, rubin_overlap_night) not in \$PRIORA_ALERTS (run 'make alerts')"
fi
# Fig. 11  DP2 precovery timeline
if [ -f "$PRIORA_ALERTS/dp2/run/score/candidate_verdict_final.parquet" ]; then
  $PY make_dp2_precovery_timeline.py >/dev/null;                 put dp2_precovery_timeline.png dp2_precovery_timeline.png
else
  echo "  skip Fig. 11: DP2 data not present (Rubin data rights required)"
fi
