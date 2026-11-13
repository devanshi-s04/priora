#!/usr/bin/env bash
# Rubin DP2 precovery test (paper Sec. 9.3): from the DP2 TAP tables to the candidate verdicts and Fig. 11.
# DP2 is available only to Rubin data-rights holders: no DP2 data are in this repository (see alerts/dp2/README.md).
# Needs: an RSP token in ~/.rsp-tap-token (read, never printed), HelioLinC (`make external`), and under $DS42_ALERTS:
#   data/mpcorb_extended.json.gz                     MPCORB of 2026 Nov 2 (data bundle)
#   dp2/mpcorb/MPCORB.DAT.gz, dp2/mpcorb/CometEls.txt MPC orbit files of 2026 Nov 4 (data bundle)
#   data_b612/obs_sbn_X05_full.parquet               Rubin's MPC submissions, public (URL in recovered_steps/07)
# Steps marked (network) query the RSP, SkyBoT or the MPC.
set -euo pipefail
: "${DS42_ALERTS:?source env.sh or run through make}"
cd "$DS42_ALERTS/dp2"
step() { echo; echo "== $*"; }
R=run/recovered_steps

step "1. DP2 DiaSources for nights 60858-60866 (network, RSP token)"
$PY nights.py
$PY pull_dp2.py
step "2. tracklets and scores (known + unknown sets)"
( cd run && bash run_tracklets.sh && $PY tracklet_table.py known && $PY tracklet_table.py unknown && bash score_all.sh )
step "3. labels from Rubin's MPC submissions; known and unknown analyses"
bash $R/07_b612_dp2nights.sh
( cd run && $PY label_b612.py && $PY analyze_known.py && $PY analyze_unknown3.py )
step "4. candidate screening: pair motion, template + trail, known-object check"
bash $R/09_candidate_pair_ratio.sh
( cd run && $PY template_trail.py )
bash $R/11_template_trail_flags.sh
bash $R/11b_trail_confirmed.sh
bash $R/12_numbers_dp2.sh
bash $R/14a_candidate_verdict.sh
bash $R/14b_candidate_neighbours.sh
( cd run && $PY mpcorb_check.py )                     # ASSIST propagation of MPCORB + comets
bash $R/16_mpcorb_annotation.sh
( cd run && $PY final_verdict.py )
step "5. MPC histories of the precovered objects (network)"
bash $R/18a_mpc_histories_nov03.sh
bash $R/18b_mpc_histories_nov05.sh
echo; echo "dp2 complete; Fig. 11 is made by 'make figures'"
