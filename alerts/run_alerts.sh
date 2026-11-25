#!/usr/bin/env bash
# Rubin alert test (paper Sec. 9): rerun every step from the Fink Data Transfers to the numbers and tables in the paper.
# Run through `make alerts`, which sets PRIORA_REPO / PRIORA_ALERTS / PRIORA_EXTERNAL / PY. Needs, under $PRIORA_ALERTS:
#   data/ftransfer_2026-11-02/ftransfer_lsst_2026-11-02_838512/*.parquet     known objects      (alerts/fink/README.md)
#   data_unk/ftransfer_2026-11-03/ftransfer_lsst_2026-11-03_596070/*.parquet  unknown, first pull
#   data_all/ftransfer_lsst_2026-11-03_913305/*.parquet                      full transfer
#   data/mpcorb_extended.json.gz                                             MPCORB of 2026 Nov 2 (data bundle)
# and HelioLinC built by `make external`. Steps marked (network) call the MPC API.
set -euo pipefail
: "${PRIORA_ALERTS:?source env.sh or run through make}"
A=$PRIORA_EXTERNAL/heliolinx-aux/tests; B=$PRIORA_EXTERNAL/heliolinx/src
mt() {  # mt <dets> <tag>: HelioLinC make_tracklets with the settings of every run in the paper (90 min, 5 deg/day, 2 deg)
  $B/make_tracklets -dets "$1" -outimgs imgs_$2.txt -pairdets pairdets_$2.csv -tracklets tracklets_$2.csv -trk2det trk2det_$2.csv \
    -colformat colformat_fink.txt -imrad 2.0 -maxtime 1.5 -maxGCR 0.5 -maxvel 5 -mintrkpts 2 -time_offset -37 \
    -earth $A/Earth1day2020s_02a.csv -obscode $A/ObsCodesNew.txt > mt_$2.log 2>&1
}
cd "$PRIORA_ALERTS"
step() { echo; echo "== $*"; }

step "1. known objects (838512): alert table, MPCORB labels, July + October nights"
bash recovered_steps/01_known_alerts_table.sh
bash recovered_steps/02_known_labels.sh
bash recovered_steps/03_known_julyoct_labelled.sh
step "2. known objects: tracklets, purity, scores"
( cd tracklets && $PY -I build_inputs.py && mt dets_julyoct.csv v5 && $PY purity.py 5 )
( cd score && $PY build_psv.py && bash run_scores.sh && $PY analyze.py > analysis.log )

step "3. unknown alerts, first pull (596070)"
$PY unk/flatten.py
( cd unk/tracklets && $PY -I build_inputs.py && mt dets_unk_r90.csv r90 )
$PY unk/fetch_mpc.py                                   # (network) MPC observations of 2026 N / 2026 U objects
( cd unk/score && $PY build_psv.py && bash run_scores.sh )
$PY unk/match/match_mpcorb.py && $PY unk/match/label_tracklets.py
( cd unk/score && $PY analyze.py > analysis.log && $PY candidates.py > candidates.log )

step "4. full transfer (913305), then the nDiaSources <= 2 rerun used in the paper"
$PY full/flatten.py
bash full/run_all.sh
bash full_nd2/run_all.sh
$PY full_nd2/followup.py > full_nd2/followup.log      # (network) MPC histories of the candidates

step "5. multi-night linking of the July tracklets (HelioLinC)"
$PY link/prep.py full_nd2/tracklets full link_nd2/full
bash link_nd2/run.sh

step "6. paper numbers and table"
$PY summary_table.py
bash recovered_steps/04_numbers_nd2.sh
bash recovered_steps/05_link_nd2_summary.sh
echo; echo "alerts complete; Figs. 9-10 are made by 'make figures'"
