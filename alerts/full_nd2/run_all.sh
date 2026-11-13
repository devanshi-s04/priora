#!/bin/bash
# Full-transfer pipeline: inputs -> make_tracklets -> PSV -> ds42/digest2 -> catalogue match -> labels -> analysis -> candidates
set -e
PY=${PY:-python}; A=$DS42_EXTERNAL/heliolinx-aux/tests
cd $DS42_ALERTS/full_nd2/tracklets
$PY -I build_inputs.py
$DS42_EXTERNAL/heliolinx/src/make_tracklets -dets dets_full_r90.csv -outimgs imgs_full.txt -pairdets pairdets_full.csv -tracklets tracklets_full.csv -trk2det trk2det_full.csv -colformat colformat_fink.txt -imrad 2.0 -maxtime 1.5 -maxGCR 0.5 -maxvel 5 -mintrkpts 2 -time_offset -37 -earth $A/Earth1day2020s_02a.csv -obscode $A/ObsCodesNew.txt > mt_full.log 2>&1
tail -3 mt_full.log
cd ../score && $PY build_psv.py && ./run_scores.sh && cat run.log && $PY join_scores.py && $PY drop_stationary.py
cd $DS42_ALERTS && $PY full_nd2/match/match_mpcorb.py
cd full_nd2 && $PY match/label_tracklets.py
cd score && $PY analyze.py > analysis.log && $PY candidates.py > candidates.log; head -2 candidates.log
echo PIPELINE DONE
