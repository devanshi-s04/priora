#!/bin/bash
# DP2: inputs and make_tracklets (same settings as the alert tests) for known and unknown sets
set -e; PY=${PY:-python}; A=$PRIORA_EXTERNAL/heliolinx-aux/tests
cd $PRIORA_ALERTS/dp2/run
for K in known unknown; do
  $PY -I build_inputs.py $K
  ( cd $K && $PRIORA_EXTERNAL/heliolinx/src/make_tracklets -dets dets.csv -outimgs imgs.txt -pairdets pairdets.csv -tracklets tracklets.csv -trk2det trk2det.csv -colformat colformat.txt -imrad 2.0 -maxtime 1.5 -maxGCR 0.5 -maxvel 5 -mintrkpts 2 -time_offset -37 -earth $A/Earth1day2020s_02a.csv -obscode $A/ObsCodesNew.txt > mt.log 2>&1; tail -3 mt.log )
done
echo TRACKLETS DONE
