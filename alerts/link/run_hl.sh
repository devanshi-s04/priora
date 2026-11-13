#!/bin/bash
# run_hl.sh <prefix: known|full> <imgs/pairdets tag> <tracklets dir> <hypothesis file> <label>
P=$1; TAG=$2; D=$3; HYP=$4; L=$5
B=$DS42_EXTERNAL/heliolinx/src; A=$DS42_EXTERNAL/heliolinx-aux/tests
start=$(date +%s)
OMP_NUM_THREADS=${WORKERS:-8} $B/heliolinc_omp -imgs $D/imgs_$TAG.txt -pairdets $D/pairdets_$TAG.csv -tracklets ${P}_tracklets.csv -trk2det ${P}_trk2det.csv \
  -obspos $A/Earth1day2020s_02a.csv -heliodist $HYP -mjd 61230.2 -outsum ${P}_${L}_sum.csv -clust2det ${P}_${L}_c2d.csv > ${P}_${L}_hl.log 2>&1
python3_fix=1; ${PY:-python} fix_sum.py ${P}_${L}_sum.csv
echo "heliolinc $P $L exit $? in $(( $(date +%s) - start )) s" >> run.log
echo "${P}_${L}_sum.csv ${P}_${L}_c2d.csv" > ${P}_${L}_lflist
start=$(date +%s)
OMP_NUM_THREADS=${WORKERS:-8} $B/link_purify -imgs $D/imgs_$TAG.txt -pairdets $D/pairdets_$TAG.csv -lflist ${P}_${L}_lflist \
  -outsum ${P}_${L}_LPsum.csv -clust2det ${P}_${L}_LPc2d.csv > ${P}_${L}_lp.log 2>&1
echo "link_purify $P $L exit $? in $(( $(date +%s) - start )) s" >> run.log
