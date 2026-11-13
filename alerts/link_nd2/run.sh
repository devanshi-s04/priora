#!/bin/bash
# HelioLinC on the nDiaSources<=2 rerun's July tracklets: NEO grid default and relaxed, main-belt grid default
B=$DS42_EXTERNAL/heliolinx/src; A=$DS42_EXTERNAL/heliolinx-aux/tests; H=$A/hypotheses; D=../full_nd2/tracklets; PY=${PY:-python}
cd $DS42_ALERTS/link_nd2
for spec in "neo $H/NEO/hihyp00aa_neo.txt 1e5 1e5 1" "mb $H/main_belt/hihyp01b_mb.txt 1e5 1e5 1" "t_aa_3e5r $H/NEO/hihyp00aa_neo.txt 3e5 4e5 2"; do set -- $spec
  OMP_NUM_THREADS=${WORKERS:-8} $B/heliolinc_omp -imgs $D/imgs_full.txt -pairdets $D/pairdets_full.csv -tracklets full_tracklets.csv -trk2det full_trk2det.csv -obspos $A/Earth1day2020s_02a.csv -heliodist $2 -mjd 61230.2 -clustrad $3 -outsum full_$1_sum.csv -clust2det full_$1_c2d.csv > full_$1_hl.log 2>&1
  $PY fix_sum.py full_$1_sum.csv >/dev/null; echo "full_$1_sum.csv full_$1_c2d.csv" > full_$1_lflist
  OMP_NUM_THREADS=${WORKERS:-8} $B/link_purify -imgs $D/imgs_full.txt -pairdets $D/pairdets_full.csv -lflist full_$1_lflist -maxrms $4 -max_astrom_rms $5 -outsum full_$1_LPsum.csv -clust2det full_$1_LPc2d.csv > full_$1_lp.log 2>&1
  echo "$1: $(tail -2 full_$1_lp.log | head -1)"
done
echo LINK DONE
