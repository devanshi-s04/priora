#!/usr/bin/env bash
# Regenerate every stored simulation / comparison output that the paper uses, from scratch.
# Run through `make simulations` (sets DS42_* and PYTHONPATH). Takes many hours; WORKERS sets parallelism.
# Seeds, sizes and settings are exactly those of the published runs (validation/output/<run>/summary.json).
set -euo pipefail
cd "$DS42_REPO"
W="${WORKERS:-8}"
O=validation/output
sc() {  # sc <run> <n> <seed> <variant> <v_max> [extra args]: generate + score with ds42 (H extended to 30)
  local run=$1 n=$2 seed=$3 var=$4 vmax=$5; shift 5
  [ -f "$O/$run/scores.npz" ] && { echo "skip $run (exists)"; return; }
  $PY -m validation.unk_noid.run_sc run --out "$O/$run" --n "$n" --seed "$seed" --variant "$var" \
      --v-min 14 --v-max "$vmax" --h-extend-to 30 --workers "$W" --no-plots "$@"
}
d2() {  # d2 <run> <out>: score the same tracklets with digest2 2.8.2 (two detections, 45 min, geocentric)
  [ -f "$2/digest2_scores.npz" ] && { echo "skip digest2 $1"; return; }
  $PY -m validation.digest2_comparison run --run "$O/$1" --out "$2" --workers "$W"
}

# Fig. 1: the All sample without the H extension (the two example objects are drawn from its generated geometry)
[ -f $O/gate_v0/scores.npz ] || $PY -m validation.unk_noid.run_sc run --out $O/gate_v0 --n 20000 --seed 20260901 \
    --variant All --v-min 14 --v-max 24 --workers "$W" --no-plots

# Sec. 5: self-consistency, four Unk samples of 2e5 (Table 2, Fig. 3) and the All samples A+B
sc gate_v0unk200k_hext30        200000 20260902 Unk 24
sc gate_v0unk200k_s3_hext30     200000 20260903 Unk 24
sc gate_v0unk200k_s4_hext30     200000 20260904 Unk 24
sc gate_v0unk200k_s5y27_hext30  200000 20260905 Unk 24 --epoch-start 2027-01-01T00:00:00
sc calibration_1e5_hext30       100000 20260903 All 24
sc calibration_1e5_hext30_seed2 100000 20260904 All 24

# Sec. 6: the controlled digest2 comparison (Figs. 4-6, Table 3) and the All / 0.1" checks of Sec. 6.5
sc gate_v0unk_hext30  20000 20260901 Unk 24
sc gate_v0_hext30     20000 20260901 All 24
d2 gate_v0unk_hext30 $O/digest2_comparison_unk
d2 gate_v0_hext30    $O/digest2_comparison_hext30/gate_v0_hext30
DIGEST2_CONFIG="$DS42_REPO/validation/configs/digest2_MPC_obserr0p1.config" d2 gate_v0_hext30 $O/digest2_comparison_hext30_obserr0p1/gate_v0_hext30

# Sec. 7: astrometric noise at 45 min and the arc-length sweep (Table 4)
[ -f $O/astrometric_noise_hext30/summary.json ] || $PY -m validation.astrometric_noise run \
    --source $O/gate_v0_hext30 --out $O/astrometric_noise_hext30 --h-extend-to 30 --seed 20260902 --workers "$W"
[ -f $O/astrometric_noise_arclength_hext30/summary.json ] || $PY -m validation.astrometric_noise sweep \
    --source $O/gate_v0_hext30 --out $O/astrometric_noise_arclength_hext30 --h-extend-to 30 --seed 20260902 --workers "$W"

# Sec. 8: NEOCP sample (network: MPC NEOCP archive and observations API, cached under neocp_pilot/raw), rescoring
[ -f $O/neocp_pilot/tracklets.json ] || $PY -m validation.neocp_pilot all --n 500 --workers "$W"
[ -f $O/h_extension/neocp_delta_all.json ] || $PY validation/output/h_extension/rescore_neocp.py --workers "$W" --stage both
[ -f $O/neocp_unk_ext/neocp_delta_all.json ] || $PY validation/output/neocp_unk_ext/rescore_neocp_unk.py --workers "$W"

# Sec. 8: bright sample matched to NEOCP brightness (Table 6, Figs. 7-8)
sc gate_v0unk_bright212_hext30 3500 20261106 Unk 21.2
d2 gate_v0unk_bright212_hext30 $O/digest2_comparison_unk_bright212
$PY validation/d2_far_known/bright_neocp_match.py
$PY validation/d2_far_known/bright_test.py

# Appendix B: quadrature accuracy study (on the All 1e5 sample without the H extension)
[ -f $O/calibration_1e5/scores.npz ] || $PY -m validation.unk_noid.run_sc run --out $O/calibration_1e5 --n 100000 --seed 20260903 \
    --variant All --v-min 14 --v-max 24 --workers "$W" --no-plots
[ -f $O/h_interpolation/grid_study.json ] || $PY -m validation.h_interpolation_study grid --run $O/calibration_1e5 --select midp \
    --out $O/h_interpolation --out-name grid_study.json --ref-grid 9600 400 --ref-check-grid 19200 800 --threads "$W" \
    --time-reps 5 --time-tracklets 48

echo "simulations complete; now run: make analysis"
