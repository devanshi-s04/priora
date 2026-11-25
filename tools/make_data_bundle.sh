#!/usr/bin/env bash
# Maintainer: pack the stored outputs that 'make analysis' needs into priora_outputs.tar.gz for Zenodo.
# Contains only simulated data and public NEOCP/MPC-derived results (no DP2 detections, no unpublished data); the two MPC catalogue snapshots are public.
set -euo pipefail
cd "$PRIORA_REPO"
tar -czf priora_outputs.tar.gz \
  validation/output/gate_v0unk200k_hext30 validation/output/gate_v0unk200k_s3_hext30 validation/output/gate_v0unk200k_s4_hext30 \
  validation/output/gate_v0unk200k_s5y27_hext30 validation/output/calibration_1e5_hext30 validation/output/calibration_1e5_hext30_seed2 \
  validation/output/gate_v0 validation/output/gate_v0unk_hext30 validation/output/gate_v0_hext30 validation/output/gate_v0unk_bright212_hext30 \
  validation/output/digest2_comparison_unk validation/output/digest2_comparison_hext30 validation/output/digest2_comparison_hext30_obserr0p1 \
  validation/output/digest2_comparison_unk_bright212 validation/output/astrometric_noise_hext30 validation/output/astrometric_noise_arclength_hext30 \
  validation/output/neocp_pilot validation/output/neocp_unk_ext validation/output/h_extension validation/output/h_interpolation validation/output/calibration_1e5 \
  alerts/dp2/mpcorb/MPCORB.DAT.gz alerts/dp2/mpcorb/CometEls.txt alerts/data/mpcorb_extended.json.gz
ls -lh priora_outputs.tar.gz
