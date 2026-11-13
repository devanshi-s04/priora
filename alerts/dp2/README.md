# DP2 precovery test

**Data rights.** Rubin Data Preview 2 is available only to Rubin data-rights holders. This folder holds scripts
only. No DP2 detections, tracklets, scores, cutouts or candidate lists are in this repository or its Zenodo bundle.
Do not commit anything that `run_dp2.sh` writes.

**To rerun** (data-rights holders):

1. Put a Rubin Science Platform token in `~/.rsp-tap-token`. The scripts read it and never print it.
2. Run `make external`. Put the MPC files from the Zenodo bundle in `dp2/mpcorb/` and `data/`.
3. Download Rubin's public MPC submissions to `$DS42_ALERTS/data_b612/` (URL in `run/recovered_steps/07_b612_dp2nights.sh`).
4. Run `make dp2`, then `make figures` for Fig. 11.

The `run/recovered_steps/` scripts are one-off analysis commands, saved verbatim. Each was rerun and checked to
give identical output.
