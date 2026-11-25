# Data

Nothing in this folder is versioned. Every input comes from a `make` target or from the source named below.

| Path | Target | What it is |
|---|---|---|
| `digest2.model.csv` | `make model` | digest2 population model (MPC), the absolute population priora integrates over |
| `external/opsim/baseline_v5.3.7_10yrs.db` | `make external` | LSST baseline v5.3.7 cadence simulation (Rubin sim-data, SLAC) |
| `external/ar_cache/` | `make external` | Sorcha 1.1.1 / ASSIST ephemeris files (`sorcha bootstrap`) |
| `external/heliolinx`, `external/heliolinx-aux` | `make external` | HelioLinC at commits `e979817` and `03e3411` (make_tracklets, heliolinc_omp, link_purify) |
| `../validation/output/` | `make data` or `make simulations` | stored simulated samples, digest2 comparisons, NEOCP sample (Zenodo; or regenerate from the seeds in `analysis/simulations.sh`) |

## Inputs that are not redistributed

* **Imminent-impactor detections (Sec. 10).** The Sorcha simulation of LSST detections of imminent impactors
  (`validation/impactors/all_obs_precovery.csv`) was provided by Ian Chow and is based on the population of
  Chow et al. (2026). It is unpublished and is not included; ask Ian Chow for it. The public data release of
  Chow et al. (2026) is in `validation/impactors/chow2026/`.
* **Rubin alerts (Sec. 9).** The three Fink Data Transfers are not included. `alerts/fink/` gives the transfer
  settings needed to request the same data from Fink.
* **Rubin DP2 (Sec. 9.3).** DP2 is available only to Rubin data-rights holders. Only the scripts are here; see
  `alerts/dp2/README.md`.
* **MPC catalogue snapshots.** The MPCORB extended file of 2026 November 2 and MPCORB.DAT / CometEls.txt of
  2026 November 4 are part of the Zenodo bundle, because the MPC files change daily.
