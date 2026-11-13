# Fink Data Transfers

The Rubin alert test (paper Sec. 9) used three Fink Data Transfers (https://fink-portal.org, Data Transfer service).
The alert data are not redistributed here. These files describe each request, so that the same data can be requested again.

| Transfer | Request | Used for | Unpack under `$DS42_ALERTS` |
|---|---|---|---|
| `ftransfer_lsst_2026-11-02_838512` | `ftransfer_lsst_2026-11-02_838512.yml` (reconstructed) | known-object test | `data/ftransfer_2026-11-02/` |
| `ftransfer_lsst_2026-11-03_596070` | `ftransfer_lsst_2026-11-03_596070.yml` (reconstructed) | unknown alerts, first pull | `data_unk/ftransfer_2026-11-03/` |
| `ftransfer_lsst_2026-11-03_913305` | `ftransfer_lsst_2026-11-03_913305.yml` (the original request) | full transfer, the paper's main sample | `data_all/` |

Only the request for 913305 was saved as Fink returned it. The other two were rebuilt from the delivered data:
the date range, the columns and the selection used in the scripts. `schema/` holds the Avro schema Fink sent with
each transfer. It contains column definitions only, no alert data.

All three cover alerts from 2026-06-30 22:56 to 2026-10-31 07:38 UTC. After unpacking, run `make alerts`.
