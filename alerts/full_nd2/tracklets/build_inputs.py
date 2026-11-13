"""HelioLinC input for the full transfer (ftransfer_lsst_2026-11-03_913305: no filters, no blocks).
Unknown = no Rubin known-object match (designation and packed designation both empty). Same cuts as the b_is_new test:
drop 2026-06-30; psfFlux > 0; not dipole, glint or streak; reliability >= 0.9; nDiaSources <= 2 (drop detections on Rubin objects with 3+ detections: static/variable sources). Hidden ID = diaSourceId."""
import numpy as np, pandas as pd
d = pd.read_parquet("../../data_all/ftransfer_913305.parquet")
known = d.packed_primary_provisional_designation.notna() | d.designation.notna()
night = np.floor(d.midpointMjdTai - 0.5).astype(int)
m = ~known & (night != 61221) & (d.psfFlux > 0) & ~d.isDipole.fillna(False) & ~d.glint_trail.fillna(False) & ~d.pixelFlags_streak.fillna(False) & (d.reliability >= 0.9) & (d.nDiaSources <= 2)
d = d[m].sort_values("midpointMjdTai").reset_index(drop=True)
out = pd.DataFrame({"index": np.arange(len(d)), "ObjID": d.diaSourceId.astype(str).values, "MJD_TAI": d.midpointMjdTai.values,
                    "RA": d.ra.values, "Dec": d.dec.values, "mag": 31.4 - 2.5 * np.log10(d.psfFlux.values), "band": d.band.values,
                    "obscode": "X05", "diaSourceId": d.diaSourceId.values})
with open("dets_full_r90.csv", "w") as f:
    f.write("#" + ",".join(out.columns) + "\n"); out.to_csv(f, header=False, index=False, float_format="%.8f")
d.to_parquet("dets_full_r90.parquet"); print(len(out), "detections; repeat-object", int((d.nDiaSources > 1).sum()))
