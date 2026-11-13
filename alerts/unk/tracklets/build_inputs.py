"""HelioLinC input for the unknown-alert transfer (2026-11-03), same format as tracklets/build_inputs.py.
Cuts: drop the 2026-06-30 night; psfFlux > 0; not dipole, glint or streak; reliability >= REL (default 0.9, which keeps
95.4% of the known-asteroid detections). Hidden ID = diaSourceId (no designations exist yet)."""
import sys, numpy as np, pandas as pd
REL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
d = pd.read_parquet("../../data_unk/ftransfer_2026-11-03_flat.parquet")
night = np.floor(d.midpointMjdTai - 0.5).astype(int)
m = (night != 61221) & (d.psfFlux > 0) & ~d.isDipole.fillna(False) & ~d.glint_trail.fillna(False) & ~d.pixelFlags_streak.fillna(False) & (d.reliability >= REL)
d = d[m].sort_values("midpointMjdTai").reset_index(drop=True)
out = pd.DataFrame({"index": np.arange(len(d)), "ObjID": d.diaSourceId.astype(str).values, "MJD_TAI": d.midpointMjdTai.values,
                    "RA": d.ra.values, "Dec": d.dec.values, "mag": 31.4 - 2.5 * np.log10(d.psfFlux.values), "band": d.band.values,
                    "obscode": "X05", "diaSourceId": d.diaSourceId.values})
tag = f"r{int(round(REL * 100))}"
with open(f"dets_unk_{tag}.csv", "w") as f:
    f.write("#" + ",".join(out.columns) + "\n"); out.to_csv(f, header=False, index=False, float_format="%.8f")
d.to_parquet(f"dets_unk_{tag}.parquet")
print(tag, len(out), "detections")
