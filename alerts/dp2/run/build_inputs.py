"""DP2 -> HelioLinC input, July 2025 nights 60858-60866. Both sets already have reliability >= 0.9, psfFlux > 0, not dipole
(server side). Here: no glint/streak; unknown also nDiaSources <= 2. mag = AB from psfFlux (nJy). Hidden ID: designation
(known) or diaSourceId (unknown). Usage: python3 build_inputs.py known|unknown"""
import sys, glob, numpy as np, pandas as pd
kind = sys.argv[1]
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"../data/{kind}_*.parquet"))], ignore_index=True)
n0 = len(d)
m = ~d.glint_trail.fillna(False).astype(bool) & ~d.pixelFlags_streak.fillna(False).astype(bool)
if kind == "unknown": m &= d.nDiaSources <= 2
d = d[m].sort_values("midpointMjdTai").reset_index(drop=True)
idc = d.designation.str.replace(" ", "_") if kind == "known" else d.diaSourceId.astype(str)
out = pd.DataFrame({"index": np.arange(len(d)), "ObjID": idc.values, "MJD_TAI": d.midpointMjdTai.values, "RA": d.ra.values, "Dec": d.dec.values,
                    "mag": 31.4 - 2.5 * np.log10(d.psfFlux.values), "band": d.band.values, "obscode": "X05", "diaSourceId": d.diaSourceId.values})
with open(f"{kind}/dets.csv", "w") as f:
    f.write("#" + ",".join(out.columns) + "\n"); out.to_csv(f, header=False, index=False, float_format="%.8f")
d.to_parquet(f"{kind}/dets.parquet")
open(f"{kind}/colformat.txt", "w").write("IDCOL 2\nMJDCOL 3\nRACOL 4\nDECCOL 5\nMAGCOL 6\nBANDCOL 7\nOBSCODECOL 8\n")
print(kind, "downloaded", n0, "-> after cuts", len(out), "| nights", sorted(np.floor(d.midpointMjdTai - 0.5).astype(int).unique()))
