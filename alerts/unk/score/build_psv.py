"""Unknown-alert test, step 3 input: HelioLinC maxvel-5 tracklets (reliability >= 0.9 detections) with arc >= 15 min
-> tracklets_r90_arc15.psv (trkSub N%07d = row of meta.parquet). Same format as ../../score/build_psv.py."""
import numpy as np, pandas as pd
from astropy.time import Time
T = "../tracklets/"
pdets = pd.read_csv(T + "pairdets_r90.csv"); pdets.columns = [c.lstrip("#") for c in pdets.columns]
t2d = pd.read_csv(T + "trk2det_r90.csv"); t2d.columns = [c.lstrip("#") for c in t2d.columns]
t2d = t2d.join(pdets, on="detnum")
g = t2d.groupby("trk_ID").agg(n=("MJD", "size"), t0=("MJD", "min"), t1=("MJD", "max"))
g["arc"] = (g.t1 - g.t0) * 1440; g["night"] = np.floor(g.t0 - 0.5).astype(int)
keep = g[g.arc >= 15].copy()
print(f"tracklets {len(g):,}; arc >= 15 min {len(keep):,}; npts {keep.n.value_counts().sort_index().head(6).to_dict()}")
keep["row"] = np.arange(len(keep)); keep["trkSub"] = ["N%07d" % k for k in keep.row]
t2d = t2d[t2d.trk_ID.isin(keep.index)].copy(); t2d["trkSub"] = t2d.trk_ID.map(keep.trkSub)
t2d = t2d.sort_values(["trkSub", "MJD"])
iso = Time(t2d.MJD.values, format="mjd", scale="tai").utc.isot
with open("tracklets_r90_arc15.psv", "w") as f:
    f.write("# Unknown-alert test: HelioLinC maxvel 5 deg/day tracklets, reliability >= 0.9, arc >= 15 min, Fink transfer 2026-11-03\n")
    f.write("permID |provID |trkSub  |mode |stn  |obsTime                     |ra          |dec         |mag   |band |rmsRA |rmsDec |astCat\n")
    for s, t, ra, de, m, b in zip(t2d.trkSub.values, iso, t2d.RA.values, t2d.Dec.values, t2d.mag.values, t2d.band.values):
        f.write("       |       |%-8s|CCD  |X05  |%-27s |%12.8f |%+12.8f |%6.2f |%-5s|      |       |Gaia3\n" % (s, t + "Z", ra, de, m, b))
keep.to_parquet("meta.parquet"); t2d[["trkSub", "trk_ID", "detnum", "MJD", "RA", "Dec", "mag", "band", "idstring"]].to_parquet("trk_dets.parquet")
print("rows", len(t2d))
