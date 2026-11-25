"""Step 3 input: HelioLinC maxvel-5 tracklets with arc >= 15 min -> one ADES PSV both codes score (trkSub = Nnnnnnnn = row
in meta.parquet). Native Rubin band kept; priora and digest2 convert to V with the same digest2 table. TAI -> UTC via astropy."""
import numpy as np, pandas as pd
from astropy.time import Time
T = "../tracklets/"
pdets = pd.read_csv(T + "pairdets_v5.csv"); pdets.columns = [c.lstrip("#") for c in pdets.columns]
t2d = pd.read_csv(T + "trk2det_v5.csv"); t2d.columns = [c.lstrip("#") for c in t2d.columns]
g = pd.read_parquet(T + "tracklet_labels_v5.parquet")
lab = pd.read_parquet("../data/labels_ftransfer_2026-11-02.parquet").set_index("packed")
keep = g[g.arc >= 15].copy()
print(f"tracklets {len(g):,}; arc >= 15 min {len(keep):,} (pure {keep.pure.sum():,}, mixed {(~keep.pure).sum():,}, pure NEO {keep.neo.sum()})")
keep["row"] = np.arange(len(keep)); keep["trkSub"] = ["N%07d" % k for k in keep.row]
keep["q"] = keep.obj.map(lab.q).where(keep.pure); keep["e"] = keep.obj.map(lab.e).where(keep.pure) if "e" in lab else np.nan
t2d = t2d[t2d.trk_ID.isin(keep.index)].copy()
t2d = t2d.join(pdets, on="detnum"); t2d["trkSub"] = t2d.trk_ID.map(keep.trkSub)
t2d = t2d.sort_values(["trkSub", "MJD"])
iso = Time(t2d.MJD.values, format="mjd", scale="tai").utc.isot
mag = t2d.mag.values
with open("tracklets_v5_arc15.psv", "w") as f:
    f.write("# LSST alerts test: HelioLinC maxvel 5 deg/day tracklets, arc >= 15 min, Fink transfer 2026-11-02\n")
    f.write("permID |provID |trkSub  |mode |stn  |obsTime                     |ra          |dec         |mag   |band |rmsRA |rmsDec |astCat\n")
    for s, t, ra, de, m, b in zip(t2d.trkSub.values, iso, t2d.RA.values, t2d.Dec.values, mag, t2d.band.values):
        f.write("       |       |%-8s|CCD  |X05  |%-27s |%12.8f |%+12.8f |%6.2f |%-5s|      |       |Gaia3\n" % (s, t + "Z", ra, de, m, b))
keep.to_parquet("meta.parquet")
print("rows", len(t2d), "->", "tracklets_v5_arc15.psv")
