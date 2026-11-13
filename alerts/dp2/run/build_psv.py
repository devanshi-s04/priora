"""ADES PSV for scoring a DP2 tracklet subset (same format as the alert tests; trkSub = N%08d = row of <out>_meta.parquet).
Usage: build_psv.py <set> <subset: all|3plus> <out stem>"""
import sys, numpy as np, pandas as pd
from astropy.time import Time
k, sub, stem = sys.argv[1:4]
g = pd.read_parquet(f"{k}/tracklets_table.parquet")
if sub == "3plus": g = g[g.n >= 3]
g = g.copy(); g["row"] = np.arange(len(g)); g["trkSub"] = ["N%08d" % r for r in g.row]
pdet = pd.read_csv(f"{k}/pairdets.csv"); pdet.columns = [c.lstrip("#") for c in pdet.columns]
t2d = pd.read_csv(f"{k}/trk2det.csv"); t2d.columns = ["trk_ID", "detnum"]
t2d = t2d[t2d.trk_ID.isin(g.index)].join(pdet[["MJD", "RA", "Dec", "mag", "band", "idstring"]], on="detnum")
t2d["trkSub"] = t2d.trk_ID.map(g.trkSub); t2d = t2d.sort_values(["trkSub", "MJD"])
iso = Time(t2d.MJD.values, format="mjd", scale="tai").utc.isot
with open(f"{stem}.psv", "w") as f:
    f.write(f"# DP2 {k} {sub}: HelioLinC maxvel 5 deg/day tracklets, arc >= 15 min, moving\n")
    f.write("permID |provID |trkSub   |mode |stn  |obsTime                     |ra          |dec         |mag   |band |rmsRA |rmsDec |astCat\n")
    f.writelines("       |       |%-9s|CCD  |X05  |%-27s |%12.8f |%+12.8f |%6.2f |%-5s|      |       |Gaia3\n" % (s, t + "Z", ra, de, mg, b)
                 for s, t, ra, de, mg, b in zip(t2d.trkSub.values, iso, t2d.RA.values, t2d.Dec.values, t2d.mag.values, t2d.band.values))
g.to_parquet(f"{stem}_meta.parquet"); t2d[["trkSub", "trk_ID", "detnum", "MJD", "RA", "Dec", "idstring"]].to_parquet(f"{stem}_dets.parquet")
print(stem, len(g), "tracklets", len(t2d), "rows")
