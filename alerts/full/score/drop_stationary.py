"""Full pull only: the repeat-object detections include stationary variable sources, which make_tracklets (minvel 0)
pairs with themselves. Drop tracklets that move < 0.5 arcsec in total (stationary median 0.07", 99% < 0.35"; real slow objects merged onto one diaObject: median 0.46").
scores_joined_nolabel_all.parquet (kept) -> scores_joined_nolabel.parquet (moving only)."""
import os, numpy as np, pandas as pd
if not os.path.exists("scores_joined_nolabel_all.parquet"):
    os.rename("scores_joined_nolabel.parquet", "scores_joined_nolabel_all.parquet")
m = pd.read_parquet("scores_joined_nolabel_all.parquet")
td = pd.read_parquet("trk_dets.parquet"); td["diaSourceId"] = td.idstring.astype("int64")
D = pd.read_parquet("../tracklets/dets_full_r90.parquet")[["diaSourceId", "diaObjectId", "nDiaSources"]]
td = td.merge(D, on="diaSourceId", how="left").sort_values(["trkSub", "MJD"])
g = td.groupby("trkSub").agg(nobj=("diaObjectId", "nunique"), nrep=("nDiaSources", lambda s: int((s > 1).sum())),
                             ra0=("RA", "first"), ra1=("RA", "last"), de0=("Dec", "first"), de1=("Dec", "last"))
g["disp"] = np.hypot(((g.ra1 - g.ra0 + 180) % 360 - 180) * np.cos(np.radians(g.de0)), g.de1 - g.de0) * 3600
m = m.join(g[["nobj", "nrep", "disp"]])
stat = m.disp < 0.5
print(f"tracklets {len(m):,}; stationary dropped {stat.sum():,} (< 0.5\"); moving {(~stat).sum():,}; "
      f"moving with repeat-object detections {((~stat) & (m.nrep > 0)).sum():,}")
m[~stat].to_parquet("scores_joined_nolabel.parquet")
