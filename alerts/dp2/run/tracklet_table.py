"""Per-tracklet table for a DP2 set: n, arc (min), total motion (arcsec), rate, night; purity for the known set (hidden ID =
designation). Keeps arc >= 15 min and motion >= 0.5" (moving). Usage: tracklet_table.py known|unknown -> <set>/tracklets_table.parquet"""
import sys, numpy as np, pandas as pd
k = sys.argv[1]
pdet = pd.read_csv(f"{k}/pairdets.csv"); pdet.columns = [c.lstrip("#") for c in pdet.columns]
t2d = pd.read_csv(f"{k}/trk2det.csv"); t2d.columns = ["trk_ID", "detnum"]
t2d = t2d.join(pdet[["MJD", "RA", "Dec", "idstring"]], on="detnum").sort_values(["trk_ID", "MJD"])
g = t2d.groupby("trk_ID").agg(n=("detnum", "size"), t0=("MJD", "first"), t1=("MJD", "last"), ra0=("RA", "first"), ra1=("RA", "last"),
                              de0=("Dec", "first"), de1=("Dec", "last"), nobj=("idstring", "nunique"), obj=("idstring", "first"))
g["arc"] = (g.t1 - g.t0) * 1440; g["night"] = np.floor(g.t0 - 0.5).astype(int)
g["disp"] = np.hypot(((g.ra1 - g.ra0 + 180) % 360 - 180) * np.cos(np.radians(g.de0)), g.de1 - g.de0) * 3600
g["rate"] = g.disp / 3600 / ((g.t1 - g.t0).clip(lower=1e-6))
keep = (g.arc >= 15) & (g.disp >= 0.5)
print(f"{k}: tracklets {len(g):,}; arc>=15 {int((g.arc >= 15).sum()):,}; + moving {int(keep.sum()):,}; by n {g[keep].n.clip(upper=4).value_counts().sort_index().to_dict()}")
if k == "known":
    K = g[keep]; print(f"   pure {int((K.nobj == 1).sum()):,} ({(K.nobj == 1).mean():.1%}); mixed {int((K.nobj > 1).sum()):,}; pure 3+ {int(((K.nobj == 1) & (K.n >= 3)).sum()):,}")
g[keep].drop(columns=["ra0", "ra1", "de0", "de1"]).to_parquet(f"{k}/tracklets_table.parquet")
