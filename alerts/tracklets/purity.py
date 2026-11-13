"""Purity and completeness of make_tracklets output, using the hidden designation (idstring) of each paired detection."""
import sys, numpy as np, pandas as pd
v = sys.argv[1]
pd_ = pd.read_csv(f"pairdets_v{v}.csv"); pd_.columns = [c.lstrip("#") for c in pd_.columns]
t2d = pd.read_csv(f"trk2det_v{v}.csv"); t2d.columns = [c.lstrip("#") for c in t2d.columns]
trk = pd.read_csv(f"tracklets_v{v}.csv"); trk.columns = [c.lstrip("#") for c in trk.columns]
lab = pd.read_parquet("../data/labels_ftransfer_2026-11-02.parquet").set_index("packed")
t2d["obj"] = pd_.idstring.values[t2d.detnum.values]; t2d["mjd"] = pd_.MJD.values[t2d.detnum.values]
g = t2d.groupby("trk_ID").agg(n=("obj", "size"), nobj=("obj", "nunique"), obj=("obj", "first"), t0=("mjd", "min"), t1=("mjd", "max"))
g["pure"] = g.nobj == 1; g["arc"] = (g.t1 - g.t0) * 1440; g["night"] = np.floor(g.t0 - 0.5).astype(int)
g["neo"] = g.obj.map(lab.neo).fillna(False).astype(bool) & g.pure
print(f"v={v}: tracklets {len(g):,}; pure {g.pure.sum():,} ({g.pure.mean():.1%}); mixed {(~g.pure).sum():,}")
# multiplicity: how many tracklets per true object-night
p = g[g.pure]; per = p.groupby(["obj", "night"]).size()
print(f"   pure tracklets cover {len(per):,} object-nights; tracklets per object-night: median {per.median():.0f}, mean {per.mean():.2f}, max {per.max()}")
# truth object-nights with >=2 dets (from the input)
d = pd.read_csv("dets_julyoct.csv"); d.columns = [c.lstrip("#") for c in d.columns]
d["night"] = np.floor(d.MJD_TAI - 0.5).astype(int)
truth = d.groupby(["ObjID", "night"]).size(); truth = truth[truth >= 2]
rec = truth.index.isin(per.index)
neo_truth = np.asarray(pd.Series(truth.index.get_level_values(0)).map(lab.neo).fillna(False).astype(bool))
print(f"   completeness: {rec.sum():,}/{len(truth):,} object-nights ({rec.mean():.1%}); NEO {rec[neo_truth].sum()}/{neo_truth.sum()}")
print(f"   pure NEO tracklets {g.neo.sum():,}; NEO object-nights covered {p[p.neo].groupby(['obj','night']).ngroups}")
print(f"   npts distribution of all tracklets: {g.n.value_counts().sort_index().head(8).to_dict()}")
g.to_parquet(f"tracklet_labels_v{v}.parquet")
