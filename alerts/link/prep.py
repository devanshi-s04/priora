"""heliolinc inputs from a make_tracklets run: keep tracklets whose images are on July nights (dayObs 2026-07-05..13)
and that move >= 0.5 arcsec between their two representative points; renumber trk_ID; pairdets and imgs unchanged.
Usage: python3 prep.py <tracklets dir> <tag> <out prefix>"""
import sys, numpy as np, pandas as pd
D, tag, out = sys.argv[1:4]
img = pd.read_csv(f"{D}/imgs_{tag}.txt", sep=r"\s+", header=None, comment="#")
mjd = img[0].values if np.issubdtype(img[0].dtype, np.number) else img[1].values
trk = pd.read_csv(f"{D}/tracklets_{tag}.csv"); trk.columns = [c.lstrip("#") for c in trk.columns]
t2d = pd.read_csv(f"{D}/trk2det_{tag}.csv"); t2d.columns = [c.lstrip("#") for c in t2d.columns]
night = np.floor(mjd[trk.Image1.values] - 0.5).astype(int)
disp = np.hypot(((trk.RA2 - trk.RA1 + 180) % 360 - 180) * np.cos(np.radians(trk.Dec1)), trk.Dec2 - trk.Dec1) * 3600
keep = (night >= 61226) & (night <= 61234) & (disp >= 0.5)
k = trk[keep].copy(); new = dict(zip(k.trk_ID, range(len(k)))); k["trk_ID"] = k.trk_ID.map(new)
t = t2d[t2d.trk_ID.isin(new)].copy(); t["trk_ID"] = t.trk_ID.map(new)
with open(f"{out}_tracklets.csv", "w") as f:
    f.write("#" + ",".join(k.columns) + "\n"); k.to_csv(f, header=False, index=False, float_format="%.7f")
with open(f"{out}_trk2det.csv", "w") as f:
    f.write("#trk_ID,detnum\n"); t.to_csv(f, header=False, index=False)
pd.DataFrame({"old": list(new), "new": list(new.values())}).to_csv(f"{out}_trkmap.csv", index=False)
print(f"{tag}: tracklets {len(trk):,}; July {((night >= 61226) & (night <= 61234)).sum():,}; kept (July, moving) {len(k):,}; nights {sorted(set(night[keep]))}")
