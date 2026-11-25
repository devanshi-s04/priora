"""DP2 labels from Rubin's own MPC submissions (B612 export, nights 60858-60866): each submitted observation is matched to the
nearest DP2 DiaSource (known or unknown set) within 2" and 30 s (DP2-era submissions are single positions, a few s from the
visit midpoint). Each matched DiaSource gets the submission's designation (provid, else permid) and status.
q from MPCORB (principal designation, or number for numbered objects). -> b612_labels.parquet (one row per matched DiaSource)"""
import os
import gzip, json, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from astropy.time import Time
X = pd.read_parquet("../b612_x05_dp2nights.parquet")
X["desig"] = X.provid.fillna(X.permid); X = X.dropna(subset=["desig"])
def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec); return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
out = []
for kind in ("known", "unknown"):
    d = pd.read_parquet(f"{kind}/dets.parquet", columns=["diaSourceId", "midpointMjdTai", "ra", "dec"])
    d["mjd_utc"] = Time(d.midpointMjdTai.values, format="mjd", scale="tai").utc.mjd; d["night"] = np.floor(d.midpointMjdTai - 0.5).astype(int)
    for n, dn in d.groupby("night"):
        xn = X[X.night == n]
        if xn.empty: continue
        tree = cKDTree(unit(dn.ra.values, dn.dec.values)); r = 2 * np.sin(np.radians(2 / 3600) / 2)
        L = tree.query_ball_point(unit(xn.ra.values, xn.dec.values), r)
        for j, cand in enumerate(L):
            if not cand: continue
            c = np.array(cand); dt = np.abs(dn.mjd_utc.values[c] - xn.mjd.values[j]) * 86400; ok = c[dt < 30]
            if len(ok) == 0: continue
            sep = np.degrees(np.arccos(np.clip(unit(dn.ra.values[ok], dn.dec.values[ok]) @ unit(np.array([xn.ra.values[j]]), np.array([xn.dec.values[j]]))[0], -1, 1))) * 3600
            k = ok[np.argmin(sep)]
            out.append((kind, dn.diaSourceId.values[k], xn.desig.values[j], xn.status.values[j], xn.trksub.values[j], sep.min(), (dn.mjd_utc.values[k] - xn.mjd.values[j]) * 86400))
        print(kind, n, "submissions", len(xn), "matched so far", len(out), flush=True)
R = pd.DataFrame(out, columns=["set", "diaSourceId", "desig", "status", "trksub", "sep_arcsec", "dt_s"]).sort_values("sep_arcsec").drop_duplicates("diaSourceId")
J = json.load(gzip.open(os.path.join(os.environ["PRIORA_ALERTS"], "data/mpcorb_extended.json.gz")))
q = {}
for o in J:
    for key in (o.get("Principal_desig"), (o.get("Number") or "").strip("()")):
        if key: q[key] = (o.get("Perihelion_dist"), o.get("H"), o.get("Orbit_type"))
R["q"] = R.desig.map(lambda s: q.get(s, (np.nan,))[0]); R["H"] = R.desig.map(lambda s: q.get(s, (np.nan, np.nan))[1]); R["orbit_type"] = R.desig.map(lambda s: q.get(s, (None, None, None))[2])
R.to_parquet("b612_labels.parquet")
print(R.groupby("set").agg(n=("diaSourceId", "size"), with_q=("q", lambda s: int(s.notna().sum())), neo=("q", lambda s: int((s < 1.3).sum()))))
print("separation quantiles", R.sep_arcsec.quantile([.5, .9, .99]).round(3).tolist(), "| dt quantiles", R.dt_s.quantile([.1, .5, .9]).round(1).tolist())
