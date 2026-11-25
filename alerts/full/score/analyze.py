"""Unknown-alert test, step 5: scores x labels (../match/tracklet_labels.parquet). Same metrics as ../../score/analyze.py.
Sets: labelled (all), labelled & designated after the alert (truly unknown then). Other groups: flag rates only."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(20261103)
m = pd.read_parquet("scores_joined_nolabel.parquet").join(pd.read_parquet("../match/tracklet_labels.parquet")[["status", "desig", "q", "H", "neo", "designated_after", "sepmax"]])
ps = pd.read_parquet("trk_dets.parquet").sort_values(["trkSub", "MJD"]).groupby("trkSub")
e = ps.agg(ra0=("RA", "first"), ra1=("RA", "last"), de0=("Dec", "first"), de1=("Dec", "last"), t0=("MJD", "first"), t1=("MJD", "last"))
m["rate"] = np.hypot(((e.ra1 - e.ra0 + 180) % 360 - 180) * np.cos(np.radians(e.de0)), e.de1 - e.de0) / (e.t1 - e.t0)
m.to_parquet("scores_joined.parquet")
S = ("priora_All", "priora_Unk", "d2_raw", "d2_noid")
def auc_bs(y, s, B=2000):
    s = np.asarray(s); a = roc_auc_score(y, s); i1, i0 = np.where(y)[0], np.where(~y)[0]
    bs = [roc_auc_score(np.r_[np.ones(len(i1)), np.zeros(len(i0))], np.r_[s[rng.choice(i1, len(i1))], s[rng.choice(i0, len(i0))]]) for _ in range(B)]
    return [float(a), float(np.std(bs))]
out = {"groups": m.status.value_counts().to_dict()}
for tag, P in (("labelled", m[m.status == "labelled"]), ("designated_after", m[(m.status == "labelled") & m.designated_after])):
    y = P.neo.values.astype(bool); r = {"n": len(P), "n_neo": int(y.sum()), "n_objects": int(P.desig.nunique())}
    r["auc"] = {k: auc_bs(y, P[k]) for k in S}; r["sumP"] = {k: float(P[k].sum()) for k in ("priora_All", "priora_Unk")}
    fl = {}
    for name, sel in (("d2_raw>=65", P.d2_raw >= 65), ("d2_noid>=65", P.d2_noid >= 65), ("priora_All>=0.1", P.priora_All >= 0.1), ("priora_Unk>=0.1", P.priora_Unk >= 0.1),
                      ("priora_All>=0.5", P.priora_All >= 0.5), ("priora_Unk>=0.5", P.priora_Unk >= 0.5)):
        f = P[sel]; fl[name] = dict(flagged=int(len(f)), neo=int(f.neo.sum()), non_neo=int((~f.neo.astype(bool)).sum()), distant=int(((~f.neo.astype(bool)) & (f.q >= 4.5)).sum()))
    r["flags"] = fl; out[tag] = r
out["flag_rates_by_group"] = {g: {k: float((G[k] >= (65 if k.startswith("d2") else .5)).mean()) for k in S} | {"n": len(G)} for g, G in m.groupby("status")}
out["flag_rates_by_npts"] = {f"{n}": {k: float((G[k] >= (65 if k.startswith("d2") else .5)).mean()) for k in S} | {"n": len(G)} for n, G in m.groupby(m.n.clip(upper=4))}
out["rate_pct"] = {g: G.rate.quantile([.1, .5, .9]).round(2).tolist() for g, G in m.groupby("status")}
json.dump(out, open("analysis.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
