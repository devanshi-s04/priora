"""Step 5 first pass: ds42 (All, Unk) vs digest2 (raw, NoID) on the HelioLinC maxvel-5, arc>=15 min tracklets.
Pure tracklets: AUC (bootstrap), flags at digest2 >= 65 vs ds42 at matched completeness, false alarms by q. Mixed: flag rates.
-> analysis.json"""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(20261102)
m = pd.read_parquet("meta.parquet").set_index("trkSub")
for V in ("All", "Unk"):
    m[f"ds42_{V}"] = pd.read_csv(f"ds42_{V}.csv").set_index("object_id").p_neo
d2 = pd.read_csv("digest2.csv").set_index("trkSub"); m["d2_raw"] = d2.raw_neo; m["d2_noid"] = d2.noid_neo
# sky rate from the PSV endpoints
ps = pd.read_csv("tracklets_v5_arc15.psv", sep="|", comment="#", skipinitialspace=True); ps.columns = [c.strip() for c in ps.columns]
ps["trkSub"] = ps.trkSub.str.strip(); ps["t"] = pd.to_datetime(ps.obsTime.str.strip().str.rstrip("Z"))
e = ps.groupby("trkSub").agg(ra0=("ra", "first"), ra1=("ra", "last"), de0=("dec", "first"), de1=("dec", "last"), t0=("t", "first"), t1=("t", "last"))
dra = ((e.ra1 - e.ra0 + 180) % 360 - 180) * np.cos(np.radians(e.de0))
m["rate"] = np.hypot(dra, e.de1 - e.de0) / ((e.t1 - e.t0).dt.total_seconds() / 86400)
assert m[["ds42_All", "ds42_Unk", "d2_raw", "d2_noid", "rate"]].notna().all().all()
P = m[m.pure]; X = m[~m.pure]; y = P.neo.values
out = {"n_pure": len(P), "n_neo": int(y.sum()), "n_mixed": len(X)}
def auc_bs(s, B=2000):
    s = np.asarray(s); a = roc_auc_score(y, s); i1, i0 = np.where(y)[0], np.where(~y)[0]
    bs = [roc_auc_score(np.r_[np.ones(len(i1)), np.zeros(len(i0))], np.r_[s[rng.choice(i1, len(i1))], s[rng.choice(i0, len(i0))]]) for _ in range(B)]
    return [float(a), float(np.std(bs))]
S = {"ds42_All": P.ds42_All, "ds42_Unk": P.ds42_Unk, "d2_raw": P.d2_raw, "d2_noid": P.d2_noid}
out["auc"] = {k: auc_bs(v) for k, v in S.items()}
out["sumP"] = {k: float(P[k].sum()) for k in ("ds42_All", "ds42_Unk")}
def flags(sel):
    f = P[sel]; nn = f[~f.neo]
    return dict(flagged=int(len(f)), neo=int(f.neo.sum()), completeness=float(f.neo.sum() / y.sum()), non_neo=int(len(nn)),
                distant=int((nn.q >= 4.5).sum()), contamination=float(len(nn) / max(len(f), 1)))
fl = {}
for d in ("d2_raw", "d2_noid"):
    fl[f"{d}>=65"] = flags(P[d] >= 65); k = fl[f"{d}>=65"]["neo"]
    for V in ("ds42_All", "ds42_Unk"):                 # ds42 threshold giving the same number of NEOs
        thr = np.sort(P[V][P.neo].values)[::-1][k - 1] if k else 1.0
        fl[f"{V} matched to {d} (P>={thr:.3g})"] = flags(P[V] >= thr)
fl["ds42_All P>=0.5"] = flags(P.ds42_All >= 0.5)
out["flags_pure"] = fl
out["mixed"] = {k: {"n_flag": int(c.sum()), "frac": float(c.mean())} for k, c in
                {"d2_raw>=65": X.d2_raw >= 65, "d2_noid>=65": X.d2_noid >= 65, "ds42_All>=0.5": X.ds42_All >= 0.5, "ds42_Unk>=0.5": X.ds42_Unk >= 0.5}.items()}
out["mixed_rate_pct"] = X.rate.quantile([.1, .5, .9]).round(2).tolist(); out["pure_rate_pct"] = P.rate.quantile([.1, .5, .9]).round(2).tolist()
out["neo_rate_pct"] = P.rate[P.neo].quantile([.1, .5, .9]).round(2).tolist()
for c in (1.0, 2.0):
    out[f"rate>={c}"] = dict(mixed=int((X.rate >= c).sum()), pure=int((P.rate >= c).sum()), pure_neo=int((P.neo & (P.rate >= c)).sum()))
m.to_parquet("scores_joined.parquet"); json.dump(out, open("analysis.json", "w"), indent=1)
print(json.dumps(out, indent=1))
