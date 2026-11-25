"""DP2 known-object test (Early DP2, July 3-11 2025): HelioLinC tracklets of Rubin-associated detections (hidden ID = Rubin's
designation), arc >= 15 min, moving. Pure tracklets labelled with q from MPCORB (principal designation or number).
Calibration, AUC (bootstrap), flags; all, 3+, 4+ detections. -> score/known_analysis.json, score/known_joined.parquet"""
import os
import gzip, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(20251107)
meta = pd.read_parquet("score/known_all_meta.parquet").set_index("trkSub")
for V in ("All", "Unk"): meta[f"priora_{V}"] = pd.read_csv(f"score/known_all_priora_{V}.csv").set_index("object_id").p_neo
d2 = pd.read_csv("score/known_all_digest2.csv").set_index("trkSub"); meta["d2_raw"] = d2.raw_neo; meta["d2_noid"] = d2.noid_neo
J = json.load(gzip.open(os.path.join(os.environ["PRIORA_ALERTS"], "data/mpcorb_extended.json.gz"))); q = {}
for o in J:
    for key in (o.get("Principal_desig"), (o.get("Number") or "").strip("()")):
        if key: q[key.replace(" ", "_")] = o.get("Perihelion_dist")
meta["pure"] = meta.nobj == 1; meta["q"] = meta.obj.map(q).where(meta.pure); meta["neo"] = meta.q < 1.3
meta.to_parquet("score/known_joined.parquet")
P0 = meta[meta.pure & meta.q.notna()]
out = {"tracklets": len(meta), "pure": int(meta.pure.sum()), "pure_with_q": len(P0), "pure_no_q": int((meta.pure & meta.q.isna()).sum()),
       "mixed": int((~meta.pure).sum()), "mixed_flag_priora_05": int((meta[~meta.pure].priora_All >= .5).sum()), "mixed_flag_d2_65": int((meta[~meta.pure].d2_raw >= 65).sum())}
for k in (2, 3, 4):
    P = P0[P0.n >= k]; y = P.neo.values.astype(bool)
    def auc(s, B=300):
        a = roc_auc_score(y, s); i1, i0 = np.where(y)[0], np.where(~y)[0]
        return [float(a), float(np.std([roc_auc_score(np.r_[np.ones(len(i1)), np.zeros(len(i0))], np.r_[s[rng.choice(i1, len(i1))], s[rng.choice(i0, len(i0))]]) for _ in range(B)]))]
    def fl(sel):
        f = P[sel]; nn = f[~f.neo]; return dict(flagged=len(f), neo=int(f.neo.sum()), non_neo=len(nn), distant=int((nn.q >= 4.5).sum()))
    out[f"{k}plus"] = dict(n=len(P), neo=int(y.sum()), sumP_All=float(P.priora_All.sum()), sumP_Unk=float(P.priora_Unk.sum()),
                          auc={c: auc(P[c].values) for c in ("priora_All", "priora_Unk", "d2_raw", "d2_noid")},
                          d2_raw_65=fl(P.d2_raw >= 65), d2_noid_65=fl(P.d2_noid >= 65), priora_All_01=fl(P.priora_All >= 0.1), priora_All_05=fl(P.priora_All >= 0.5))
    E = np.array([0, .01, .1, .3, .5, .7, .9, 1.0001]); p = P.priora_All.values
    out[f"{k}plus"]["calib_bins"] = [dict(lo=float(l), n=int(((p >= l) & (p < h)).sum()), obs=int(y[(p >= l) & (p < h)].sum()), exp=float(p[(p >= l) & (p < h)].sum())) for l, h in zip(E[:-1], E[1:])]
json.dump(out, open("score/known_analysis.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if not k.endswith("plus")}, indent=1))
for k in ("2plus", "3plus", "4plus"):
    o = out[k]; print(k, "n", o["n"], "NEO", o["neo"], "sumP All %.1f Unk %.1f" % (o["sumP_All"], o["sumP_Unk"]), "| AUC", {c: round(v[0], 4) for c, v in o["auc"].items()})
    print("   d2 raw>=65", o["d2_raw_65"], "| priora P>=0.1", o["priora_All_01"], "| P>=0.5", o["priora_All_05"]); print("   bins", [(b["lo"], b["n"], b["obs"], round(b["exp"], 1)) for b in o["calib_bins"]])
