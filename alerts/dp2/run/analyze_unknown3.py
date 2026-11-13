"""DP2 unknown tracklets with 3+ detections (no SSSource match when DP2 was processed; nDiaSources <= 2): labels from Rubin's MPC
submissions (b612_labels.parquet, exact match); a tracklet is labelled when every detection matched the same designation.
Evaluation on labelled ones; candidates = unlabelled with ds42 P >= 0.1. -> score/unknown3_joined.parquet, score/unknown3_analysis.json"""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
meta = pd.read_parquet("score/unknown_3plus_meta.parquet").set_index("trkSub")
for V in ("All", "Unk"): meta[f"ds42_{V}"] = pd.read_csv(f"score/unknown_3plus_ds42_{V}.csv").set_index("object_id").p_neo
d2 = pd.read_csv("score/unknown_3plus_digest2.csv").set_index("trkSub"); meta["d2_raw"] = d2.raw_neo; meta["d2_noid"] = d2.noid_neo
dets = pd.read_parquet("score/unknown_3plus_dets.parquet"); dets["diaSourceId"] = dets.idstring.astype("int64")
L = pd.read_parquet("b612_labels.parquet"); L = L[L.set == "unknown"].set_index("diaSourceId")
dets["desig"] = dets.diaSourceId.map(L.desig); dets["q"] = dets.diaSourceId.map(L.q); dets["status"] = dets.diaSourceId.map(L.status)
g = dets.groupby("trkSub").agg(n_lab=("desig", "count"), n=("desig", "size"), ndes=("desig", "nunique"), desig=("desig", "first"), q=("q", "first"), st=("status", "first"))
meta = meta.join(g[["n_lab", "ndes", "desig", "q", "st"]])
meta["label"] = np.select([(meta.n_lab == meta.n) & (meta.ndes == 1), meta.n_lab > 0], ["labelled", "partial"], "unlabelled")
meta["neo"] = (meta.q < 1.3) & (meta.label == "labelled")
meta.to_parquet("score/unknown3_joined.parquet")
Lb = meta[(meta.label == "labelled") & meta.q.notna()]; y = Lb.neo.values
out = dict(n=len(meta), groups=meta.label.value_counts().to_dict(), labelled_with_q=len(Lb), neo=int(y.sum()), sumP_All=float(Lb.ds42_All.sum()), sumP_Unk=float(Lb.ds42_Unk.sum()),
           auc={c: float(roc_auc_score(y, Lb[c])) for c in ("ds42_All", "ds42_Unk", "d2_raw", "d2_noid")} if 0 < y.sum() < len(y) else {},
           d2_raw_65=dict(flagged=int((Lb.d2_raw >= 65).sum()), non_neo=int(((Lb.d2_raw >= 65) & ~Lb.neo).sum()), distant=int(((Lb.d2_raw >= 65) & ~Lb.neo & (Lb.q >= 4.5)).sum())),
           ds42_01=dict(flagged=int((Lb.ds42_All >= .1).sum()), non_neo=int(((Lb.ds42_All >= .1) & ~Lb.neo).sum()), distant=int(((Lb.ds42_All >= .1) & ~Lb.neo & (Lb.q >= 4.5)).sum())),
           hiP=int((meta.ds42_All >= .5).sum()), hiD=int((meta.d2_raw >= 65).sum()),
           candidates=int(((meta.label == "unlabelled") & (meta.ds42_All >= .1)).sum()), candidates_p05=int(((meta.label == "unlabelled") & (meta.ds42_All >= .5)).sum()))
json.dump(out, open("score/unknown3_analysis.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
print(Lb[Lb.neo][["desig", "q", "n", "ds42_All", "ds42_Unk", "d2_raw", "d2_noid", "st"]].round(3).to_string())
