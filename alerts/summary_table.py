"""One results table across the three alerts tests, for tracklets with >= 2, 3, 4 detections -> summary_table.json.
Evaluation set: known test = pure tracklets; unknown tests = catalogue-labelled tracklets."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
S = {"known": "score/scores_joined.parquet", "unk": "unk/score/scores_joined.parquet", "full": "full/score/scores_joined.parquet"}
out = {}
for t, f in S.items():
    m = pd.read_parquet(f)
    E0 = m[m.pure] if t == "known" else m[m.status == "labelled"]
    O0 = m[~m.pure] if t == "known" else m[m.status != "labelled"]
    for k in (2, 3, 4):
        E = E0[E0.n >= k]; O = O0[O0.n >= k]; y = E.neo.astype(bool).values
        def fl(sel):
            f = E[sel]; nn = f[~f.neo.astype(bool)]
            return [int(len(f)), int(f.neo.astype(bool).sum()), int(len(nn)), int((nn.q >= 4.5).sum())]
        out[f"{t}_{k}"] = dict(n=int(len(E)), neo=int(y.sum()), other=int(len(O)), all_tracklets=int((m.n >= k).sum()),
            sumP_All=float(E.ds42_All.sum()), sumP_Unk=float(E.ds42_Unk.sum()),
            auc_All=float(roc_auc_score(y, E.ds42_All)), auc_raw=float(roc_auc_score(y, E.d2_raw)),
            d2raw=fl(E.d2_raw >= 65), d2noid=fl(E.d2_noid >= 65), ds42_01=fl(E.ds42_All >= 0.1), ds42_05=fl(E.ds42_All >= 0.5),
            other_flag_d2raw=int((O.d2_raw >= 65).sum()), other_flag_ds42=int((O.ds42_All >= 0.5).sum()))
json.dump(out, open("summary_table.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
