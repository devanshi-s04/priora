"""DP2 known objects: digest2 raw >= 65 against priora All at the equal-recovery threshold (largest P, on a 0.01 grid, at
which priora selects at least as many NEOs as digest2), and at P >= 0.1 / 0.5. Pure tracklets with q from MPCORB, 2+ and
3+ detections. Run after analyze_known.py (reads score/known_joined.parquet) -> score/known_operating_points.json"""
import json, numpy as np, pandas as pd
m = pd.read_parquet("score/known_joined.parquet")
out = {}
for k in (2, 3):
    P = m[m.pure & m.q.notna() & (m.n >= k)]
    def fl(sel):
        f = P[sel]; nn = f[~f.neo]
        return dict(neo=int(f.neo.sum()), non_neo=len(nn), distant=int((nn.q >= 4.5).sum()))
    d2 = fl(P.d2_raw >= 65)
    pstar = next(t for t in np.round(np.arange(1, -1e-4, -.01), 2) if (P.neo & (P.priora_All >= t)).sum() >= d2["neo"])
    out[f"{k}plus"] = dict(tracklets=len(P), neo=int(P.neo.sum()), predicted=round(float(P.priora_All.sum()), 1),
                           digest2_raw_65=d2, p_star=float(pstar), priora_at_p_star=fl(P.priora_All >= pstar),
                           priora_01=fl(P.priora_All >= .1), priora_05=fl(P.priora_All >= .5))
json.dump(out, open("score/known_operating_points.json", "w"), indent=1)
print(json.dumps(out, indent=1))
