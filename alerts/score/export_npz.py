"""Write the pure LSST tracklets in the paper figure scripts' cache format (validation/output/<run>/{scores,gen}.npz and
digest2_scores.npz), so make_d2cmp_hext30.py / make_linear_hext30.py run unchanged via R_RUN / R_CMP / CAL_RUNS.
elements[:,0] = q, elements[:,1] = 0 so the scripts' q = a(1-e) gives the MPC perihelion distance.
Sets: all pure tracklets, and the post-2011 subset (objects first designated 2012+, i.e. unknown when the Unk model's
catalogue was frozen)."""
import os
import numpy as np, pandas as pd
from pathlib import Path
O = Path(os.environ["PRIORA_REPO"]) / "validation/output"
m = pd.read_parquet("scores_joined.parquet"); lab = pd.read_parquet("../data/labels_ftransfer_2026-11-02.parquet").set_index("packed")
P = m[m.pure].copy(); P["yr"] = P.obj.map(lab.unpacked).str.extract(r"^(\d{4}) ")[0].astype(float)
for tag, S in (("", P), ("_post2011", P[P.yr >= 2012])):
    n = len(S); idx = np.arange(n); el = np.zeros((n, 6)); el[:, 0] = S.q.values
    for V in ("All", "Unk"):
        d = O / f"lsst_alerts_{V}{tag}"; d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "scores.npz", index=idx, p_neo=S[f"priora_{V}"].values)
        np.savez(d / "gen.npz", is_neo=S.neo.values, elements=el)
        c = O / f"lsst_alerts_d2{tag}" / f"lsst_alerts_{V}{tag}"; c.mkdir(parents=True, exist_ok=True)
        np.savez(c / "digest2_scores.npz", index=idx, raw_neo=S.d2_raw.values, noid_neo=S.d2_noid.values, ok=np.ones(n, bool))
    print(tag or "all", n, "NEO", int(S.neo.sum()))
