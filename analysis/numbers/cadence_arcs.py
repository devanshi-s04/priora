"""Section 7: same-night arc lengths in the first year of the LSST baseline cadence simulation v5.3.7.
For 3000 random sky points (dec < +20 deg, seed 2), find every visit within 1.75 deg (about the LSSTCam field radius) and,
for each point-night with >= 2 visits, the time from first to last visit. Writes cadence_arcs.json.
Input: $DS42_EXTERNAL/opsim/baseline_v5.3.7_10yrs.db (rubin_sim baseline). Run: python analysis/numbers/cadence_arcs.py"""
import os
import json, sqlite3
from pathlib import Path
import numpy as np
DB = Path(os.environ["DS42_EXTERNAL"]) / "opsim/baseline_v5.3.7_10yrs.db"
a = np.array(sqlite3.connect(DB).execute("select fieldRA, fieldDec, observationStartMJD, night from observations where night < 365").fetchall(), float)
ra, dec, t, n = np.radians(a[:, 0]), np.radians(a[:, 1]), a[:, 2] * 1440, a[:, 3].astype(int)
rng = np.random.default_rng(2); m = 3000
pr = rng.uniform(0, 2 * np.pi, m); pd = np.arcsin(rng.uniform(-1, 0.35, m))
arcs = []
for i in range(m):
    c = np.sin(dec) * np.sin(pd[i]) + np.cos(dec) * np.cos(pd[i]) * np.cos(ra - pr[i]); s = np.where(c > np.cos(np.radians(1.75)))[0]
    for night in np.unique(n[s]):
        tt = np.sort(t[s][n[s] == night])
        if tt.size >= 2: arcs.append(tt[-1] - tt[0])
A = np.array(arcs)
out = dict(n_point_nights=int(A.size), percentiles_10_25_50_75_90=np.percentile(A, [10, 25, 50, 75, 90]).round(1).tolist(),
           median=float(np.median(A)), frac_lt={str(x): float((A < x).mean()) for x in (5, 10, 15, 30, 45, 60)})
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1); print(out)
