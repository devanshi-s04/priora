"""Mario's test, bright version (2026-11-06): of digest2's distant false alarms (q >= 4.5 AU, digest2 >= 65) that pre-Rubin surveys
could see (V < 22), how many are objects of a kind that is already catalogued?
Completeness per (q range, H bin) = known MPCORB objects / model population (digest2 model 'All', cell counts un-scaled with the
model's own rescaling, Eq. muk), also the model's own known fraction 1 - Unk/All. Because S3M may overpredict the distant
population, completeness is also shown normalised to its value in the brightest well-populated bins.
-> bright_numbers.json"""
import os
import gzip, json
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
M = pd.read_csv("data/digest2.model.csv"); M = M[M.Class == "SS"].fillna(0.0); Hc = [c for c in M.columns if c.startswith("H")]
Hhi = np.array([float(c[1:]) for c in Hc]); Hlo = np.r_[0.0, Hhi[:-1]]
def edges(col):
    hi = np.sort(M[col].unique()); return dict(zip(hi, np.r_[0.0, hi[:-1]]))
qL, eL, iL = edges("Q"), edges("e"), edges("i")
def unscale(df):
    q1, e1, i1 = df.Q.values, df.e.values, df.i.values; q0 = np.array([qL[v] for v in q1]); e0 = np.array([eL[v] for v in e1]); i0 = np.array([iL[v] for v in i1])
    om = lambda e: np.where(e >= 1, 0.0, 1 - e)
    with np.errstate(divide="ignore", invalid="ignore"):
        ef = np.where((om(e0) + om(e1)) > 0, (e1 - e0) / (om(e0) + om(e1)), 0.0)
    base = (q1 - q0) * ef * (i1 - i0)
    return df[Hc].values * np.sqrt(base[:, None] * (Hhi - Hlo)[None, :])
A = M[M.Model == "All"]; U = M[M.Model == "Unk"].set_index(["Q", "e", "i"]).reindex(A.set_index(["Q", "e", "i"]).index).fillna(0.0).reset_index()
cA, cU = unscale(A), unscale(U); qhi = A.Q.values
rows = []
with gzip.open(Path(os.environ["DS42_ALERTS"]) / "dp2/mpcorb/MPCORB.DAT.gz", "rt", errors="ignore") as f:
    go = False
    for L in f:
        if L.startswith("-----"): go = True; continue
        if go and len(L) > 103 and L[8:13].strip():
            try: rows.append((float(L[8:13]), float(L[92:103]) * (1 - float(L[70:79]))))
            except ValueError: pass
K = pd.DataFrame(rows, columns=["H", "q"])
def load(run, d2dir, sc):
    g = np.load(f"validation/output/{run}/gen.npz"); z = np.load(f"validation/output/{run}/scores.npz"); d = np.load(f"validation/output/{d2dir}/digest2_scores.npz")
    idx = np.intersect1d(z["index"], d["index"]); idd = np.searchsorted(d["index"], idx); el = g["elements"][idx]
    return dict(y=g["is_neo"][idx].astype(bool), q=el[:, 0] * (1 - el[:, 1]), H=g["H"][idx], V=g["v_mag"][idx], s=d[sc][idd])
res = {}
for lo, hi, name in ((4.5, 10, "Trojans/Centaurs 4.5-10 AU"), (10, 1e9, "TNOs >= 10 AU")):
    # model q bins whose upper edge lies in (lo, hi]: the q bin edges include 4.5 and 10? use cells with lower edge >= lo and upper <= hi
    qlo = np.array([qL[v] for v in qhi]); sel = (qlo >= lo - 1e-9) & (qhi <= hi + 1e-9)
    allH, unkH = cA[sel].sum(0), cU[sel].sum(0)
    kn = K[(K.q >= qlo[sel].min()) & (K.q < qhi[sel].max())]
    knH = np.array([((kn.H >= a) & (kn.H < b)).sum() for a, b in zip(Hlo, Hhi)])
    comp_cat = np.where(allH > 0, knH / np.where(allH > 0, allH, 1), np.nan); comp_mod = np.where(allH > 0, 1 - unkH / np.where(allH > 0, allH, 1), np.nan)
    table = [dict(H=f"{a:g}-{b:g}", model_all=round(float(x), 1), known_mpcorb=int(k), known_over_model=None if not np.isfinite(c) else round(float(c), 3), model_known_frac=None if not np.isfinite(m) else round(float(m), 3))
             for a, b, x, k, c, m in zip(Hlo, Hhi, allH, knH, comp_cat, comp_mod) if x > 0 or k > 0]
    fas = {}
    for nm, run, d2, sc in (("All", "gate_v0_hext30", "digest2_comparison_hext30/gate_v0_hext30", "raw_neo"), ("Unk", "gate_v0unk_hext30", "digest2_comparison_unk", "noid_neo")):
        D = load(run, d2, sc); fa = (D["s"] >= 65) & ~D["y"] & (D["q"] >= lo) & (D["q"] < hi); br = fa & (D["V"] < 22)
        kb = np.searchsorted(Hhi, D["H"][br], side="right")
        fas[nm] = dict(n_far=int(fa.sum()), n_bright=int(br.sum()), H_bright=sorted(np.round(D["H"][br], 1).tolist()),
                       mean_known_over_model=round(float(np.nanmean(comp_cat[np.clip(kb, 0, len(Hhi) - 1)])), 3) if br.any() else None,
                       mean_model_known_frac=round(float(np.nanmean(comp_mod[np.clip(kb, 0, len(Hhi) - 1)])), 3) if br.any() else None)
    res[name] = dict(q_cells=f"{qlo[sel].min():g}-{qhi[sel].max():g} AU", H_table=table, false_alarms=fas)
(OUT / "bright_numbers.json").write_text(json.dumps(res, indent=1))
for name, r in res.items():
    print("==", name, r["q_cells"])
    for t in r["H_table"]: print("  ", t)
    for nm, f in r["false_alarms"].items(): print("  FA", nm, {k: v for k, v in f.items() if k != "H_bright"}, "H of bright FAs:", f["H_bright"][:40])
