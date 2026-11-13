"""Are digest2's distant false alarms objects the surveys would already know? (Mario, 2026-11-05)
For the digest2 >= 65 non-NEO false alarms with q >= 4.5 AU in the simulated comparison samples:
 (1) H distribution against known MPCORB objects in the same q range (Trojans/Centaurs 4.5-10 AU; TNOs >= 10 AU);
 (2) the fraction of each false alarm's (q,e,i,H) cell already discovered, 1 - Unk/All from the digest2 model file itself,
     summed into an expected number of known objects among the false alarms;
 (3) apparent V and sky rate, for comparison with what surveys submit.
Samples: All (gate_v0_hext30, digest2 raw) and Unk (gate_v0unk_hext30, digest2 NoID; undiscovered by construction).
Run from repo root. -> numbers.json, h_distribution.{png,pdf}"""
import os
import gzip, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
O = Path("validation/output"); OUT = Path(__file__).resolve().parent
M = pd.read_csv("data/digest2.model.csv"); Hcols = [c for c in M.columns if c.startswith("H")]; Hedges = np.array([float(c[1:]) for c in Hcols])
M = M[M.Class == "SS"].fillna(0.0)
key = ["Q", "e", "i"]; A = M[M.Model == "All"].set_index(key)[Hcols]; U = M[M.Model == "Unk"].set_index(key)[Hcols].reindex(A.index).fillna(0.0)
qE, eE, iE = (np.sort(M[k].unique()) for k in key)
def cell_completeness(q, e, i, H):
    out = np.full(len(q), np.nan)
    for n, (a, b, c, h) in enumerate(zip(q, e, i, H)):
        kq, ke, ki, kh = (np.searchsorted(E, v, side="right") for E, v in ((qE, a), (eE, b), (iE, c), (Hedges, h)))
        if kq >= len(qE) or ke >= len(eE) or ki >= len(iE) or kh >= len(Hedges): continue
        k = (qE[kq], eE[ke], iE[ki]); al = A.loc[k].iloc[kh]; un = U.loc[k].iloc[kh]
        if al > 0: out[n] = float(np.clip(1 - un / al, 0, 1))
    return out
def load(run, d2dir, score):
    g = np.load(O / run / "gen.npz"); z = np.load(O / run / "scores.npz"); d = np.load(O / d2dir / "digest2_scores.npz")
    idx = np.intersect1d(z["index"], d["index"]); idd = np.searchsorted(d["index"], idx)
    el = g["elements"][idx]
    return dict(y=g["is_neo"][idx].astype(bool), q=el[:, 0] * (1 - el[:, 1]), e=el[:, 1], i=np.degrees(el[:, 2]), H=g["H"][idx], V=g["v_mag"][idx],
                rate=np.degrees(np.hypot(g["ra_rate"][idx] * np.cos(np.radians(g["dec"][idx])), g["dec_rate"][idx])), s=d[score][idd])
# known objects (MPCORB, downloaded 2026-11-04)
rows = []
with gzip.open(Path(os.environ["DS42_ALERTS"]) / "dp2/mpcorb/MPCORB.DAT.gz", "rt", errors="ignore") as f:
    go = False
    for L in f:
        if L.startswith("-----"): go = True; continue
        if go and len(L) > 103 and L[8:13].strip():
            try: rows.append((float(L[8:13]), float(L[92:103]) * (1 - float(L[70:79]))))
            except ValueError: pass
K = pd.DataFrame(rows, columns=["H", "q"])
res = {}; samples = {"All": ("gate_v0_hext30", "digest2_comparison_hext30/gate_v0_hext30", "raw_neo"), "Unk": ("gate_v0unk_hext30", "digest2_comparison_unk", "noid_neo")}
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.8), layout="constrained")
for ax, (lo, hi, lab) in zip(axs, ((4.5, 10, "$4.5\\leq q<10$ AU (Trojans, Centaurs)"), (10, 1e9, "$q\\geq10$ AU (TNOs)"))):
    kn = K[(K.q >= lo) & (K.q < hi)]; bins = np.arange(0, 26, 0.5)
    ax.hist(kn.H, bins=bins, density=True, histtype="stepfilled", color="#c9c6bf", label=f"known, MPCORB ({len(kn):,})")
    for nm, c in (("All", "#d4572a"), ("Unk", "#3f6fa5")):
        run, d2, sc = samples[nm]; D = load(run, d2, sc); fa = (D["s"] >= 65) & ~D["y"] & (D["q"] >= lo) & (D["q"] < hi)
        comp = cell_completeness(D["q"][fa], D["e"][fa], D["i"][fa], D["H"][fa])
        res[f"{nm} {lab}"] = dict(n=int(fa.sum()), H_pct=np.percentile(D["H"][fa], [10, 50, 90]).round(2).tolist(), V_pct=np.percentile(D["V"][fa], [10, 50, 90]).round(2).tolist(),
                                  rate_deg_day_pct=np.percentile(D["rate"][fa], [10, 50, 90]).round(4).tolist(), expected_known=round(float(np.nansum(comp)), 1),
                                  frac_known=round(float(np.nanmean(comp)), 3), known_H_pct=np.percentile(kn.H, [10, 50, 90]).round(2).tolist(), n_known=len(kn))
        ax.hist(D["H"][fa], bins=bins, density=True, histtype="step", lw=1.4, color=c, label=f"digest2 false alarms, {nm} sample ({int(fa.sum())})")
    ax.set(xlabel="$H$", title=lab); ax.set_xlim(0, 20 if hi < 1e8 else 14)
axs[0].set_ylabel("density"); axs[0].legend(frameon=False, fontsize=7)
for ext in ("png", "pdf"): fig.savefig(OUT / f"h_distribution.{ext}", dpi=200)
(OUT / "numbers.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
