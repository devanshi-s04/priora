"""Is the mid-range calibration dip on the undiscovered (Unk) sample real or chance? Four independent 200k Unk samples
(three seeds in 2026, one in 2027) plus the two full-population (All) 1e5 runs for reference. Same 0.1 bins as Fig 1.
Run from the repo root: python3 -m validation.unk_noid.seed_check  -> seed_check.json, figures/seed_check.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

REPO = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent; O = REPO / "validation/output"
RUNS = [("Unk · seed 20260902 · 2026", "gate_v0unk200k_hext30"), ("Unk · seed 20260903 · 2026", "gate_v0unk200k_s3_hext30"),
        ("Unk · seed 20260904 · 2026", "gate_v0unk200k_s4_hext30"), ("Unk · seed 20260905 · 2027", "gate_v0unk200k_s5y27_hext30")]
ALL = [("All · 1e5 run 1", "calibration_1e5_hext30"), ("All · 1e5 run 2", "calibration_1e5_hext30_seed2")]
E = np.linspace(0, 1, 11)
COL = ["#2a78d6", "#eb6834", "#2fa36b", "#8a4fc2"]


def load(d):
    with np.load(O / d / "scores.npz") as z, np.load(O / d / "gen.npz") as g:
        p = z["p_neo"]; y = g["is_neo"][z["index"]].astype(bool)
    ok = np.isfinite(p)
    return p[ok], y[ok], int((~ok).sum())


def stats(p, y):
    rows = []
    for i, (lo, hi) in enumerate(zip(E[:-1], E[1:])):
        m = (p >= lo) & ((p < hi) if i < 9 else (p <= hi)); n, k = int(m.sum()), int(y[m].sum()); e = float(p[m].sum())
        rows.append(dict(lo=float(lo), n=n, obs=k, exp=e, pbar=e / n if n else np.nan, sd=float(np.sqrt((p[m] * (1 - p[m])).sum())),
                         flo=float(beta.ppf(0.16, k + .5, n - k + .5)) if n else np.nan, fhi=float(beta.ppf(0.84, k + .5, n - k + .5)) if n else np.nan))
    mid = (p >= 0.2) & (p < 0.7); e = p[mid].sum(); sd = np.sqrt((p[mid] * (1 - p[mid])).sum())
    return dict(bins=rows, n=int(p.size), n_neo=int(y.sum()), pred_obs=float(p.sum() / y.sum()),
                mid=dict(n=int(mid.sum()), obs=int(y[mid].sum()), exp=float(e), z=float((y[mid].sum() - e) / sd)),
                nonneo_p1=int(((~y) & (p == 1)).sum()))


res = {}
for lab, d in RUNS + ALL:
    if not (O / d / "scores.npz").exists():
        continue
    p, y, nan = load(d); res[lab] = stats(p, y) | {"run": d, "undefined": nan}
unk = [l for l, _ in RUNS if l in res]
P = np.concatenate([load(d)[0] for l, d in RUNS if l in res]); Y = np.concatenate([load(d)[1] for l, d in RUNS if l in res])
res["Unk · pooled"] = stats(P, Y) | {"run": "pooled"}
(OUT / "seed_check.json").write_text(json.dumps(res, indent=1, default=float))

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10, "xtick.direction": "in", "ytick.direction": "in"})
fig, (a, r) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}, layout="constrained")
a.plot([0, 1], [0, 1], ":", color="0.35"); r.axhline(1, ls=":", color="0.35")
for j, lab in enumerate(unk + ["Unk · pooled"]):
    B = res[lab]["bins"]; off = (j - 2) * 0.008
    x = np.array([b["pbar"] for b in B]) + off; f = np.array([b["obs"] / b["n"] if b["n"] else np.nan for b in B])
    lo = f - np.array([b["flo"] for b in B]); hi = np.array([b["fhi"] for b in B]) - f
    pooled = lab.endswith("pooled")
    kw = dict(fmt="s" if pooled else "o", ms=6 if pooled else 4, mfc="black" if pooled else "white",
              mec="black" if pooled else COL[j], color="black" if pooled else COL[j], elinewidth=1.4 if pooled else 1, capsize=0)
    a.errorbar(x, f, yerr=[lo, hi], label=f"{lab}  (mid z = {res[lab]['mid']['z']:+.1f})", **kw)
    ratio = np.array([b["obs"] / b["exp"] if b["exp"] > 0 else np.nan for b in B]); re = np.array([b["sd"] / b["exp"] if b["exp"] > 0 else np.nan for b in B])
    r.errorbar(x, ratio, yerr=re, **kw)
a.set(xlim=(0, 1), ylim=(-0.02, 1.02), ylabel="observed NEO fraction"); a.legend(fontsize=8, frameon=False, loc="upper left")
r.set(ylim=(0.3, 1.7), xlabel="mean predicted P(NEO) in bin", ylabel="observed / expected")
a.set_xticks(E); r.set_xticks(E)
(OUT / "figures").mkdir(exist_ok=True); fig.savefig(OUT / "figures/seed_check.png", dpi=200); plt.close(fig)
for lab, v in res.items():
    print(f"{lab:30s} n {v['n']:7d} NEO {v['n_neo']:5d} pred/obs {v['pred_obs']:.3f} mid obs {v['mid']['obs']:3d} exp {v['mid']['exp']:6.1f} z {v['mid']['z']:+.2f} nonNEO@P=1 {v['nonneo_p1']}")
