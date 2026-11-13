"""Subset of one alerts test restricted to tracklets with >= NMIN detections: numbers + the same figures as the main sections.
Usage: python3 subset_figs.py <test> <nmin>    test in {known, unk, full}
 known: $DS42_ALERTS/score/scores_joined.parquet, evaluation set = pure tracklets, page $DS42_ALERTS/page
 unk:   $DS42_ALERTS/unk/score/scores_joined.parquet, evaluation set = labelled tracklets, page unk/page
 full:  $DS42_ALERTS/full/score/scores_joined.parquet, labelled tracklets, page full/page
Writes <score dir>/analysis_<n>plus.json, figures into <page>/img with tag <Tag><n>plus, paper-script caches/figures
(validation/output/lsst_<test>_{All,Unk}_<n>plus, paper/figures/*_<Tag>{All,Unk}<n>plus.*; new filenames only)."""
import json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

H = Path.home(); T = Path(os.environ["DS42_ALERTS"]); PY = sys.executable
test, nmin = sys.argv[1], int(sys.argv[2])
cfg = {"known": (T / "score", T / "page", "lsst"), "unk": (T / "unk/score", T / "unk/page", "lsstUnkLab"),
       "full": (T / "full/score", T / "full/page", "lsstFullLab")}[test]
SD, PG, TAG = cfg
m = pd.read_parquet(SD / "scores_joined.parquet")
if test == "known":
    E, other = m[m.pure].copy(), m[~m.pure]
else:
    E, other = m[m.status == "labelled"].copy(), m[m.status != "labelled"]
E["neo"] = E.neo.astype(bool); E = E[E.n >= nmin]; other = other[other.n >= nmin]
y = E.neo.values; rng = np.random.default_rng(20261103 + nmin)


def auc_bs(s, B=2000):
    s = np.asarray(s); a = roc_auc_score(y, s); i1, i0 = np.where(y)[0], np.where(~y)[0]
    bs = [roc_auc_score(np.r_[np.ones(len(i1)), np.zeros(len(i0))], np.r_[s[rng.choice(i1, len(i1))], s[rng.choice(i0, len(i0))]]) for _ in range(B)]
    return [float(a), float(np.std(bs))]


out = {"test": test, "nmin": nmin, "n": len(E), "n_neo": int(y.sum()), "n_other": len(other),
       "npts": {str(k): int(v) for k, v in E.n.value_counts().sort_index().items()}}
out["auc"] = {k: auc_bs(E[k]) for k in ("ds42_All", "ds42_Unk", "d2_raw", "d2_noid")} if 0 < y.sum() < len(y) else {}
out["sumP"] = {k: float(E[k].sum()) for k in ("ds42_All", "ds42_Unk")}
fl = {}
for name, sel in (("d2_raw>=65", E.d2_raw >= 65), ("d2_noid>=65", E.d2_noid >= 65), ("ds42_All>=0.1", E.ds42_All >= 0.1),
                  ("ds42_Unk>=0.1", E.ds42_Unk >= 0.1), ("ds42_All>=0.5", E.ds42_All >= 0.5)):
    f = E[sel]; nn = f[~f.neo]
    fl[name] = dict(flagged=len(f), neo=int(f.neo.sum()), non_neo=len(nn), distant=int((nn.q >= 4.5).sum()))
out["flags"] = fl
out["other_flagged"] = {c: int((other[c] >= (65 if c.startswith("d2") else .5)).sum()) for c in ("d2_raw", "d2_noid", "ds42_All", "ds42_Unk")}
# paper-script caches
O = Path(os.environ["DS42_REPO"]) / "validation/output"; n = len(E); idx = np.arange(n); el = np.zeros((n, 6)); el[:, 0] = E.q.values
runs = {}
for V in ("All", "Unk"):
    r = f"lsst_{test}_{V}_{nmin}plus"; d = O / r; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "scores.npz", index=idx, p_neo=E[f"ds42_{V}"].values); np.savez(d / "gen.npz", is_neo=E.neo.values, elements=el)
    c = O / f"lsst_{test}_d2_{nmin}plus" / r; c.mkdir(parents=True, exist_ok=True)
    np.savez(c / "digest2_scores.npz", index=idx, raw_neo=E.d2_raw.values, noid_neo=E.d2_noid.values, ok=np.ones(n, bool))
    runs[V] = r
FIG = Path(os.environ["DS42_REPO"]) / "paper/figures"; img = PG / "img"; img.mkdir(exist_ok=True)
sfx = {V: f"_{TAG}{V}{nmin}plus" for V in ("All", "Unk")}
for V, d2 in (("All", "raw"), ("Unk", "noid")):
    env = dict(os.environ, R_RUN=runs[V], R_CMP=f"lsst_{test}_d2_{nmin}plus/{runs[V]}", D2_SCORE=d2, OUT_SUFFIX=sfx[V],
               CAL_RUNS=runs[V], CAL_SFX=sfx[V])
    for s in ("make_d2cmp_hext30.py", "make_linear_hext30.py"):
        r = subprocess.run([PY, s], cwd=FIG, env=env, capture_output=True, text=True)
        if r.returncode: print(s, V, "FAILED", r.stderr[-400:])
    prov = FIG / f"d2cmp_hext30_provenance{'' if d2 == 'raw' else '_noid'}{sfx[V]}.json"
    if prov.exists():
        p = json.loads(prov.read_text()); out[f"dAUC_{V}"] = [p["auc"]["delta"], p["auc"]["delta_ci"]]; out[f"d2_over_100_{V}"] = p["predicted"]["digest2_over_100"]
names = {"cal_all": f"calibration_hext30_reliability_linear{sfx['All']}{sfx['All']}.png",
         "cal_unk": f"calibration_hext30_reliability_linear{sfx['Unk']}_noid{sfx['Unk']}.png",
         "counts": f"d2cmp_hext30_counts{sfx['All']}.png", "rel": f"d2cmp_hext30_reliability{sfx['All']}.png", "roc": f"d2cmp_hext30_roc{sfx['All']}.png"}
for k, f in list(names.items()):
    if (FIG / f).exists(): shutil.copy(FIG / f, img / f)
    else: print("missing", f); names.pop(k)
E2 = E.copy(); E2["pure"] = True; src = SD / f"scores_for_figs_{nmin}plus.parquet"; E2.to_parquet(src)
st = f"{TAG}{nmin}plus"
subprocess.run([PY, "fig_scatter_axes.py"], cwd=T / "score", env=dict(os.environ, SRC=str(src), OUTDIR=str(img), TAG=st), check=True)
names |= {f"scatter_{mo}": f"scatter_{mo}_{st}_p10.png" for mo in ("logx", "loglog", "lin100")}
out["figures"] = names
(SD / f"analysis_{nmin}plus.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k != "figures"}, indent=1, default=float))
