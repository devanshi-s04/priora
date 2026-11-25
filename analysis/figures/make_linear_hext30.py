"""Linear-axis versions of the paper's Figures 1-3.

Same inputs as make_reliability_hext30.py (Fig 1) and make_d2cmp_hext30.py (Figs 2-4); no re-scoring.
  Fig 1  calibration_hext30_reliability_linear   pooled 2x10^5 reliability, ten bins of width 0.1, linear axes
  Fig 2  d2cmp_hext30_roc_linear                 ROC with a linear false-positive axis
  Fig 3  d2cmp_hext30_counts_linear              expected vs observed NEOs, ten bins of width 0.1, linear counts
  (Fig 4 with a P = 0.1 line is built by make_d2cmp_jacobian.py: d2cmp_hext30_scatter_p10.)
Bins are [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]; the last bin includes P = 1 exactly.
Run from paper/figures/:  python3 make_linear_hext30.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from validation.digest2_comparison import _roc, auc_bootstrap  # noqa: E402

OUT = Path(__file__).resolve().parent
import os as _os
CAL_RUNS = ["calibration_1e5_hext30", "calibration_1e5_hext30_seed2"]
CAL_RUNS = _os.environ["CAL_RUNS"].split(",") if _os.environ.get("CAL_RUNS") else CAL_RUNS   # e.g. the Unk sample
CAL_SFX = _os.environ.get("CAL_SFX", "")
RUN = REPO / "validation/output/gate_v0_hext30"
CMP = REPO / "validation/output/digest2_comparison_hext30/gate_v0_hext30"
# Unk 20k sample: R_RUN=gate_v0unk_hext30 R_CMP=digest2_comparison_unk D2_SCORE=noid OUT_SUFFIX=_unk20k
RUN = REPO / "validation/output" / _os.environ["R_RUN"] if _os.environ.get("R_RUN") else RUN
CMP = REPO / "validation/output" / _os.environ["R_CMP"] if _os.environ.get("R_CMP") else CMP
C42, CD2, CINK, CSUB, COTH, COUT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#9a9993", "#6b3fa0"
THRESH = 65.0
EDGES = np.round(np.arange(0, 1.01, 0.1), 1)

# digest2 score mode: D2_SCORE=raw (paper default) or D2_SCORE=noid. NoID outputs get a "_noid" filename suffix,
# so the raw-score figures are never overwritten.
import os as _os
D2_SCORE = _os.environ.get("D2_SCORE", "raw")
assert D2_SCORE in ("raw", "noid"), D2_SCORE
SFX = "" if D2_SCORE == "raw" else "_noid"
D2_LABEL = {"raw": "raw", "noid": "NoID"}[D2_SCORE]
SFX += _os.environ.get("OUT_SUFFIX", "")

mpl.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.8,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def save(fig, stem):
    stem = stem + SFX
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        fig.savefig(OUT / f"{stem}.{ext}", **kw)
    plt.close(fig)


def linbins(p, y):
    rows = []
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        m = (p >= lo) & ((p < hi) if i < len(EDGES) - 2 else (p <= hi))
        n, k = int(m.sum()), int(y[m].sum())
        exp = float(p[m].sum())
        rows.append(dict(lo=float(lo), hi=float(hi), n=n, obs=k, exp=exp, pbar=exp / n if n else np.nan,
                         sd=float(np.sqrt((p[m] * (1 - p[m])).sum())),
                         f_lo=float(beta.ppf(0.16, k + 0.5, n - k + 0.5)) if n else np.nan,
                         f_hi=float(beta.ppf(0.84, k + 0.5, n - k + 0.5)) if n else np.nan,
                         n_p1=int((m & (p == 1.0)).sum())))
    assert sum(r["n"] for r in rows) == p.size
    return rows


# --- Fig 1: pooled reliability, linear ------------------------------------------------------------------------
P, Y = [], []
for r in CAL_RUNS:
    d = REPO / "validation/output" / r
    with np.load(d / "scores.npz") as z, np.load(d / "gen.npz") as g:
        p = z["p_neo"]
        y = g["is_neo"][z["index"]].astype(bool)
        # CAL_EXCLUDE=1: drop tracklets whose true distance is beyond the 100 AU integration limit and the D-19
        # triage-policy rows (both Lambdas 0, P set to 1), as in the paper's calibration sample.
        drop = np.zeros(p.size, bool)
        if _os.environ.get("CAL_EXCLUDE") == "1":
            drop = ~z["truth_rho_in_range"].astype(bool) | ((z["lambda_neo"] == 0) & (z["lambda_other"] == 0))
    ok = np.isfinite(p) & ~drop
    P.append(p[ok])
    Y.append(y[ok])
pc, yc = np.concatenate(P), np.concatenate(Y)
RC = [r for r in linbins(pc, yc) if r["n"]]                    # small real-data samples can leave a bin empty

fig, (ax, ar) = plt.subplots(2, 1, figsize=(3.4, 4.3), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [2.3, 1]})
x = np.array([r["pbar"] for r in RC])
f = np.array([r["obs"] / r["n"] for r in RC])
flo = np.array([r["f_lo"] for r in RC])
fhi = np.array([r["f_hi"] for r in RC])
ax.plot([0, 1], [0, 1], color=CSUB, lw=0.9, ls=":", zorder=1, label="perfect calibration")
ax.errorbar(x, f, yerr=[np.clip(f - flo, 0, None), np.clip(fhi - f, 0, None)], fmt="o", ms=4, mfc="white", mec=C42, mew=1.1, color=C42,
            elinewidth=1.0, capsize=0, zorder=3, label="observed NEO fraction (68%)")
for xi, fi, r in zip(x, f, RC):                                 # beside the point, clear of its error bar
    right = xi < 0.95
    ax.text(xi + (0.022 if right else -0.022), fi, f"{r['n']:,}", ha="left" if right else "right", va="center",
            fontsize=6.2, color=CSUB)
for a in (ax, ar):
    a.set_xlim(0, 1)
    a.set_xticks(EDGES)
    a.set_xticks([], minor=True)
ax.set_ylim(-0.02, 1.02)
ax.set_yticks(EDGES)
ax.set_ylabel("observed NEO fraction")
ax.legend(loc="upper left", frameon=False, handletextpad=0.4, borderaxespad=0.3)
ax.text(0.97, 0.04, f"{pc.size:,} tracklets; numbers give tracklets per bin\n"
        f"last bin includes {RC[-1]['n_p1']:,} with $P=1$ exactly",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=CINK, linespacing=1.3)
ratio = np.array([r["obs"] / r["exp"] for r in RC])
rerr = np.array([r["sd"] / r["exp"] for r in RC])
ar.axhline(1, color=CSUB, lw=0.9, ls=":")
ar.errorbar(x, ratio, yerr=rerr, fmt="o", ms=4, mfc="white", mec=C42, mew=1.1, color=C42, elinewidth=1.0, capsize=0)
ar.set_ylim(0.4, 1.6)
ar.set_ylabel("observed / expected")
ar.set_xlabel(r"mean predicted $P(\mathrm{NEO})$ in bin")
ar.tick_params(axis="x", labelsize=7.5)
save(fig, "calibration_hext30_reliability_linear" + CAL_SFX)

# --- Figs 2-4 data ----------------------------------------------------------------------------------------------
with np.load(RUN / "scores.npz") as z, np.load(RUN / "gen.npz") as g, np.load(CMP / "digest2_scores.npz") as d:
    idx = z["index"]
    keep = np.isin(idx, d["index"])                      # the Unk digest2 cache omits priora's undefined tracklets
    assert np.array_equal(d["index"], idx[keep]), "digest2 cache does not match the run"
    idx = idx[keep]
    p42 = z["p_neo"][keep].astype(float)
    s2 = d[f"{D2_SCORE}_neo"].astype(float)
    ok2 = d["ok"].astype(bool)
    y = g["is_neo"][idx].astype(bool)
    el = g["elements"][idx]
q = el[:, 0] * (1 - el[:, 1])
good = np.isfinite(p42) & ok2 & np.isfinite(s2)
p42, s2, y, q = p42[good], s2[good], y[good], q[good]
pd2 = np.clip(s2 / 100.0, 0.0, 1.0)
w = np.ones_like(p42)
auc = auc_bootstrap(p42, s2, y, w, n_boot=2000, seed=7)        # same settings as make_d2cmp_hext30.py

# --- Fig 2: ROC, linear ------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.9), layout="constrained")
for sc, col, lbl, key in ((p42, C42, "priora $P(\\mathrm{NEO})$", "auc_a"), (s2, CD2, f"digest2 {D2_LABEL} score", "auc_b")):
    fpr, tpr = _roc(sc, y, w)
    ax.plot(fpr, tpr, color=col, lw=1.5, drawstyle="steps-post", label=f"{lbl}, AUC = {auc[key]:.4f}")
ax.plot([0, 1], [0, 1], color=CSUB, lw=0.8, ls=":")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xticks(EDGES[::2])
ax.set_xlabel("false-positive rate")
ax.set_ylabel("true-positive rate (completeness)")
ax.legend(loc="lower right", frameon=False)
ax.text(0.97, 0.30, f"$\\Delta$AUC = {auc['delta']:+.4f}\n[{auc['delta_ci'][0]:+.4f}, {auc['delta_ci'][1]:+.4f}]",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=CINK)
save(fig, "d2cmp_hext30_roc_linear")

# --- Fig 3: counts, linear bins and axis -------------------------------------------------------------------------
R42, RD2 = linbins(p42, y), linbins(pd2, y)
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), layout="constrained", sharey=True)
for ax, rows, col, name in ((axes[0], R42, C42, "priora"), (axes[1], RD2, CD2, "digest2 (score / 100)")):
    xc = np.array([(r["lo"] + r["hi"]) / 2 for r in rows])
    exp = np.array([r["exp"] for r in rows])
    obs = np.array([r["obs"] for r in rows])
    sd = np.array([r["sd"] for r in rows])
    ax.bar(xc, exp, width=0.066, color=col, alpha=0.30, lw=0, label="predicted ($\\sum P$)")
    ax.errorbar(xc, obs, yerr=sd, fmt="o", ms=3.6, color=col, elinewidth=1.0, capsize=0,
                label="observed ($\\pm1\\sigma$ if calibrated)")
    ax.set_xlim(0, 1)
    ax.set_xticks(EDGES)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("predicted $P(\\mathrm{NEO})$ bin")
    ax.set_title(f"{name}: predicted {exp.sum():.0f}, observed {obs.sum()}", fontsize=9)
ymax = max(max(r["exp"], r["obs"] + r["sd"]) for r in R42 + RD2) * 1.18
for ax, rows in ((axes[0], R42), (axes[1], RD2)):
    for r in rows:
        top = max(r["exp"], r["obs"] + r["sd"])
        ax.text((r["lo"] + r["hi"]) / 2, top + ymax * 0.012, f"{r['n']:,}", ha="center", va="bottom",
                fontsize=5.8, color=CSUB, rotation=90 if r["n"] >= 1000 else 0)
axes[0].set_ylim(0, ymax * 1.08)
axes[0].set_ylabel("NEOs in bin")
axes[0].legend(loc="upper left", frameon=False, handletextpad=0.4)
save(fig, "d2cmp_hext30_counts_linear")

# --- provenance -------------------------------------------------------------------------------------------------
sh = lambda *c: subprocess.run(c, cwd=REPO, capture_output=True, text=True).stdout.strip()
quad = {
    "d2_ge65_p_ge0.1": int(((s2 >= THRESH) & (p42 >= 0.1)).sum()),
    "d2_ge65_p_lt0.1": int(((s2 >= THRESH) & (p42 < 0.1)).sum()),
    "d2_lt65_p_ge0.1": int(((s2 < THRESH) & (p42 >= 0.1)).sum()),
    "neo_d2_ge65_p_lt0.1": int(((s2 >= THRESH) & (p42 < 0.1) & y).sum()),
    "neo_d2_lt65_p_ge0.1": int(((s2 < THRESH) & (p42 >= 0.1) & y).sum()),
}
prov = {"digest2_score": D2_SCORE,
    "script": Path(__file__).name, "commit": sh("git", "rev-parse", "--short", "HEAD"),
    "cal_runs": CAL_RUNS, "run": str(RUN.relative_to(REPO)), "digest2_cache": str(CMP.relative_to(REPO)),
    "model_sha256": hashlib.sha256((REPO / "data/digest2.model.csv").read_bytes()).hexdigest(),
    "fig1_bins": RC, "auc": {k: auc[k] for k in ("auc_a", "auc_b", "delta", "delta_ci")},
    "fig3_bins": {"priora": R42, "digest2": RD2}, "fig4_quadrants_at_p0.1": quad,
}
(OUT / f"linear_hext30_provenance{SFX}.json").write_text(json.dumps(prov, indent=1, default=float))
for name, rows in (("fig1", RC), ("priora", R42), ("digest2", RD2)):
    print(name, "  ".join(f"[{r['lo']:.1f},{r['hi']:.1f}] n={r['n']} {r['obs']}/{r['exp']:.1f}" for r in rows))
print(json.dumps({"auc": prov["auc"], "quad": quad}, indent=1, default=float))
