"""Fig 4 variant: predicted vs observed NEO counts for priora and digest2 (NoID / 100) on the Unk 20k sample
(gate_v0unk_hext30 + digest2_comparison_unk, run M), showing only the bins with P >= 0.1 and no per-bin tracklet counts.
Same bins, errors and styling as make_d2cmp_hext30.py. New files only: d2cmp_unk_counts_p01.{pdf,png}. Run from paper/figures."""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # reproduce copy: outputs next to this script
from pathlib import Path
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[2]
RUN, CMP = REPO / "validation/output/gate_v0unk_hext30", REPO / "validation/output/digest2_comparison_unk"
C42, CD2 = "#2a78d6", "#eb6834"
EDGES = [0.1, 0.5, 0.9, 0.99, 1.0]
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.8, "axes.linewidth": 0.8, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42})
with np.load(RUN / "scores.npz") as z, np.load(RUN / "gen.npz") as g, np.load(CMP / "digest2_scores.npz") as d:
    idx = z["index"]; keep = np.isin(idx, d["index"]); assert np.array_equal(d["index"], idx[keep]); idx = idx[keep]
    p42 = z["p_neo"][keep].astype(float); s2 = d["noid_neo"].astype(float); ok2 = d["ok"].astype(bool)
    y = g["is_neo"][idx].astype(bool)
good = np.isfinite(p42) & ok2 & np.isfinite(s2)
p42, s2, y = p42[good], s2[good], y[good]
pd2 = np.clip(s2 / 100.0, 0.0, 1.0)
print("n", p42.size, "NEOs", y.sum(), "| all bins: priora pred", round(p42.sum(), 1), "d2 pred", round(pd2.sum(), 1))
def rows(p):
    ms = [(p >= lo) & (p < hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])] + [p == 1.0]
    return [(float(p[m].sum()), int(y[m].sum()), float(np.sqrt((p[m] * (1 - p[m])).sum())), int(m.sum())) for m in ms]
LAB = ["0.1–0.5", "0.5–0.9", "0.9–0.99", "0.99–<1", "$=1$"]
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), layout="constrained", sharey=True)
for ax, p, col, name in ((axes[0], p42, C42, "priora"), (axes[1], pd2, CD2, "digest2 (score / 100)")):
    R = rows(p); x = np.arange(len(R)); exp = np.array([r[0] for r in R]); obs = np.array([r[1] for r in R]); sd = np.array([r[2] for r in R])
    ax.bar(x, exp, width=0.66, color=col, alpha=0.30, lw=0, label="predicted ($\\sum P$)")
    ax.errorbar(x, obs, yerr=sd, fmt="o", ms=3.6, color=col, elinewidth=1.0, capsize=0, label="observed ($\\pm1\\sigma$ if calibrated)")
    ax.set_yscale("symlog", linthresh=1); ax.set_ylim(0, 4000)
    ax.set_yticks([0, 1, 10, 100, 1000], ["0", "1", "10", "100", "1000"])
    ax.set_xticks(x, LAB, rotation=35, ha="right", fontsize=7.5); ax.set_xlabel("predicted $P(\\mathrm{NEO})$ bin")
    ax.set_title(f"{name}, $P\\geq0.1$: predicted {exp.sum():.0f}, observed {obs.sum()}", fontsize=9)
    print(name, [(LAB[i], round(r[0], 1), r[1], r[3]) for i, r in enumerate(R)], "sum", round(exp.sum(), 1), obs.sum())
axes[0].set_ylabel("NEOs in bin"); axes[0].legend(loc="upper left", frameon=False, handletextpad=0.4)
for ext, kw in (("pdf", {}), ("png", {"dpi": 400})): fig.savefig(f"d2cmp_unk_counts_p01.{ext}", **kw)
