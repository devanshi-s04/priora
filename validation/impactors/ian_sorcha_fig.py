"""Timeline figure for the Chow et al. (2026) Sorcha impactor test: one row per impactor, one marker per Rubin tracklet at
its time before impact; filled = ds42 Unk P >= 0.5, open = P < 0.5; a cross marks digest2 NoID >= 65. New file only."""
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
T = pd.read_csv("validation/impactors/ian_sorcha_scores.csv")
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in", "savefig.bbox": "tight", "pdf.fonttype": 42})
objs = T.groupby("obj").dbi.min().sort_values().index[::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.4), layout="constrained")
for k, o in enumerate(objs):
    t = T[T.obj == o]; ax.plot(-t.dbi, [k] * len(t), color="#d4d2cc", lw=0.8, zorder=1)
    hi = t.pU >= 0.5
    ax.scatter(-t.dbi[hi], [k] * hi.sum(), s=34, color="#2a78d6", zorder=3)
    ax.scatter(-t.dbi[~hi], [k] * (~hi).sum(), s=34, facecolor="white", edgecolor="#2a78d6", lw=1.1, zorder=3)
    d2 = t.noid >= 65; ax.scatter(-t.dbi[d2], [k] * d2.sum(), s=12, marker="x", color="#eb6834", lw=1.0, zorder=4)
ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlim(100, 0.3)
ax.set_yticks(range(len(objs)), [o[:4] + "-" + o[4:6] + "-" + o[6:8] for o in objs], fontsize=7.5)
ax.set_xlabel("days before impact"); ax.set_ylabel("impactor (impact date)")
h = [plt.Line2D([], [], marker="o", ls="", color="#2a78d6", label="ds42 $P\\geq0.5$"),
     plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#2a78d6", label="ds42 $P<0.5$"),
     plt.Line2D([], [], marker="x", ls="", color="#eb6834", label="digest2 NoID $\\geq65$")]
ax.legend(handles=h, loc="lower left", frameon=False, fontsize=7.5)
for ext in ("png", "pdf"): fig.savefig(f"validation/impactors/page/impactor_timeline.{ext}", dpi=300)
