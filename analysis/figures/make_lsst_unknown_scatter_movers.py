"""LSST alerts section: plain scatter of every tracklet with 3+ detections from the unknown-alert pull (rerun with nDiaSources <= 2;
moving, arc >= 15 min), digest2 raw against priora All, with the NEO candidates confirmed as real movers in the embargo images marked.
Left out: tracklets reviewed as variable stars, supernovae or artefacts (full_nd2/score/image_review.csv) and candidates flagged
static by the catalogue tests (back-to-back detections do not move, or a template source at 2+ detections; candidates_annot.csv).
-> lsst_unknown_scatter_movers.{pdf,png} (new file)"""
import os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
S = Path(os.environ["PRIORA_ALERTS"]) / "full_nd2/score"; OUT = Path(__file__).resolve().parent
m = pd.read_parquet(S / "scores_joined.parquet").rename(columns=lambda c: c.replace("ds42_", "priora_"))  # columns written before the rename
s = m[m.n >= 3]
RV = pd.read_csv(S / "image_review.csv").dropna(subset=["current_id"]).set_index("current_id").review
C = pd.read_csv(S / "candidates_annot.csv", index_col=0); static = C.index[C.static_flag.str.startswith("static")]
s = s[~s.index.isin(RV.index[RV == "artefact"]) & ~s.index.isin(static)]; mov = s.index.isin(RV.index[RV == "real mover"])
CPT, CSTAR, CINK = "#55607a", "#f2b01e", "#0b0b0b"
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 7.8, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
    "ytick.right": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
x = np.clip(s.priora_All.values, 1e-8, 1); y = s.d2_raw.values
fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
ax.scatter(x[~mov], y[~mov], s=5, color=CPT, alpha=0.45, lw=0, rasterized=True, label=f"tracklets with 3+ detections ({len(s):,})")
ax.scatter(x[mov], y[mov], s=110, marker="*", color=CSTAR, ec=CINK, lw=0.6, zorder=5, label=f"NEO candidates ({int(mov.sum())})")
ax.axhline(65, color=CINK, lw=0.8, ls="--"); ax.axvline(0.1, color=CINK, lw=0.8, ls=":")   # candidate cut, P = 0.1
ax.set(xscale="log", xlim=(1e-7, 1.6), ylim=(-2, 102), xlabel=r"priora $P(\mathrm{NEO})$", ylabel="digest2 raw NEO score")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False, handletextpad=0.3)
for ext in ("pdf", "png"): fig.savefig(OUT / f"lsst_unknown_scatter_movers.{ext}", dpi=300)
print("static removed:", int(s.index.isin(static).sum()) if False else len(static), "| movers flagged static:", len(set(static) & set(RV.index[RV == "real mover"])), "|", len(s), "tracklets;", int(mov.sum()), "movers:", list(s.index[mov]))
