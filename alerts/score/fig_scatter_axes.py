"""Score scatter variants for the test page, so both axes use the same kind of scale:
  scatter_loglog_lsstAll:  ds42 P(NEO) and digest2 score, both log (digest2 0.05-100)
  scatter_lin100_lsstAll:  ds42 100 x P(NEO) and digest2 score, both linear 0-100
  scatter_logx_lsstAll:    ds42 log, digest2 linear (the paper's Figure 4 axes)
All get a dashed line at ds42 P = 0.1 (10%); files end in _p10 so the earlier versions stay.
Each has digest2 raw (a) and NoID (b) panels. Pure tracklets, ds42 All prior. Style and colours from make_d2cmp_hext30.py."""
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
C42, CD2, CINK, CSUB, COTH, COUT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#9a9993", "#6b3fa0"
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.8, "axes.linewidth": 0.8, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
import os
SRC = os.environ.get("SRC", "scores_joined.parquet"); OUTDIR = os.environ.get("OUTDIR", "../page/img"); TAG = os.environ.get("TAG", "lsstAll")
P = pd.read_parquet(SRC); P = P[P.pure]
y = P.neo.values; q = P.q.values; p = P.ds42_All.values
inner, outer = ~y & (q < 4.5), ~y & (q >= 4.5)
for mode in ("loglog", "lin100", "logx"):
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.1), layout="constrained", sharey=True)
    for ax, (col, lab, L) in zip(axs, (("d2_raw", "raw", "a"), ("d2_noid", "NoID", "b"))):
        s = P[col].values; x = 100 * p if mode == "lin100" else p
        ax.scatter(x[inner], s[inner], s=2, alpha=0.18, color=COTH, lw=0, rasterized=True, label="non-NEO, $q<4.5$ AU")
        ax.scatter(x[outer], s[outer], s=5, alpha=0.75, color=COUT, lw=0, rasterized=True, label="non-NEO, $q\\geq4.5$ AU")
        ax.scatter(x[y], s[y], s=9, alpha=0.95, color=CD2, lw=0, label="NEO")
        ax.axhline(65, color=CINK, lw=0.9, ls="--")
        if mode == "loglog":
            ax.set(xscale="log", yscale="log", xlim=(1e-7, 1.6), ylim=(0.05, 130), xlabel="ds42 $P(\\mathrm{NEO})$")
            ax.set_yticks([0.1, 1, 10, 65, 100], ["0.1", "1", "10", "65", "100"])
        elif mode == "logx":
            ax.set(xscale="log", xlim=(1e-7, 1.6), ylim=(-2, 102), xlabel="ds42 $P(\\mathrm{NEO})$")
        else:
            ax.set(xlim=(-2, 102), ylim=(-2, 102), xlabel="ds42 $100\\times P(\\mathrm{NEO})$")
            ax.axvline(50, color=CSUB, lw=0.8, ls=":")
        x10 = 0.1 if mode != "lin100" else 10.0
        ax.axvline(x10, color=CINK, lw=0.9, ls="--")
        ax.text(x10 / 1.3 if mode != "lin100" else x10 - 1.2, 0.5, "$P=0.1$ (10%)", transform=ax.get_xaxis_transform(),
                fontsize=7.5, color=CINK, ha="center", va="center", rotation=90,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
        nf, nn = int((s >= 65).sum()), int(((s >= 65) & ~y).sum())
        ax.set_title(f"({L}) digest2 {lab}: {nf:,} flagged at $\\geq$65, {nn:,} non-NEO", fontsize=9.5)
    axs[0].set_ylabel("digest2 NEO score")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False, markerscale=2.2, fontsize=7.5)
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(f"{OUTDIR}/scatter_{mode}_{TAG}_p10.{ext}", **kw)
    plt.close(fig)
print("ok")
