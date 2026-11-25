"""priora vs digest2 on the matched Unk comparison (validation/unk_noid run M: sample Unk, priora Unk, digest2 NoID; 19,999
tracklets, 287 NEOs), drawn so the low-false-positive regime is visible:
  d2cmp_unk_roc_logfpr   ROC with a log false-positive-rate axis (the follow-up regime)
  d2cmp_unk_pr           precision (purity) against recall (completeness)
Operating points marked: digest2 NoID >= 65 (NEOCP threshold), priora at matched completeness, priora P >= 0.5.
New files only. Run from paper/figures."""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # reproduce copy: outputs next to this script
from pathlib import Path
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from validation.unk_noid.analyze import load
D = load("gate_v0unk_hext30", "digest2_comparison_unk"); y, p, s = D["y"], D["p"], D["noid"]
C42, CD2, CINK = "#2a78d6", "#eb6834", "#0b0b0b"
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 7.8, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
    "ytick.right": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
def curve(sc):
    o = np.argsort(-sc, kind="stable"); ys = y[o]; tp = np.cumsum(ys); fp = np.cumsum(~ys)
    last = np.r_[np.diff(sc[o]) != 0, True]           # one point per distinct threshold
    return tp[last] / y.sum(), fp[last] / (~y).sum(), tp[last] / (tp[last] + fp[last])
def point(sel): tp = (sel & y).sum(); fp = (sel & ~y).sum(); return tp / y.sum(), fp / (~y).sum(), tp / (tp + fp), int(tp), int(fp)
d2pt = point(s >= 65); cut = np.sort(p[y])[::-1][d2pt[3] - 1]; dpt = point(p >= cut); hpt = point(p >= 0.5)
T42, F42, P42 = curve(p); TD2, FD2, PD2 = curve(s)
# ROC, log FPR
fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
ax.plot(np.maximum(F42, 1e-5), T42, color=C42, lw=1.5, label="priora $P$(NEO)"); ax.plot(np.maximum(FD2, 1e-5), TD2, color=CD2, lw=1.5, label="digest2 NoID score")
ax.plot(d2pt[1], d2pt[0], "o", color=CD2, ms=6, mec=CINK, mew=0.6, zorder=5); ax.plot(dpt[1], dpt[0], "o", color=C42, ms=6, mec=CINK, mew=0.6, zorder=5)
ax.annotate(f"digest2 $\\geq$65: {d2pt[4]} non-NEOs", (d2pt[1], d2pt[0]), xytext=(1.6e-3, 0.62), fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.6, color=CINK))
ax.annotate(f"priora, same completeness: {dpt[4]} non-NEOs", (dpt[1], dpt[0]), xytext=(1.2e-4, 0.78), fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.6, color=CINK))
ax.set(xscale="log", xlim=(1e-4, 1), ylim=(0.5, 1.005), xlabel="false-positive rate", ylabel="true-positive rate (completeness)")
ax.legend(loc="lower right", frameon=False)
for ext in ("pdf", "png"): fig.savefig(f"d2cmp_unk_roc_logfpr.{ext}", dpi=300)
plt.close(fig)
# precision-recall
fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
ax.plot(T42, P42, color=C42, lw=1.5, label="priora $P$(NEO)"); ax.plot(TD2, PD2, color=CD2, lw=1.5, label="digest2 NoID score")
ax.plot(d2pt[0], d2pt[2], "o", color=CD2, ms=6, mec=CINK, mew=0.6, zorder=5); ax.plot(dpt[0], dpt[2], "o", color=C42, ms=6, mec=CINK, mew=0.6, zorder=5)
ax.plot(hpt[0], hpt[2], "s", color=C42, ms=5, mec=CINK, mew=0.6, zorder=5)
ax.annotate(f"digest2 $\\geq$65\n{100 * d2pt[2]:.1f}% pure", (d2pt[0], d2pt[2]), xytext=(0.78, 0.30), fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.6, color=CINK))
ax.annotate(f"priora, same completeness\n{100 * dpt[2]:.1f}% pure", (dpt[0], dpt[2]), xytext=(0.77, 0.74), fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.6, color=CINK))
ax.annotate(f"priora $P\\geq0.5$: {100 * hpt[2]:.1f}%", (hpt[0], hpt[2]), xytext=(0.80, 1.025), fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.6, color=CINK))
ax.set(xlim=(0.75, 1.005), ylim=(0, 1.08), xlabel="completeness (recall)", ylabel="purity (precision)")
ax.legend(loc="lower left", frameon=False)
for ext in ("pdf", "png"): fig.savefig(f"d2cmp_unk_pr.{ext}", dpi=300)
print("d2>=65", d2pt, "| priora matched cut", round(float(cut), 3), dpt, "| P>=0.5", hpt)
for f in (1e-3, 1e-2):
    print(f"FPR {f}: TPR priora {T42[F42 <= f].max():.3f} digest2 {TD2[FD2 <= f].max():.3f}")
