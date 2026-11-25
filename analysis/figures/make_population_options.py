"""Options for showing All vs Unk in the digest2 population model (data/digest2.model.csv, un-scaled counts).
New files only: popopt_{a,b,c,d,e}.{pdf,png}. Run from paper/figures."""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # reproduce copy: outputs next to this script
from pathlib import Path
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "src"))
from priora.population import read_digest2_model, Q_EDGES, H_EDGES
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.8, "axes.linewidth": 0.8, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42})
T = read_digest2_model(str(REPO / "data/digest2.model.csv"))
A, U = T[("All", "SS")], T[("Unk", "SS")]
qhi = Q_EDGES[1:]; w = np.diff(H_EDGES); hc = 0.5 * (H_EDGES[1:] + H_EDGES[:-1])
GROUPS = [("NEO", qhi <= 1.3 + 1e-9, "#2a78d6"), ("Mars-crosser", (qhi > 1.3 + 1e-9) & (qhi <= 1.67 + 1e-9), "#7a5cc4"),
          ("main belt", (qhi > 1.67 + 1e-9) & (qhi <= 4.5 + 1e-9), "#eb6834"), ("Trojan/Centaur", (qhi > 4.5 + 1e-9) & (qhi <= 10 + 1e-9), "#3d7a4f"),
          ("TNO", qhi > 10 + 1e-9, "#8a6a2a")]
print("q edges", Q_EDGES)
def hist(X, m): return X[m].sum(axis=(0, 1, 2))
def save(fig, s):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})): fig.savefig(f"popopt_{s}.{ext}", **kw)
    plt.close(fig)
# (a) known fraction vs H, one line per class
fig, ax = plt.subplots(figsize=(3.5, 2.8), layout="constrained")
for name, m, c in GROUPS:
    a, u = hist(A, m), hist(U, m); ok = a >= 5
    ax.plot(hc[ok], 1 - u[ok] / a[ok], "o-", color=c, ms=3, lw=1.3, label=name)
ax.set(xlabel="absolute magnitude $H$", ylabel="fraction already known\n(All $-$ Unk) / All", xlim=(0, 25.5), ylim=(0, 1))
ax.legend(frameon=False, loc="upper left", fontsize=7.2); save(fig, "a")
# (b) All and Unk as lines, NEO and non-NEO in one panel
fig, ax = plt.subplots(figsize=(3.5, 2.8), layout="constrained")
nn = GROUPS[0][1]
for m, c, nm in ((nn, "#2a78d6", "NEO"), (~nn, "#eb6834", "non-NEO")):
    ax.stairs(hist(A, m) / w, H_EDGES, color=c, lw=1.4, label=f"{nm}, All")
    ax.stairs(hist(U, m) / w, H_EDGES, color=c, lw=1.4, ls="--", label=f"{nm}, Unk")
ax.set(yscale="log", xlim=(5, 25.5), ylim=(0.5, 6e6), xlabel="absolute magnitude $H$", ylabel="objects per magnitude")
ax.legend(frameon=False, loc="upper left", fontsize=7.2); save(fig, "b")
# (c) 100% stacked bars: known vs undiscovered per H bin, NEO and non-NEO
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), layout="constrained", sharey=True)
for ax, (m, c, nm) in zip(axes, ((nn, "#2a78d6", "NEOs"), (~nn, "#eb6834", "non-NEOs"))):
    a, u = hist(A, m), hist(U, m); ok = a >= 5; k = np.where(ok, (a - u) / np.where(ok, a, 1), np.nan)
    ax.bar(hc[ok], k[ok], width=w[ok] * 0.9, color="#9a9993", label="known")
    ax.bar(hc[ok], 1 - k[ok], bottom=k[ok], width=w[ok] * 0.9, color="#4a8fb8", alpha=0.85, label="undiscovered (Unk)")
    ax.set(xlim=(0, 25.5), ylim=(0, 1), xlabel="absolute magnitude $H$", title=nm)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
for ax in axes: pass
axes[0].set_ylabel("share of All"); save(fig, "c")
# (d) q-H map of the known fraction
a2, u2 = A.sum(axis=(1, 2)), U.sum(axis=(1, 2))
with np.errstate(invalid="ignore", divide="ignore"): kf = np.where(a2 >= 1, 1 - u2 / a2, np.nan)
fig, ax = plt.subplots(figsize=(3.6, 3.0), layout="constrained")
qe = np.r_[0.25, Q_EDGES[1:]] ; pc = ax.pcolormesh(H_EDGES, qe, kf, cmap="viridis", vmin=0, vmax=1, shading="flat")
ax.set(yscale="log", xlabel="absolute magnitude $H$", ylabel="perihelion $q$ (AU)", ylim=(0.25, 100), xlim=(0, 25.5))
for qq in (1.3, 4.5, 10): ax.axhline(qq, color="w", lw=0.6, ls=":")
fig.colorbar(pc, ax=ax, label="fraction already known"); save(fig, "d")
# (e) totals per class, All vs Unk, horizontal bars (log)
fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
y = np.arange(len(GROUPS))[::-1]
for yi, (name, m, c) in zip(y, GROUPS):
    a, u = A[m].sum(), U[m].sum()
    ax.barh(yi + 0.18, a, height=0.34, color="#9a9993"); ax.barh(yi - 0.18, u, height=0.34, color=c)
    ax.text(a * 1.15, yi + 0.18, f"{a:,.0f}", va="center", fontsize=7); ax.text(u * 1.15, yi - 0.18, f"{u:,.0f} ({100*(1-u/a):.0f}% known)", va="center", fontsize=7)
    print(name, a, u, 1 - u / a)
ax.set(xscale="log", xlim=(1e4, 3e8), yticks=y, yticklabels=[g[0] for g in GROUPS], xlabel="objects in model (gray All, colour Unk)")
save(fig, "e")
