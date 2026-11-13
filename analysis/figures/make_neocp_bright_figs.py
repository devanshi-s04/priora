"""§8 figures for the NEOCP comparison (2026-11-06), from the bright-only Unk sample (V 14-21.2; validation/output/gate_v0unk_bright212_hext30,
digest2 NoID from validation/output/digest2_comparison_unk_bright212):
  neocp_bright_scatter  digest2 NoID vs ds42 Unk after removing known distant objects (q >= 4.5 AU and H brighter than the MPCORB
                        known-object H peak for its range: Trojan/Centaur 14.75, TNO 7.25); one column
  neocp_bright_hdist    H of the bright sample's distant digest2 >= 65 false alarms against known MPCORB objects, by q range, with the
                        peak used as the 'known' limit; full width
New files only. Run from the repo root."""
import sys, numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.argv = ["x"]; from pathlib import Path as _P; _REPO = _P(__file__).resolve().parents[2]; import os; os.chdir(_REPO)
exec(open("validation/d2_far_known/bright_neocp_match.py").read().split('res = {"known_peak_H": peaks}')[0])
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 7.6, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
    "ytick.right": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
F = str(_P(__file__).resolve().parent) + "/"   # reproduce copy: write next to this script
D = load("gate_v0unk_bright212_hext30", "digest2_comparison_unk_bright212", "noid_neo"); b = D["V"] < 21.2
far = D["q"] >= 4.5; keep = b & ~(D["known"] & far)
x = np.clip(D["p"][keep], 1e-8, 1); yv = D["s"][keep]; y = D["y"][keep]; fr = far[keep] & ~y
fig, ax = plt.subplots(figsize=(3.5, 3.1), layout="constrained")
ax.scatter(x[~y & ~fr], yv[~y & ~fr], s=5, color="#b9b6ae", lw=0, rasterized=True, label="non-NEO, $q<4.5$ AU")
ax.scatter(x[fr], yv[fr], s=10, color="#6b3fa0", lw=0, label="non-NEO, $q\\geq4.5$ AU")
ax.scatter(x[y], yv[y], s=14, color="#eb6834", lw=0, label="NEO")
ax.axhline(65, color="k", lw=0.8, ls="--"); ax.axvline(0.1, color="k", lw=0.8, ls=":")
ax.set(xscale="log", xlim=(1e-8, 1.6), ylim=(-2, 102), xlabel="ds42 $P(\\mathrm{NEO})$", ylabel="digest2 NoID NEO score")
fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False, markerscale=1.8, handletextpad=0.2, columnspacing=0.8)
for ext in ("pdf", "png"): fig.savefig(F + f"neocp_bright_scatter.{ext}", dpi=300)
plt.close(fig)
fa = b & far & ~D["y"] & (D["s"] >= 65)
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.7), layout="constrained")
for ax, (lo, hi, nm, lab, xl) in zip(axs, ((4.5, 10, "Trojan/Centaur", "(a) $4.5\\leq q<10$ AU (Trojans, Centaurs)", (6, 20)), (10, 1e9, "TNO", "(b) $q\\geq10$ AU (TNOs)", (0, 14)))):
    bins = np.arange(0, 26, 0.5); kn = Hs[(qs >= lo) & (qs < hi)]; m = fa & (D["q"] >= lo) & (D["q"] < hi)
    ax.hist(kn, bins=bins, density=True, histtype="stepfilled", color="#c9c6bf", label=f"known (MPCORB, {len(kn):,})")
    ax.hist(D["H"][m], bins=bins, density=True, histtype="step", lw=1.5, color="#6b3fa0", label=f"digest2 false alarms, $V<21.2$ ({int(m.sum())})")
    ax.axvline(peaks[nm], color="k", lw=0.9, ls="--"); ax.text(peaks[nm] + 0.2, ax.get_ylim()[1] * 0.92, f"peak $H={peaks[nm]:.2f}$", fontsize=7.5, va="top")
    ax.set(xlabel="$H$", xlim=xl, title=lab); ax.title.set_fontsize(9)
    print(nm, "false alarms", int(m.sum()), "brighter than peak", int((D["H"][m] < peaks[nm]).sum()), "H median", np.median(D["H"][m]).round(2))
axs[0].set_ylabel("density"); axs[0].legend(loc="upper left", frameon=False)
for ext in ("pdf", "png"): fig.savefig(F + f"neocp_bright_hdist.{ext}", dpi=300)
