"""Figure for Mario's test, final version: bright-only Unk sample (V < 21.2, digest2 NoID). (a) all; (b) known objects removed
for q >= 4.5 AU only (brighter than the MPCORB known-object H peak: Trojan/Centaur 14.75, TNO 7.25). The near-boundary objects are
kept because the same rule calls 55% of the real (all newly discovered) NEOCP NEOs 'known' (bright_neocp_match.py notes).
-> bright_far_known.{png,pdf}"""
import sys, numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.argv = ["x"]; exec(open("validation/d2_far_known/bright_neocp_match.py").read().split('res = {"known_peak_H": peaks}')[0])
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 7.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "savefig.bbox": "tight", "pdf.fonttype": 42})
D = load("gate_v0unk_bright212_hext30", "digest2_comparison_unk_bright212", "noid_neo"); b = D["V"] < 21.2; k = b & ~(D["known"] & (D["q"] >= 4.5))
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.1), layout="constrained", sharey=True)
for ax, (sel, ttl) in zip(axs, ((b, "(a) $V<21.2$"), (k, "(b) $V<21.2$, known distant objects removed"))):
    x = np.clip(D["p"][sel], 1e-8, 1); yv = D["s"][sel]; y = D["y"][sel]; far = (D["q"][sel] >= 4.5) & ~y
    ax.scatter(x[~y & ~far], yv[~y & ~far], s=6, color="#b9b6ae", lw=0, label="non-NEO, $q<4.5$ AU"); ax.scatter(x[far], yv[far], s=10, color="#6b3fa0", lw=0, label="non-NEO, $q\\geq4.5$ AU")
    ax.scatter(x[y], yv[y], s=14, color="#eb6834", lw=0, label="NEO"); ax.axhline(65, color="k", lw=0.8, ls="--"); ax.axvline(0.1, color="k", lw=0.8, ls=":")
    r = ops(D["y"][sel], D["q"][sel], D["p"][sel], D["s"][sel])
    ax.set(xscale="log", xlim=(1e-8, 1.6), ylim=(-2, 102), xlabel="ds42 $P(\\mathrm{NEO})$"); ax.set_title(f"{ttl}\n{r['n']:,} tracklets, {r['neos']} NEOs", fontsize=9)
    ax.text(0.03, 0.97, f"digest2$\\geq$65: {100*r['d2_complete']:.1f}% complete, {100*r['d2_contam']:.1f}% contamination", transform=ax.transAxes, va="top", fontsize=7.5, bbox=dict(fc="white", ec="none", alpha=0.85))
axs[1].text(0.03, 0.88, "NEOCP (real): 98.4% complete, 31.8% contamination", transform=axs[1].transAxes, va="top", fontsize=7.5, color="#2a78d6", bbox=dict(fc="white", ec="none", alpha=0.85))
axs[0].set_ylabel("digest2 NoID NEO score"); fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False, markerscale=1.8)
for ext in ("png", "pdf"): fig.savefig(f"validation/d2_far_known/bright_far_known.{ext}", dpi=250)
print("ok")
