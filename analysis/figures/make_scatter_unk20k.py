"""Single-panel digest2 vs priora scatter on the 20k undiscovered-population (Unk) sample.

Same style as panel (a) of make_d2cmp_jacobian.py (Figure 4), one panel, no Jacobian-off comparison.
Inputs (no re-scoring):
  validation/output/gate_v0unk_hext30/          priora scores + truth, Unk prior, h_extend_to = 30
  validation/output/digest2_comparison_unk/     digest2 scores on the same tracklets (NoID used)
Outputs (new filenames): d2cmp_scatter_unk20k.{pdf,png}, d2cmp_scatter_unk20k_provenance.json
Run from paper/figures/:  python3 make_scatter_unk20k.py
"""
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = REPO / "validation/output/gate_v0unk_hext30"
CMP = REPO / "validation/output/digest2_comparison_unk"
CD2, CINK, COTH, COUT = "#eb6834", "#0b0b0b", "#9a9993", "#6b3fa0"
THRESH, P10 = 65.0, 0.1

mpl.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.8,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

with np.load(RUN / "scores.npz") as z, np.load(RUN / "gen.npz") as g, np.load(CMP / "digest2_scores.npz") as d:
    idx = z["index"]
    keep = np.isin(idx, d["index"])                       # digest2 cache omits priora's undefined tracklets
    assert np.array_equal(d["index"], idx[keep]), "digest2 cache does not match the run"
    p42 = z["p_neo"][keep].astype(float)
    s2 = d["noid_neo"].astype(float); ok2 = d["ok"].astype(bool)
    y = g["is_neo"][idx[keep]].astype(bool); el = g["elements"][idx[keep]]
q = el[:, 0] * (1 - el[:, 1])
good = np.isfinite(p42) & ok2 & np.isfinite(s2)
p42, s2, y, q = p42[good], s2[good], y[good], q[good]
N = p42.size

fig, ax = plt.subplots(figsize=(3.4, 2.9), layout="constrained")
xs = np.clip(p42, 1e-8, 1.0)
inner, outer = ~y & (q < 4.5), ~y & (q >= 4.5)
ax.scatter(xs[inner], s2[inner], s=2, alpha=0.18, color=COTH, lw=0, rasterized=True, label="non-NEO, $q<4.5$ AU")
ax.scatter(xs[outer], s2[outer], s=5, alpha=0.75, color=COUT, lw=0, rasterized=True, label="non-NEO, $q\\geq4.5$ AU")
ax.scatter(xs[y], s2[y], s=9, alpha=0.95, color=CD2, lw=0, label="NEO")
ax.axhline(THRESH, color=CINK, lw=0.9, ls="--")
ax.axvline(P10, color=CINK, lw=0.9, ls=":")
ax.text(1.5e-8, THRESH + 1.5, "digest2 NEOCP threshold", fontsize=7.5, color=CINK, va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
ax.text(P10 / 1.35, 34, "$P=0.1$ (10%)", fontsize=7.5, color=CINK, ha="center", va="bottom", rotation=90)
ax.set(xscale="log", xlim=(1e-8, 1.6), ylim=(-2, 102), xlabel="priora $P(\\mathrm{NEO})$", ylabel="digest2 NoID NEO score")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, markerscale=2.2,
          handletextpad=0.1, columnspacing=0.8, fontsize=7.5)
for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
    fig.savefig(OUT / f"d2cmp_scatter_unk20k.{ext}", **kw)
plt.close(fig)

# numbers for the caption / text
fa = ~y & (s2 >= THRESH)
compl_d2 = float((s2[y] >= THRESH).mean())
t = float(np.quantile(p42[y], 1 - compl_d2))                     # priora threshold at digest2's completeness
fa42 = ~y & (p42 >= t)
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
prov = {
    "script": "make_scatter_unk20k.py", "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                                                capture_output=True, text=True).stdout.strip(),
    "run": str(RUN.relative_to(REPO)), "digest2_cache": str(CMP.relative_to(REPO)), "digest2_score": "noid",
    "model_sha256": sha(REPO / "data/digest2.model.csv"), "n": int(N), "n_neo": int(y.sum()),
    "n_clipped_below_1e-8": int((p42 < 1e-8).sum()),
    "digest2_ge65": dict(flagged=int((s2 >= THRESH).sum()), completeness=compl_d2, false_alarms=int(fa.sum()),
                         false_alarms_q_ge_4p5=int((fa & (q >= 4.5)).sum()),
                         false_alarms_with_P_ge_0p1=int((fa & (p42 >= P10)).sum())),
    "priora_at_same_completeness": dict(threshold=t, flagged=int((p42 >= t).sum()), completeness=float((p42[y] >= t).mean()),
                                      false_alarms=int(fa42.sum()), false_alarms_q_ge_4p5=int((fa42 & (q >= 4.5)).sum())),
    "priora_P_ge_0p5": dict(flagged=int((p42 >= .5).sum()), completeness=float((p42[y] >= .5).mean()),
                          false_alarms=int((~y & (p42 >= .5)).sum())),
}
(OUT / "d2cmp_scatter_unk20k_provenance.json").write_text(json.dumps(prov, indent=1))
print(json.dumps(prov, indent=1))
