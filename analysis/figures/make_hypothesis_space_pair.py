"""Methods figure: hypothesis-space posteriors for the example NEO and main-belt asteroid, side by side.

Style of option I on the figure-options page (rho on the y-axis, white background, cells below 1e-4 of the
peak faded, orange truth star, dashed q = 1.3 AU curve), as two panels with one shared colour scale.
The objects, grid and per-node bookkeeping come from ds42_example_figures.ipynb itself: its first six code
cells are executed here (they select gate objects by a printed rule, re-observe them from X05 and assert
the Lambda / P(NEO) / mass bookkeeping), so this figure cannot drift from the notebook.
Run from paper/figures/ with the Rubin-stack python:  python3 make_hypothesis_space_pair.py
Writes ds42_hypothesis_space_pair.{pdf,png} and ds42_hypothesis_space_pair_provenance.json.
"""
import json
import os
from pathlib import Path

import nbformat
import numpy as np

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
nb = nbformat.read("ds42_example_figures.ipynb", 4)
G = {}
for src in [c.source for c in nb.cells if c.cell_type == "code"][:6]:
    src = "\n".join(l for l in src.splitlines() if not l.startswith("%config"))
    src = src.replace('save(fig, f"ds42_{lab.lower()}_hypothesis_space"); plt.show()', "plt.close(fig)")
    exec(src, G)

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator

DATA, LIM, CMAP, C_NONE, q13 = G["DATA"], G["LIM"], G["CMAP"], G["C_NONE"], G["q13_branches"]
C_NEGL, C_STAR = "#f1efe9", "#FF8A1F"
LETTERS = os.environ.get("PANEL_LETTERS", "a,b").split(",")     # e.g. PANEL_LETTERS=a,b for a composite figure
STEM = os.environ.get("OUT_STEM", "ds42_hypothesis_space_pair")  # change it so the paper figure is not overwritten
NAME = {"NEO": f"({LETTERS[0]}) NEO", "MBA": f"({LETTERS[1]}) main-belt asteroid"}
FS = 9
CROP = {"MBA": 0.1}   # panel (b) shows rho >= 0.1 AU only, cropped in rho_dot to the cells holding probability
if os.environ.get("NO_CROP") == "1":                     # NO_CROP=1: full rho range in both panels (the paper version)
    CROP = {}
CROPPED = {}
Q13 = dict(color="white", lw=1.0, ls=(0, (5, 3)),
           path_effects=[pe.Stroke(linewidth=2.1, foreground="black"), pe.Normal()])

rc = {"font.size": FS, "axes.labelsize": FS + 0.5, "axes.titlesize": FS + 0.5, "xtick.labelsize": FS - 0.5,
      "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 0.5, "savefig.facecolor": "white"}
with mpl.rc_context(rc):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.4), layout="constrained")
    norm = LogNorm(*LIM["dens"])
    floor = LIM["dens"][0]
    for ax, lab in zip(axes, ("NEO", "MBA")):
        D = DATA[lab]
        y0, y1 = D["rho_dot"] - D["d_rhodot"] / 2, D["rho_dot"] + D["d_rhodot"] / 2
        verts = np.stack([np.c_[D["lo_e"], y0], np.c_[D["hi_e"], y0],
                          np.c_[D["hi_e"], y1], np.c_[D["lo_e"], y1]], 1)[..., ::-1].copy()   # (rho_dot, rho)
        dens = D["dens"]
        pos, show = dens > 0, dens >= floor
        negl = pos & ~show
        ax.add_collection(PolyCollection(verts[~pos], facecolors=C_NONE, edgecolors=C_NONE, lw=0.3, rasterized=True))
        ax.add_collection(PolyCollection(verts[negl], facecolors=C_NEGL, edgecolors=C_NEGL, lw=0.3, rasterized=True))
        pc = PolyCollection(verts[show], array=dens[show], cmap=CMAP, norm=norm, edgecolors="face", lw=0.3,
                            rasterized=True)
        ax.add_collection(pc)
        for seg in q13(D):
            ax.plot(seg[:, 1], seg[:, 0], zorder=4, **Q13)
        ax.scatter([D["truth"]["rho_dot"]], [D["truth"]["rho"]], marker="*", s=95, c=C_STAR, lw=0, zorder=6)
        ax.set_yscale("log")
        if lab in CROP:   # zoom to the cells that hold probability, so both panels show their posterior at a similar size
            keep = show & (D["lo_e"] >= CROP[lab])
            pad = 0.06 * (y1[keep].max() - y0[keep].min())
            ax.set_ylim(CROP[lab], LIM["rho"][1])
            ax.set_xlim(y0[keep].min() - pad, y1[keep].max() + pad)
            CROPPED[lab] = {"rho_min": CROP[lab], "rhodot": [float(y0[keep].min() - pad), float(y1[keep].max() + pad)],
                            "mass_outside_view": float(D["mass"][~((D["lo_e"] >= CROP[lab]) & (y0 >= y0[keep].min() - pad)
                                                                  & (y1 <= y1[keep].max() + pad))].sum())}
        else:
            ax.set_ylim(*LIM["rho"])
            pad = 0.05 * (y1.max() - y0.min())
            ax.set_xlim(y0.min() - pad, y1.max() + pad)
        ax.set_xlabel(r"range-rate $\dot\rho$ (AU day$^{-1}$)")
        ax.set_title(f"{NAME[lab]}, $P(\\mathrm{{NEO}})={D['res'].p_neo:.3f}$", pad=5)
    axes[1].set_ylabel(r"topocentric distance $\rho$ (AU)")
    axes[0].set_ylabel(r"topocentric distance $\rho$ (AU)")

    cb = fig.colorbar(pc, ax=axes, pad=0.015, fraction=0.035, aspect=30, extend="min", extendfrac=0.03,
                      extendrect=True)
    cb.set_label(r"posterior density $dP/d\ln\rho\,d\dot\rho$ (day AU$^{-1}$)", labelpad=5)
    cb.ax.yaxis.set_major_locator(LogLocator(numticks=12))
    cb.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    cb.ax.minorticks_off()
    cb.ax.tick_params(direction="out", length=3)

    h = [mpl.lines.Line2D([], [], marker="*", ms=10, mfc=C_STAR, mec="none", ls="none", label=r"true $(\rho,\dot\rho)$"),
         mpl.lines.Line2D([], [], label="$q=1.3$ AU", **Q13),
         mpl.patches.Patch(facecolor=C_NEGL, edgecolor="0.6", lw=0.5, label=r"$<10^{-4}$ of peak density"),
         mpl.patches.Patch(facecolor=C_NONE, edgecolor="0.55", lw=0.5, label="no catalogued objects")]
    fig.legend(handles=h, loc="outside lower center", ncol=4, frameon=False, handlelength=1.6,
               columnspacing=1.2, handletextpad=0.4)
    for ext, kw in (("pdf", {"dpi": 600}), ("png", {"dpi": 400})):
        fig.savefig(HERE / f"{STEM}.{ext}", **kw)
    plt.close(fig)

prov = {"script": Path(__file__).name, "notebook": "ds42_example_figures.ipynb", "commit": G["COMMIT"],
        "dirty_src": G["DIRTY"], "model_sha256": G["MODEL_SHA"], "obscode": G["OBSCODE"],
        "grid": [G["N_RHO"], G["N_RHODOT"]], "rho_range": [G["RHO_MIN"], G["RHO_MAX"]],
        "h_extend_to": G["H_EXTEND_TO"], "colour_floor_rel": G["DYN"], "dens_range": list(map(float, LIM["dens"]))}
prov["crop"] = CROPPED
for lab, D in DATA.items():
    t = D["truth"]
    prov[lab] = {"gate_index": int(D["k"]), "p_neo": float(D["res"].p_neo),
                 "n_nodes": int(D["dens"].size), "n_occupied": int((D["dens"] > 0).sum()),
                 "mass_within_x2_of_true_rho": float(D["within2"]),
                 **{k: float(v) for k, v in t.items()}}
(HERE / f"{STEM}_provenance.json").write_text(json.dumps(prov, indent=1))
print(json.dumps(prov, indent=1))
