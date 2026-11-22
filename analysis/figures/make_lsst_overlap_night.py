"""Appendix D: where Rubin gets three detections. One real night (61227) of the known-object alert transfer.
(a) The night's visits (footprints approximated as circles of radius 1.75 deg about the image centres that HelioLinC
reconstructs from each image's detections), the sky shaded by how many visits cover it, visit outlines coloured by
time in the night, and the known-object tracklets: two detections dim, three or more bright, NEOs ringed.
(b) Share of tracklets with 3+ detections against the number of visits covering their position.
(c) Close-up of one overlap patch, tracklets as streaks of their motion.
Sample: pure known-object tracklets, arcs >= 15 min, with a perihelion in MPCORB (as make_lsst_known_flow.py).
Inputs ($DS42_ALERTS): score/scores_joined.parquet, tracklets/{tracklet_labels_v5.parquet, pairdets_v5.csv, trk2det_v5.csv, imgs_v5.txt}.
-> rubin_overlap_night.{png,pdf}"""
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

T = Path(os.environ["DS42_ALERTS"])
OUT = Path(__file__).resolve().parent
NIGHT, RAD = 61227, 1.75
FR = dict(ra=(352, 298), dec=(-33, -1))
ZW = dict(ra=(316, 311), dec=(-20.5, -16.5))
img = pd.read_csv(T / "tracklets/imgs_v5.txt", sep=r"\s+", header=None)
I = img[np.floor(img[0] - 0.5).astype(int) == NIGHT].copy(); I["h"] = (I[0] - I[0].min()) * 24
S = pd.read_parquet(T / "score/scores_joined.parquet")
S = S[S.pure & (S.arc >= 15) & S.q.notna()].copy()
g = pd.read_parquet(T / "tracklets/tracklet_labels_v5.parquet"); keep = g[g.arc >= 15]
trk_id = keep.index.values[S.row.values]                       # score/build_psv.py: row = position in keep
pdets = pd.read_csv(T / "tracklets/pairdets_v5.csv"); pdets.columns = [c.lstrip("#") for c in pdets.columns]
t2d = pd.read_csv(T / "tracklets/trk2det_v5.csv"); t2d.columns = [c.lstrip("#") for c in t2d.columns]
t2d = t2d[t2d.trk_ID.isin(trk_id)].join(pdets[["MJD", "RA", "Dec"]], on="detnum").sort_values("MJD")
gg = t2d.groupby("trk_ID"); f, l = gg.first().loc[trk_id], gg.last().loc[trk_id]
assert np.allclose(f.MJD.values, S.t0.values, atol=1e-6) and np.allclose(l.MJD.values, S.t1.values, atol=1e-6)
S["RA0"], S["Dec0"], S["RA1"], S["Dec1"] = f.RA.values, f.Dec.values, l.RA.values, l.Dec.values
S["MJD0"], S["MJD1"] = f.MJD.values, l.MJD.values
S = S[S.night == NIGHT].copy()
gx = np.linspace(I[1].min() - 3, I[1].max() + 3, 1400); gy = np.linspace(I[2].min() - 3, I[2].max() + 3, 900)
GX, GY = np.meshgrid(gx, gy); cov = np.zeros_like(GX, int)
for _, r in I.iterrows():
    cov += (((GX - r[1]) * np.cos(np.radians(r[2]))) ** 2 + (GY - r[2]) ** 2) < RAD ** 2
ii = np.clip(np.searchsorted(gy, S.Dec0.values), 0, gy.size - 1); jj = np.clip(np.searchsorted(gx, S.RA0.values), 0, gx.size - 1)
S["nvis"] = cov[ii, jj]; n3 = (S.n >= 3).values
BG, INK, DIM = "#ffffff", "#111111", "#555555"
covcmap = LinearSegmentedColormap.from_list("c", ["#f4f6fa", "#dde5f1", "#b7cbe6", "#86a9d6", "#5585c2", "#2e5ea8"], N=6)
covnorm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 99], 6)
tcmap = mpl.colors.ListedColormap(mpl.colormaps["plasma"](np.linspace(0, 0.85, 256))); th = np.linspace(0, 2 * np.pi, 160)
C2, C3, CN = "#9aa0ae", "#d9480f", "#000000"


def sky(ax, lw_outline, s2, s3, ring, streak=None):
    ax.set_facecolor(BG)
    m = ax.pcolormesh(gx, gy, np.ma.masked_equal(cov, 0), cmap=covcmap, norm=covnorm, shading="auto", rasterized=True)
    for _, r in I.iterrows():
        ax.plot(r[1] + RAD * np.cos(th) / np.cos(np.radians(r[2])), r[2] + RAD * np.sin(th), color=tcmap(0.1 + 0.85 * r.h / I.h.max()),
                lw=lw_outline, alpha=0.7)
    if streak:
        for sel, c, w in ((~n3, C2, 0.5), (n3, C3, 1.1)):
            dt = np.clip((S.MJD1 - S.MJD0)[sel], 1e-4, None)                 # true RA/Dec motion of each tracklet
            vra = (((S.RA1 - S.RA0)[sel] + 180) % 360 - 180) / dt; vde = (S.Dec1 - S.Dec0)[sel] / dt
            seg = np.stack([np.c_[S.RA0[sel], S.Dec0[sel]], np.c_[S.RA0[sel] + vra * streak, S.Dec0[sel] + vde * streak]], 1)
            ax.add_collection(LineCollection(seg, colors=c, linewidths=w, alpha=0.85 if c == C3 else 0.6))
    else:
        ax.scatter(S.RA0[~n3], S.Dec0[~n3], s=s2, color=C2, alpha=0.55, lw=0, rasterized=True)
        ax.scatter(S.RA0[n3], S.Dec0[n3], s=s3, color=C3, alpha=0.9, lw=0, rasterized=True)
    nn = S[S.neo]
    ax.scatter(nn.RA0, nn.Dec0, s=ring, facecolor="none", edgecolor=CN, lw=1.0, zorder=6)
    return m


rc = {"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "text.color": INK, "axes.labelcolor": INK,
      "xtick.color": DIM, "ytick.color": DIM, "axes.edgecolor": "#333333", "xtick.direction": "in", "ytick.direction": "in"}
with mpl.rc_context(rc):
    fig = plt.figure(figsize=(7.2, 7.0), facecolor=BG, layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1), width_ratios=(0.9, 1.1))
    ax = fig.add_subplot(gs[0, :])
    m = sky(ax, 0.45, 0.7, 2.4, 30)
    ax.add_patch(plt.Rectangle((ZW["ra"][1], ZW["dec"][0]), ZW["ra"][0] - ZW["ra"][1], ZW["dec"][1] - ZW["dec"][0], fill=False, ec=INK, lw=0.8, ls="--"))
    ax.set(xlim=FR["ra"], ylim=FR["dec"]); ax.set_aspect(1 / np.cos(np.radians(-17)), adjustable="datalim")
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
    ax.set_title(f"(a) one Rubin night, 2026 July 6: {len(I)} visits, {len(S):,} tracklets of known asteroids", fontsize=9.5, loc="left")
    h = [Line2D([], [], marker="o", ls="none", ms=2.5, color=C2, label="2 detections"), Line2D([], [], marker="o", ls="none", ms=3.5, color=C3, label="3+ detections"),
         Line2D([], [], marker="o", ls="none", ms=7, mfc="none", mec=CN, label="NEO"),
         Line2D([], [], color=tcmap(0.12), lw=1.2, label="visit, early"), Line2D([], [], color=tcmap(0.92), lw=1.2, label="visit, late")]
    lg = ax.legend(handles=h, loc="upper right", fontsize=7.3, frameon=True, framealpha=0.8, facecolor=BG, edgecolor="none", ncol=1)
    for t in lg.get_texts(): t.set_color(INK)
    cb = fig.colorbar(m, ax=ax, location="right", shrink=0.85, pad=0.01, ticks=[1, 2, 3, 4, 5, 6])
    cb.ax.set_yticklabels(["1", "2", "3", "4", "5", "6+"]); cb.set_label("visits covering the sky"); cb.outline.set_visible(False)
    cb.ax.tick_params(colors=DIM); cb.ax.yaxis.label.set_color(INK)

    bx = fig.add_subplot(gs[1, 0]); bx.set_facecolor(BG)
    ks = np.arange(1, 7); nv = np.minimum(S.nvis.values, 6)
    fr = np.array([n3[nv == k].mean() for k in ks]); cnt = np.array([(nv == k).sum() for k in ks])
    bx.bar(ks, fr, width=0.72, color=[covcmap(covnorm(k)) for k in ks], edgecolor="#2e5ea8", lw=0.6)
    for k, f, c in zip(ks, fr, cnt):
        bx.text(k, f + 0.025, f"{100 * f:.0f}%", ha="center", fontsize=8)
        bx.text(k, -0.075, f"{c:,}", ha="center", fontsize=6.5, color=DIM)
    bx.set(ylim=(-0.11, 0.72), xticks=ks, xticklabels=["1", "2", "3", "4", "5", "6+"], yticks=[0, .2, .4, .6])
    bx.set_yticklabels(["0%", "20%", "40%", "60%"])
    bx.set_xlabel("visits covering the tracklet  (grey: tracklets)"); bx.set_ylabel("share with 3+ detections")
    bx.set_title("(b) overlaps make 3-detection tracklets", fontsize=9.5, loc="left")
    for sp in ("top", "right"): bx.spines[sp].set_visible(False)

    cx = fig.add_subplot(gs[1, 1])
    sky(cx, 1.1, 3, 8, 70, streak=1.5)
    cx.set(xlim=ZW["ra"], ylim=ZW["dec"]); cx.set_aspect(1 / np.cos(np.radians(-18.5)))
    w = S.RA0.between(ZW["ra"][1], ZW["ra"][0]) & S.Dec0.between(*ZW["dec"])
    cx.set_title(f"(c) close-up: {int(w.sum()):,} tracklets, streaks = 1.5 days of motion", fontsize=9.5, loc="left")
    cx.set_xlabel("RA [deg]")
    fig.savefig(OUT / "rubin_overlap_night.png", dpi=250, facecolor=BG); fig.savefig(OUT / "rubin_overlap_night.pdf", facecolor=BG)
print("visits", len(I), "tracklets", len(S), "3+", int(n3.sum()))
for k, f, c in zip(ks, fr, cnt): print(k, c, round(f, 3))
print("3+ tracklets in >=3-visit sky:", round(n3[nv >= 3].sum() / n3.sum(), 3), "| sky with >=3 visits:", round(np.mean(cov[cov > 0] >= 3), 3))
