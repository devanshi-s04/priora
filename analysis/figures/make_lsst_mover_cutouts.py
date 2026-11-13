"""LSST alerts section: alert cutouts of the nine NEO candidates confirmed as real movers in the image review
(full_nd2/score/image_review.csv). Public ALeRCE stamps (science | template | difference, 30x30 px) at each tracklet's
highest-SNR detection, from $DS42_ALERTS/cutouts/alerce_nd2/<diaSourceId>.fits. Arrow on the science stamp: direction
of motion (first to last detection, via the stamp WCS); label: rate. North/East compass in the first cell.
-> lsst_mover_cutouts.{pdf,png} (new file)"""
import os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
T = Path(os.environ["DS42_ALERTS"]); OUT = Path(__file__).resolve().parent
RV = pd.read_csv(T / "full_nd2/score/image_review.csv"); RV = RV[RV.review == "real mover"].set_index("current_id")
H = pd.read_csv(T / "handoff/lsst_neo_candidates_nd2_detections.csv", dtype={"diaSourceId": str})
S = pd.read_parquet(T / "full_nd2/score/scores_joined.parquet")
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
order = S.loc[RV.index, "ds42_All"].sort_values(ascending=False).index
fig = plt.figure(figsize=(7.1, 3.75)); outer = fig.add_gridspec(3, 3, wspace=0.08, hspace=0.42, left=0.01, right=0.99, top=0.95, bottom=0.06)
for k, t in enumerate(order):
    g = H[H.tracklet_id == t].sort_values("obs_mjd_tai")
    have = g[[ (T / f"cutouts/alerce_nd2/{d}.fits").exists() for d in g.diaSourceId ]]
    r = have.sort_values("snr", ascending=False).iloc[0]
    hdul = fits.open(T / f"cutouts/alerce_nd2/{r.diaSourceId}.fits")
    w = WCS(hdul[1].header); x0, y0 = w.world_to_pixel_values(g.ra.iloc[0], g.dec.iloc[0]); x1, y1 = w.world_to_pixel_values(g.ra.iloc[-1], g.dec.iloc[-1])
    v = np.array([x1 - x0, y1 - y0]); v = v / np.hypot(*v)
    rate = np.hypot((g.ra.iloc[-1] - g.ra.iloc[0]) * np.cos(np.radians(g.dec.iloc[0])), g.dec.iloc[-1] - g.dec.iloc[0]) / ((g.obs_mjd_tai.iloc[-1] - g.obs_mjd_tai.iloc[0]))
    inner = outer[k].subgridspec(1, 3, wspace=0.04)
    for j, (ext, lab) in enumerate(((1, "science"), (2, "template"), (3, "difference"))):
        ax = fig.add_subplot(inner[j]); a = hdul[ext].data.astype(float)
        ax.imshow(a, origin="lower", cmap="gray", norm=ImageNormalize(a[np.isfinite(a)], interval=ZScaleInterval(), stretch=AsinhStretch(0.5)))
        ax.set_xticks([]); ax.set_yticks([])
        if k < 3: ax.set_title(lab, fontsize=7.5, pad=2)
        if j == 0:
            xs, ys = w.world_to_pixel_values(r.ra, r.dec); c = np.array([xs, ys]); n = a.shape[0]
            perp = np.array([-v[1], v[0]]); mid = c + perp * 0.30 * n   # arrow beside the source, along the motion
            ax.annotate("", xy=mid + v * 0.22 * n, xytext=mid - v * 0.22 * n,
                        arrowprops=dict(arrowstyle="-|>", color="#ffcc33", lw=1.6, mutation_scale=9, shrinkA=0, shrinkB=0))
        if j == 0 and k == 0:   # compass from the WCS: N and E unit vectors in pixels
            base = np.array([0.18, 0.18]) * a.shape[0]; ra, de = w.pixel_to_world_values(*base)
            for dd, name in (((0, 1 / 3600), "N"), ((1 / 3600 / np.cos(np.radians(de)), 0), "E")):
                px = np.array(w.world_to_pixel_values(ra + dd[0], de + dd[1])) - base; px = px / np.hypot(*px) * 0.22 * a.shape[0]
                ax.annotate("", xy=base + px, xytext=base, arrowprops=dict(arrowstyle="-", color="white", lw=1.0))
                ax.text(*(base + px * 1.3), name, color="white", fontsize=6.5, ha="center", va="center")
    first = RV.loc[t, "first_pull_id"]
    fig.text(0.5 * (fig.axes[-3].get_position().x0 + fig.axes[-1].get_position().x1), fig.axes[-1].get_position().y0 - 0.008,
             f"{first}   $P$ = {S.loc[t, 'ds42_All']:.2f}   d2 = {S.loc[t, 'd2_raw']:.0f}   {rate:.2f}$^\\circ$/d   {r.band}",
             ha="center", va="top", fontsize=7.0)
for ext in ("pdf", "png"): fig.savefig(OUT / f"lsst_mover_cutouts.{ext}", dpi=300)
print([(RV.loc[t, "first_pull_id"], t) for t in order])
