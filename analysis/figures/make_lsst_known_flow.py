"""LSST alerts section (Sec. 6.2.2): known objects in the Rubin alert stream as main-belt flow and flagged tracklets.
Sample: the pure known-object tracklets with arcs >= 15 min and a perihelion in MPCORB (64 937 of the paper's 64 938).
Grey arrows: median motion of tracklets with pneo All P(NEO) < 0.1 per 2 x 2 deg cell and night (cells with >= 20).
Coloured segments: tracklets with P >= 0.1, drawn as 4 days of motion, coloured by P; circles: true NEOs.
Coordinates: longitude from opposition (lambda - lambda_sun - 180) and ecliptic latitude, at the first detection.
Inputs ($DS42_ALERTS): score/scores_joined.parquet, tracklets/{tracklet_labels_v5.parquet, pairdets_v5.csv, trk2det_v5.csv}.
-> rubin_known_flow.{pdf,png} (single column, region 22-62 deg from opposition)"""
import os
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord, get_sun
from astropy.time import Time
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm

T = Path(os.environ["DS42_ALERTS"]); OUT = Path(__file__).resolve().parent
S = pd.read_parquet(T / "score/scores_joined.parquet")
S = S[S.pure & (S.arc >= 15) & S.q.notna()].copy()
g = pd.read_parquet(T / "tracklets/tracklet_labels_v5.parquet"); keep = g[g.arc >= 15]
trk_id = keep.index.values[S.row.values]                       # score/build_psv.py: row = position in keep
pdets = pd.read_csv(T / "tracklets/pairdets_v5.csv"); pdets.columns = [c.lstrip("#") for c in pdets.columns]
t2d = pd.read_csv(T / "tracklets/trk2det_v5.csv"); t2d.columns = [c.lstrip("#") for c in t2d.columns]
t2d = t2d[t2d.trk_ID.isin(trk_id)].join(pdets[["MJD", "RA", "Dec"]], on="detnum").sort_values("MJD")
gg = t2d.groupby("trk_ID"); f, l = gg.first().loc[trk_id], gg.last().loc[trk_id]
assert np.allclose(f.MJD.values, S.t0.values, atol=1e-6) and np.allclose(l.MJD.values, S.t1.values, atol=1e-6)
fr = GeocentricTrueEcliptic(equinox=Time("J2000"))
c0 = SkyCoord(f.RA.values * u.deg, f.Dec.values * u.deg).transform_to(fr)
c1 = SkyCoord(l.RA.values * u.deg, l.Dec.values * u.deg).transform_to(fr)
sun = get_sun(Time(f.MJD.values, format="mjd", scale="tai")).transform_to(fr)
dt = np.clip(l.MJD.values - f.MJD.values, 1e-4, None)
S["bet"] = c0.lat.deg
S["theta"] = ((c0.lon.deg - sun.lon.deg) % 360) - 180
S["vl"] = (((c1.lon.deg - c0.lon.deg + 180) % 360) - 180) * np.cos(np.radians(c0.lat.deg)) / dt
S["vb"] = (c1.lat.deg - c0.lat.deg) / dt

S["cx"] = np.floor(S.theta / 2).astype(int); S["cy"] = np.floor(S.bet / 2).astype(int)
fl = S[S.ds42_All < 0.1].groupby(["night", "cx", "cy"]).agg(fl=("vl", "median"), fb=("vb", "median"), nn=("vl", "size")).reset_index()
fl = fl[fl.nn >= 20]; fl["x"] = fl.cx * 2 + 1; fl["y"] = fl.cy * 2 + 1
F = S[S.ds42_All >= 0.1].sort_values("ds42_All")
CM = LinearSegmentedColormap.from_list("o", ["#f4a261", "#e6550d", "#a63603", "#4a1486"]); NORM = LogNorm(0.1, 1)
XR, YR = (62, 22), (-12, 8)
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.7, "xtick.direction": "in",
                     "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.4, 2.32), layout="constrained")
ax.quiver(fl.x, fl.y, fl.fl, fl.fb, color="#9097a6", angles="xy", scale_units="xy", scale=0.15, width=0.004, headwidth=3, headlength=3.5)
seg = np.stack([np.c_[F.theta, F.bet], np.c_[F.theta + F.vl * 4, F.bet + F.vb * 4]], 1)
ax.add_collection(LineCollection(seg, colors=CM(NORM(F.ds42_All.values)), linewidths=1.3, capstyle="round", zorder=4))
ax.scatter(S.theta[S.neo], S.bet[S.neo], s=9, facecolor="none", edgecolor="k", lw=0.5, zorder=5)
ax.set(xlim=XR, ylim=YR, aspect="equal", xlabel="longitude from opposition [deg]", ylabel="ecliptic latitude [deg]")
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=NORM, cmap=CM), ax=ax, location="top", shrink=0.7, aspect=30, pad=0.02, ticks=[0.1, 0.2, 0.5, 1])
cb.ax.set_xticklabels(["0.1", "0.2", "0.5", "1"]); cb.ax.minorticks_off(); cb.set_label(r"pneo $P(\mathrm{NEO})$", fontsize=8)
fig.savefig(OUT / "rubin_known_flow.png", dpi=300); fig.savefig(OUT / "rubin_known_flow.pdf"); plt.close(fig)
w = S.theta.between(XR[1], XR[0]) & S.bet.between(*YR)
print(f"tracklets {len(S)}, flagged {len(F)} ({int(F.neo.sum())} NEOs), NEOs {int(S.neo.sum())}; "
      f"in region: {int(w.sum())} tracklets, {int((w & (S.ds42_All >= .1)).sum())} flagged, {int((w & S.neo).sum())} NEOs")
