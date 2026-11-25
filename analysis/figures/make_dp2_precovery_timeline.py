"""DP2 section, small figure: known NEOs that priora scored highly among the DP2 unknown candidates (local MPCORB check) and that
Rubin never reported from those detections. One row per object: MPC observations by night (other stations; Rubin X05 submissions
in blue), our DP2 tracklets (stars, with priora P), and the DP2 nights we searched (shaded). MPC histories from the MPC get-obs API
(public designations only), saved in $PRIORA_ALERTS/dp2/precovery_fig/. -> dp2_precovery_timeline.{pdf,png} (new file)"""
import os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.dates as mdates
mpl.use("Agg"); import matplotlib.pyplot as plt
from astropy.time import Time
H = Path(os.environ["PRIORA_ALERTS"]) / "dp2"; OUT = Path(__file__).resolve().parent
V = pd.read_parquet(H / "run/score/candidate_verdict_final.parquet").rename(columns=lambda c: c.replace("ds42_", "priora_"))  # columns written before the rename
V = V[V.verdict == "known NEO"]
T = pd.read_parquet(H / "run/score/unknown_3plus_dets.parquet"); first = T.groupby("trkSub").MJD.min()
V["date"] = pd.to_datetime(Time(first.reindex(V.index).values, format="mjd", scale="tai").utc.isot)
objs = V.groupby("mpc_object").date.min().sort_values().index.tolist()
CO, CX, CS, CINK = "#8a8780", "#2a78d6", "#d4572a", "#0b0b0b"
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 7.6, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
X0, X1 = pd.Timestamp("2025-06-26"), pd.Timestamp("2025-08-06")
fig, ax = plt.subplots(figsize=(7.1, 2.9), layout="constrained")
ax.axvspan(pd.Timestamp("2025-07-03"), pd.Timestamp("2025-07-12"), color="#f1ede4", zorder=0)
ax.text(pd.Timestamp("2025-07-07 12:00"), len(objs) - 0.42, "DP2 nights searched", ha="center", va="center", fontsize=7.6, color="#6b675f")
rows = []
for i, o in enumerate(objs):
    y = len(objs) - 1 - i
    d = pd.read_csv(H / f"precovery_fig/mpc_obs_{o.replace(' ', '_')}.csv"); t = pd.to_datetime(d.obstime, format="ISO8601").dt.tz_localize(None)
    nights = d.assign(t=t.dt.floor("D")).groupby(["t", d.stn == "X05"]).size().reset_index()
    for (_, r) in nights.iterrows():
        if X0 <= r.t <= X1: ax.plot(r.t + pd.Timedelta(hours=12), y, "|", ms=11, mew=1.6, color=CX if r.stn else CO, zorder=2)
    before = int((t < X0).sum()); after = int((t > X1).sum())
    if before: ax.annotate(f"← {before} obs since {t.min():%Y}", (X0, y + 0.28), fontsize=7, color=CO, va="center")
    if after: ax.annotate(f"{after} more →", (X1, y + 0.28), fontsize=7, color=CO, va="center", ha="right")
    v = V[V.mpc_object == o]
    ax.scatter(v.date, [y] * len(v), marker="*", s=120, color=CS, ec=CINK, lw=0.5, zorder=4)
    first_mpc = t.min(); pre = v.date.min() < first_mpc
    lab = f"{o}" + (" (precovery)" if pre else "")
    ax.text(X1 + pd.Timedelta(days=0.8), y, f"P = {', '.join(f'{p:.2f}' for p in v.priora_All)}", va="center", fontsize=7.6)
    rows.append(dict(object=o, label=lab, y=y, first_mpc=first_mpc.date(), dp2_first=v.date.min().date(), precovery=pre, n_dp2_tracklets=len(v),
                     X05_submitted=int((d.stn == "X05").sum()), priora=", ".join(f"{p:.3f}" for p in v.priora_All)))
R = pd.DataFrame(rows)
ax.set_yticks(R.y, R.label); ax.set_ylim(-0.6, len(objs) - 0.15); ax.set_xlim(X0, X1)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.set_xlabel("2025 (UTC)"); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", top=False)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="*", ls="", color=CS, markeredgecolor=CINK, markersize=10, label="DP2 tracklet, never reported by Rubin"),
                   Line2D([], [], marker="|", ls="", color=CO, markersize=9, mew=1.6, label="MPC observations, other stations"),
                   Line2D([], [], marker="|", ls="", color=CX, markersize=9, mew=1.6, label="MPC observations, Rubin (X05)")],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handletextpad=0.3)
for ext in ("pdf", "png"): fig.savefig(OUT / f"dp2_precovery_timeline.{ext}", dpi=300)
R.to_csv(OUT / "precovery_summary.csv", index=False)  # reproduce copy: keep the summary next to this script; print(R.drop(columns="y").to_string(index=False))
