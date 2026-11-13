"""Contact sheet per tier from the ALeRCE cutouts: one row per detection (science | template | difference), grouped by tracklet,
with the static flag in the row label. -> sheets/tier_<tier>.png"""
import sys, glob, csv, pathlib, numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tier = sys.argv[1]
H = [r for r in csv.DictReader(open("../handoff/lsst_neo_candidates_detections_2026-11-03.csv")) if r["tier"] == tier]
H.sort(key=lambda r: (-float(r["ds42_P_NEO_All"]), r["tracklet_id"], int(r["det_index_in_tracklet"])))
fig, axes = plt.subplots(len(H), 3, figsize=(5.4, 1.75 * len(H)))
for row, r in zip(axes, H):
    f = glob.glob(f"alerce/{tier.replace(' ', '_')}/{r['tracklet_id']}/det{r['det_index_in_tracklet']}_*.fits")[0]
    hl = fits.open(f)
    for x, k in zip(row, ("SCIENCE", "TEMPLATE", "DIFFERENCE")):
        im = np.nan_to_num(hl[k].data.astype(float)); lo, hi = np.percentile(im, [1, 99.5])
        x.imshow(im, origin="lower", cmap="gray", vmin=lo, vmax=hi); x.set_xticks([]); x.set_yticks([])
    flag = r.get("static_flag", "")
    row[0].set_ylabel(f"{r['tracklet_id']} d{r['det_index_in_tracklet']}\n{r['obs_time_utc'][5:16]} {r['band']}\nP={float(r['ds42_P_NEO_All']):.2f}" + ("\nSTATIC" if flag.startswith("static") else ("\nlikely static" if flag.startswith("likely") else "")), fontsize=6.5, rotation=0, ha="right", va="center")
axes[0][0].set_title("science", fontsize=8); axes[0][1].set_title("template", fontsize=8); axes[0][2].set_title("difference", fontsize=8)
pathlib.Path("sheets").mkdir(exist_ok=True); out = f"sheets/tier_{tier.replace(' ', '_')}.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); print(out, len(H), "rows")
