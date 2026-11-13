"""Cutouts for every detection in the hand-off CSV from ALeRCE's LSST stamps API (alerce client, survey="lsst",
oid = diaObjectId, measurement_id = diaSourceId). Saves one FITS per detection (Science, Template, Difference HDUs) and a
PNG strip (science | template | difference, each with a 1-99.5 percentile stretch). -> alerce/<tier>/<tracklet>/"""
import csv, pathlib, time, numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from alerce.core import Alerce
a = Alerce()
H = list(csv.DictReader(open("../handoff/lsst_neo_candidates_detections_2026-11-03.csv")))
order = {"A": 0, "B": 1, "set aside": 2}; H.sort(key=lambda r: (order[r["tier"]], r["tracklet_id"], int(r["det_index_in_tracklet"])))
ok = fail = 0
for r in H:
    d = pathlib.Path("alerce") / r["tier"].replace(" ", "_") / r["tracklet_id"]; d.mkdir(parents=True, exist_ok=True)
    stem = d / f"det{r['det_index_in_tracklet']}_{r['diaSourceId']}"
    if stem.with_suffix(".fits").exists(): continue
    h = None
    for att in range(4):
        try:
            h = a.get_stamps(oid=int(r["diaObjectId"]), measurement_id=int(r["diaSourceId"]), survey="lsst"); break
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"; time.sleep(10 * (att + 1))
    if not h:
        fail += 1; print("FAIL", r["tracklet_id"], r["det_index_in_tracklet"], err, flush=True); continue
    hl = fits.HDUList([fits.PrimaryHDU()] + [fits.ImageHDU(h[k].data, header=h[k].header, name=k.replace("cutout", "").upper()) for k in ("cutoutScience", "cutoutTemplate", "cutoutDifference")])
    for k, v in (("TRACKLET", r["tracklet_id"]), ("TIER", r["tier"]), ("DIASRC", r["diaSourceId"]), ("DIAOBJ", r["diaObjectId"]), ("VISIT", r["visit"]),
                 ("DETECTOR", r["detector"]), ("OBSUTC", r["obs_time_utc"]), ("RA", float(r["ra"])), ("DEC", float(r["dec"])), ("BAND", r["band"]), ("STATIC", r.get("static_flag", "")[:60])):
        hl[0].header[k] = v
    hl.writeto(stem.with_suffix(".fits"), overwrite=True)
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.4))
    for x, k in zip(ax, ("cutoutScience", "cutoutTemplate", "cutoutDifference")):
        im = np.nan_to_num(h[k].data.astype(float)); lo, hi = np.percentile(im, [1, 99.5])
        x.imshow(im, origin="lower", cmap="gray", vmin=lo, vmax=hi); x.set_xticks([]); x.set_yticks([]); x.set_title(k.replace("cutout", ""), fontsize=8)
    fig.suptitle(f"{r['tracklet_id']} (tier {r['tier']}) det {r['det_index_in_tracklet']}  {r['obs_time_utc'][:19]}  {r['band']}", fontsize=8)
    fig.savefig(stem.with_suffix(".png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    ok += 1; print(r["tier"], r["tracklet_id"], r["det_index_in_tracklet"], "ok", flush=True)
    time.sleep(0.5)
print("DONE ok", ok, "fail", fail)
