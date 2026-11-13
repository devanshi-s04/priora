"""Template test: is there a source at the cutout centre in the TEMPLATE (a static object) ?
For each cutout: sum in a 5x5 box at the centre minus median background (outer ring, r >= 12 px), divided by
the background noise scaled to the box (robust std x 5). template_snr > 5 -> source in template.
Science cutout measured the same way for reference. -> template_test.csv (per detection) and per-tracklet summary."""
import glob, re, numpy as np, pandas as pd
from astropy.io import fits
def snr(im):
    im = np.nan_to_num(im.astype(float)); n = im.shape[0]; c = n // 2
    yy, xx = np.mgrid[:n, :n]; ring = np.hypot(yy - c, xx - c) >= 12
    bg = np.median(im[ring]); sd = 1.4826 * np.median(np.abs(im[ring] - bg))
    box = im[c - 2:c + 3, c - 2:c + 3] - bg
    return box.sum() / (sd * 5 + 1e-12)
rows = []
for f in glob.glob("alerce/*/*/det*_*.fits"):
    tier, trk, det = re.search(r"alerce/([^/]+)/([^/]+)/det(\d+)_", f).groups()
    h = fits.open(f)
    rows.append(dict(tier=tier.replace("_", " "), tracklet_id=trk, det=int(det), template_snr=snr(h["TEMPLATE"].data), science_snr=snr(h["SCIENCE"].data), diff_snr=snr(h["DIFFERENCE"].data)))
R = pd.DataFrame(rows).sort_values(["tier", "tracklet_id", "det"]); R.to_csv("template_test.csv", index=False)
S = R.groupby(["tier", "tracklet_id"]).agg(n=("det", "size"), n_template_src=("template_snr", lambda x: int((x > 5).sum())), max_template_snr=("template_snr", "max"))
S.to_csv("template_test_tracklets.csv")
pd.set_option("display.width", 200); print(S.round(1).to_string())
