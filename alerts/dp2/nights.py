"""Visits per night in DP2 (dayObs = floor(MJD - 0.5)), to choose dense runs of consecutive nights. -> nights.csv"""
import pathlib, requests, pyvo, numpy as np, pandas as pd
from astropy.time import Time
s = requests.Session(); s.headers["Authorization"] = "Bearer " + (pathlib.Path.home() / ".rsp-tap-token").read_text().strip()
tap = pyvo.dal.TAPService("https://data.lsst.cloud/api/tap", session=s)
v = tap.run_sync("SELECT visit, expMidptMJD, band, ra, dec, expTime FROM dp2.Visit").to_table().to_pandas()
v["night"] = np.floor(v.expMidptMJD - 0.5).astype(int)
n = v.groupby("night").agg(visits=("visit", "size"), bands=("band", lambda b: "".join(sorted(set(b)))), ra_span=("ra", lambda r: float(r.max() - r.min())))
n["date"] = Time(n.index.values + 0.5, format="mjd").iso; n["date"] = n.date.str[:10]
n.to_csv("nights.csv"); v.to_parquet("visits.parquet")
print("visits", len(v), "| nights", len(n), "| range", n.date.iloc[0], "to", n.date.iloc[-1])
# best 9-night windows by total visits, counting nights with >= 100 visits
idx = n.index.values; dense = n.visits.where(n.visits >= 100, 0)
best = sorted(((int(dense[(idx >= a) & (idx < a + 9)].sum()), int(((idx >= a) & (idx < a + 9) & (n.visits >= 100)).sum()), a) for a in idx), reverse=True)[:8]
for tot, k, a in best: print(f"  window from {Time(a + 0.5, format='mjd').iso[:10]}: {k} nights with >=100 visits, {tot} visits")
print(n.sort_values("visits", ascending=False).head(12).to_string())
