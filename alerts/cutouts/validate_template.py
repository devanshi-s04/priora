"""Validate the template test on REAL asteroids: ALeRCE cutouts for 60 random catalogue-labelled 3+ detection tracklets
import os
from the full unknown pull (known objects Rubin did not associate), same SNR measurement as template_test.py.
-> validate/real_template_test.csv"""
import pathlib, re, time, numpy as np, pandas as pd
from astropy.io import fits
from alerce.core import Alerce
exec(open("template_test.py").read().split("rows = []")[0])          # reuse snr()
T = os.path.join(os.environ["PRIORA_ALERTS"], "full")
m = pd.read_parquet(f"{T}/score/scores_joined.parquet"); L = m[(m.status == "labelled") & (m.n >= 3)]
L = L.sample(60, random_state=20261103)
td = pd.read_parquet(f"{T}/score/trk_dets.parquet"); td = td[td.trkSub.isin(L.index)]
D = pd.read_parquet(f"{T}/tracklets/dets_full_r90.parquet")[["diaSourceId", "diaObjectId"]]
td["diaSourceId"] = td.idstring.astype("int64"); td = td.merge(D, on="diaSourceId")
a = Alerce(); out = pathlib.Path("validate"); out.mkdir(exist_ok=True); rows = []
for _, r in td.iterrows():
    try:
        h = a.get_stamps(oid=int(r.diaObjectId), measurement_id=int(r.diaSourceId), survey="lsst")
        rows.append(dict(trkSub=r.trkSub, diaSourceId=r.diaSourceId, template_snr=snr(h["cutoutTemplate"].data), science_snr=snr(h["cutoutScience"].data)))
    except Exception as e:
        rows.append(dict(trkSub=r.trkSub, diaSourceId=r.diaSourceId, template_snr=np.nan, science_snr=np.nan))
    time.sleep(0.3)
R = pd.DataFrame(rows); R.to_csv(out / "real_template_test.csv", index=False)
S = R.groupby("trkSub").agg(n=("template_snr", "size"), got=("template_snr", "count"), n_src=("template_snr", lambda x: int((x > 5).sum())))
print("real tracklets", len(S), "| detections", len(R), "got", int(R.template_snr.notna().sum()))
print("detections with template source (>5):", int((R.template_snr > 5).sum()), f"({(R.template_snr > 5).mean():.1%})")
for k in (1, 2, 3): print(f"tracklets with >= {k} template-source detections:", int((S.n_src >= k).sum()), "of", len(S))
