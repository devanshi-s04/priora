"""Match Rubin's MPC-submitted X05 observations (B612 / Asteroid Institute export, obs_sbn_X05_full.parquet, 2026-11-03)
for our nights against (a) every unknown detection of the full pull, (b) every known-object detection, by time (< 2 s)
and position (< 2"). Then report which NEO candidates have submitted detections. -> b612_matches.parquet"""
import os
import numpy as np, pandas as pd
from astropy.time import Time
T = os.environ["PRIORA_ALERTS"]
X = pd.read_parquet(f"{T}/data_b612/x05_our_nights.parquet")
X["mjd_utc"] = X.mjd
def match(D, name, tcol, racol, decol, idcol):
    t_utc = Time(D[tcol].values, format="mjd", scale="tai").utc.mjd
    out = []
    for j, x in X.iterrows():
        k = np.where(np.abs(t_utc - x.mjd_utc) * 86400 < 2)[0]
        if len(k) == 0: continue
        s = np.hypot((D[racol].values[k] - x.ra) * np.cos(np.radians(x.dec)), D[decol].values[k] - x.dec) * 3600
        b = np.argmin(s)
        if s[b] < 2: out.append(dict(x_id=x.id, trksub=x.trksub, provid=x.provid, permid=x.permid, status=x.status, sep=s[b], dt=(t_utc[k[b]] - x.mjd_utc) * 86400, set=name, det=D[idcol].values[k[b]]))
    return pd.DataFrame(out)
U = pd.read_parquet(f"{T}/full/tracklets/dets_full_r90.parquet")
K = pd.read_parquet(f"{T}/data/ftransfer_julyoct_labelled.parquet")
mu = match(U, "unknown (full pull, after cuts)", "midpointMjdTai", "ra", "dec", "diaSourceId")
mk = match(K, "known (Rubin-associated)", "midpointMjdTai", "ra", "dec", "diaSourceId")
Ua = pd.read_parquet(f"{T}/data_all/ftransfer_913305.parquet")
ma = match(Ua, "any alert (full transfer, no cuts)", "midpointMjdTai", "ra", "dec", "diaSourceId")
M = pd.concat([mu, mk, ma]); M.to_parquet(f"{T}/link/b612_matches.parquet")
for n, m in (("unknown after cuts", mu), ("known", mk), ("any alert", ma)):
    print(f"{n}: {len(m)} of {len(X)} submitted observations matched; designations {sorted(m.provid.dropna().unique())[:15]}")
# candidates
t2d = pd.read_csv(f"{T}/full/tracklets/trk2det_full.csv"); t2d.columns = ["trk_ID", "detnum"]
pdf = pd.read_csv(f"{T}/full/tracklets/pairdets_full.csv"); pdf.columns = [c.lstrip("#") for c in pdf.columns]
meta = pd.read_parquet(f"{T}/full/score/meta.parquet").reset_index().set_index("trkSub")
C = pd.read_csv(f"{T}/full/score/candidates_annot.csv", index_col=0)
cd = t2d[t2d.trk_ID.isin(meta.loc[C.index, "trk_ID"])].merge(meta.reset_index()[["trkSub", "trk_ID"]], on="trk_ID")
cd["diaSourceId"] = pdf.idstring.values[cd.detnum.values].astype("int64")
hit = cd.merge(mu, left_on="diaSourceId", right_on="det")
print("candidate detections:", len(cd), "| matched to a Rubin MPC submission:", len(hit), hit[["trkSub", "trksub", "provid", "status", "sep"]].to_string() if len(hit) else "")
