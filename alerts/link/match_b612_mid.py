"""Rubin submits each detection as two positions (start and end of the 30 s exposure). Pair consecutive observations of a
trksub that are 25-35 s apart, average them to the exposure midpoint, and match midpoints to alerts (< 1 s, < 1").
Inputs: data_b612/x05_our_nights.parquet; data_all/ftransfer_913305.parquet (every alert). -> link/b612_mid_matches.csv"""
import numpy as np, pandas as pd
from astropy.time import Time
X = pd.read_parquet("data_b612/x05_our_nights.parquet")
X["trksub"] = X.trksub.fillna("desig:" + X.provid.fillna("?"))   # 67 observations carry only a designation
X = X.sort_values(["trksub", "mjd"])
mids = []
for t, g in X.groupby("trksub"):
    g = g.reset_index(drop=True); i = 0
    while i < len(g):
        if i + 1 < len(g) and 25 <= (g.mjd[i + 1] - g.mjd[i]) * 86400 <= 35:
            mids.append(dict(trksub=t, mjd=(g.mjd[i] + g.mjd[i + 1]) / 2, ra=(g.ra[i] + g.ra[i + 1]) / 2, dec=(g.dec[i] + g.dec[i + 1]) / 2,
                             provid=g.provid[i], permid=g.permid[i], status=g.status[i], created=g.created_at[i], pair=True)); i += 2
        else:
            mids.append(dict(trksub=t, mjd=g.mjd[i], ra=g.ra[i], dec=g.dec[i], provid=g.provid[i], permid=g.permid[i], status=g.status[i], created=g.created_at[i], pair=False)); i += 1
Mi = pd.DataFrame(mids)
A = pd.read_parquet("data_all/ftransfer_913305.parquet", columns=["midpointMjdTai", "ra", "dec", "diaSourceId", "designation", "packed_primary_provisional_designation"])
tA = Time(A.midpointMjdTai.values, format="mjd", scale="tai").utc.mjd; o = np.argsort(tA); tA = tA[o]; A = A.iloc[o].reset_index(drop=True)
res = []
for _, m in Mi.iterrows():
    lo, hi = np.searchsorted(tA, m.mjd - 1 / 86400), np.searchsorted(tA, m.mjd + 1 / 86400)
    if hi <= lo: res.append(None); continue
    s = np.hypot((A.ra.values[lo:hi] - m.ra) * np.cos(np.radians(m.dec)), A.dec.values[lo:hi] - m.dec) * 3600; b = np.argmin(s)
    res.append((A.diaSourceId[lo + b], s[b], A.designation[lo + b] if pd.notna(A.designation[lo + b]) else A.packed_primary_provisional_designation[lo + b]) if s[b] < 1 else None)
Mi["diaSourceId"] = pd.array([int(r[0]) if r else None for r in res], dtype="Int64"); Mi["sep"] = [r[1] if r else np.nan for r in res]; Mi["rubin_known"] = [r[2] if r else None for r in res]
Mi.astype({"diaSourceId": "string"}).to_csv("link/b612_mid_matches.csv", index=False)
print(f"submitted observations {len(X)} -> {len(Mi)} detections ({int(Mi.pair.sum())} paired); matched to an alert: {Mi.diaSourceId.notna().sum()}")
print("matched alerts that Rubin itself called known:", int(Mi.rubin_known.notna().sum()))
# which of OUR unknown detections / tracklets / candidates
t2d = pd.read_csv("full/tracklets/trk2det_full.csv"); t2d.columns = ["trk_ID", "detnum"]
pdf = pd.read_csv("full/tracklets/pairdets_full.csv"); pdf.columns = [c.lstrip("#") for c in pdf.columns]
meta = pd.read_parquet("full/score/meta.parquet").reset_index()
t2d["diaSourceId"] = pdf.idstring.values[t2d.detnum.values].astype("int64")
hit = t2d.merge(Mi.dropna(subset=["diaSourceId"]).astype({"diaSourceId": "int64"}), on="diaSourceId").merge(meta[["trk_ID", "trkSub", "n"]], on="trk_ID")
C = pd.read_csv("full/score/candidates_annot.csv", index_col=0)
hit["candidate"] = hit.trkSub.isin(C.index)
g = hit.groupby("trkSub").agg(n=("n", "first"), trksubs=("trksub", lambda x: ",".join(sorted(set(x)))), provid=("provid", "first"), status=("status", "first"), cand=("candidate", "first"))
pd.set_option("display.width", 250); print(f"our scored tracklets with a Rubin-submitted detection: {len(g)} (candidates: {int(g.cand.sum())})"); print(g.to_string())
g.to_csv("link/b612_tracklet_hits.csv")
