"""Unknown-alert NEO candidates: tracklets with 3+ detections, no catalogue match (status unmatched), priora All P >= 0.1.
Cross-checks: nearest catalogue object to any detection (match/stage2_all.parquet, any separation) and Rubin (X05)
lines in the MPC Isolated Tracklet File (data_unk/mpc/itf.txt.gz) within 5 arcsec and 5 s. -> candidates.csv"""
import gzip, numpy as np, pandas as pd
from astropy.time import Time
m = pd.read_parquet("scores_joined.parquet")
m["V"] = pd.read_csv("priora_All.csv").set_index("object_id").V
C = m[(m.status == "unmatched") & (m.n >= 3) & (m.priora_All >= 0.1)].copy()
td = pd.read_parquet("trk_dets.parquet"); td = td[td.trkSub.isin(C.index)]
first = td.sort_values("MJD").groupby("trkSub").first()
C["mjd_tai"] = first.MJD; C["ra"] = first.RA; C["dec"] = first.Dec
C["utc"] = [t[:16].replace("T", " ") for t in Time(C.mjd_tai.values, format="mjd", scale="tai").utc.isot]
S = pd.read_parquet("../match/stage2_all.parquet")
near = S[S.det.isin(pd.read_parquet("../match/match.parquet").reset_index().set_index("detnum").loc[td.detnum.unique(), "index"] if False else S.det)]  # placeholder, replaced below
M = pd.read_parquet("../match/match.parquet")                      # row index = det; detnum column
d2row = dict(zip(M.detnum, M.index))
td["det"] = td.detnum.map(d2row)
best = S.sort_values("sep_arcsec").drop_duplicates("det").set_index("det")
td["near_desig"] = td.det.map(best.desig); td["near_sep"] = td.det.map(best.sep_arcsec)
nb = td.sort_values("near_sep").groupby("trkSub").first()
C["nearest_catalogue"] = nb.near_desig; C["nearest_sep_arcsec"] = nb.near_sep
# ITF X05 lines
rows = []
for L in gzip.open("../../data_unk/mpc/itf.txt.gz", "rt"):
    if L[77:80] != "X05" or L[15:19] != "2026": continue
    y, mo, dd = int(L[15:19]), int(L[20:22]), float(L[23:32])
    ra = (int(L[32:34]) + int(L[35:37]) / 60 + float(L[38:44]) / 3600) * 15
    s = -1 if L[44] == "-" else 1; de = s * (int(L[45:47]) + int(L[48:50]) / 60 + float(L[51:56]) / 3600)
    rows.append((Time(f"{y}-{mo:02d}-01", scale="utc").mjd + dd - 1, ra, de, L[5:12].strip()))
I = pd.DataFrame(rows, columns=["mjd_utc", "ra", "dec", "trksub"])
td["mjd_utc"] = Time(td.MJD.values, format="mjd", scale="tai").utc.mjd
hit = {}
for _, r in td.iterrows():
    k = I[(abs(I.mjd_utc - r.mjd_utc) < 5 / 86400)]
    if len(k):
        sep = np.hypot((k.ra - r.RA) * np.cos(np.radians(r.Dec)), k.dec - r.Dec) * 3600
        if (sep < 5).any(): hit.setdefault(r.trkSub, set()).add(k.trksub[sep < 5].iloc[0])
C["itf_trksub"] = [",".join(sorted(hit.get(t, []))) for t in C.index]
C = C.sort_values("priora_All", ascending=False)
cols = ["night", "utc", "ra", "dec", "n", "arc", "rate", "V", "priora_All", "priora_Unk", "d2_raw", "d2_noid", "nearest_catalogue", "nearest_sep_arcsec", "itf_trksub"]
C[cols].to_csv("candidates.csv")
print(len(C), "candidates;", (C.priora_All >= 0.5).sum(), "at P>=0.5; ITF hits", (C.itf_trksub != "").sum(), "; ITF X05 2026 lines", len(I))
pd.set_option("display.width", 250); print(C[cols].round(3).to_string())
