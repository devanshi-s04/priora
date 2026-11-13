"""Hand-off CSV for image pulls: the 23 strong DP2 candidates (verdict group 3, isolated), one row per detection.
DP2 data: data-rights holders only. -> ../handoff/dp2_strong_candidates_detections.csv"""
import numpy as np, pandas as pd
from astropy.time import Time
S = "score/"
v = pd.read_parquet(S + "candidate_verdict.parquet"); v = v[v.cat.str.startswith("3") & (v.neighbours <= 1)]
c = pd.read_parquet(S + "unknown3_candidates_flags.parquet"); u = pd.read_parquet(S + "unknown3_joined.parquet")
td = pd.read_parquet(S + "unknown_3plus_dets.parquet"); td = td[td.trkSub.isin(v.index)].copy(); td["diaSourceId"] = td.idstring.astype("int64")
det = pd.read_parquet("unknown/dets.parquet"); det["diaSourceId"] = det.diaSourceId.astype("int64")
x = td.merge(det, on="diaSourceId", how="left")
x["mag"] = np.where(x.psfFlux > 0, -2.5 * np.log10(x.psfFlux.clip(lower=1e-9)) + 31.4, np.nan)   # nJy -> AB
x["utc"] = Time(x.midpointMjdTai.values, format="mjd", scale="tai").utc.isot
x = x.join(u[["n", "ds42_All", "ds42_Unk", "d2_raw", "d2_noid"]], on="trkSub").join(c[["rate", "gal_b", "trail_ratio_med", "pair_ratio"]], on="trkSub")
x["group"] = np.where(x.n >= 4, "strong 4+", "strong 3")
x = x.sort_values(["trkSub", "midpointMjdTai"]); x["detnum"] = x.groupby("trkSub").cumcount() + 1
out = x.sort_values(["group", "ds42_All", "trkSub", "midpointMjdTai"], ascending=[False, False, True, True])[
    ["group", "trkSub", "n", "detnum", "diaSourceId", "diaObjectId", "visit", "detector", "band", "midpointMjdTai", "utc", "ra", "dec", "raErr", "decErr",
     "mag", "snr", "trailLength", "reliability", "ds42_All", "ds42_Unk", "d2_raw", "d2_noid", "rate", "gal_b", "trail_ratio_med", "pair_ratio"]].rename(
    columns={"trkSub": "tracklet", "n": "n_dets", "detnum": "det_index", "midpointMjdTai": "mjd_tai", "utc": "utc_iso", "rate": "rate_deg_per_day",
             "trailLength": "trail_length_arcsec", "trail_ratio_med": "trail_measured_over_expected"})
out["diaSourceId"] = out.diaSourceId.astype(str); out["diaObjectId"] = out.diaObjectId.astype("Int64").astype(str)
out = out.round({"mjd_tai": 8, "ra": 7, "dec": 7, "raErr": 8, "decErr": 8, "mag": 3, "snr": 2, "trail_length_arcsec": 3, "reliability": 4, "ds42_All": 5, "ds42_Unk": 5,
                  "d2_raw": 2, "d2_noid": 2, "rate_deg_per_day": 4, "gal_b": 3, "trail_measured_over_expected": 3, "pair_ratio": 3})
out.to_csv("../handoff/dp2_strong_candidates_detections.csv", index=False)
print(len(out), "rows,", out.tracklet.nunique(), "tracklets", out.groupby("group").tracklet.nunique().to_dict(), "missing visit:", int(out.visit.isna().sum()))
