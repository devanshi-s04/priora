"""Follow-up for the nDiaSources <= 2 rerun's NEO candidates (full_nd2/score/candidates.csv):
tiers (isolated & |b| >= 15: A if P >= 0.5, B otherwise; else set aside), old ID from the first full pull (shared detections),
pair-motion test (consecutive detections < 5 min apart moving < 0.5 x rate-predicted, prediction >= 0.5"), ALeRCE cutouts
(reusing files already downloaded) and template test (source SNR > 5 at the centre of the template at >= 2 detections),
Rubin MPC submissions (B612 export, exposure-midpoint match), SkyBoT and JPL (reused when the old candidate shares the first
detection, otherwise queried). -> score/candidates_annot.csv, ../handoff/lsst_neo_candidates_nd2_detections.csv"""
import json, glob, math, os, pathlib, re, shutil, time
import numpy as np, pandas as pd, requests
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
import astropy.units as u
from astropy.io import fits
T = pathlib.Path(os.environ["PRIORA_ALERTS"]); F = T / "full_nd2"
C = pd.read_csv(F / "score/candidates.csv", index_col=0)
m = pd.read_parquet(F / "score/scores_joined.parquet")
td = pd.read_parquet(F / "score/trk_dets.parquet"); td = td[td.trkSub.isin(C.index)].copy(); td["diaSourceId"] = td.idstring.astype("int64")
D = pd.read_parquet(F / "tracklets/dets_full_r90.parquet")
td = td.merge(D[["diaSourceId", "diaObjectId", "nDiaSources", "visit", "detector", "midpointMjdTai", "ra", "dec", "raErr", "decErr", "band", "psfFlux",
                 "psfFluxErr", "snr", "reliability", "extendedness", "trailLength"]], on="diaSourceId").sort_values(["trkSub", "midpointMjdTai"])
# --- geometry, grouping, tiers
g = SkyCoord(C.ra.values * u.deg, C.dec.values * u.deg).galactic; C["gal_b"] = g.b.deg
t0 = pd.to_datetime(C.utc)
C["same_field"] = [int((((t0 - t0.iloc[i]).abs().dt.total_seconds() / 60 <= 10) & (np.hypot((C.ra - r.ra) * np.cos(np.radians(r.dec)), C.dec - r.dec) < 2)).sum() - 1) for i, (_, r) in enumerate(C.iterrows())]
iso = (C.gal_b.abs() >= 15) & (C.same_field == 0)
C["tier"] = np.where(iso & (C.priora_All >= 0.5), "A", np.where(iso, "B", "set aside"))
# --- old IDs (first full pull) via shared detections
otd = pd.read_parquet(T / "full/score/trk_dets.parquet"); otd["diaSourceId"] = otd.idstring.astype("int64")
OC = pd.read_csv(T / "full/score/candidates_annot.csv", index_col=0)
iso_o = (OC.gal_b.abs() >= 15) & (OC.same_field == 0); OC["tier"] = np.where(iso_o & (OC.priora_All >= 0.5), "A", np.where(iso_o, "B", "set aside"))
otd = otd[otd.trkSub.isin(OC.index)]
def old_id(t):
    s = set(td[td.trkSub == t].diaSourceId); best = otd[otd.diaSourceId.isin(s)].trkSub.value_counts()
    return (best.index[0], int(best.iloc[0])) if len(best) else (None, 0)
C[["old_id", "old_shared_dets"]] = [old_id(t) for t in C.index]
C["old_tier"] = C.old_id.map(OC.tier)
# --- pair test
def pair(g):
    t, ra, de = g.midpointMjdTai.values, g.ra.values, g.dec.values; cosd = np.cos(np.radians(de[0]))
    rate = np.hypot(((ra[-1] - ra[0] + 180) % 360 - 180) * cosd, de[-1] - de[0]) / (t[-1] - t[0]); rs = []
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        if dt * 1440 < 5 and rate * dt * 3600 >= 0.5:
            rs.append(np.hypot(((ra[i + 1] - ra[i] + 180) % 360 - 180) * cosd, de[i + 1] - de[i]) * 3600 / (rate * dt * 3600))
    return min(rs) if rs else np.nan
C["pair_motion_ratio_min"] = td.groupby("trkSub").apply(pair)
C["pair_static"] = C.pair_motion_ratio_min < 0.5
# --- cutouts (ALeRCE) + template test
import sys
from alerce.core import Alerce
A = Alerce()
def snr(im):
    im = np.nan_to_num(im.astype(float)); n = im.shape[0]; c = n // 2
    yy, xx = np.mgrid[:n, :n]; ring = np.hypot(yy - c, xx - c) >= 12; bg = np.median(im[ring]); sd = 1.4826 * np.median(np.abs(im[ring] - bg))
    return (im[c - 2:c + 3, c - 2:c + 3] - bg).sum() / (sd * 5 + 1e-12)
cut = T / "cutouts/alerce_nd2"; cut.mkdir(exist_ok=True)
tsnr = {}
for _, r in td.iterrows():
    f = cut / f"{r.diaSourceId}.fits"
    if not f.exists():
        old = glob.glob(str(T / f"cutouts/alerce/*/*/det*_{r.diaSourceId}.fits"))
        if old: shutil.copy(old[0], f)
        else:
            for att in range(4):
                try:
                    h = A.get_stamps(oid=int(r.diaObjectId), measurement_id=int(r.diaSourceId), survey="lsst")
                    fits.HDUList([fits.PrimaryHDU()] + [fits.ImageHDU(h[k].data, header=h[k].header, name=k.replace("cutout", "").upper())
                                                        for k in ("cutoutScience", "cutoutTemplate", "cutoutDifference")]).writeto(f); break
                except Exception: time.sleep(10 * (att + 1))
    if f.exists(): tsnr[r.diaSourceId] = snr(fits.open(f)["TEMPLATE"].data)
td["template_snr"] = td.diaSourceId.map(tsnr)
C["n_dets_with_template_source"] = td.groupby("trkSub").template_snr.apply(lambda s: int((s > 5).sum()))
C["cutouts_missing"] = td.groupby("trkSub").template_snr.apply(lambda s: int(s.isna().sum()))
C["template_static"] = C.n_dets_with_template_source >= 2
C["static_flag"] = np.select([C.pair_static, C.template_static], ["static: back-to-back detections do not move", "static: source in the template image at 2+ detections"], "no static signature")
# --- Rubin MPC submissions (B612)
B = pd.read_csv(T / "link/b612_mid_matches.csv", dtype={"diaSourceId": "string"}).dropna(subset=["diaSourceId"]); B["diaSourceId"] = B.diaSourceId.astype("int64")
hb = td.merge(B[["diaSourceId", "trksub", "provid", "status", "created"]], on="diaSourceId")
def rub(t):
    h = hb[hb.trkSub == t]
    return "not among Rubin's MPC submissions" if h.empty else f"Rubin tracklet {h.trksub.iloc[0]} ({'unlinked, isolated tracklet file' if h.status.iloc[0] == 'I' else h.provid.iloc[0]}), submitted {str(h.created.iloc[0])[:10]}"
C["rubin_mpc_submission"] = [rub(t) for t in C.index]
# --- SkyBoT / JPL: reuse old answers when the first detection is the same, else query
OD = T / "link/dbcheck"; ND = F / "dbcheck"; ND.mkdir(exist_ok=True)
first = td.groupby("trkSub").first()
ofirst = otd.sort_values("MJD").groupby("trkSub").first()
def sky(t, r):
    oid = C.loc[t, "old_id"]; f = ND / f"skybot_{t}.json"
    if not f.exists():
        if oid and ofirst.loc[oid, "diaSourceId"] == r.diaSourceId and (OD / f"skybot_{oid}.json").exists(): shutil.copy(OD / f"skybot_{oid}.json", f)
        else:
            jd = Time(r.midpointMjdTai, format="mjd", scale="tai").utc.jd
            x = requests.get(f"https://ssp.imcce.fr/webservices/skybot/api/conesearch.php?-ep={jd:.6f}&-ra={r.ra:.6f}&-dec={r.dec:.6f}&-rd={120 / 3600:.5f}&-loc=X05&-mime=json&-output=basic", timeout=120)
            f.write_text(x.text if x.status_code == 200 else "[]"); time.sleep(1)
    try: Dd = pd.DataFrame(json.loads(f.read_text() or "[]"))
    except Exception: Dd = pd.DataFrame()
    s = td[td.trkSub == t]; dt = (s.midpointMjdTai.iloc[-1] - s.midpointMjdTai.iloc[0]) * 24
    ov = np.array([((s.ra.iloc[-1] - s.ra.iloc[0] + 180) % 360 - 180) * np.cos(np.radians(s.dec.iloc[0])) * 3600 / dt, (s.dec.iloc[-1] - s.dec.iloc[0]) * 3600 / dt])
    if Dd.empty: return "none within 120″", None
    b = Dd.sort_values("d (arcsec)").iloc[0]; pv = np.array([b["dRA (arcsec/h)"], b["dDEC (arcsec/h)"]])
    ang = np.degrees(np.arccos(np.clip(pv @ ov / (np.linalg.norm(pv) * np.linalg.norm(ov) + 1e-9), -1, 1))); ratio = np.linalg.norm(pv) / (np.linalg.norm(ov) + 1e-9)
    same = ang < 15 and 0.7 < ratio < 1.4
    return f"{b['Name']} ({b['Class']}) at {b['d (arcsec)']:.0f}″, " + ("motion matches" if same else f"motion differs ({ang:.0f}° off, speed ×{ratio:.2f})"), (b["Name"] if same else None)
res = [sky(t, first.loc[t]) for t in C.index]
C["skybot_nearest"] = [r[0] for r in res]; known = [r[1] for r in res]
def jpl(t, r):
    oid = C.loc[t, "old_id"]; f = ND / f"jpl_{t}.json"
    if not f.exists():
        if oid and ofirst.loc[oid, "diaSourceId"] == r.diaSourceId and (OD / f"jpl_{oid}.json").exists(): shutil.copy(OD / f"jpl_{oid}.json", f)
        else:
            ts = Time(r.midpointMjdTai, format="mjd", scale="tai").utc.strftime("%Y-%m-%d_%H:%M:%S")
            ra = Angle(r.ra * u.deg).to_string(unit=u.hour, sep="-", precision=2, pad=True)
            de = Angle(abs(r.dec) * u.deg).to_string(unit=u.deg, sep="-", precision=1, pad=True); de = ("M" if r.dec < 0 else "") + de
            for att in range(3):
                try:
                    x = requests.get(f"https://ssd-api.jpl.nasa.gov/sb_ident.api?mpc-code=X05&obs-time={ts}&fov-ra-center={ra}&fov-dec-center={de}&fov-ra-hwidth=0.1&fov-dec-hwidth=0.1&two-pass=true&suppress-first-pass=true", timeout=600)
                    x.raise_for_status(); f.write_text(x.text); break
                except Exception: time.sleep(20)
    if not f.exists(): return "JPL query failed"
    d = json.loads(f.read_text()); rows = d.get("data_second_pass", [])
    if not rows: return "none within 120″"
    Dd = pd.DataFrame(rows, columns=d["fields_second"])
    ra2 = Angle(Dd["Astrometric RA (hh:mm:ss)"].values, unit=u.hour).deg
    de2 = Angle([s.replace("'", " ").replace('"', "") for s in Dd["Astrometric Dec (dd mm'ss\")"]], unit=u.deg).deg
    sep = np.degrees(np.arccos(np.clip(np.sin(np.radians(r.dec)) * np.sin(np.radians(de2)) + np.cos(np.radians(r.dec)) * np.cos(np.radians(de2)) * np.cos(np.radians(r.ra - ra2)), -1, 1))) * 3600
    j = int(np.argmin(sep)); return f"{Dd['Object name'].iloc[j]} at {sep[j]:.0f}″" if sep[j] < 120 else "none within 120″"
C["jpl_nearest"] = [jpl(t, first.loc[t]) for t in C.index]
C["matched_to"] = ["; ".join(x for x in ((f"known object {k}" if k else None), (f"Rubin MPC submission {s.split()[2]}" + (" (unlinked)" if "unlinked" in s else "") if s.startswith("Rubin") else None)) if x) or "nothing"
                   for k, s in zip(known, C.rubin_mpc_submission)]
C.to_csv(F / "score/candidates_annot.csv")
# --- hand-off CSV (one row per detection)
td["obs_time_utc"] = Time(td.midpointMjdTai.values, format="mjd", scale="tai").utc.isot; td["mag_ab"] = 31.4 - 2.5 * np.log10(td.psfFlux)
td["det_index_in_tracklet"] = td.groupby("trkSub").cumcount() + 1
cols = ["tier", "n", "arc", "rate", "V", "gal_b", "same_field", "priora_All", "priora_Unk", "d2_raw", "d2_noid", "static_flag", "n_dets_with_template_source", "pair_motion_ratio_min",
        "matched_to", "rubin_mpc_submission", "skybot_nearest", "jpl_nearest", "nearest_catalogue", "nearest_sep_arcsec", "old_id", "old_tier"]
H = td.merge(C[cols], left_on="trkSub", right_index=True).rename(columns={"trkSub": "tracklet_id", "n": "tracklet_n_detections", "midpointMjdTai": "obs_mjd_tai", "band_x": "band", "arc": "tracklet_arc_min", "rate": "tracklet_rate_deg_per_day", "V": "tracklet_V_mag", "gal_b": "galactic_b_deg", "same_field": "other_candidates_same_field_10min_2deg", "priora_All": "priora_P_NEO_All", "priora_Unk": "priora_P_NEO_Unk", "d2_raw": "digest2_raw_NEO", "d2_noid": "digest2_NoID_NEO", "nearest_catalogue": "nearest_MPCORB_object", "nearest_sep_arcsec": "nearest_MPCORB_sep_arcsec"})
H["tier_rank"] = H.tier.map({"A": 1, "B": 2, "set aside": 3}); H = H.sort_values(["tier_rank", "priora_P_NEO_All", "tracklet_id", "obs_mjd_tai"], ascending=[True, False, True, True]).drop(columns="tier_rank")
H["obs_mjd_tai"] = H.obs_mjd_tai.map(lambda x: f"{x:.8f}"); H["ra"] = H.ra.map(lambda x: f"{x:.7f}"); H["dec"] = H.dec.map(lambda x: f"{x:.7f}")
first_cols = ["tier", "tracklet_id", "det_index_in_tracklet", "tracklet_n_detections", "diaSourceId", "visit", "detector", "diaObjectId", "nDiaSources", "obs_time_utc", "obs_mjd_tai", "ra", "dec"]
H = H[first_cols + [c for c in H.columns if c not in first_cols and c not in ("detnum", "trk_ID", "idstring", "MJD", "RA", "Dec", "mag", "band_y")]]
H.to_csv(T / "handoff/lsst_neo_candidates_nd2_detections.csv", index=False)
print(pd.crosstab(C.tier, C.static_flag)); print("matched:", C.matched_to.value_counts().to_dict()); print("old tiers of new candidates:", C.old_tier.value_counts(dropna=False).to_dict())
print("cutouts missing:", int(C.cutouts_missing.sum()))
print(C[C.tier != "set aside"][["tier", "old_id", "old_tier", "priora_All", "static_flag", "matched_to"]].to_string())
