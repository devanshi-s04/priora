"""Check the 73 full-pull NEO candidates against JPL's Small-Body Identification API (sb_ident, two-pass) and IMCCE SkyBoT,
at each tracklet's first detection, observatory X05. Raw answers kept in dbcheck/; summary -> dbcheck_summary.csv.
Tier A and B first. JPL: 3 requests at a time (each takes ~1-2 min); SkyBoT: sequential, radius 120"."""
import os
import json, time, math, concurrent.futures as cf
from pathlib import Path
import numpy as np, pandas as pd, requests
from astropy.time import Time
from astropy.coordinates import Angle
import astropy.units as u
T = Path(os.environ["DS42_ALERTS"]); OUT = T / "link/dbcheck"; OUT.mkdir(exist_ok=True)
C = pd.read_csv(T / "full/score/candidates_annot.csv", index_col=0)
C["tier"] = np.where((C.gal_b.abs() >= 15) & (C.same_field == 0) & (C.ds42_All >= 0.5), "A", np.where((C.gal_b.abs() >= 15) & (C.same_field == 0), "B", "set aside"))
C = C.assign(o=C.tier.map({"A": 0, "B": 1, "set aside": 2})).sort_values(["o", "ds42_All"], ascending=[True, False])
td = pd.read_parquet(T / "full/score/trk_dets.parquet"); first = td.sort_values("MJD").groupby("trkSub").first()
C["mjd_tai"] = first.loc[C.index, "MJD"].values
C["jd_utc"] = Time(C.mjd_tai.values, format="mjd", scale="tai").utc.jd


def sep(ra1, de1, ra2, de2):
    return np.degrees(np.arccos(np.clip(np.sin(np.radians(de1)) * np.sin(np.radians(de2)) + np.cos(np.radians(de1)) * np.cos(np.radians(de2)) * np.cos(np.radians(ra1 - ra2)), -1, 1))) * 3600


def jpl(t, r):
    f = OUT / f"jpl_{t}.json"
    if not f.exists():
        ts = Time(r.jd_utc, format="jd", scale="utc").strftime("%Y-%m-%d_%H:%M:%S")
        ra = Angle(r.ra * u.deg).to_string(unit=u.hour, sep="-", precision=2, pad=True)
        de = Angle(r.dec * u.deg).to_string(unit=u.deg, sep="-", precision=1, pad=True, alwayssign=True).replace("-", "M", 1) if r.dec < 0 else Angle(r.dec * u.deg).to_string(unit=u.deg, sep="-", precision=1, pad=True)
        de = de.replace("+", "")
        url = ("https://ssd-api.jpl.nasa.gov/sb_ident.api?mpc-code=X05&obs-time=" + ts + "&fov-ra-center=" + ra + "&fov-dec-center=" + de +
               "&fov-ra-hwidth=0.1&fov-dec-hwidth=0.1&two-pass=true&suppress-first-pass=true")
        for a in range(3):
            try:
                x = requests.get(url, timeout=600); x.raise_for_status(); f.write_text(x.text); break
            except Exception as e:
                time.sleep(20)
    if not f.exists():
        return dict(trkSub=t, jpl_status="failed")
    d = json.loads(f.read_text()); rows = d.get("data_second_pass", []); flds = d.get("fields_second", [])
    if not rows:
        return dict(trkSub=t, jpl_status="ok", jpl_n=0)
    D = pd.DataFrame(rows, columns=flds)
    ra = Angle(D["Astrometric RA (hh:mm:ss)"].values, unit=u.hour).deg
    de = Angle([s.replace("'", " ").replace('"', "") for s in D["Astrometric Dec (dd mm'ss\")"]], unit=u.deg).deg
    D["sep"] = sep(r.ra, r.dec, ra, de); b = D.sort_values("sep").iloc[0]
    rate = math.hypot(float(b["RA rate (\"/h)"]), float(b["Dec rate (\"/h)"])) * 24 / 3600
    return dict(trkSub=t, jpl_status="ok", jpl_n=len(D), jpl_nearest=b["Object name"], jpl_sep=float(b.sep), jpl_V=b["Visual magnitude (V)"],
                jpl_rate=rate, jpl_err=b.get("Est. error RA (\")"))


def skybot(t, r):
    f = OUT / f"skybot_{t}.json"
    if not f.exists():
        url = f"https://ssp.imcce.fr/webservices/skybot/api/conesearch.php?-ep={r.jd_utc:.6f}&-ra={r.ra:.6f}&-dec={r.dec:.6f}&-rd={120 / 3600:.5f}&-loc=X05&-mime=json&-output=basic"
        x = requests.get(url, timeout=120)
        f.write_text(x.text if x.status_code == 200 else "[]")
        time.sleep(1)
    try: D = pd.DataFrame(json.loads(f.read_text() or "[]"))
    except Exception: D = pd.DataFrame()
    if D.empty:
        return dict(trkSub=t, skybot_n=0)
    b = D.sort_values("d (arcsec)").iloc[0]
    rate = math.hypot(b["dRA (arcsec/h)"], b["dDEC (arcsec/h)"]) * 24 / 3600
    return dict(trkSub=t, skybot_n=len(D), skybot_nearest=str(b.get("Name")), skybot_sep=float(b["d (arcsec)"]), skybot_V=b["VMag (mag)"],
                skybot_rate=rate, skybot_class=b.get("Class"), skybot_err=b.get("Err (arcsec)"))


sb = [skybot(t, r) for t, r in C.iterrows()]
print("SkyBoT done", flush=True)
with cf.ThreadPoolExecutor(3) as ex:
    jp = list(ex.map(lambda tr: jpl(*tr), C.iterrows()))
S = C[["tier", "utc", "ra", "dec", "n", "rate", "V", "ds42_All"]].join(pd.DataFrame(sb).set_index("trkSub")).join(pd.DataFrame(jp).set_index("trkSub"))
S.to_csv(T / "link/dbcheck_summary.csv")
pd.set_option("display.width", 250); print(S.round(2).to_string())
