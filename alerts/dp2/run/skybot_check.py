"""SkyBoT (IMCCE) cone search, 120" around each DP2 candidate's first detection, observatory X05 (user approved SkyBoT only, no JPL).
Candidates: all 4+ and the unflagged 3+ (unlabelled, ds42 All P >= 0.1). Failed requests are retried and never cached as 'no match'.
-> ../skybot/skybot_<trkSub>.json, score/skybot_candidates.parquet"""
import json, math, time
from pathlib import Path
import numpy as np, pandas as pd, requests
from astropy.time import Time
S = Path("score"); OUT = Path("../skybot")
c = pd.read_parquet(S / "unknown3_candidates_flags.parquet"); u = pd.read_parquet(S / "unknown3_joined.parquet")
c["n"] = u.n.reindex(c.index); c = c[(c.n >= 4) | ~c.any_flag2]
d = pd.read_parquet(S / "unknown_3plus_dets.parquet").sort_values(["trkSub", "MJD"]).groupby("trkSub").first()
c = c.join(d[["MJD", "RA", "Dec"]]); c["jd_utc"] = Time(c.MJD.values, format="mjd", scale="tai").utc.jd
def skybot(t, r):
    f = OUT / f"skybot_{t}.json"
    if not f.exists():
        url = (f"https://ssp.imcce.fr/webservices/skybot/api/conesearch.php?-ep={r.jd_utc:.6f}&-ra={r.RA:.6f}&-dec={r.Dec:.6f}"
               f"&-rd={120 / 3600:.5f}&-loc=X05&-mime=json&-output=basic")
        for k in range(4):
            try:
                x = requests.get(url, timeout=120)
                if x.status_code == 200: f.write_text(x.text); break
                if x.status_code in (204, 404) or "No solar system object" in x.text: f.write_text("[]"); break   # SkyBoT: nothing in the cone
            except requests.RequestException: pass
            time.sleep(5 * (k + 1))
        time.sleep(1)
    if not f.exists(): return dict(trkSub=t, skybot_status="failed")
    try: D = pd.DataFrame(json.loads(f.read_text() or "[]"))
    except Exception: return dict(trkSub=t, skybot_status="unparsed")
    if D.empty: return dict(trkSub=t, skybot_status="ok", skybot_n=0)
    b = D.sort_values("d (arcsec)").iloc[0]
    return dict(trkSub=t, skybot_status="ok", skybot_n=len(D), skybot_nearest=str(b.get("Name")), skybot_sep=float(b["d (arcsec)"]),
                skybot_V=b["VMag (mag)"], skybot_rate=math.hypot(b["dRA (arcsec/h)"], b["dDEC (arcsec/h)"]) * 24 / 3600,
                skybot_class=b.get("Class"), skybot_err=b.get("Err (arcsec)"))
R = pd.DataFrame([skybot(t, r) for t, r in c.iterrows()]).set_index("trkSub")
R = R.join(c[["n", "rate", "ds42_All", "gal_b", "trail_confirmed", "any_flag2"]]); R.to_parquet(S / "skybot_candidates.parquet")
print(R.skybot_status.value_counts().to_dict(), "queried", len(R))
