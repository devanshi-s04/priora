"""Decide SkyBoT matches for the DP2 candidates. A known object counts as a candidate match when its speed is within 20% and its
direction within 20 deg of the tracklet's (first to last detection). For those, SkyBoT is queried again at the LAST detection:
a real match keeps the same offset (orbit error moves with the object); a chance neighbour does not.
match: offset at both ends <= 10" ; same-object (orbit error): offsets agree within 5" ; else not the same object.
-> score/skybot_matches.parquet (one row per candidate; nothing removed)"""
import json, math, time
from pathlib import Path
import numpy as np, pandas as pd, requests
from astropy.time import Time
S = Path("score"); OUT = Path("../skybot")
R = pd.read_parquet(S / "skybot_candidates.parquet")
d = pd.read_parquet(S / "unknown_3plus_dets.parquet").sort_values(["trkSub", "MJD"])
F, L = d.groupby("trkSub").first(), d.groupby("trkSub").last()
def pa_rate(ra0, de0, ra1, de1, dt):
    dx = ((ra1 - ra0 + 180) % 360 - 180) * math.cos(math.radians(de0)); dy = de1 - de0
    return math.degrees(math.atan2(dx, dy)) % 360, math.hypot(dx, dy) / dt
def query(tag, jd, ra, de):
    f = OUT / f"skybot_{tag}.json"
    if not f.exists():
        url = f"https://ssp.imcce.fr/webservices/skybot/api/conesearch.php?-ep={jd:.6f}&-ra={ra:.6f}&-dec={de:.6f}&-rd={120 / 3600:.5f}&-loc=X05&-mime=json&-output=basic"
        for k in range(4):
            try:
                x = requests.get(url, timeout=120)
                if x.status_code == 200: f.write_text(x.text); break
                if x.status_code in (204, 404): f.write_text("[]"); break
            except requests.RequestException: pass
            time.sleep(5 * (k + 1))
        time.sleep(1)
    return json.loads(f.read_text() or "[]") if f.exists() else None
rows = []
for t in R.index:
    f, l = F.loc[t], L.loc[t]; pa, rate = pa_rate(f.RA, f.Dec, l.RA, l.Dec, l.MJD - f.MJD)
    D = json.loads((OUT / f"skybot_{t}.json").read_text() or "[]"); best = dict(trkSub=t, skybot_result="no known object within 120\"")
    cands = []
    for o in D:
        dra, dde = float(o["dRA (arcsec/h)"]), float(o["dDEC (arcsec/h)"]); sr = math.hypot(dra, dde) * 24 / 3600
        spa = math.degrees(math.atan2(dra, dde)) % 360; dpa = abs((spa - pa + 180) % 360 - 180)
        if abs(sr - rate) / rate < 0.2 and dpa < 20: cands.append((float(o["d (arcsec)"]), o, sr, dpa))
    if D and not cands: best["skybot_result"] = "known object nearby, motion differs"; best["skybot_nearest"] = min(D, key=lambda o: float(o["d (arcsec)"]))["Name"]
    for s0, o, sr, dpa in sorted(cands, key=lambda c: c[0]):
        jd = Time(l.MJD, format="mjd", scale="tai").utc.jd; E = query(f"{t}_last", jd, l.RA, l.Dec)
        s1 = next((float(e["d (arcsec)"]) for e in (E or []) if e["Name"] == o["Name"]), np.nan)
        res = ("match" if s0 <= 10 and s1 <= 10 else "same object, orbit offset" if abs(s1 - s0) <= 5 else "not the same object")
        best = dict(trkSub=t, skybot_result=res, skybot_object=o["Name"], skybot_class=o.get("Class"), sep_first=s0, sep_last=s1,
                    rate=rate, skybot_rate=sr, dpa=dpa)
        if res != "not the same object": break
    rows.append(best)
M = pd.DataFrame(rows).set_index("trkSub").join(R[["n", "priora_All", "gal_b", "trail_confirmed", "any_flag2"]]); M.to_parquet(S / "skybot_matches.parquet")
print(M.skybot_result.value_counts().to_string())
print(M[M.skybot_object.notna()][["n", "skybot_result", "skybot_object", "skybot_class", "sep_first", "sep_last", "rate", "skybot_rate", "dpa", "priora_All"]].round(2).sort_values("skybot_result").to_string())
for k in (3, 4):
    m = M[M.n >= k]; print(f"{k}+: checked {len(m)}, still unknown {int((~m.skybot_result.isin(['match', 'same object, orbit offset'])).sum())}")
