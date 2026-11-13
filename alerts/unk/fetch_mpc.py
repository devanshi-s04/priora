"""MPC observations (get-obs API, ADES_DF) for every object first designated 2026 N (Jul 1-15) or 2026 U (Oct 16-31)
in data/mpcorb_extended.json.gz -> data_unk/mpc/obs_2026NU.parquet (all stations; X05 used for matching)."""
import gzip, json, time, requests, pandas as pd
J = json.load(gzip.open("data/mpcorb_extended.json.gz"))
des = [o["Principal_desig"] for o in J if o.get("Principal_desig", "").startswith(("2026 N", "2026 U"))]
rows = []
for k in range(0, len(des), 1):
    b = des[k:k + 1]
    for a in range(3):
        try:
            r = requests.get("https://data.minorplanetcenter.net/api/get-obs", json={"desigs": b, "output_format": ["ADES_DF"]}, timeout=120)
            r.raise_for_status(); break
        except Exception as e:
            print("retry", k, e); time.sleep(5)
    for blk in (r.json() if isinstance(r.json(), list) else []):
        for o in blk.get("ADES_DF", []):
            rows.append({c: o.get(c) for c in ("provid", "permid", "trksub", "stn", "obstime", "ra", "dec", "mag", "band", "obsid", "ref", "disc")})
    time.sleep(0.3)
df = pd.DataFrame(rows); df.to_parquet("data_unk/mpc/obs_2026NU.parquet")
print(len(des), "objects;", len(df), "observations;", (df.stn == "X05").sum(), "from X05;", df[df.stn == "X05"].provid.nunique(), "objects with X05")
