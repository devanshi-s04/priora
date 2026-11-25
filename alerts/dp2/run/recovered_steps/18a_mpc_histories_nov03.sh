#!/usr/bin/env bash
# MPC observation histories (network; fetched 2026-11-03): ../precovery_fig/mpc_obs_*.csv
# Recovered verbatim from the analysis session (log line 28880); run with 'source env.sh' first.
# (The session wrote these to a scratch folder, then moved them to dp2/precovery_fig; this version writes there directly.)
set -e
cd $PRIORA_ALERTS/dp2/precovery_fig && ${PY:-python} - <<'EOF'
import requests, pandas as pd, io
for k in ("2025 OL4","2025 OT12","2025 OZ"):
    r=requests.get("https://data.minorplanetcenter.net/api/get-obs", json={"desigs":[k],"output_format":["ADES_DF"]}, timeout=120)
    print(k, r.status_code)
    try:
        j=r.json(); df=pd.DataFrame(j[0]["ADES_DF"])
        df.to_csv(f"mpc_obs_{k.replace(' ','_')}.csv",index=False)
        print(" n obs", len(df), "stations", df.stn.value_counts().head(8).to_dict())
        print(" first", df.obstime.min(), "last", df.obstime.max())
        x=df[df.stn=="X05"]; print(" X05:", len(x), x.obstime.min() if len(x) else "")
        e=df[(df.obstime>="2025-07-03")&(df.obstime<="2025-07-12")]; print(" during DP2 nights (Jul 3-12):", len(e), e.stn.value_counts().to_dict())
    except Exception as ex: print(" err", ex, r.text[:300])
EOF