#!/usr/bin/env bash
# MPC observation histories for the remaining objects (network; fetched 2026-11-05): ../precovery_fig/mpc_obs_*.csv
# Recovered verbatim from the analysis session (log line 32642); run with 'source ds42_paper/env.sh' first.
set -e
cd $DS42_ALERTS/dp2/precovery_fig && ${PY:-python} - <<'EOF'
import requests, pandas as pd, os
for k in ("2025 NA", "2025 ML93", "2017 EC3", "2025 QO13", "2025 OL4", "2025 OT12", "2025 OZ"):
    f = f"mpc_obs_{k.replace(' ', '_')}.csv"
    if not os.path.exists(f):
        r = requests.get("https://data.minorplanetcenter.net/api/get-obs", json={"desigs": [k], "output_format": ["ADES_DF"]}, timeout=120)
        pd.DataFrame(r.json()[0]["ADES_DF"]).to_csv(f, index=False)
    d = pd.read_csv(f); t = pd.to_datetime(d.obstime, format="ISO8601")
    w = (t >= "2025-07-03") & (t < "2025-07-12")
    print(f"{k:10s} n={len(d):4d} first={t.min():%Y-%m-%d} last={t.max():%Y-%m-%d} X05={int((d.stn=='X05').sum())} | during DP2 window: {int(w.sum())} {d[w].stn.value_counts().to_dict()} | 2025 obs dates: {sorted(set(t[t.dt.year>=2025].dt.strftime('%m-%d')))[:12]}")
EOF