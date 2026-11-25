#!/usr/bin/env bash
# Rubin's own MPC submissions (X05) on the DP2 nights 60858-60866: dp2/b612_x05_dp2nights.parquet (input to label_b612.py)
# Source: Asteroid Institute public export of the MPC obs_sbn table for X05 (downloaded 2026-11-03), saved as data_b612/:
#   https://storage.googleapis.com/asteroid-institute-public/production/rubin/mpc/obs_sbn/full/parquet/obs_sbn_X05_full.parquet
# Recovered verbatim from the analysis session (log line 26439); run with 'source env.sh' first.
set -e
cd $PRIORA_ALERTS && ${PY:-python} -I - <<'EOF2'
import pyarrow.parquet as pq, pandas as pd, numpy as np
t=pq.read_table("data_b612/obs_sbn_X05_full.parquet",columns=["id","trksub","status","permid","provid","obstime","ra","dec","mag","band","created_at","disc"]).to_pandas()
t["mjd"]=(t.obstime-pd.Timestamp("1858-11-17")).dt.total_seconds()/86400; t["night"]=np.floor(t.mjd-0.5).astype(int)
c=t[(t.night>=60858)&(t.night<=60866)].copy()
print("X05 obs on DP2 nights 60858-60866:", len(c), c.night.value_counts().sort_index().to_dict())
print("status:", c.status.value_counts().to_dict(), "| with provid:", c.provid.notna().sum(), "| permid:", c.permid.notna().sum(), "| trksubs:", c.trksub.nunique(), "| distinct objects:", c.provid.nunique())
print("submitted (created_at) range:", c.created_at.min(), c.created_at.max())
c["ra"]=c.ra.astype(float); c["dec"]=c.dec.astype(float); c.to_parquet("dp2/b612_x05_dp2nights.parquet")
EOF2
