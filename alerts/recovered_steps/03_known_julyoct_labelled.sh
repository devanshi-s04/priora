#!/usr/bin/env bash
# July+October known detections with labels: data/ftransfer_julyoct_labelled.parquet
# Recovered verbatim from the analysis session (log line 18875); run with 'source ds42_paper/env.sh' first.
set -e
cd $DS42_ALERTS/data && ${PY:-python} - <<'EOF'
import pandas as pd, numpy as np
from astropy.time import Time
df=pd.read_parquet("ftransfer_2026-11-02_all.parquet")
L=pd.read_parquet("labels_ftransfer_2026-11-02.parquet").set_index("packed")
df["dayobs"]=[t[:10] for t in Time(np.floor(df.midpointMjdTai.values-0.5)+0.5, format="mjd", scale="tai").utc.iso]
df=df[df.dayobs.str.slice(0,7).isin(["2026-07","2026-10"])].copy()
df["neo"]=df.packed_primary_provisional_designation.map(L.neo); df["q"]=df.packed_primary_provisional_designation.map(L.q)
on=df.groupby(["packed_primary_provisional_designation","dayobs"]).agg(n=("ra","size"),t0=("midpointMjdTai","min"),t1=("midpointMjdTai","max"),neo=("neo","first"),q=("q","first")).reset_index()
on["arc"]=(on.t1-on.t0)*1440
for lab,m in [("all >=2 dets",on.n>=2),(">=2 dets, arc>=15 min",(on.n>=2)&(on.arc>=15))]:
    s=on[m]; print(f"{lab:24s} object-nights {len(s):6d}  NEO {int(s.neo.sum()):4d}  q>=4.5 {int((s.q>=4.5).sum()):4d}  | July NEO {int(s[s.dayobs<'2026-08'].neo.sum())}  Oct NEO {int(s[s.dayobs>'2026-08'].neo.sum())}")
print("NEO objects seen:", df[df.neo==True].packed_primary_provisional_designation.nunique())
df.to_parquet("ftransfer_julyoct_labelled.parquet")
EOF