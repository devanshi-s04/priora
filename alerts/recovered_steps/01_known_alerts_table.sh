#!/usr/bin/env bash
# Known-object alert table: data/ftransfer_2026-11-02_all.parquet from the Fink transfer ftransfer_lsst_2026-11-02_838512
# Recovered verbatim from the analysis session (log line 18748); run with 'source ds42_paper/env.sh' first.
set -e
cd $DS42_ALERTS/data/ftransfer_2026-11-02 && find . -maxdepth 2 ! -name "*.parquet" | head; ${PY:-python} - <<'EOF'
import pyarrow.parquet as pq, pyarrow as pa, glob, numpy as np, pandas as pd
from astropy.time import Time
fs=sorted(glob.glob("ftransfer_lsst_2026-11-02_838512/*.parquet"))
df=pd.concat([pq.read_table(f).to_pandas() for f in fs], ignore_index=True)
print("rows",len(df),"unique diaSourceId",df.diaSourceId.nunique(),"objects",df.packed_primary_provisional_designation.nunique())
t=Time(df.midpointMjdTai.values,format="mjd",scale="tai").utc
df["month"]=[x[:7] for x in t.iso]
print(df.month.value_counts().sort_index())
print("MJD range", df.midpointMjdTai.min(), df.midpointMjdTai.max())
print(df[["band","reliability","snr","diaDistanceRank"]].describe(include="all").T.head(8))
df.to_parquet("../ftransfer_2026-11-02_all.parquet")
EOF