#!/usr/bin/env bash
# Adds same-field neighbour counts to score/candidate_verdict.parquet
# Recovered verbatim from the analysis session (log line 29216); run with 'source env.sh' first.
set -e
cd $PRIORA_ALERTS/dp2/run/score; ${PY:-python} - <<'EOF'
import pandas as pd, numpy as np
v=pd.read_parquet('candidate_verdict.parquet'); c=pd.read_parquet('unknown3_candidates_flags.parquet'); u=pd.read_parquet('unknown3_joined.parquet')
d=pd.read_parquet('unknown_3plus_dets.parquet').sort_values('MJD').groupby('trkSub').first()
x=v.join(d[['MJD','RA','Dec']]).join(u[['n']])
# same-field, same-time: other candidates (any category) whose first detection is within 1 deg and 30 min
allc=pd.read_parquet('unknown3_candidates.parquet').join(d[['MJD','RA','Dec']])
def crowd(r):
    s=((allc.MJD-r.MJD).abs()<30/1440)&(np.hypot((allc.RA-r.RA)*np.cos(np.radians(r.Dec)),allc.Dec-r.Dec)<1.0)
    return int(s.sum())-1
x["neighbours"]=[crowd(r) for _,r in x.iterrows()]
s=x[x.cat.str.startswith(("3","4","5"))]
for k in ("3","4","5"):
    t=s[s.cat.str.startswith(k)]; print(k, len(t), "isolated (0-1 neighbours):", int((t.neighbours<=1).sum()), "crowded (>=2):", int((t.neighbours>=2).sum()), "| 4+ isolated:", int(((t.neighbours<=1)&(t.n>=4)).sum()))
st=x[x.cat.str.startswith("3")&(x.neighbours<=1)].join(c[['gal_b','priora_All','trail_ratio_med','rate']]).sort_values(['n','priora_All'],ascending=False)
print(st[['n','neighbours','rate','gal_b','priora_All','trail_ratio_med']].round(2).to_string())
x.to_parquet('candidate_verdict.parquet')
EOF