#!/usr/bin/env bash
# Candidate categories: score/candidate_verdict.parquet
# Recovered verbatim from the analysis session (log line 29202); run with 'source ds42_paper/env.sh' first.
set -e
cd $DS42_ALERTS/dp2/run/score; ${PY:-python} - <<'EOF'
import pandas as pd, numpy as np
K=pd.read_parquet('template_trail_known.parquet'); kt=K.trail_ratio_med.dropna()
print("real asteroids trail-testable", len(kt), "flagged at <0.4:", int((kt<.4).sum()), "<0.6:", int((kt<.6).sum()), "<0.7:", int((kt<.7).sum()), "quantiles", kt.quantile([.001,.01,.05]).round(2).tolist())
c=pd.read_parquet('unknown3_candidates_flags.parquet'); u=pd.read_parquet('unknown3_joined.parquet'); c["n"]=u.n.reindex(c.index)
m1=pd.read_parquet('skybot_matches.parquet'); m2=pd.read_parquet('skybot_matches_r30.parquet')
known={'N00000054':'C/2025 M3','N00001028':'C/2024 G6','N00001133':'2025 OT12','N00000738':'2025 OL4','N00005436':'2025 OK382'}
poss={'N00005170':'2025 OZ','N00007367':'2025 OV28'}
c["trail7"]=c.trail_ratio_med<0.7
c["flag"]=c.any_flag2|c.trail7
c["tested"]=c.pair_ratio.notna()|c.trail_ratio_med.notna()
def cat(t,r):
    if t in known: return "1 known object"
    if t in poss: return "2 possible known"
    if r.flag: return "6 rejected (flag)"
    if r.trail_confirmed: return "3 strong: trail-confirmed"
    if r.tested: return "4 passed a test"
    return "5 untested"
c["cat"]=[cat(t,r) for t,r in c.iterrows()]
print(pd.crosstab(c.cat, c.n>=4).to_string())
s=c[c.cat.str.startswith("3")].sort_values(["n","ds42_All"],ascending=False)
d=pd.read_parquet('unknown_3plus_dets.parquet').groupby('trkSub').MJD.min()
s["date"]=pd.to_datetime(d.reindex(s.index)-40587,unit='D').dt.strftime('%m-%d %H:%M')
print(s[['n','date','rate','gal_b','ds42_All','trail_ratio_med','pair_ratio']].round(2).to_string())
c[["cat"]].to_parquet("candidate_verdict.parquet")
EOF