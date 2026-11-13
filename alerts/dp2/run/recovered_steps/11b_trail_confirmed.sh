#!/usr/bin/env bash
# Adds trail_confirmed (and related columns) to score/unknown3_candidates_flags.parquet; run after 11_template_trail_flags.sh
# Recovered verbatim from the analysis session (log line 27428); run with 'source ds42_paper/env.sh' first.
set -e
cd $DS42_ALERTS/dp2/run && ${PY:-python} - <<'EOF'
import pandas as pd
c=pd.read_parquet("score/unknown3_candidates_flags.parquet"); um=pd.read_parquet("score/unknown3_joined.parquet")
cl=c[~c.any_flag2]
tc=cl[(cl.n_fast>=2)&(cl.trail_ratio_med.between(0.6,1.6))]
print("clean", len(cl), "| trail-confirmed (2+ fast dets, ratio 0.6-1.6):", len(tc), "| |b|>=15:", int((tc.gal_b.abs()>=15).sum()), "| P>=0.5:", int((tc.ds42_All>=.5).sum()))
print("trail-confirmed rate quantiles", um.loc[tc.index,"rate"].quantile([.1,.5,.9]).round(2).tolist(), "| d2 raw >=65:", int((um.loc[tc.index,"d2_raw"]>=65).sum()), "| nights", um.loc[tc.index,"night"].value_counts().sort_index().to_dict())
c["trail_confirmed"]=c.index.isin(tc.index); c.to_parquet("score/unknown3_candidates_flags.parquet")
EOF