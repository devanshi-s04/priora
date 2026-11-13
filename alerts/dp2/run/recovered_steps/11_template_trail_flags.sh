#!/usr/bin/env bash
# Adds template_all / any_flag2 / trail flags to score/unknown3_candidates_flags.parquet (run after template_trail.py)
# Recovered verbatim from the analysis session (log line 27413); run with 'source ds42_paper/env.sh' first.
set -e
cd $DS42_ALERTS/dp2/run && ${PY:-python} - <<'EOF'
import pandas as pd, numpy as np
K=pd.read_parquet("score/template_trail_known.parquet"); U=pd.read_parquet("score/template_trail_unknown.parquet")
km=pd.read_parquet("score/known_joined.parquet"); um=pd.read_parquet("score/unknown3_joined.parquet"); c=pd.read_parquet("score/unknown3_candidates_flags.parquet")
neo=set(km.index[km.neo]); lab=um.index[um.label=="labelled"]
for thr in (5,10,20):
    pass
for rule,fk,fu,fc in (("all dets template SNR>5", K.n_tmpl==K.n, U.loc[lab].n_tmpl==U.loc[lab].n, c.n_tmpl==c.n),):
    print(rule, "| known real", int(fk.sum()), f"({fk.mean():.2%})", "NEO", int((fk & K.index.isin(neo)).sum()), "| unknown labelled real", int(fu.sum()), f"({fu.mean():.2%})", "| candidates", int(fc.sum()))
# stronger SNR on all dets
det=pd.read_parquet("unknown/dets.parquet",columns=["diaSourceId","templateFlux","templateFluxErr"])
print("known: fraction of detections with template SNR>5:", "n/a here")
c["template_all"]=c.n_tmpl==c.n
c["any_flag2"]=c.pair_static|c.template_all|c.trail_fake
print("candidates: pair", int(c.pair_static.sum()), "| template at all dets", int(c.template_all.sum()), "| trail", int(c.trail_fake.sum()), "| any", int(c.any_flag2.sum()), "| clean", int((~c.any_flag2).sum()), "| clean & |b|>=15", int(((~c.any_flag2)&(c.gal_b.abs()>=15)).sum()), "| clean & P>=0.5", int(((~c.any_flag2)&(c.ds42_All>=.5)).sum()))
print("clean by night:", c[~c.any_flag2].index.map(um.night).value_counts().sort_index().to_dict(), "| clean rate quantiles", um.loc[c.index[~c.any_flag2],"rate"].quantile([.1,.5,.9]).round(2).tolist(), "| trail testable among clean", int(c[~c.any_flag2].trail_ratio_med.notna().sum()))
c.to_parquet("score/unknown3_candidates_flags.parquet")
EOF