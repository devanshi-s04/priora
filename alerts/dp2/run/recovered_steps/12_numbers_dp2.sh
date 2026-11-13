#!/usr/bin/env bash
# Counts for the DP2 page and paper: dp2/page/numbers_dp2.json
# Recovered verbatim from the analysis session (log line 27498); run with 'source ds42_paper/env.sh' first.
set -e
mkdir -p $DS42_ALERTS/dp2/page/img && cd $DS42_ALERTS/dp2/run && ${PY:-python} - <<'EOF'
import json, pandas as pd
k = json.load(open("score/known_analysis.json")); u = json.load(open("score/unknown3_analysis.json"))
c = pd.read_parquet("score/unknown3_candidates_flags.parquet")
K = pd.read_parquet("score/template_trail_known.parquet"); km = pd.read_parquet("score/known_joined.parquet")
um = pd.read_parquet("score/unknown3_joined.parquet"); lab = um.index[um.label == "labelled"]; U = pd.read_parquet("score/template_trail_unknown.parquet")
neo = set(km.index[km.neo])
N = dict(known=k, unknown3=u,
  tests=dict(known_n=len(K), tmpl2_real=int(K.template_static.sum()), tmpl2_real_neo=int((K.template_static & K.index.isin(neo)).sum()),
             tmplall_real=int((K.n_tmpl == K.n).sum()), tmplall_real_neo=int(((K.n_tmpl == K.n) & K.index.isin(neo)).sum()),
             trail_real=int(K.trail_fake.sum()), trail_testable_real=int(K.trail_ratio_med.notna().sum()),
             trail_q=[float(x) for x in K.trail_ratio_med.quantile([.01, .5, .99]).round(2)],
             tmplall_unklab=int((U.loc[lab].n_tmpl == U.loc[lab].n).sum()), unklab=len(lab)),
  cand=dict(n=len(c), pair=int(c.pair_static.sum()), pair_testable=int(c.pair_ratio.notna().sum()), tmplall=int(c.template_all.sum()),
            trail=int(c.trail_fake.sum()), trail_testable=int(c.trail_ratio_med.notna().sum()), any=int(c.any_flag2.sum()), clean=int((~c.any_flag2).sum()),
            clean_b15=int(((~c.any_flag2) & (c.gal_b.abs() >= 15)).sum()), trail_conf=int(c.trail_confirmed.sum()),
            trail_conf_b15=int((c.trail_confirmed & (c.gal_b.abs() >= 15)).sum())))
json.dump(N, open("../page/numbers_dp2.json", "w"), indent=1, default=float); print("ok", N["tests"], N["cand"])
EOF