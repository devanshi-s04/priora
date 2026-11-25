#!/usr/bin/env bash
# Known-object annotation of candidates: score/mpcorb_candidates_annot.parquet (run after mpcorb_check.py)
# Recovered verbatim from the analysis session (log line 29828); run with 'source env.sh' first.
set -e
cd $PRIORA_ALERTS/dp2/run/score; ${PY:-python} - <<'EOF'
import pandas as pd, numpy as np
R=pd.read_parquet("mpcorb_matches.parquet"); C=R[R.set=="candidate"].copy()
v=pd.read_parquet("candidate_verdict.parquet"); u=pd.read_parquet("unknown3_joined.parquet"); c=pd.read_parquet("unknown3_candidates_flags.parquet")
C=C.join(v[["cat","neighbours"]]).join(u[["n","priora_All","rate"]])
C["m"]=pd.cut(C.sep_med,[-1,5,30,120,1e9],labels=["<5\"","5-30\"","30-120\"",">120\" / none"]); C.loc[C.sep_med.isna(),"m"]=">120\" / none"
print(pd.crosstab(C.cat,C.m).to_string())
print("\n4+ dets:", pd.crosstab(C.n>=4,C.m).to_string())
print("\nall matches <30\" (non-rejected first):")
x=C[C.sep_med<30].sort_values(["cat","sep_med"])
print(x[["cat","n","mpc_object","sep_med","sep_max","priora_All","rate"]].round(2).to_string())
print("\nstrong 23 nearest:")
s=C[C.cat.str.startswith("3")&(C.neighbours<=1)].sort_values("sep_med"); print(s[["n","mpc_object","sep_med","sep_max"]].round(1).to_string())
C.to_parquet("mpcorb_candidates_annot.parquet")
EOF