#!/usr/bin/env bash
# Counts for the rerun (nDiaSources<=2) sample: paper_section/numbers_nd2.json
# Recovered verbatim from the analysis session (log line 26933); run with 'source env.sh' first.
set -e
cd $PRIORA_ALERTS && ${PY:-python} - <<'EOF'
import json, pandas as pd, numpy as np
F="full_nd2"
m=pd.read_parquet(f"{F}/score/scores_joined.parquet"); allj=pd.read_parquet(f"{F}/score/scores_joined_nolabel_all.parquet")
t=pd.read_csv(f"{F}/tracklets/tracklets_full.csv"); meta=pd.read_parquet(f"{F}/score/meta.parquet")
D=pd.read_parquet(f"{F}/tracklets/dets_full_r90.parquet"); D0=pd.read_parquet("full/tracklets/dets_full_r90.parquet")
C=pd.read_csv(f"{F}/score/candidates_annot.csv",index_col=0)
s3=m[m.n>=3]
N=dict(dets_before_nd=len(D0), dets=len(D), nd_removed=len(D0)-len(D), tracklets=len(t), arc15=len(meta), stationary=len(allj)-len(m), moving=len(m),
       n2=int((m.n==2).sum()), n3=int((m.n==3).sum()), n4=int((m.n>=4).sum()),
       hiP=int((m.priora_All>=.5).sum()), hiD=int((m.d2_raw>=65).sum()), hiP_and_hiD=int(((m.priora_All>=.5)&(m.d2_raw>=65)).sum()),
       s3=len(s3), s3_hiP=int((s3.priora_All>=.5).sum()), s3_hiD=int((s3.d2_raw>=65).sum()), s3_p01=int((s3.priora_All>=.1).sum()),
       cand=len(C), cand_p05=int((C.priora_All>=.5).sum()), static=int((C.static_flag!="no static signature").sum()), clean=int((C.static_flag=="no static signature").sum()),
       clean_A=int(((C.static_flag=="no static signature")&(C.tier=="A")).sum()), clean_B=int(((C.static_flag=="no static signature")&(C.tier=="B")).sum()),
       clean_S=int(((C.static_flag=="no static signature")&(C.tier=="set aside")).sum()), tierA=int((C.tier=="A").sum()), tierB=int((C.tier=="B").sum()), tierS=int((C.tier=="set aside").sum()),
       matched=int((C.matched_to!="nothing").sum()))
a=json.load(open(f"{F}/score/analysis.json")); L=a["labelled"]
N.update(lab=L["n"], lab_neo=L["n_neo"], lab_sumP=round(L["sumP"]["priora_All"],1), d2_fa=L["flags"]["d2_raw>=65"]["non_neo"], d2_fa_dist=L["flags"]["d2_raw>=65"]["distant"],
         ds_fa=L["flags"]["priora_All>=0.1"]["non_neo"], ds_fa_dist=L["flags"]["priora_All>=0.1"]["distant"])
json.dump(N, open("paper_section/numbers_nd2.json","w"), indent=1); print(N)
EOF