#!/usr/bin/env bash
# Multi-night linking summary: link_nd2/summary.csv
# Recovered verbatim from the analysis session (log line 27177); run with 'source ds42_paper/env.sh' first.
set -e
cd $DS42_ALERTS/link_nd2 && cat run.log && ${PY:-python} - <<'EOF'
import pandas as pd, numpy as np
T="$DS42_ALERTS"
meta=pd.read_parquet(f"{T}/full_nd2/score/meta.parquet").reset_index().set_index("trkSub")
t2d=pd.read_csv(f"{T}/full_nd2/tracklets/trk2det_full.csv"); t2d.columns=["trk_ID","detnum"]
C=pd.read_csv(f"{T}/full_nd2/score/candidates_annot.csv",index_col=0)
cd=t2d[t2d.trk_ID.isin(meta.loc[C.index,"trk_ID"])].merge(meta.reset_index()[["trkSub","trk_ID"]],on="trk_ID")
M=pd.read_parquet(f"{T}/full_nd2/match/match.parquet").set_index("detnum")
rows=[]
for L in ("neo","mb","t_aa_3e5r"):
    lp=pd.read_csv(f"full_{L}_LPc2d.csv"); lp.columns=["cl","detnum"]; raw=pd.read_csv(f"full_{L}_c2d.csv"); raw.columns=["cl","detnum"]
    s=pd.read_csv(f"full_{L}_LPsum.csv"); s.columns=[c.lstrip("#") for c in s.columns]; s["q"]=s.orbit_a*(1-s.orbit_e)
    lp["ok"]=lp.detnum.map(M.sep_arcsec)<10; cat=lp.groupby("cl").ok.all()
    unc=s[s.clusternum.map(lambda c: not cat.get(c,False))]
    rows.append(dict(run=L, linkages=len(s), catalogued=int(cat.sum()), uncat=len(unc), uncat_q13=int((unc.q<1.3).sum()), uncat_q13_bound=int(((unc.q<1.3)&(unc.orbit_e<1)&(unc.orbit_a>0)).sum()),
                     cand_final=lp.merge(cd,on="detnum").trkSub.nunique(), cand_raw=raw.merge(cd,on="detnum").trkSub.nunique()))
R=pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv("summary.csv",index=False)
EOF