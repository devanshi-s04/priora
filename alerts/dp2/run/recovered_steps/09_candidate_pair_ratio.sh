#!/usr/bin/env bash
# Unknown 3+ candidates with pair-motion ratio and galactic latitude: score/unknown3_candidates.parquet (run after analyze_unknown3.py)
# Recovered verbatim from the analysis session (log line 27338); run with 'source ds42_paper/env.sh' first.
set -e
cd $DS42_ALERTS/dp2/run && ${PY:-python} - <<'EOF'
import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord; import astropy.units as u
m=pd.read_parquet("score/unknown3_joined.parquet"); C=m[(m.label=="unlabelled")&(m.ds42_All>=.1)].copy()
d=pd.read_parquet("score/unknown_3plus_dets.parquet"); d=d[d.trkSub.isin(C.index)].sort_values(["trkSub","MJD"])
def pair(g):
    t,ra,de=g.MJD.values,g.RA.values,g.Dec.values; c=np.cos(np.radians(de[0])); rate=np.hypot(((ra[-1]-ra[0]+180)%360-180)*c,de[-1]-de[0])/(t[-1]-t[0]); r=[]
    for i in range(len(t)-1):
        dt=t[i+1]-t[i]
        if dt*1440<5 and rate*dt*3600>=0.5: r.append(np.hypot(((ra[i+1]-ra[i]+180)%360-180)*c,de[i+1]-de[i])*3600/(rate*dt*3600))
    return min(r) if r else np.nan
C["pair_ratio"]=d.groupby("trkSub")[["MJD","RA","Dec"]].apply(pair)
f=d.groupby("trkSub").first(); C["ra"]=f.RA; C["dec"]=f.Dec; C["gal_b"]=SkyCoord(C.ra.values*u.deg,C.dec.values*u.deg).galactic.b.deg
print("candidates", len(C), "| pair-static (<0.5):", int((C.pair_ratio<0.5).sum()), "| testable:", int(C.pair_ratio.notna().sum()))
print("|b|<15:", int((C.gal_b.abs()<15).sum()), "| rate deg/day quantiles", C.rate.quantile([.1,.5,.9]).round(2).tolist(), "| nights", C.night.value_counts().sort_index().to_dict())
cl=C[~(C.pair_ratio<0.5)]; print("after pair test:", len(cl), "| of which |b|>=15:", int((cl.gal_b.abs()>=15).sum()), "| P>=0.5:", int((cl.ds42_All>=.5).sum()))
C.to_parquet("score/unknown3_candidates.parquet")
EOF