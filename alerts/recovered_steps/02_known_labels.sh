#!/usr/bin/env bash
# MPCORB labels for the Rubin-associated objects: data/labels_ftransfer_2026-11-02.parquet
# Recovered verbatim from the analysis session (log line 18859); run with 'source env.sh' first.
set -e
cd $PRIORA_ALERTS/data && ${PY:-python} - <<'EOF'
import gzip, json, pandas as pd, numpy as np
recs=json.load(gzip.open("mpcorb_extended.json.gz"))
rows=[]; alias={}
for r in recs:
    p=r.get("Principal_desig")
    rows.append((p, r.get("Number"), r.get("Name"), r.get("Perihelion_dist"), r.get("e"), r.get("i"), r.get("a"), r.get("H"), r.get("Num_opps"), r.get("Orbit_type")))
    for o in (r.get("Other_desigs") or []): alias[o]=p
M=pd.DataFrame(rows,columns=["desig","number","name","q","e","i","a","H","n_opp","orbit_type"]).drop_duplicates("desig").set_index("desig")
df=pd.read_parquet("ftransfer_2026-11-02_all.parquet",columns=["packed_primary_provisional_designation","unpacked_primary_provisional_designation"]).drop_duplicates("packed_primary_provisional_designation")
u=df.unpacked_primary_provisional_designation
direct=u.isin(M.index); via=~direct & u.map(lambda x: alias.get(x)).isin(M.index)
print("objects",len(u),"matched principal",direct.sum(),"via other desig",via.sum(),"unmatched",(~direct&~via).sum())
key=np.where(direct,u,u.map(lambda x: alias.get(x)))
L=M.reindex(key).reset_index(drop=True); L.insert(0,"packed",df.packed_primary_provisional_designation.values); L.insert(1,"unpacked",u.values)
L["neo"]=L.q<1.3
print("NEOs (q<1.3):",int(L.neo.sum()),"| q>=4.5:",int((L.q>=4.5).sum()),"| no orbit:",int(L.q.isna().sum()))
print("unmatched examples:", list(u[~direct&~via][:8]))
L.to_parquet("labels_ftransfer_2026-11-02.parquet")
EOF