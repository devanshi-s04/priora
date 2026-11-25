"""Evaluate heliolinc + link_purify linkages (July nights) for the known-object validation and the full unknown search.
Known: purity (one designation per linkage), recovery of objects with tracklets on >= 3 July nights, fitted q vs MPCORB q.
Full: catalogue label of each linkage (all detections matched to one MPCORB object within 10"), which linkages contain the
NEO candidates (tiers from full/score/candidates_annot.csv), and uncatalogued linkages with fitted q < 1.3 AU.
-> eval_known.json, eval_full.json, linkages_full.parquet, candidate_links.csv"""
import os
import json, numpy as np, pandas as pd
T = os.environ["PRIORA_ALERTS"]
def rd(p):
    d = pd.read_csv(p); d.columns = [c.lstrip("#") for c in d.columns]; return d
def load(P, L, pairdets):
    s = rd(f"{P}_{L}_LPsum.csv"); c = rd(f"{P}_{L}_LPc2d.csv")
    s["q_fit"] = s.orbit_a * (1 - s.orbit_e); s["run"] = L
    c = c.join(pairdets[["idstring", "MJD"]], on="detnum"); c["run"] = L
    return s, c
out = {}
# ---------------- known validation ----------------
pdk = rd(f"{T}/tracklets/pairdets_v5.csv")
lab = pd.read_parquet(f"{T}/data/labels_ftransfer_2026-11-02.parquet").set_index("packed")
g = pd.read_parquet(f"{T}/tracklets/tracklet_labels_v5.parquet")
g = g[g.pure & (g.night >= 61226) & (g.night <= 61234)]
nights = g.groupby("obj").night.nunique(); recov = set(nights[nights >= 3].index)
ek = {}
for L in ("neo", "mb", "t_aa_3e5r"):
    s, c = load("known", L, pdk)
    agg = c.groupby("clusternum").agg(nobj=("idstring", "nunique"), obj=("idstring", lambda x: x.value_counts().index[0]),
                                      frac=("idstring", lambda x: x.value_counts().iloc[0] / len(x)), n=("idstring", "size"))
    s = s.join(agg, on="clusternum"); s["pure"] = s.nobj == 1
    s["q_cat"] = s.obj.map(lab.q); s["neo_cat"] = s.obj.map(lab.neo).fillna(False).astype(bool)
    P = s[s.pure]; found = set(P.obj)
    neo_r = {o for o in recov if lab.neo.get(o, False)}
    dq = (P.q_fit - P.q_cat) / P.q_cat
    ek[L] = dict(linkages=len(s), pure=int(s.pure.sum()), mostly_one=int((s.frac >= 0.9).sum()),
                 objects_found=len(found), recoverable=len(recov), recovered=len(found & recov),
                 neo_recoverable=len(neo_r), neo_recovered=len(found & neo_r),
                 neo_linkages=int((P.neo_cat).sum()), fitted_q_lt_1p3_pure=int((P.q_fit < 1.3).sum()),
                 fitted_q_lt_1p3_true_neo=int(((P.q_fit < 1.3) & P.neo_cat).sum()),
                 dq_frac_median=float(dq.abs().median()), dq_frac_p90=float(dq.abs().quantile(.9)),
                 neo_q_fit=P[P.neo_cat][["obj", "q_cat", "q_fit", "obsnights", "uniquepoints", "astromRMS"]].round(4).to_dict("records"))
    s.to_parquet(f"known_{L}_linkages.parquet")
both = set()
for L in ("neo", "mb", "t_aa_3e5r"):
    s = pd.read_parquet(f"known_{L}_linkages.parquet"); both |= set(s[s.pure].obj)
ek["union"] = dict(objects_found=len(both), recovered=len(both & recov), recoverable=len(recov),
                   neo_recovered=len(both & {o for o in recov if lab.neo.get(o, False)}), neo_recoverable=len({o for o in recov if lab.neo.get(o, False)}))
out["known"] = ek
json.dump(ek, open("eval_known.json", "w"), indent=1, default=float)
# ---------------- full search ----------------
pdf = rd(f"{T}/full/tracklets/pairdets_full.csv")
M = pd.read_parquet(f"{T}/full/match/match.parquet").set_index("detnum")      # closest catalogue orbit per tracklet detection
meta = pd.read_parquet(f"{T}/full/score/meta.parquet").reset_index().set_index("trkSub")   # keeps trk_ID (make_tracklets index)
t2d = rd(f"{T}/full/tracklets/trk2det_full.csv")
C = pd.read_csv(f"{T}/full/score/candidates_annot.csv", index_col=0)
C["iso"] = (C.gal_b.abs() >= 15) & (C.same_field == 0)
C["tier"] = np.where(C.iso & (C.priora_All >= 0.5), "A", np.where(C.iso, "B", "set aside"))
C["trk_ID"] = meta.loc[C.index, "trk_ID"].values
cand_det = t2d[t2d.trk_ID.isin(C.trk_ID)].merge(C.reset_index()[["trkSub", "trk_ID", "tier"]], on="trk_ID")
ef, allL = {}, []
for L in ("neo", "mb", "t_aa_3e5r"):
    s, c = load("full", L, pdf)
    c["desig"] = c.detnum.map(M.desig); c["sep"] = c.detnum.map(M.sep_arcsec); c["ok"] = c.sep < 10
    agg = c.groupby("clusternum").agg(n=("detnum", "size"), nok=("ok", "sum"), ndes=("desig", lambda x: x[c.loc[x.index, "ok"]].nunique()),
                                      desig=("desig", lambda x: x[c.loc[x.index, "ok"]].value_counts().index[0] if c.loc[x.index, "ok"].any() else None))
    s = s.join(agg, on="clusternum")
    s["status"] = np.select([(s.nok == s.n) & (s.ndes == 1), s.nok > 0], ["catalogued", "partly catalogued"], "uncatalogued")
    cc = c.merge(cand_det[["detnum", "trkSub", "tier"]], on="detnum")
    hit = cc.groupby("clusternum").agg(cands=("trkSub", lambda x: ",".join(sorted(set(x)))), tiers=("tier", lambda x: ",".join(sorted(set(x)))))
    s = s.join(hit, on="clusternum")
    ef[L] = dict(linkages=len(s), by_status=s.status.value_counts().to_dict(),
                 uncat_q_lt_1p3=int(((s.status == "uncatalogued") & (s.q_fit < 1.3)).sum()),
                 with_candidates=int(s.cands.notna().sum()))
    allL.append(s)
S = pd.concat(allL, ignore_index=True); S.to_parquet("linkages_full.parquet")
rows = []
for t, r in C.iterrows():
    h = S[S.cands.fillna("").str.split(",").apply(lambda v: t in v)]
    if len(h) == 0:
        rows.append(dict(trkSub=t, tier=r.tier, linked=False))
        continue
    b = h.sort_values(["obsnights", "astromRMS"], ascending=[False, True]).iloc[0]
    rows.append(dict(trkSub=t, tier=r.tier, linked=True, n_linkages=len(h), run=b.run, obsnights=int(b.obsnights), points=int(b.uniquepoints),
                     timespan=float(b.timespan), astromRMS=float(b.astromRMS), a=float(b.orbit_a), e=float(b.orbit_e), i=float(b.orbit_incl),
                     q_fit=float(b.q_fit), status=b.status, desig=b.desig))
CL = pd.DataFrame(rows); CL.to_csv("candidate_links.csv", index=False)
ef["candidates"] = {t: dict(n=int((CL.tier == t).sum()), linked=int(CL[CL.tier == t].linked.sum())) for t in ("A", "B", "set aside")}
json.dump(ef, open("eval_full.json", "w"), indent=1, default=float)
print(json.dumps(out | {"full": ef}, indent=1, default=float)[:6000])
print(CL.to_string())
