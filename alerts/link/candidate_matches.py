"""One 'matched to' record per NEO candidate (all 73 kept): nearest known object from SkyBoT (120" cone) and JPL sb_ident,
with a motion comparison; Rubin's own MPC submissions (B612 export, exposure-midpoint match); HelioLinC linking.
-> candidate_matches.csv"""
import json, os, numpy as np, pandas as pd
T = os.environ["DS42_ALERTS"]
C = pd.read_csv(f"{T}/full/score/candidates_annot.csv", index_col=0)
iso = (C.gal_b.abs() >= 15) & (C.same_field == 0)
C["tier"] = np.where(iso & (C.ds42_All >= 0.5), "A", np.where(iso, "B", "set aside"))
S = pd.read_csv("dbcheck_summary.csv", index_col=0)
td = pd.read_parquet(f"{T}/full/score/trk_dets.parquet").sort_values("MJD")
B = pd.read_csv("b612_mid_matches.csv", dtype={"diaSourceId": "string"}).dropna(subset=["diaSourceId"])
t2d = pd.read_csv(f"{T}/full/tracklets/trk2det_full.csv"); t2d.columns = ["trk_ID", "detnum"]
pdf = pd.read_csv(f"{T}/full/tracklets/pairdets_full.csv"); pdf.columns = [c.lstrip("#") for c in pdf.columns]
meta = pd.read_parquet(f"{T}/full/score/meta.parquet").reset_index()
t2d["diaSourceId"] = pdf.idstring.values[t2d.detnum.values].astype(str)
cb = t2d.merge(meta[["trk_ID", "trkSub"]], on="trk_ID").merge(B[["diaSourceId", "trksub", "provid", "status", "created"]], on="diaSourceId")
CL = pd.read_csv("candidate_links.csv").set_index("trkSub")
raw_link = {"N0005470", "N0005481"}   # appear only in mixed heliolinc linkages rejected by link_purify (see page)
rows = []
for t, r in C.iterrows():
    s = td[td.trkSub == t]; dt = (s.MJD.iloc[-1] - s.MJD.iloc[0]) * 24
    vra = ((s.RA.iloc[-1] - s.RA.iloc[0] + 180) % 360 - 180) * np.cos(np.radians(s.Dec.iloc[0])) * 3600 / dt; vde = (s.Dec.iloc[-1] - s.Dec.iloc[0]) * 3600 / dt
    sk = "none within 120″"; motion = ""
    f = f"dbcheck/skybot_{t}.json"
    D = pd.DataFrame(json.loads(open(f).read() or "[]")) if os.path.exists(f) else pd.DataFrame()
    if len(D):
        b = D.sort_values("d (arcsec)").iloc[0]; pv = np.array([b["dRA (arcsec/h)"], b["dDEC (arcsec/h)"]]); ov = np.array([vra, vde])
        ang = np.degrees(np.arccos(np.clip(pv @ ov / (np.linalg.norm(pv) * np.linalg.norm(ov) + 1e-9), -1, 1))); ratio = np.linalg.norm(pv) / (np.linalg.norm(ov) + 1e-9)
        same = ang < 15 and 0.7 < ratio < 1.4
        motion = "motion matches" if same else f"motion differs ({ang:.0f}° off, speed ×{ratio:.2f})"
        sk = f"{b['Name']} ({b['Class']}) at {b['d (arcsec)']:.0f}″, {motion}"
    jp = S.loc[t, "jpl_nearest"] if t in S.index else None
    jtxt = f"{jp} at {S.loc[t, 'jpl_sep']:.0f}″" if isinstance(jp, str) and S.loc[t, "jpl_sep"] < 120 else "none within 120″"
    rb = cb[cb.trkSub == t]
    rtxt = (f"Rubin tracklet {rb.trksub.iloc[0]} ({'unlinked, isolated tracklet file' if rb.status.iloc[0] == 'I' else rb.provid.iloc[0]}), submitted {str(rb.created.iloc[0])[:10]}"
            if len(rb) else "not among Rubin's MPC submissions")
    ltxt = "only in mixed linkages rejected by link_purify" if t in raw_link else ("linked" if CL.loc[t, "linked"] else "no multi-night linkage")
    summary = []
    if motion == "motion matches": summary.append(f"known object {sk.split(' (')[0]}")
    if len(rb): summary.append(f"Rubin MPC submission {rb.trksub.iloc[0]}" + (" (unlinked)" if rb.status.iloc[0] == "I" else ""))
    rows.append(dict(trkSub=t, tier=C.loc[t, "tier"], skybot=sk, jpl=jtxt, rubin_mpc=rtxt, linking=ltxt, matched_to="; ".join(summary) or "nothing"))
R = pd.DataFrame(rows).set_index("trkSub"); R.to_csv("candidate_matches.csv")
print(R.matched_to.value_counts().to_dict()); print(R[R.matched_to != "nothing"].to_string())
