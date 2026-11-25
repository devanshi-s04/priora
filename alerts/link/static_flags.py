"""Static-source flags for tracklets, tested on real asteroids before use.
pair test: for consecutive detections < 5 min apart, compare measured motion with rate x dt (rate = whole-tracklet rate).
  static pair = predicted >= 0.5" and measured < 0.25 x predicted.
history test: any detection on a Rubin diaObject with nDiaSources > 1 (full pull only; SSO alerts carry no diaObject).
Applied to: full-pull NEO candidates (73), full-pull catalogue-labelled tracklets (real asteroids missed by Rubin),
known-object pure tracklets (Rubin-associated asteroids). -> link/static_flags.csv"""
import os
import numpy as np, pandas as pd
def pair_flag(td):
    td = td.sort_values(["trk", "t"]); out = {}
    for k, g in td.groupby("trk"):
        t, ra, de = g.t.values, g.ra.values, g.dec.values
        if len(t) < 2: out[k] = (False, np.nan); continue
        cosd = np.cos(np.radians(de[0]))
        rate = np.hypot(((ra[-1] - ra[0] + 180) % 360 - 180) * cosd, de[-1] - de[0]) / (t[-1] - t[0])      # deg/day
        flags, ratios = [], []
        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            if dt * 1440 >= 5: continue
            pred = rate * dt * 3600; meas = np.hypot(((ra[i + 1] - ra[i] + 180) % 360 - 180) * cosd, de[i + 1] - de[i]) * 3600
            if pred >= 0.5: flags.append(meas < 0.25 * pred); ratios.append(meas / pred)
        out[k] = (any(flags), min(ratios) if ratios else np.nan)
    return out
T = os.environ["PRIORA_ALERTS"]
# full pull: candidates + labelled
m = pd.read_parquet(f"{T}/full/score/scores_joined.parquet")
td = pd.read_parquet(f"{T}/full/score/trk_dets.parquet"); td["diaSourceId"] = td.idstring.astype("int64")
D = pd.read_parquet(f"{T}/full/tracklets/dets_full_r90.parquet")[["diaSourceId", "nDiaSources"]]
td = td.merge(D, on="diaSourceId", how="left")
C = pd.read_csv(f"{T}/full/score/candidates_annot.csv", index_col=0)
sel = m.index[(m.index.isin(C.index)) | (m.status == "labelled")]
x = td[td.trkSub.isin(sel)].rename(columns={"trkSub": "trk", "MJD": "t", "RA": "ra", "Dec": "dec"})
pf = pair_flag(x)
hist = x.groupby("trk").nDiaSources.agg(lambda s: int((s > 1).sum()))
F = pd.DataFrame({"pair_static": {k: v[0] for k, v in pf.items()}, "min_motion_ratio": {k: v[1] for k, v in pf.items()}, "n_dets_on_object_with_history": hist})
F["group"] = np.where(F.index.isin(C.index), "candidate", "labelled (real, full pull)")
F["neo"] = F.index.map(m.neo).fillna(False).astype(bool); F["n"] = F.index.map(m.n)
# known pure tracklets
g = pd.read_parquet(f"{T}/tracklets/tracklet_labels_v5.parquet"); g = g[g.pure & (g.arc >= 15)]
pdk = pd.read_csv(f"{T}/tracklets/pairdets_v5.csv"); pdk.columns = [c.lstrip("#") for c in pdk.columns]
t2k = pd.read_csv(f"{T}/tracklets/trk2det_v5.csv"); t2k.columns = ["trk_ID", "detnum"]
t2k = t2k[t2k.trk_ID.isin(g.index)].join(pdk[["MJD", "RA", "Dec"]], on="detnum").rename(columns={"trk_ID": "trk", "MJD": "t", "RA": "ra", "Dec": "dec"})
pk = pair_flag(t2k)
K = pd.DataFrame({"pair_static": {k: v[0] for k, v in pk.items()}, "min_motion_ratio": {k: v[1] for k, v in pk.items()}})
K["group"] = "known (Rubin-associated)"; K["neo"] = g.neo.reindex(K.index).fillna(False).values; K["n"] = g.n.reindex(K.index).values; K["n_dets_on_object_with_history"] = np.nan
A = pd.concat([F, K]); A.to_csv(f"{T}/link/static_flags.csv")
for grp, G in A.groupby("group"):
    G3 = G[G.n >= 3]
    print(f"{grp}: tracklets {len(G):,} (3+: {len(G3):,}) | pair-static {int(G.pair_static.sum())} ({G.pair_static.mean():.2%}); 3+: {int(G3.pair_static.sum())}"
          + ("" if grp.startswith("known") else f" | any det on object with history {int((G.n_dets_on_object_with_history > 0).sum())} ({(G.n_dets_on_object_with_history > 0).mean():.2%})")
          + f" | NEOs flagged by pair test {int((G.neo & G.pair_static).sum())} of {int(G.neo.sum())}")
c = A[A.group == "candidate"].join(C[["priora_All"]]); 
iso = (C.gal_b.abs() >= 15) & (C.same_field == 0); tier = pd.Series(np.where(iso & (C.priora_All >= 0.5), "A", np.where(iso, "B", "set aside")), index=C.index)
c["tier"] = tier
pd.set_option("display.width", 200); print(c[["tier", "n", "pair_static", "min_motion_ratio", "n_dets_on_object_with_history"]].sort_values(["tier", "pair_static"]).round(3).to_string())
