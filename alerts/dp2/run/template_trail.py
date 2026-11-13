"""DP2 static / fake-mover tests from catalogue columns (no images needed):
template test: templateFlux / templateFluxErr > 5 (a source in the template) at >= 2 of a tracklet's detections.
trail test:   expected trail = tracklet rate x exposure time; for detections with expected >= 1.5" (fast movers), compare
              with the measured trailLength. A real fast mover is trailed; a chance alignment of point sources is not.
              Flag when the median measured/expected over those detections is < 0.4.
Thresholds checked on REAL asteroids: known-set pure tracklets with 3+ detections, and unknown-set tracklets later labelled from
Rubin's MPC submissions. -> score/template_trail_{known,unknown}.parquet; prints false-flag rates and candidate results."""
import numpy as np, pandas as pd
V = pd.read_parquet("../visits.parquet")[["visit", "expTime"]].drop_duplicates("visit").set_index("visit").expTime
COLS = ["diaSourceId", "visit", "midpointMjdTai", "ra", "dec", "templateFlux", "templateFluxErr", "trailLength", "snr"]
def per_tracklet(kind, stem, keep):
    det = pd.read_parquet(f"{kind}/dets.parquet", columns=COLS)
    td = pd.read_parquet(f"score/{stem}_dets.parquet"); td = td[td.trkSub.isin(keep)]
    if kind == "unknown":
        td["diaSourceId"] = td.idstring.astype("int64"); td = td.merge(det, on="diaSourceId", how="left")
    else:   # hidden ID is the designation: join on time and position
        det["k"] = (det.midpointMjdTai.round(6).astype(str) + "_" + det.ra.round(5).astype(str) + "_" + det.dec.round(5).astype(str))
        td["k"] = (td.MJD.round(6).astype(str) + "_" + td.RA.round(5).astype(str) + "_" + td.Dec.round(5).astype(str))
        td = td.merge(det.drop_duplicates("k"), on="k", how="left")
    td["exp"] = td.visit.map(V); td["tsnr"] = td.templateFlux / td.templateFluxErr
    g = td.sort_values(["trkSub", "MJD"]).groupby("trkSub")
    a = g.agg(ra0=("RA", "first"), ra1=("RA", "last"), de0=("Dec", "first"), de1=("Dec", "last"), t0=("MJD", "first"), t1=("MJD", "last"),
              n_tmpl=("tsnr", lambda s: int((s > 5).sum())), n=("tsnr", "size"), joined=("visit", "count"))
    a["rate"] = np.hypot(((a.ra1 - a.ra0 + 180) % 360 - 180) * np.cos(np.radians(a.de0)), a.de1 - a.de0) / (a.t1 - a.t0)   # deg/day
    td["exp_trail"] = td.trkSub.map(a.rate) * td.exp / 24.0                                                              # arcsec
    td["ratio"] = np.where(td.exp_trail >= 1.5, td.trailLength / td.exp_trail, np.nan)
    a["trail_ratio_med"] = td.groupby("trkSub").ratio.median(); a["n_fast"] = td.groupby("trkSub").ratio.count()
    a["template_static"] = a.n_tmpl >= 2; a["trail_fake"] = a.trail_ratio_med < 0.4
    return a.drop(columns=["ra0", "ra1", "de0", "de1", "t0", "t1"])
km = pd.read_parquet("score/known_joined.parquet"); K3 = km[km.pure & (km.n >= 3)]
K = per_tracklet("known", "known_all", K3.index)
um = pd.read_parquet("score/unknown3_joined.parquet"); C = pd.read_parquet("score/unknown3_candidates.parquet")
U = per_tracklet("unknown", "unknown_3plus", um.index)
K.to_parquet("score/template_trail_known.parquet"); U.to_parquet("score/template_trail_unknown.parquet")
print(f"known pure 3+: {len(K):,} (detections joined {int(K.joined.sum()):,}/{int(K.n.sum()):,})")
for name, col in (("template (2+ dets)", "template_static"), ("trail (median ratio < 0.4)", "trail_fake")):
    kneo = K[col] & K.index.isin(K3.index[K3.neo])
    print(f"  {name}: flags {int(K[col].sum()):,} of {len(K):,} real ({K[col].mean():.2%}); NEOs flagged {int(kneo.sum())} of {int(K3.neo.sum())}; testable {int(K.trail_ratio_med.notna().sum()) if 'trail' in name else len(K):,}")
lab = um[um.label == "labelled"].index
for name, col in (("template", "template_static"), ("trail", "trail_fake")):
    print(f"unknown labelled real ({len(lab):,}): {name} flags {int(U.loc[lab, col].sum())}")
print("measured/expected trail on known fast movers: quantiles", K.trail_ratio_med.quantile([.01, .05, .5, .95]).round(2).tolist())
c = U.loc[C.index].join(C[["pair_ratio", "gal_b", "ds42_All"]])
c["pair_static"] = c.pair_ratio < 0.5
c["any_flag"] = c.pair_static | c.template_static | c.trail_fake
print(f"\ncandidates {len(c)}: pair {int(c.pair_static.sum())}, template {int(c.template_static.sum())}, trail {int(c.trail_fake.sum())} (testable {int(c.trail_ratio_med.notna().sum())}); any {int(c.any_flag.sum())}; clean {int((~c.any_flag).sum())}; clean & |b|>=15 {int(((~c.any_flag) & (c.gal_b.abs() >= 15)).sum())}; clean & P>=0.5 {int(((~c.any_flag) & (c.ds42_All >= .5)).sum())}")
c.to_parquet("score/unknown3_candidates_flags.parquet")
