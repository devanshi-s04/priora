"""Tracklet labels for the unknown-alert test from match/match.parquet (closest catalogue orbit per detection).
A tracklet is 'labelled' if every detection matches the SAME object within SEP arcsec (10); 'mixed_known' if all match
but to different objects; 'partial' if only some detections match; 'unmatched' otherwise.
Known at alert time vs designated later: from the provisional designation's year and half-month letter."""
import numpy as np, pandas as pd
SEP = 10.0
M = pd.read_parquet("match/match.parquet")
td = pd.read_parquet("score/trk_dets.parquet")
td = td.merge(M[["detnum", "desig", "sep_arcsec", "q", "H"]], on="detnum", how="left")
td["ok"] = td.sep_arcsec < SEP
g = td.groupby("trkSub").agg(n=("detnum", "size"), nok=("ok", "sum"), ndes=("desig", lambda s: s[td.loc[s.index, "ok"]].nunique()),
                             desig=("desig", "first"), q=("q", "first"), H=("H", "first"), sepmax=("sep_arcsec", "max"))
g["status"] = np.select([(g.nok == g.n) & (g.ndes == 1), (g.nok == g.n), g.nok > 0], ["labelled", "mixed_known", "partial"], "unmatched")
g.loc[g.status != "labelled", ["desig", "q", "H"]] = np.nan
meta = pd.read_parquet("score/meta.parquet").set_index("trkSub")
g = g.join(meta[["night", "arc"]])
HM = "ABCDEFGHJKLMNOPQRSTUVWXY"                                     # half-month letters (no I, Z)
def half_month(mjd):
    t = pd.to_datetime(mjd + 2400000.5 - 2440587.5, unit="D")
    return t.year, HM[(t.month - 1) * 2 + (t.day > 15)]
def later(row):
    d = row.desig
    if not isinstance(d, str) or len(d) < 6 or not d[:4].isdigit() or d[4] != " ":
        return False
    y, L = half_month(row.night + 0.5)
    return (int(d[:4]), HM.index(d[5]) if d[5] in HM else -1) >= (y, HM.index(L))
g["designated_after"] = g.apply(later, axis=1)
g["neo"] = g.q < 1.3
g.to_parquet("match/tracklet_labels.parquet")
print(g.status.value_counts().to_dict())
L = g[g.status == "labelled"]
print(f"labelled {len(L):,} tracklets, {L.desig.nunique():,} objects; NEO {L.neo.sum()} tracklets ({L[L.neo].desig.nunique()} objects); "
      f"designated after the alert {L.designated_after.sum()} ({L[L.designated_after].neo.sum()} NEO); q>=4.5 {(L.q >= 4.5).sum()}")
print("labelled by npts", L.n.value_counts().sort_index().to_dict(), "| max sep arcsec quantiles", L.sepmax.quantile([.5, .9, .99]).round(2).tolist())
print("designation year of labelled objects", L.drop_duplicates("desig").desig.str[:4].value_counts().head(8).to_dict())
