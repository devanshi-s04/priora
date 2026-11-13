"""Write HelioLinC make_tracklets input for the Fink Data Transfer of 2026-11-02 (known SSOs), July + October nights only.
Magnitude: AB mag from psfFlux (nJy): 31.4 - 2.5 log10(flux). Time: midpointMjdTai (make_tracklets -time_offset -37 -> UTC).
The designation goes in IDCOL as a hidden label (used only to score tracklet purity afterwards, never for pairing)."""
import numpy as np, pandas as pd
df = pd.read_parquet("../data/ftransfer_julyoct_labelled.parquet")
df = df[df.psfFlux > 0].sort_values("midpointMjdTai").reset_index(drop=True)
out = pd.DataFrame({"index": np.arange(len(df)), "ObjID": df.packed_primary_provisional_designation.values,
                    "MJD_TAI": df.midpointMjdTai.values, "RA": df.ra.values, "Dec": df.dec.values,
                    "mag": 31.4 - 2.5 * np.log10(df.psfFlux.values), "band": df.band.values, "obscode": "X05",
                    "diaSourceId": df.diaSourceId.values})
with open("dets_julyoct.csv", "w") as f:
    f.write("#" + ",".join(out.columns) + "\n"); out.to_csv(f, header=False, index=False, float_format="%.8f")
open("colformat_fink.txt", "w").write("IDCOL 2\nMJDCOL 3\nRACOL 4\nDECCOL 5\nMAGCOL 6\nBANDCOL 7\nOBSCODECOL 8\n")
print(len(out), "detections written; dropped non-positive flux:", int((pd.read_parquet('../data/ftransfer_julyoct_labelled.parquet').psfFlux <= 0).sum()))
