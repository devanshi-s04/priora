"""Join ds42 (All, Unk) and digest2 (raw, NoID) scores onto meta.parquet -> scores_joined_nolabel.parquet."""
import pandas as pd
m = pd.read_parquet("meta.parquet").set_index("trkSub")
for V in ("All", "Unk"): m[f"ds42_{V}"] = pd.read_csv(f"ds42_{V}.csv").set_index("object_id").p_neo
d = pd.read_csv("digest2.csv").set_index("trkSub"); m["d2_raw"] = d.raw_neo; m["d2_noid"] = d.noid_neo
assert m[["ds42_All", "ds42_Unk", "d2_raw", "d2_noid"]].notna().all().all()
m.to_parquet("scores_joined_nolabel.parquet"); print(len(m), "tracklets scored")
