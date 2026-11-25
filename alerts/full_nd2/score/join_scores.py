"""Join priora (All, Unk) and digest2 (raw, NoID) scores onto meta.parquet -> scores_joined_nolabel.parquet."""
import pandas as pd
m = pd.read_parquet("meta.parquet").set_index("trkSub")
for V in ("All", "Unk"): m[f"priora_{V}"] = pd.read_csv(f"priora_{V}.csv").set_index("object_id").p_neo
d = pd.read_csv("digest2.csv").set_index("trkSub"); m["d2_raw"] = d.raw_neo; m["d2_noid"] = d.noid_neo
assert m[["priora_All", "priora_Unk", "d2_raw", "d2_noid"]].notna().all().all()
m.to_parquet("scores_joined_nolabel.parquet"); print(len(m), "tracklets scored")
