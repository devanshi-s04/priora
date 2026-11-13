"""Flatten the Fink Data Transfer of 2026-11-03 (unknown alerts: b_is_new, NOT b_is_solar_system, NOT is_solar_system)
to one row per diaSource with the columns the test needs -> data_unk/ftransfer_2026-11-03_flat.parquet."""
import glob, numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, pyarrow as pa
fs = sorted(glob.glob("data_unk/ftransfer_2026-11-03/ftransfer_lsst_2026-11-03_596070/*.parquet"))
D = ["diaSourceId", "visit", "detector", "diaObjectId", "ssObjectId", "midpointMjdTai", "ra", "dec", "raErr", "decErr",
     "snr", "psfFlux", "psfFluxErr", "band", "reliability", "extendedness", "trailLength", "isDipole", "isNegative",
     "pixelFlags", "pixelFlags_cr", "pixelFlags_streak", "pixelFlags_saturated", "glint_trail", "centroid_flag", "psfFlux_flag"]
cols = {k: pc.field("diaSource", k) for k in D}
cols |= {"ss_designation": pc.field("ssSource", "designation"), "mpc_designation": pc.field("mpc_orbits", "designation"),
         "obj_nDiaSources": pc.field("diaObject", "nDiaSources"), "obj_first": pc.field("diaObject", "firstDiaSourceMjdTai")}
t = ds.dataset(fs).to_table(columns=cols)
df = t.to_pandas(); df.to_parquet("data_unk/ftransfer_2026-11-03_flat.parquet")
print(len(df), "rows")
