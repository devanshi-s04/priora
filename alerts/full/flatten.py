"""Combine the full Fink Data Transfer ftransfer_lsst_2026-11-03_913305 (no filters, no blocks; 24 flat columns, see
data_all/datatransfer_20261103_064021.yml) into one table -> data_all/ftransfer_913305.parquet.
designation / packed_primary_provisional_designation = Rubin's known-object match; nDiaSources from diaObject."""
import glob, pyarrow.dataset as ds
fs = sorted(glob.glob("data_all/ftransfer_lsst_2026-11-03_913305/*.parquet"))
t = ds.dataset(fs).to_table(); t.to_pandas().to_parquet("data_all/ftransfer_913305.parquet"); print(t.num_rows, "rows", t.num_columns, "columns")
