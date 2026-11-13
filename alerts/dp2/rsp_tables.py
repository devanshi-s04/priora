"""Connect to the Rubin Science Platform TAP service with a personal token (read from ~/.rsp-tap-token, never printed) and
list the DP2 tables and the columns of the DIA / solar-system tables. Usage: python3 rsp_tables.py"""
import os, sys, pathlib, requests, pyvo
TOK = pathlib.Path.home() / ".rsp-tap-token"
if not TOK.exists(): sys.exit(f"No token file at {TOK}. Create one at https://data.lsst.cloud (Settings → Access tokens, scope read:tap).")
if oct(TOK.stat().st_mode)[-3:] not in ("600", "400"): print(f"warning: {TOK} is readable by others; run: chmod 600 {TOK}")
s = requests.Session(); s.headers["Authorization"] = "Bearer " + TOK.read_text().strip()
tap = pyvo.dal.TAPService("https://data.lsst.cloud/api/tap", session=s)
t = tap.run_sync("SELECT schema_name, table_name FROM tap_schema.tables ORDER BY schema_name, table_name").to_table()
schemas = sorted(set(t["schema_name"])); print("schemas:", schemas)
dp2 = [r["table_name"] for r in t if "dp2" in str(r["schema_name"]).lower()]
print("DP2 tables:", dp2)
for want in ("DiaSource", "DiaObject", "SSSource", "SSObject", "MPCORB", "Visit"):
    hits = [x for x in dp2 if x.split(".")[-1].lower() == want.lower()]
    for h in hits:
        c = tap.run_sync(f"SELECT column_name, datatype, unit FROM tap_schema.columns WHERE table_name = '{h}'").to_table()
        print(f"\n{h}: {len(c)} columns"); print(", ".join(c["column_name"]))
