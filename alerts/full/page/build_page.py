"""Builds lsst-full-unknown-alerts.html from body_template.html + ../score/candidates_annot.csv, with the known-test page's styles."""
import os
import pandas as pd
from pathlib import Path
C = pd.read_csv("../score/candidates_annot.csv", index_col=0).sort_values("ds42_All", ascending=False)
def flag(r):
    f = []
    if abs(r.gal_b) < 15: f.append("near plane")
    if r.same_field >= 2: f.append(f"group of {int(r.same_field) + 1}")
    if r.rate < 0.05: f.append("very slow")
    return ", ".join(f) or "isolated"
rows = []
for t, r in C.iterrows():
    cls = (["hi"] if r.ds42_All >= 0.5 else []) + (["clean"] if abs(r.gal_b) >= 15 and r.same_field == 0 else [])
    rows.append(f'<tr class="{" ".join(cls)}"><td>{t}</td><td>{r.utc}</td><td>{r.ra:.4f}</td><td>{r.dec:+.4f}</td><td>{r.gal_b:+.0f}</td><td>{int(r.n)}</td>'
                f'<td>{r.rate:.2f}</td><td>{r.V:.1f}</td><td>{r.ds42_All:.3f}</td><td>{r.ds42_Unk:.3f}</td><td>{r.d2_raw:.1f}</td><td>{r.d2_noid:.1f}</td>'
                f'<td>{"yes" if r.in_bnew else "new"}</td><td>{flag(r)}</td><td>{r.nearest_sep_arcsec:.0f}″</td></tr>\n')
body = Path("body_template.html").read_text().replace("{{ROWS}}", "".join(rows))
old = Path(os.path.join(os.environ["DS42_ALERTS"], "page/lsst-alerts-test.html")).read_text()
head = old.split("</style>")[0].replace("<title>LSST Alerts Test</title>", "<title>LSST Full Unknown Alerts</title>", 1)
head += ".cand td,.cand th{padding:6px 9px}.cand tr.hi td{background:color-mix(in srgb,var(--neo) 10%,transparent)}.cand tr.clean td:first-child{box-shadow:inset 3px 0 0 var(--done)}\n</style>\n"
Path("lsst-full-unknown-alerts.html").write_text(head + body); print("ok", len(C), "rows")
