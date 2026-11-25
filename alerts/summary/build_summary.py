"""LSST alerts tests overview page, numbers from ../summary_table.json and the full-pull candidate list."""
import os
import json, pandas as pd
from pathlib import Path
S = json.loads(Path("../summary_table.json").read_text())
C = pd.read_csv("../full_nd2/score/candidates_annot.csv", index_col=0)   # rerun: nDiaSources <= 2 before linking
NN = json.loads(Path("../paper_section/numbers_nd2.json").read_text())
iso = C[(C.gal_b.abs() >= 15) & (C.same_field == 0)]
U = {"known": "https://claude.ai/artifact/LYdeun36bZ8MH1qfDonBoR", "unk": "https://claude.ai/artifact/J7SxiHg7WosbwdYYgEUQ2T",
     "full": "https://claude.ai/artifact/3iBdZafWdmyPz9fZtkNyLz"}
NAME = {"known": "Known objects", "unk": "Unknown, new Rubin objects", "full": "All unknown detections"}
def row(t, k):
    s = S[f"{t}_{k}"]; r, d1, d5 = s["d2raw"], s["priora_01"], s["priora_05"]
    return (f'<tr><td>{NAME[t]}</td><td>{k}+</td><td>{s["n"]:,}</td><td class="neo">{s["neo"]}</td><td>{s["sumP_All"]:.1f}</td><td>{s["sumP_Unk"]:.1f}</td>'
            f'<td>{r[0]} / {r[2]} / {r[3]}</td><td>{d1[0]} / {d1[2]} / {d1[3]}</td><td>{d5[0]} / {d5[2]}</td><td>{s["auc_All"]:.4f}</td><td>{s["auc_raw"]:.4f}</td></tr>')
rows = "".join(row(t, k) for t in ("known", "unk", "full") for k in (2, 3, 4))
def chance(t):
    s2 = S[f"{t}_2"]; return s2["other"], s2["other_flag_priora"], s2["other_flag_d2raw"]
kc, uc, fc = chance("known"), chance("unk"), chance("full")
# candidate tiers (full pull): A = isolated, |b| >= 15, P >= 0.5; B = isolated, |b| >= 15, 0.1 <= P < 0.5; rest set aside
C["iso"] = (C.gal_b.abs() >= 15) & (C.same_field == 0)
A = C[C.iso & (C.priora_All >= 0.5)].sort_values(["n", "priora_All"], ascending=[False, False])
B = C[C.iso & (C.priora_All < 0.5)].sort_values(["n", "priora_All"], ascending=[False, False])
X = C[~C.iso]; X4 = X[X.n >= 4].sort_values("priora_All", ascending=False)
MTc = C[["matched_to"]]
SFc = C[["static_flag"]]
def note(r):
    f = []
    mt = MTc.loc[r.name, "matched_to"] if r.name in MTc.index else "nothing"
    if mt != "nothing": f.append(f"<b>matched to: {mt}</b>")
    sf = SFc.loc[r.name, "static_flag"] if r.name in SFc.index else "no static signature"
    if sf != "no static signature": f.append(f"<b>{sf}</b>")
    if r.rate < 0.05: f.append("very slow: possibly stationary")
    if pd.isna(r.nearest_sep_arcsec): f.append("no catalogued object within 10′")
    if abs(r.gal_b) < 15: f.append("near plane")
    if r.same_field >= 1: f.append(f"{int(r.same_field) + 1} candidates in the same field")
    f.append(f"first-pull ID {r.old_id}" if isinstance(r.old_id, str) and r.old_id not in ("None", "nan") else "new in this run")
    return "; ".join(f)
def ctab(D):
    return "".join(f'<tr><td>{t}</td><td>{r.utc}</td><td>{r.ra:.4f}</td><td>{r.dec:+.4f}</td><td>{r.gal_b:+.0f}</td><td>{"<b>4+</b>" if r.n >= 4 else int(r.n)}</td>'
                   f'<td>{r.rate:.2f}</td><td>{r.V:.1f}</td><td>{r.priora_All:.3f}</td><td>{r.priora_Unk:.3f}</td><td>{r.d2_raw:.1f}</td><td>{r.d2_noid:.1f}</td><td>{note(r)}</td></tr>'
                   for t, r in D.iterrows())
CH = "<thead><tr><th>tracklet</th><th>first detection (UTC)</th><th>RA (°)</th><th>Dec (°)</th><th>gal. b</th><th>dets</th><th>rate (°/day)</th><th>V</th><th>priora All</th><th>priora Unk</th><th>digest2 raw</th><th>digest2 NoID</th><th>notes</th></tr></thead>"
k3 = S["known_3"]
old = Path(os.path.join(os.environ["PRIORA_ALERTS"], "page/lsst-alerts-test.html")).read_text()
head = old.split("</style>")[0].replace("<title>LSST Alerts Test</title>", "<title>LSST Alerts Tests Overview</title>", 1)
head += """.tests{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,250px),1fr));gap:12px}
.test{background:var(--sheet);border:1px solid var(--rule);border-radius:8px;padding:14px 16px;display:flex;flex-direction:column;gap:6px;min-width:0}
.test h3 a{color:var(--accent);text-decoration:none} .test h3 a:hover{text-decoration:underline}
.test p{font-size:14px;color:var(--ink-2)} .test .q{font:500 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
.flow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font:500 12.5px var(--mono);color:var(--ink-2)}
.cand td,.cand th{padding:6px 9px}.cand td:last-child{white-space:normal;min-width:14em;text-align:left}
.flow span{background:var(--sheet);border:1px solid var(--rule);border-radius:5px;padding:4px 8px}
</style>
"""
body = f'''<main>
<header>
  <p class="kicker">priora paper · real-data tests · Rubin alerts, July and October 2026</p>
  <h1>LSST alerts tests overview</h1>
  <p class="lede">Three tests of priora against digest2 on real Rubin alerts, and how they fit together. Each test has its own page with every table and plot; this page holds the shared results and the next steps.</p>
</header>

<section>
  <h2>The pipeline (same in all three)</h2>
  <div class="flow"><span>Fink alerts</span>→<span>cuts</span>→<span>HelioLinC make_tracklets, maxvel 5 °/day</span>→<span>arc ≥ 15 min</span>→<span>priora All, Unk · digest2 raw, NoID</span>→<span>labels</span>→<span>evaluation</span></div>
  <p>Nights: July 5–13 and October 29–30, 2026. Only the source of the alerts and of the labels changes between tests.</p>
</section>

<section>
  <h2>The three tests</h2>
  <div class="tests">
    <div class="test"><span class="q">1 · calibration on known objects</span><h3><a href="{U["known"]}">LSST alerts test</a></h3>
      <p>Alerts Rubin matched to known asteroids (291,305). Labels from the MPC catalogue. Answers: is priora calibrated on real data, and does it beat digest2?</p></div>
    <div class="test"><span class="q">2 · first look at unknowns</span><h3><a href="{U["unk"]}">LSST unknown alerts test</a></h3>
      <p>Alerts on new Rubin objects with no known match (Fink <code>b_is_new</code>, 2,141,791). Labels from matching today's MPC catalogue. Answers: how do the codes do on objects unknown at alert time?</p></div>
    <div class="test"><span class="q">3 · every unknown detection</span><h3><a href="{U["full"]}">LSST full unknown alerts</a></h3>
      <p>Every alert with nothing filtered at Fink (3,789,349); known objects removed by us. Adds 100,985 clean detections that <code>b_is_new</code> dropped. Answers: what NEO candidates does the full stream give?</p></div>
  </div>
  <ul>
    <li><b>1 → 2:</b> the known test fixes the settings (maxvel 5, arc ≥ 15 min, reliability threshold) and shows priora All is calibrated; test 2 applies them to unknowns.</li>
    <li><b>2 → 3:</b> test 3 contains all of test 2 (its 190,192 detections reproduce exactly) plus the repeat-object detections. Its evaluation numbers barely move; the candidate list grows.</li>
  </ul>
</section>

<section>
  <h2>Results side by side</h2>
  <p>Evaluation set: pure tracklets (test 1) or catalogue-labelled tracklets (tests 2 and 3). Flag columns: flagged / non-NEOs / of those with q ≥ 4.5 AU.</p>
  <div class="tbl"><table>
    <thead><tr><th>test</th><th>detections</th><th>tracklets</th><th>NEOs</th><th>priora All predicts</th><th>priora Unk predicts</th><th>digest2 raw ≥ 65</th><th>priora All P ≥ 0.1</th><th>priora All P ≥ 0.5</th><th>AUC priora All</th><th>AUC digest2 raw</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</section>

<section>
  <h2>What holds across all three</h2>
  <ul>
    <li><b>priora with the All prior is calibrated on real data.</b> Predicted vs observed NEOs: {S["known_2"]["sumP_All"]:.1f} vs {S["known_2"]["neo"]} (known), {S["unk_2"]["sumP_All"]:.1f} vs {S["unk_2"]["neo"]} (unknown), and {S["known_3"]["sumP_All"]:.1f} vs {S["known_3"]["neo"]} on known 3+ tracklets.</li>
    <li><b>digest2's false alarms are the distant objects; priora has none of them.</b> digest2 raw ≥ 65 flags {S["known_2"]["d2raw"][3]} distant objects in the known test and {S["unk_2"]["d2raw"][3]} in the unknown tests; priora All at P ≥ 0.1 flags none in any test.</li>
    <li><b>The Unk prior over-predicts on known objects</b> ({S["known_2"]["sumP_Unk"]:.0f} vs {S["known_2"]["neo"]}) but is close on the unknowns ({S["unk_2"]["sumP_Unk"]:.1f} vs {S["unk_2"]["neo"]}), as a prior for undiscovered objects should be.</li>
    <li><b>2-detection tracklets are dominated by chance pairs.</b> Unlabelled or mixed 2+ tracklets flagged by priora All (P ≥ 0.5): {kc[1]:,} of {kc[0]:,} (known), {uc[1]:,} of {uc[0]:,} (unknown), {fc[1]:,} of {fc[0]:,} (full); digest2 flags as many. Neither code can catch a mis-link. With 3+ detections almost all of them go away.</li>
    <li><b>With 3+ or 4+ detections both codes rank NEOs almost perfectly;</b> the difference is in false alarms. Known 3+: digest2 raw {S["known_3"]["d2raw"][2]} non-NEOs, priora All (P ≥ 0.1) {S["known_3"]["priora_01"][2]}. But the NEO counts there are small ({S["known_3"]["neo"]}, {S["unk_3"]["neo"]}, {S["full_3"]["neo"]}).</li>
  </ul>
</section>

<section id="candidates">
  <h2>NEO candidates</h2>
  <p>From the <a href="{U["full"]}">full unknown pull</a>, rerun with nDiaSources ≤ 2 before linking: tracklets with 3 or more detections that move, match no catalogued object, and have priora All P ≥ 0.1. <b>{len(C)}</b> in all, {int((C.n >= 4).sum())} with 4+ detections. Ranked by how likely they are to be real. For scale: on known objects with 3+ detections, priora flagged {k3["priora_05"][0]} tracklets at P ≥ 0.5 and all {k3["priora_05"][1]} were NEOs.</p>
  <h3>Tier A: isolated, |b| ≥ 15°, priora P ≥ 0.5 ({len(A)})</h3>
  <div class="tbl"><table class="cand">{CH}<tbody>{ctab(A)}</tbody></table></div>
  <h3>Tier B: isolated, |b| ≥ 15°, 0.1 ≤ priora P &lt; 0.5 ({len(B)})</h3>
  <div class="tbl"><table class="cand">{CH}<tbody>{ctab(B)}</tbody></table></div>
  <h3>Set aside: crowded or grouped ({len(X)})</h3>
  <p>Within 15° of the Galactic plane or sharing a field and minute with other candidates, including {int((C.utc.str.startswith("2026-07-13 05:06")).sum())} in one bulge field on July 13 at 05:06. Most are probably chance alignments of unrelated detections. The {len(X4)} with 4+ detections, worth a second look:</p>
  <div class="tbl"><table class="cand">{CH}<tbody>{ctab(X4)}</tbody></table></div>
  <ul>
    <li><b>Rerun with nDiaSources ≤ 2.</b> Detections on Rubin objects with more than 2 detections (static and variable sources) are now removed before linking: {NN["nd_removed"]:,} of {NN["dets_before_nd"]:,} unknown detections, but only 0.05% of real asteroid detections. The labelled evaluation is unchanged ({NN["lab"]:,} tracklets, {NN["lab_neo"]} NEOs, priora All predicts {NN["lab_sumP"]}); the candidate list drops from 73 to {NN["cand"]}. Candidate IDs changed; each row's first-pull ID is in the hand-off CSV and the <a href="https://claude.ai/artifact/UQaBaBVMD43HEMjNZCeqSm">cutout gallery</a>.</li>
    <li><b>Static-source checks</b>, both validated on real asteroids: back-to-back detections that move less than half the expected amount, and a source in the template image at 2+ detections. They flag {NN["static"]} of the {NN["cand"]}; <b>{NN["clean"]} have no static signature: tier A {NN["clean_A"]}, tier B {NN["clean_B"]}, set aside {NN["clean_S"]}.</b></li>
    <li><b>Matches:</b> SkyBoT and JPL find no known object within 10″ of any candidate. Two candidates match something and stay on the list: one is the known Kuiper-belt object 2025 NB54 (46–73″ from its predicted position, matching motion and brightness), and one is Rubin's own unlinked MPC submission RL021CV (B612 export).</li>
    <li><b>None is confirmed.</b> All are single-night tracklets. HelioLinC finds no multi-night linkage for any of them, as expected: only 3 of 73 known July NEOs had tracklets on 3+ nights (<a href="https://claude.ai/artifact/8XSa7kdDS3WnyuW7iA63si">candidate linking page</a>).</li>
    <li><b>To confirm:</b> image cutouts for tiers A and B, then link across nights with HelioLinC. A candidate that joins an orbit with q &lt; 1.3 AU and is still not catalogued is a missed-NEO candidate.</li>
  </ul>
  <p class="src">Full list: $PRIORA_ALERTS/handoff/lsst_neo_candidates_nd2_detections.csv and the cutout gallery; earlier 73-candidate list: $PRIORA_ALERTS/full/score/candidates_annot.csv.</p>
</section>

<section>
  <h2>Lessons and gotchas</h2>
  <ul>
    <li>Fink names objects by provisional designation; label with MPCORB's principal designation, not the NEA list.</li>
    <li>Rubin's own association misses catalogued objects: about 3,100 tracklets in the unknown tests belong to objects in the MPC catalogue with good orbits.</li>
    <li>Repeat-object alerts include stationary variables, which make_tracklets pairs with themselves; cut tracklets moving less than 0.5″.</li>
    <li>Crowded fields near the Galactic plane produce chance alignments of 3 detections; 16 candidates in one bulge field on July 13.</li>
    <li>Rubin's observations of these nights are not at the MPC yet, so unknown-object labels come from today's catalogue, not from MPC linking.</li>
    <li>NEOCP check: neither digest2 score reproduces the NEOCP's displayed Score from one snapshot; two papers say posting uses NoID ≥ 65.</li>
  </ul>
</section>

<section>
  <h2>Next steps</h2>
  <ol>
    <li><b>Multi-night linking.</b> Link the unknown tracklets across the July nights with HelioLinC (heliolinc), using the full set. A candidate that joins an orbit across 2–3 nights is real; chance pairs and alignments will not link. This also rescues real 2-detection tracklets.</li>
    <li><b>Cutouts for the isolated candidates.</b> {len(iso)} of the full set's {len(C)} candidates are isolated and at |b| ≥ 15°; check their image cutouts first.</li>
    <li><b>Missed NEOs.</b> Any linked object with q &lt; 1.3 AU that is still not in the MPC catalogue (and not in the NEOCP history or the Isolated Tracklet File) is a missed-NEO candidate for Mario.</li>
    <li><b>Relabel later.</b> Rerun the catalogue match and MPC observation match in a few weeks, once Rubin's observations of these nights reach the MPC. That grows the labelled unknown sample and the number of NEOs.</li>
  </ol>
  <p class="src">Numbers: $PRIORA_ALERTS/summary_table.py → summary_table.json; page: summary/build_summary.py.</p>
</section>
</main>
'''
Path("lsst-alerts-overview.html").write_text(head + body); print("ok")
