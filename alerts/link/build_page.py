"""'LSST Candidate Linking' page: HelioLinC multi-night linking (validation on known objects, search on full unknown set)
and known-object database checks (JPL sb_ident, IMCCE SkyBoT) for the 73 full-pull NEO candidates."""
import os
import json, numpy as np, pandas as pd
from pathlib import Path
T = Path(os.environ["PRIORA_ALERTS"]); L = T / "link"
EK = json.loads((L / "eval_known.json").read_text()); EF = json.loads((L / "eval_full.json").read_text())
lab = pd.read_parquet(T / "data/labels_ftransfer_2026-11-02.parquet").set_index("packed")
pdk = pd.read_csv(T / "tracklets/pairdets_v5.csv")
def known_run(run):
    c = pd.read_csv(L / f"known_{run}_LPc2d.csv"); c.columns = ["cl", "det"]; c["obj"] = pdk.idstring.values[c.det.values]
    s = pd.read_csv(L / f"known_{run}_LPsum.csv"); s.columns = [x.lstrip("#") for x in s.columns]
    g = c.groupby("cl").agg(nobj=("obj", "nunique"), obj=("obj", lambda x: x.value_counts().index[0]))
    s = s.join(g, on="clusternum"); s["q_fit"] = s.orbit_a * (1 - s.orbit_e); s["q_cat"] = s.obj.map(lab.q); P = s[s.nobj == 1]
    return dict(n=len(s), pure=len(P), objs=P.obj.nunique(), dq=float(((P.q_fit - P.q_cat).abs() / P.q_cat).median()),
                qlt=int((P.q_fit < 1.3).sum()), qlt_true=int(((P.q_fit < 1.3) & (P.q_cat < 1.3)).sum()))
KR = {r: known_run(r) for r in ("neo", "mb", "t_aa_3e5r")}
g = pd.read_parquet(T / "tracklets/tracklet_labels_v5.parquet"); g = g[g.pure & (g.night >= 61226) & (g.night <= 61234)]
nn = g.groupby("obj").night.nunique(); isneo = nn.index.map(lambda o: bool(lab.neo.get(o, False)))
cov_all = nn.value_counts().sort_index().to_dict(); cov_neo = nn[isneo].value_counts().sort_index().to_dict()
M = pd.read_parquet(L / "linkages_full.parquet")
U = M[(M.run == "t_aa_3e5r") & (M.status == "uncatalogued") & (M.q_fit < 1.3)]
S = pd.read_csv(L / "dbcheck_summary.csv", index_col=0)
RUN = {"neo": "NEO grid, default", "mb": "main-belt grid, default", "t_aa_3e5r": "NEO grid, relaxed"}
krow = "".join(f"<tr><td>{RUN[r]}</td><td>{v['n']:,}</td><td>{v['pure'] / v['n']:.1%}</td><td>{v['objs']:,} of {EK['neo']['recoverable']:,}</td>"
               f"<td>{EK[r]['neo_recovered']} of {EK[r]['neo_recoverable']}</td><td>{v['dq']:.2%}</td><td>{v['qlt']} ({v['qlt_true']} real NEO)</td></tr>" for r, v in KR.items())
frow = "".join(f"<tr><td>{RUN[r]}</td><td>{EF[r]['linkages']}</td><td>{EF[r]['by_status'].get('catalogued', 0)}</td><td>{EF[r]['by_status'].get('uncatalogued', 0)}</td>"
               f"<td>{EF[r]['uncat_q_lt_1p3']}</td><td>{EF[r]['with_candidates']}</td></tr>" for r in ("neo", "mb", "t_aa_3e5r"))
urow = "".join(f"<tr><td>{int(r.uniquepoints)}</td><td>{int(r.obsnights)}</td><td>{r.timespan:.1f}</td><td>{r.astromRMS:.2f}</td><td>{r.orbit_a:.2f}</td><td>{r.orbit_e:.2f}</td><td>{r.q_fit:.2f}</td></tr>" for _, r in U.iterrows())
def fmt(x, f="{:.0f}"):
    return "–" if pd.isna(x) else f.format(x)
MT = pd.read_csv(L / "candidate_matches.csv", index_col=0)      # every candidate kept; this only says what it matched
B6 = pd.read_csv(L / "b612_trksubs.csv", index_col=0)
X05 = pd.read_parquet(T / "data_b612/x05_our_nights.parquet"); n_x05 = len(X05); n_notrk = int(X05.trksub.isna().sum())
def verdict(r):
    return MT.loc[r.name, "matched_to"]
drow = "".join(f'<tr class="{"hi" if r.tier == "A" else ""}"><td>{t}</td><td>{r.tier}</td><td>{r.rate:.2f}</td><td>{r.V:.1f}</td>'
               f'<td>{r.skybot_nearest if not pd.isna(r.skybot_sep) else "none"}</td><td>{fmt(r.skybot_sep)}</td><td>{fmt(r.skybot_rate, "{:.2f}")}</td>'
               f'<td>{r.jpl_nearest if isinstance(r.jpl_nearest, str) else "none"}</td><td>{fmt(r.jpl_sep)}</td><td>{fmt(r.jpl_rate, "{:.2f}")}</td><td>{MT.loc[t, "rubin_mpc"]}</td><td><b>{verdict(r)}</b></td></tr>'
               for t, r in S.iterrows())
n_sky = int((S.skybot_sep < 10).sum()); n_jpl = int((S.jpl_sep < 10).sum()); n_jpl_done = int((S.jpl_status == "ok").sum())
neo_cells = "".join(f'<td class="neo">{cov_neo.get(k, 0)}</td>' for k in sorted(cov_all))
old = (T / "page/lsst-alerts-test.html").read_text()
head = old.split("</style>")[0].replace("<title>LSST Alerts Test</title>", "<title>LSST Candidate Linking</title>", 1)
head += ".cand td,.cand th{padding:6px 9px}.cand tr.hi td{background:color-mix(in srgb,var(--neo) 10%,transparent)}.cand td:last-child{white-space:normal;min-width:16em;text-align:left}\n</style>\n"
body = f'''<main>
<header>
  <p class="kicker">priora paper · real-data test · follow-up of the NEO candidates · 2026-11-03</p>
  <h1>LSST candidate linking</h1>
  <p class="lede">Can the NEO candidates from the <a href="https://claude.ai/artifact/3iBdZafWdmyPz9fZtkNyLz">full unknown pull</a> be confirmed? Two checks: HelioLinC multi-night linking across the July nights, and the known-object databases at JPL and IMCCE. Part of the <a href="https://claude.ai/artifact/XndqoKKVFWJTqxAZhjXvAH">LSST alerts tests overview</a>.</p>
</header>

<section class="callout">
  <h3>Answer so far</h3>
  <p><b>No candidate links across nights. Two match something, and both stay on the list:</b> N0004194 (Tier B) is the known distant object 2025 NB54, and N0003116 (set aside) is Rubin's own unlinked MPC submission RL021CV. The other {len(S) - 2} match no known object in SkyBoT or JPL and are not among Rubin's MPC submissions. Not linking does not rule them out: in July only {cov_neo.get(3, 0) + cov_neo.get(4, 0)} of {sum(cov_neo.values())} known NEOs have tracklets on 3 or more nights, which is what linking needs.</p>
</section>

<section id="coverage">
  <h2>Why linking can only see a few objects</h2>
  <p>Nights with a tracklet in July (dayObs July 5–13, 8 nights with data), for known objects in the known-object test:</p>
  <div class="tbl"><table>
    <thead><tr><th>nights with a tracklet</th>{"".join(f"<th>{k}</th>" for k in sorted(cov_all))}</tr></thead>
    <tbody>
      <tr><td>all known objects</td>{"".join(f"<td>{cov_all[k]:,}</td>" for k in sorted(cov_all))}</tr>
      <tr><td>known NEOs</td>{neo_cells}</tr>
    </tbody>
  </table></div>
  <p>HelioLinC needs tracklets on at least 3 nights. 59 of the 73 July NEOs were seen on one night only. October has 2 nights and cannot be linked at all.</p>
</section>

<section id="validation">
  <h2>Validation on known objects</h2>
  <p>heliolinc_omp + link_purify on the 247,424 known-object July tracklets (stationary ones removed), reference MJD 61230.2. Grids from heliolinx-aux: NEO <code>hihyp00aa_neo</code> (1,884 hypotheses), main belt <code>hihyp01b_mb</code> (324). Relaxed: clustering radius 3×10⁵ km, link_purify <code>-maxrms 4e5 -max_astrom_rms 2</code>.</p>
  <div class="tbl"><table>
    <thead><tr><th>run</th><th>linkages</th><th>pure</th><th>objects recovered (3+ nights)</th><th>NEOs recovered</th><th>median |Δq|/q</th><th>fitted q &lt; 1.3 AU</th></tr></thead>
    <tbody>{krow}</tbody>
  </table></div>
  <ul>
    <li>Linkages are essentially always one object, and fitted orbits are good to about 0.2% in q.</li>
    <li>The 3 NEOs with 3+ nights are hard cases: mostly 2 detections per night, on 3 nights spread over 5–7 days. Default settings recover none; relaxed settings recover K08W00L with fitted q 0.970 AU against the catalogue's 0.972, at the cost of one false NEO-like orbit.</li>
    <li>All three runs together recover {EK["union"]["recovered"]:,} of the {EK["union"]["recoverable"]:,} objects with 3+ nights.</li>
  </ul>
</section>

<section id="search">
  <h2>Search on the full unknown set</h2>
  <p>Same runs on the 261,776 July tracklets of the full unknown pull (moving only).</p>
  <div class="tbl"><table>
    <thead><tr><th>run</th><th>linkages</th><th>catalogued object</th><th>uncatalogued</th><th>uncatalogued, fitted q &lt; 1.3</th><th>containing a candidate</th></tr></thead>
    <tbody>{frow}</tbody>
  </table></div>
  <ul>
    <li><b>No candidate is in any final linkage</b> (Tier A 0 of 9, Tier B 0 of 8, set aside 0 of 56). Before link_purify, 2 set-aside candidates (N0005470, N0005481) appear only in large mixed linkages of 15–39 detections, which link_purify rightly rejects.</li>
    <li><b>The {len(U)} uncatalogued NEO-like linkages are spurious:</b> all have unbound orbits (negative a, e &gt; 1) and residuals of 0.9–2″.</li>
    <li>The other uncatalogued linkages have bound orbits with q &gt; 1.3 AU: candidate new non-NEO objects (not checked further here).</li>
  </ul>
  <div class="tbl"><table>
    <thead><tr><th>points</th><th>nights</th><th>span (d)</th><th>RMS (″)</th><th>a (AU)</th><th>e</th><th>q (AU)</th></tr></thead>
    <tbody>{urow}</tbody>
  </table></div>
</section>

<section id="databases">
  <h2>Known-object databases</h2>
  <p>Each candidate's first detection, observatory X05, searched with IMCCE SkyBoT (cone 120″) and JPL's Small-Body Identification API (two-pass, 0.2° box). JPL answered for {n_jpl_done} of {len(S)}. Within 10″: SkyBoT {n_sky}, JPL {n_jpl}. Tier A shaded.</p>
  <div class="tbl"><table class="cand">
    <thead><tr><th>tracklet</th><th>tier</th><th>rate (°/day)</th><th>V</th><th>SkyBoT nearest</th><th>sep (″)</th><th>its rate</th><th>JPL nearest</th><th>sep (″)</th><th>its rate</th><th>Rubin MPC submission</th><th>matched to</th></tr></thead>
    <tbody>{drow}</tbody>
  </table></div>
  <ul>
    <li><b>N0004194 = 2025 NB54,</b> a 2:1 resonant Kuiper-belt object: predicted motion −5.3, −3.0 ″/h against measured −5.4, −3.0, V 22.5 against 22.7, 46″ away (its orbit uncertainty). Our 10″ catalogue match missed it.</li>
    <li>Three other near-misses (2016 GT34, 2008 DN26, 1997 AS16) match in speed but move in a different direction.</li>
  </ul>
</section>

<section id="b612">
  <h2>Rubin's own MPC submissions (B612 export)</h2>
  <p>The Asteroid Institute's public export of every Rubin (X05) observation in the MPC database (<code>obs_sbn_X05_full.parquet</code>, updated 2026-11-03, 8,125,096 observations). Rubin reports each detection as two positions, at the start and end of the 30 s exposure; we match their midpoint to the alert (within 1 s and 1″).</p>
  <ul>
    <li><b>Rubin submitted only {n_x05} observations from our nights</b> ({len(B6)} named tracklets, plus {n_notrk} observations without a tracklet name), all from July; none from October 29–30. Most come from exposures that are not in the alert stream, so they cannot match an alert.</li>
    <li>{int(B6.provid.notna().sum())} of those tracklets now have designations; {int((B6.status == "I").sum())} are unlinked tracklets in the Isolated Tracklet File (status I), submitted 2026-09-09 to 09-11.</li>
    <li><b>N0003116 is Rubin tracklet RL021CV</b>, one of those unlinked submissions: three of its four detections match exactly. It is a set-aside candidate (4 detections, 2.38 °/day, priora P = 1.00, digest2 100) and stays on the list.</li>
    <li>Three more unlinked Rubin tracklets (RL021CX, RL021CY, RL021Ce) are in our data only as 2-detection tracklets, so they are not on the 3+ list; priora and digest2 give each P = 1.0 and 100 (1.4–2.7 °/day).</li>
    <li>Designated objects found in our unknown set: the Apollo NEO 2026 OL3 (q 0.75 AU; priora P = 1.00, digest2 100), comet P/2026 N2 (priora P = 0.0001, digest2 1.5) and the distant object 2026 FP35 (q 41 AU).</li>
  </ul>
</section>

<section id="next">
  <h2>What would confirm a candidate</h2>
  <ul>
    <li><b>Image cutouts</b> for Tier A and B, to rule out artifacts.</li>
    <li><b>Precovery and recovery outside these nights:</b> Rubin nights after July 13 (the next monthly bulk file or a Fink transfer for July 14–31) and other surveys, searched along each candidate's range of possible orbits.</li>
    <li><b>Rubin's MPC submissions</b> (B612 clone), above.</li>
  </ul>
  <p class="src">Work directory: $PRIORA_ALERTS/link (prep.py, run_hl.sh, fix_sum.py, eval_links.py, check_known_dbs.py, build_page.py). HelioLinC fix: heliolinc_omp writes 31 summary columns but link_purify reads 32 (orbit_incl); fix_sum.py inserts it.</p>
</section>
</main>
'''
(L / "page").mkdir(exist_ok=True); (L / "page/lsst-candidate-linking.html").write_text(head + body); print("ok")
