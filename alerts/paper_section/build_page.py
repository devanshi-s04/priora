"""Working page for the new paper section on LSST alerts: figure options, section outline with verified numbers, plans for
the candidate table and the cutout block. Styles from the LSST alerts test page."""
import os
from pathlib import Path
T = Path(os.environ["DS42_ALERTS"])
old = (T / "page/lsst-alerts-test.html").read_text()
head = old.split("</style>")[0].replace("<title>LSST Alerts Test</title>", "<title>LSST Alerts Paper Section</title>", 1)
head += """.opt{display:flex;flex-direction:column;gap:8px;background:var(--sheet);border:1px solid var(--rule);border-radius:10px;padding:14px}
.opt h3{font:600 15px var(--sans)} .opt img{background:#fff;border-radius:6px;width:100%}
.outline{counter-reset:s} .outline > li{margin:10px 0}
.num{font-variant-numeric:tabular-nums}
.todo{color:var(--run);font:500 12px var(--mono);letter-spacing:.06em;text-transform:uppercase}
blockquote{margin:0;padding:10px 14px;border-left:3px solid var(--accent);background:var(--sheet);border-radius:0 6px 6px 0;max-width:72ch}
</style>
"""
import json
N = json.loads((Path(__file__).parent / "numbers_nd2.json").read_text())
f = lambda k: f"{N[k]:,}"
pct = lambda a, b: f"{100 * N[a] / N[b]:.0f}%"
body = f"""<main>
<header>
  <p class="kicker">ds42 paper · new section · LSST alerts · working draft · rerun with nDiaSources ≤ 2</p>
  <h1>LSST alerts paper section</h1>
  <p class="lede">Where we build the new section on running ds42 and digest2 over Rubin's unknown alerts: download, tracklets, cuts, how many unknown tracklets score high, which of those are NEO candidates, one scatter figure, one candidate table and one block of cutouts. All numbers are generated from the rerun's result files (<code>paper_section/numbers_nd2.json</code>), in which detections on Rubin objects with more than 2 detections are removed before linking. See also the <a href="https://claude.ai/artifact/XndqoKKVFWJTqxAZhjXvAH">overview</a> and the <a href="https://claude.ai/artifact/UQaBaBVMD43HEMjNZCeqSm">cutout gallery</a>.</p>
</header>

<section class="callout">
  <h3>Step 1: choose the scatter figure</h3>
  <p>Three options from the rerun, all in the paper's style: digest2 raw against ds42 P(NEO) with the All prior, coloured by P ≥ 0.5 (orange) and P &lt; 0.5 (slate). Dashed line: digest2 = 65; dotted: P = 0.5. Corner numbers count tracklets in each quadrant. Files: <code>paper/figures/lsst_unknown_scatter_*_nd2.{{pdf,png}}</code>, script <code>make_lsst_unknown_scatter.py</code> (LSST_RUN=full_nd2 OUT_SUFFIX=_nd2).</p>
</section>

<section>
  <div class="opt"><h3>Option A · two panels: all tracklets and 3+ detections (full width)</h3>
    <img src="img/lsst_unknown_scatter_2panel_nd2.png" alt="Two panels. Left: all {f('moving')} tracklets, a dense cloud of chance pairs at top right. Right: {f('s3')} tracklets with 3+ detections, {f('s3_hiP')} orange points at top right.">
    <p>Panel (a) shows that most 2-detection tracklets land in the NEO corner for both codes (chance pairs); panel (b) shows that with 3+ detections only {f('s3_hiP')} do. My recommendation.</p></div>
  <div class="opt"><h3>Option B · 3+ detections only (one column)</h3>
    <img src="img/lsst_unknown_scatter_3plus_nd2.png" alt="Single panel of {f('s3')} tracklets with 3+ detections.">
    <p>Compact; the chance-pair point would then be made in the text.</p></div>
  <div class="opt"><h3>Option C · 3+ detections with score histograms (one column)</h3>
    <img src="img/lsst_unknown_scatter_3plus_hist_nd2.png" alt="Same as option B with histograms of both scores.">
    <p>Adds the two score distributions: ds42 is two-peaked, digest2 has a long tail of intermediate scores.</p></div>
</section>

<section>
  <h2>Section outline (draft, generated numbers)</h2>
  <ol class="outline">
    <li><b>Alerts.</b> Every Rubin alert from the nights of 2026 July 5–13 (8 nights with data) and October 29–30, downloaded through the Fink Data Transfer service with no broker filtering (3,789,349 alerts; one June 30 night dropped). 291,305 were associated by Rubin with known solar-system objects; the remaining 3,498,044 are unknown at alert time.</li>
    <li><b>Detection cuts.</b> Positive PSF flux; not a dipole, glint or streak; reliability ≥ 0.9 (keeps 95% of the detections of known asteroids): {f('dets_before_nd')} unknown detections. Then detections on Rubin objects with more than 2 detections (nDiaSources &gt; 2: static and variable sources) are removed: {f('nd_removed')} ({pct('nd_removed', 'dets_before_nd')}), leaving <span class="num">{f('dets')}</span>. That cut removes 0.05% of the detections of real asteroids.</li>
    <li><b>Tracklets.</b> HelioLinC <code>make_tracklets</code> (maximum rate 5 deg/day, maximum 1.5 h between detections, image radius 2°, great-circle residual ≤ 0.5″): <span class="num">{f('tracklets')}</span> tracklets; <span class="num">{f('arc15')}</span> with arcs ≥ 15 min; {f('stationary')} stationary tracklets (total motion &lt; 0.5″) removed, leaving <span class="num">{f('moving')}</span> moving tracklets: {f('n2')} with 2 detections, {f('n3')} with 3, {f('n4')} with 4 or more.</li>
    <li><b>Scores.</b> ds42 (All prior) and digest2 2.8.2 on every tracklet. With 2+ detections, <span class="num">{f('hiP')}</span> ({pct('hiP', 'moving')}) have P(NEO) ≥ 0.5 and <span class="num">{f('hiD')}</span> ({pct('hiD', 'moving')}) have digest2 ≥ 65; every ds42 P ≥ 0.5 tracklet also has digest2 ≥ 65. These are almost all 2-detection chance pairs of unrelated detections, which look like fast movers to both codes; neither code can reject a mis-link. With 3+ detections, <span class="num">{f('s3_hiP')}</span> of {f('s3')} ({100 * N['s3_hiP'] / N['s3']:.1f}%) have P ≥ 0.5 and {f('s3_hiD')} ({100 * N['s3_hiD'] / N['s3']:.1f}%) have digest2 ≥ 65. → <b>Figure</b> (scatter).</li>
    <li><b>Labelled subset (sanity check).</b> Matching every orbit in the MPC catalogue to the detections labels {f('lab')} tracklets ({N['lab_neo']} NEOs): ds42 All predicts {N['lab_sumP']} NEOs against {N['lab_neo']} observed; at digest2 ≥ 65 there are {N['d2_fa']} non-NEOs ({N['d2_fa_dist']} with q ≥ 4.5 AU), at ds42 P ≥ 0.1 there are {N['ds_fa']} ({N['ds_fa_dist']} distant).</li>
    <li><b>Candidates.</b> 3+ detections, no catalogue match, ds42 P ≥ 0.1: <span class="num">{f('cand')}</span> tracklets ({f('cand_p05')} at P ≥ 0.5): tier A {N['tierA']}, tier B {N['tierB']}, set aside {N['tierS']}. Static-source checks (back-to-back detections that move less than half the expected amount; a source in the template image at 2+ detections), validated on real asteroids, flag {N['static']}; <b>{N['clean']} have no static signature</b> (tier A {N['clean_A']}, B {N['clean_B']}, set aside {N['clean_S']}). Known-object databases (SkyBoT, JPL) and Rubin's own MPC submissions match {N['matched']}: 2025 NB54 (a Kuiper-belt object) and Rubin's unlinked tracklet RL021CV. → <b>Table</b> and <b>cutout block</b>.</li>
  </ol>
</section>

<section>
  <h2>Still to build</h2>
  <ul>
    <li><b>Candidate table:</b> ID, first detection (UTC), RA, Dec, detections, arc, rate, V, ds42 P (All, Unk), digest2 (raw, NoID), static flag, match. The {N['clean']} with no static signature in the paper; all {N['cand']} in a machine-readable table.</li>
    <li><b>Cutout block:</b> science | difference | template for each detection of a few candidates, with the motion vector drawn. Source: ALeRCE alert stamps (public), not the embargoed Rubin cutouts.</li>
    <li><b>HelioLinC on the rerun</b> and <b>DP2 alongside</b> (Early DP2, July 2025: calibration on 116k known tracklets with 3+ detections; labels from Rubin's MPC submissions): running.</li>
  </ul>
</section>
</main>
"""
(Path(__file__).parent / "lsst-alerts-paper-section.html").write_text(head + body); print("ok")
