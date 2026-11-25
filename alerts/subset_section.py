"""HTML section for one subset (analysis_<n>plus.json from subset_figs.py), inserted before a marker in a test page.
Usage: python3 subset_section.py <test> <nmin> <section number label> <marker>"""
import os
import json, sys
from pathlib import Path
T = Path(os.environ["PRIORA_ALERTS"])
test, nmin, num, marker = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
SD, page = {"known": (T / "score", T / "page/lsst-alerts-test.html"), "unk": (T / "unk/score", T / "unk/page/lsst-unknown-alerts-test.html"),
            "full": (T / "full/score", T / "full/page/lsst-full-unknown-alerts.html")}[test]
a = json.loads((SD / f"analysis_{nmin}plus.json").read_text())
ev = "pure" if test == "known" else "labelled"
oth = "mixed" if test == "known" else "unlabelled (mixed known, partial, unmatched)"
F, fl, au = a["figures"], a["flags"], a["auc"]
f2 = lambda x: f"{x:,}"
def auc(k):
    v = au.get(k); return "–" if not v else (f"{v[0]:.5f} ± {v[1]:.5f}" if v[0] > 0.999 else f"{v[0]:.4f} ± {v[1]:.4f}")
rows = "".join(f"<tr><td>{lab}</td><td>{fl[k]['flagged']}</td><td>{fl[k]['neo']}</td><td>{fl[k]['non_neo']}</td><td>{fl[k]['distant']}</td></tr>"
               for lab, k in (("digest2 raw ≥ 65", "d2_raw>=65"), ("digest2 NoID ≥ 65", "d2_noid>=65"), ("priora All, P ≥ 0.1", "priora_All>=0.1"),
                              ("priora Unk, P ≥ 0.1", "priora_Unk>=0.1"), ("priora All, P ≥ 0.5", "priora_All>=0.5")))
npts = "; ".join(f"{k}: {v:,}" for k, v in a["npts"].items())
d = a.get("dAUC_All")
dtxt = f"ΔAUC (priora All − digest2 raw) = {d[0]:+.4f}, 95% CI {d[1][0]:+.4f} to {d[1][1]:+.4f}." if d else ""
fig = lambda k, cap, alt: f'<figure><img src="img/{F[k]}" alt="{alt}"><figcaption>{cap}</figcaption></figure>' if k in F else ""
oth_f = a["other_flagged"]
sec = f'''<section id="det{nmin}plus">
  <div class="step-head"><span class="num">{num}</span><h2>Only tracklets with {nmin}+ detections</h2><span class="pill run">first look</span></div>
  <p>The same tables and plots, restricted to tracklets with at least {nmin} detections in the night (arc ≥ 15 min). {ev.capitalize()} tracklets: <b>{f2(a["n"])}</b> ({npts}), of which <b class="neo">{a["n_neo"]} NEOs</b>. {oth.capitalize()}: {f2(a["n_other"])}.</p>
  <div class="tbl"><table>
    <thead><tr><th>score ({nmin}+)</th><th>AUC (bootstrap σ)</th><th>NEOs predicted ({a["n_neo"]} observed)</th></tr></thead>
    <tbody>
      <tr><td>priora All</td><td>{auc("priora_All")}</td><td>{a["sumP"]["priora_All"]:.1f}</td></tr>
      <tr><td>priora Unk</td><td>{auc("priora_Unk")}</td><td>{a["sumP"]["priora_Unk"]:.1f}</td></tr>
      <tr><td>digest2 raw</td><td>{auc("d2_raw")}</td><td>{a.get("d2_over_100_All", float("nan")):.0f} if read as score/100</td></tr>
      <tr><td>digest2 NoID</td><td>{auc("d2_noid")}</td><td>–</td></tr>
    </tbody>
  </table></div>
  <div class="tbl"><table>
    <thead><tr><th>flagged ({nmin}+)</th><th>flagged</th><th>NEOs</th><th>non-NEOs</th><th>q ≥ 4.5 AU</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  <ul>
    <li>{dtxt} With only {a["n_neo"]} NEOs, AUC barely separates the codes; the false-alarm counts are the clearer comparison.</li>
    <li>{oth.capitalize()} tracklets with {nmin}+ detections flagged: digest2 raw {oth_f["d2_raw"]}, NoID {oth_f["d2_noid"]}; priora All {oth_f["priora_All"]}, Unk {oth_f["priora_Unk"]} (P ≥ 0.5).</li>
  </ul>
  <div class="gallery wide">
    {fig("scatter_logx", f"{nmin}+ detections, the paper's Figure 4 axes. Dashed: digest2 = 65 and priora P = 0.1.", f"Two panels: digest2 score against priora P(NEO), {nmin}+ detection tracklets.")}
    {fig("scatter_loglog", "Both axes log.", "Same data, both axes log.")}
    {fig("scatter_lin100", "Both axes linear, 0–100.", "Same data, both axes linear.")}
  </div>
  <div class="gallery">
    {fig("cal_all", f"priora All calibration, {nmin}+ detections.", "Linear reliability diagram for priora All.")}
    {fig("cal_unk", f"priora Unk calibration, {nmin}+ detections.", "Linear reliability diagram for priora Unk.")}
    {fig("counts", "Predicted vs observed NEOs per bin.", "Predicted versus observed NEOs per bin, priora and digest2.")}
    {fig("rel", "Reliability, logit axes.", "Reliability on logit axes.")}
    {fig("roc", "ROC, log false-positive axis.", "ROC curves.")}
  </div>
</section>

'''
s = page.read_text()
if f'<section id="det{nmin}plus">' in s:
    i = s.index(f'<section id="det{nmin}plus">'); j = s.index("</section>", i) + len("</section>\n\n")
    s = s[:i] + s[j:]
assert s.count(marker) == 1, marker
page.write_text(s.replace(marker, sec + marker))
print(page.name, "ok", [F[k] for k in F])
