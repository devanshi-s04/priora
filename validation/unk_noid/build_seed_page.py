"""Builds the 'Calibration Seed Check' artifact page from seed_check.json (run seed_check.py first). Images in <out>/img/."""
import html, json, sys
from pathlib import Path
R = json.loads((Path(__file__).resolve().parent / "seed_check.json").read_text())
rows = ""
for lab, v in R.items():
    m = v["mid"]; z = m["z"]
    cls = "hi" if abs(z) >= 3 else ("md" if abs(z) >= 2 else "")
    rows += (f"<tr class='{'pool' if 'pooled' in lab else ''}'><td>{html.escape(lab)}</td><td class='n'>{v['n']:,}</td><td class='n'>{v['n_neo']:,}</td>"
             f"<td class='n'>{v['pred_obs']:.3f}</td><td class='n'>{m['obs']} / {m['exp']:.1f}</td><td class='n {cls}'>{z:+.2f}</td>"
             f"<td class='n'>{v['nonneo_p1']}</td></tr>")
P = R["Unk · pooled"]
HTML = f"""<title>Calibration Seed Check</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,650&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; the verdict first, then the table of runs, then the plots */
:root{{--paper:#f4f5f2;--sheet:#fcfcfa;--ink:#1c2227;--ink-2:#4b555d;--ink-3:#7a848b;--rule:#d8dbd3;--accent:#1f6f78;--warn:#a5521a;--warn-bg:#f7e6da;--soft:#e0eeee;
  --serif:"Source Serif 4",Georgia,serif;--sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--paper:#121517;--sheet:#1a1e21;--ink:#e6e9eb;--ink-2:#b3bbc1;--ink-3:#858f98;--rule:#2e3439;--accent:#6cc0c8;--warn:#f0a070;--warn-bg:#3a2418;--soft:#163338;color-scheme:dark}}}}
:root[data-theme="dark"]{{--paper:#121517;--sheet:#1a1e21;--ink:#e6e9eb;--ink-2:#b3bbc1;--ink-3:#858f98;--rule:#2e3439;--accent:#6cc0c8;--warn:#f0a070;--warn-bg:#3a2418;--soft:#163338;color-scheme:dark}}
*{{box-sizing:border-box}}
body{{background:var(--paper);color:var(--ink);font:15px/1.6 var(--sans);margin:0;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:32px 64px}}
main{{max-width:900px;margin:0 auto;display:flex;flex-direction:column;gap:28px}}
.kicker{{font:500 11.5px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);margin:0 0 8px}}
h1{{font:650 32px/1.15 var(--serif);margin:0 0 10px;text-wrap:balance}} h2{{font:650 22px/1.25 var(--serif);margin:0 0 10px}}
p{{margin:0 0 10px;max-width:70ch}} ul{{margin:0;padding-left:20px;max-width:74ch}} li{{margin:5px 0}}
.lede{{color:var(--ink-2);font-size:16px}}
.verdict{{background:var(--sheet);border:1px solid var(--rule);border-left:4px solid var(--accent);border-radius:6px;padding:16px 20px}}
.tbl{{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--sheet)}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--rule)}} th{{font:600 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}}
tr:last-child td{{border-bottom:0}} .n{{text-align:right;white-space:nowrap}} tr.pool td{{font-weight:600;background:var(--soft)}}
td.hi{{color:var(--warn);background:var(--warn-bg);font-weight:600}} td.md{{color:var(--warn)}}
figure{{margin:0;background:#fff;border:1px solid var(--rule);border-radius:6px;padding:10px}} figure img{{display:block;width:100%;height:auto}}
figcaption{{font-size:13px;color:#4b555d;margin-top:6px}}
.pair{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px}}
.act{{font:500 12.5px var(--sans);color:#1c2227;background:#f2f2ee;border:1px solid #d8dbd3;border-radius:5px;padding:6px 10px;cursor:pointer;margin-left:8px}}
.act:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}} .msg{{font-size:12px;color:#7a848b;margin-left:6px}}
.note{{font-size:13.5px;color:var(--ink-2)}} code{{font:12.5px var(--mono)}}
</style>
<main>
<header><p class="kicker">ds42 · calibration on the undiscovered population · 2026-10-24</p>
<h1>Was the mid-range dip real?</h1>
<p class="lede">The first 200,000-tracklet sample from the undiscovered (Unk) population showed fewer NEOs than predicted between P = 0.2 and 0.7. We drew three more independent samples (two new seeds in 2026, one in 2027), scored them with ds42 exactly as before, and compared.</p></header>
<section class="verdict"><h2>Verdict: it was a fluctuation</h2><ul>
<li>The dip does not repeat. The mid-range z is −3.2 in the first sample, then +2.1, +0.8 and +0.0 in the new ones.</li>
<li>Pooled over all four samples ({P['n']:,} tracklets, {P['n_neo']:,} NEOs), the mid range has {P['mid']['obs']} NEOs against {P['mid']['exp']:.1f} expected (z = {P['mid']['z']:+.2f}), and predicted/observed overall is {P['pred_obs']:.3f}.</li>
<li>The first sample's −3.2 is also inflated because the 0.2–0.7 range was picked after seeing that plot.</li>
<li>One thing does repeat: a handful of very bright, very distant non-NEOs (perihelion 31–98 AU, H 0.3–3.6) get P = 1 in every sample, {P['nonneo_p1']} in 800,000 tracklets. The undiscovered population has almost no bright distant objects, so ds42 finds no non-NEO alternative there. This is the same prior-coverage gap as Pluto, and it is why each run's built-in check reports FAIL.</li>
</ul></section>
<section><h2>The runs</h2><div class="tbl"><table>
<tr><th>Sample</th><th class="n">Tracklets</th><th class="n">NEOs</th><th class="n">Pred/obs</th><th class="n">Mid range obs / exp</th><th class="n">Mid z</th><th class="n">Non-NEOs at P=1</th></tr>{rows}</table></div>
<p class="note">Mid range = P between 0.2 and 0.7. Unk runs: 200,000 tracklets each, h_extend_to = 30, the same generator as the paper. All runs: the paper's two 100,000-tracklet full-population runs, for reference.</p></section>
<section><h2>All four samples on one plot</h2>
<figure><img src="img/seed_check.png" alt="Reliability diagram with four coloured runs and black pooled points; pooled points sit on the diagonal"><figcaption>Coloured: each sample, offset slightly in x so error bars don't overlap. Black squares: all four pooled. 68% intervals; lower panel is observed / expected.</figcaption></figure></section>
<section><h2>Talk options</h2><div class="pair">
<figure><img src="img/calibration_undiscovered_800k.png" alt="Calibration plot, 800,000 undiscovered-population tracklets"><figcaption>Undiscovered population, 4 samples pooled (799,963 tracklets) <button class="act" data-f="calibration_undiscovered_800k.png">Save</button><span class="msg"></span></figcaption></figure>
<figure><img src="img/calibration_all_200k.png" alt="Calibration plot, 200,000 full-population tracklets"><figcaption>Full population (the paper's Fig 1, 199,997 tracklets) <button class="act" data-f="calibration_all_200k.png">Save</button><span class="msg"></span></figcaption></figure>
</div><p class="note">Either works for the talk now. The pooled undiscovered-population plot matches the main scatter's population. The first single sample (seed 20260902) is kept for the record but shouldn't be shown alone.</p></section>
<section><h2>Files</h2><p class="note"><code>validation/unk_noid/seed_check.py</code> (numbers, figure) and <code>seed_check.json</code>; runs in <code>validation/output/gate_v0unk200k{{,_s3,_s4,_s5y27}}_hext30/</code>; pooled plot from <code>paper/figures/make_linear_hext30.py</code> with <code>CAL_RUNS</code>. Not committed.</p></section>
</main>
<script>
let dl=null; const ready=(window.claude&&claude.use?claude.use("downloads"):Promise.resolve(null)).then(d=>dl=d).catch(()=>null);
document.querySelectorAll("[data-f]").forEach(b=>b.addEventListener("click",async()=>{{const m=b.nextElementSibling;await ready;
 if(!dl){{m.textContent="Right-click the image to save it.";return;}}
 try{{await dl.save({{filename:b.dataset.f,data:await(await fetch("img/"+b.dataset.f)).blob()}});m.textContent="Saved";}}catch(e){{m.textContent=e&&e.code==="declined"?"Cancelled":"Right-click the image to save it.";}}}}));
</script>
"""
Path(sys.argv[1]).write_text(HTML); print(len(HTML))
