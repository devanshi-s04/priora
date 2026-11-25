"""Builds the results page for the NoID / Unk comparison from results.json and figures/ (claude.ai artifact).
Run: python3 validation/unk_noid/build_page.py <out.html>"""
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
SC = json.loads((HERE.parents[0] / "output/gate_v0unk_hext30/summary.json").read_text()) if False else None
IMG = {k: "data:image/png;base64," + base64.b64encode((HERE / "figures" / f"figs1to4_{k}.png").read_bytes()).decode()
       for k in ("P", "N", "U", "M")}
ORDER = ["P", "N", "U", "M"]
SETUP = {"P": ("All", "All", "raw", "Paper as it stands"),
         "N": ("All", "All", "NoID", "Mario's swap"),
         "U": ("All", "Unk", "NoID", "Unk prior on the wrong sample"),
         "M": ("Unk", "Unk", "NoID", "Fully matched")}


def pct(x):
    return f"{100 * x:.1f}%"


def row(label, f, cls=""):
    cells = "".join(f'<td class="num{(" " + cls) if k == "M" else ""}">{f(R[k])}</td>' for k in ORDER)
    return f"<tr><td>{label}</td>{cells}</tr>"


rows = "".join([
    row("NEOs in the sample", lambda r: f"{r['n_neo']}"),
    row("AUC, priora", lambda r: f"{r['auc_priora']:.4f}"),
    row("AUC, digest2", lambda r: f"{r['auc_d2']:.4f}"),
    row("ΔAUC (priora − digest2), 95% interval",
        lambda r: f"{r['delta_auc']:+.4f}<br><span class='ci'>[{r['delta_ci'][0]:+.4f}, {r['delta_ci'][1]:+.4f}]</span>"),
    row("Predicted / observed NEOs, priora", lambda r: f"{r['pred_obs_priora']:.2f}"),
    row("Predicted / observed NEOs, digest2", lambda r: f"{r['pred_obs_d2']:.2f}"),
    row("digest2 ≥ 65: flagged", lambda r: f"{r['d2_ge65']['flagged']}"),
    row("digest2 ≥ 65: completeness", lambda r: pct(r['d2_ge65']['completeness'])),
    row("digest2 ≥ 65: false alarms", lambda r: f"{r['d2_ge65']['fa']}"),
    row("&nbsp;&nbsp;of which q ≥ 4.5 AU (distant)", lambda r: f"{r['d2_ge65']['fa_far']}"),
    row("&nbsp;&nbsp;of which q 1.67–4.5 AU", lambda r: f"{r['d2_ge65']['fa_q_1p67_4p5']}"),
    row("&nbsp;&nbsp;of which q &lt; 1.67 AU", lambda r: f"{r['d2_ge65']['fa_q_lt_1p67']}"),
    row("digest2 ≥ 65: contamination", lambda r: pct(r['d2_ge65']['contamination'])),
    row("priora at the same completeness: cut", lambda r: f"P ≥ {r['priora_matched']['cut']:.3f}"),
    row("priora at the same completeness: false alarms", lambda r: f"{r['priora_matched']['fa']}"),
    row("priora at the same completeness: contamination", lambda r: pct(r['priora_matched']['contamination']), "hl"),
    row("priora at P ≥ 0.5: completeness / contamination",
        lambda r: f"{pct(r['priora_p05']['completeness'])} / {pct(r['priora_p05']['contamination'])}"),
])
setup_rows = "".join(
    f"<tr><td><b>{k}</b></td><td>{SETUP[k][3]}</td><td>{SETUP[k][0]}</td><td>{SETUP[k][1]}</td><td>{SETUP[k][2]}</td></tr>"
    for k in ORDER)
heads = "".join(f"<th class='num'>{k}</th>" for k in ORDER)
buttons = "".join(f'<button type="button" data-k="{k}" aria-pressed="{str(k == "M").lower()}">{k} · {SETUP[k][3]}</button>'
                  for k in ORDER)
Mr = R["M"]

HTML = f"""<title>NoID and Unk Comparison</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,650&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; the four setups run as table columns P → N → U → M, the matched one (M) highlighted */
:root{{
  --paper:#f5f6f3; --sheet:#fdfdfb; --ink:#1c2227; --ink-2:#4b555d; --ink-3:#7a848b; --rule:#d8dbd3;
  --accent:#1f6f78; --accent-soft:#e0eeee; --warn:#9a6a12; --warn-bg:#f6ecd6;
  --serif:"Source Serif 4",Georgia,serif; --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --paper:#121517; --sheet:#1a1e21; --ink:#e6e9eb; --ink-2:#b3bbc1; --ink-3:#858f98; --rule:#2e3439;
  --accent:#6cc0c8; --accent-soft:#163338; --warn:#e3b45e; --warn-bg:#3a2d14; color-scheme:dark}}}}
:root[data-theme="dark"]{{
  --paper:#121517; --sheet:#1a1e21; --ink:#e6e9eb; --ink-2:#b3bbc1; --ink-3:#858f98; --rule:#2e3439;
  --accent:#6cc0c8; --accent-soft:#163338; --warn:#e3b45e; --warn-bg:#3a2d14; color-scheme:dark}}
*{{box-sizing:border-box}}
body{{background:var(--paper);color:var(--ink);font:15px/1.6 var(--sans);margin:0;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:32px 64px}}
main{{max-width:900px;margin:0 auto;display:flex;flex-direction:column;gap:32px}}
.kicker{{font:500 11.5px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);margin:0 0 8px}}
h1{{font:650 33px/1.15 var(--serif);margin:0 0 10px;text-wrap:balance}}
h2{{font:650 23px/1.25 var(--serif);margin:0 0 12px;text-wrap:balance}}
p{{margin:0 0 10px;max-width:70ch}} ul{{margin:0 0 10px;padding-left:20px;max-width:74ch}} li{{margin:5px 0}}
.lede{{color:var(--ink-2);font-size:16px}}
.bottom{{background:var(--sheet);border:1px solid var(--rule);border-left:4px solid var(--accent);border-radius:6px;padding:16px 20px}}
.defs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}}
.defs div{{background:var(--sheet);border:1px solid var(--rule);border-radius:6px;padding:12px 14px;min-width:0}}
.defs b{{display:block;font:600 12px var(--mono);letter-spacing:.06em;color:var(--accent);margin-bottom:4px}}
.defs span{{font-size:14px;color:var(--ink-2)}}
.tbl{{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--sheet)}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:7px 11px;border-bottom:1px solid var(--rule);vertical-align:top}}
th{{font:600 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);background:var(--paper)}}
tr:last-child td{{border-bottom:0}}
.num{{text-align:right;white-space:nowrap}}
th.num:last-child,td.num:last-child{{background:var(--accent-soft)}}
td.hl{{font-weight:600}}
.ci{{font-size:12px;color:var(--ink-3)}}
.seg{{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 12px}}
.seg button{{font:500 13px/1 var(--sans);color:var(--ink-2);background:var(--sheet);border:1px solid var(--rule);border-radius:6px;padding:9px 12px;cursor:pointer}}
.seg button[aria-pressed="true"]{{color:var(--ink);border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}}
.seg button:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
figure{{margin:0;background:#fff;border:1px solid var(--rule);border-radius:6px;padding:10px}}
figure img{{display:block;width:100%;height:auto}}
.note{{font-size:13.5px;color:var(--ink-2)}}
.warn{{background:var(--warn-bg);border-radius:6px;padding:10px 14px;font-size:14px}}
.warn b{{color:var(--warn)}}
code{{font:13px var(--mono);overflow-wrap:anywhere}}
@media (max-width:520px){{h1{{font-size:27px}}h2{{font-size:20px}}}}
</style>
<main>
<header>
  <p class="kicker">priora · paper 1 · §5–§6 · 2026-10-21</p>
  <h1>Switching digest2 to NoID, and matching priora to it</h1>
  <p class="lede">Mario asked to compare against digest2's NoID score (the one the NEOCP uses) instead of its raw score. We did that, then matched priora to the same population (Unk), then drew a fresh test sample from that population so priora's probabilities can be checked on the same footing. Four setups, same 20,000-tracklet design, all with the extended model (H to 30).</p>
</header>

<section class="bottom">
  <h2>Bottom line</h2>
  <ul>
    <li><b>priora still wins in every setup.</b> In the fully matched setup (M) its AUC is {Mr['auc_priora']:.4f} vs {Mr['auc_d2']:.4f}, ΔAUC {Mr['delta_auc']:+.4f} [{Mr['delta_ci'][0]:+.4f}, {Mr['delta_ci'][1]:+.4f}]. The lead is smaller than in the paper (+0.016), because both codes do better on this sample.</li>
    <li><b>priora stays calibrated when matched</b>: predicted/observed NEOs {Mr['pred_obs_priora']:.2f}, and the new sample passes every self-consistency check (observed 287 vs expected 284 NEOs). digest2 NoID overpredicts by {Mr['pred_obs_d2']:.1f}×.</li>
    <li><b>The false-alarm gap holds</b>: at digest2's completeness ({pct(Mr['d2_ge65']['completeness'])}), digest2 NoID has {Mr['d2_ge65']['fa']} false alarms ({pct(Mr['d2_ge65']['contamination'])} contamination), priora {Mr['priora_matched']['fa']} ({pct(Mr['priora_matched']['contamination'])}).</li>
    <li><b>The distant-object story comes back when matched.</b> {Mr['d2_ge65']['fa_far']} of digest2's {Mr['d2_ge65']['fa']} false alarms in M are distant (q ≥ 4.5 AU). In setup N it looked like the main belt took over; that was the mismatch between NoID and an All sample.</li>
    <li><b>Mixing populations hurts priora too.</b> Unk priora on the All sample (U) overpredicts by {R['U']['pred_obs_priora']:.2f}× and has {R['U']['priora_matched']['fa']} false alarms. That is the prior being wrong for the sample, not the method.</li>
  </ul>
</section>

<section>
  <h2>What raw, NoID, All and Unk mean</h2>
  <div class="defs">
    <div><b>raw (digest2)</b><span>Fraction of all model objects in the bins a tracklet could fall in that are NEOs.</span></div>
    <div><b>NoID (digest2)</b><span>Same, but counting only objects not yet discovered. What the NEOCP uses. Main-belt objects are mostly known, so scores rise.</span></div>
    <div><b>All (priora)</b><span>priora with the whole population. Counterpart of raw.</span></div>
    <div><b>Unk (priora)</b><span>priora with the undiscovered population. Counterpart of NoID.</span></div>
  </div>
</section>

<section>
  <h2>The four setups</h2>
  <div class="tbl"><table>
    <tr><th></th><th>What it is</th><th>Sample drawn from</th><th>priora</th><th>digest2</th></tr>
    {setup_rows}
  </table></div>
  <p class="note">P, N and U use the paper's 20,000-tracklet sample. M is a new 20,000-tracklet sample drawn the same way (seed, epochs, magnitudes, geocentric observer) from the Unk population; it contains more NEOs (287) because NEOs are a larger share of what is undiscovered.</p>
</section>

<section>
  <h2>Results</h2>
  <div class="tbl"><table>
    <tr><th>Quantity</th>{heads}</tr>
    {rows}
  </table></div>
  <p class="note">Intervals are from a paired bootstrap (2,000 resamples, NEOs and non-NEOs resampled separately). Setup M also scored digest2 raw for reference: AUC {R['Mraw']['auc_d2']:.4f}, {R['Mraw']['d2_ge65']['fa']} false alarms at ≥ 65 ({R['Mraw']['d2_ge65']['fa_far']} distant), {pct(R['Mraw']['d2_ge65']['contamination'])} contamination.</p>
</section>

<section>
  <h2>Figures 1–4 for each setup</h2>
  <div class="seg" role="group" aria-label="Choose setup">{buttons}</div>
  <figure><img id="fig" src="{IMG['M']}" alt="Four panels for the selected setup: priora reliability, ROC, predicted vs observed NEOs per bin, digest2 score vs priora P(NEO)"></figure>
  <p class="note">(1) priora reliability in ten bins of 0.1, with tracklet counts; (2) ROC, log false-positive rate; (3) predicted (bar) vs observed (dot) NEOs per score bin, digest2 score divided by 100; (4) the Figure-4 scatter with the digest2 65 line and P = 0.1. These are quick-look panels; paper-style NoID versions of Figs 2–4 for setup N are in <code>paper/figures/*_noid.*</code>.</p>
</section>

<section>
  <h2>Checks and caveats</h2>
  <ul>
    <li>The paper's raw-score figures were re-made after adding the NoID switch and reproduce the paper's numbers exactly.</li>
    <li>Setup M's sample passes all self-consistency checks: reliability, HPD rank uniformity, collapse coverage, truth in domain, scores defined. Fitted odds scale ĉ = 1.13 [0.76, 1.68].</li>
    <li>digest2 skipped 1 of the 20,000 tracklets in M (the emitter refused it); it is dropped from both codes.</li>
    <li>Middle reliability bins hold only a few tracklets each at N = 20,000, so panel 1 is noisy there; the paper's pooled 2 × 10⁵ run is what Fig 1 uses.</li>
    <li>The ephemeris was pinned to astropy's built-in Earth ephemeris, as in the paper's runs.</li>
  </ul>
  <p class="warn"><b>For the paper:</b> setup M is the clean NoID comparison. If the paper switches to NoID, the §6 numbers, Fig 4 and the "most false alarms are distant" sentence should come from M, not from N.</p>
</section>

<section>
  <h2>Files</h2>
  <ul>
    <li><code>validation/unk_noid/analyze.py</code> computes everything on this page (<code>results.json</code>, <code>figures/</code>); <code>build_page.py</code> builds the page; <code>run_sc.py</code> runs the generator with the ephemeris pinned.</li>
    <li>Runs: <code>validation/output/gate_v0_hext30_scoreUnk/</code> (priora Unk on the paper sample), <code>gate_v0unk_hext30/</code> (new Unk sample), <code>digest2_comparison_unk/</code> (digest2 on it).</li>
    <li>Paper-style NoID figures for setup N: <code>paper/figures/*_noid.*</code>; scripts take <code>D2_SCORE=noid</code>.</li>
    <li>Nothing is committed yet.</li>
  </ul>
</section>
</main>
<script>
const IMG = {json.dumps(IMG)};
const btns = document.querySelectorAll(".seg button");
btns.forEach(b => b.addEventListener("click", () => {{
  btns.forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  document.getElementById("fig").src = IMG[b.dataset.k];
}}));
</script>
"""
out = Path(sys.argv[1] if len(sys.argv) > 1 else "noid-unk.html")
out.write_text(HTML)
print(out, len(HTML))
