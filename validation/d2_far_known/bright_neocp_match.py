"""Mario's test (2026-11-06): can the simulated comparison reproduce the NEOCP operating point (Table 5) once it is cut to
NEOCP-like brightness and known objects are removed?
  1. V < 21.2 (NEOCP sample median 21.14 + 0.1).
  2. 'Known' = brighter than the peak of the known-object H distribution in MPCORB for the object's orbit range
     (NEO q<1.3, Mars-crosser 1.3-1.67, inner/main 1.67-4.5, Trojan/Centaur 4.5-10, TNO >= 10 AU), applied to every class.
  3. Completeness / contamination at digest2 >= 65 and at priora's matched cut, before and after removing known objects,
     against the NEOCP numbers for the same pairing (All/raw = Table 5; Unk/NoID = validation/output/neocp_unk_ext).
Also the same numbers for the Unk sample with 3-detection digest2 tracklets. -> bright_match.json, bright_scatter_*.png/pdf"""
import os
import gzip, json
from pathlib import Path
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent; O = Path("validation/output")
QR = [(0, 1.3, "NEO"), (1.3, 1.667, "Mars-crosser"), (1.667, 4.5, "main belt"), (4.5, 10, "Trojan/Centaur"), (10, 1e9, "TNO")]
Hs, qs = [], []
with gzip.open(Path(os.environ["PRIORA_ALERTS"]) / "dp2/mpcorb/MPCORB.DAT.gz", "rt", errors="ignore") as f:
    go = False
    for L in f:
        if L.startswith("-----"): go = True; continue
        if go and len(L) > 103 and L[8:13].strip():
            try: Hs.append(float(L[8:13])); qs.append(float(L[92:103]) * (1 - float(L[70:79])))
            except ValueError: pass
Hs, qs = np.array(Hs), np.array(qs); bins = np.arange(-1, 32, 0.5); peaks = {}
for lo, hi, nm in QR:
    h = np.histogram(Hs[(qs >= lo) & (qs < hi)], bins=bins)[0]; peaks[nm] = float(bins[np.argmax(h)] + 0.25)
def load(run, d2dir, sc):
    g = np.load(O / run / "gen.npz"); z = np.load(O / run / "scores.npz"); d = np.load(O / d2dir / "digest2_scores.npz")
    idx = np.intersect1d(z["index"], d["index"]); iz = np.searchsorted(z["index"], idx); idd = np.searchsorted(d["index"], idx); el = g["elements"][idx]
    D = dict(y=g["is_neo"][idx].astype(bool), q=el[:, 0] * (1 - el[:, 1]), H=g["H"][idx], V=g["v_mag"][idx], p=z["p_neo"][iz], s=d[sc][idd])
    pk = np.zeros(len(idx))
    for lo, hi, nm in QR: pk[(D["q"] >= lo) & (D["q"] < hi)] = peaks[nm]
    D["known"] = D["H"] < pk; m = np.isfinite(D["p"]); return {k: v[m] for k, v in D.items()}
def ops(y, q, p, s):
    sel = s >= 65; tp, fp = int((sel & y).sum()), int((sel & ~y).sum()); r = dict(n=int(len(y)), neos=int(y.sum()), d2_flagged=int(sel.sum()), d2_complete=tp / y.sum(), d2_contam=fp / sel.sum() if sel.sum() else np.nan,
        d2_fa_q=[int((sel & ~y & (q < 1.67)).sum()), int((sel & ~y & (q >= 1.67) & (q < 4.5)).sum()), int((sel & ~y & (q >= 4.5)).sum())])
    if tp:
        cut = np.sort(p[y])[::-1][tp - 1]; s2 = p >= cut
        r.update(priora_cut=float(cut), priora_flagged=int(s2.sum()), priora_contam=float((s2 & ~y).sum() / s2.sum()), priora_fa_q=[int((s2 & ~y & (q < 1.67)).sum()), int((s2 & ~y & (q >= 1.67) & (q < 4.5)).sum()), int((s2 & ~y & (q >= 4.5)).sum())])
    s5 = p >= 0.5; r.update(p05_flagged=int(s5.sum()), p05_complete=float((s5 & y).sum() / y.sum()), p05_contam=float((s5 & ~y).sum() / s5.sum()) if s5.sum() else np.nan)
    return r
res = {"known_peak_H": peaks}
CFG = {"Unk / NoID": ("gate_v0unk_hext30", "digest2_comparison_unk", "noid_neo"), "All / raw": ("gate_v0_hext30", "digest2_comparison_hext30/gate_v0_hext30", "raw_neo"),
       "Unk / NoID, 3 detections": ("gate_v0unk_hext30", "digest2_comparison_unk_3det", "noid_neo"),
       "Unk / NoID, bright sample": ("gate_v0unk_bright212_hext30", "digest2_comparison_unk_bright212", "noid_neo")}
for nm, (run, d2, sc) in CFG.items():
    D = load(run, d2, sc); b = D["V"] < 21.2; bu = b & ~D["known"]
    res[nm] = {"all V": ops(D["y"], D["q"], D["p"], D["s"]), "V<21.2": ops(D["y"][b], D["q"][b], D["p"][b], D["s"][b]),
               "V<21.2, known removed": ops(D["y"][bu], D["q"][bu], D["p"][bu], D["s"][bu]),
               "known fraction among V<21.2": {"NEO": float(D["known"][b & D["y"]].mean()), "non-NEO": float(D["known"][b & ~D["y"]].mean())}}
    if "3 det" in nm: continue
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.0), layout="constrained", sharey=True)
    for ax, (sel, ttl) in zip(axs, ((b, f"(a) $V<21.2$"), (bu, f"(b) $V<21.2$, known objects removed"))):
        x = np.clip(D["p"][sel], 1e-8, 1); yv = D["s"][sel]; y = D["y"][sel]; far = (D["q"][sel] >= 4.5) & ~y
        ax.scatter(x[~y & ~far], yv[~y & ~far], s=6, color="#b9b6ae", lw=0, label="non-NEO, $q<4.5$ AU"); ax.scatter(x[far], yv[far], s=10, color="#6b3fa0", lw=0, label="non-NEO, $q\\geq4.5$ AU")
        ax.scatter(x[y], yv[y], s=16, color="#eb6834", lw=0, label="NEO"); ax.axhline(65, color="k", lw=0.8, ls="--"); ax.axvline(0.1, color="k", lw=0.8, ls=":")
        r = ops(D["y"][sel], D["q"][sel], D["p"][sel], D["s"][sel])
        ax.set(xscale="log", xlim=(1e-8, 1.6), ylim=(-2, 102), xlabel="priora $P(\\mathrm{NEO})$", title=f"{ttl}: {r['n']:,} tracklets, {r['neos']} NEOs")
        ax.text(0.03, 0.97, f"digest2$\\geq$65: {100 * r['d2_complete']:.0f}% complete, {100 * r['d2_contam']:.0f}% contam.", transform=ax.transAxes, va="top", fontsize=7.5, bbox=dict(fc="white", ec="none", alpha=0.85))
    axs[0].set_ylabel(f"digest2 {nm.split(' / ')[1]} NEO score"); axs[0].legend(loc="center left", frameon=False, fontsize=7)
    tag = nm.split(" / ")[0].lower() + ("_brightsample" if "bright sample" in nm else "")
    for ext in ("png", "pdf"): fig.savefig(OUT / f"bright_scatter_{tag}.{ext}", dpi=250)
    plt.close(fig)
(OUT / "bright_match.json").write_text(json.dumps(res, indent=1, default=float))
print("known peaks:", peaks)
for nm in CFG:
    print("==", nm, "| known frac V<21.2:", {k: round(v, 2) for k, v in res[nm]["known fraction among V<21.2"].items()})
    for k in ("all V", "V<21.2", "V<21.2, known removed"):
        r = res[nm][k]; print(f"  {k:24s} n={r['n']:6d} NEOs={r['neos']:4d} | d2>=65: {r['d2_flagged']} flagged, {100*r['d2_complete']:.1f}% complete, {100*r['d2_contam']:.1f}% contam, FA(q<1.67/1.67-4.5/>=4.5)={r['d2_fa_q']} | priora matched {r.get('priora_cut', float('nan')):.3f}: {r.get('priora_flagged')} flagged, {100*r.get('priora_contam', float('nan')):.1f}% contam FA={r.get('priora_fa_q')} | P>=0.5: {r['p05_flagged']}, {100*r['p05_complete']:.1f}%, {100*r['p05_contam']:.1f}%")
