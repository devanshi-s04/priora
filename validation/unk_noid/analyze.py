"""priora vs digest2 under four score/population pairings (Mario, 2026-10-21: switch digest2 to NoID).

  P  paper:      sample All, priora All, digest2 raw
  N  NoID swap:  sample All, priora All, digest2 NoID
  U  matched:    sample All, priora Unk, digest2 NoID
  M  matched+:   sample Unk, priora Unk, digest2 NoID    (both codes and the truth use the undiscovered population)
Inputs (all under validation/output/): gate_v0_hext30 (+ digest2_comparison_hext30), gate_v0_hext30_scoreUnk
(priora Unk on the v0 sample), gate_v0unk_hext30 (+ digest2_comparison_unk). Writes results.json and figures/ here.
Run from the repo root:  python3 -m validation.unk_noid.analyze
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
O = REPO / "validation/output"
THRESH = 65.0
RNG = np.random.default_rng(7)
C42, CD2, CINK, COTH, COUT = "#2a78d6", "#eb6834", "#0b0b0b", "#9a9993", "#6b3fa0"


def load(run, d2dir, priora_scores=None):
    with np.load(O / run / "gen.npz") as g, np.load(O / (priora_scores or run) / "scores.npz") as z, \
            np.load(O / d2dir / "digest2_scores.npz") as d:
        idx = np.intersect1d(z["index"], d["index"])          # digest2 may skip a tracklet the emitter refused
        iz, idd = np.searchsorted(z["index"], idx), np.searchsorted(d["index"], idx)
        el = g["elements"][idx]
        out = dict(p=z["p_neo"][iz].astype(float), raw=d["raw_neo"][idd].astype(float),
                   noid=d["noid_neo"][idd].astype(float), ok=d["ok"][idd].astype(bool),
                   y=g["is_neo"][idx].astype(bool), q=el[:, 0] * (1 - el[:, 1]))
        n_missing = len(z["index"]) - len(idx)
    m = np.isfinite(out["p"]) & out["ok"] & np.isfinite(out["raw"]) & np.isfinite(out["noid"])
    return {k: v[m] for k, v in out.items() if k != "ok"} | {"n_dropped": int((~m).sum()) + n_missing}


def auc(y, s):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def delta_auc_ci(y, a, b, n=2000):
    d = auc(y, a) - auc(y, b); boots = []
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    for _ in range(n):
        i = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        boots.append(auc(y[i], a[i]) - auc(y[i], b[i]))
    boots = np.array(boots)
    return d, [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))], float((boots <= 0).mean())


def metrics(D, d2key):
    p, s, y, q = D["p"], D[d2key], D["y"], D["q"]
    f2 = s >= THRESH; tp = int((f2 & y).sum())
    cut = np.sort(p[y])[::-1][tp - 1]; f42 = p >= cut; h = p >= 0.5
    dlt, ci, ple0 = delta_auc_ci(y, p, s)
    far = ~y & (q >= 4.5)
    return {
        "n": int(len(y)), "n_neo": int(y.sum()), "n_dropped": D["n_dropped"],
        "auc_priora": auc(y, p), "auc_d2": auc(y, s), "delta_auc": dlt, "delta_ci": ci, "p_delta_le_0": ple0,
        "pred_obs_priora": float(p.sum() / y.sum()), "pred_obs_d2": float(s.sum() / 100 / y.sum()),
        "d2_ge65": {"flagged": int(f2.sum()), "neo": tp, "fa": int((f2 & ~y).sum()), "fa_far": int((f2 & far).sum()),
                    "fa_q_lt_1p67": int((f2 & ~y & (q < 1.67)).sum()),
                    "fa_q_1p67_4p5": int((f2 & ~y & (q >= 1.67) & (q < 4.5)).sum()),
                    "completeness": tp / y.sum(), "contamination": float((f2 & ~y).sum() / f2.sum())},
        "priora_matched": {"cut": float(cut), "flagged": int(f42.sum()), "neo": int((f42 & y).sum()),
                         "fa": int((f42 & ~y).sum()), "fa_far": int((f42 & far).sum()),
                         "contamination": float((f42 & ~y).sum() / f42.sum())},
        "priora_p05": {"flagged": int(h.sum()), "completeness": float((h & y).sum() / y.sum()),
                     "contamination": float((h & ~y).sum() / max(h.sum(), 1))},
    }


def figs(D, d2key, tag, title):
    p, s, y, q = D["p"], D[d2key], D["y"], D["q"]
    lab = {"raw": "raw", "noid": "NoID"}[d2key]
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
                         "xtick.direction": "in", "ytick.direction": "in", "savefig.bbox": "tight"})
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.6), layout="constrained")
    # Fig 1: reliability (priora), linear bins of 0.1
    e = np.linspace(0, 1, 11); k = np.clip(np.digitize(p, e) - 1, 0, 9)
    for j in range(10):
        m = k == j
        if m.sum() == 0:
            continue
        obs, pr = y[m].mean(), p[m].mean(); err = np.sqrt(max(p[m] * (1 - p[m])).sum()) / m.sum() if False else np.sqrt((p[m] * (1 - p[m])).sum()) / m.sum()
        ax[0, 0].errorbar(pr, obs, yerr=err, fmt="o", color=C42, ms=4, capsize=2)
        ax[0, 0].text(pr, obs + 0.04, str(int(m.sum())), fontsize=6.5, ha="center")
    ax[0, 0].plot([0, 1], [0, 1], ":", color=CINK, lw=0.8)
    ax[0, 0].set(xlim=(0, 1), ylim=(0, 1.08), xlabel="mean predicted P(NEO) in bin", ylabel="observed NEO fraction",
                 title=f"(1) priora reliability  ΣP/N = {p.sum() / y.sum():.2f}")
    # Fig 2: ROC
    for sc, c, l in ((p, C42, "priora"), (s, CD2, f"digest2 {lab}")):
        o = np.argsort(-sc); tpr = np.cumsum(y[o]) / y.sum(); fpr = np.cumsum(~y[o]) / (~y).sum()
        ax[0, 1].plot(fpr, tpr, color=c, lw=1.2, label=f"{l} (AUC {auc(y, sc):.4f})")
    ax[0, 1].set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1.02), xlabel="false-positive rate", ylabel="completeness",
                 title="(2) ROC"); ax[0, 1].legend(loc="lower right", fontsize=7.5, frameon=False)
    # Fig 3: predicted vs observed NEOs per score bin, both codes on P = score/100
    e3 = np.linspace(0, 1, 11)
    for sc, c, l, off in ((p, C42, "priora", -0.012), (s / 100, CD2, f"digest2 {lab}/100", 0.012)):
        kk = np.clip(np.digitize(sc, e3) - 1, 0, 9)
        pred = np.bincount(kk, weights=sc, minlength=10); obs = np.bincount(kk, weights=y, minlength=10)
        ax[1, 0].plot((e3[:-1] + 0.05) + off, pred, "_", color=c, ms=10, mew=2)
        ax[1, 0].plot((e3[:-1] + 0.05) + off, obs, "o", color=c, ms=3.5, label=f"{l}: predicted (bar) vs observed (dot)")
    ax[1, 0].set(yscale="symlog", xlim=(0, 1), xlabel="score / probability bin", ylabel="NEOs in bin",
                 title="(3) predicted vs observed NEOs"); ax[1, 0].legend(fontsize=6.8, frameon=False, loc="upper left")
    # Fig 4: scatter
    xs = np.clip(p, 1e-8, 1); inner, outer = ~y & (q < 4.5), ~y & (q >= 4.5)
    a4 = ax[1, 1]
    a4.scatter(xs[inner], s[inner], s=2, alpha=0.18, color=COTH, lw=0, rasterized=True, label="non-NEO, $q<4.5$ AU")
    a4.scatter(xs[outer], s[outer], s=5, alpha=0.75, color=COUT, lw=0, rasterized=True, label="non-NEO, $q\\geq4.5$ AU")
    a4.scatter(xs[y], s[y], s=9, alpha=0.95, color=CD2, lw=0, label="NEO")
    a4.axhline(THRESH, color=CINK, lw=0.9, ls="--"); a4.axvline(0.1, color=CINK, lw=0.9, ls=":")
    a4.set(xscale="log", xlim=(1e-8, 1.6), ylim=(-2, 102), xlabel="priora P(NEO)", ylabel=f"digest2 {lab} NEO score",
           title="(4) digest2 vs priora"); a4.legend(fontsize=6.8, frameon=False, loc="center left", markerscale=2)
    fig.suptitle(title, fontsize=10)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"figs1to4_{tag}.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    A = load("gate_v0_hext30", "digest2_comparison_hext30/gate_v0_hext30")
    AU = load("gate_v0_hext30", "digest2_comparison_hext30/gate_v0_hext30", priora_scores="gate_v0_hext30_scoreUnk")
    UU = load("gate_v0unk_hext30", "digest2_comparison_unk")
    cases = {
        "P": ("sample All · priora All · digest2 raw (paper)", A, "raw"),
        "N": ("sample All · priora All · digest2 NoID", A, "noid"),
        "U": ("sample All · priora Unk · digest2 NoID", AU, "noid"),
        "M": ("sample Unk · priora Unk · digest2 NoID", UU, "noid"),
        "Mraw": ("sample Unk · priora Unk · digest2 raw", UU, "raw"),
    }
    res = {}
    for k, (title, D, key) in cases.items():
        res[k] = {"label": title} | metrics(D, key)
        figs(D, key, k, title)
        r = res[k]
        print(f"{k:5s} {title:42s} NEO {r['n_neo']:4d} AUC {r['auc_priora']:.4f}/{r['auc_d2']:.4f} "
              f"dAUC {r['delta_auc']:+.4f} {np.round(r['delta_ci'], 4)} pred/obs {r['pred_obs_priora']:.2f}/{r['pred_obs_d2']:.2f} "
              f"d2>=65 {r['d2_ge65']['flagged']}/{r['d2_ge65']['fa']}FA({r['d2_ge65']['fa_far']} far) "
              f"compl {r['d2_ge65']['completeness']:.3f} contam {r['d2_ge65']['contamination']:.3f} | priora match "
              f"p>={r['priora_matched']['cut']:.3f} {r['priora_matched']['flagged']}/{r['priora_matched']['fa']}FA "
              f"contam {r['priora_matched']['contamination']:.3f}")
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
