#!/usr/bin/env python3
"""Turn the D-19 validation artifacts into the tables of REPORT.md.

Reads (all produced by the two drivers in this directory, plus the gate run):

    neocp_rescore.json    the 11 H-wall tracklets under four density pairs
    neocp_delta_all.json  all 499 pilot tracklets under the same four
    gate_delta.npz        the gate_v0 catalogue (20k) under the same four
    ../gate_v0_hext30/summary.json   the extended-mode gate run

and prints Markdown.  No fitting, no bootstrapping — every number here is a
direct statistic of the stored arrays, so the report can be regenerated and
diffed.

    python validation/output/h_extension/analyze.py > /tmp/tables.md
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GATE_HEXT = os.path.join(os.path.dirname(HERE), "gate_v0_hext30")
KINDS = ("truncated", "extended", "extended_pchip", "clamped")


def _pct(x):
    return "-" if not np.isfinite(x) else f"{x:.4g}"


def wall_table() -> None:
    b = json.load(open(os.path.join(HERE, "neocp_rescore.json")))
    d, idx = b["designations"], b["indices"]
    print("### The 11 H-wall tracklets, re-scored\n")
    print("| designation | true NEO | q [AU] | V | rate [deg/d] | "
          "p (extended) | p (ext+pchip) | p (clamp) | p (policy-only) | "
          "status (policy-only) |")
    print("|---|:--:|---:|---:|---:|---:|---:|---:|---:|---|")
    for k, i in enumerate(idx):
        s = str(i)
        r = {c: b["configs"][c][s] for c in KINDS}
        print(f"| {d[k]} | {'yes' if b['is_neo'][k] else 'no'} "
              f"| {b['true_q_au'][k]:.3f} | {b['v_mag'][k]:.2f} "
              f"| {b['rate_deg_day'][k]:.2f} "
              f"| {r['extended']['p_neo']:.6f} "
              f"| {r['extended_pchip']['p_neo']:.6f} "
              f"| {r['clamped']['p_neo']:.6f} "
              f"| {r['truncated']['p_neo']:.6f} "
              f"| `{r['truncated']['status']}` |")

    print("\n### The D-19 trigger's mechanism evidence on these 11\n")
    print("| designation | measure frac >= support end | node-count frac | "
          "min implied H | n evaluable nodes | fires at 0.95? | fires at 1.0? |")
    print("|---|---:|---:|---:|---:|:--:|:--:|")
    mf = []
    for k, i in enumerate(idx):
        r = b["configs"]["truncated"][str(i)]
        fr = r["h_measure_frac_above_support"]
        nf = r["n_h_at_or_above_25p5"] / max(r["n_h_finite"], 1)
        mf.append(fr)
        print(f"| {d[k]} | {fr:.6f} | {nf:.6f} | {r['min_node_H']:.3f} "
              f"| {r['n_h_finite']} | {'yes' if fr >= 0.95 else '**NO**'} "
              f"| {'yes' if fr >= 1.0 else '**NO**'} |")
    print(f"\nMinimum measure fraction over the 11: **{min(mf):.6f}**; "
          f"**{sum(1 for x in mf if x < 1.0)} of 11 are below 1.0**, which is "
          "why the criterion is 0.95 and not 1.0.")

    print("\n### Lambda_NEO and the residual tail above H = 30\n")
    print("| designation | Lambda_NEO ext | Lambda_NEO clamp | rel. diff | "
          "Lambda_other (all four) | f(>25.5) | **f(>30)** | f(>34) | "
          "max implied H |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, i in enumerate(idx):
        s = str(i)
        e = b["configs"]["extended"][s]
        c = b["configs"]["clamped"][s]
        rel = ((e["lambda_neo"] - c["lambda_neo"]) / c["lambda_neo"]
               if c["lambda_neo"] > 0 else float("nan"))
        print(f"| {d[k]} | {e['lambda_neo']:.6g} | {c['lambda_neo']:.6g} "
              f"| {rel:+.2e} | {e['lambda_other']:.0f} "
              f"| {e['ar_frac_H_above_25p5']:.4f} "
              f"| **{e['ar_frac_H_above_30']:.4f}** "
              f"| {e['ar_frac_H_above_34']:.4f} | {e['max_node_H']:.3f} |")


def neocp_full() -> None:
    f = json.load(open(os.path.join(HERE, "neocp_delta_all.json")))
    C, n = f["configs"], f["n"]
    y = np.array(f["is_neo"])
    p = {k: np.array(C[k]["p_neo"]) for k in KINDS}
    pol = {k: np.array(C[k]["policy_assigned"]) for k in KINDS}
    lik = ~pol["truncated"]           # scorable by likelihood under D-4

    print(f"\n### All {n} pilot tracklets\n")
    print("| pair | non-finite p | policy-assigned | median &#124;dp&#124; vs "
          "D-4 (488 likelihood-scored) | p90 | max | n(&#124;dp&#124;>1e-3) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for k in KINDS:
        dd = np.abs(p[k][lik] - p["truncated"][lik])
        print(f"| {k} | {int((~np.isfinite(p[k])).sum())} "
              f"| {int(pol[k].sum())} | {np.median(dd):.3e} "
              f"| {np.percentile(dd, 90):.3e} | {dd.max():.3e} "
              f"| {int((dd > 1e-3).sum())} |")
    for a, bb in (("extended", "clamped"), ("extended", "extended_pchip")):
        dd = np.abs(p[a] - p[bb])
        print(f"\n`{a}` vs `{bb}` over all {n}: median |dp| = "
              f"{np.median(dd):.3e}, p90 = {np.percentile(dd, 90):.3e}, "
              f"max = {dd.max():.3e}, n(>1e-3) = {int((dd > 1e-3).sum())}")

    print("\n### Calibration on the pilot sample (pred/obs NEO count)\n")
    print("| pair | predicted | observed | ratio | n | "
          "predicted excl. policy | observed excl. policy | ratio |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in KINDS:
        m = np.isfinite(p[k])
        mk = m & ~pol[k]
        print(f"| {k} | {p[k][m].sum():.1f} | {int(y[m].sum())} "
              f"| {p[k][m].sum() / y[m].sum():.3f} | {int(m.sum())} "
              f"| {p[k][mk].sum():.1f} | {int(y[mk].sum())} "
              f"| {p[k][mk].sum() / max(y[mk].sum(), 1):.3f} |")


def gate_delta() -> None:
    path = os.path.join(HERE, "gate_delta.npz")
    if not os.path.exists(path):
        print("\n(gate_delta.npz absent)")
        return
    z = np.load(path)
    n = z["index"].size
    p = {k: z[f"p_{k}"] for k in KINDS}
    ar25 = z["ar_frac_H_above_25p5"]
    ar30 = z["ar_frac_H_above_30"]
    maxH = z["max_node_H"]
    y = z["is_neo"].astype(bool)
    st = {k: z[f"status_{k}"] for k in KINDS}

    print(f"\n### gate_v0 catalogue, N = {n}: scoring-mode comparison\n")
    print("Statuses (the D-19 policy on a synthetic catalogue):\n")
    for k in KINDS:
        u, c = np.unique(st[k], return_counts=True)
        pol = (z[f"policy_{k}"] if f"policy_{k}" in z.files
               else np.zeros(n, dtype=bool))
        print(f"* `{k}`: " + ", ".join(f"{a} x {b}" for a, b in zip(u, c))
              + f"   (policy_assigned: {int(pol.sum())})")

    print("\nAdmissible-region measure above the two boundaries "
          "(D-4 pair, DESIGN §2.6(b)):\n")
    for name, arr, thr in (("H > 25.5", ar25, 25.5), ("H > 30", ar30, 30.0)):
        print(f"* {name}: median {np.median(arr):.3e}, mean "
              f"{np.mean(arr):.3e}, frac>1e-3 {np.mean(arr > 1e-3):.4f}, "
              f"frac>1e-2 {np.mean(arr > 1e-2):.4f}, "
              f"frac>1e-1 {np.mean(arr > 1e-1):.4f}, max {arr.max():.4f}")
    print(f"* max implied H over the whole catalogue: {np.nanmax(maxH):.3f}; "
          f"p50/p90/p99 = {np.nanpercentile(maxH, 50):.2f} / "
          f"{np.nanpercentile(maxH, 90):.2f} / "
          f"{np.nanpercentile(maxH, 99):.2f}")

    print("\n| comparison | median &#124;dp&#124; | p90 | p99 | max | "
          "n(>1e-6) | n(>1e-3) | n(>1e-2) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    pairs = [("extended", "truncated"), ("clamped", "truncated"),
             ("extended", "clamped"), ("extended_pchip", "extended"),
             ("extended_pchip", "truncated")]
    for a, bb in pairs:
        dd = np.abs(p[a] - p[bb])
        dd = dd[np.isfinite(dd)]
        print(f"| {a} - {bb} | {np.median(dd):.3e} "
              f"| {np.percentile(dd, 90):.3e} | {np.percentile(dd, 99):.3e} "
              f"| {dd.max():.3e} | {int((dd > 1e-6).sum())} "
              f"| {int((dd > 1e-3).sum())} | {int((dd > 1e-2).sum())} |")

    print("\n`extended` - `clamped` conditioned on the measure above H = 25.5 "
          "(the prediction: 0 where the truncation never bound):\n")
    print("| f(H>25.5) bin | n | median &#124;dp(ext-trunc)&#124; | "
          "max &#124;dp(ext-trunc)&#124; | median &#124;dp(ext-clamp)&#124; | "
          "max &#124;dp(ext-clamp)&#124; | median f(H>30) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    edges = [0.0, 1e-12, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.01]
    labels = ["exactly 0", "(0, 1e-6]", "(1e-6, 1e-4]", "(1e-4, 1e-3]",
              "(1e-3, 1e-2]", "(1e-2, 1e-1]", "> 1e-1"]
    d_et = np.abs(p["extended"] - p["truncated"])
    d_ec = np.abs(p["extended"] - p["clamped"])
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        m = (ar25 > lo) & (ar25 <= hi) if k else (ar25 <= 0.0)
        if not m.any():
            print(f"| {labels[k]} | 0 | - | - | - | - | - |")
            continue
        print(f"| {labels[k]} | {int(m.sum())} | {np.median(d_et[m]):.3e} "
              f"| {d_et[m].max():.3e} | {np.median(d_ec[m]):.3e} "
              f"| {d_ec[m].max():.3e} | {np.median(ar30[m]):.3e} |")

    # Where they differ: characterise the tail of extended - truncated.
    order = np.argsort(-d_et)[:12]
    print("\nThe 12 largest `extended - truncated` shifts, and what they "
          "look like:\n")
    print("| # | V | true NEO | f(H>25.5) | f(H>30) | p trunc | p ext | "
          "p clamp | dp(ext-trunc) |")
    print("|---|---:|:--:|---:|---:|---:|---:|---:|---:|")
    for j in order:
        print(f"| {int(z['index'][j])} | {z['v_mag'][j]:.2f} "
              f"| {'yes' if y[j] else 'no'} | {ar25[j]:.4f} | {ar30[j]:.4f} "
              f"| {p['truncated'][j]:.6f} | {p['extended'][j]:.6f} "
              f"| {p['clamped'][j]:.6f} | {p['extended'][j] - p['truncated'][j]:+.3e} |")

    # AUC, unweighted (gate mode carries unit weights).
    def auc(s):
        m = np.isfinite(s)
        o = np.argsort(s[m], kind="mergesort")
        yy = y[m][o]
        r = np.empty(yy.size, dtype=float)
        ss = s[m][o]
        i = 0
        while i < ss.size:
            j = i
            while j + 1 < ss.size and ss[j + 1] == ss[i]:
                j += 1
            r[i:j + 1] = 0.5 * (i + j) + 1.0
            i = j + 1
        npos, nneg = yy.sum(), (~yy).sum()
        return (r[yy].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

    print("\nROC AUC on the gate_v0 catalogue (same tracklets, four models):\n")
    for k in KINDS:
        print(f"* `{k}`: {auc(p[k]):.6f}")


def gate_summary() -> None:
    path = os.path.join(GATE_HEXT, "summary.json")
    if not os.path.exists(path):
        print("\n(extended-mode gate summary absent)")
        return
    s = json.load(open(path))
    r = s["reliability"]
    print("\n### Extended-mode gate run (DESIGN §4.1)\n")
    print(f"* run dir: `{s['run_dir']}`")
    print(f"* N = {s['n_scored']}, true NEOs {s['n_neo_true']}, weighted NEO "
          f"fraction {s['weighted_neo_fraction']:.5f}")
    print(f"* generator/scorer h_extend_to = "
          f"{s['generation_config'].get('h_extend_to')} / "
          f"{s['scoring_config'].get('h_extend_to')}")
    print(f"* undefined p: {s['n_undefined_p']}; policy-assigned: "
          f"{s['n_policy_assigned_p']}")
    print(f"* Z_tot = {r['global_z']:.4f} (bootstrap 99th pct "
          f"{r['global_z_threshold']:.3f}, p = {r['global_z_p_value']:.4f})")
    o = r.get("odds_scale", {})
    print(f"* c_hat = {o.get('c_hat', float('nan')):.4f} "
          f"[{o.get('lo', float('nan')):.4f}, "
          f"{o.get('hi', float('nan')):.4f}]")
    print(f"* max|z| = {r['max_abs_z']:.4f} vs bootstrap 99th pct "
          f"{r['max_abs_z_threshold']:.3f} (p = {r['max_abs_z_p_value']:.4f})")
    print(f"* AUC = {s['auc']:.6f}")
    c = s["collapse"]
    print(f"* HPD KS D: nearest-node {c['hpd_ks_D']:.5f}, exact "
          f"{c['hpd_exact_ks_D']:.5f} (max allowed "
          f"{c['hpd_ks_D_max_allowed']}), gap {c['hpd_estimator_gap_D']:.5f}")
    print(f"* coverage {c['coverage_frac_truth_in_AR']:.6f}; truth density "
          f"positive {c['frac_truth_density_positive']:.6f}")
    d = s["diagnostics_2_6"]
    print(f"* §2.6(b) measure above H_max^model (= 30 here): median "
          f"{_pct(d['median_ar_frac_H_above_Hmax'])}, mean "
          f"{_pct(d['mean_ar_frac_H_above_Hmax'])}, "
          f"frac>1e-3 {_pct(d['frac_tracklets_ar_frac_H_above_gt_1e-3'])}, "
          f"frac>1e-2 {_pct(d['frac_tracklets_ar_frac_H_above_gt_1e-2'])}, "
          f"frac>1e-1 {_pct(d['frac_tracklets_ar_frac_H_above_gt_1e-1'])}")
    print("* verdicts: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}"
                                     for k, v in s["verdicts"].items()))
    print(f"* **OVERALL {'PASS' if s['overall_pass'] else 'FAIL'}**")


if __name__ == "__main__":
    wall_table()
    neocp_full()
    gate_summary()
    gate_delta()
