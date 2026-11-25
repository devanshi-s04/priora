"""Recompute every number in the simulation-based sections of the priora paper (draft 5, 2026-11-12) (Sections 2.8, 3, 5, 6, 7, 8, 12 and
Appendix C) from the stored outputs, and compare each with the value printed in the paper.

Nothing is re-scored: every value comes from stored arrays (scores.npz / gen.npz / digest2_scores.npz / npz tables)
or from stored summary JSON. Writes check_sim_numbers.md (one row per number) and check_sim_numbers.json next to this
file, and prints the rows that do not match.

Run from the repository root:  PYTHONPATH=src python analysis/numbers/check_sim_numbers.py
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "validation"))
from priora.population import read_digest2_model, Digest2Histogram, Q_EDGES, H_EDGES, E_EDGES  # noqa: E402
from self_consistency import fit_odds_scale                                                     # noqa: E402
from validation.digest2_comparison import auc_bootstrap                                          # noqa: E402

O = REPO / "validation/output"
ROWS = []


def _half_unit(v):
    """Half a unit in the last printed digit of a paper value (the rounding tolerance)."""
    t = repr(float(v)) if not float(v).is_integer() else "0"
    return 0.5 * 10 ** -len(t.split(".")[1]) + 1e-12 if "." in t and not float(v).is_integer() else 1e-9


def chk(sec, what, paper, got, tol, src, fmt="{:.4g}"):
    """Record one number. tol='auto': equal after rounding to the digits printed in the paper."""
    pa, ga = np.atleast_1d(np.asarray(paper, float)), np.atleast_1d(np.asarray(got, float))
    tl = np.array([_half_unit(v) for v in pa]) if tol == "auto" else tol
    ok = bool(np.all(np.abs(ga - pa) <= tl))
    g = fmt.format(got) if np.ndim(got) == 0 else str([fmt.format(x) for x in got])
    ROWS.append(dict(section=sec, quantity=what, paper=str(paper), computed=g, status="OK" if ok else "MISMATCH", source=src))


def bound(sec, what, rel, limit, got, src, fmt="{:.4g}"):
    """Record a claim of the form 'X < limit' or 'X <= limit' (rel is '<', '<=', '>', '>=')."""
    ok = {"<": got < limit, "<=": got <= limit, ">": got > limit, ">=": got >= limit}[rel]
    ROWS.append(dict(section=sec, quantity=what, paper=f"{rel} {limit}", computed=fmt.format(got), status="OK" if ok else "MISMATCH", source=src))


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ztot(p, y):
    return (y - p).sum() / np.sqrt((p * (1 - p)).sum())


# ---------------------------------------------------------------------------------------------- Section 3: model
T = read_digest2_model(str(REPO / "data/digest2.model.csv"))
neo_q = Q_EDGES[1:] <= 1.3 + 1e-9
A = T[("All", "SS")]
chk("3 / Table 1", "total objects", 14542147, A.sum(), 0.5, "data/digest2.model.csv via population.read_digest2_model")
chk("3 / Table 1", "NEO objects", 277469, A[neo_q].sum(), 0.5, "same")
chk("3 / Table 1", "other objects", 14264678, A[~neo_q].sum(), 0.5, "same")
chk("3 / Table 1", "NEO fraction (%)", 1.9, 100 * A[neo_q].sum() / A.sum(), 0.05, "same")
chk("3 / Table 1", "cells (q x e x i x H)", 45936, int(np.prod(A.shape[:3]) * A.shape[3]), 0, "same")
nX, oX = Digest2Histogram.neo_other_pair(str(REPO / "data/digest2.model.csv"), "All", h_extend_to=30.0)
chk("3 / Table 1", "extended NEO / other", [742145, 14868899], [nX.total_count, oX.total_count], 0.5, "Digest2Histogram(h_extend_to=30).total_count")
chk("3 / Table 1", "extended NEO fraction (%)", 4.8, 100 * nX.total_count / (nX.total_count + oX.total_count), 0.05, "same")
chk("3.4", "NEO fraction in highest-H bin (%)", 55.8, 100 * A[neo_q][..., -1].sum() / A[neo_q].sum(), 0.05, "read_digest2_model")
top_e = E_EDGES[-2] < 1.0 < E_EDGES[-1]
frac_unbound = lambda X: 0.5 * X[..., -1, :, :].sum() / X.sum()   # top e bin 0.9-1.1 is half above e = 1 (uniform in e)
chk("3.4", "NEO draws unbound, All (%)", 1.93, 100 * frac_unbound(A[neo_q]), 0.02, "bin counts (DESIGN.md 4.1: 1.92% exact, 1.93% MC)")
U = T[("Unk", "SS")]
chk("5.2", "NEO draws unbound, Unk (%)", 2.2, 100 * frac_unbound(U[neo_q]), 0.05, "bin counts (DESIGN.md 4.1: 2.23% exact, 2.24% MC)")

# ---------------------------------------------------------------------------------------------- Section 5: self-consistency
UNK = ["gate_v0unk200k_hext30", "gate_v0unk200k_s3_hext30", "gate_v0unk200k_s4_hext30", "gate_v0unk200k_s5y27_hext30"]
ALL = ["calibration_1e5_hext30", "calibration_1e5_hext30_seed2"]


def load_gate(run):
    z = np.load(O / run / "scores.npz"); g = np.load(O / run / "gen.npz"); i = z["index"]
    p = z["p_neo"]; y = g["is_neo"][i].astype(bool)
    pol = (z["lambda_neo"] == 0) & (z["lambda_other"] == 0) & np.isfinite(p)
    inr = z["truth_rho_in_range"].astype(bool)
    return dict(p=p, y=y, pol=pol, inr=inr, nan=np.isnan(p), g=g, i=i, z=z)


rows = {}
P_all, Y_all = [], []
tot = dict(beyond=0, nan=0, pol=0, p1=0, p1neo=0, nonneo_p1_beyond=0)
pol_q, nan_q, beyond_nn = [], [], []
for k, run in enumerate(UNK):
    d = load_gate(run)
    keep = ~d["nan"] & d["inr"] & ~d["pol"]
    p, y = d["p"][keep], d["y"][keep].astype(float)
    o = fit_odds_scale(p, y)
    rows[k] = (int(y.sum()), p.sum(), ztot(p, y), o["c_hat"], o["lo"], o["hi"])
    P_all.append(p); Y_all.append(y)
    tot["beyond"] += int((~d["nan"] & ~d["inr"]).sum()); tot["nan"] += int(d["nan"].sum()); tot["pol"] += int(d["pol"].sum())
    tot["p1"] += int((p == 1).sum()); tot["p1neo"] += int(((p == 1) & (y == 1)).sum())
    m = (~d["y"]) & (d["p"] == 1)
    tot["nonneo_p1_beyond"] += int((m & ~d["inr"]).sum())
    el = d["g"]["elements"]
    beyond_nn += [(el[j, 0] * (1 - el[j, 1]), d["g"]["H"][j]) for j in d["i"][m]]
    pol_q += list((el[:, 0] * (1 - el[:, 1]))[d["i"][d["pol"]]]); nan_q += list((el[:, 0] * (1 - el[:, 1]))[d["i"][d["nan"]]])
paper_tab = {0: (2591, 2599.2, -0.55, 0.96, 0.84, 1.10), 1: (2675, 2654.6, 1.40, 1.10, 0.96, 1.25),
             2: (2642, 2650.3, -0.57, 0.96, 0.84, 1.10), 3: (2642, 2638.6, 0.23, 1.02, 0.89, 1.16)}
for k in range(4):
    chk("5.4 / Table 2", f"Unk {k+1} obs, exp, Z_tot, c, lo, hi", list(paper_tab[k]), list(rows[k]), "auto",
        f"validation/output/{UNK[k]}/scores.npz, gen.npz; self_consistency.fit_odds_scale", fmt="{:.4f}")
p, y = np.concatenate(P_all), np.concatenate(Y_all); o = fit_odds_scale(p, y)
chk("5.4 / Table 2", "Unk pooled obs, exp, Z_tot, c, lo, hi", [10550, 10542.6, 0.25, 1.01, 0.94, 1.08],
    [y.sum(), p.sum(), ztot(p, y), o["c_hat"], o["lo"], o["hi"]], "auto", "four Unk runs pooled", fmt="{:.4f}")
chk("5.4", "pooled calibration sample size", 799906, p.size, 0, "four Unk runs")
chk("5.2", "NEO fraction of accepted tracklets (%)", 1.3, 100 * y.mean(), 0.05, "four Unk runs")
chk("5.4", "excluded beyond 100 AU / no posterior / triage (Unk)", [52, 37, 5], [tot["beyond"], tot["nan"], tot["pol"]], 0, "scores.npz truth_rho_in_range, p_neo, lambdas")
chk("5.4", "non-NEOs at P=1 beyond 100 AU", 20, tot["nonneo_p1_beyond"], 0, "same")
bq, bH = np.array(beyond_nn).T
bound("5.4", "those 20: max H", "<", 4, bH.max(), "gen.npz H", fmt="{:.2f}")
bound("5.4", "those 20: min q (AU)", ">", 30, bq.min(), "gen.npz elements", fmt="{:.2f}")
chk("5.4", "tracklets at P=1 within domain, all NEOs", [8828, 8828], [tot["p1"], tot["p1neo"]], 0, "pooled sample")
bound("5.4 / 2.8", "no-posterior NEOs: max q (AU)", "<", 0.004, max(nan_q), "gen.npz elements", fmt="{:.5f}")
bound("2.8 / 5.4", "triage NEOs: max q (AU)", "<", 0.003, max(pol_q), "gen.npz elements", fmt="{:.5f}")
ks = [json.load(open(O / r / "summary.json"))["collapse"] for r in UNK]
bound("5.4", "max KS D over Unk runs", "<=", 0.064, max(max(c["hpd_ks_D"], c["hpd_exact_ks_D"]) for c in ks), "summary.json collapse.hpd_ks_D / hpd_exact_ks_D", fmt="{:.5f}")
chk("5.4", "median posterior mass within x2 of true rho (%)", 99.98, 100 * min(c["median_mass_within_factor2_rho"] for c in ks), 0.005, "summary.json collapse")
chk("5.3", "c_hat 95% half-width, one run / pooled (%)", [14, 7], [100 * 1.96 * np.mean([0.0688, 0.0665, 0.0695, 0.0662]), 100 * (o["hi"] / o["lo"]) ** 0.5 - 100], 1.0, "fit_odds_scale log_c_se (per-run summary.json)")
Pa, Ya = [], []
for run in ALL:
    d = load_gate(run); keep = ~d["nan"] & d["inr"] & ~d["pol"]; Pa.append(d["p"][keep]); Ya.append(d["y"][keep].astype(float))
pa, ya = np.concatenate(Pa), np.concatenate(Ya); oa = fit_odds_scale(pa, ya)
chk("5.4 / Table 2", "All A+B obs, exp, Z_tot, c, lo, hi", [1719, 1743.8, -1.42, 0.92, 0.82, 1.03],
    [ya.sum(), pa.sum(), ztot(pa, ya), oa["c_hat"], oa["lo"], oa["hi"]], "auto", "calibration_1e5_hext30{,_seed2}", fmt="{:.4f}")
low = (p < 0.1)
chk("12.2", "NEOs at P<0.1: observed / predicted", [540, 535.7], [y[low].sum(), p[low].sum()], 0.05, "pooled Unk sample", fmt="{:.1f}")

# low-P NEO characteristics (Section 12.2)
G = []
for run in UNK:
    d = load_gate(run); keep = ~d["nan"] & d["inr"] & ~d["pol"] & d["y"]
    j = d["i"][keep]; g = d["g"]; el = g["elements"][j]
    rate = np.degrees(np.hypot(g["ra_rate"][j] * np.cos(g["dec"][j]), g["dec_rate"][j]))
    G.append(np.c_[d["p"][keep], el[:, 0] * (1 - el[:, 1]), rate, g["rho_true"][j]])
G = np.vstack(G); pn, qn, rn, dn = G.T; lo = pn < 0.1
chk("12.2", "median sky rate low-P NEOs / all NEOs (deg/day)", [0.41, 1.5], [np.median(rn[lo]), np.median(rn)], 0.01, "gen.npz ra_rate, dec_rate", fmt="{:.3f}")
chk("12.2", "median distance low-P NEOs / all NEOs (AU)", [1.27, 1.04], [np.median(dn[lo]), np.median(dn)], 0.005, "gen.npz rho_true", fmt="{:.3f}")
b = (qn >= 1.2) & (qn < 1.3)
chk("12.2", "fraction of 1.2<=q<1.3 NEOs with P<0.1 (quarter)", 0.25, lo[b].mean(), 0.02, "pooled Unk", fmt="{:.3f}")
chk("12.2", "NEOs faster than 1 deg/day with P<0.1", 0, int((lo & (rn > 1)).sum()), 0, "pooled Unk")

# ---------------------------------------------------------------------------------------------- Section 6: digest2 comparison
RUN, CMP = O / "gate_v0unk_hext30", O / "digest2_comparison_unk"
with np.load(RUN / "scores.npz") as z, np.load(RUN / "gen.npz") as g, np.load(CMP / "digest2_scores.npz") as dd:
    idx = z["index"]; keep = np.isin(idx, dd["index"]); idx = idx[keep]
    p42 = z["p_neo"][keep]; fr = z["ar_frac_h_above"][keep]; s2 = dd["noid_neo"].astype(float); ok2 = dd["ok"].astype(bool)
    yb = g["is_neo"][idx].astype(bool); el = g["elements"][idx]; rho = g["rho_true"][idx]
    u = np.c_[np.cos(g["dec"][idx]) * np.cos(g["ra"][idx]), np.cos(g["dec"][idx]) * np.sin(g["ra"][idx]), np.sin(g["dec"][idx])]
    rh = np.linalg.norm(g["R_obs"][idx] + rho[:, None] * u, axis=1)
good = np.isfinite(p42) & ok2 & np.isfinite(s2)
p42, s2, yb, el, fr, rh = p42[good], s2[good], yb[good], el[good], fr[good], rh[good]
q = el[:, 0] * (1 - el[:, 1]); pd2 = np.clip(s2 / 100, 0, 1)
src6 = "validation/output/gate_v0unk_hext30 + digest2_comparison_unk (run M)"
chk("6.1", "tracklets / NEOs", [19999, 287], [p42.size, yb.sum()], 0, src6)
chk("6.1", "tracklets with >1% of measure beyond H=30 (%)", 2, 100 * (fr > 0.01).mean(), 0.1, src6 + " ar_frac_h_above", fmt="{:.2f}")
chk("6.2", "priora predicted / digest2 predicted (all)", [284.0, 1349], [p42.sum(), pd2.sum()], 0.5, src6, fmt="{:.1f}")
chk("6.2", "digest2 / observed ratio", 4.7, pd2.sum() / yb.sum(), 0.05, src6, fmt="{:.2f}")
lo2, lo4 = pd2 < 0.1, p42 < 0.1
chk("6.2", "digest2 <10: n, predicted, observed", [17263, 290.5, 9], [lo2.sum(), pd2[lo2].sum(), yb[lo2].sum()], 0.05, src6, fmt="{:.1f}")
chk("6.2", "digest2 >=10: n, predicted, observed", [2736, 1058, 278], [(~lo2).sum(), pd2[~lo2].sum(), yb[~lo2].sum()], 0.5, src6, fmt="{:.1f}")
chk("6.2", "priora <0.1 / >=0.1: predicted, observed", [14.1, 17, 269.9, 270], [p42[lo4].sum(), yb[lo4].sum(), p42[~lo4].sum(), yb[~lo4].sum()], 0.05, src6, fmt="{:.1f}")
chk("6.2", "priora P=1 tracklets, all NEOs", [237, 237], [(p42 == 1).sum(), yb[p42 == 1].sum()], 0, src6)
a42, ad2 = auc(p42, yb), auc(s2, yb)
chk("6.3", "AUC priora / digest2", [0.9945, 0.9846], [a42, ad2], 0.00005, src6, fmt="{:.5f}")
res_M = json.load(open(REPO / "validation/unk_noid/results.json"))["M"]
chk("6.3", "Delta AUC and 95% CI", [0.0099, 0.0040, 0.0176], [res_M["delta_auc"]] + list(res_M["delta_ci"]), 0.00005, "validation/unk_noid/results.json M", fmt="{:.5f}")
chk("6.3", "non-NEO : NEO ratio (69 to 1)", 69, (~yb).sum() / yb.sum(), 0.5, src6, fmt="{:.1f}")
sel_d2 = s2 >= 65
cut = np.sort(p42[yb])[::-1][yb[sel_d2].sum() - 1]; sel42 = p42 >= cut; sel5 = p42 >= 0.5
chk("6.4 / Table 3", "digest2>=65: NEO, non-NEO, compl %, contam %", [269, 287, 93.7, 51.6],
    [yb[sel_d2].sum(), (~yb[sel_d2]).sum(), 100 * yb[sel_d2].sum() / yb.sum(), 100 * (~yb[sel_d2]).mean()], 0.05, src6, fmt="{:.2f}")
chk("6.4 / Table 3", "priora matched cut value", 0.30, cut, 0.005, src6, fmt="{:.4f}")
chk("6.4 / Table 3", "priora P>=0.30: NEO, non-NEO, contam %", [269, 11, 3.9], [yb[sel42].sum(), (~yb[sel42]).sum(), 100 * (~yb[sel42]).mean()], 0.05, src6, fmt="{:.2f}")
chk("6.4 / Table 3", "priora P>=0.5: NEO, non-NEO, compl %, contam %", [264, 3, 92.0, 1.1],
    [yb[sel5].sum(), (~yb[sel5]).sum(), 100 * yb[sel5].sum() / yb.sum(), 100 * (~yb[sel5]).mean()], 0.05, src6, fmt="{:.2f}")
chk("6.3", "purity digest2>=65 / priora matched / priora P>=0.5 (%)", [48.4, 96.1, 98.9],
    [100 * yb[sel_d2].mean(), 100 * yb[sel42].mean(), 100 * yb[sel5].mean()], 0.05, src6, fmt="{:.2f}")
both = sel_d2 & sel42 & yb
chk("6.4", "NEOs selected by both / each only", [267, 2, 2], [both.sum(), (sel_d2 & yb & ~sel42).sum(), (sel42 & yb & ~sel_d2).sum()], 0, src6)
fa2 = sel_d2 & ~yb
chk("6.5", "digest2 FPs with q>=4.5 / q>=10", [213, 145], [(fa2 & (q >= 4.5)).sum(), (fa2 & (q >= 10)).sum()], 0, src6)
chk("12.1", "digest2 FPs with 1.67<=q<4.5", 58, (fa2 & (q >= 1.67) & (q < 4.5)).sum(), 0, src6)
chk("6.5", "heliocentric distance range of q>=10 FPs (AU)", [13, 89], [rh[fa2 & (q >= 10)].min(), rh[fa2 & (q >= 10)].max()], 0.5, src6 + " (R_obs + rho u)", fmt="{:.1f}")
bound("6.5", "max priora P among digest2 q>=4.5 FPs", "<", 0.1, p42[fa2 & (q >= 4.5)].max(), src6, fmt="{:.3f}")
fa42 = sel42 & ~yb
chk("6.5 / 12.1", "priora FPs: Mars-crossers / main-belt range / q>=4.5", [10, 1, 0],
    [(fa42 & (q >= 1.3) & (q < 1.67)).sum(), (fa42 & (q >= 1.67) & (q < 4.5)).sum(), (fa42 & (q >= 4.5)).sum()], 0, src6)
# All sample, digest2 raw, default 1" vs 0.1" observer error
gz = np.load(O / "gate_v0_hext30/gen.npz"); zz = np.load(O / "gate_v0_hext30/scores.npz")
fp = []
for cdir in ("digest2_comparison_hext30/gate_v0_hext30", "digest2_comparison_hext30_obserr0p1/gate_v0_hext30"):
    dd = np.load(O / cdir / "digest2_scores.npz"); ii = dd["index"]; yy = gz["is_neo"][ii].astype(bool)
    elx = gz["elements"][ii]; qq = elx[:, 0] * (1 - elx[:, 1]); ss = dd["raw_neo"]
    f = (ss >= 65) & ~yy; fp.append((f.sum(), (f & (qq >= 4.5)).mean() / f.mean()))
chk("6.5", "All/raw digest2 FPs at 1\" -> 0.1\", fraction q>=4.5 at 0.1\" (%)", [221, 276, 73], [fp[0][0], fp[1][0], 100 * fp[1][1]], 0.5,
    "digest2_comparison_hext30{,_obserr0p1}/gate_v0_hext30/digest2_scores.npz", fmt="{:.1f}")

# ---------------------------------------------------------------------------------------------- Section 7: astrometric noise
S45 = json.load(open(O / "astrometric_noise_hext30/summary.json"))
chk("7", "All sample: tracklets / NEOs", [20000, 170], [S45["n"], S45["n_neo"]], 0, "astrometric_noise_hext30/summary.json")
sig = {r["V"] if "V" in r else r.get("v"): r for r in S45["sigma_table"]} if isinstance(S45["sigma_table"][0], dict) else {}
chk("7", "median per-coordinate sigma (mas)", 35, S45["sigma_population"]["median_mas"], 0.5, "summary.json sigma_population")
chk("2.9", "45-min rate error / own motion, median (%)", 0.2, 100 * S45["sigma_population"]["rate_error_over_rate"]["p50"], 0.01, "summary.json")
chk("7", "99th pct |dP| at 45 min (< 7e-4)", 7e-4, S45["deltas"]["p_neo"]["all"]["abs_p99"], 0.4e-4, "summary.json deltas.p_neo.all.abs_p99", fmt="{:.2e}")
cross = [t for t in S45["threshold_stability"] if t["n_cross_any"] > 0]
chk("7", "tracklets crossing any threshold at 45 min (one, at P=0.7)", [1, 0.7], [sum(t["n_cross_any"] for t in cross), cross[0]["threshold"]], 0, "summary.json threshold_stability")
gz0 = np.load(O / "gate_v0_hext30/gen.npz"); z0 = np.load(O / "gate_v0_hext30/scores.npz")
y0 = gz0["is_neo"][z0["index"]].astype(bool); p0 = z0["p_neo"]
ARC = json.load(open(O / "astrometric_noise_arclength_hext30/summary.json"))
re = {int(r["dt_min"]): r["rate_error"]["rate_error_over_own_motion"]["p50"] for r in ARC["per_dt"]}
d2a = np.load(O / "digest2_comparison_hext30/gate_v0_hext30/digest2_scores.npz"); assert np.array_equal(d2a["index"], z0["index"])
n148 = int(((d2a["raw_neo"] >= 65) & y0).sum()); CUT7 = np.sort(p0[y0])[::-1][n148 - 1]
chk("7 / Table 4", "matched cut (digest2 raw>=65 completeness on All: 148/170) ~ 0.214", [148, 0.214], [n148, CUT7], "auto", "gate_v0_hext30 + digest2_comparison_hext30", fmt="{:.4f}")
paper_arc = {None: (166.1, 0.71, 6.3), 60: (166.2, 0.69, 6.4), 30: (166.3, 0.67, 6.4), 15: (170.2, -0.04, 8.7), 10: (176.0, -1.06, 12.5), 5: (197.9, -4.49, 25.6)}
paper_re = {60: 0.14, 30: 0.28, 15: 0.57, 10: 0.85, 5: 1.70}
for dt, (sp, zp, cp) in paper_arc.items():
    if dt is None:
        pp, src = p0, "gate_v0_hext30/scores.npz"
    else:
        nz = np.load(O / f"astrometric_noise_arclength_hext30/dt_{dt}min/noisy_scores.npz")
        assert np.array_equal(nz["index"], z0["index"]); pp, src = nz["p_neo"], f"astrometric_noise_arclength_hext30/dt_{dt}min/noisy_scores.npz"
    s = pp >= CUT7
    chk("7 / Table 4", f"arc {dt or 'none'}: sum P, Z_tot, contam % at the matched cut", [sp, zp, cp], [pp.sum(), ztot(pp, y0), 100 * (~y0[s]).mean()], [0.05, 0.005, 0.05], src, fmt="{:.2f}")
    bound("7 / Table 4", f"arc {dt or 'none'}: completeness >= 86.5 %", ">=", 86.5 - 0.05, 100 * y0[s].sum() / y0.sum(), src, fmt="{:.2f}")
    bound("7 / Table 4", f"arc {dt or 'none'}: completeness <= 87.1 %", "<=", 87.1 + 0.05, 100 * y0[s].sum() / y0.sum(), src, fmt="{:.2f}")
    if dt:
        chk("7 / Table 4", f"arc {dt}: median rate error / motion (%)", paper_re[dt], 100 * re[dt], 0.005, "arclength summary.json per_dt.rate_error", fmt="{:.3f}")
nz5 = np.load(O / "astrometric_noise_arclength_hext30/dt_5min/noisy_scores.npz")
new = (nz5["p_neo"] >= CUT7) & (p0 < CUT7) & ~y0
el0 = gz0["elements"][z0["index"]]; q0 = el0[:, 0] * (1 - el0[:, 1]); rho0 = gz0["rho_true"][z0["index"]]
tno = new & (q0 >= 10)   # 'trans-Neptunian' here: q >= 10 AU (EXTENDED_RERUN.md 7 quotes q 27-54 AU)
chk("7", "5-min new FPs / of which TNOs / their distance range (AU)", [41, 36, 31, 58], [new.sum(), tno.sum(), rho0[tno].min(), rho0[tno].max()], 0.5, "dt_5min noisy vs noiseless", fmt="{:.1f}")
CAD = REPO / "analysis/numbers/cadence_arcs.json"
if CAD.exists():
    c = json.load(open(CAD))
    chk("7", "LSST v5.3.7 yr-1 arcs: median (min), frac >=15 min, frac <10 min", [23, 0.70, 0.15], [c["median"], 1 - c["frac_lt"]["15"], c["frac_lt"]["10"]], 0.5, "analysis/numbers/cadence_arcs.py", fmt="{:.2f}")

# ---------------------------------------------------------------------------------------------- Section 8: NEOCP
N = np.load(O / "neocp_unk_ext/table5_inputs.npz"); yN = N["y"].astype(bool); pU = N["pU"]; noid = N["noid"]; qN = N["q"]; VN = N["V"]
srcN = "validation/output/neocp_unk_ext/table5_inputs.npz (rescore_neocp_unk.py)"
chk("8", "tracklets / NEOs / non-NEOs", [499, 305, 194], [yN.size, yN.sum(), (~yN).sum()], 0, srcN)
TK = json.load(open(O / "neocp_pilot/tracklets.json"))["tracklets"]
chk("8", "median detections / median arc (min)", [4, 35], [np.median([t["n_det"] for t in TK]), 60 * np.median([t["arc_hours"] for t in TK])], 0.5, "validation/output/neocp_pilot/tracklets.json", fmt="{:.1f}")
chk("8", "priora predicted / digest2 NoID/100 predicted", [312.0, 436], [pU.sum(), (noid / 100).sum()], [0.05, 0.5], srcN, fmt="{:.1f}")
s65 = noid >= 65; cN = np.sort(pU[yN])[::-1][yN[s65].sum() - 1]; sN = pU >= cN; s5 = pU >= 0.5
chk("8 / Table 5", "digest2>=65 NEO, non-NEO, compl, contam", [300, 140, 98.4, 31.8], [yN[s65].sum(), (~yN[s65]).sum(), 100 * yN[s65].sum() / 305, 100 * (~yN[s65]).mean()], 0.05, srcN, fmt="{:.2f}")
chk("8 / Table 5", "priora matched cut", 0.030, cN, 0.0005, srcN, fmt="{:.4f}")
chk("8 / Table 5", "priora matched NEO, non-NEO, contam", [300, 85, 22.1], [yN[sN].sum(), (~yN[sN]).sum(), 100 * (~yN[sN]).mean()], 0.05, srcN, fmt="{:.2f}")
chk("8 / Table 5", "priora P>=0.5 NEO, non-NEO, compl, contam", [286, 22, 93.8, 7.1], [yN[s5].sum(), (~yN[s5]).sum(), 100 * yN[s5].sum() / 305, 100 * (~yN[s5]).mean()], 0.05, srcN, fmt="{:.2f}")
chk("8 / Table 5", "fraction of sample below digest2 65 (%)", 11.8, 100 * (~s65).mean(), 0.05, srcN, fmt="{:.2f}")
chk("8", "selected tracklets: digest2 - priora matched", 55, s65.sum() - sN.sum(), 0, srcN)
chk("8", "AUC priora Unk / digest2 NoID", [0.9663, 0.9539], [auc(pU, yN), auc(noid, yN)], 0.00005, srcN, fmt="{:.5f}")
bt = auc_bootstrap(pU, noid, yN, np.ones_like(pU), n_boot=2000, seed=7)
chk("8", "Delta AUC and 95% CI (paired, stratified bootstrap, seed 7, 2000)", [0.0123, 0.0050, 0.0211], [bt["delta"]] + list(bt["delta_ci"]), 0.00005, srcN + "; digest2_comparison.auc_bootstrap", fmt="{:.4f}")
fN = s5 & ~yN
chk("8", "priora P>=0.5 non-NEOs: q<2.3 / q<1.55", [22, 17], [(fN & (qN < 2.3)).sum(), (fN & (qN < 1.55)).sum()], 0, srcN)
fD = s65 & ~yN
chk("8", "digest2 FPs: q<1.67 of total / q>=4.5", [91, 140, 4], [(fD & (qN < 1.67)).sum(), fD.sum(), (fD & (qN >= 4.5)).sum()], 0, srcN)
chk("8", "NEOCP median V (paper: 21.1, cut = +0.1 = 21.2)", 21.14, np.median(VN), 0.005, srcN, fmt="{:.3f}")
HX = json.load(open(O / "h_extension/neocp_delta_all.json"))
tr = HX["configs"]["truncated"]; zero = (np.array(tr["lambda_neo"]) == 0) & (np.array(tr["lambda_other"]) == 0)
neo_hx = np.array(HX["is_neo"], bool)
rates = np.array([np.degrees(np.hypot(t["ra_rate"] * np.cos(t["dec"]), t["dec_rate"])) for t in TK]); vm = np.array([t["v_mag"] for t in TK])
chk("2.8", "NEOCP tracklets with all measure at H>25.5 (both Lambda 0), all NEOs", [11, 11], [zero.sum(), (zero & neo_hx).sum()], 0, "validation/output/h_extension/neocp_delta_all.json (truncated)")
same_order = [t["designation"] for t in TK] == HX["designations"]
if same_order:
    chk("2.8", "those 11: median rate (deg/day), median V", [6.8, 20.7], [np.median(rates[zero]), np.median(vm[zero])], 0.05, "neocp_pilot/tracklets.json", fmt="{:.2f}")
ext = HX["configs"]["extended"]
chk("2.8", "with H to 30 the 11 get P=1", 11, int((np.array(ext["p_neo"])[zero] == 1).sum()), 0, "neocp_delta_all.json (extended)")
fa30 = np.array(ext["ar_frac_H_above_30"])[zero]
chk("App C", "boundary tracklets keeping 7-10% of measure above H=30", [4, 0.07, 0.10], [int((fa30 > 0.05).sum()), fa30[fa30 > 0.05].min(), fa30[fa30 > 0.05].max()], 0.006, "neocp_delta_all.json ar_frac_H_above_30", fmt="{:.3f}")
chk("App C", "min fraction of measure at H>=25.5 among the 11 (>= 0.986)", 0.986, np.array(tr["ar_frac_H_above_25p5"])[zero].min(), 0.0005, "neocp_delta_all.json", fmt="{:.4f}")

# bright sample (Section 8, Table 6, Figure 8/9)
PEAK = {"TC": 14.75, "TNO": 7.25}
bz = np.load(O / "gate_v0unk_bright212_hext30/scores.npz"); bg = np.load(O / "gate_v0unk_bright212_hext30/gen.npz"); bd = np.load(O / "digest2_comparison_unk_bright212/digest2_scores.npz")
bi = bz["index"]; kb = np.isin(bi, bd["index"]); bi = bi[kb]; bp = bz["p_neo"][kb]; bs = bd["noid_neo"].astype(float); bok = bd["ok"].astype(bool)
by = bg["is_neo"][bi].astype(bool); be = bg["elements"][bi]; bq = be[:, 0] * (1 - be[:, 1]); bH = bg["H"][bi]; bV = bg["v_mag"][bi]
gb = np.isfinite(bp) & bok & np.isfinite(bs); bp, bs, by, bq, bH, bV = bp[gb], bs[gb], by[gb], bq[gb], bH[gb], bV[gb]
srcB = "validation/output/gate_v0unk_bright212_hext30 + digest2_comparison_unk_bright212"
chk("8", "bright sample: tracklets / NEOs / max V", [3498, 215, 21.2], [bp.size, by.sum(), bV.max()], 0.05, srcB, fmt="{:.2f}")
known = ((bq >= 4.5) & (bq < 10) & (bH < PEAK["TC"])) | ((bq >= 10) & (bH < PEAK["TNO"]))


def op(sel_y, p, s):
    d2 = s >= 65; comp = sel_y[d2].sum() / sel_y.sum(); cont = (~sel_y[d2]).mean()
    c = np.sort(p[sel_y])[::-1][sel_y[d2].sum() - 1]; dsel = p >= c
    return comp, cont, d2, (~sel_y[dsel]).mean()


comp, cont, d2b, c42 = op(by, bp, bs)
chk("8 / Table 6", "V<21.2: compl, contam, distant FPs, priora contam", [98.1, 54.2, 143, 17.6], [100 * comp, 100 * cont, (d2b & ~by & (bq >= 4.5)).sum(), 100 * c42], 0.05, srcB, fmt="{:.2f}")
m = ~known
comp2, cont2, d2k, c42k = op(by[m], bp[m], bs[m])
chk("8 / Table 6", "known distant removed: compl, contam, distant FPs, priora contam", [98.1, 33.9, 1, 17.3], [100 * comp2, 100 * cont2, (d2k & ~by[m] & (bq[m] >= 4.5)).sum(), 100 * c42k], 0.05, srcB + "; distant-only H-peak removal", fmt="{:.2f}")
dfa = d2b & ~by & (bq >= 4.5)
tc, tn = dfa & (bq < 10), dfa & (bq >= 10)
chk("8 / Fig 9", "distant FPs brighter than the known peak (142 of 143)", [142, 143], [(dfa & known).sum(), dfa.sum()], 0, srcB)
chk("8 / Fig 9", "median H of distant FPs: 4.5-10 AU / >=10 AU", [11.5, 2.8], [np.median(bH[tc]), np.median(bH[tn])], 0.05, srcB, fmt="{:.2f}")
BN = json.load(open(REPO / "validation/d2_far_known/bright_numbers.json"))
sel_bins = {"TNOs >= 10 AU": ["0-6", "6-8"], "Trojans/Centaurs 4.5-10 AU": ["8-10", "11-12", "12-13", "13-14"]}
kom = [r["known_over_model"] for k, hs in sel_bins.items() for r in BN[k]["H_table"] if r["H"] in hs]
chk("8", "known/model in brightest complete bins: min, max (14-30%)", [0.14, 0.30], [min(kom), max(kom)], 0.006,
    "validation/d2_far_known/bright_numbers.json (bright_test.py): TNO H 0-8, Trojan/Centaur H 8-10 and 11-14", fmt="{:.3f}")
pk = json.load(open(REPO / "validation/d2_far_known/bright_match.json"))["known_peak_H"]
chk("8", "known-object H peaks: 4.5-10 AU / >=10 AU", [14.75, 7.25], [pk["Trojan/Centaur"], pk["TNO"]], 0, "validation/d2_far_known/bright_match.json (MPCORB 2026-11-04)")

# ---------------------------------------------------------------------------------------------- Appendix C: H_max table
r = np.array([1.3, 1.45, 1.55, 1.67, 1.8, 2.0])
chk("App C", "H_max(r) for V=24", [26.0, 24.9, 24.3, 23.8, 23.2, 22.5], list(24 - 5 * np.log10(r * (r - 1))), 0.051, "Equation (hmax)", fmt="{:.2f}")
GS = json.load(open(O / "h_interpolation/grid_study.json"))
chk("App B", "quadrature: mid-P p90 / max error vs 9600x400", [8e-3, 0.23], [GS["target_mid_p90"], GS["histogram_baseline"]["vs_fine_reference"]["max"]], "auto", "validation/output/h_interpolation/grid_study.json", fmt="{:.4f}")
chk("App B", "mid-P tracklets", 497, GS["n_mid_p"], 0, "grid_study.json")

# ---------------------------------------------------------------------------------------------- output
out = Path(__file__).with_suffix("")
json.dump(ROWS, open(f"{out}.json", "w"), indent=1, default=str)
with open(f"{out}.md", "w") as fh:
    fh.write("| Section | Quantity | Paper | Computed | Status | Source |\n|---|---|---|---|---|---|\n")
    for r_ in ROWS:
        fh.write(f"| {r_['section']} | {r_['quantity']} | {r_['paper']} | {r_['computed']} | {r_['status']} | `{r_['source']}` |\n")
bad = [r_ for r_ in ROWS if r_["status"] == "MISMATCH"]
print(f"{len(ROWS)} checks, {len(bad)} mismatches")
for r_ in bad:
    print(f"  MISMATCH  {r_['section']}: {r_['quantity']}: paper {r_['paper']}, computed {r_['computed']}")
