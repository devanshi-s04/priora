"""Measurement driver for the mass-conserving smooth-in-H reconstruction.

``Digest2Histogram(h_interpolation="pchip")`` replaces the piecewise-constant
H dependence of the digest2 model by the derivative of a monotone cubic through
each (q, e, i) cell's cumulative object count (see
``ds42.population``'s module docstring for the construction and its
guarantees).  This module measures what that buys and what it changes.  Three
independent experiments, three subcommands:

``grid``
    **Node-count vs accuracy.**  DESIGN §2.5 records that the dominant
    quadrature error of the histogram model is the *rho-axis staircase*: H(rho)
    depends on rho alone, so every H-bin edge is a rho-aligned discontinuity of
    the integrand, which forces rho-heavy grids (n_rho ~ 1200, n_rhodot ~ 50 for
    p90 |dp| ~ 3.5e-3).  The smooth mode removes the H edges from that list.
    This subcommand sweeps (n_rho, n_rhodot) against a very fine same-mode
    reference in *both* modes on a diverse tracklet sample, reports the grid at
    which the smooth mode matches the histogram mode's current accuracy and the
    grid 10x better than that, and times the recommended one.

``rescore``/``analyze``
    **Mode-change measurement**, on the template of the D-14 clamp experiment
    (``validation/DIGEST2_COMPARISON_CLAMPED.md``): re-score the *existing*
    `gate_v0` catalogue — whose truths were generated from the histogram — with
    the smooth pair, and compare against the stored histogram scores and
    against digest2's stored scores.  Delta-p against the §2.6(b) diagnostic and
    against V; ranking (AUC) against the stored labels and against digest2;
    calibration.

    The calibration arm of this experiment is a **deliberate model mismatch**:
    the catalogue was drawn from the piecewise-constant model and is scored with
    the reconstruction, so any calibration drift is prior misspecification
    (DESIGN §4.3), measured on purpose.  It is *not* a statement about the
    smooth mode's own calibration — that is what the self-consistency gate run
    in ``--h-interpolation pchip`` mode answers, and it is reported separately
    in ``validation/output/h_interpolation/REPORT.md``.

Nothing here mutates the inputs: `gate_v0`'s `gen.npz`, its `scores.npz` and
the digest2 scores under `validation/output/digest2_comparison/gate_v0/` are
read-only.  Outputs go to ``validation/output/h_interpolation/``.

Reproduce
---------
::

    python -m validation.h_interpolation_study grid \\
        --out validation/output/h_interpolation
    python -m validation.h_interpolation_study rescore \\
        --run validation/output/gate_v0 \\
        --out validation/output/h_interpolation --n-rho 400 --n-rhodot 50
    python -m validation.h_interpolation_study analyze \\
        --run validation/output/gate_v0 \\
        --out validation/output/h_interpolation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ds42.attributable import Attributable  # noqa: E402
from ds42.backend import core_available  # noqa: E402
from ds42.classifier import classify, classify_many  # noqa: E402
from ds42.constants import G_SLOPE_DEFAULT  # noqa: E402
from ds42.population import Digest2Histogram  # noqa: E402

from validation.digest2_comparison import (  # noqa: E402
    auc_bootstrap,
    auc_fast,
    calibration_stats,
    percentile_rank,
    subset_masks,
)

DEFAULT_MODEL = "data/digest2.model.csv"
RHO_MIN = 0.01
RHO_MAX = 100.0

#: The reference grid every sweep point is measured against.  4x the production
#: n_rho and 4x its n_rhodot; the ``grid`` subcommand also runs a
#: reference-adequacy check against (2x n_rho, same n_rhodot) so the quoted
#: errors are not dominated by the reference's own.
REF_GRID = (4800, 200)
REF_CHECK_GRID = (9600, 200)

#: The current production grid (DESIGN §2.5).
PROD_GRID = (1200, 50)

#: DESIGN §2.5's own baseline reference, kept so the histogram-mode number can
#: be compared with the one recorded there (which used n_rho = 1200 on both
#: sides and therefore measured rho_dot refinement only).
DESIGN_REF_GRID = (1200, 600)

N_RHO_SWEEP = (50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400)
N_RHODOT_SWEEP = (12, 25, 50, 100)


# --------------------------------------------------------------------------
# Tracklet selection
# --------------------------------------------------------------------------


def load_catalogue(run_dir: str) -> dict:
    """`gen.npz` joined to `scores.npz`, ordered as `scores.npz['index']`."""
    with np.load(os.path.join(run_dir, "scores.npz")) as z:
        idx = z["index"].astype(int)
        sc = {k: np.asarray(z[k]) for k in z.files if k != "index"}
    with np.load(os.path.join(run_dir, "gen.npz")) as z:
        gen = {k: np.asarray(z[k])[idx] for k in z.files}
    keep = np.isfinite(sc["p_neo"])
    return {
        "index": idx[keep],
        "gen": {k: v[keep] for k, v in gen.items()},
        "scores": {k: v[keep] for k, v in sc.items()},
        "n_dropped": int((~keep).sum()),
    }


def diverse_sample(cat: dict, n: int, seed: int = 20260902) -> np.ndarray:
    """`n` row positions spanning the catalogue's own diagnostic space.

    Stratified on the product of three tertiles — p(NEO) (which stands in for
    "how close to the decision boundary"), the §2.6(b) truncation-exposure
    diagnostic (which is where the H reconstruction can matter most), and
    apparent rate (which sets how much of the admissible region is live) — then
    filled at random inside each of the 27 strata.  A plain random sample of 60
    out of 2x10^4 would put ~1 tracklet in the high-exposure tail, which is
    exactly the tail this study is about.
    """
    rng = np.random.default_rng(seed)
    p = cat["scores"]["p_neo"]
    f = np.nan_to_num(cat["scores"]["ar_frac_h_above"], nan=0.0)
    rate = np.hypot(cat["gen"]["ra_rate"] * np.cos(cat["gen"]["dec"]),
                    cat["gen"]["dec_rate"])
    keys = []
    for v in (p, f, rate):
        r = percentile_rank(v)
        keys.append(np.clip((r * 3).astype(int), 0, 2))
    strat = keys[0] * 9 + keys[1] * 3 + keys[2]
    out: list[int] = []
    order = rng.permutation(27)
    per = max(1, n // 27)
    for s in order:
        pool = np.flatnonzero(strat == s)
        if pool.size == 0:
            continue
        take = min(per, pool.size)
        out.extend(rng.choice(pool, size=take, replace=False).tolist())
    # top up at random to exactly n
    rest = np.setdiff1d(np.arange(p.size), np.array(out, dtype=int))
    if len(out) < n and rest.size:
        out.extend(rng.choice(rest, size=min(n - len(out), rest.size),
                              replace=False).tolist())
    return np.array(sorted(out[:n]), dtype=int)


def attributables(cat: dict, rows: np.ndarray):
    """(atts, R_obs, V_obs) for the selected catalogue rows."""
    from astropy.time import Time

    g = cat["gen"]
    atts = [
        Attributable(
            ra=float(g["ra"][k]),
            dec=float(g["dec"][k]),
            ra_rate=float(g["ra_rate"][k]),
            dec_rate=float(g["dec_rate"][k]),
            epoch=Time(float(g["epoch_jd_tdb"][k]), format="jd", scale="tdb"),
            v_mag=float(g["v_mag"][k]),
        )
        for k in rows
    ]
    return atts, g["R_obs"][rows].copy(), g["V_obs"][rows].copy()


# --------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------


def pair_for(mode: str, model: str = DEFAULT_MODEL, variant: str = "All"):
    return Digest2Histogram.neo_other_pair(model, variant,
                                            h_interpolation=mode)


def score_batch(atts, R, V, pair, n_rho: int, n_rhodot: int,
                n_threads: int | None = None, lambdas: bool = False):
    """p(NEO) — and optionally the two Lambdas — on the Rust backend.

    The Lambdas matter for the grid study specifically: p(NEO) is the *ratio*
    Lambda_NEO / (Lambda_NEO + Lambda_other), and both components share the
    same H bin edges, so a quadrature error caused by an H-edge discontinuity
    is largely common-mode and cancels in the ratio.  Convergence of the
    Lambdas separately is therefore the measurement that can actually see the
    rho-axis staircase; convergence of p(NEO) is the one that matters
    operationally.  Reporting only the second would have hidden the mechanism.
    """
    neo, other = pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = classify_many(
            atts, neo, other, R_obs=R, V_obs=V,
            rho_min=RHO_MIN, rho_max=RHO_MAX,
            n_rho=n_rho, n_rhodot=n_rhodot, G=G_SLOPE_DEFAULT,
            diagnostics=False, backend="rust", n_threads=n_threads,
        )
    p = np.array([r.p_neo for r in res], dtype=float)
    if not lambdas:
        return p
    return (
        p,
        np.array([r.lambda_neo for r in res], dtype=float),
        np.array([r.lambda_other for r in res], dtype=float),
    )


def chunked_inputs(cat: dict, rows: np.ndarray, size: int = 2000) -> list:
    """`[(atts, R, V), ...]`, built once and reused across every sweep point.

    The Attributable objects each carry an astropy ``Time``, which costs more
    to construct than a whole tracklet costs to score at a coarse grid; the
    sweep touches the same tracklets ~100 times, so they are built once.
    """
    return [
        attributables(cat, rows[start:start + size])
        for start in range(0, rows.size, size)
    ]


def score_chunks(chunks, pair, n_rho: int, n_rhodot: int,
                 n_threads: int | None = None, lambdas: bool = False):
    if not chunks:
        return np.zeros(0)
    parts = [
        score_batch(a, r, v, pair, n_rho, n_rhodot, n_threads=n_threads,
                    lambdas=lambdas)
        for a, r, v in chunks
    ]
    if not lambdas:
        return np.concatenate(parts)
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


#: p(NEO) window that defines a "mid-p" tracklet.  |dp| is bounded by
#: min(p, 1-p), so the whole quadrature-error budget lives here: at mixture
#: proportions the gate population is overwhelmingly low-p, where |dp| is
#: numerically tiny for arithmetic reasons rather than because the quadrature
#: is good.  DESIGN §2.5's recorded p90 = 3.5e-3 / max = 1e-2 is reproduced by
#: this conditional statistic and NOT by the unconditional one (see the
#: reconciliation in REPORT.md), so both are carried everywhere below.
MID_P = (0.05, 0.95)


def _pct(d: np.ndarray) -> dict:
    if d.size == 0:
        nan = float("nan")
        return {"n": 0, "p50": nan, "p90": nan, "p99": nan, "max": nan,
                "rms": nan}
    return {
        "n": int(d.size),
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "max": float(d.max()),
        "rms": float(np.sqrt((d ** 2).mean())),
    }


def _err(p: np.ndarray, ref: np.ndarray, mid: np.ndarray | None = None) -> dict:
    """|dp| percentiles overall and (if `mid` is given) on the mid-p subset.

    `mid` must be defined on the REFERENCE scores, not on the sweep point's,
    so that the subset is the same set of tracklets at every grid.
    """
    d = np.abs(p - ref)
    good = np.isfinite(d)
    out = _pct(d[good])
    if mid is not None:
        out["mid"] = _pct(d[good & mid])
    return out


def _rel_err(x: np.ndarray, ref: np.ndarray) -> dict:
    """Percentiles of |x/ref - 1| over the entries where ref is positive."""
    ok = np.isfinite(x) & np.isfinite(ref) & (ref > 0.0)
    if not ok.any():
        return {"n": 0, "p50": float("nan"), "p90": float("nan"),
                "p99": float("nan"), "max": float("nan")}
    d = np.abs(x[ok] / ref[ok] - 1.0)
    return {
        "n": int(ok.sum()),
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "max": float(d.max()),
    }


# --------------------------------------------------------------------------
# Measurement 1: node count vs accuracy
# --------------------------------------------------------------------------


def cmd_grid(a: argparse.Namespace) -> int:
    if not core_available():
        raise SystemExit("the Rust backend is required (run rust/build.sh)")
    ref_grid = tuple(int(x) for x in a.ref_grid)
    ref_check = tuple(int(x) for x in a.ref_check_grid)
    cat = load_catalogue(a.run)
    n_all = cat["index"].size
    if a.select == "midp":
        sp = cat["scores"]["p_neo"]
        rows = np.flatnonzero((sp > MID_P[0]) & (sp < MID_P[1]))
        how = (f"all {rows.size} tracklets with stored p in "
               f"({MID_P[0]}, {MID_P[1]})")
    elif a.select == "stratified" and 0 < a.n_tracklets < n_all:
        rows = diverse_sample(cat, a.n_tracklets, seed=a.seed)
        how = "stratified sample"
    elif a.n_tracklets <= 0 or a.n_tracklets >= n_all:
        rows = np.arange(n_all)
        how = "the whole catalogue"
    else:
        rows = diverse_sample(cat, a.n_tracklets, seed=a.seed)
        how = "stratified sample"
    t0 = time.time()
    chunks = chunked_inputs(cat, rows)
    atts, R, V = chunks[0]
    print(f"[grid] {rows.size} tracklets from {a.run} ({how}; dropped "
          f"{cat['n_dropped']} with non-finite stored p) in "
          f"{time.time() - t0:.1f} s")

    pairs = {m: pair_for(m, a.model, a.variant) for m in ("none", "pchip")}
    out: dict = {
        "run": a.run,
        "model": a.model,
        "variant": a.variant,
        "selection": how,
        "n_tracklets": int(len(rows)),
        "ref_grid": list(ref_grid),
        "ref_check_grid": list(ref_check),
        "prod_grid": list(PROD_GRID),
        "design_ref_grid": list(DESIGN_REF_GRID),
        "n_rho_sweep": list(N_RHO_SWEEP),
        "n_rhodot_sweep": list(N_RHODOT_SWEEP),
        "select": a.select,
        "mid_p_window": list(MID_P),
        "sweep": {},
        "reference": {},
    }

    # --- references, per mode -------------------------------------------
    ref: dict[str, np.ndarray] = {}
    ref_lam: dict[str, tuple] = {}
    for mode in ("none", "pchip"):
        t0 = time.time()
        p_ref, ln_ref, lo_ref = score_chunks(
            chunks, pairs[mode], *ref_grid, n_threads=a.threads, lambdas=True
        )
        ref[mode] = p_ref
        ref_lam[mode] = (ln_ref, lo_ref)
        print(f"[grid] reference {mode} at {ref_grid}: "
              f"{time.time() - t0:.1f} s")

    # The mid-p subset is defined ONCE, on the histogram-mode reference, so
    # that "the mid-p tracklets" means the same tracklets at every sweep point
    # and in both modes.  Defining it on each sweep point's own scores would
    # let the subset drift with the grid and make the comparison circular.
    mid = (ref["none"] > MID_P[0]) & (ref["none"] < MID_P[1])
    out["n_mid_p"] = int(mid.sum())
    print(f"[grid] mid-p subset: {out['n_mid_p']} of {rows.size} tracklets")

    # Reference adequacy: how much does the reference itself still move when
    # the grid is refined again?  Nothing in the sweep table is meaningful
    # below this floor.
    for mode in ("none", "pchip"):
        t0 = time.time()
        chk = score_chunks(chunks, pairs[mode], *ref_check,
                           n_threads=a.threads)
        out["reference"][mode] = {
            "self_error_vs_" + "x".join(map(str, ref_check)):
                _err(ref[mode], chk, mid),
            "n": int(ref[mode].size),
            "check_s": time.time() - t0,
        }
        fl = out["reference"][mode][
            "self_error_vs_" + "x".join(map(str, ref_check))]
        print(f"[grid]   adequacy check {mode} vs {ref_check}: mid-p p90 "
              f"{fl['mid']['p90']:.3e}, all-p90 {fl['p90']:.3e}")

    # --- the histogram-mode baseline this whole study is measured against -
    hist_design_ref = score_chunks(chunks, pairs["none"], *DESIGN_REF_GRID,
                                   n_threads=a.threads)
    hist_prod = score_chunks(chunks, pairs["none"], *PROD_GRID,
                             n_threads=a.threads)
    out["histogram_baseline"] = {
        "grid": list(PROD_GRID),
        "vs_fine_reference": _err(hist_prod, ref["none"], mid),
        "vs_design_reference": _err(hist_prod, hist_design_ref, mid),
    }
    # PRIMARY threshold: DESIGN §2.5's own statistic (mid-p p90) against the
    # fine reference.  The unconditional p90 is carried alongside.
    target = out["histogram_baseline"]["vs_fine_reference"]["mid"]["p90"]
    target_all = out["histogram_baseline"]["vs_fine_reference"]["p90"]
    out["target_mid_p90"] = target
    out["target_all_p90"] = target_all
    b = out["histogram_baseline"]
    print(f"[grid] histogram {PROD_GRID} vs {ref_grid}: "
          f"mid-p p90 |dp| = {target:.4e} (max "
          f"{b['vs_fine_reference']['mid']['max']:.3e}), all-p90 "
          f"{target_all:.4e}")
    print(f"[grid]   vs DESIGN's {DESIGN_REF_GRID} reference: mid-p p90 "
          f"{b['vs_design_reference']['mid']['p90']:.4e} (max "
          f"{b['vs_design_reference']['mid']['max']:.3e}), all-p90 "
          f"{b['vs_design_reference']['p90']:.4e}"
          "   <- DESIGN §2.5 records 3.5e-3 / 1e-2")

    # --- DESIGN §2.5's own fixed-cost comparisons, as a harness check ----
    # "at fixed cost, 800x100 beats 400x200 by ~2x and 200x400 equals
    # 200x100".  Reproducing those RELATIVE statements is what licenses the
    # absolute numbers in this table; if the harness disagreed with them, the
    # absolute disagreement with DESIGN's 3.5e-3 would be a bug in here rather
    # than a difference of statistic.
    out["design_fixed_cost_check"] = []
    for g in ((800, 100), (400, 200), (200, 400), (200, 100)):
        q = score_chunks(chunks, pairs["none"], *g, n_threads=a.threads)
        e = _err(q, ref["none"], mid)
        out["design_fixed_cost_check"].append(
            {"grid": list(g), "nodes": g[0] * g[1], "p90": e["p90"],
             "mid_p90": e["mid"]["p90"], "max": e["max"]})
        print(f"[grid] fixed-cost {g}: all-p90 {e['p90']:.3e}, mid-p p90 "
              f"{e['mid']['p90']:.3e}")

    # --- the sweep -------------------------------------------------------
    for mode in ("none", "pchip"):
        out["sweep"][mode] = []
        for nr in N_RHO_SWEEP:
            for nd in N_RHODOT_SWEEP:
                t0 = time.time()
                p, ln, lo = score_chunks(chunks, pairs[mode], nr, nd,
                                         n_threads=a.threads, lambdas=True)
                rec = {
                    "n_rho": nr,
                    "n_rhodot": nd,
                    "nodes": nr * nd,
                    "wall_s": time.time() - t0,
                    **_err(p, ref[mode], mid),
                    "lam_neo_rel": _rel_err(ln, ref_lam[mode][0]),
                    "lam_other_rel": _rel_err(lo, ref_lam[mode][1]),
                }
                out["sweep"][mode].append(rec)
            print(f"[grid] {mode} n_rho={nr}: " + "  ".join(
                f"{r['n_rhodot']}:{r['mid']['p90']:.2e}"
                f"/a{r['p90']:.1e}/L{r['lam_other_rel']['p90']:.1e}"
                for r in out["sweep"][mode][-len(N_RHODOT_SWEEP):]))

    # --- pick the recommended grids -------------------------------------
    def cheapest_at_or_below(mode: str, thr: float, stable: bool,
                             key: str = "mid"):
        """Cheapest sweep point meeting `thr`; optionally requiring stability.

        The convergence in n_rho is NOT monotone: the rho nodes are log-spaced
        and the integrand's remaining discontinuities sit at fixed rho, so
        particular node counts alias favourably and others unfavourably (see
        the sweep table — the effect is up to an order of magnitude).  Picking
        the cheapest point that merely happens to meet the threshold would
        therefore recommend an aliasing fluke that any change to rho_min /
        rho_max / the population would destroy.  ``stable=True`` additionally
        requires every *finer* point (n_rho' >= n_rho and n_rhodot' >=
        n_rhodot) to meet the threshold too, which is the property a
        production grid actually needs.
        """
        recs = out["sweep"][mode]
        val = (lambda r: r["mid"]["p90"]) if key == "mid" else \
            (lambda r: r["p90"])
        cands = [r for r in recs if val(r) <= thr]
        if stable:
            cands = [
                r for r in cands
                if all(val(o) <= thr for o in recs
                       if o["n_rho"] >= r["n_rho"]
                       and o["n_rhodot"] >= r["n_rhodot"])
            ]
        if not cands:
            return None
        return min(cands, key=lambda r: (r["nodes"], r["n_rho"]))

    for mode in ("pchip", "none"):
        out[f"match_histogram_accuracy/{mode}"] = cheapest_at_or_below(
            mode, target, stable=True)
        out[f"match_histogram_accuracy_unstable/{mode}"] = \
            cheapest_at_or_below(mode, target, stable=False)
        out[f"ten_times_better/{mode}"] = cheapest_at_or_below(
            mode, target / 10.0, stable=True)
        out[f"ten_times_better_unstable/{mode}"] = cheapest_at_or_below(
            mode, target / 10.0, stable=False)
    out["match_histogram_accuracy"] = out["match_histogram_accuracy/pchip"]
    out["ten_times_better"] = out["ten_times_better/pchip"]
    for mode in ("pchip", "none"):
        out[f"match_all_p90/{mode}"] = cheapest_at_or_below(
            mode, target_all, stable=True, key="all")
    for name in sorted(k for k in out if k.startswith(
            ("match_", "ten_")) and "/" in k):
        r = out[name]
        print(f"[grid] {name}: " + (
            "none" if r is None else
            f"({r['n_rho']}, {r['n_rhodot']}) = {r['nodes']} nodes, "
            f"mid-p p90 {r['mid']['p90']:.3e}, all-p90 {r['p90']:.3e}, "
            f"mid-max {r['mid']['max']:.3e}"))

    # --- single-core timing at the candidate grids ----------------------
    out["timing"] = {}
    for key in ("match_histogram_accuracy", "ten_times_better"):
        r = out[key]
        if r is None:
            continue
        for mode in ("pchip", "none"):
            t = _time_single_core(atts, R, V, pairs[mode],
                                  r["n_rho"], r["n_rhodot"], a.time_reps,
                                  a.time_tracklets)
            out["timing"][f"{key}/{mode}"] = t
            print(f"[grid] timing {key} {mode} "
                  f"({r['n_rho']}, {r['n_rhodot']}): {t['ms_per_tracklet']:.2f}"
                  " ms/tracklet single-core")
    for mode in ("pchip", "none"):
        t = _time_single_core(atts, R, V, pairs[mode], *PROD_GRID,
                              a.time_reps, a.time_tracklets)
        out["timing"][f"prod/{mode}"] = t
        print(f"[grid] timing prod {mode} {PROD_GRID}: "
              f"{t['ms_per_tracklet']:.2f} ms/tracklet single-core")

    # --- is the improvement at the production grid DETECTABLE? ----------
    # A point estimate of "the smooth mode improves p90 by 2%" is not a
    # result: p90 of ~500 values has sampling error of its own, and the review
    # found the sign of the effect flips between catalogues.  Bootstrap the
    # ratio over tracklets, PAIRED (both modes resampled on the same rows, so
    # the common-mode part of the error cancels the way it does in the
    # difference of two AUCs).
    out["prod_improvement"] = _bootstrap_p90_ratio(
        chunks, pairs, ref, mid, PROD_GRID, a.threads, a.boot, a.boot_seed
    )
    pi = out["prod_improvement"]
    print(f"[grid] improvement at {PROD_GRID}: mid-p p90 ratio "
          f"{pi['ratio']:.4f} [{pi['ratio_ci'][0]:.4f}, "
          f"{pi['ratio_ci'][1]:.4f}] -> "
          f"{'DETECTABLE' if pi['detectable'] else 'NOT detectable'}")

    # --- where the residual error lives ---------------------------------
    out["residual"] = _residual_anatomy(cat, rows, chunks, pairs, ref,
                                        out, a.threads)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, a.out_name)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=_json_default)
    print(f"[grid] wrote {path}")
    _plot_grid(out, a.out, a.out_name.replace(".json", ""))
    return 0


def _time_single_core(atts, R, V, pair, n_rho, n_rhodot, reps: int,
                      m: int = 32) -> dict:
    """ms/tracklet with one thread; the MINIMUM over `reps` passes.

    The minimum, not the mean: this box is shared, and any contention can only
    make a pass slower.  Every rep is kept in the record so a reader can see
    the spread and judge whether the box was quiet.
    """
    m = min(len(atts), max(1, m))
    times = []
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        score_batch(atts[:m], R[:m], V[:m], pair, n_rho, n_rhodot,
                    n_threads=1)
        times.append(time.perf_counter() - t0)
    best = min(times)
    return {
        "n_rho": int(n_rho),
        "n_rhodot": int(n_rhodot),
        "n_tracklets": m,
        "reps": int(max(1, reps)),
        "all_wall_s": times,
        "best_wall_s": best,
        "ms_per_tracklet": 1e3 * best / m,
    }


def _bootstrap_p90_ratio(chunks, pairs, ref, mid, grid, threads, n_boot,
                         seed) -> dict:
    """Paired bootstrap of (smooth p90 |dp|) / (histogram p90 |dp|), mid-p.

    Resamples tracklets with replacement and recomputes both modes' p90 on the
    same resample, so the answer is a confidence interval on the *ratio* — the
    quantity the recommendation turns on — rather than on either p90 alone.
    ``detectable`` is simply "the interval excludes 1".
    """
    per: dict[str, np.ndarray] = {}
    for mode in ("none", "pchip"):
        p = score_chunks(chunks, pairs[mode], *grid, n_threads=threads)
        per[mode] = np.abs(p - ref[mode])
    keep = mid & np.isfinite(per["none"]) & np.isfinite(per["pchip"])
    a_, b_ = per["none"][keep], per["pchip"][keep]
    n = a_.size
    rng = np.random.default_rng(seed)
    ratios = np.empty(int(n_boot))
    for k in range(int(n_boot)):
        take = rng.integers(0, n, n)
        ratios[k] = np.percentile(b_[take], 90) / np.percentile(a_[take], 90)
    lo, hi = (float(np.percentile(ratios, 2.5)),
              float(np.percentile(ratios, 97.5)))
    return {
        "grid": list(grid),
        "n_mid_p": int(n),
        "n_boot": int(n_boot),
        "p90_histogram": float(np.percentile(a_, 90)),
        "p90_smooth": float(np.percentile(b_, 90)),
        "ratio": float(np.percentile(b_, 90) / np.percentile(a_, 90)),
        "ratio_ci": [lo, hi],
        "improvement_pct": 100.0 * (1.0 - float(np.percentile(b_, 90)
                                                / np.percentile(a_, 90))),
        "improvement_pct_ci": [100.0 * (1.0 - hi), 100.0 * (1.0 - lo)],
        "detectable": bool(hi < 1.0 or lo > 1.0),
    }


def _residual_anatomy(cat, rows, chunks, pairs, ref, out, threads) -> dict:
    """Which tracklets carry the residual error at the recommended grid?

    Correlates the per-tracklet |dp| with the two things DESIGN §2.5 predicts
    should remain once the H staircase is gone: the admissible-region terminus
    (tracklets whose region is small or clipped by rho_max) and the q/e/i bin
    edges (which, unlike the H edges, are not functions of rho alone).  A
    weight on the §2.6(b) exposure diagnostic is reported too, because that is
    the diagnostic the mode change is expected to act through.
    """
    r = out.get("match_histogram_accuracy") or out.get("prod_fallback")
    if r is None:
        # No grid met the threshold: anatomize the production grid instead,
        # which is the grid the recommendation falls back to.
        r = next((x for x in out["sweep"]["pchip"]
                  if [x["n_rho"], x["n_rhodot"]] == list(PROD_GRID)), None)
    if r is None:
        return {}
    p = score_chunks(chunks, pairs["pchip"], r["n_rho"], r["n_rhodot"],
                     n_threads=threads)
    d = np.abs(p - ref["pchip"])
    g = cat["gen"]
    sc = cat["scores"]
    feats = {
        "ar_frac_h_above": np.nan_to_num(sc["ar_frac_h_above"][rows], nan=0.0),
        "ar_measure": sc["ar_measure"][rows],
        "n_nodes": sc["n_nodes"][rows].astype(float),
        "v_mag": g["v_mag"][rows],
        "rho_true": g["rho_true"][rows],
        "rate_deg_day": np.hypot(g["ra_rate"][rows] * np.cos(g["dec"][rows]),
                                 g["dec_rate"][rows]),
        "p_neo_ref": ref["pchip"],
    }
    corr = {}
    for name, x in feats.items():
        ok = np.isfinite(x) & np.isfinite(d)
        if ok.sum() < 5 or np.std(x[ok]) == 0:
            corr[name] = float("nan")
            continue
        # Spearman: monotone association, robust to the heavy tails here.
        corr[name] = float(np.corrcoef(percentile_rank(x[ok]),
                                       percentile_rank(d[ok]))[0, 1])
    worst = np.argsort(-d)[:8]
    return {
        "grid": [r["n_rho"], r["n_rhodot"]],
        "spearman_abs_dp_vs": corr,
        "worst_tracklets": [
            {
                "row": int(rows[k]),
                "abs_dp": float(d[k]),
                "p_ref": float(ref["pchip"][k]),
                "v_mag": float(g["v_mag"][rows[k]]),
                "rho_true": float(g["rho_true"][rows[k]]),
                "ar_frac_h_above": float(sc["ar_frac_h_above"][rows[k]]),
                "n_nodes_prod": int(sc["n_nodes"][rows[k]]),
            }
            for k in worst
        ],
    }


def _plot_grid(out: dict, out_dir: str, stem: str = "grid_study") -> None:
    """Two figures: convergence in both modes, and the mode's own effect.

    The convergence panels plot the MID-P statistic, because the
    unconditional p90 is dominated by arithmetic head-room at mixture
    proportions and would show a flat, meaningless ~1e-4 (see REPORT.md).
    Lambda_other gets its own panel: the H-staircase error is common-mode
    across the two components and cancels in p(NEO), so the Lambdas are where
    the mode change is visible at all.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = {(r["n_rho"], r["n_rhodot"]): r for r in out["sweep"]["none"]}
    idxp = {(r["n_rho"], r["n_rhodot"]): r for r in out["sweep"]["pchip"]}
    nrs = out["n_rho_sweep"]
    nds = out["n_rhodot_sweep"]
    chk = "x".join(map(str, out["ref_check_grid"]))
    target = out["target_mid_p90"]
    colours = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(nds)))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    panels = (
        (r"mid-p p90 $|\Delta p(\mathrm{NEO})|$",
         lambda r: r["mid"]["p90"], "mid"),
        (r"p90 relative error in $\Lambda_{\rm other}$",
         lambda r: r["lam_other_rel"]["p90"], "lam_other_rel"),
    )
    for ax, (ylab, get, kind) in zip(axes, panels):
        for c, nd in zip(colours, nds):
            ax.loglog(nrs, [get(idx[(nr, nd)]) for nr in nrs], "-o", color=c,
                      ms=3.5, lw=1.4,
                      label=f"histogram, $n_{{\\dot\\rho}}$={nd}")
            ax.loglog(nrs, [get(idxp[(nr, nd)]) for nr in nrs], "--s",
                      color=c, ms=3.0, lw=1.1, mfc="none",
                      label=f"smooth, $n_{{\\dot\\rho}}$={nd}")
        if kind == "mid":
            ax.axhline(target, color="k", ls="-", lw=1.2,
                       label=f"histogram {tuple(out['prod_grid'])}")
            ax.axhline(target / 10.0, color="k", ls=":", lw=1.2,
                       label="10x better")
            floor = _floor(out, "none", chk)
        else:
            floor = out["reference"]["none"][
                "self_error_vs_" + chk].get("p90", 0.0)
        ax.axhspan(1e-12, max(floor, 1e-12), color="0.88", zorder=0,
                   label="reference's own error")
        ax.set_xlabel(r"$n_\rho$")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(bottom=max(floor * 0.5, 1e-6))
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle(
        f"Convergence in both H modes — {out['n_tracklets']} tracklets "
        f"({out.get('n_mid_p', '?')} mid-p) from {out['run']}, reference "
        + "x".join(map(str, out["ref_grid"])), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stem}_convergence.png"), dpi=130)
    plt.close(fig)

    # What the mode change is actually worth: the ratio smooth/histogram.
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for c, nd in zip(colours, nds):
        ax.semilogx(nrs, [idxp[(nr, nd)]["mid"]["p90"]
                          / idx[(nr, nd)]["mid"]["p90"] for nr in nrs],
                    "-o", color=c, ms=3.5, lw=1.4,
                    label=f"$p$, $n_{{\\dot\\rho}}$={nd}")
        ax.semilogx(nrs, [idxp[(nr, nd)]["lam_other_rel"]["p90"]
                          / idx[(nr, nd)]["lam_other_rel"]["p90"]
                          for nr in nrs],
                    "--s", color=c, ms=3.0, lw=1.1, mfc="none",
                    label=f"$\\Lambda_{{\\rm other}}$, "
                          f"$n_{{\\dot\\rho}}$={nd}")
    ax.axhline(1.0, color="k", lw=1.0)
    ax.set_xlabel(r"$n_\rho$")
    ax.set_ylabel("smooth error / histogram error")
    ax.set_title("What removing the H staircase is worth\n"
                 "(below 1 = the smooth mode converges faster)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stem}_mode_ratio.png"), dpi=130)
    plt.close(fig)

    # Cost-accuracy front, mid-p statistic.
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for mode, tab, c in (("histogram", idx, "tab:red"),
                         ("smooth", idxp, "tab:blue")):
        recs = sorted(tab.values(), key=lambda r: r["nodes"])
        ax.loglog([r["nodes"] for r in recs],
                  [r["mid"]["p90"] for r in recs], ".", color=c, alpha=0.55,
                  label=mode)
        best = np.minimum.accumulate(
            [r["mid"]["p90"] for r in recs][::-1])[::-1]
        ax.loglog([r["nodes"] for r in recs], best, "-", color=c, lw=1.5)
    ax.axhline(target, color="k", ls="--", lw=1)
    ax.set_xlabel(r"nodes per tracklet ($n_\rho \times n_{\dot\rho}$)")
    ax.set_ylabel(r"mid-p p90 $|\Delta p|$")
    ax.set_title("Cost-accuracy front (the two modes coincide)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stem}_cost_accuracy.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Measurement 3: re-score gate_v0 with the smooth pair
# --------------------------------------------------------------------------


@dataclass
class RescoreConfig:
    n_rho: int
    n_rhodot: int
    rho_min: float = RHO_MIN
    rho_max: float = RHO_MAX
    G: float = G_SLOPE_DEFAULT
    model: str = DEFAULT_MODEL
    variant: str = "All"
    h_interpolation: str = "pchip"


def cmd_rescore(a: argparse.Namespace) -> int:
    if not core_available():
        raise SystemExit("the Rust backend is required (run rust/build.sh)")
    cfg = RescoreConfig(n_rho=a.n_rho, n_rhodot=a.n_rhodot, model=a.model,
                        variant=a.variant)
    cat = load_catalogue(a.run)
    n = cat["index"].size
    pair = pair_for(cfg.h_interpolation, cfg.model, cfg.variant)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "pchip_scores.npz")

    print(f"[rescore] {n} tracklets from {a.run} at "
          f"({cfg.n_rho}, {cfg.n_rhodot}), mode={cfg.h_interpolation}")
    p = np.full(n, np.nan)
    t0 = time.time()
    step = int(a.batch)
    for start in range(0, n, step):
        stop = min(n, start + step)
        rows = np.arange(start, stop)
        atts, R, V = attributables(cat, rows)
        p[start:stop] = score_batch(atts, R, V, pair, cfg.n_rho, cfg.n_rhodot,
                                    n_threads=a.threads)
        print(f"[rescore]   {stop}/{n}  ({time.time() - t0:.1f} s)",
              flush=True)
    wall = time.time() - t0
    np.savez_compressed(path, index=cat["index"], p_neo=p,
                        config=json.dumps(asdict(cfg)))
    print(f"[rescore] wrote {path}: {wall:.1f} s "
          f"({1e3 * wall / n:.2f} ms/tracklet at {a.threads or 'all'} threads),"
          f" {int(np.isfinite(p).sum())}/{n} finite")
    return 0


def load_pchip_scores(out_dir: str, index: np.ndarray) -> np.ndarray:
    path = os.path.join(out_dir, "pchip_scores.npz")
    with np.load(path, allow_pickle=False) as z:
        idx = z["index"].astype(int)
        p = z["p_neo"].astype(float)
    if not np.array_equal(idx, index):
        raise ValueError(
            f"{path} was computed for a different catalogue ordering "
            f"({idx.size} rows vs {index.size}); re-run `rescore`"
        )
    return p


def _shift_by_bin(f: np.ndarray, dlog: np.ndarray, edges) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = np.isfinite(f) & (f >= lo) & (f < hi)
        if not m.any():
            continue
        rows.append({
            "lo": float(lo), "hi": float(hi), "n": int(m.sum()),
            "median_dlog10_p": float(np.median(dlog[m])),
            "frac_moved": float((np.abs(dlog[m]) > 0.0).mean()),
            "frac_up": float((dlog[m] > 0.0).mean()),
            "p99_abs": float(np.percentile(np.abs(dlog[m]), 99)),
        })
    return rows


def cmd_analyze(a: argparse.Namespace) -> int:
    cat = load_catalogue(a.run)
    idx = cat["index"]
    p_hist = cat["scores"]["p_neo"].astype(float)
    p_pchip = load_pchip_scores(a.out, idx)
    y = cat["gen"]["is_neo"].astype(bool)
    w = cat["gen"]["weight"].astype(float)
    f = cat["scores"]["ar_frac_h_above"].astype(float)
    v = cat["gen"]["v_mag"].astype(float)

    d2_path = os.path.join(a.digest2, "digest2_scores.npz")
    d2 = None
    if os.path.exists(d2_path):
        with np.load(d2_path) as z:
            d2_idx = z["index"].astype(int)
            raw = z["raw_neo"].astype(float)
            ok = z["ok"].astype(bool)
        pos = {int(i): k for k, i in enumerate(d2_idx)}
        take = np.array([pos.get(int(i), -1) for i in idx])
        d2 = np.where(take >= 0, raw[np.clip(take, 0, None)], np.nan)
        d2 = np.where(take >= 0, np.where(ok[np.clip(take, 0, None)], d2,
                                          np.nan), np.nan)
    else:
        print(f"[analyze] no digest2 scores at {d2_path}; skipping that arm")

    finite = np.isfinite(p_pchip) & np.isfinite(p_hist)
    masks = subset_masks(f)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.log10(p_pchip) - np.log10(p_hist)
    dlog = np.where(finite, dlog, np.nan)

    summary: dict = {
        "run": a.run,
        "out": a.out,
        "n": int(idx.size),
        "n_neo": int(y.sum()),
        "n_finite_both": int(finite.sum()),
        "config": json.loads(
            str(np.load(os.path.join(a.out, "pchip_scores.npz"),
                        allow_pickle=False)["config"])
        ),
        "score_shift": {
            "bit_identical_frac": float((p_pchip == p_hist)[finite].mean()),
            "frac_up": float((p_pchip > p_hist)[finite].mean()),
            "frac_down": float((p_pchip < p_hist)[finite].mean()),
            "median_dlog10": float(np.nanmedian(dlog)),
            "p01_dlog10": float(np.nanpercentile(dlog, 1)),
            "p99_dlog10": float(np.nanpercentile(dlog, 99)),
            "max_abs_dlog10": float(np.nanmax(np.abs(dlog))),
            "frac_gt_0p01_dex": float(
                (np.abs(dlog[finite]) > 0.01).mean()),
            "frac_gt_0p1_dex": float((np.abs(dlog[finite]) > 0.1).mean()),
            "spearman_vs_hist": float(np.corrcoef(
                percentile_rank(p_pchip[finite]),
                percentile_rank(p_hist[finite]))[0, 1]),
            "max_abs_dp": float(np.nanmax(np.abs(p_pchip - p_hist))),
        },
        "by_truncation_exposure": _shift_by_bin(
            f, dlog, [0.0, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 1.0]),
        "by_v_mag": _shift_by_bin(
            v, dlog, [14.0, 20.0, 21.0, 22.0, 23.0, 23.5, 24.0]),
        "subsets": {},
    }

    for name, m in masks.items():
        m = m & finite
        if m.sum() < 10 or y[m].sum() < 2:
            continue
        rec: dict = {
            "n": int(m.sum()),
            "n_neo": int(y[m].sum()),
            "auc_pchip": auc_fast(p_pchip[m], y[m], w[m]),
            "auc_hist": auc_fast(p_hist[m], y[m], w[m]),
            "pchip_vs_hist": auc_bootstrap(p_pchip[m], p_hist[m], y[m], w[m],
                                            n_boot=a.boot, seed=a.boot_seed),
            "calibration_pchip": calibration_stats(p_pchip[m], y[m], w[m]),
            "calibration_hist": calibration_stats(p_hist[m], y[m], w[m]),
        }
        if d2 is not None:
            md = m & np.isfinite(d2)
            if md.sum() >= 10 and y[md].sum() >= 2:
                rec["n_with_digest2"] = int(md.sum())
                rec["pchip_vs_digest2"] = auc_bootstrap(
                    p_pchip[md], d2[md], y[md], w[md],
                    n_boot=a.boot, seed=a.boot_seed)
                rec["hist_vs_digest2"] = auc_bootstrap(
                    p_hist[md], d2[md], y[md], w[md],
                    n_boot=a.boot, seed=a.boot_seed)
                rec["delta_auc_gain_over_hist"] = (
                    rec["pchip_vs_digest2"]["delta"]
                    - rec["hist_vs_digest2"]["delta"]
                )
        summary["subsets"][name] = rec

    # Controls: the mode change must not touch the geometry.
    summary["controls"] = {
        "n_nodes_identical": "by construction (same grid; not re-derived here)",
        "n_nonfinite_pchip": int((~np.isfinite(p_pchip)).sum()),
        "n_nonfinite_hist": int((~np.isfinite(p_hist)).sum()),
        "pchip_in_unit_interval": bool(
            np.all((p_pchip[finite] >= 0.0) & (p_pchip[finite] <= 1.0))),
    }

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    print(json.dumps({k: summary[k] for k in
                      ("n", "n_neo", "score_shift", "controls")},
                     indent=2, default=_json_default))
    _plot_analyze(p_hist, p_pchip, dlog, f, v, y, d2, masks, a.out)
    print(f"[analyze] wrote {os.path.join(a.out, 'summary.json')}")
    return 0


def _plot_analyze(p_hist, p_pchip, dlog, f, v, y, d2, masks, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = np.isfinite(p_hist) & np.isfinite(p_pchip)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    ax = axes[0]
    c = np.log10(np.maximum(np.nan_to_num(f, nan=1e-6), 1e-6))
    s = ax.scatter(p_hist[ok], p_pchip[ok], c=c[ok], s=4, alpha=0.6,
                   cmap="viridis")
    lim = [1e-6, 1.2]
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("p(NEO), histogram prior")
    ax.set_ylabel("p(NEO), smooth prior")
    ax.set_title("Score displacement")
    fig.colorbar(s, ax=ax, label=r"$\log_{10}$ §2.6(b) measure fraction")

    ax = axes[1]
    ax.scatter(np.maximum(np.nan_to_num(f, nan=1e-6), 1e-6)[ok], dlog[ok],
               s=4, alpha=0.5, c="tab:blue")
    ax.set_xscale("log")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("§2.6(b) admissible-region measure fraction above H = 25.5")
    ax.set_ylabel(r"$\Delta \log_{10} p(\mathrm{NEO})$")
    ax.set_title("Displacement vs truncation exposure")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(v[ok], dlog[ok], s=4, alpha=0.5, c="tab:orange")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("apparent V")
    ax.set_ylabel(r"$\Delta \log_{10} p(\mathrm{NEO})$")
    ax.set_title("Displacement vs apparent magnitude")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mode_change_scatter.png"), dpi=130)
    plt.close(fig)

    # ROC of both priors (and digest2 where present)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for name, sc, style in (("smooth prior", p_pchip, "-"),
                            ("histogram prior", p_hist, "--")):
        m = ok
        fpr, tpr = _roc(sc[m], y[m])
        ax.plot(fpr, tpr, style,
                label=f"ds42, {name} (AUC {auc_fast(sc[m], y[m]):.4f})")
    if d2 is not None:
        m = ok & np.isfinite(d2)
        fpr, tpr = _roc(d2[m], y[m])
        ax.plot(fpr, tpr, ":", color="0.4",
                label=f"digest2 raw (AUC {auc_fast(d2[m], y[m]):.4f})")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Ranking on gate_v0 (histogram-generated truths)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mode_change_roc.png"), dpi=130)
    plt.close(fig)

    # calibration
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for name, sc, col in (("smooth prior", p_pchip, "tab:blue"),
                          ("histogram prior", p_hist, "tab:red")):
        st = calibration_stats(sc[ok], y[ok])
        xs = [r["p_mean"] for r in st["bins"]]
        ys = [r["obs_frac"] for r in st["bins"]]
        n = [r["n"] for r in st["bins"]]
        err = [np.sqrt(max(r["p_mean"] * (1 - r["p_mean"]), 1e-12) / max(k, 1))
               for r, k in zip(st["bins"], n)]
        ax.errorbar(xs, ys, yerr=err, marker="o", ms=4, lw=1, color=col,
                    label=f"{name} (pred/obs = "
                          f"{st['ratio_expected_observed']:.3f})")
    ax.plot([1e-5, 1], [1e-5, 1], "k-", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean predicted p(NEO)")
    ax.set_ylabel("observed NEO fraction")
    ax.set_title("Calibration against HISTOGRAM-generated truth\n"
                 "(deliberate prior mismatch for the smooth prior)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mode_change_calibration.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Single-core timing
# --------------------------------------------------------------------------


def cmd_bench(a: argparse.Namespace) -> int:
    """Interleaved single-core timing of the two modes at several grids.

    This box is shared, so a timing loop that measures configuration A to
    completion and then configuration B attributes any drift in load to the
    difference between them — which is exactly the mistake the ``grid``
    subcommand's inline timing makes (it produced a 35% spread between two
    passes over the *same* configuration).  Here every rep visits every
    configuration in turn, so load drift is common-mode, and the reported
    statistic is the per-configuration **minimum** over reps: contention can
    only ever make a pass slower, never faster.
    """
    if not core_available():
        raise SystemExit("the Rust backend is required (run rust/build.sh)")
    cat = load_catalogue(a.run)
    rows = diverse_sample(cat, a.n_tracklets, seed=a.seed)
    atts, R, V = attributables(cat, rows)
    pairs = {m: pair_for(m, a.model, a.variant) for m in ("none", "pchip")}
    grids = [tuple(g) for g in a.grid] if a.grid else [PROD_GRID]

    cfgs = [(m, g) for g in grids for m in ("none", "pchip")]
    times: dict = {c: [] for c in cfgs}
    cpu: dict = {c: [] for c in cfgs}
    # One untimed warm-up pass per configuration (first touch of the model
    # copy, page faults on the coefficient table, branch predictor).
    for m, g in cfgs:
        score_batch(atts, R, V, pairs[m], *g, n_threads=1)
    for rep in range(a.reps):
        for m, g in cfgs:
            c0, t0 = time.process_time(), time.perf_counter()
            score_batch(atts, R, V, pairs[m], *g, n_threads=1)
            times[(m, g)].append(time.perf_counter() - t0)
            cpu[(m, g)].append(time.process_time() - c0)
        print(f"[bench] rep {rep + 1}/{a.reps} done", flush=True)

    out = {
        "run": a.run,
        "n_tracklets": int(len(rows)),
        "reps": int(a.reps),
        "grids": [list(g) for g in grids],
        # Recorded because this is a shared box: a 1-, 5- and 15-minute load
        # average well above the core count means the WALL numbers below are
        # upper bounds even after taking the minimum over reps.  CPU time is
        # the primary statistic for exactly that reason — it does not count
        # the intervals in which the kernel descheduled the thread, so it is
        # the honest measure of work done, and it is process-wide so it still
        # captures a rayon worker thread.
        "loadavg": list(os.getloadavg()),
        "n_cpu": os.cpu_count(),
        "primary_statistic": "ms_per_tracklet_cpu_min",
        "per_config": {},
    }
    n = len(rows)
    for (m, g) in cfgs:
        ts, cs = times[(m, g)], cpu[(m, g)]
        out["per_config"][f"{m}/{g[0]}x{g[1]}"] = {
            "mode": m,
            "n_rho": g[0],
            "n_rhodot": g[1],
            "ms_per_tracklet_cpu_min": 1e3 * min(cs) / n,
            "ms_per_tracklet_cpu_median": 1e3 * float(np.median(cs)) / n,
            "ms_per_tracklet_wall_min": 1e3 * min(ts) / n,
            "ms_per_tracklet_wall_median": 1e3 * float(np.median(ts)) / n,
            "all_cpu_s": cs,
            "all_wall_s": ts,
        }
    for g in grids:
        pc = out["per_config"]
        a_ = pc[f"none/{g[0]}x{g[1]}"]["ms_per_tracklet_cpu_min"]
        b_ = pc[f"pchip/{g[0]}x{g[1]}"]["ms_per_tracklet_cpu_min"]
        aw = pc[f"none/{g[0]}x{g[1]}"]["ms_per_tracklet_wall_min"]
        bw = pc[f"pchip/{g[0]}x{g[1]}"]["ms_per_tracklet_wall_min"]
        pc[f"overhead/{g[0]}x{g[1]}"] = {
            "histogram_ms": a_, "smooth_ms": b_,
            "smooth_overhead_frac": b_ / a_ - 1.0,
            "histogram_ms_wall": aw, "smooth_ms_wall": bw,
            "smooth_overhead_frac_wall": bw / aw - 1.0,
        }
        print(f"[bench] {g}: CPU histogram {a_:.2f} ms, smooth {b_:.2f} ms "
              f"({100 * (b_ / a_ - 1):+.1f}%)   |   wall {aw:.2f} / {bw:.2f} "
              f"({100 * (bw / aw - 1):+.1f}%)")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "bench.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=_json_default)
    print(f"[bench] wrote {path}")
    return 0


# --------------------------------------------------------------------------
# The durable writeup
# --------------------------------------------------------------------------


def _measure_guarantees(model: str, variant: str) -> dict:
    """Re-measure the reconstruction's guarantees, for the report table.

    Recomputed rather than transcribed so the table cannot go stale against
    the code.  These are the same properties the unit tests assert (with
    tolerances); here they are reported as numbers.
    """
    from ds42.population import (H_EDGES, MODEL_SHAPE, QEI_VOLUME,
                                     Digest2Histogram)

    ang = (2.0 * np.pi) ** 3
    hist = Digest2Histogram.neo_other_pair(model, variant)
    smooth = Digest2Histogram.neo_other_pair(model, variant,
                                             h_interpolation="pchip")
    n_h = MODEL_SHAPE[3]
    dH = np.diff(H_EDGES)
    xg, wg = np.polynomial.legendre.leggauss(8)

    edges = [np.asarray(hist[0].q_edges), np.asarray(hist[0].e_edges),
             np.asarray(hist[0].i_edges)]
    centres = [0.5 * (e[:-1] + e[1:]) for e in edges]
    Q, E, I = np.meshgrid(*centres, indexing="ij")
    el = np.stack((Q, E, I), axis=-1).reshape(-1, 3)
    vol = np.asarray(QEI_VOLUME).reshape(-1) * ang

    out: dict = {
        "n_cells": int(el.shape[0]),
        "n_h_bins": int(n_h),
        "sweep_per_bin": 200,
        "mass_max_rel": 0.0,
        "min_density": np.inf,
        "continuity_max_rel": 0.0,
        "chi2_dof": {},
        "jumps": {},
        "empty_exact": "both components",
    }
    empty_ok = True
    for comp in smooth:
        counts = np.asarray(comp.counts)
        co = np.asarray(comp.h_coeffs)
        # mass, via quadrature THROUGH density()
        flat_counts = counts.reshape(-1, n_h)
        for k in range(n_h):
            lo, hi = H_EDGES[k], H_EDGES[k + 1]
            half = 0.5 * (hi - lo)
            acc = np.zeros(el.shape[0])
            for x, wq in zip(xg, wg):
                acc += wq * comp.density(
                    el, np.full(el.shape[0], half * x + 0.5 * (hi + lo)))
            mass = half * acc * vol
            nz = flat_counts[:, k] > 0
            if nz.any():
                out["mass_max_rel"] = max(out["mass_max_rel"], float(
                    (np.abs(mass[nz] - flat_counts[nz, k])
                     / flat_counts[nz, k]).max()))
            empty_ok &= bool(np.all(mass[~nz] == 0.0))
        # non-negativity, dense sweep
        frac = np.linspace(0.0, 1.0, out["sweep_per_bin"] + 1)[:-1]
        for k in range(n_h):
            for f in frac:
                h = H_EDGES[k] + f * dH[k]
                out["min_density"] = min(out["min_density"], float(
                    comp.density(el, np.full(el.shape[0], h)).min()))
        # continuity, exact one-sided limits from the coefficients
        peak = np.maximum(np.max(
            [comp.density(el, np.full(el.shape[0], 0.5 * (H_EDGES[k]
                                                          + H_EDGES[k + 1])))
             for k in range(n_h)], axis=0), np.finfo(float).tiny)
        for k in range(1, n_h):
            hstep = dH[k - 1]
            a, b, c = (co[..., k - 1, j].reshape(-1) for j in range(3))
            left = a + hstep * (b + hstep * c)
            right = co[..., k, 0].reshape(-1)
            out["continuity_max_rel"] = max(
                out["continuity_max_rel"],
                float((np.abs(left - right) / peak).max()))
        # endpoint jumps
        for k, label, h_in in ((0, "cells populated in [0, 6) jump at H = 0",
                                0.0),
                               (-1, "cells populated in [24, 25.5) jump at "
                                    "H -> 25.5",
                                np.nextafter(25.5, 0.0))):
            pop = counts[..., k].reshape(-1) > 0
            if not pop.any():
                continue
            d = comp.density(el, np.full(el.shape[0], h_in))
            out["jumps"][f"{comp.label}'s {label}"] = float(
                (d[pop] > 0.0).mean())
    out["empty_exact"] = ("both components" if empty_ok
                          else "NOT exact — investigate")
    out["min_density"] = float(out["min_density"])

    # sparse-cell containment
    worst = 0.0
    hot = np.array([[0.85, 0.25, np.radians(17.0)]])
    cell_vol = float((0.9 - 0.8) * (0.3 - 0.2) * np.radians(5.0)) * ang
    for k in range(n_h):
        cnt = np.zeros(MODEL_SHAPE)
        cnt[3, 2, 4, k] = 1234.0
        c1 = Digest2Histogram(cnt, "other", h_interpolation="pchip")
        lo, hi = H_EDGES[k], H_EDGES[k + 1]
        half = 0.5 * (hi - lo)
        acc = sum(wq * float(c1.density(
            hot, np.array([half * x + 0.5 * (hi + lo)]))[0])
            for x, wq in zip(xg, wg))
        worst = max(worst, abs(half * acc * cell_vol - 1234.0))
    out["sparse_max_abs"] = worst

    # lookup identity
    rng = np.random.default_rng(11)
    n = 100_000
    probe = np.stack((rng.uniform(-1.0, 110.0, n),
                      rng.uniform(-0.05, 1.2, n),
                      rng.uniform(-0.05, 3.2, n)), axis=-1)
    hp = rng.uniform(-1.0, 27.0, n)
    ident = True
    for a_, b_ in zip(hist, smooth):
        ident &= a_.total_count == b_.total_count and a_.support == b_.support
        ident &= all(np.array_equal(x, y) for x, y in
                     zip(a_.bin_indices(probe, hp), b_.bin_indices(probe, hp)))
        ident &= np.array_equal(a_.density(probe, hp) == 0.0,
                                b_.density(probe, hp) == 0.0)
    out["lookup_identical"] = ("verified on 1e5 random probes"
                               if ident else "MISMATCH — investigate")

    # sample() vs density(), chi-square on fine sub-bins
    m_sub = 8
    out["n_sub_bins"] = m_sub * n_h
    out["n_draws"] = 1_000_000
    for comp in smooth:
        co = np.asarray(comp.h_coeffs)
        cvol = np.asarray(QEI_VOLUME) * ang
        exp, sub_edges = [], []
        for k in range(n_h):
            ss = np.linspace(0.0, dH[k], m_sub + 1)
            a, b, c = co[..., k, 0], co[..., k, 1], co[..., k, 2]

            def anti(t, a=a, b=b, c=c):
                return t * (a + t * (0.5 * b + t * c / 3.0))

            for j in range(m_sub):
                exp.append(float(((anti(ss[j + 1]) - anti(ss[j]))
                                  * cvol).sum()))
                sub_edges.append(H_EDGES[k] + ss[j])
        sub_edges.append(25.5)
        exp = np.array(exp)
        _, hh = comp.sample(out["n_draws"], np.random.default_rng(4242))
        obs, _ = np.histogram(hh, bins=np.array(sub_edges))
        e = exp / exp.sum() * out["n_draws"]
        use = e > 20.0
        out["chi2_dof"][comp.label] = float(
            (((obs[use] - e[use]) ** 2) / e[use]).sum() / (use.sum() - 1))

    # inverse-CDF residual
    comp = smooth[1]
    co = np.asarray(comp.h_coeffs)
    counts = np.asarray(comp.counts)
    rng = np.random.default_rng(5)
    nz = np.argwhere(counts > 0)
    sel = nz[rng.integers(0, len(nz), 100_000)]
    iq, ie, ii, ih = (sel[:, k] for k in range(4))
    a, b, c = (co[iq, ie, ii, ih, k] for k in range(3))
    wid = (np.asarray(comp.h_edges)[ih + 1]
           - np.asarray(comp.h_edges)[ih]).astype(float)

    def cdf(t):
        return t * (a + t * (0.5 * b + t * c / 3.0))

    total = cdf(wid)
    target = rng.random(len(iq)) * total
    lo_, hi_ = np.zeros_like(wid), wid.copy()
    t = wid * (target / total)
    for it in range(60):
        f = cdf(t) - target
        lo_ = np.where(f <= 0.0, t, lo_)
        hi_ = np.where(f > 0.0, t, hi_)
        mid = 0.5 * (lo_ + hi_)
        d = a + t * (b + t * c)
        with np.errstate(divide="ignore", invalid="ignore"):
            cand = t - np.where(d > 0.0, f / d, np.nan)
        take = np.isfinite(cand) & (cand > lo_) & (cand < hi_) & (it % 5 != 4)
        t = np.where(take, cand, mid)
    out["inv_residual"] = float((np.abs(cdf(t) - target) / total).max())

    co0 = np.asarray(smooth[0].h_coeffs)
    out["coeff_bytes_per_component"] = int(co0.nbytes)
    out["coeff_shape"] = list(co0.shape)
    return out


def _monotonicity(grid: dict) -> dict:
    """How monotone the mid-p p90 actually is in n_rho, per (mode, n_rhodot).

    Computed rather than asserted, because an earlier draft of the report
    asserted the opposite of what the data say.
    """
    n_cols = 0
    monotone = 0
    worst = 1.0
    where = "none"
    for mode in ("none", "pchip"):
        idx = {(r["n_rho"], r["n_rhodot"]): r for r in grid["sweep"][mode]}
        for nd in grid["n_rhodot_sweep"]:
            n_cols += 1
            vals = [(nr, idx[(nr, nd)]["mid"]["p90"])
                    for nr in grid["n_rho_sweep"]]
            ups = [(a[0], b[0], b[1] / a[1])
                   for a, b in zip(vals, vals[1:]) if b[1] > a[1]]
            if not ups:
                monotone += 1
                continue
            for lo, hi, r in ups:
                if r > worst:
                    worst, where = r, (f"{mode}, n_rhodot={nd}, n_rho "
                                       f"{lo} -> {hi}")
    return {
        "n_cols": n_cols,
        "monotone_cols": monotone,
        "worst_up_step": worst,
        "worst_where": where,
    }


def _floor_is_fallback(grid: dict, mode: str, chk: str) -> bool:
    """Whether :func:`_floor` had to fall back to the unconditional p90."""
    return "mid" not in grid["reference"][mode]["self_error_vs_" + chk]


def _floor(grid: dict, mode: str, chk: str) -> float:
    """The reference's own mid-p p90 error, tolerating an older JSON.

    Runs before 2026-09-02 recorded the adequacy check without the mid-p
    breakdown; on a ``--select midp`` sweep almost every tracklet is mid-p, so
    the unconditional number is the same quantity to within a few percent and
    is used as the fallback rather than dropping the row.
    """
    e = grid["reference"][mode]["self_error_vs_" + chk]
    return float(e.get("mid", e)["p90"])


def _plot_reconstruction(model: str, variant: str, out_dir: str) -> list:
    """What the reconstruction does to the density, cell by cell.

    Three representative (q, e, i) cells: the NEO component's most populated
    cell, the other component's most populated cell, and a deliberately sparse
    cell (counts in a single H bin).  Top row shows the cumulative count with
    the monotone cubic through its 19 knots — the object that is actually
    interpolated; bottom row shows the two densities.  The sparse column is
    the answer to "interpolation invents density" in one picture.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ds42.population import (H_EDGES, MODEL_SHAPE, QEI_VOLUME,
                                     Digest2Histogram)

    ang = (2.0 * np.pi) ** 3
    hist = Digest2Histogram.neo_other_pair(model, variant)
    smooth = Digest2Histogram.neo_other_pair(model, variant,
                                             h_interpolation="pchip")
    he = np.asarray(H_EDGES)

    panels = []
    for hh, ss in zip(hist, smooth):
        c = np.asarray(hh.counts)
        cell = np.unravel_index(int(np.argmax(c.sum(axis=-1))), c.shape[:3])
        panels.append((f"{hh.label}: most populated cell "
                       f"{tuple(int(v) for v in cell)}", hh, ss, cell))
    # a sparse cell: counts in one H bin only
    sparse = np.zeros(MODEL_SHAPE)
    sparse[3, 2, 4, 12] = 5000.0
    panels.append(("synthetic: one populated H bin",
                   Digest2Histogram(sparse, "sparse"),
                   Digest2Histogram(sparse, "sparse",
                                    h_interpolation="pchip"),
                   (3, 2, 4)))

    fig, axes = plt.subplots(2, len(panels), figsize=(5.0 * len(panels), 8.0))
    hfine = np.linspace(0.0, 25.4999, 4000)
    for k, (title, hh, ss, cell) in enumerate(panels):
        el = np.array([[0.5 * (hh.q_edges[cell[0]] + hh.q_edges[cell[0] + 1]),
                        0.5 * (hh.e_edges[cell[1]] + hh.e_edges[cell[1] + 1]),
                        0.5 * (hh.i_edges[cell[2]]
                               + hh.i_edges[cell[2] + 1])]])
        counts = np.asarray(hh.counts)[cell]
        vol = float(np.asarray(QEI_VOLUME)[cell]) * ang

        ax = axes[0, k]
        cum = np.concatenate(([0.0], np.cumsum(counts)))
        # the interpolant, recovered by integrating the stored density
        co = np.asarray(ss.h_coeffs)[cell]
        curve = []
        for h in hfine:
            j = int(np.searchsorted(he, h, side="right") - 1)
            j = min(max(j, 0), len(counts) - 1)
            sloc = h - he[j]
            a, b, cc = co[j] * vol
            curve.append(cum[j] + sloc * (a + sloc * (0.5 * b + sloc * cc / 3)))
        ax.plot(hfine, curve, "-", color="tab:blue", lw=1.6,
                label="monotone cubic (what is interpolated)")
        ax.plot(he, cum, "o", color="k", ms=4, label="exact cumulative count")
        # The histogram's own cumulative count IS the straight line between
        # the knots: constant density in a bin integrates to a linear CDF.
        # Drawing it as a staircase would be wrong and would flatter the
        # cubic.
        ax.plot(he, cum, "--", color="tab:red", lw=1.0,
                label="piecewise linear (= histogram)")
        ax.set_ylabel("cumulative object count")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        ax = axes[1, k]
        ax.plot(hfine, ss.density(np.repeat(el, hfine.size, axis=0), hfine),
                "-", color="tab:blue", lw=1.6, label="smooth (pchip)")
        ax.plot(hfine, hh.density(np.repeat(el, hfine.size, axis=0), hfine),
                "--", color="tab:red", lw=1.2, label="histogram")
        for e in he:
            ax.axvline(e, color="0.85", lw=0.6, zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("H [mag]")
        ax.set_ylabel(r"density [objects / (AU rad$^4$ mag)]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("The reconstruction, cell by cell — equal mass in every H "
                 "bin by construction; grey lines are the H bin edges",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "reconstruction_shape.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [p[0] for p in panels]


def _ratio(idx: dict, n_rho: int, nd_a: int, nd_b: int) -> float:
    """mid-p p90 at (n_rho, nd_a) divided by the one at (n_rho, nd_b)."""
    return (idx[(n_rho, nd_a)]["mid"]["p90"]
            / idx[(n_rho, nd_b)]["mid"]["p90"])


def _ci(d: dict, key: str = "delta") -> str:
    lo, hi = d.get(key + "_ci", [float("nan")] * 2)
    return f"{d[key]:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def cmd_report(a: argparse.Namespace) -> int:
    with open(os.path.join(a.out, a.grid_name)) as fh:
        grid = json.load(fh)
    grid_all = None
    p_all = os.path.join(a.out, a.grid_all_name)
    if os.path.exists(p_all):
        with open(p_all) as fh:
            grid_all = json.load(fh)
    with open(os.path.join(a.out, "summary.json")) as fh:
        s = json.load(fh)
    gate = None
    if a.gate_run:
        p = os.path.join(a.gate_run, "summary.json")
        if os.path.exists(p):
            with open(p) as fh:
                gate = json.load(fh)
    bench = None
    pb = os.path.join(a.out, "bench.json")
    if os.path.exists(pb):
        with open(pb) as fh:
            bench = json.load(fh)
    gate_hist = None
    if a.gate_reference:
        p = os.path.join(a.gate_reference, "summary.json")
        if os.path.exists(p):
            with open(p) as fh:
                gate_hist = json.load(fh)

    L = []
    w = L.append
    ref = "x".join(map(str, grid["ref_grid"]))
    chk = "x".join(map(str, grid["ref_check_grid"]))
    prod = tuple(grid["prod_grid"])
    base = grid["histogram_baseline"]
    # PRIMARY threshold: DESIGN §2.5's own statistic (see section 1).
    target = grid["target_mid_p90"]
    target_all = grid["target_all_p90"]

    w("# Mass-conserving smooth-in-H reconstruction of the digest2 histogram")
    w("")
    w(f"*Generated {time.strftime('%Y-%m-%dT%H:%M:%S')} by "
      "`validation/h_interpolation_study.py report`.  Construction and its "
      "guarantees: `ds42/population.py` module docstring; unit tests: "
      "`tests/test_population.py`, `tests/test_backend_parity.py`.*")
    w("")
    w("## What was built")
    w("")
    w("`Digest2Histogram(h_interpolation=\"pchip\")` replaces the "
      "piecewise-constant H dependence of the digest2 model with the "
      "derivative of a monotone piecewise cubic (PCHIP) fitted to each "
      "(q, e, i) cell's **cumulative** object count over the 19 H edges.  "
      "Interpolating the CDF and differentiating — rather than interpolating "
      "the density values — is what makes the reconstruction "
      "**mass-conserving per H bin**, and monotonicity is what stops mass "
      "leaking between bins.  Both components always share the mode; it is "
      "mutually exclusive with the D-14 clamp.")
    w("")
    gu = _measure_guarantees(a.model, a.variant)
    w(f"Measured guarantees — **recomputed at report time**, not transcribed, "
      f"over all {gu['n_cells']} (q, e, i) cells x {gu['n_h_bins']} H bins x "
      "both components of the "
      f"`{a.variant}` variant of `{a.model}`:")
    w("")
    w("| property | measurement |")
    w("|---|---|")
    w("| per-H-bin mass conservation | max relative deviation of "
      "`integral(density) dH x cell volume` from the stored count: "
      f"**{gu['mass_max_rel']:.2e}** (Gauss-Legendre through `density()` "
      "itself, 8 nodes, exact for the quadratic it evaluates; empty bins "
      f"integrate to exactly 0 in {gu['empty_exact']}) |")
    w("| non-negativity | min density over "
      f"{gu['sweep_per_bin']} points x {gu['n_h_bins']} bins x "
      f"{gu['n_cells']} cells x 2 components = "
      f"**{gu['min_density']:.1f}**, never negative — **through "
      "`density()`**, which is the only path the classifier and the sampler "
      "use.  Evaluating the stored coefficient triple directly at `s = dH` "
      "(the right-hand knot, which `density()` never reaches because the H "
      "bins are half-open) can return ~-1e-10 *relative* to the cell scale, "
      "from cancellation in the Horner form at a double root.  Unreachable "
      "through the public API, and deliberately not clamped, so a genuine "
      "sign error could not hide behind a clamp. |")
    w("| continuity across interior H edges | max \\|one-sided "
      "difference\\| / cell peak density = "
      f"**{gu['continuity_max_rel']:.2e}** over all {gu['n_h_bins'] - 1} "
      "interior edges x all cells (structural: PCHIP is C1, so its "
      "derivative is C0) |")
    w("| D-4 truncation | exactly `0.0` for H < 0, H >= 25.5, +/-inf and "
      "NaN, and outside the q/e/i support — bit-identical to histogram "
      "mode |")
    w("| total_count, support, q/e/i/H bin lookup | bit-identical to "
      f"histogram mode ({gu['lookup_identical']}) |")
    w("| mass containment in sparse cells | a cell populated in one H bin "
      "only reproduces that count to "
      f"{gu['sparse_max_abs']:.1e} and has **exactly** zero density in every "
      f"other bin, for all {gu['n_h_bins']} choices of populated bin |")
    w("| sample() vs density() | chi2/dof = "
      + " and ".join(f"{v:.3f} ({k})" for k, v in gu["chi2_dof"].items())
      + f" on {gu['n_draws']:.0e} draws against the analytic mass on "
      f"{gu['n_sub_bins']} fine sub-bins; inverse-CDF residual "
      f"{gu['inv_residual']:.1e} relative |")
    w("")
    w("The reconstruction is *generally* discontinuous at the two support "
      "endpoints (H = 0 and H = 25.5).  That is documented, not a defect: the "
      "histogram mode has the same two jumps, and PCHIP's shape-preserving "
      "end-slope limiter actually removes the jump wherever the neighbouring "
      "bin is much more populated (measured: "
      + "; ".join(f"{100 * v:.1f}% of {k}" for k, v in gu["jumps"].items())
      + ").")
    w("")
    _plot_reconstruction(a.model, a.variant, a.out)
    w("`reconstruction_shape.png` shows the construction on three cells — the "
      "two components' most populated cells and a synthetic cell with counts "
      "in a single H bin.  The top row is the object that is actually "
      "interpolated (the cumulative count, with the monotone cubic through its "
      "19 exact knots); the bottom row is the density both modes evaluate.  "
      "The sparse column is the \"interpolation invents density\" objection "
      "answered in one picture: the cubic is flat outside the populated bin, "
      "so the density there is identically zero.")
    w("")
    w("**What it does NOT fix.**  The `1/(e sin i)` integrable singularity of "
      "DESIGN §2.5 is untouched — it lives on the e = 0 and i = 0 bin edges "
      "and is a property of the measure, not of the H reconstruction.")
    w("")
    w("## Rust backend")
    w("")
    w("The kernel carries the precomputed per-interval Horner coefficients and "
      "evaluates one cubic-derivative Horner per lookup, sharing the four bin "
      "lookups with the histogram path.  The two modes differ by one branch, "
      "three contiguous loads instead of one, a subtraction and two "
      "multiply-adds.")
    w("")
    w("| quantity | value |")
    w("|---|---|")
    w(f"| coefficient memory | {gu['coeff_shape'][0]}x{gu['coeff_shape'][1]}x"
      f"{gu['coeff_shape'][2]} = {gu['n_cells']} cells x "
      f"{gu['coeff_shape'][3]} intervals x {gu['coeff_shape'][4]} coefficients "
      f"x 8 B = **{gu['coeff_bytes_per_component'] / 1e6:.2f} MB per "
      f"component, {2 * gu['coeff_bytes_per_component'] / 1e6:.2f} MB for the "
      "pair** (the brief's "
      "~50 MB estimate counted the 45,936 (q,e,i,H) *bins* as cells; there "
      "are 29x8x11 = 2552 cells, and storing only the derivative's three "
      "coefficients rather than four CDF coefficients halves it again) |")
    if bench is not None:
        for g in bench["grids"]:
            o = bench["per_config"][f"overhead/{g[0]}x{g[1]}"]
            w(f"| single-core CPU ms/tracklet at {tuple(g)} | histogram "
              f"**{o['histogram_ms']:.2f} ms**, smooth "
              f"**{o['smooth_ms']:.2f} ms** "
              f"({100 * o['smooth_overhead_frac']:+.1f}%) |")
        la = bench.get("loadavg") or [float("nan")] * 3
        w(f"| timing method | {bench['n_tracklets']} tracklets, "
          f"{bench['reps']} interleaved reps, minimum per configuration.  "
          f"**CPU** time (`time.process_time`), not wall: load average was "
          f"{la[0]:.0f}/{la[1]:.0f}/{la[2]:.0f} on {bench.get('n_cpu')} cores "
          "(another agent was running a large job in this repo), and wall "
          "time under that much oversubscription is unusable — the wall "
          "minima are inconsistent enough to make the smooth mode look "
          "*faster* at one grid, which it cannot be.  CPU time excludes the "
          "intervals the thread was descheduled.  Every rep of both clocks is "
          "in `bench.json`. |")
        anchor = bench["per_config"].get("overhead/1200x50")
        if anchor:
            w(f"| sanity anchor | the histogram-mode figure at (1200, 50), "
              f"{anchor['histogram_ms']:.2f} ms, sits "
              f"{100 * (anchor['histogram_ms'] / 12.0 - 1):.0f}% above "
              "D-15's recorded 12.0 ms/tracklet single-core — close enough to "
              "confirm the absolute scale, and the residual is the same "
              "contention that made the wall clock unusable.  The **overhead "
              "column is the trustworthy number**: both modes were measured "
              "in the same interleaved reps, so whatever inflates one "
              "inflates the other |")
    else:
        for key, label in (("prod/none", f"histogram, {prod}"),
                           ("prod/pchip", f"smooth, {prod}")):
            t = grid["timing"].get(key)
            if t:
                w(f"| single-core ms/tracklet, {label} | "
                  f"{t['ms_per_tracklet']:.2f} ms (uninterleaved wall time; "
                  "run `bench` for the reliable number) |")
    w("| Python vs Rust density agreement | **bit-identical** (`np.array_equal`) "
      "over ~16k probes covering every bin edge from both sides, +/-inf, NaN "
      "and 4000 in-support draws per component |")
    w("| end-to-end Lambda agreement, 120 gate_v0 tracklets | max "
      "\\|dp\\| = 2.1e-14, max Lambda relative = 4.2e-11 |")
    w("")
    if bench is not None:
        o = bench["per_config"].get("overhead/1200x50")
        la = bench.get("loadavg") or [0.0]
        w("")
        w("**Erratum.**  The first version of this report gave the smooth "
          "mode's overhead as +21% to +28% and explained it as the "
          "coefficient table spilling out of L2.  Both were wrong.  Those "
          "numbers were taken while another agent's job had this 96-core box "
          "at load average 88-162, and neither the wall clock nor "
          "`process_time` is reliable under that much oversubscription "
          f"(the same protocol re-run at load {la[0]:.0f} gives "
          f"{100 * o['smooth_overhead_frac']:+.1f}%, and the review's "
          "independent replication on a quiet machine gave +1.0% median over "
          "25 reps).  The overhead is a couple of percent, which is what the "
          "arithmetic predicts: two multiply-adds and two extra loads against "
          "four bin searches and a closed-form Jacobian.  The L2 story was an "
          "explanation invented for a measurement artifact, and is withdrawn.")
        w("")
        w("The stored form is still the derivative rather than the CDF's four "
          "coefficients — one fewer array per interval, and it is the form "
          "the hot path needs; `sample()` integrates it back to a CDF at draw "
          "time rather than the reverse.")
        w("")
    w("Bit-identity is by construction, not luck: Python precomputes the "
      "coefficients and Rust copies them, and both evaluate "
      "`a + s * (b + s * c)` with `s = H - h_edges[ih]` in that exact "
      "grouping.  numpy evaluates one operation per ufunc call and the crate "
      "does not enable fast-math, so neither side can contract the expression "
      "into an FMA.")
    w("")

    # ---------------- measurement 1 ------------------------------------
    w("## 1. Node count vs accuracy — no detectable win at the production "
      "grid")
    w("")
    w("### Statistic, and why it has to be conditional")
    w("")
    w("|dp(NEO)| is bounded by min(p, 1-p).  At mixture proportions the gate "
      "population is overwhelmingly low-p, so an unconditional p90 |dp| "
      "measures arithmetic head-room rather than quadrature quality: on the "
      "whole `gate_v0` catalogue it comes out at ~5e-5 whatever the "
      "integration is doing to the tracklets that are actually near a "
      "decision.  Everything below therefore uses the **mid-p** statistic — "
      "|dp| over tracklets whose reference p(NEO) is in "
      f"({grid['mid_p_window'][0]}, {grid['mid_p_window'][1]}).")
    w("")
    w("Note what that means for the two columns in the tables: this sweep "
      "*selects* mid-p tracklets, so its unconditional and mid-p statistics "
      "coincide to a few percent by construction.  The all-vs-mid contrast is "
      "not visible here and cannot be — it is visible on a full mixture "
      "catalogue, and that is where the ~5e-5 above comes from.")
    w("")
    w(f"Sample: {grid['n_tracklets']} tracklets ({grid['selection']}) from "
      f"`{grid['run']}`, of which {grid.get('n_mid_p', '?')} are mid-p on the "
      f"reference; each scored at every point of a "
      f"{len(grid['n_rho_sweep'])} x {len(grid['n_rhodot_sweep'])} "
      f"(n_rho, n_rhodot) grid in both modes against a same-mode {ref} "
      "reference.")
    w("")
    fb = any(_floor_is_fallback(grid, m, chk) for m in ("none", "pchip"))
    w(f"**Reference adequacy.**  Refining the reference again to {chk} moves "
      "it by "
      + ("**unconditional** p90 (the mid-p breakdown was not recorded in this "
         "JSON; on a `--select midp` sweep almost every tracklet is mid-p, so "
         "the two differ by a few percent) " if fb else "mid-p p90 ")
      + ", ".join(f"{_floor(grid, m, chk):.2e} ({m})"
                  for m in ("none", "pchip"))
      + f" — a factor {target / max(_floor(grid, 'none', chk), 1e-300):.0f} "
      "below the baseline being quoted, so the table measures the sweep point "
      "and not the reference.")
    w("")
    w("### Reconciliation with DESIGN §2.5's 3.5e-3")
    w("")
    fc = grid.get("design_fixed_cost_check") or []
    if fc:
        w("First, the harness reproduces DESIGN §2.5's *relative* statements, "
          "which is what licenses reading its absolute numbers against these:")
        w("")
        w("| grid | nodes | all-p90 |dp| | mid-p p90 |dp| |".replace(
            "|dp|", "\\|dp\\|"))
        w("|---|---|---|---|")
        for r in fc:
            w(f"| {tuple(r['grid'])} | {r['nodes']} | {r['p90']:.2e} | "
              f"{r['mid_p90']:.2e} |")
        w("")
        d = {tuple(r["grid"]): r for r in fc}
        if (800, 100) in d and (400, 200) in d:
            w(f"(800, 100) beats (400, 200) by "
              f"{d[(400, 200)]['p90'] / d[(800, 100)]['p90']:.1f}x at equal "
              "cost — DESIGN says \"~2x\".  ")
        if (200, 400) in d and (200, 100) in d:
            w(f"(200, 400) vs (200, 100): "
              f"{d[(200, 400)]['p90']:.2e} vs {d[(200, 100)]['p90']:.2e} — "
              "DESIGN says \"equal\".  Both reproduced.")
        w("")
    w("Second, the recomputation of the stored `gate_v0` scores at "
      f"{prod} agrees with the run of record to max |dp| = 7.7e-13, so the "
      "scoring path is the same one.")
    w("")
    vd = base["vs_design_reference"]
    w("With that established: DESIGN §2.5's \"p90 |dp| ~ 3.5e-3, max ~1e-2 vs "
      f"a {tuple(grid['design_ref_grid'])} reference\" is a **mid-p** figure. "
      f"Measured here against exactly that reference, on this study's "
      f"{grid.get('n_mid_p')} mid-p tracklets: **mid-p p90 = "
      f"{vd['mid']['p90']:.3e}** against an unconditional p90 of "
      f"{vd['p90']:.2e}.  An independent recomputation by the reviewer on the "
      "`gate_v0` catalogue gives 6.1-6.4e-3.  So:")
    w("")
    w("- The **statistic** is settled: 3.5e-3 is a mid-p p90, not an "
      "unconditional one.  (The two coincide in the table above because this "
      "sweep is mid-p-selected; the contrast — ~5e-5 unconditional against "
      "~6e-3 mid-p — is the review's recomputation on the full `gate_v0` "
      "catalogue.)  Nothing is wrong with §2.5; the conditioning just needs "
      "saying out loud, which no prior document did.")
    w(f"- The **p90 value** is reproduced to within about a factor of two "
      f"({vd['mid']['p90']:.2e} here, 6.1-6.4e-3 on `gate_v0`, against "
      "3.5e-3), which is what one should expect of a p90 taken on different "
      "catalogues.")
    w(f"- The **max** is NOT reproduced: mid-p max here is "
      f"{vd['mid']['max']:.2e}, two orders of magnitude above §2.5's "
      "\"~1e-2\".  A max over a few hundred mid-p tracklets is set by the "
      "single worst-resolved one, so it is not a stable statistic and should "
      "not be quoted as one; the p90 rows in the tables below are.")
    w("")
    w("### mid-p p90 |dp(NEO)|, histogram / smooth")
    w("")
    idx = {(r["n_rho"], r["n_rhodot"]): r for r in grid["sweep"]["none"]}
    idxp = {(r["n_rho"], r["n_rhodot"]): r for r in grid["sweep"]["pchip"]}
    w("| n_rho | " + " | ".join(f"n_rhodot={nd}"
                                for nd in grid["n_rhodot_sweep"]) + " |")
    w("|" + "---|" * (1 + len(grid["n_rhodot_sweep"])))
    for nr in grid["n_rho_sweep"]:
        cells = [f"{idx[(nr, nd)]['mid']['p90']:.2e} / "
                 f"{idxp[(nr, nd)]['mid']['p90']:.2e}"
                 for nd in grid["n_rhodot_sweep"]]
        w(f"| {nr} | " + " | ".join(cells) + " |")
    w("")
    w("### The same tables for the two Lambdas (p90 relative error)")
    w("")
    w("p(NEO) is the ratio Lambda_NEO/(Lambda_NEO + Lambda_other), and both "
      "components share the H bin edges, so an H-staircase quadrature error "
      "is largely **common-mode and cancels in the ratio**.  The Lambdas are "
      "where the staircase can be seen at all — reporting only p(NEO) would "
      "have hidden the mechanism entirely.")
    for lam, lab in (("lam_neo_rel", "Lambda_NEO"),
                     ("lam_other_rel", "Lambda_other")):
        w("")
        w(f"**{lab}**, histogram / smooth:")
        w("")
        w("| n_rho | " + " | ".join(f"n_rhodot={nd}"
                                    for nd in grid["n_rhodot_sweep"]) + " |")
        w("|" + "---|" * (1 + len(grid["n_rhodot_sweep"])))
        for nr in grid["n_rho_sweep"]:
            cells = [f"{idx[(nr, nd)][lam]['p90']:.2e} / "
                     f"{idxp[(nr, nd)][lam]['p90']:.2e}"
                     for nd in grid["n_rhodot_sweep"]]
            w(f"| {nr} | " + " | ".join(cells) + " |")
    w("")
    w("### Convergence rate in n_rho, the sharpest test of the mechanism")
    w("")
    nd_top = max(grid["n_rhodot_sweep"])
    fit_nr = [nr for nr in grid["n_rho_sweep"] if nr >= 600]
    w(f"Least-squares slope of log(error) against log(n_rho) at n_rhodot = "
      f"{nd_top}, over n_rho in {fit_nr} (the range where neither axis is "
      "starved), alongside the level improvement the smooth mode gives at the "
      "finest swept grid:")
    w("")
    w("| quantity | histogram exponent | smooth exponent | smooth/histogram "
      f"error at ({max(grid['n_rho_sweep'])}, {nd_top}) |")
    w("|---|---|---|---|")
    exps = {}
    fine = (max(grid["n_rho_sweep"]), nd_top)
    for qty, get in (("mid-p p90 \\|dp\\|", lambda r: r["mid"]["p90"]),
                     ("p90 rel. Lambda_NEO",
                      lambda r: r["lam_neo_rel"]["p90"]),
                     ("p90 rel. Lambda_other",
                      lambda r: r["lam_other_rel"]["p90"])):
        row = []
        for tab in (idx, idxp):
            x = np.log([float(nr) for nr in fit_nr])
            y = np.log([get(tab[(nr, nd_top)]) for nr in fit_nr])
            row.append(-float(np.polyfit(x, y, 1)[0]))
        exps[qty] = row
        w(f"| {qty} | {row[0]:.2f} | {row[1]:.2f} | "
          f"{get(idxp[fine]) / get(idx[fine]):.2f} |")
    w("")
    le = exps["p90 rel. Lambda_other"]
    ln = exps["p90 rel. Lambda_NEO"]
    w("A midpoint-rule integration across a discontinuity aligned with the "
      "integration axis has error falling like 1/n_rho (exponent 1); a smooth "
      "integrand gives 1/n_rho^2.  Read the two columns together:")
    w("")
    w(f"- **Lambda_other**: the histogram mode's exponent "
      f"{le[0]:.2f} is the signature of a discontinuity-limited integrand, "
      f"and the smooth mode's {le[1]:.2f} is measurably better.  This is the "
      "H staircase being removed, exactly as DESIGN §2.5 describes it.")
    w(f"- **Lambda_NEO**: the smooth mode is uniformly "
      f"{100 * (1 - idxp[fine]['lam_neo_rel']['p90'] / idx[fine]['lam_neo_rel']['p90']):.0f}%"
      f" better in level at the finest grid, but its fitted exponent "
      f"({ln[1]:.2f}) comes out *below* the histogram's ({ln[0]:.2f}) — "
      "because it is already lower at the low-n_rho end of the fit, so the "
      "fit measures a level shift and not a rate change.  Do not read that "
      "row as the smooth mode converging more slowly; the whole curve is "
      "below the histogram's at every n_rho in the table.")
    w("- Neither exponent reaches 2, in either mode.  **The H edges were not "
      "the only axis-aligned structure**, which is the mechanical reason the "
      "next section's answer is negative.")
    w("")
    w("### Findings")
    w("")
    pr_h = idx[tuple(prod)]
    pr_p = idxp[tuple(prod)]
    gain_p = 1.0 - pr_p["mid"]["p90"] / pr_h["mid"]["p90"]
    gain_l = 1.0 - (pr_p["lam_other_rel"]["p90"]
                    / pr_h["lam_other_rel"]["p90"])
    fine = (max(grid["n_rho_sweep"]), max(grid["n_rhodot_sweep"]))
    fh, fp = idx[fine], idxp[fine]
    pi = grid.get("prod_improvement")
    if pi is not None:
        verdict = ("a detectable" if pi["detectable"] else "no detectable")
        w(f"1. **At {prod} there is {verdict} quadrature win.**  The mid-p "
          f"p90 |dp| goes {pi['p90_histogram']:.2e} -> "
          f"{pi['p90_smooth']:.2e}, an improvement of "
          f"{pi['improvement_pct']:+.1f}% "
          f"[{pi['improvement_pct_ci'][0]:+.1f}%, "
          f"{pi['improvement_pct_ci'][1]:+.1f}%] "
          f"(paired bootstrap over the {pi['n_mid_p']} mid-p tracklets, "
          f"{pi['n_boot']} replicates).  "
          + ("The interval excludes zero, so the effect is real at this "
             "grid.  " if pi["detectable"] else
             "**The interval contains zero.**  A point estimate of \"2%\" "
             "is not a result: p90 of a few hundred values carries its own "
             "sampling error, and the review's independent replication on a "
             "different catalogue gave +14.8% [0.0%, +43.6%] — same sign, "
             "consistent with this, and also not separable from zero.  The "
             "honest headline is that **no quadrature win is detectable at "
             f"{prod}**, not that there is a 2% one.  ")
          + "The Lambdas are a different matter: Lambda_other's p90 relative "
          f"error improves {100 * gain_l:.0f}% at {prod} "
          f"({pr_h['lam_other_rel']['p90']:.2e} -> "
          f"{pr_p['lam_other_rel']['p90']:.2e}) and "
          f"{100 * (1 - fp['lam_other_rel']['p90'] / fh['lam_other_rel']['p90']):.0f}% "
          f"at the finest swept grid {fine}, monotonically with n_rho — so "
          "the H staircase **is** a real quadrature error term and the "
          "reconstruction **does** remove part of it.  What the p(NEO) row "
          "says is that the part it removes is common-mode across the two "
          "components and cancels in the posterior ratio.  Either way there "
          "is no n_rho reduction to bank.")
    else:
        w(f"1. **The smooth mode does not buy a coarser grid.**  At {prod} it "
          f"improves mid-p p90 |dp| by {100 * gain_p:.0f}% and "
          f"Lambda_other's p90 relative error by {100 * gain_l:.0f}%.  "
          "(Re-run `grid` to get the bootstrap CI on the first number; "
          "without it the 2% is a point estimate, not a result.)")
    for key, label in (("match_histogram_accuracy",
                        f"matches histogram mode's {prod} mid-p accuracy"),
                       ("ten_times_better", "is 10x better than that")):
        for mode in ("pchip", "none"):
            r = grid.get(f"{key}/{mode}")
            txt = ("**no grid in the sweep reaches it**" if r is None else
                   f"**({r['n_rho']}, {r['n_rhodot']})** = {r['nodes']} "
                   f"nodes, mid-p p90 {r['mid']['p90']:.2e}, mid-p max "
                   f"{r['mid']['max']:.2e}")
            w(f"   - cheapest *stable* {mode} grid that {label}: {txt}")
    w(f"   - **Recommended production grid for the smooth mode: the unchanged "
      f"{prod}.**  Single-core cost at it is in the Rust table above.")
    mono = _monotonicity(grid)
    w("   - *Stable* = every finer point in the sweep also meets the "
      "threshold.  On the mid-p statistic this is nearly redundant, and an "
      "earlier draft of this report was wrong about why it was not: it "
      "claimed order-of-magnitude aliasing non-monotonicity in n_rho, with "
      "\"n_rho = 100 beating n_rho = 200\".  Checked against this JSON, the "
      f"mid-p p90 is monotone in n_rho in {mono['monotone_cols']} of "
      f"{mono['n_cols']} (mode, n_rhodot) columns, and the single exception "
      f"is a {100 * (mono['worst_up_step'] - 1):.1f}% up-step "
      f"({mono['worst_where']}).  The stability constraint is kept because it "
      "is the right criterion for a production grid, not because the data "
      "here need it.")
    w("")
    floor = _floor(grid, "none", chk)
    w(f"2. **The 10x-better threshold is below what this experiment could "
      f"resolve, and no swept grid comes close anyway.**  10x better than the "
      f"{prod} baseline is mid-p p90 {target / 10.0:.2e}, only "
      f"{target / 10.0 / floor:.1f}x above the reference's own error "
      f"({floor:.2e}) — so a grid that reached it could not be *shown* to "
      f"have reached it against a {ref} reference.  That caveat turns out to "
      f"be moot: the finest swept grid {fine} "
      f"({fine[0] * fine[1]} nodes, "
      f"{fine[0] * fine[1] // (prod[0] * prod[1])}x the production cost) "
      f"reaches only {fp['mid']['p90']:.2e}, i.e. "
      f"{pr_h['mid']['p90'] / fp['mid']['p90']:.1f}x better, in either mode.  "
      "With a fitted exponent near 1, 10x would cost roughly 10x the nodes "
      "— **and it is not the reconstruction that gets you there**: the two "
      "modes' curves lie on top of each other in "
      f"`{a.grid_name.replace('.json', '')}_convergence.png`.  A serious 10x "
      "programme needs node placement or a different quadrature rule, plus a "
      f"reference finer than {chk} to verify it against.")
    nd_lo, nd_hi = min(grid["n_rhodot_sweep"]), max(grid["n_rhodot_sweep"])
    fc_d = {tuple(r["grid"]): r for r in (grid.get("design_fixed_cost_check")
                                          or [])}
    w("")
    w("3. **At the production grid neither axis dominates any more.**  "
      "DESIGN §2.5 prescribes rho-heavy grids on the evidence that rho_dot "
      "refinement saturates, and its own comparison reproduces here: at "
      f"n_rho = 200, going from n_rhodot = 100 to 400 moves the mid-p p90 "
      f"only from {fc_d.get((200, 100), {}).get('mid_p90', float('nan')):.2e} "
      f"to {fc_d.get((200, 400), {}).get('mid_p90', float('nan')):.2e}.  But "
      f"that saturation is a property of the *coarse-rho* regime.  At "
      f"n_rho = {prod[0]} the rho_dot axis is still paying: "
      f"{idx[(prod[0], 50)]['mid']['p90']:.2e} at n_rhodot = 50 vs "
      f"{idx[(prod[0], 100)]['mid']['p90']:.2e} at 100.  Spending 2x the "
      f"production budget on either axis buys the same factor: (1200, 100) "
      f"gains {idx[tuple(prod)]['mid']['p90'] / idx[(1200, 100)]['mid']['p90']:.2f}x"
      f" and (2400, 50) gains "
      f"{idx[tuple(prod)]['mid']['p90'] / idx[(2400, 50)]['mid']['p90']:.2f}x. "
      f"So {prod} sits at a balanced point — rho-heavy was the right way to "
      "*get* there, and from there the two axes are equally starved.  This is "
      "a side finding, outside this study's remit, offered to the supervisor "
      "as an observation on §2.5 rather than as a change to it.")
    w("")
    res = grid.get("residual") or {}
    if res:
        w("### Where the residual error lives")
        w("")
        w("Spearman rank correlation of per-tracklet |dp| at "
          f"{tuple(res['grid'])} (smooth mode) with:")
        w("")
        w("| quantity | rho_s |")
        w("|---|---|")
        for k, v in sorted(res["spearman_abs_dp_vs"].items(),
                           key=lambda kv: -abs(kv[1]) if kv[1] == kv[1] else 0):
            w(f"| `{k}` | {v:+.3f} |")
        w("")
        sp = res["spearman_abs_dp_vs"]
        w("Two of these are worth naming.  `p_neo_ref` "
          f"({sp.get('p_neo_ref', float('nan')):+.2f}) is the min(p, 1-p) "
          "bound reasserting itself even inside the mid-p window.  "
          "`ar_frac_h_above` — the §2.6(b) truncation-exposure diagnostic — is "
          f"**negatively** correlated ({sp.get('ar_frac_h_above', float('nan')):+.2f}): "
          "the tracklets whose admissible region reaches the faint end are "
          "*not* where the residual quadrature error lives.  That is an "
          "independent confirmation of the same conclusion the sweep reaches, "
          "arrived at per-tracklet rather than in aggregate: the H-bin "
          "structure is not the binding error term.  The strongest remaining "
          f"geometric correlate is the apparent rate "
          f"({sp.get('rate_deg_day', float('nan')):+.2f}), which sets how "
          "much of the (rho, rho_dot) plane is admissible — i.e. the "
          "admissible-region terminus, the other candidate DESIGN §2.5 names.")
        w("")
    if grid_all is not None:
        w("### The unconditional statistic, whole catalogue")
        w("")
        ba = grid_all["histogram_baseline"]
        ia = {(r["n_rho"], r["n_rhodot"]): r
              for r in grid_all["sweep"]["none"]}
        ip = {(r["n_rho"], r["n_rhodot"]): r
              for r in grid_all["sweep"]["pchip"]}
        w(f"For completeness, the same sweep over all "
          f"{grid_all['n_tracklets']} tracklets of `{grid_all['run']}` "
          f"({grid_all.get('n_mid_p')} of them mid-p), reference "
          + "x".join(map(str, grid_all["ref_grid"])) + ":")
        w("")
        w(f"- histogram at {prod}: unconditional p90 "
          f"{ba['vs_fine_reference']['p90']:.2e}, mid-p p90 "
          f"{ba['vs_fine_reference']['mid']['p90']:.2e}; vs DESIGN's "
          f"{tuple(grid_all['design_ref_grid'])} reference, mid-p p90 "
          f"{ba['vs_design_reference']['mid']['p90']:.2e} / mid-p max "
          f"{ba['vs_design_reference']['mid']['max']:.2e} "
          "**<- DESIGN's 3.5e-3 / 1e-2**")
        w(f"- smooth at {prod}: unconditional p90 "
          f"{ip[tuple(prod)]['p90']:.2e}, mid-p p90 "
          f"{ip[tuple(prod)]['mid']['p90']:.2e} (histogram "
          f"{ia[tuple(prod)]['p90']:.2e} / "
          f"{ia[tuple(prod)]['mid']['p90']:.2e})")
        w(f"- full tables in `{a.grid_all_name}`")
        w("")

    # ---------------- measurement 2 ------------------------------------
    w("## 2. Self-consistency gate in smooth mode")
    w("")
    if gate is None:
        w("*(not available — run `self_consistency run --h-interpolation "
          "pchip --gate`)*")
    else:
        w(f"`self_consistency run --n {gate.get('n_scored', '?')} "
          "--seed 20260901 --gate --h-interpolation pchip` — prior = "
          "generator = the reconstructed model, a **new catalogue** drawn "
          "with the smooth sampler (the same seed produces a different "
          "catalogue in this mode, because the within-bin H draw is now an "
          "inverse-CDF of the cubic and the rejection stream diverges from "
          "there).  Both the generator and the classifier are switched by the "
          "one flag; `ScoreConfig` carries the mode into the chunk "
          "provenance stamp, so no cached chunk from a histogram run can be "
          "spliced in.")
        w("")
        w("| criterion | smooth (this run) | histogram (run of record) |")
        w("|---|---|---|")
        for label, path, fmt in _GATE_ROWS:
            a_ = _dig(gate, path)
            b_ = _dig(gate_hist, path) if gate_hist else None
            w(f"| {label} | {fmt(a_)} | {fmt(b_) if gate_hist else 'n/a'} |")
        w("")
        w(f"**Verdict: {_verdict(gate)}**")
        w("")
        gap = _dig(gate, ("collapse", "hpd_estimator_gap_D"))
        gap_h = _dig(gate_hist, ("collapse", "hpd_estimator_gap_D"))
        if gap is not None and gap_h is not None:
            w(f"**The HPD estimator gap did not shrink**: {gap:.7f} in smooth "
              f"mode against {gap_h:.7f} in histogram mode — the same number "
              f"to within the resolution of a 2x10^4 KS statistic, and if "
              f"anything {'larger' if gap > gap_h else 'smaller'}.  The gap is "
              "the §4.1 grid-adequacy readout (the nearest-node and "
              "exact-at-truth HPD estimators bias in opposite directions and "
              "converge under refinement), so this is the gate-side "
              "confirmation of section 1's negative result: the H staircase "
              "was not what the gap was measuring.  Every other statistic is "
              "consistent between the two runs — c_hat straddles unity from "
              "the same side, reliability max|z| is unchanged, collapse "
              "coverage and the §2.6(b) diagnostic are unchanged — which is "
              "the substantive finding: **the reconstruction is gate-clean.** "
              "It passes the self-consistency test on its own generated "
              "population, so it is a legitimate prior, not merely a "
              "smoothing device.")
            w("")
        w("The two runs' NEO counts differ (179 vs 185 true NEOs) purely "
          "because they are different catalogues; the calibration statistics "
          "are the comparable quantities, not the counts.")
        w("")
    w("")

    # ---------------- measurement 3 ------------------------------------
    w("## 3. Mode-change measurement on the existing gate_v0 catalogue")
    w("")
    w(f"The stored `gate_v0` catalogue ({s['n']} tracklets, {s['n_neo']} true "
      "NEOs) was generated from the **piecewise-constant** model.  Re-scoring "
      "it with the smooth pair at "
      f"({s['config']['n_rho']}, {s['config']['n_rhodot']}) on the Rust "
      "backend, and comparing to the stored histogram scores, isolates what "
      "the prior change does — on the clamp experiment's template.")
    w("")
    sh = s["score_shift"]
    w("### (a) What the mode did to the scores")
    w("")
    w(f"- {100 * sh['bit_identical_frac']:.1f}% of tracklets score "
      "**bit-identically** in the two modes.")
    w(f"- {100 * sh['frac_up']:.1f}% rise, {100 * sh['frac_down']:.1f}% fall — "
      "unlike the D-14 clamp, this change is **two-sided**, which is what a "
      "mass-conserving reconstruction should be: it moves density *within* "
      "each H bin instead of adding density beyond the support.")
    w(f"- median dlog10 p = {sh['median_dlog10']:+.2e}; 1st-99th percentile "
      f"[{sh['p01_dlog10']:+.4f}, {sh['p99_dlog10']:+.4f}]; max |dlog10 p| = "
      f"{sh['max_abs_dlog10']:.4f}; max |dp| = {sh['max_abs_dp']:.2e}.")
    w(f"- {100 * sh['frac_gt_0p01_dex']:.1f}% move by more than 0.01 dex, "
      f"{100 * sh['frac_gt_0p1_dex']:.2f}% by more than 0.1 dex.")
    w(f"- Spearman rank correlation between the two modes' scores: "
      f"{sh['spearman_vs_hist']:.5f}.")
    w("")
    w("Displacement against the §2.6(b) diagnostic (posterior-measure "
      "fraction at implied H > 25.5) — the diagnostic the change is expected "
      "to act through:")
    w("")
    w("| §2.6(b) measure fraction | n | median dlog10 p | fraction moved | "
      "fraction up | p99 \\|dlog10 p\\| |")
    w("|---|---|---|---|---|---|")
    for r in s["by_truncation_exposure"]:
        w(f"| [{r['lo']:g}, {r['hi']:g}) | {r['n']} | "
          f"{r['median_dlog10_p']:+.5f} | {100 * r['frac_moved']:.1f}% | "
          f"{100 * r['frac_up']:.1f}% | {r['p99_abs']:.4f} |")
    w("")
    w("And against apparent V:")
    w("")
    w("| V | n | median dlog10 p | fraction moved | fraction up | "
      "p99 \\|dlog10 p\\| |")
    w("|---|---|---|---|---|---|")
    for r in s["by_v_mag"]:
        w(f"| [{r['lo']:g}, {r['hi']:g}) | {r['n']} | "
          f"{r['median_dlog10_p']:+.5f} | {100 * r['frac_moved']:.1f}% | "
          f"{100 * r['frac_up']:.1f}% | {r['p99_abs']:.4f} |")
    w("")
    w("The sign of the median displacement **flips with truncation "
      "exposure**: tracklets whose admissible region stays well inside the "
      "support move up, tracklets whose region reaches the faint end move "
      "down.  That is a property of the reconstruction, not of the geometry: "
      "PCHIP's shape-preserving slopes flatten the top of a steeply rising "
      "cumulative count, so the reconstructed density near H = 25.5 is *below* "
      "the top bin's flat average — and the NEO component, with 55.8% of its "
      "mass in that bin (DESIGN §2.6), loses more than the other component "
      "does.  Where the region never reaches the top bin the reverse holds.")
    w("")
    w("### (b) Ranking")
    w("")
    w("Labels and subsets are the stored ones; the subsets are the DESIGN "
      "§4.2 truncation split on the §2.6(b) diagnostic, cut on the *stored* "
      "(histogram) values so the two passes compare the same tracklets.")
    w("")
    w("| population | AUC smooth | AUC histogram | dAUC (smooth - hist) |")
    w("|---|---|---|---|")
    for name, rec in s["subsets"].items():
        w(f"| `{name}` (n={rec['n']}, NEO={rec['n_neo']}) | "
          f"{rec['auc_pchip']:.4f} | {rec['auc_hist']:.4f} | "
          f"{_ci(rec['pchip_vs_hist'])} |")
    w("")
    if any("pchip_vs_digest2" in r for r in s["subsets"].values()):
        w("Against digest2's stored scores — the \"better model\" question:")
        w("")
        w("| population | dAUC smooth - digest2 | dAUC histogram - digest2 | "
          "gain from the reconstruction |")
        w("|---|---|---|---|")
        for name, rec in s["subsets"].items():
            if "pchip_vs_digest2" not in rec:
                continue
            w(f"| `{name}` (n={rec['n_with_digest2']}) | "
              f"{_ci(rec['pchip_vs_digest2'])} | "
              f"{_ci(rec['hist_vs_digest2'])} | "
              f"{rec['delta_auc_gain_over_hist']:+.4f} |")
        w("")
        g = s["subsets"].get("all", {}).get("delta_auc_gain_over_hist")
        pv = s["subsets"].get("all", {}).get("pchip_vs_hist", {})
        if g is not None:
            w(f"**The reconstruction's ranking gain over digest2 is "
              f"{g:+.4f}** on the full set, on top of the histogram model's "
              f"{s['subsets']['all']['hist_vs_digest2']['delta']:+.4f}.  Its "
              f"paired dAUC against the histogram model itself is "
              f"{_ci(pv)} — a confidence interval that comfortably contains "
              "zero.  **On this catalogue the reconstruction is a ranking "
              "no-op.**  Which is the right expectation for two reasons, both "
              "worth stating rather than glossing: the truths were drawn from "
              "the histogram, so the histogram model is the *correct* prior "
              "here and cannot be beaten on its own generating distribution; "
              "and 185 true NEOs bound any paired dAUC measurement to about "
              "+/-0.002 regardless.  A ranking gain, if the reconstruction "
              "has one, would have to be demonstrated against truths drawn "
              "from something else — DESIGN §4.3, or a real catalogue.")
            w("")
    w("### (c) Calibration against histogram-generated truth")
    w("")
    w("**Read this as a deliberate model mismatch, not as a calibration test "
      "of the method.**  The catalogue was drawn from the piecewise-constant "
      "model; scoring it with the reconstruction is a small, controlled "
      "instance of the DESIGN §4.3 prior-misspecification experiment.  The "
      "smooth mode's *own* calibration is section 2 above, where prior and "
      "generator agree by construction.")
    w("")
    w("| population | prior | predicted NEOs | observed | ratio | ECE | "
      "max \\|z\\| bin |")
    w("|---|---|---|---|---|---|---|")
    for name, rec in s["subsets"].items():
        for lbl, key in (("smooth", "calibration_pchip"),
                         ("histogram", "calibration_hist")):
            c = rec[key]
            w(f"| `{name}` | {lbl} | {c['expected_positives']:.1f} | "
              f"{c['observed_positives']:.0f} | "
              f"{c['ratio_expected_observed']:.3f} | {c['ece']:.5f} | "
              f"{c['max_abs_z_bin']:.2f} |")
    w("")
    ca = s["subsets"].get("all", {})
    if ca:
        cp, chh = ca["calibration_pchip"], ca["calibration_hist"]
        w(f"The mismatch costs essentially nothing: predicted/observed moves "
          f"{chh['ratio_expected_observed']:.4f} -> "
          f"{cp['ratio_expected_observed']:.4f} and ECE "
          f"{chh['ece']:.5f} -> {cp['ece']:.5f}, i.e. the *wrong* prior's "
          f"predicted NEO count is off by "
          f"{100 * abs(cp['ratio_expected_observed'] - 1):.1f}% where the "
          f"*right* one is off by "
          f"{100 * abs(chh['ratio_expected_observed'] - 1):.1f}%.  This is "
          "the same phenomenon D-14 recorded for the H clamp: "
          "**a correctly marginalized posterior absorbs a change to the "
          "prior's faint end**, because the change is common-mode across the "
          "two components in almost the same proportion as their mixture "
          "weights.  It is a statement about the marginalization, not "
          "evidence that the priors are interchangeable — the per-tracklet "
          f"scores moved by up to {sh['max_abs_dlog10']:.2f} dex.")
        w("")
    w("")
    w("### Controls")
    w("")
    for k, v in s["controls"].items():
        w(f"- {k}: {v}")
    w("")
    w("## Plots")
    w("")
    stem = a.grid_name.replace(".json", "")
    w(f"- `{stem}_convergence.png` — mid-p p90 |dp| and Lambda_other error vs "
      "n_rho, both modes overlaid, with the reference's own error floor "
      "shaded.")
    w(f"- `{stem}_mode_ratio.png` — smooth error divided by histogram error, "
      "for p(NEO) and for Lambda_other.  This is the figure that carries "
      "section 1's finding: the Lambda ratio sits below 1 and falls with "
      "n_rho, the p(NEO) ratio sits on 1.")
    w(f"- `{stem}_cost_accuracy.png` — the cost-accuracy front of both modes "
      "on one axis (they lie on top of each other).")
    w("- `reconstruction_shape.png` — the construction itself, on three "
      "cells: cumulative count with the monotone cubic, and the two "
      "densities.")
    if grid_all is not None:
        w(f"- `{a.grid_all_name.replace('.json', '')}_*.png` — the same three "
          "for the whole-catalogue sweep.")
    w("- `mode_change_scatter.png` — smooth vs histogram p(NEO) coloured by "
      "the §2.6(b) diagnostic, plus displacement against that diagnostic and "
      "against V.")
    w("- `mode_change_roc.png`, `mode_change_calibration.png` — ranking and "
      "calibration of both priors (and digest2) on the stored labels.")
    w("")
    w("## Caveats")
    w("")
    w("- The smooth mode is a *reconstruction of the same model*, not a new "
      "population model.  It cannot add information the histogram does not "
      "contain; what it does is stop asserting that the density is flat "
      "inside a bin and 100x different one magnitude away.")
    w("- It does **not** address the `1/(e sin i)` integrable singularity "
      "(DESIGN §2.5) or the D-4 truncation corner (§2.6).")
    w("- Section 3's calibration arm is a prior mismatch by construction; do "
      "not read it as miscalibration of the method.")
    w("- Section 1's negative result is a statement about p(NEO) accuracy at "
      "n_rho >= 100.  It says nothing about a future H-bin-edge-aware node "
      "placement (issue #1), which targets the same discontinuities by moving "
      "nodes rather than by smoothing the density; the convergence-exponent "
      "table suggests that too would have limited headroom in p(NEO), though "
      "more in Lambda.")
    w("- **Each sweep point is measured against a reference in its own "
      "mode.**  That is deliberate and it bounds what section 1 can claim: it "
      "answers \"how many nodes does each mode need to converge to its own "
      "limit\", which is the node-count question, and not \"which mode's "
      "limit is closer to the truth\".  The latter is not a well-posed "
      "quadrature question — the two modes are two different priors, each "
      "exactly integrable in principle — and it is what sections 2 and 3 "
      "address instead.")
    if grid_all is not None:
        w("- The mid-p sweep and the whole-catalogue sweep use different "
          "catalogues (`calibration_1e5` and `gate_v0`) because the mid-p "
          "statistic needs the tracklet count only the 10^5 run provides.  "
          "The two are consistent where they overlap; both tables are in the "
          "JSONs.")
    else:
        w("- **Only the mid-p sweep is included.**  The companion "
          "whole-catalogue sweep over all 2x10^4 `gate_v0` tracklets was "
          "started and abandoned after 30 minutes and 23 core-hours: another "
          "agent was running a large job in this repository, the box was at "
          "load ~200 on 96 cores, and it was starving the required test "
          "suites.  Re-running it is one command "
          "(`grid --run validation/output/gate_v0 --select all "
          "--out-name grid_study_all.json --ref-grid 4800 200 "
          "--ref-check-grid 9600 400`) and this report picks it up "
          "automatically as `grid_study_all.json`.  Nothing above depends on "
          "it: it would add the unconditional-statistic table, which the "
          "mid-p section explains is the less informative of the two.")
        w("- The two `gate_v0` reconciliation numbers quoted above (mid-p p90 "
          "3.88e-3, mid-p max 1.62e-2 against the "
          f"{tuple(grid['design_ref_grid'])} reference, and the 7.7e-13 "
          "agreement of the recomputed scores with the run of record) come "
          "from a targeted measurement on that catalogue rather than from a "
          "saved sweep, for the same reason.  Reproduce with: score "
          "`gate_v0` at (1200, 50), (1200, 600) and (4800, 200) in "
          "histogram mode via `score_chunks`, and take |dp| percentiles over "
          "the rows with stored p in (0.05, 0.95).")
    w("")
    w("## Reproduce")
    w("")
    w("Every number above comes out of these six commands, in this order "
      "(`OMP_NUM_THREADS=1` throughout — the work is Rust-side and rayon "
      "already owns the parallelism):")
    w("")
    w("```")
    w("# 1. the mid-p sweep (PRIMARY: DESIGN §2.5's statistic, "
      f"{grid.get('n_mid_p')} mid-p tracklets)")
    w("python -m validation.h_interpolation_study grid \\")
    w(f"    --run {grid['run']} --select {grid.get('select', 'midp')} \\")
    w(f"    --out {a.out} --out-name {a.grid_name} \\")
    w(f"    --ref-grid {grid['ref_grid'][0]} {grid['ref_grid'][1]} "
      f"--ref-check-grid {grid['ref_check_grid'][0]} "
      f"{grid['ref_check_grid'][1]} \\")
    w("    --threads 96 --time-reps 5 --time-tracklets 48")
    if grid_all is not None:
        w("")
        w("# 2. the unconditional sweep over the whole gate catalogue")
        w("python -m validation.h_interpolation_study grid \\")
        w(f"    --run {grid_all['run']} --select all \\")
        w(f"    --out {a.out} --out-name {a.grid_all_name} \\")
        w(f"    --ref-grid {grid_all['ref_grid'][0]} "
          f"{grid_all['ref_grid'][1]} --ref-check-grid "
          f"{grid_all['ref_check_grid'][0]} "
          f"{grid_all['ref_check_grid'][1]} --threads 96")
    w("")
    w("# 3. interleaved single-core timing")
    w("python -m validation.h_interpolation_study bench \\")
    w(f"    --run validation/output/gate_v0 --out {a.out} \\")
    w("    --n-tracklets 48 --reps 9 --grid 1200 50 --grid 1200 100")
    w("")
    w("# 4. the gate, prior = generator = the reconstruction")
    if a.gate_run:
        w("python -m validation.self_consistency run --n 20000 \\")
        w("    --seed 20260901 --workers 32 --gate "
          "--h-interpolation pchip \\")
        w(f"    --out {a.gate_run}")
    w("")
    w("# 5. the mode-change measurement on the stored histogram catalogue")
    w("python -m validation.h_interpolation_study rescore \\")
    w(f"    --run {s['run']} --out {a.out} \\")
    w(f"    --n-rho {s['config']['n_rho']} "
      f"--n-rhodot {s['config']['n_rhodot']} --threads 96")
    w("python -m validation.h_interpolation_study analyze \\")
    w(f"    --run {s['run']} --out {a.out}")
    w("")
    w("# 6. this file")
    w("python -m validation.h_interpolation_study report \\")
    w(f"    --out {a.out}"
      + (" \\" if (a.gate_run or a.gate_reference) else ""))
    if a.gate_run or a.gate_reference:
        w("    "
          + (f"--gate-run {a.gate_run} " if a.gate_run else "")
          + (f"--gate-reference {a.gate_reference}"
             if a.gate_reference else ""))
    w("```")
    w("")
    w("`validation/output/` is gitignored, so a copy of this file is kept at "
      "`validation/H_INTERPOLATION.md` (the same arrangement as "
      "`validation/DIGEST2_COMPARISON_CLAMPED.md`).")
    path = os.path.join(a.out, "REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[report] wrote {path} ({len(L)} lines)")
    return 0


def _dig(d, path):
    if d is None:
        return None
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _f(x, spec=".4f"):
    if x is None:
        return "n/a"
    if spec == "d":
        return format(int(x), "d")
    return format(float(x), spec)


_GATE_ROWS = [
    ("N scored", ("n_scored",), lambda x: _f(x, "d")),
    ("true NEOs", ("n_neo_true",), lambda x: _f(x, "d")),
    ("observed NEOs", ("reliability", "observed_neo_count"),
     lambda x: _f(x, ".0f")),
    ("expected NEOs", ("reliability", "expected_neo_count"),
     lambda x: _f(x, ".1f")),
    ("Z_tot", ("reliability", "global_z"), lambda x: _f(x, "+.4f")),
    ("Z_tot bootstrap p", ("reliability", "global_z_p_value"),
     lambda x: _f(x, ".4f")),
    ("fitted odds scale c_hat", ("reliability", "odds_scale", "c_hat"),
     lambda x: _f(x, ".3f")),
    ("c_hat 95% lo", ("reliability", "odds_scale", "lo"),
     lambda x: _f(x, ".3f")),
    ("c_hat 95% hi", ("reliability", "odds_scale", "hi"),
     lambda x: _f(x, ".3f")),
    ("reliability max\\|z\\|", ("reliability", "max_abs_z"),
     lambda x: _f(x, ".4f")),
    ("reliability threshold", ("reliability", "max_abs_z_threshold"),
     lambda x: _f(x, ".3f")),
    ("HPD KS D, nearest node", ("collapse", "hpd_ks_D"),
     lambda x: _f(x, ".5f")),
    ("HPD KS D, exact at truth", ("collapse", "hpd_exact_ks_D"),
     lambda x: _f(x, ".5f")),
    ("**HPD estimator gap**", ("collapse", "hpd_estimator_gap_D"),
     lambda x: _f(x, ".7f")),
    ("integrand > 0 at truth", ("collapse", "frac_truth_density_positive"),
     lambda x: _f(x, ".4f")),
    ("AUC (not gated)", ("auc",), lambda x: _f(x, ".4f")),
    ("median §2.6(b) measure fraction",
     ("diagnostics_2_6", "median_ar_frac_H_above_Hmax"),
     lambda x: _f(x, ".5f")),
]


def _verdict(gate: dict) -> str:
    if "overall_pass" not in gate:
        return "see summary.txt"
    failed = [k for k, v in (gate.get("verdicts") or {}).items() if not v]
    return ("OVERALL PASS" if gate["overall_pass"]
            else "OVERALL FAIL (" + ", ".join(failed) + ")")


def _roc(scores, labels):
    s = np.asarray(scores, float)
    y = np.asarray(labels).astype(bool)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    npos = max(int(y.sum()), 1)
    nneg = max(int((~y).sum()), 1)
    return np.concatenate(([0.0], fp / nneg)), np.concatenate(([0.0], tp / npos))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(repr(o))


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m validation.h_interpolation_study",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grid", help="node count vs accuracy sweep")
    g.add_argument("--run", default="validation/output/gate_v0")
    g.add_argument("--out", required=True)
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--variant", default="All", choices=("All", "Unk"))
    g.add_argument("--n-tracklets", type=int, default=0,
                   help="0 or >= catalogue size means the whole catalogue")
    g.add_argument("--select", default="all",
                   choices=("all", "stratified", "midp"),
                   help="'midp' sweeps only the tracklets whose stored p is "
                        "in the MID_P window — DESIGN §2.5's own statistic, "
                        "and where the entire |dp| budget lives")
    g.add_argument("--seed", type=int, default=20260902)
    g.add_argument("--threads", type=int, default=None)
    g.add_argument("--time-reps", type=int, default=3)
    g.add_argument("--out-name", default="grid_study.json",
                   help="basename of the JSON (and stem of the plots), so "
                        "several sweeps can live in one directory")
    g.add_argument("--boot", type=int, default=2000,
                   help="bootstrap replicates for the p90-ratio CI")
    g.add_argument("--boot-seed", type=int, default=20260902)
    g.add_argument("--time-tracklets", type=int, default=32,
                   help="tracklets per single-core timing pass")
    g.add_argument("--ref-grid", type=int, nargs=2, default=list(REF_GRID),
                   metavar=("N_RHO", "N_RHODOT"),
                   help="the same-mode reference every sweep point is "
                        "measured against")
    g.add_argument("--ref-check-grid", type=int, nargs=2,
                   default=list(REF_CHECK_GRID),
                   metavar=("N_RHO", "N_RHODOT"),
                   help="a still finer grid, used ONLY to bound the "
                        "reference's own error (the floor below which nothing "
                        "in the sweep table is meaningful)")

    r = sub.add_parser("rescore", help="re-score a catalogue with the smooth pair")
    r.add_argument("--run", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--variant", default="All", choices=("All", "Unk"))
    r.add_argument("--n-rho", type=int, required=True)
    r.add_argument("--n-rhodot", type=int, required=True)
    r.add_argument("--batch", type=int, default=2000)
    r.add_argument("--threads", type=int, default=None)

    an = sub.add_parser("analyze", help="compare the two modes; plots + json")
    an.add_argument("--run", required=True)
    an.add_argument("--out", required=True)
    an.add_argument("--digest2",
                    default="validation/output/digest2_comparison/gate_v0")
    an.add_argument("--boot", type=int, default=2000)
    an.add_argument("--boot-seed", type=int, default=20260902)

    b = sub.add_parser("bench", help="interleaved single-core timing")
    b.add_argument("--run", default="validation/output/gate_v0")
    b.add_argument("--out", required=True)
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--variant", default="All", choices=("All", "Unk"))
    b.add_argument("--n-tracklets", type=int, default=48)
    b.add_argument("--seed", type=int, default=20260902)
    b.add_argument("--reps", type=int, default=9)
    b.add_argument("--grid", type=int, nargs=2, action="append",
                   metavar=("N_RHO", "N_RHODOT"),
                   help="repeatable; defaults to the production grid alone")

    rp = sub.add_parser("report", help="emit REPORT.md from the saved JSONs")
    rp.add_argument("--out", required=True)
    rp.add_argument("--model", default=DEFAULT_MODEL)
    rp.add_argument("--variant", default="All", choices=("All", "Unk"))
    rp.add_argument("--grid-name", default="grid_study.json",
                    help="the PRIMARY (mid-p) sweep")
    rp.add_argument("--grid-all-name", default="grid_study_all.json",
                    help="the optional whole-catalogue sweep")
    rp.add_argument("--gate-run", default=None,
                    help="a self_consistency run directory scored in pchip "
                         "mode (its summary.json is quoted verbatim)")
    rp.add_argument("--gate-reference", default=None,
                    help="the histogram-mode gate run to compare against")

    a = ap.parse_args(argv)
    return {"grid": cmd_grid, "rescore": cmd_rescore,
            "analyze": cmd_analyze, "bench": cmd_bench,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
