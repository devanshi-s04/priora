#!/usr/bin/env python3
"""Golden A/B: score the whole gate_v0 catalogue on the Rust backend.

Three comparisons, all on ``validation/output/gate_v0``:

1. **Golden** — the Rust backend's ``p_neo``/``log_lr``/``Lambda``/``status``
   (D-19; the stored reference predates the field, so its status is
   reconstructed from the recorded (n_nodes, Lambda) triple) against the
   *stored* ``scores.npz``, which was produced by the Python reference at the
   grid and model recorded in ``score_config.json``.  This is the acceptance
   check from issue #1: ``max |dp_neo| <= 1e-6`` and ``Lambda`` relative error
   ``<= 1e-5``.
2. **Node counts** — ``n_nodes`` must agree exactly with the stored run
   (identical grids, identical admissible-region truncation).
3. **Masks** — the stored file does not carry the mask counters, so a *fresh*
   Python-reference run (parallelized with ``multiprocessing``) supplies them.
   ``n_masked_nonfinite`` (the physical unbound-node mask) must agree exactly.
   ``n_masked_det`` is reported rather than asserted, because it is the one
   place the two Jacobian routes could in principle diverge: the Python path
   masks where its own 6x6 inversion is ill-conditioned, and the Rust kernel
   reproduces that test in ``rust/src/guard.rs``.  Measured residual over the
   whole catalogue: **zero** — 397 900 masked nodes on both sides out of
   8.0e8 nodes.

Outputs ``validation/output/backend_parity/summary.json`` and ``summary.txt``.

Usage
-----
    python3 validation/backend_parity.py                    # everything
    python3 validation/backend_parity.py --python-n 500     # subsample the ref
    python3 validation/backend_parity.py --no-python        # golden only
    python3 validation/backend_parity.py --rust-workers 64
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

GATE = os.path.join(REPO, "validation", "output", "gate_v0")
OUTDIR = os.path.join(REPO, "validation", "output", "backend_parity")

# Acceptance bars (issue #1).
P_TOL = 1e-6
LAM_RTOL = 1e-5


# ---------------------------------------------------------------------------
# Python-reference worker (multiprocessing)
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(model_path, variant, cfg, gen_path, pair_kw=None):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    from ds42.population import Digest2Histogram

    warnings.simplefilter("ignore", RuntimeWarning)
    g = np.load(gen_path)
    # The density-mode kwargs (h_interpolation [D-17], h_extend_to [D-19]) come
    # from the run's own score_config.json.  Ignoring them scored the reference
    # backend against a DIFFERENT model than the run of record used, which is
    # exactly the silent mismatch the WP6 harness now hard-fails on — and it
    # would have shown up here as a spurious backend "disagreement".
    _W["neo"], _W["other"] = Digest2Histogram.neo_other_pair(
        model_path, variant, **(pair_kw or {})
    )
    _W["cfg"] = cfg
    _W["gen"] = {
        k: g[k] for k in (
            "ra", "dec", "ra_rate", "dec_rate", "v_mag", "epoch_jd_tdb",
            "R_obs", "V_obs",
        )
    }


def _score_python(k: int):
    from astropy.time import Time

    from ds42.attributable import Attributable
    from ds42.classifier import classify

    g = _W["gen"]
    cfg = _W["cfg"]
    att = Attributable(
        ra=float(g["ra"][k]),
        dec=float(g["dec"][k]),
        ra_rate=float(g["ra_rate"][k]),
        dec_rate=float(g["dec_rate"][k]),
        epoch=Time(float(g["epoch_jd_tdb"][k]), format="jd", scale="tdb"),
        v_mag=float(g["v_mag"][k]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = classify(
            att, _W["neo"], _W["other"], g["R_obs"][k], g["V_obs"][k],
            backend="python", diagnostics=False, **cfg,
        )
    return (
        k, r.p_neo, r.log_lr, r.lambda_neo, r.lambda_other, r.n_nodes,
        r.n_masked_nonfinite, r.n_masked_det, r.status,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rel(new, ref):
    """Relative error |new - ref| / |ref|, NaN where ref is 0 or non-finite."""
    new = np.asarray(new, dtype=float)
    ref = np.asarray(ref, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs(new - ref) / np.abs(ref)
    out = np.where(np.isfinite(ref) & (ref != 0.0), out, np.nan)
    # Exactly equal (including both zero / both inf) counts as zero error.
    same = (new == ref) | (np.isnan(new) & np.isnan(ref))
    return np.where(same, 0.0, out)


def _dist(x, name):
    x = np.asarray(x, dtype=float)
    good = x[np.isfinite(x)]
    if good.size == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(good.size),
        "n_undefined": int(x.size - good.size),
        "median": float(np.median(good)),
        "p90": float(np.percentile(good, 90)),
        "p99": float(np.percentile(good, 99)),
        "p99_9": float(np.percentile(good, 99.9)),
        "max": float(good.max()),
        "n_over_1e-9": int(np.count_nonzero(good > 1e-9)),
        "n_over_1e-6": int(np.count_nonzero(good > 1e-6)),
        "n_over_1e-5": int(np.count_nonzero(good > 1e-5)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default=GATE)
    ap.add_argument("--out", default=OUTDIR)
    ap.add_argument("--n", type=int, default=None,
                    help="limit the number of tracklets (default: all)")
    ap.add_argument("--rust-workers", type=int, default=None,
                    help="rayon workers (default: one per core)")
    ap.add_argument("--python-n", type=int, default=None,
                    help="tracklets to re-score with the Python reference "
                         "(default: all)")
    ap.add_argument("--python-workers", type=int, default=None)
    ap.add_argument("--no-python", action="store_true",
                    help="skip the fresh Python run (no mask comparison)")
    ap.add_argument("--chunk", type=int, default=2000,
                    help="tracklets per Rust batch call")
    args = ap.parse_args()

    from astropy.time import Time

    from ds42.attributable import Attributable
    from ds42.backend import core_available
    from ds42.classifier import classify_many
    from ds42.population import Digest2Histogram

    if not core_available():
        print("ds42._core is not built; run rust/build.sh", file=sys.stderr)
        return 2

    gen_path = os.path.join(args.gate, "gen.npz")
    scores_path = os.path.join(args.gate, "scores.npz")
    cfg_path = os.path.join(args.gate, "score_config.json")
    for p in (gen_path, scores_path, cfg_path):
        if not os.path.exists(p):
            print(f"missing {p}", file=sys.stderr)
            return 2

    with open(cfg_path) as fh:
        score_cfg = json.load(fh)
    cfg = dict(
        rho_min=float(score_cfg["rho_min"]),
        rho_max=float(score_cfg["rho_max"]),
        n_rho=int(score_cfg["n_rho"]),
        n_rhodot=int(score_cfg["n_rhodot"]),
        G=float(score_cfg["G"]),
    )
    model_path = os.path.join(REPO, score_cfg["model"])
    variant = score_cfg.get("variant", "All")
    # Density-mode kwargs, read from the run's own config so this harness
    # cannot compare the two backends on a model neither of them was asked
    # for.  `h_extend_to` is D-19; `h_interpolation` is D-17.  Absent keys mean
    # the defaults, i.e. an older run.
    pair_kw = {}
    if score_cfg.get("h_interpolation", "none") != "none":
        pair_kw["h_interpolation"] = str(score_cfg["h_interpolation"])
    if score_cfg.get("h_extend_to") is not None:
        pair_kw["h_extend_to"] = float(score_cfg["h_extend_to"])

    g = np.load(gen_path)
    ref = np.load(scores_path)
    n_all = len(g["ra"])
    n = n_all if args.n is None else min(args.n, n_all)
    print(f"gate_v0: {n_all} tracklets ({n} scored); grid "
          f"({cfg['n_rho']}, {cfg['n_rhodot']}); model {score_cfg['model']} "
          f"[{variant}]")

    neo, other = Digest2Histogram.neo_other_pair(model_path, variant,
                                                 **pair_kw)
    if pair_kw:
        print(f"density mode: {pair_kw}")

    # -- Rust pass ---------------------------------------------------------
    got = {k: np.empty(n, dtype=float) for k in
           ("p_neo", "log_lr", "lambda_neo", "lambda_other")}
    for k in ("n_nodes", "n_masked_nonfinite", "n_masked_det"):
        got[k] = np.empty(n, dtype=np.int64)
    got["status"] = np.empty(n, dtype="U28")

    warnings.simplefilter("ignore", RuntimeWarning)
    # Build the Attributables (and their astropy Time epochs) up front: that
    # cost belongs to the harness, not to the kernel being timed.
    epochs = Time(g["epoch_jd_tdb"][:n], format="jd", scale="tdb")
    all_atts = [
        Attributable(
            ra=float(g["ra"][k]),
            dec=float(g["dec"][k]),
            ra_rate=float(g["ra_rate"][k]),
            dec_rate=float(g["dec_rate"][k]),
            epoch=epochs[k],
            v_mag=float(g["v_mag"][k]),
        )
        for k in range(n)
    ]
    t0 = time.perf_counter()
    done = 0
    while done < n:
        hi = min(done + args.chunk, n)
        idx = np.arange(done, hi)
        atts = all_atts[done:hi]
        res = classify_many(
            atts, neo, other, R_obs=g["R_obs"][idx], V_obs=g["V_obs"][idx],
            backend="rust", diagnostics=False, n_threads=args.rust_workers,
            **cfg,
        )
        for j, r in zip(idx, res):
            got["p_neo"][j] = r.p_neo
            got["log_lr"][j] = r.log_lr
            got["lambda_neo"][j] = r.lambda_neo
            got["lambda_other"][j] = r.lambda_other
            got["n_nodes"][j] = r.n_nodes
            got["n_masked_nonfinite"][j] = r.n_masked_nonfinite
            got["n_masked_det"][j] = r.n_masked_det
            got["status"][j] = r.status
        done = hi
        print(f"  rust: {done}/{n}  ({time.perf_counter() - t0:.1f} s)",
              flush=True)
    rust_s = time.perf_counter() - t0
    print(f"rust pass: {rust_s:.1f} s total, "
          f"{rust_s / n * 1e3:.3f} ms/tracklet wall")

    # -- Golden comparison -------------------------------------------------
    summary: dict = {
        "n_tracklets": int(n),
        "grid": cfg,
        "model": score_cfg["model"],
        "variant": variant,
        "rust_wall_s": rust_s,
        "rust_ms_per_tracklet_wall": rust_s / n * 1e3,
        "rust_workers": args.rust_workers,
        "tolerances": {"p_neo": P_TOL, "lambda_rel": LAM_RTOL},
    }

    p_ref = np.asarray(ref["p_neo"], dtype=float)[:n]
    p_got = got["p_neo"]
    nan_ref = np.isnan(p_ref)
    nan_got = np.isnan(p_got)
    summary["p_neo_nan_agreement"] = bool(np.array_equal(nan_ref, nan_got))
    summary["n_nan_ref"] = int(nan_ref.sum())
    summary["n_nan_got"] = int(nan_got.sum())
    both = ~nan_ref & ~nan_got
    dp = np.abs(p_got[both] - p_ref[both])
    summary["p_neo_max_abs_diff"] = float(dp.max()) if dp.size else 0.0
    summary["p_neo_p99_9_abs_diff"] = (
        float(np.percentile(dp, 99.9)) if dp.size else 0.0
    )
    summary["p_neo_n_over_tol"] = int(np.count_nonzero(dp > P_TOL))
    summary["p_neo_pass"] = bool(
        summary["p_neo_max_abs_diff"] <= P_TOL
        and summary["p_neo_nan_agreement"]
    )

    # Needed by the status reconstruction below, which runs before the
    # n_nodes block defines its own copy.
    nn_ref_pre = np.asarray(ref["n_nodes"], dtype=np.int64)[:n]

    lam_dists = []
    lam_pass = True
    for name in ("lambda_neo", "lambda_other"):
        rel = _rel(got[name], np.asarray(ref[name], dtype=float)[:n])
        d = _dist(rel, name)
        lam_dists.append(d)
        if d.get("n", 0):
            lam_pass = lam_pass and d["max"] <= LAM_RTOL
    summary["lambda_rel_error"] = lam_dists
    summary["lambda_pass"] = bool(lam_pass)

    ll_ref = np.asarray(ref["log_lr"], dtype=float)[:n]
    ll_rel = _rel(got["log_lr"], ll_ref)
    summary["log_lr_rel_error"] = _dist(ll_rel, "log_lr")
    # NaN-aware, because a NaN log_lr now carries meaning: since D-19 it is the
    # marker of a triage-policy assignment (p = 1.0 with no likelihood ratio
    # behind it), so the two backends disagreeing about which tracklets have
    # one is a real defect rather than a formality.
    summary["log_lr_nan_agreement"] = bool(
        np.array_equal(np.isnan(ll_ref), np.isnan(got["log_lr"]))
    )
    summary["log_lr_n_nan"] = int(np.count_nonzero(np.isnan(got["log_lr"])))

    # D-19 statuses.  The stored reference predates the field, so its status is
    # RECONSTRUCTED from the recorded (n_nodes, Lambda) triple exactly as the
    # WP6 harness reconstructs it — including the isfinite(p) term, which keeps
    # a pre-D-19 nan-p row out of the policy column.
    ref_lam_n = np.asarray(ref["lambda_neo"], dtype=float)[:n]
    ref_lam_o = np.asarray(ref["lambda_other"], dtype=float)[:n]
    ref_policy = (
        np.isfinite(p_ref) & (nn_ref_pre > 0)
        & (ref_lam_n == 0.0) & (ref_lam_o == 0.0)
    )
    got_policy = got["status"] == "fully_truncated_flagged"
    summary["policy_assigned_n_rust"] = int(np.count_nonzero(got_policy))
    summary["policy_assigned_n_reference_reconstructed"] = int(
        np.count_nonzero(ref_policy)
    )
    summary["policy_assigned_agreement"] = bool(
        np.array_equal(ref_policy, got_policy)
    )
    summary["status_counts_rust"] = {
        str(k): int(v) for k, v in zip(*np.unique(got["status"],
                                                  return_counts=True))
    }

    nn_ref = np.asarray(ref["n_nodes"], dtype=np.int64)[:n]
    summary["n_nodes_exact_match"] = bool(
        np.array_equal(nn_ref, got["n_nodes"])
    )
    summary["n_nodes_n_mismatch"] = int(
        np.count_nonzero(nn_ref != got["n_nodes"])
    )
    summary["n_empty_region"] = int(np.count_nonzero(got["n_nodes"] == 0))

    # -- Fresh Python reference (mask counters) ----------------------------
    if args.no_python:
        summary["python_reference"] = None
    else:
        pn = n if args.python_n is None else min(args.python_n, n)
        # The reference path allocates ~200-300 MB of (N, 6, 6) Jacobians per
        # worker at the production grid, so the process count is capped well
        # below the core count: past ~64 workers the run goes memory-bound and
        # the extra processes only add page-cache pressure.
        workers = args.python_workers or min(
            64, max(1, (os.cpu_count() or 8) - 4)
        )
        print(f"python reference: {pn} tracklets on {workers} processes ...",
              flush=True)
        t1 = time.perf_counter()
        ctx = mp.get_context("fork")
        with ctx.Pool(
            workers,
            initializer=_init_worker,
            initargs=(model_path, variant,
                      {k: v for k, v in cfg.items()}, gen_path, pair_kw),
        ) as pool:
            rows = pool.map(_score_python, range(pn), chunksize=8)
        py_s = time.perf_counter() - t1
        rows.sort()
        py = {
            "p_neo": np.array([r[1] for r in rows]),
            "log_lr": np.array([r[2] for r in rows]),
            "lambda_neo": np.array([r[3] for r in rows]),
            "lambda_other": np.array([r[4] for r in rows]),
            "n_nodes": np.array([r[5] for r in rows], dtype=np.int64),
            "n_masked_nonfinite": np.array([r[6] for r in rows],
                                           dtype=np.int64),
            "n_masked_det": np.array([r[7] for r in rows], dtype=np.int64),
            "status": np.array([r[8] for r in rows], dtype="U28"),
        }
        mnf_eq = np.array_equal(
            py["n_masked_nonfinite"], got["n_masked_nonfinite"][:pn]
        )
        det_py = py["n_masked_det"]
        det_rs = got["n_masked_det"][:pn]
        tot_nodes = int(py["n_nodes"].sum())
        pdp = np.abs(py["p_neo"] - got["p_neo"][:pn])
        pdp = pdp[np.isfinite(pdp)]
        rel_neo = _rel(got["lambda_neo"][:pn], py["lambda_neo"])
        rel_other = _rel(got["lambda_other"][:pn], py["lambda_other"])
        summary["python_reference"] = {
            "n_tracklets": int(pn),
            "workers": int(workers),
            "wall_s": py_s,
            "ms_per_tracklet_single_core": py_s * workers / pn * 1e3,
            "p_neo_max_abs_diff": float(pdp.max()) if pdp.size else 0.0,
            "p_neo_nan_agreement": bool(
                np.array_equal(np.isnan(py["p_neo"]),
                               np.isnan(got["p_neo"][:pn]))
            ),
            "n_nodes_exact_match": bool(
                np.array_equal(py["n_nodes"], got["n_nodes"][:pn])
            ),
            # D-19: `status` and a NaN-aware `log_lr` are part of the contract
            # the two backends share (both come out of
            # `classifier._posterior`), so a mismatch here means one backend
            # fed it different mechanism evidence.
            "status_exact_match": bool(
                np.array_equal(py["status"], got["status"][:pn])
            ),
            "status_n_differing": int(
                np.count_nonzero(py["status"] != got["status"][:pn])
            ),
            "log_lr_nan_agreement": bool(
                np.array_equal(np.isnan(py["log_lr"]),
                               np.isnan(got["log_lr"][:pn]))
            ),
            "policy_assigned_n": int(np.count_nonzero(
                py["status"] == "fully_truncated_flagged"
            )),
            "n_masked_nonfinite_exact_match": bool(mnf_eq),
            "n_masked_nonfinite_total": int(py["n_masked_nonfinite"].sum()),
            "n_masked_det_total_python": int(det_py.sum()),
            "n_masked_det_total_rust": int(det_rs.sum()),
            "n_masked_det_abs_diff_total": int(np.abs(det_py - det_rs).sum()),
            "n_masked_det_tracklets_differing": int(
                np.count_nonzero(det_py != det_rs)
            ),
            "n_nodes_total": tot_nodes,
            "n_masked_det_disagreement_fraction_of_nodes": (
                float(np.abs(det_py - det_rs).sum() / tot_nodes)
                if tot_nodes else 0.0
            ),
            "lambda_rel_error": [
                _dist(rel_neo, "lambda_neo"), _dist(rel_other, "lambda_other"),
            ],
        }

    summary["PASS"] = bool(
        summary["p_neo_pass"]
        and summary["lambda_pass"]
        and summary["n_nodes_exact_match"]
        # D-19: the status vocabulary and the NaN pattern of log_lr are part of
        # the shared contract, so they are verdict criteria, not just readouts.
        and summary["log_lr_nan_agreement"]
        and summary["policy_assigned_agreement"]
        and (
            summary["python_reference"] is None
            or (
                summary["python_reference"]["n_masked_nonfinite_exact_match"]
                and summary["python_reference"]["status_exact_match"]
                and summary["python_reference"]["log_lr_nan_agreement"]
            )
        )
    )

    # -- Report ------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    lines = []
    L = lines.append
    L("ds42 backend A/B — Rust core kernel vs the stored gate_v0 scores")
    L("=" * 72)
    L(f"tracklets                 : {summary['n_tracklets']}")
    L(f"grid                      : ({cfg['n_rho']}, {cfg['n_rhodot']}), "
      f"rho in [{cfg['rho_min']:g}, {cfg['rho_max']:g}] AU, G = {cfg['G']}")
    L(f"model                     : {score_cfg['model']} [{variant}]")
    L(f"rust wall time            : {rust_s:.1f} s "
      f"({summary['rust_ms_per_tracklet_wall']:.3f} ms/tracklet, "
      f"{args.rust_workers or 'all'} workers)")
    L("")
    L("1. Golden comparison against validation/output/gate_v0/scores.npz")
    L(f"   max |dp_neo|           : {summary['p_neo_max_abs_diff']:.3e}   "
      f"(tol {P_TOL:g})  ->  "
      f"{'PASS' if summary['p_neo_pass'] else 'FAIL'}")
    L(f"   p99.9 |dp_neo|         : {summary['p_neo_p99_9_abs_diff']:.3e}")
    L(f"   tracklets over tol     : {summary['p_neo_n_over_tol']}")
    L(f"   NaN p_neo agreement    : {summary['p_neo_nan_agreement']} "
      f"({summary['n_nan_ref']} stored / {summary['n_nan_got']} rust)")
    L(f"   NaN log_lr agreement   : {summary['log_lr_nan_agreement']} "
      f"({summary['log_lr_n_nan']} rust)")
    L(f"   D-19 policy rows       : {summary['policy_assigned_n_rust']} rust / "
      f"{summary['policy_assigned_n_reference_reconstructed']} reconstructed "
      f"from the stored (n_nodes, Lambda) triple  ->  "
      f"{'PASS' if summary['policy_assigned_agreement'] else 'FAIL'}")
    L(f"   rust status counts     : {summary['status_counts_rust']}")
    for d in lam_dists:
        if not d.get("n"):
            continue
        L(f"   {d['name']:<14} rel   : median {d['median']:.2e}  "
          f"p99 {d['p99']:.2e}  p99.9 {d['p99_9']:.2e}  max {d['max']:.2e}"
          f"   (tol {LAM_RTOL:g})")
        L(f"   {'':<14}         : n>1e-9 {d['n_over_1e-9']}, "
          f"n>1e-6 {d['n_over_1e-6']}, n>1e-5 {d['n_over_1e-5']}")
    d = summary["log_lr_rel_error"]
    if d.get("n"):
        L(f"   log_lr         rel   : median {d['median']:.2e}  "
          f"max {d['max']:.2e}")
    L(f"   Lambda verdict         : "
      f"{'PASS' if summary['lambda_pass'] else 'FAIL'}")
    L("")
    L("2. Node counts")
    L(f"   n_nodes exact match    : {summary['n_nodes_exact_match']} "
      f"({summary['n_nodes_n_mismatch']} mismatched)")
    L(f"   empty admissible region: {summary['n_empty_region']} tracklets")
    L("")
    pr = summary["python_reference"]
    if pr is None:
        L("3. Mask counters: skipped (--no-python)")
    else:
        L(f"3. Fresh Python reference ({pr['n_tracklets']} tracklets, "
          f"{pr['workers']} processes, {pr['wall_s']:.1f} s wall; "
          f"{pr['ms_per_tracklet_single_core']:.0f} ms/tracklet single-core)")
        L(f"   max |dp_neo| vs fresh  : {pr['p_neo_max_abs_diff']:.3e}")
        L(f"   status exact match     : {pr['status_exact_match']} "
          f"({pr['status_n_differing']} differing; "
          f"{pr['policy_assigned_n']} D-19 policy rows)")
        L(f"   NaN log_lr agreement   : {pr['log_lr_nan_agreement']}")
        L(f"   n_nodes exact match    : {pr['n_nodes_exact_match']}")
        L(f"   n_masked_nonfinite     : EXACT MATCH = "
          f"{pr['n_masked_nonfinite_exact_match']} "
          f"(total {pr['n_masked_nonfinite_total']})")
        L(f"   n_masked_det totals    : python {pr['n_masked_det_total_python']}"
          f"  rust {pr['n_masked_det_total_rust']}")
        L(f"   n_masked_det |diff|    : {pr['n_masked_det_abs_diff_total']} "
          f"nodes over {pr['n_nodes_total']} "
          f"({pr['n_masked_det_disagreement_fraction_of_nodes']:.2e} of all "
          f"nodes), on {pr['n_masked_det_tracklets_differing']} tracklets")
        for dd in pr["lambda_rel_error"]:
            if dd.get("n"):
                L(f"   {dd['name']:<14} rel   : median {dd['median']:.2e}  "
                  f"max {dd['max']:.2e}")
    L("")
    L(f"VERDICT: {'PASS' if summary['PASS'] else 'FAIL'}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(text)
    print()
    print(text)
    print(f"wrote {args.out}/summary.json and summary.txt")
    return 0 if summary["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
