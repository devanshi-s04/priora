#!/usr/bin/env python3
"""COPY of output/h_extension/rescore_neocp.py (2026-11-06) with the population variant switched to Unk and only the extended config:
the NEOCP test redone as ds42 Unk (extended, h_extend_to=30) to pair with digest2 NoID. Writes neocp_delta_all.json here.
Original docstring follows.

Read-only with respect to ``validation/neocp_pilot.py`` and everything under
``validation/output/neocp_pilot/`` — it imports the pilot module for its
``Tracklet`` dataclass and loader, reads the cached ``tracklets.json`` /
``scores.npz``, and writes only into ``validation/output/h_extension/``.

What it produces
----------------
1. ``neocp_rescore.json`` / ``.md`` — the 11 previously-unscorable tracklets
   (the ones the pilot reports at ``p_neo = nan``) re-scored under

     (a) the D-19 extended pair              h_extend_to=30
     (b) extended + pchip                    h_extend_to=30, h_interpolation="pchip"
     (c) policy-only, unextended             the D-4 pair, D-19 triage policy
     (d) the D-14 clamped pair               reproduced here for comparison

   with, per tracklet, the residual admissible-region measure fraction ABOVE
   H = 30 — i.e. how much of the region the extension still truncates.  At
   rho_min = 0.01 AU the implied H reaches ~34 for these objects, so that
   residual is not zero and is reported rather than hidden.

2. ``neocp_delta_all.json`` — the same four configurations on ALL 499 pilot
   tracklets, so the "changes nothing else" claim is a measurement: the |Δp|
   distribution of extended-vs-truncated and extended-vs-clamped over the
   tracklets that were already scorable.

Usage
-----
    OMP_NUM_THREADS=1 python validation/output/h_extension/rescore_neocp.py \
        [--workers 90] [--stage wall|all|both]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from validation import neocp_pilot as pilot  # noqa: E402

PILOT_DIR = os.path.join(REPO, "validation", "output", "neocp_pilot")
OUT_DIR = HERE
MODEL = os.path.join(REPO, "data", "digest2.model.csv")

H_EXTEND_TO = 30.0
N_RHO = 1200
N_RHODOT = 50
RHO_MIN = 0.01
RHO_MAX = 100.0

#: The four configurations under test.  ``kind`` selects the pair; the
#: classifier is otherwise identical in all four (same grid, same rho window,
#: same G, same observer resolution), so any difference is the model's.
CONFIGS = {
    "extended": dict(h_extend_to=H_EXTEND_TO),              # D-19, Unk
}

_W: dict = {}


def _pair():
    return _W["neo"], _W["other"]


def _init_worker(kind: str) -> None:
    from ds42.population import Digest2Histogram

    neo, other = Digest2Histogram.neo_other_pair(MODEL, "Unk", **CONFIGS[kind])
    _W["neo"] = neo
    _W["other"] = other
    _W["kind"] = kind


def _score_one(task: tuple) -> dict:
    from astropy.time import Time

    from ds42 import Attributable, classify
    from ds42.classifier import h_support_end

    (idx, ra, dec, ra_rate, dec_rate, jd, v_mag, lon, lat, height) = task
    att = Attributable(
        ra=float(ra), dec=float(dec), ra_rate=float(ra_rate),
        dec_rate=float(dec_rate),
        epoch=Time(float(jd), format="jd", scale="utc"), v_mag=float(v_mag),
    )
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = classify(
            att, _W["neo"], _W["other"],
            lon_deg=float(lon), lat_deg=float(lat), height_m=float(height),
            rho_min=RHO_MIN, rho_max=RHO_MAX, n_rho=N_RHO, n_rhodot=N_RHODOT,
            backend="rust",
        )
    h = np.asarray(res.node_H, dtype=float)
    fin = np.isfinite(h)
    out = {
        "index": int(idx),
        "p_neo": float(res.p_neo),
        # D-19 mechanism evidence, so the report can show that the trigger
        # fired for the right reason on these 11 and only these.
        "n_h_finite": int(fin.sum()),
        "n_h_at_or_above_25p5": int((h[fin] >= 25.5).sum()) if fin.any() else 0,
        "min_node_H": float(h[fin].min()) if fin.any() else float("nan"),
        # The D-19 trigger's own quantity: the MEASURE fraction at/above the
        # pair's H support end (not a node-count fraction).
        "h_measure_frac_above_support": float(
            res.measure_fraction_H_above(
                np.nextafter(h_support_end(*_pair()), 0.0)
            )
        ) if res.n_nodes else float("nan"),
        "log_lr": float(res.log_lr),
        "lambda_neo": float(res.lambda_neo),
        "lambda_other": float(res.lambda_other),
        "n_nodes": int(res.n_nodes),
        "status": str(res.status),
        "policy_assigned": bool(res.policy_assigned),
        "ar_frac_H_above_25p5": float(res.measure_fraction_H_above(25.5)),
        "ar_frac_H_above_30": float(res.measure_fraction_H_above(30.0)),
        "ar_frac_H_above_34": float(res.measure_fraction_H_above(34.0)),
        "max_node_H": float(np.nanmax(res.node_H)) if res.n_nodes else float("nan"),
        "seconds": time.time() - t0,
    }
    return out


def _tasks(tracklets, indices):
    for i in indices:
        t = tracklets[i]
        yield (
            i, t.ra, t.dec, t.ra_rate, t.dec_rate, t.epoch_jd_utc, t.v_mag,
            t.lon_deg, t.lat_deg, t.height_m,
        )


def run_config(tracklets, indices, kind: str, workers: int) -> dict[int, dict]:
    tasks = list(_tasks(tracklets, indices))
    rows: dict[int, dict] = {}
    if workers <= 1:
        _init_worker(kind)
        for task in tasks:
            r = _score_one(task)
            rows[r["index"]] = r
        return rows
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(kind,)) as ex:
        for r in ex.map(_score_one, tasks, chunksize=1):
            rows[r["index"]] = r
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=90)
    ap.add_argument("--stage", default="both", choices=("wall", "all", "both"))
    a = ap.parse_args(argv)

    tracklets, census = pilot.load_tracklets(PILOT_DIR)
    z = np.load(os.path.join(PILOT_DIR, "scores.npz"))
    p_all = z["d42_all_p_neo"]
    idx_all = z["d42_all_index"].astype(int)
    order = np.argsort(idx_all)
    assert np.array_equal(idx_all[order], np.arange(len(tracklets)))
    p_all = p_all[order]
    p_clamp_cached = z["d42_allclamp_p_neo"][
        np.argsort(z["d42_allclamp_index"].astype(int))
    ]
    is_neo = z["is_neo"].astype(bool)

    wall = np.flatnonzero(~np.isfinite(p_all))
    print(f"[rescore] pilot reports {wall.size} unscorable tracklets: "
          f"{[tracklets[i].designation for i in wall]}")

    results = {}
    if a.stage in ("wall", "both"):
        for kind in CONFIGS:
            t0 = time.time()
            results[kind] = run_config(tracklets, wall, kind, a.workers)
            print(f"[rescore] wall/{kind}: {len(results[kind])} tracklets in "
                  f"{time.time() - t0:.1f} s")
        blob = {
            "designations": [tracklets[i].designation for i in wall],
            "indices": [int(i) for i in wall],
            "is_neo": [bool(is_neo[i]) for i in wall],
            "true_q_au": [float(tracklets[i].q_au) for i in wall],
            "v_mag": [float(tracklets[i].v_mag) for i in wall],
            "rate_deg_day": [
                float(np.degrees(np.hypot(
                    tracklets[i].ra_rate * np.cos(tracklets[i].dec),
                    tracklets[i].dec_rate)))
                for i in wall
            ],
            "p_clamp_cached": [float(p_clamp_cached[i]) for i in wall],
            "configs": {k: {str(i): results[k][i] for i in wall}
                        for k in CONFIGS},
            "meta": {
                "model": MODEL, "h_extend_to": H_EXTEND_TO,
                "n_rho": N_RHO, "n_rhodot": N_RHODOT,
                "rho_min": RHO_MIN, "rho_max": RHO_MAX, "backend": "rust",
            },
        }
        with open(os.path.join(OUT_DIR, "neocp_rescore.json"), "w") as fh:
            json.dump(blob, fh, indent=1, default=float)
        print("[rescore] wrote neocp_rescore.json")

    if a.stage in ("all", "both"):
        every = np.arange(len(tracklets))
        full = {}
        for kind in CONFIGS:
            t0 = time.time()
            full[kind] = run_config(tracklets, every, kind, a.workers)
            print(f"[rescore] all/{kind}: {len(full[kind])} tracklets in "
                  f"{time.time() - t0:.1f} s")
        blob = {
            "n": int(len(tracklets)),
            "designations": [t.designation for t in tracklets],
            "is_neo": [bool(v) for v in is_neo],
            "configs": {
                k: {
                    "p_neo": [full[k][i]["p_neo"] for i in every],
                    "status": [full[k][i]["status"] for i in every],
                    "policy_assigned": [full[k][i]["policy_assigned"]
                                        for i in every],
                    "lambda_neo": [full[k][i]["lambda_neo"] for i in every],
                    "lambda_other": [full[k][i]["lambda_other"] for i in every],
                    "ar_frac_H_above_25p5": [
                        full[k][i]["ar_frac_H_above_25p5"] for i in every],
                    "ar_frac_H_above_30": [
                        full[k][i]["ar_frac_H_above_30"] for i in every],
                }
                for k in CONFIGS
            },
            "meta": {"model": MODEL, "h_extend_to": H_EXTEND_TO,
                     "n_rho": N_RHO, "n_rhodot": N_RHODOT, "backend": "rust"},
        }
        with open(os.path.join(OUT_DIR, "neocp_delta_all.json"), "w") as fh:
            json.dump(blob, fh, indent=1, default=float)
        print("[rescore] wrote neocp_delta_all.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
