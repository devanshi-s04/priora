#!/usr/bin/env python3
"""D-19 validation driver: Δp vs the D-14 clamp on the gate_v0 catalogue.

Scores the *stored* ``gate_v0`` catalogue (which was generated from the
UNEXTENDED model — this is a scoring comparison, not a gate run; the harness
correctly refuses to call it a gate) under four density pairs:

    truncated       D-4, the run of record
    extended        D-19, h_extend_to = 30
    extended_pchip  D-19 + D-17
    clamped         D-14, h_top_bin_infinite = True

and records, per tracklet, each pair's p(NEO) together with the DESIGN §2.6(b)
admissible-region measure fractions above H = 25.5 and above H = 30.  Those
two columns are what makes the Δp distribution interpretable: the extension can
only change a tracklet's score through region it has above 25.5, so "Δp ≈ 0
where the measure above 25.5 is negligible" is a prediction to be checked, not
an excuse.

Read-only with respect to ``validation/output/gate_v0/``; writes
``gate_delta.npz`` into this directory.

Usage
-----
    OMP_NUM_THREADS=1 python validation/output/h_extension/gate_delta.py \
        [--n 20000] [--chunk 250] [--threads 90]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from astropy.time import Time  # noqa: E402

from priora.attributable import Attributable  # noqa: E402
from priora.classifier import classify_many  # noqa: E402
from priora.population import Digest2Histogram  # noqa: E402

GATE_DIR = os.path.join(REPO, "validation", "output", "gate_v0")
MODEL = os.path.join(REPO, "data", "digest2.model.csv")
H_EXTEND_TO = 30.0

PAIRS = {
    "truncated": dict(),
    "extended": dict(h_extend_to=H_EXTEND_TO),
    "extended_pchip": dict(h_extend_to=H_EXTEND_TO, h_interpolation="pchip"),
    "clamped": dict(h_top_bin_infinite=True),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--threads", type=int, default=90)
    a = ap.parse_args(argv)

    g = np.load(os.path.join(GATE_DIR, "gen.npz"))
    cfg_path = os.path.join(GATE_DIR, "gen_config.json")
    cfg = {}
    for cand in (cfg_path, os.path.join(GATE_DIR, "summary.json")):
        if os.path.exists(cand):
            with open(cand) as fh:
                blob = json.load(fh)
            cfg = blob.get("generation_config", blob)
            break
    n = min(a.n, len(g["ra"]))
    print(f"[gate_delta] gate_v0 catalogue: {len(g['ra'])} tracklets, using {n}")
    print(f"[gate_delta] generator h_extend_to = "
          f"{cfg.get('h_extend_to', 'None')} (this is a SCORING comparison)")

    pairs = {k: Digest2Histogram.neo_other_pair(MODEL, "All", **v)
             for k, v in PAIRS.items()}

    out = {k: np.full(n, np.nan) for k in PAIRS}
    status = {k: np.zeros(n, dtype="U28") for k in PAIRS}
    policy = {k: np.zeros(n, dtype=bool) for k in PAIRS}
    ar25 = np.full(n, np.nan)
    ar30 = np.full(n, np.nan)
    maxH = np.full(n, np.nan)
    t0 = time.time()
    for s in range(0, n, a.chunk):
        e = min(s + a.chunk, n)
        atts = [
            Attributable(
                ra=float(g["ra"][i]), dec=float(g["dec"][i]),
                ra_rate=float(g["ra_rate"][i]),
                dec_rate=float(g["dec_rate"][i]),
                epoch=Time(float(g["epoch_jd_tdb"][i]), format="jd",
                           scale="tdb"),
                v_mag=float(g["v_mag"][i]),
            )
            for i in range(s, e)
        ]
        R = np.ascontiguousarray(g["R_obs"][s:e])
        V = np.ascontiguousarray(g["V_obs"][s:e])
        for kind, (neo, other) in pairs.items():
            diag = kind == "truncated"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                res = classify_many(
                    atts, neo, other, R_obs=R, V_obs=V, backend="rust",
                    diagnostics=diag, n_threads=a.threads,
                )
            for k, r in enumerate(res):
                out[kind][s + k] = r.p_neo
                status[kind][s + k] = r.status
                policy[kind][s + k] = r.policy_assigned
                if diag:
                    ar25[s + k] = r.measure_fraction_H_above(25.5)
                    ar30[s + k] = r.measure_fraction_H_above(30.0)
                    maxH[s + k] = (np.nanmax(r.node_H) if r.n_nodes
                                   else np.nan)
        if (s // a.chunk) % 8 == 0:
            print(f"[gate_delta] {e}/{n} in {time.time() - t0:.0f} s",
                  flush=True)

    np.savez_compressed(
        os.path.join(HERE, "gate_delta.npz"),
        index=np.arange(n),
        is_neo=g["is_neo"][:n],
        v_mag=g["v_mag"][:n],
        ar_frac_H_above_25p5=ar25,
        ar_frac_H_above_30=ar30,
        max_node_H=maxH,
        **{f"p_{k}": out[k] for k in PAIRS},
        **{f"status_{k}": status[k] for k in PAIRS},
        **{f"policy_{k}": policy[k] for k in PAIRS},
    )
    print(f"[gate_delta] wrote gate_delta.npz in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
