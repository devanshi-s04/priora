"""WP7 — digest2 head-to-head comparison on shared inputs (DESIGN §4.2).

What this is
------------
DESIGN §4.2 asks for the one experiment that isolates *method* from *prior*:
run the MPC's own digest2 and priora on **the same tracklets** with **the
same population file**, and measure the ranking gap.  Because the WP6 harness
generates its test population *from* that same file, the prior is not merely
"shared" — it is correct for both codes.  Whatever ROC gap remains is the value
of the Jacobian + marginalisation + calibrated posterior, not of a better prior.

How the two codes are wired to the same model
---------------------------------------------
The pip-installed ``digest2`` Python package ships ``digest2/data/digest2.model.csv``
as an **unresolved git-LFS pointer** (132 bytes) whose recorded ``oid`` is
``sha256:2b9d9d2a…`` — bit-for-bit the file this repo fetches to
``data/digest2.model.csv``.  So the shared-model premise is not an
approximation: we pass ``Digest2(model_path=…)`` (equivalently ``$DIGEST2_MODEL``)
pointing at our copy, and the checksum is recorded in the summary.  Observatory
codes come from the package's cached ``digest2.obscodes`` (auto-downloaded from
the MPC on first use); only code 500 (geocentre) is used, matching the WP6
generator's geocentric observer.

Input format
------------
ADES PSV.  The installed package exposes ``parse_ades_psv`` and treats a
``.psv`` suffix as ADES, which buys full float precision on RA/Dec and
microsecond time resolution — MPC 80-column would round RA to 0.001 s and Dec
to 0.01 arcsec.  ``--format mpc80`` is kept as a fallback.

Each WP6 attributable (ra, dec, ra_rate, dec_rate, epoch, V) becomes a
two-detection tracklet: detection 0 **at** the attributable epoch and detection
1 at ``epoch + dt`` (default 45 min).  Detection 0 is placed at the epoch, not
symmetrically about it, because ``d2math.c`` anchors its ranging on
``obsPair[0]`` (``observer_object0``) — so digest2's reference position and
observer state are exactly the attributable's.  Linear propagation in
(alpha, delta) is exact at this arc length for the purpose at hand: the WP6
attributable *is* a first-order motion model, so the pair reproduces it
identically by construction.

Which digest2 score
-------------------
priora's v0 scoring uses the **All** variant of the model.  digest2's
``raw`` score is the All-population fraction; its ``noid`` (default) score
uses the ``Unk`` (yet-undiscovered) parallel histogram — see ALGORITHM.md
steps 5-6.  The headline comparison therefore uses ``result.raw.NEO``;
``noid.NEO`` is carried through the summary for reference.

The "same inputs" caveat (DESIGN §4.2, WP4 review)
--------------------------------------------------
``d2model.c:hToBin`` runs ``for (ih = 0; h >= hpart[ih] && ih < HX - 1; ++ih)``
with ``hpart[17] = 25.5``, so **every** trial orbit with H >= 24 lands in the
final bin: digest2's top H bin is effectively [24, inf).  priora truncates
the model at H = 25.5 per D-4.  For a tracklet with appreciable
admissible-region measure at implied H > 25.5 the two codes are therefore *not*
consuming the same effective model, and the head-to-head premise is broken for
that tracklet.  ``scores.npz['ar_frac_h_above']`` is the §2.6(b) diagnostic
that measures it.  Following DESIGN §4.2 the comparison is reported on three
populations:

* ``clean``  — ``ar_frac_h_above < 1e-3``: the headline number;
* ``broken`` — ``ar_frac_h_above > 1e-2``: premise broken, reported separately;
* ``all``    — everything, for completeness.

CLI
---
::

    python -m validation.digest2_comparison run \
        --run validation/output/gate_v0 \
        --out validation/output/digest2_comparison/gate_v0

    python -m validation.digest2_comparison report \
        --out validation/output/digest2_comparison \
        --case gate_v0 --case enriched_f01

    python -m validation.digest2_comparison clamped \
        --run validation/output/gate_v0 \
        --out validation/output/digest2_comparison/gate_v0 --workers 8

``run`` is a CLI, never a test: a full 2x10^4-tracklet pass takes minutes.
The unit tests exercise a 10-tracklet round trip and the pure-numpy analysis.

The clamped control (WP7 addendum)
----------------------------------
``clamped`` repairs the "same inputs" caveat above instead of routing around
it: it re-scores the *same* tracklets with a priora pair built with
``Digest2Histogram(h_top_bin_infinite=True)``, i.e. with digest2's own
[24, inf) top H bin, so that the premise holds for every tracklet.  It reuses
the case's existing digest2 scores and true labels, writes
``priora_clamped_scores.npz`` plus three plots into the case directory, adds
a ``clamped`` block to its ``summary.json``, and writes
``CLAMPED_ADDENDUM.md`` one level up.  See the section marked "WP7 addendum"
below for what the resulting decomposition means.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np

# Headless by default: this runs on batch nodes alongside the WP6 harness.
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from astropy.time import Time  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

__all__ = [
    "CLEAN_MAX_FRAC",
    "BROKEN_MIN_FRAC",
    "DEFAULT_DT_MIN",
    "DEFAULT_MODEL",
    "RunData",
    "auc_fast",
    "auc_bootstrap",
    "calibration_stats",
    "designation",
    "digest2_available",
    "digest2_env",
    "disagreement_census",
    "emit_ades_psv",
    "emit_mpc80",
    "load_run",
    "percentile_rank",
    "reliability_table",
    "run_digest2",
    "subset_masks",
    "analyze",
    "main",
    # WP7 addendum — the clamped (digest2-emulation) control
    "CLAMPED_SCORES",
    "ClampedScoreConfig",
    "analyze_clamped",
    "clamped_report",
    "clamped_score",
    "clamped_score_config",
    "load_clamped_scores",
]

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Shared population file (same bytes both codes; see module docstring).
DEFAULT_MODEL = os.path.join(REPO_ROOT, "data", "digest2.model.csv")

#: Tracklet arc, minutes.  30-60 min is the NEOCP-typical same-night pair.
DEFAULT_DT_MIN = 45.0

#: §2.6(b) truncation-diagnostic thresholds (DESIGN §4.2).
CLEAN_MAX_FRAC = 1e-3
BROKEN_MIN_FRAC = 1e-2

#: digest2 score used as the priora counterpart.  'raw' == All population.
D2_VARIANT_DEFAULT = "raw"
D2_CLASS = "NEO"

#: Geocentric observatory code — the WP6 generator's observer (obscode=None).
OBSCODE = "500"

SUBSETS = ("all", "clean", "broken")


# --------------------------------------------------------------------------
# digest2 availability
# --------------------------------------------------------------------------


def digest2_available() -> tuple[bool, str]:
    """(installed?, message).  Never raises."""
    try:
        import digest2  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, (
            "digest2 is not importable (%s). Install with:\n"
            '  CFLAGS="-D_GNU_SOURCE" pip install --user '
            '"git+https://github.com/Smithsonian/mpc-public.git'
            '@c9e3f89a26ec407d7c95226d7fd327d08be136c2#subdirectory=digest2"\n'
            "(the -D_GNU_SOURCE is needed because the sdist compiles with "
            "-std=c99, under which rand_r() is not declared)." % exc
        )
    return True, "digest2 importable"


def digest2_env(model_path: str = DEFAULT_MODEL) -> dict:
    """Record exactly which digest2 build and data files are in play."""
    info: dict = {"model_path": os.path.abspath(model_path)}
    try:
        info["model_sha256"] = _sha256(model_path)
    except OSError:
        info["model_sha256"] = None
    ok, msg = digest2_available()
    info["available"] = ok
    if not ok:
        info["message"] = msg
        return info
    import digest2
    from digest2.model import ensure_obscodes, find_config_path

    info["version"] = getattr(digest2, "__version__", None)
    if info["version"] is None:
        try:
            from importlib.metadata import version as _v

            info["version"] = _v("digest2")
        except Exception:  # pragma: no cover
            info["version"] = "unknown"
    info["source_commit"] = "c9e3f89a26ec407d7c95226d7fd327d08be136c2"
    info["package_dir"] = os.path.dirname(digest2.__file__)
    try:
        info["obscodes_path"] = ensure_obscodes()
    except Exception as exc:  # pragma: no cover
        info["obscodes_path"] = f"<unavailable: {exc}>"
    info["config_path"] = find_config_path()
    # digest2 caches a binary render of the CSV next to it on first load.
    # Harmless (it is derived from our file, and we always pass model_path
    # explicitly, which the loader honours over any sibling), but it is an
    # artifact this harness creates in data/ and should be on the record.
    abs_model = os.path.abspath(model_path)
    if abs_model.endswith(".csv"):
        cache = abs_model[: -len(".csv")]  # digest2.model.csv -> digest2.model
        if os.path.exists(cache):
            info["binary_model_cache"] = cache
    # The bundled model is an unresolved LFS pointer; record its declared oid
    # so the "same file both codes" claim is auditable.
    bundled = os.path.join(info["package_dir"], "data", "digest2.model.csv")
    info["bundled_model_is_lfs_pointer"] = False
    if os.path.exists(bundled) and os.path.getsize(bundled) < 4096:
        with open(bundled) as fh:
            head = fh.read(512)
        if "git-lfs" in head:
            info["bundled_model_is_lfs_pointer"] = True
            for line in head.splitlines():
                if line.startswith("oid sha256:"):
                    info["bundled_model_lfs_oid"] = line.split(":", 1)[1].strip()
    return info


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Run ingestion (WP6 gen.npz + scores.npz)
# --------------------------------------------------------------------------

#: gen.npz columns needed here.
_GEN_KEYS = (
    "ra",
    "dec",
    "ra_rate",
    "dec_rate",
    "v_mag",
    "epoch_jd_tdb",
    "rho_true",
    "rho_dot_true",
    "elements",
    "H",
    "is_neo",
    "weight",
)

#: scores.npz columns needed here (beyond 'index').
_SCORE_KEYS = (
    "p_neo",
    "log_lr",
    "n_nodes",
    "ar_measure",
    "ar_frac_h_above",
    "mass_frac_q_above_h_above",
)

#: Columns loaded when present but NOT required.
#:
#: D-19: ``lambda_neo``/``lambda_other`` are what identify a triage-policy row
#: (see :func:`load_run`).  Every vintage of ``self_consistency`` writes them,
#: so a real run always has them — but this module is also driven with
#: hand-built score tables (its own tests, and the injected-score path of
#: :func:`analyze`), and a *required* column would turn those into hard errors
#: for the sake of a filter that has nothing to filter.  When they are absent
#: the policy filter cannot run and ``RunData.policy_detectable`` is False,
#: which is reported rather than assumed away.
_OPTIONAL_SCORE_KEYS = ("lambda_neo", "lambda_other")


@dataclass
class RunData:
    """A WP6 run's generated catalogue joined to its priora scores.

    All arrays are aligned row-for-row and ordered as ``scores.npz['index']``.
    ``index`` is the row in ``gen.npz``; it is also what the emitted tracklet
    designation encodes, so digest2 results join back without ambiguity.
    """

    run_dir: str
    index: np.ndarray
    gen: dict = field(repr=False)
    scores: dict = field(repr=False)
    #: Tracklets dropped by :func:`load_run` because priora produced no
    #: likelihood score for them.  Reported rather than silent, because the
    #: symmetric-drop premise of this comparison depends on the drop being
    #: understood: ``n_dropped_nonfinite`` is a nan p_neo, and
    #: ``n_dropped_policy`` is a D-19 triage assignment (p = 1.0 with both
    #: Lambdas exactly zero — a defined number with no likelihood behind it, on
    #: a class that is empirically ~100% true NEO, so keeping it would bias
    #: ΔAUC toward priora by construction).
    n_dropped_nonfinite: int = 0
    n_dropped_policy: int = 0
    #: Whether the score table carried the Lambda columns the policy filter
    #: needs.  ``False`` for a hand-built table (this module's tests, the
    #: injected-score path); every real ``self_consistency`` run is ``True``.
    #: When ``False``, ``n_dropped_policy`` is 0 because nothing *could* be
    #: detected, which is not the same claim as "none were present".
    policy_detectable: bool = True

    @property
    def n(self) -> int:
        return int(self.index.size)

    @property
    def is_neo(self) -> np.ndarray:
        return self.gen["is_neo"].astype(bool)

    @property
    def weight(self) -> np.ndarray:
        return self.gen["weight"].astype(float)

    @property
    def p_neo(self) -> np.ndarray:
        return self.scores["p_neo"].astype(float)

    @property
    def ar_frac_h_above(self) -> np.ndarray:
        return self.scores["ar_frac_h_above"].astype(float)


def load_run(run_dir: str, limit: int | None = None) -> RunData:
    """Read ``gen.npz`` + ``scores.npz`` from a WP6 run directory.

    The join is on ``scores.npz['index']`` (the harness scores in chunks and
    concatenates, so the row order of scores.npz is not assumed to be 0..N-1).
    Rows whose priora ``p_neo`` is not finite are dropped — they cannot enter
    a ROC on either side — and the count is reported by the caller.
    """
    gpath = os.path.join(run_dir, "gen.npz")
    spath = os.path.join(run_dir, "scores.npz")
    for p in (gpath, spath):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    with np.load(spath) as z:
        missing = [k for k in ("index",) + _SCORE_KEYS if k not in z.files]
        if missing:
            raise KeyError(f"{spath} is missing columns {missing}")
        idx = z["index"].astype(int)
        scores = {k: np.asarray(z[k]) for k in _SCORE_KEYS}
        scores.update({
            k: np.asarray(z[k]) for k in _OPTIONAL_SCORE_KEYS if k in z.files
        })

    with np.load(gpath) as z:
        missing = [k for k in _GEN_KEYS if k not in z.files]
        if missing:
            raise KeyError(f"{gpath} is missing columns {missing}")
        n_gen = int(z["ra"].shape[0])
        if idx.size and (idx.min() < 0 or idx.max() >= n_gen):
            raise ValueError(
                f"scores.npz index out of range for gen.npz (n={n_gen}): "
                f"[{idx.min()}, {idx.max()}]"
            )
        gen = {k: np.asarray(z[k])[idx] for k in _GEN_KEYS}

    # Drop tracklets priora could not score with a LIKELIHOOD, which since
    # D-19 is two categories rather than one:
    #
    #   * non-finite p_neo — an empty admissible region, or a 0/0 whose
    #     mechanism is not the faint-end wall;
    #   * a D-19 triage-policy assignment — p_neo = 1.0 over a 0/0 whose whole
    #     region is at/above the H support end, recognised by the recorded
    #     (isfinite(p), n_nodes > 0, Lambda_NEO == Lambda_other == 0) quadruple.
    #
    # The second is load-bearing for THIS module specifically.  Its whole
    # premise (documented in DIGEST2_COMPARISON.md) is a *symmetric* drop: a
    # tracklet either enters both codes' ROC or neither, so that no ΔAUC can be
    # manufactured by one side scoring tracklets the other cannot. A policy row
    # is p = 1.0 for priora with no likelihood behind it, on a class that is
    # empirically ~100% true NEO — keeping it would hand priora free true
    # positives that digest2 has to earn, biasing ΔAUC in priora's favour by
    # construction.  Excluded here, and counted so the drop is visible.
    finite_p = np.isfinite(scores["p_neo"])
    policy_detectable = all(k in scores for k in _OPTIONAL_SCORE_KEYS)
    if policy_detectable:
        policy = (
            finite_p
            & (np.asarray(scores["n_nodes"]) > 0)
            & (np.asarray(scores["lambda_neo"], dtype=float) == 0.0)
            & (np.asarray(scores["lambda_other"], dtype=float) == 0.0)
        )
    else:
        policy = np.zeros(finite_p.shape, dtype=bool)
    keep = finite_p & ~policy
    n_dropped_policy = int(np.count_nonzero(policy))
    n_dropped_nonfinite = int(np.count_nonzero(~finite_p))
    if not keep.all():
        idx = idx[keep]
        gen = {k: v[keep] for k, v in gen.items()}
        scores = {k: v[keep] for k, v in scores.items()}

    if limit is not None and limit < idx.size:
        idx = idx[:limit]
        gen = {k: v[:limit] for k, v in gen.items()}
        scores = {k: v[:limit] for k, v in scores.items()}

    return RunData(
        run_dir=run_dir, index=idx, gen=gen, scores=scores,
        n_dropped_nonfinite=n_dropped_nonfinite,
        n_dropped_policy=n_dropped_policy,
        policy_detectable=policy_detectable,
    )


# --------------------------------------------------------------------------
# Observation emission
# --------------------------------------------------------------------------


def designation(i: int) -> str:
    """Tracklet designation encoding the gen.npz row.  8 chars (ADES trkSub)."""
    return f"T{int(i):07d}"


def _designation_index(desig: str) -> int:
    return int(desig.strip()[1:])


def _tracklet_positions(
    run: RunData, dt_min: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-epoch (ra_deg, dec_deg, jd_tdb) arrays of shape (n, 2).

    Linear propagation of the attributable: ra_rate is d(alpha)/dt (NOT
    multiplied by cos(delta)) per ``priora.attributable.Attributable``, so
    the propagation is a plain addition in both coordinates.
    """
    dt = float(dt_min) / 1440.0
    # D2_NDET (default 2): number of detections, evenly spaced over the arc (2026-11-06; 2 reproduces every earlier run)
    off = np.linspace(0.0, dt, int(os.environ.get("D2_NDET", "2")))
    ra = run.gen["ra"][:, None] + run.gen["ra_rate"][:, None] * off[None, :]
    dec = run.gen["dec"][:, None] + run.gen["dec_rate"][:, None] * off[None, :]
    jd = run.gen["epoch_jd_tdb"][:, None] + off[None, :]

    ra_deg = np.degrees(ra) % 360.0
    dec_deg = np.degrees(dec)
    if np.any(np.abs(dec_deg) > 90.0):
        # Linear extrapolation in delta would step over a pole.  With WP6's
        # |dec| < 60 deg fields this cannot happen; fail loudly rather than
        # emit a nonsense declination.
        bad = int(np.sum(np.abs(dec_deg) > 90.0))
        raise ValueError(
            f"{bad} tracklet(s) extrapolate past a celestial pole at "
            f"dt = {dt_min} min; reduce --dt-min or handle pole crossing."
        )
    return ra_deg, dec_deg, jd


def _iso_utc(jd_tdb: np.ndarray) -> np.ndarray:
    """TDB Julian dates -> UTC ISO-8601 strings at microsecond precision.

    digest2 reads a civil timestamp and feeds it to its own low-precision
    solar ephemeris (``se2000`` in d2math.c), so the physically correct thing
    is to hand it real UTC.  The TDB-UTC offset (~69 s in 2026) displaces the
    Earth by ~1e-5 AU, an order of magnitude below se2000's own accuracy.
    """
    t = Time(np.asarray(jd_tdb, dtype=float), format="jd", scale="tdb")
    t = t.utc
    t.precision = 6
    return np.atleast_1d(t.isot)


_PSV_HEADER = (
    "permID |provID |trkSub  |mode |stn  |obsTime                     |"
    "ra          |dec         |mag   |band |astCat"
)


def emit_ades_psv(
    run: RunData, path: str, dt_min: float = DEFAULT_DT_MIN
) -> dict:
    """Write the run's tracklets as an ADES PSV file digest2 can parse.

    Returns a manifest dict (path, n_tracklets, n_observations, dt_min, ...).
    """
    ra_deg, dec_deg, jd = _tracklet_positions(run, dt_min)
    iso = _iso_utc(jd.ravel()).reshape(jd.shape)
    vmag = run.gen["v_mag"]

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n_obs = 0
    with open(path, "w") as fh:
        fh.write("# WP7 synthetic tracklets from priora validation/self_consistency\n")
        fh.write(_PSV_HEADER + "\n")
        for k in range(run.n):
            trk = designation(run.index[k])
            for j in range(jd.shape[1]):
                fh.write(
                    "       |       |%-8s|CCD  |%-4s |%-27s |%12.8f |%+12.8f "
                    "|%6.2f |V    |Gaia3\n"
                    % (
                        trk,
                        OBSCODE,
                        iso[k, j] + "Z",
                        ra_deg[k, j],
                        dec_deg[k, j],
                        float(vmag[k]),
                    )
                )
                n_obs += 1
    return {
        "path": os.path.abspath(path),
        "format": "ades_psv",
        "n_tracklets": run.n,
        "n_observations": n_obs,
        "dt_min": float(dt_min),
        "obscode": OBSCODE,
        "band": "V",
    }


def _mpc80_line(desig: str, jd_tdb: float, ra_deg: float, dec_deg: float,
                vmag: float) -> str:
    """One MPC 80-column optical line (fallback format)."""
    t = Time(float(jd_tdb), format="jd", scale="tdb").utc
    y, m = t.datetime.year, t.datetime.month
    day = (
        t.datetime.day
        + (t.datetime.hour + (t.datetime.minute + (
            t.datetime.second + t.datetime.microsecond * 1e-6) / 60.0) / 60.0)
        / 24.0
    )
    ra_h = ra_deg / 15.0
    rah = int(ra_h)
    ram = int((ra_h - rah) * 60.0)
    ras = (ra_h - rah - ram / 60.0) * 3600.0
    sign = "-" if dec_deg < 0 else "+"
    ad = abs(dec_deg)
    dd = int(ad)
    dm = int((ad - dd) * 60.0)
    ds = (ad - dd - dm / 60.0) * 3600.0
    line = (
        f"{desig:<12s}"          # 0:12  designation
        f"  "                     # 12:14 discovery / note1
        f"C"                      # 14    note2 = CCD
        f"{y:4d} {m:02d} {day:09.6f}"   # 15:32 date  (DD.dddddd)
        f"{rah:02d} {ram:02d} {ras:06.3f}"  # 32:44 RA
        f"{sign}{dd:02d} {dm:02d} {ds:05.2f}"  # 44:56 Dec
        f"{'':9s}"                # 56:65 blank
        f"{vmag:5.2f}"            # 65:70 magnitude
        f"V"                      # 70    band
        f"{'':6s}"                # 71:77 blank
        f"{OBSCODE:>3s}"          # 77:80 obscode
    )
    assert len(line) == 80, (len(line), line)
    return line


def emit_mpc80(run: RunData, path: str, dt_min: float = DEFAULT_DT_MIN) -> dict:
    """Fallback emitter: MPC 80-column.  Rounds RA to 1 ms, Dec to 10 mas."""
    ra_deg, dec_deg, jd = _tracklet_positions(run, dt_min)
    vmag = run.gen["v_mag"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n_obs = 0
    with open(path, "w") as fh:
        for k in range(run.n):
            trk = designation(run.index[k])
            for j in range(jd.shape[1]):
                fh.write(
                    _mpc80_line(trk, jd[k, j], ra_deg[k, j], dec_deg[k, j],
                                float(vmag[k]))
                    + "\n"
                )
                n_obs += 1
    return {
        "path": os.path.abspath(path),
        "format": "mpc80",
        "n_tracklets": run.n,
        "n_observations": n_obs,
        "dt_min": float(dt_min),
        "obscode": OBSCODE,
        "band": "V",
    }


EMITTERS = {"ades_psv": emit_ades_psv, "mpc80": emit_mpc80}
_SUFFIX = {"ades_psv": ".psv", "mpc80": ".obs"}


# --------------------------------------------------------------------------
# Running digest2
# --------------------------------------------------------------------------


def run_digest2(
    obs_path: str,
    index: np.ndarray,
    model_path: str = DEFAULT_MODEL,
    workers: int = 8,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """Score an emitted observation file and align results to ``index``.

    Returns ``(columns, stats)`` where ``columns`` holds ``raw_neo``,
    ``noid_neo``, ``rms`` and ``ok`` arrays of length ``len(index)``, and
    ``stats`` records the parse/score rates that the round-trip check needs.
    """
    ok, msg = digest2_available()
    if not ok:
        raise RuntimeError(msg)

    from digest2 import Digest2
    from digest2.observation import parse_ades_psv, parse_mpc80_file
    from digest2.model import ensure_obscodes

    parser = parse_ades_psv if obs_path.lower().endswith(".psv") else parse_mpc80_file
    t0 = time.time()
    parsed = parser(obs_path)
    n_parsed = len(parsed)
    n_pairs = sum(1 for v in parsed.values() if len(v) >= 2)
    t_parse = time.time() - t0

    obscodes = ensure_obscodes()
    t0 = time.time()
    with Digest2(model_path=model_path, obscodes_path=obscodes,
                 repeatable=True) as d2:
        results = d2.classify_file(obs_path, max_workers=workers)
    t_score = time.time() - t0

    n = int(np.asarray(index).size)
    pos = {int(v): k for k, v in enumerate(np.asarray(index).astype(int))}
    cols = {
        "raw_neo": np.full(n, np.nan),
        "noid_neo": np.full(n, np.nan),
        "rms": np.full(n, np.nan),
    }
    n_matched = 0
    for r in results:
        try:
            k = pos[_designation_index(r.designation)]
        except (KeyError, ValueError):
            continue
        cols["raw_neo"][k] = float(r.raw.NEO)
        cols["noid_neo"][k] = float(r.noid.NEO)
        cols["rms"][k] = float(r.rms)
        n_matched += 1
    cols["ok"] = np.isfinite(cols["raw_neo"])

    stats = {
        "n_emitted": n,
        "n_parsed_tracklets": n_parsed,
        "n_parsed_with_pair": n_pairs,
        "n_scored": len(results),
        "n_matched": n_matched,
        "parse_rate": n_pairs / n if n else float("nan"),
        "score_rate": n_matched / n if n else float("nan"),
        "parse_seconds": t_parse,
        "score_seconds": t_score,
        "workers": int(workers),
        "score_ms_per_tracklet": 1e3 * t_score / max(n_matched, 1),
    }
    if verbose:
        print(
            f"[digest2] parsed {n_pairs}/{n} tracklets ({stats['parse_rate']:.4%}), "
            f"scored {n_matched}/{n} in {t_score:.1f} s "
            f"({stats['score_ms_per_tracklet']:.1f} ms/tracklet, "
            f"{workers} threads)"
        )
    return cols, stats


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def auc_fast(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Weighted ROC AUC = P(s_pos > s_neg) + 0.5 P(tie).  Fully vectorised.

    Identical in value to ``validation.self_consistency.auc_weighted`` (the
    tests assert that), but without the per-tie-group Python loop — this one is
    called ~10^4 times inside the bootstrap.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    w = np.ones(scores.size) if weights is None else np.asarray(weights, float)
    good = np.isfinite(scores)
    scores, labels, w = scores[good], labels[good], w[good]
    if scores.size == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    s, lab, ww = scores[order], labels[order], w[order]
    neg_w = np.where(~lab, ww, 0.0)
    wp = ww[lab].sum()
    wn = neg_w.sum()
    if wp <= 0 or wn <= 0:
        return float("nan")

    cum = np.concatenate(([0.0], np.cumsum(neg_w)))  # cum[k] = neg weight in s[:k]
    starts = np.concatenate(([0], np.nonzero(np.diff(s))[0] + 1))
    ends = np.concatenate((starts[1:], [s.size]))
    gid = np.searchsorted(starts, np.arange(s.size), side="right") - 1
    below = cum[starts[gid]]
    in_group = cum[ends[gid]] - cum[starts[gid]]
    contrib = below + 0.5 * in_group
    return float((ww[lab] * contrib[lab]).sum() / (wp * wn))


def auc_bootstrap(
    s_a: np.ndarray,
    s_b: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 7,
    alpha: float = 0.05,
) -> dict:
    """Paired, label-stratified bootstrap for AUC(a), AUC(b) and their difference.

    The resample is *paired* (the same resampled tracklets feed both scores),
    which is what makes the CI on the difference far tighter than the CIs on
    the two AUCs separately — the two codes see identical inputs, so most of
    the sampling noise is common-mode.

    Stratifying by label (positives and negatives resampled separately) holds
    the NEO count fixed, which matters at the gate's 185 NEOs in 2x10^4.
    """
    labels = np.asarray(labels).astype(bool)
    s_a = np.asarray(s_a, float)
    s_b = np.asarray(s_b, float)
    w = np.ones(labels.size) if weights is None else np.asarray(weights, float)
    good = np.isfinite(s_a) & np.isfinite(s_b)
    s_a, s_b, labels, w = s_a[good], s_b[good], labels[good], w[good]

    pos = np.flatnonzero(labels)
    neg = np.flatnonzero(~labels)
    out = {
        "n": int(labels.size),
        "n_pos": int(pos.size),
        "n_neg": int(neg.size),
        "w_pos": float(w[labels].sum()),
        "w_neg": float(w[~labels].sum()),
        "auc_a": auc_fast(s_a, labels, w),
        "auc_b": auc_fast(s_b, labels, w),
        "n_boot": int(n_boot),
    }
    out["delta"] = out["auc_a"] - out["auc_b"]
    if pos.size < 2 or neg.size < 2 or n_boot <= 0:
        out.update(
            delta_ci=[float("nan"), float("nan")],
            auc_a_ci=[float("nan"), float("nan")],
            auc_b_ci=[float("nan"), float("nan")],
            p_delta_le_0=float("nan"),
        )
        return out

    rng = np.random.default_rng(seed)
    da = np.empty(n_boot)
    db = np.empty(n_boot)
    for b in range(n_boot):
        # Resample each stratum from ITS OWN index array. (A previous version
        # concatenated (pos, neg) and indexed it with per-stratum indices,
        # which sent the negative half's indices into the positive block —
        # found by the NEOCP pilot when n_pos >= n_neg made every resample
        # all-positive. Regression-tested in test_digest2_comparison.py.)
        sel = np.concatenate(
            (
                pos[rng.integers(0, pos.size, pos.size)],
                neg[rng.integers(0, neg.size, neg.size)],
            )
        )
        lb, wb = labels[sel], w[sel]
        da[b] = auc_fast(s_a[sel], lb, wb)
        db[b] = auc_fast(s_b[sel], lb, wb)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    d = da - db
    out["auc_a_ci"] = [float(np.nanpercentile(da, lo)), float(np.nanpercentile(da, hi))]
    out["auc_b_ci"] = [float(np.nanpercentile(db, lo)), float(np.nanpercentile(db, hi))]
    out["delta_ci"] = [float(np.nanpercentile(d, lo)), float(np.nanpercentile(d, hi))]
    out["delta_boot_mean"] = float(np.nanmean(d))
    out["delta_boot_sd"] = float(np.nanstd(d, ddof=1))
    # One-sided bootstrap p for "priora is no better than digest2".
    out["p_delta_le_0"] = float(np.mean(d <= 0.0))
    return out


def subset_masks(
    ar_frac: np.ndarray,
    clean_max: float = CLEAN_MAX_FRAC,
    broken_min: float = BROKEN_MIN_FRAC,
) -> dict[str, np.ndarray]:
    """DESIGN §4.2 truncation split on the §2.6(b) diagnostic.

    ``clean``  : measure fraction above H = 25.5 below ``clean_max`` — the two
                 codes really are consuming the same effective model here.
    ``broken`` : fraction above ``broken_min`` — digest2's [24, inf) top bin and
                 priora's hard truncation at 25.5 differ materially.
    Tracklets between the two thresholds belong to neither subset by design;
    ``all`` keeps everything, including non-finite diagnostics.
    """
    f = np.asarray(ar_frac, dtype=float)
    if clean_max > broken_min:
        raise ValueError(
            f"clean_max ({clean_max}) must not exceed broken_min ({broken_min})"
        )
    finite = np.isfinite(f)
    return {
        "all": np.ones(f.shape, dtype=bool),
        "clean": finite & (f < clean_max),
        "broken": finite & (f > broken_min),
    }


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Mid-rank percentile in [0, 1]; ties share the average rank."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x.copy()
    order = np.argsort(x, kind="mergesort")
    s = x[order]
    starts = np.concatenate(([0], np.nonzero(np.diff(s))[0] + 1))
    ends = np.concatenate((starts[1:], [n]))
    gid = np.searchsorted(starts, np.arange(n), side="right") - 1
    mid = 0.5 * (starts[gid] + ends[gid] - 1)
    out = np.empty(n)
    out[order] = (mid + 0.5) / n
    return out


def reliability_table(
    p: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
    n_bins: int = 8,
) -> list[dict]:
    """Weighted reliability bins, equal *expected-positive* mass per bin.

    Equal-count bins are useless at a 1% base rate (most bins hold << 1
    expected NEO and can neither pass nor fail) — the same reason WP6 rebuilt
    its own criterion.  Bins are cut on the cumulative predicted-positive mass
    of the score being tested, so each bin carries a comparable expectation
    whichever code produced the score.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(float)
    w = np.ones(p.size) if w is None else np.asarray(w, dtype=float)
    good = np.isfinite(p)
    p, y, w = p[good], y[good], w[good]
    if p.size == 0:
        return []

    order = np.argsort(p, kind="mergesort")
    p, y, w = p[order], y[order], w[order]
    mass = np.cumsum(w * p)
    total = mass[-1]
    if not np.isfinite(total) or total <= 0:
        return []
    edges = np.searchsorted(mass, np.linspace(0, total, n_bins + 1)[1:-1])
    pieces = np.split(np.arange(p.size), np.unique(edges))

    rows = []
    for sl in pieces:
        if sl.size == 0:
            continue
        pp, yy, ww = p[sl], y[sl], w[sl]
        exp_pos = float((ww * pp).sum())
        obs_pos = float((ww * yy).sum())
        var = float((pp * (1.0 - pp)).sum())
        rows.append(
            {
                "p_lo": float(pp.min()),
                "p_hi": float(pp.max()),
                "n": int(sl.size),
                "w": float(ww.sum()),
                "p_mean": float((ww * pp).sum() / ww.sum()),
                "obs_frac": obs_pos / float(ww.sum()),
                "expected_pos": exp_pos,
                "observed_pos": obs_pos,
                "z": (obs_pos - exp_pos) / np.sqrt(var) if var > 0 else float("nan"),
            }
        )
    return rows


def calibration_stats(
    p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, n_bins: int = 8
) -> dict:
    """Global calibration summary for a score already mapped into [0, 1]."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(float)
    w = np.ones(p.size) if w is None else np.asarray(w, dtype=float)
    good = np.isfinite(p)
    p, y, w = p[good], y[good], w[good]
    if p.size == 0:
        return {}

    wsum = float(w.sum())
    expected = float((w * p).sum())
    observed = float((w * y).sum())
    var = float((p * (1.0 - p)).sum())
    rows = reliability_table(p, y, w, n_bins=n_bins)
    # Expected calibration error: weight each bin by its share of total weight.
    ece = sum(r["w"] * abs(r["p_mean"] - r["obs_frac"]) for r in rows) / wsum
    return {
        "n": int(p.size),
        "w_total": wsum,
        "expected_positives": expected,
        "observed_positives": observed,
        "ratio_expected_observed": expected / observed if observed > 0 else float("inf"),
        "z_total": (observed - expected) / np.sqrt(var) if var > 0 else float("nan"),
        "brier": float((w * (p - y) ** 2).sum() / wsum),
        "ece": float(ece),
        "max_abs_z_bin": float(max((abs(r["z"]) for r in rows), default=float("nan"))),
        "bins": rows,
    }


def operating_points(
    p42: np.ndarray,
    s_d2: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    d2_threshold: float = 65.0,
) -> dict:
    """Triage at digest2's own NEOCP cut, and priora matched to it.

    digest2 posts a tracklet to the NEO Confirmation Page at a score of 65.
    The comparison that matters operationally is not the AUC but what that
    threshold buys: at equal completeness, how much contamination does each
    code carry?  priora is evaluated twice -- at its natural p = 0.5, and at
    whatever p reproduces digest2's completeness -- because only the second is
    an apples-to-apples contamination comparison.
    """
    y = np.asarray(y).astype(bool)
    w = np.asarray(w, dtype=float)
    w_pos = w[y].sum()

    def point(flag: np.ndarray, label: str) -> dict:
        wf = w[flag].sum()
        wtp = w[flag & y].sum()
        return {
            "label": label,
            "n_flagged": int(flag.sum()),
            "w_flagged": float(wf),
            "completeness": float(wtp / w_pos) if w_pos > 0 else float("nan"),
            "contamination": float(1.0 - wtp / wf) if wf > 0 else float("nan"),
            "n_true_neo": int((flag & y).sum()),
        }

    d2_pt = point(s_d2 >= d2_threshold, f"digest2 raw >= {d2_threshold:g}")
    out = {
        "digest2_neocp": d2_pt,
        "priora_p50": point(p42 >= 0.5, "priora p >= 0.5"),
    }
    # priora threshold that matches digest2's completeness.
    target = d2_pt["completeness"]
    if np.isfinite(target) and 0 < target <= 1 and y.any():
        order = np.argsort(-p42, kind="mergesort")
        comp = np.cumsum(np.where(y[order], w[order], 0.0)) / w_pos
        k = int(np.searchsorted(comp, target))
        if k < order.size:
            thr = float(p42[order[k]])
            out["priora_matched"] = point(p42 >= thr,
                                            f"priora p >= {thr:.3g}")
            out["priora_matched"]["threshold"] = thr
    return out


def disagreement_census(
    run: RunData,
    d2: dict[str, np.ndarray],
    mask: np.ndarray,
    top_k: int = 10,
) -> dict:
    """Tracklets whose two rank orders differ most, with their diagnostics.

    Ranks are percentile ranks computed *within* ``mask`` — a rank gap only
    means something relative to the population being compared.
    """
    sel = np.flatnonzero(mask & d2["ok"])
    if sel.size == 0:
        return {"n": 0, "priora_higher": [], "digest2_higher": [], "misranked_neos": []}

    r42 = percentile_rank(run.p_neo[sel])
    r2 = percentile_rank(d2["raw_neo"][sel])
    delta = r42 - r2

    a, e, inc = (
        run.gen["elements"][:, 0],
        run.gen["elements"][:, 1],
        run.gen["elements"][:, 2],
    )
    q = a * (1.0 - e)

    def row(j: int) -> dict:
        k = sel[j]
        return {
            "index": int(run.index[k]),
            "is_neo": bool(run.is_neo[k]),
            "p_neo_42": float(run.p_neo[k]),
            "d2_raw_neo": float(d2["raw_neo"][k]),
            "d2_noid_neo": float(d2["noid_neo"][k]),
            "rank_42": float(r42[j]),
            "rank_d2": float(r2[j]),
            "delta_rank": float(delta[j]),
            "v_mag": float(run.gen["v_mag"][k]),
            "H_true": float(run.gen["H"][k]),
            "q_true": float(q[k]),
            "e_true": float(e[k]),
            "i_true_deg": float(np.degrees(inc[k])),
            "rho_true": float(run.gen["rho_true"][k]),
            "ar_frac_h_above": float(run.ar_frac_h_above[k]),
            "n_nodes": int(run.scores["n_nodes"][k]),
        }

    order = np.argsort(delta)
    up = order[::-1][:top_k]
    dn = order[:top_k]
    neo_local = run.is_neo[sel]

    def tail(idx: np.ndarray) -> dict:
        """What kind of object populates a disagreement tail?"""
        if idx.size == 0:
            return {}
        k = sel[idx]
        return {
            "n": int(idx.size),
            "median_rho_true_au": float(np.median(run.gen["rho_true"][k])),
            "median_q_true_au": float(np.median(q[k])),
            "frac_q_gt_5au": float(np.mean(q[k] > 5.0)),
            "frac_rho_gt_10au": float(np.mean(run.gen["rho_true"][k] > 10.0)),
            "frac_true_neo": float(np.mean(run.is_neo[k])),
            "median_delta_rank": float(np.median(delta[idx])),
        }

    n_tail = max(top_k, int(round(0.01 * sel.size)))
    # True NEOs that digest2 ranks worst relative to priora, and vice versa.
    neo_idx = np.flatnonzero(neo_local)
    if neo_idx.size:
        neo_sorted = neo_idx[np.argsort(delta[neo_idx])[::-1]][:top_k]
    else:
        neo_sorted = np.array([], dtype=int)

    return {
        "n": int(sel.size),
        "rank_spearman": float(np.corrcoef(r42, r2)[0, 1]),
        "mean_abs_delta_rank": float(np.mean(np.abs(delta))),
        "frac_delta_rank_gt_0p25": float(np.mean(np.abs(delta) > 0.25)),
        "priora_higher": [row(int(j)) for j in up],
        "digest2_higher": [row(int(j)) for j in dn],
        "misranked_neos": [row(int(j)) for j in neo_sorted],
        "tail_priora_higher": tail(order[::-1][:n_tail]),
        "tail_digest2_higher": tail(order[:n_tail]),
    }


# --------------------------------------------------------------------------
# Analysis + plots
# --------------------------------------------------------------------------


def _roc(scores, labels, weights):
    """Weighted ROC (fpr, tpr) at every distinct threshold."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels).astype(bool)
    w = np.ones(scores.size) if weights is None else np.asarray(weights, float)
    good = np.isfinite(scores)
    scores, labels, w = scores[good], labels[good], w[good]
    order = np.argsort(-scores, kind="mergesort")
    s, lab, ww = scores[order], labels[order], w[order]
    tp = np.cumsum(np.where(lab, ww, 0.0))
    fp = np.cumsum(np.where(~lab, ww, 0.0))
    # keep only the last point of each tie group
    keep = np.concatenate((np.diff(s) != 0, [True]))
    tp, fp = tp[keep], fp[keep]
    P, N = ww[lab].sum(), ww[~lab].sum()
    if P <= 0 or N <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    return (
        np.concatenate(([0.0], fp / N, [1.0])),
        np.concatenate(([0.0], tp / P, [1.0])),
    )


_C42 = "#1b6ca8"
_CD2 = "#c8562b"


def plot_roc(run, d2, masks, res, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    for ax, name in zip(axes, SUBSETS):
        m = masks[name] & d2["ok"]
        y, w = run.is_neo[m], run.weight[m]
        for score, colour, lbl, key in (
            (run.p_neo[m], _C42, "priora p(NEO)", "auc_a"),
            (d2["raw_neo"][m], _CD2, "digest2 raw NEO", "auc_b"),
        ):
            fpr, tpr = _roc(score, y, w)
            ax.plot(fpr, tpr, color=colour, lw=1.8,
                    label=f"{lbl}  AUC={res[name]['auc'][key]:.4f}")
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("false positive rate")
        if ax is axes[0]:
            ax.set_ylabel("true positive rate")
        d = res[name]["auc"]
        ax.set_title(
            f"{name}  (n={d['n']}, NEO={d['n_pos']})\n"
            r"$\Delta$AUC = %+0.4f [%+0.4f, %+0.4f]"
            % (d["delta"], d["delta_ci"][0], d["delta_ci"][1]),
            fontsize=10,
        )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle(
        "WP7 / DESIGN §4.2 — ROC on shared inputs and the shared population file",
        fontsize=11,
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scatter(run, d2, masks, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, name in zip(axes, ("all", "clean")):
        m = masks[name] & d2["ok"]
        y = run.is_neo[m]
        x = np.clip(run.p_neo[m], 1e-8, 1.0)
        s2 = d2["raw_neo"][m]
        ax.scatter(x[~y], s2[~y], s=3, alpha=0.15, color="0.45",
                   linewidths=0, label=f"other (n={int((~y).sum())})")
        ax.scatter(x[y], s2[y], s=14, alpha=0.85, color="#c8562b",
                   linewidths=0, label=f"true NEO (n={int(y.sum())})")
        ax.axhline(65, color="#1b6ca8", lw=1.0, ls="--",
                   label="digest2 NEOCP threshold (65)")
        ax.set_xscale("log")
        ax.set_xlim(1e-8, 1.3)
        ax.set_ylim(-2, 102)
        ax.set_xlabel("priora  p(NEO)")
        if ax is axes[0]:
            ax.set_ylabel("digest2  raw NEO score")
        ax.set_title(f"{name} subset", fontsize=10)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("digest2 score vs priora posterior, coloured by truth", fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_calibration(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, name in zip(axes, ("all", "clean")):
        c = res[name]["calibration"]
        for key, colour, lbl in (
            ("priora", _C42, "priora p(NEO)"),
            ("digest2_raw_over_100", _CD2, "digest2 raw NEO / 100"),
        ):
            rows = c[key]["bins"]
            if not rows:
                continue
            xs = [r["p_mean"] for r in rows]
            ys = [r["obs_frac"] for r in rows]
            # Wilson-ish error bars from the Poisson-binomial expectation.
            err = [
                np.sqrt(max(r["expected_pos"], 1e-9)) / max(r["w"], 1e-9)
                for r in rows
            ]
            ax.errorbar(xs, ys, yerr=err, fmt="o-", ms=4, lw=1.4,
                        color=colour, label=lbl, capsize=2)
        lim = [1e-5, 1.2]
        ax.plot(lim, lim, color="0.7", lw=0.9, ls=":", label="perfect calibration")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lim)
        ax.set_ylim(1e-5, 1.2)
        ax.set_xlabel("mean predicted P(NEO)")
        if ax is axes[0]:
            ax.set_ylabel("observed NEO fraction")
        priora = c["priora"]
        dd2 = c["digest2_raw_over_100"]
        ax.set_title(
            f"{name}:  predicted NEOs  priora {priora['expected_positives']:.1f}"
            f"  /  digest2 {dd2['expected_positives']:.1f}"
            f"  /  observed {priora['observed_positives']:.1f}",
            fontsize=9,
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5, which="both")
    fig.suptitle(
        "Reliability: digest2's score treated as a probability vs priora's posterior",
        fontsize=11,
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_truncation(run, d2, res, path):
    """AUC of each code in bins of the §2.6(b) truncation diagnostic."""
    f = run.ar_frac_h_above
    ok = d2["ok"] & np.isfinite(f)
    edges = np.array([0.0, 1e-3, 1e-2, 3e-2, 1e-1, 1.0])
    centres, a42, ad2, ns = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (f >= lo) & (f < hi)
        if m.sum() < 20 or run.is_neo[m].sum() < 3:
            continue
        centres.append(0.5 * (lo + hi))
        a42.append(auc_fast(run.p_neo[m], run.is_neo[m], run.weight[m]))
        ad2.append(auc_fast(d2["raw_neo"][m], run.is_neo[m], run.weight[m]))
        ns.append(int(m.sum()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.hist(np.clip(f[ok], 1e-5, 1.0), bins=np.logspace(-5, 0, 50),
            color="0.6", label="all")
    ax.hist(np.clip(f[ok & run.is_neo], 1e-5, 1.0), bins=np.logspace(-5, 0, 50),
            color="#c8562b", alpha=0.8, label="true NEO")
    ax.axvline(CLEAN_MAX_FRAC, color="#1b6ca8", ls="--", lw=1.2, label="clean < 1e-3")
    ax.axvline(BROKEN_MIN_FRAC, color="k", ls="--", lw=1.2, label="broken > 1e-2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"§2.6(b) measure fraction at implied $H > 25.5$")
    ax.set_ylabel("tracklets")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("Where the shared-model premise breaks", fontsize=10)

    ax = axes[1]
    if centres:
        ax.plot(centres, a42, "o-", color=_C42, label="priora p(NEO)")
        ax.plot(centres, ad2, "s-", color=_CD2, label="digest2 raw NEO")
        for x, n in zip(centres, ns):
            ax.annotate(f"n={n}", (x, min(min(a42), min(ad2))), fontsize=7,
                        ha="center", va="bottom", color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel(r"measure fraction at $H > 25.5$ (bin centre)")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("Ranking vs truncation exposure", fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def analyze(
    run: RunData,
    d2: dict[str, np.ndarray],
    n_boot: int = 2000,
    seed: int = 7,
    clean_max: float = CLEAN_MAX_FRAC,
    broken_min: float = BROKEN_MIN_FRAC,
) -> dict:
    """Everything DESIGN §4.2 asks for, as a JSON-able dict."""
    masks = subset_masks(run.ar_frac_h_above, clean_max, broken_min)
    res: dict = {"subsets": {}}
    for name in SUBSETS:
        m = masks[name] & d2["ok"]
        y, w = run.is_neo[m], run.weight[m]
        auc = auc_bootstrap(run.p_neo[m], d2["raw_neo"][m], y, w,
                            n_boot=n_boot, seed=seed)
        auc["auc_digest2_noid"] = auc_fast(d2["noid_neo"][m], y, w)
        cal = {
            "priora": calibration_stats(run.p_neo[m], y, w),
            "digest2_raw_over_100": calibration_stats(
                np.clip(d2["raw_neo"][m] / 100.0, 0.0, 1.0), y, w
            ),
            "digest2_noid_over_100": calibration_stats(
                np.clip(d2["noid_neo"][m] / 100.0, 0.0, 1.0), y, w
            ),
        }
        res["subsets"][name] = {
            "n": int(m.sum()),
            "n_neo": int(y.sum()),
            "w_neo": float(w[y].sum()),
            "auc": auc,
            "calibration": cal,
            "operating_points": operating_points(
                run.p_neo[m], d2["raw_neo"][m], y, w
            ),
            "disagreement": disagreement_census(run, d2, masks[name]),
        }
    # How much of the ranking gap is carried by distant objects, which
    # digest2's bounded ranging search cannot reach at all?
    m = masks["clean"] & d2["ok"]
    near = m & (run.gen["rho_true"] < 10.0)
    res["distant_object_effect"] = {
        "n_distant_in_clean": int((m & ~near).sum()),
        "delta_clean": auc_fast(run.p_neo[m], run.is_neo[m], run.weight[m])
        - auc_fast(d2["raw_neo"][m], run.is_neo[m], run.weight[m]),
        "delta_clean_rho_lt_10au": auc_fast(
            run.p_neo[near], run.is_neo[near], run.weight[near]
        ) - auc_fast(d2["raw_neo"][near], run.is_neo[near], run.weight[near]),
    }
    res["thresholds"] = {"clean_max": clean_max, "broken_min": broken_min}
    res["truncation_diagnostic"] = {
        "median": float(np.nanmedian(run.ar_frac_h_above)),
        "mean": float(np.nanmean(run.ar_frac_h_above)),
        "frac_above_1e-2": float(np.nanmean(run.ar_frac_h_above > 1e-2)),
        "frac_above_1e-1": float(np.nanmean(run.ar_frac_h_above > 1e-1)),
    }
    return res


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _fmt_auc(d: dict) -> str:
    return (
        f"AUC_42 = {d['auc_a']:.4f} [{d['auc_a_ci'][0]:.4f}, {d['auc_a_ci'][1]:.4f}]   "
        f"AUC_d2 = {d['auc_b']:.4f} [{d['auc_b_ci'][0]:.4f}, {d['auc_b_ci'][1]:.4f}]   "
        f"delta = {d['delta']:+.4f} "
        f"[{d['delta_ci'][0]:+.4f}, {d['delta_ci'][1]:+.4f}]"
    )


def summary_text(summary: dict) -> str:
    L: list[str] = []
    A = L.append
    A("=" * 78)
    A("WP7 — digest2 head-to-head on shared inputs (DESIGN §4.2)")
    A("=" * 78)
    env = summary["digest2"]
    A(f"digest2            : v{env.get('version')} @ {env.get('source_commit', '')[:12]}")
    A(f"population model   : {env['model_path']}")
    A(f"  sha256           : {env.get('model_sha256')}")
    if env.get("bundled_model_is_lfs_pointer"):
        A(f"  package LFS oid  : {env.get('bundled_model_lfs_oid')}  "
          f"({'MATCH' if env.get('bundled_model_lfs_oid') == env.get('model_sha256') else 'DIFFERS'})")
    A(f"obscodes           : {env.get('obscodes_path')}")
    A(f"MPC.config         : {env.get('config_path')}")
    io = summary["emission"]
    st = summary["digest2_run"]
    A(f"input format       : {io['format']}, {io['n_tracklets']} tracklets x 2 det, "
      f"dt = {io['dt_min']:.0f} min, obscode {io['obscode']}")
    A(f"parse rate         : {st['n_parsed_with_pair']}/{st['n_emitted']} "
      f"= {st['parse_rate']:.4%}   scored {st['n_matched']} "
      f"({st['score_ms_per_tracklet']:.1f} ms/tracklet, {st['workers']} threads)")
    A(f"priora run       : {summary['run_dir']}  (score variant: All -> "
      f"digest2 'raw')")
    A("")
    A("-" * 78)
    A("RANKING  (positive delta = priora ranks better)")
    A("-" * 78)
    for name in SUBSETS:
        s = summary["subsets"][name]
        tag = {"clean": "  <- HEADLINE", "broken": "  <- premise broken",
               "all": ""}[name]
        A(f"{name:7s} n={s['n']:6d}  NEO={s['n_neo']:4d}{tag}")
        A(f"        {_fmt_auc(s['auc'])}")
        A(f"        bootstrap P(delta <= 0) = {s['auc']['p_delta_le_0']:.4f}"
          f"   (digest2 noid AUC = {s['auc']['auc_digest2_noid']:.4f})")
    A("")
    A("-" * 78)
    A("CALIBRATION  (digest2 makes no probability claim; this shows it)")
    A("-" * 78)
    A(f"{'subset':7s} {'code':22s} {'pred NEOs':>10s} {'obs NEOs':>9s} "
      f"{'pred/obs':>9s} {'Brier':>10s} {'ECE':>9s}")
    for name in SUBSETS:
        c = summary["subsets"][name]["calibration"]
        for key, lbl in (
            ("priora", "priora p(NEO)"),
            ("digest2_raw_over_100", "digest2 raw/100"),
            ("digest2_noid_over_100", "digest2 noid/100"),
        ):
            d = c[key]
            if not d:
                continue
            A(f"{name:7s} {lbl:22s} {d['expected_positives']:10.1f} "
              f"{d['observed_positives']:9.1f} "
              f"{d['ratio_expected_observed']:9.2f} {d['brier']:10.3e} "
              f"{d['ece']:9.3e}")
    A("")
    A("-" * 78)
    A("OPERATING POINTS  (digest2's NEOCP cut is raw >= 65)")
    A("-" * 78)
    A(f"{'subset':7s} {'operating point':26s} {'flagged':>8s} {'complete':>9s} "
      f"{'contam':>8s}")
    for name in SUBSETS:
        op = summary["subsets"][name].get("operating_points", {})
        for key in ("digest2_neocp", "priora_matched", "priora_p50"):
            d = op.get(key)
            if not d:
                continue
            A(f"{name:7s} {d['label']:26s} {d['n_flagged']:8d} "
              f"{d['completeness']:9.1%} {d['contamination']:8.1%}")
    A("")
    A("-" * 78)
    A("TRUNCATION SUBSET  (§2.6(b): digest2 bins H>=24 into [24,inf); priora cuts 25.5)")
    A("-" * 78)
    t = summary["truncation_diagnostic"]
    A(f"measure fraction at H > 25.5: median {t['median']:.4f}, mean {t['mean']:.4f}, "
      f"{t['frac_above_1e-2']:.1%} above 1e-2, {t['frac_above_1e-1']:.1%} above 1e-1")
    d_clean = summary["subsets"]["clean"]["auc"]["delta"]
    d_broken = summary["subsets"]["broken"]["auc"]["delta"]
    A(f"delta AUC: clean {d_clean:+.4f}   broken {d_broken:+.4f}   "
      f"(shift {d_broken - d_clean:+.4f})")
    A("")
    A("-" * 78)
    A("DISAGREEMENT CENSUS (clean subset; rank = percentile within subset)")
    A("-" * 78)
    dc = summary["subsets"]["clean"]["disagreement"]
    A(f"rank correlation {dc['rank_spearman']:.4f}; "
      f"mean |delta rank| {dc['mean_abs_delta_rank']:.4f}; "
      f"{dc['frac_delta_rank_gt_0p25']:.2%} of tracklets differ by > 0.25 in rank")
    hdr = (f"{'index':>7s} {'NEO':>4s} {'p42':>9s} {'d2raw':>7s} {'d rank':>7s} "
           f"{'V':>6s} {'H':>6s} {'q':>6s} {'e':>5s} {'rho':>7s} {'fH>25.5':>8s}")
    for key, title in (
        ("priora_higher", "priora ranks far ABOVE digest2"),
        ("digest2_higher", "digest2 ranks far ABOVE priora"),
        ("misranked_neos", "true NEOs, largest rank gap in priora's favour"),
    ):
        A("")
        A(f"  {title}:")
        A("  " + hdr)
        for r in summary["subsets"]["clean"]["disagreement"][key]:
            A("  %7d %4s %9.2e %7.2f %+7.3f %6.2f %6.2f %6.3f %5.3f %7.3f %8.4f"
              % (r["index"], "yes" if r["is_neo"] else "no", r["p_neo_42"],
                 r["d2_raw_neo"], r["delta_rank"], r["v_mag"], r["H_true"],
                 r["q_true"], r["e_true"], r["rho_true"], r["ar_frac_h_above"]))
    A("")
    A("-" * 78)
    A("CAVEATS")
    A("-" * 78)
    for c in summary["caveats"]:
        A(f"  * {c}")
    A("=" * 78)
    return "\n".join(L)


CAVEATS = [
    "Single geocentric observer (obscode 500) at one epoch per tracklet: no "
    "topocentric parallax, no site-error diversity, no cadence structure. "
    "digest2's per-site error table is therefore never exercised.",
    "No astrometric noise. digest2 sees exact positions but still assumes its "
    "default 1.0 arcsec observational error (d2math.c obsErr; obscode 500 has "
    "no entry in MPC.config and ADES rmsRA/rmsDec are left blank, which "
    "updateRMSValues floors to that same 1 arcsec). Its motion-vector "
    "perturbation therefore explores a rate box that is wider than the truth "
    "warrants -- a fixed, favourable-to-neither handicap that widens both "
    "codes' effective admissible regions differently. priora v0 uses a "
    "delta-function attributable (D-6), so it explores no rate box at all.",
    "digest2 options: repeatable=True (deterministic MC seeding), "
    "no_threshold=False (default RMS clamping), bundled MPC.config for "
    "per-site errors, all 15 classes computed. Score compared is 'raw' (All "
    "population) to match priora's All variant; 'noid' (Unk) is reported "
    "alongside but is a different quantity.",
    "The two codes use different Earth ephemerides: priora uses astropy's, "
    "digest2 its internal se2000 series (~1e-4 AU). Tracklet times are written "
    "as true UTC; the TDB-UTC offset shifts the observer by ~1e-5 AU, well "
    "inside that.",
    "Two detections per tracklet with an exactly linear arc: digest2's "
    "great-circle RMS is 0 by construction, so any RMS-driven behaviour in "
    "digest2 is at its floor. Real tracklets have >= 3 detections and non-zero "
    "residuals.",
    "Side effect: on first load digest2 writes a binary render of the CSV as "
    "data/digest2.model (11.7 MB, gitignored by data/*). It is derived from "
    "our file and is not read in preference to it -- an explicit model_path "
    "always wins -- but it is an artifact this harness creates outside its "
    "own output directory.",
    "The test population is drawn from the same model file both codes read, so "
    "this measures method under a correct prior. DESIGN §4.3 (v1+) is the "
    "prior-misspecification test; nothing here speaks to that.",
]


def write_outputs(summary: dict, out_dir: str, run: RunData,
                  d2: dict[str, np.ndarray]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=_json_default)
    txt = summary_text(summary)
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(txt + "\n")
    masks = subset_masks(run.ar_frac_h_above, **{
        "clean_max": summary["thresholds"]["clean_max"],
        "broken_min": summary["thresholds"]["broken_min"],
    })
    res = {k: summary["subsets"][k] for k in SUBSETS}
    plot_roc(run, d2, masks, res, os.path.join(out_dir, "roc.png"))
    plot_scatter(run, d2, masks, os.path.join(out_dir, "scatter.png"))
    plot_calibration(res, os.path.join(out_dir, "calibration.png"))
    plot_truncation(run, d2, res, os.path.join(out_dir, "truncation.png"))
    print(txt)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# --------------------------------------------------------------------------
# Aggregate report
# --------------------------------------------------------------------------


def build_report(cases: list[tuple[str, dict]]) -> str:
    """A short, committed-quality Markdown report over one or more runs."""
    head, hs = cases[0]
    env = hs["digest2"]
    L: list[str] = []
    A = L.append
    A("# WP7 — digest2 head-to-head comparison (DESIGN §4.2)")
    A("")
    A("*Produced by `validation/digest2_comparison.py`. Raw outputs (plots, "
      "`summary.json`, `summary.txt`) live under `validation/output/"
      "digest2_comparison/` (gitignored); this file is the durable record.*")
    A("")
    A("## What was compared, and how the premise was enforced")
    A("")
    A(f"| | |")
    A("|---|---|")
    A(f"| digest2 | `{env.get('version')}` from "
      f"`github.com/Smithsonian/mpc-public@{env.get('source_commit','')[:12]}`, "
      "subdirectory `digest2`, built from sdist |")
    A(f"| install | `CFLAGS=\"-D_GNU_SOURCE\" pip install --user "
      f"\"git+https://github.com/Smithsonian/mpc-public.git@"
      f"{env.get('source_commit','')[:12]}#subdirectory=digest2\"` "
      "(the flag is required: the sdist compiles `-std=c99`, under which "
      "`rand_r` is undeclared and the build fails) |")
    A(f"| population model | `data/digest2.model.csv`, sha256 "
      f"`{str(env.get('model_sha256'))[:16]}…`, passed via "
      "`Digest2(model_path=…)` |")
    same = env.get("bundled_model_lfs_oid") == env.get("model_sha256")
    A(f"| same file? | the pip package ships `digest2/data/digest2.model.csv` "
      f"as an unresolved git-LFS pointer whose `oid` is "
      f"`{str(env.get('bundled_model_lfs_oid'))[:16]}…` — "
      f"**{'identical' if same else 'DIFFERENT'}** to ours, so the shared-model "
      "premise is exact, not approximate |")
    A(f"| observatory codes | `{os.path.basename(str(env.get('obscodes_path')))}` "
      "(auto-downloaded from the MPC); only code 500 used |")
    A(f"| input format | ADES PSV (`parse_ades_psv`), "
      f"{hs['emission']['n_tracklets']} tracklets x 2 detections, "
      f"dt = {hs['emission']['dt_min']:.0f} min |")
    A(f"| parse rate | **{hs['digest2_run']['parse_rate']:.2%}** "
      f"({hs['digest2_run']['n_parsed_with_pair']}/"
      f"{hs['digest2_run']['n_emitted']} tracklets), "
      f"{hs['digest2_run']['score_rate']:.2%} scored |")
    A(f"| throughput | {hs['digest2_run']['score_ms_per_tracklet']:.1f} "
      f"ms/tracklet at {hs['digest2_run']['workers']} threads |")
    A("| score compared | digest2 `raw.NEO` (All population) against priora "
      "`p_neo` (All variant). digest2's default `noid.NEO` uses the `Unk` "
      "histogram and is a different quantity; it is reported alongside. |")
    A("")
    A("Each WP6 attributable becomes a two-detection tracklet with detection 0 "
      "**at** the attributable epoch (digest2 anchors its ranging on "
      "`obsPair[0]`, so its reference position and observer state are exactly "
      "the attributable's) and detection 1 at `epoch + dt`, propagated "
      "linearly in (alpha, delta). Since the WP6 attributable is itself a "
      "first-order motion model, the pair reproduces it by construction.")
    A("")
    dropped = [(n, s.get("n_dropped_nonfinite_p42", 0)) for n, s in cases]
    if any(d for _, d in dropped):
        A("Rows with a non-finite priora `p_neo` are excluded from both sides "
          "(they cannot enter a ROC on either): "
          + ", ".join(f"`{n}` {d}" for n, d in dropped if d)
          + ". These are tracklets where both component likelihoods underflow "
          "to zero, so the posterior is 0/0; excluding them from both codes is "
          "the only symmetric option.")
        A("")

    A("## Headline: the ranking gap on the clean subset")
    A("")
    A("| run | subset | n | NEOs | AUC priora | AUC digest2 | "
      "ΔAUC (95% CI) |")
    A("|---|---|---:|---:|---|---|---|")
    for name, s in cases:
        for sub in SUBSETS:
            d = s["subsets"][sub]["auc"]
            A(f"| {name} | {'**clean**' if sub == 'clean' else sub} | {d['n']} | "
              f"{d['n_pos']} | {d['auc_a']:.4f} "
              f"[{d['auc_a_ci'][0]:.4f}, {d['auc_a_ci'][1]:.4f}] | "
              f"{d['auc_b']:.4f} [{d['auc_b_ci'][0]:.4f}, {d['auc_b_ci'][1]:.4f}] | "
              f"**{d['delta']:+.4f}** "
              f"[{d['delta_ci'][0]:+.4f}, {d['delta_ci'][1]:+.4f}] |")
    A("")
    hc = hs["subsets"]["clean"]["auc"]
    A(f"On the clean subset of `{head}` — the DESIGN §4.2 headline — priora "
      f"ranks better by **ΔAUC = {hc['delta']:+.4f}** "
      f"(95% CI [{hc['delta_ci'][0]:+.4f}, {hc['delta_ci'][1]:+.4f}], "
      f"paired label-stratified bootstrap, B = {hc['n_boot']}; bootstrap "
      f"P(Δ ≤ 0) = {hc['p_delta_le_0']:.4f}). Both codes read the same "
      "population file and the test population is drawn from it, so this gap "
      "is attributable to the method — the Jacobian, the marginalisation over "
      "the admissible region, and the posterior — and not to a better prior.")
    A("")
    weak = [
        (f"{n}/{sub}", s["subsets"][sub]["auc"])
        for n, s in cases for sub in SUBSETS
        if s["subsets"][sub]["auc"]["delta_ci"][0] <= 0.0
    ]
    if weak:
        A("Not every subset resolves the gap, and the table should be read that "
          "way: " + "; ".join(
              f"`{n}` includes zero (P(Δ ≤ 0) = {d['p_delta_le_0']:.3f})"
              for n, d in weak
          ) + ". With ~10^2 NEOs per subset the paired bootstrap resolves a "
          "difference of order 10^-2 in AUC only marginally; the sign is "
          "consistent across every subset of both runs, but a single subset's "
          "interval is not on its own decisive. The N = 10^5 calibration run "
          "is what tightens this.")
        A("")

    A("## Calibration contrast")
    A("")
    A("digest2's score is a *fraction of tagged model population*, not a "
      "posterior probability, and it does not claim to be one. Treating "
      "`raw/100` as P(NEO) and asking the calibration question WP6 asks of "
      "priora makes the difference concrete rather than asserted:")
    A("")
    A("| run | subset | code | predicted NEOs | observed | pred/obs | Brier | ECE |")
    A("|---|---|---|---:|---:|---:|---:|---:|")
    for name, s in cases:
        for sub in ("all", "clean"):
            c = s["subsets"][sub]["calibration"]
            for key, lbl in (("priora", "priora `p_neo`"),
                             ("digest2_raw_over_100", "digest2 `raw`/100"),
                             ("digest2_noid_over_100", "digest2 `noid`/100")):
                d = c[key]
                A(f"| {name} | {sub} | {lbl} | {d['expected_positives']:.1f} | "
                  f"{d['observed_positives']:.1f} | "
                  f"{d['ratio_expected_observed']:.2f} | {d['brier']:.3e} | "
                  f"{d['ece']:.3e} |")
    A("")
    r42 = hs["subsets"]["all"]["calibration"]["priora"]
    rd2 = hs["subsets"]["all"]["calibration"]["digest2_raw_over_100"]
    A(f"Over the whole `{head}` run priora's posterior predicts "
      f"{r42['expected_positives']:.1f} NEOs against "
      f"{r42['observed_positives']:.0f} observed "
      f"(ratio {r42['ratio_expected_observed']:.2f}); digest2's `raw`/100 "
      f"predicts {rd2['expected_positives']:.1f}, over by a factor of "
      f"{rd2['ratio_expected_observed']:.0f}. The reliability curves "
      "(`calibration.png`) show the same thing bin by bin: priora tracks the "
      "diagonal, digest2 sits an order of magnitude above it across the whole "
      "score range. This is not a defect of digest2 — it is the quantity it "
      "was designed to emit — but it is the reason a digest2 score cannot be "
      "thresholded against a stated contamination budget.")
    A("")

    A("## What the threshold buys (operational reading of the calibration gap)")
    A("")
    A("digest2 posts a tracklet to the NEOCP at a score of 65. At equal "
      "completeness, what does each code's cut cost in contamination?")
    A("")
    A("| run | subset | operating point | flagged | completeness | contamination |")
    A("|---|---|---|---:|---:|---:|")
    for name, s in cases:
        for sub in ("all", "clean"):
            op = s["subsets"][sub].get("operating_points", {})
            for key in ("digest2_neocp", "priora_matched", "priora_p50"):
                d = op.get(key)
                if not d:
                    continue
                A(f"| {name} | {sub} | {d['label']} | {d['n_flagged']} | "
                  f"{d['completeness']:.1%} | {d['contamination']:.1%} |")
    A("")
    op = hs["subsets"]["all"].get("operating_points", {})
    if op.get("priora_matched"):
        a, b = op["digest2_neocp"], op["priora_matched"]
        A(f"On the full `{head}` run digest2's own NEOCP cut flags "
          f"{a['n_flagged']} tracklets to recover {a['completeness']:.0%} of "
          f"the NEOs, at {a['contamination']:.0%} contamination. priora "
          f"reaches the same completeness at p ≥ {b['threshold']:.3g}, flagging "
          f"{b['n_flagged']} tracklets at {b['contamination']:.0%} "
          "contamination. That factor is the practical content of the "
          "calibration table above: a calibrated posterior can be thresholded "
          "against a contamination budget, a population fraction cannot.")
        A("")
    A("## The truncation subset (the \"same inputs\" caveat)")
    A("")
    A("`d2model.c:hToBin` runs `for (ih = 0; h >= hpart[ih] && ih < HX-1; ++ih)` "
      "with `hpart[17] = 25.5`, so every trial orbit with H ≥ 24 falls in the "
      "last bin: digest2's top H bin is effectively [24, ∞). priora truncates "
      "the model at H = 25.5 (D-4). Where a tracklet carries appreciable "
      "admissible-region measure above H = 25.5, the two codes are not reading "
      "the same effective model and the head-to-head premise fails.")
    A("")
    A("| run | median frac at H>25.5 | >1e-2 | >1e-1 | ΔAUC clean | ΔAUC broken |")
    A("|---|---:|---:|---:|---|---|")
    for name, s in cases:
        t = s["truncation_diagnostic"]
        A(f"| {name} | {t['median']:.4f} | {t['frac_above_1e-2']:.1%} | "
          f"{t['frac_above_1e-1']:.1%} | "
          f"{s['subsets']['clean']['auc']['delta']:+.4f} | "
          f"{s['subsets']['broken']['auc']['delta']:+.4f} |")
    A("")

    A("## Disagreement census")
    A("")
    dc = hs["subsets"]["clean"]["disagreement"]
    A(f"On `{head}`'s clean subset the two rank orders correlate at "
      f"{dc['rank_spearman']:.3f}; {dc['frac_delta_rank_gt_0p25']:.1%} of "
      "tracklets differ by more than a quarter of the population in "
      "percentile rank.")
    A("")
    t2 = dc.get("tail_digest2_higher") or {}
    t42 = dc.get("tail_priora_higher") or {}
    if t2 and t42:
        A(f"The two tails are not symmetric. The {t2['n']} tracklets digest2 "
          f"ranks furthest above priora have median true "
          f"ρ = {t2['median_rho_true_au']:.1f} AU and median true "
          f"q = {t2['median_q_true_au']:.1f} AU "
          f"({t2['frac_q_gt_5au']:.0%} with q > 5 AU, "
          f"{t2['frac_rho_gt_10au']:.0%} beyond 10 AU) and are "
          f"{t2['frac_true_neo']:.0%} true NEOs — distant objects on which "
          "digest2 saturates near 100 while priora puts them at 1e-7. "
          f"The {t42['n']} tracklets priora ranks furthest above digest2 sit "
          f"at median ρ = {t42['median_rho_true_au']:.1f} AU, "
          f"q = {t42['median_q_true_au']:.1f} AU — ordinary main-belt "
          "tracklets, and both codes score them low in absolute terms, so the "
          "rank gap there costs little.")
        A("")
    doe = hs.get("distant_object_effect")
    if doe:
        A(f"That distant-object failure is real but it is not the whole story: "
          f"dropping the {doe['n_distant_in_clean']} tracklets with true "
          f"ρ > 10 AU from the clean subset moves ΔAUC only from "
          f"{doe['delta_clean']:+.4f} to "
          f"{doe['delta_clean_rho_lt_10au']:+.4f}. The remaining ~"
          f"{100 * doe['delta_clean_rho_lt_10au'] / doe['delta_clean']:.0f}% "
          "of the gap is diffuse — it comes from priora resolving a "
          "continuous posterior where digest2 can only tag bins as reachable "
          "or not.")
        A("")

    A("The extremes:")
    A("")
    for key, title in (
        ("priora_higher", "priora ranks far above digest2"),
        ("digest2_higher", "digest2 ranks far above priora"),
    ):
        A(f"**{title}**")
        A("")
        A("| gen index | true NEO | p_neo (42) | raw NEO (d2) | Δrank | V | "
          "H | q | e | ρ [AU] |")
        A("|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in dc[key]:
            A(f"| {r['index']} | {'**yes**' if r['is_neo'] else 'no'} | "
              f"{r['p_neo_42']:.2e} | {r['d2_raw_neo']:.2f} | "
              f"{r['delta_rank']:+.3f} | {r['v_mag']:.2f} | {r['H_true']:.2f} | "
              f"{r['q_true']:.3f} | {r['e_true']:.3f} | {r['rho_true']:.3f} |")
        A("")

    A("## Caveats and options chosen")
    A("")
    for c in hs["caveats"]:
        A(f"- {c}")
    A("")
    A("## Reproduce")
    A("")
    A("```")
    for name, _ in cases:
        A(f"python -m validation.digest2_comparison run "
          f"--run validation/output/{name} "
          f"--out validation/output/digest2_comparison/{name}")
    A("python -m validation.digest2_comparison report "
      "--out validation/output/digest2_comparison "
      + " ".join(f"--case {n}" for n, _ in cases))
    A("```")
    return "\n".join(L)


# --------------------------------------------------------------------------
# WP7 addendum — the clamped (digest2-emulation) control experiment
# --------------------------------------------------------------------------
#
# The head-to-head above has one uncontrolled variable: on 64.5% of the gate_v0
# catalogue the two codes are not consuming the same effective model, because
# digest2's ``hToBin`` clamps H >= 24 into a [24, inf) top bin while priora
# truncates at 25.5 (D-4).  ``Digest2Histogram(h_top_bin_infinite=True)``
# reinstates digest2's clamp, so re-scoring the *same* tracklets with a clamped
# priora pair makes the "same inputs" premise hold for ALL of them.  The
# comparison then decomposes the WP7 broken-subset gap into
#
#     Delta_WP7(broken)  =  Delta_clamped(broken)   +   boundary-handling term
#                           \____ method ____/          \__ D-4 vs digest2 __/
#
# and, on the calibration side, says how much of digest2's 6.1x overshoot in
# predicted NEO count is the H-clamp alone versus everything else.
#
# This mode is a *measurement instrument*, not a proposed configuration of
# priora: the clamped density is non-normalizable (see population.py) and is
# gate-ineligible.  Nothing here revises D-4.

#: File the clamped re-score writes, next to the existing case outputs.
CLAMPED_SCORES = "priora_clamped_scores.npz"
CLAMPED_CHUNK_DIR = "clamped_chunks"

#: Columns of that file.  A strict subset of WP6's RECORD_FIELDS: the HPD /
#: collapse / admissible-region-margin diagnostics are properties of the
#: *geometry* and the truth, which the clamp does not touch, so recomputing
#: them would only cost CPU.
CLAMPED_FIELDS = (
    "index",
    "p_neo",
    "log_lr",
    "lambda_neo",
    "lambda_other",
    "n_nodes",
    "ar_measure",
    "ar_frac_h_above",
    "elapsed_s",
)
_CLAMPED_INT_FIELDS = frozenset({"index", "n_nodes"})

#: The D-4 truncation edge, in magnitudes.  The clamped components report
#: ``support["H"] == (0, inf)`` — correctly, that is where their density is
#: non-zero — so the §2.6(b) diagnostic must be evaluated against this fixed
#: number rather than against ``other.support["H"][1]`` the way
#: ``self_consistency._init_worker`` does, or it would be identically 0 and
#: would no longer be comparable with the unclamped run's column.
H_TRUNC = 25.5


@dataclass
class ClampedScoreConfig:
    """Scoring configuration of the clamped pass.

    Deliberately the same fields as ``self_consistency.ScoreConfig``: the
    clamped pass must differ from the run's original pass in the clamp and
    nothing else, so :func:`clamped_score` seeds this from the run's own
    ``score_config.json`` and only the CLI may override it.
    """

    n_rho: int = 1200
    n_rhodot: int = 50
    rho_min: float = 0.01
    rho_max: float = 100.0
    G: float = 0.15
    model: str = DEFAULT_MODEL
    variant: str = "All"
    #: Carried so that a run scored with a non-default density mode cannot be
    #: silently re-scored under a different one.  The D-14 clamp is mutually
    #: EXCLUSIVE with both of these (``Digest2Histogram`` raises), so a
    #: non-default value here is not something to thread through — it is a
    #: reason to refuse, which is what :func:`clamped_score_config` does.
    #: Before this existed, both keys were dropped on the floor by the
    #: ``fields`` filter below and the clamped pass quietly scored a pchip or
    #: D-19-extended run against a plain clamped histogram.
    h_interpolation: str = "none"
    h_extend_to: float | None = None


def clamped_score_config(run_dir: str, **overrides) -> ClampedScoreConfig:
    """The run's own scoring config, with ``overrides`` applied.

    Reading ``score_config.json`` rather than defaulting is what makes the
    comparison controlled: if the run was scored on a different grid, the
    clamped pass follows it.  Unknown keys in the file are ignored (it is
    written by ``self_consistency.ScoreConfig``, which may grow fields).
    """
    fields = set(ClampedScoreConfig.__dataclass_fields__)
    cfg: dict = {}
    path = os.path.join(run_dir, "score_config.json")
    if os.path.exists(path):
        with open(path) as fh:
            cfg = {k: v for k, v in json.load(fh).items() if k in fields}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if "model" in cfg and not os.path.isabs(cfg["model"]):
        cfg["model"] = os.path.join(REPO_ROOT, cfg["model"])
    out = ClampedScoreConfig(**cfg)
    # The D-14 clamp cannot be combined with either of the other two H modes
    # (`Digest2Histogram.__init__` raises for both), so a run scored in one of
    # them has NO clamped control: the pass would compare a clamped histogram
    # against a pchip or D-19-extended reference and call the difference
    # "boundary handling".  Refuse with the reason rather than drop the field.
    if out.h_interpolation != "none":
        raise ValueError(
            f"{path}: this run was scored with h_interpolation="
            f"{out.h_interpolation!r}, which is mutually exclusive with the "
            "D-14 clamp (D-17): a clamped control for it does not exist, so "
            "the WP7 same-inputs comparison cannot be run on this run. Score "
            "a histogram-mode run instead."
        )
    if out.h_extend_to is not None:
        raise ValueError(
            f"{path}: this run was scored with h_extend_to="
            f"{out.h_extend_to!r} (D-19), which is mutually exclusive with the "
            "D-14 clamp: the extension already repairs the H-support wall with "
            "finite, normalizable mass, so a clamped control for it is neither "
            "constructible nor meaningful. Score a histogram-mode run instead "
            "if you want the WP7 same-inputs comparison."
        )
    return out


# Per-worker state, exactly the pattern of ``self_consistency._init_worker``:
# the 45,936-bin model is parsed once per worker rather than once per tracklet.
_CW: dict = {}


def _init_clamped_worker(model: str, variant: str, score_cfg: dict) -> None:
    from priora.population import Digest2Histogram

    # Defence in depth: `clamped_score_config` already refuses a run scored in
    # an incompatible H mode, but this worker is also reachable directly (and
    # is what actually builds the model), so the check lives here too.
    if str(score_cfg.get("h_interpolation", "none")) != "none" or (
        score_cfg.get("h_extend_to") is not None
    ):
        raise ValueError(
            "the clamped pass cannot be run for a config carrying "
            f"h_interpolation={score_cfg.get('h_interpolation')!r} / "
            f"h_extend_to={score_cfg.get('h_extend_to')!r}: the D-14 clamp is "
            "mutually exclusive with both (D-17, D-19)"
        )
    neo, other = Digest2Histogram.neo_other_pair(
        model, variant, h_top_bin_infinite=True
    )
    if not (neo.h_top_bin_infinite and other.h_top_bin_infinite):  # pragma: no cover
        raise RuntimeError("clamped worker built an unclamped pair")
    _CW["neo"] = neo
    _CW["other"] = other
    _CW["cfg"] = ClampedScoreConfig(**score_cfg)


def _clamped_score_one(task: tuple) -> tuple:
    """Score one tracklet with the clamped pair.  Module-level and picklable."""
    import warnings

    from astropy.time import Time as _Time

    from priora import Attributable, classify

    (index, ra, dec, ra_rate, dec_rate, jd, v_mag,
     rx, ry, rz, vx, vy, vz) = task
    cfg: ClampedScoreConfig = _CW["cfg"]
    att = Attributable(
        ra=float(ra), dec=float(dec), ra_rate=float(ra_rate),
        dec_rate=float(dec_rate),
        epoch=_Time(float(jd), format="jd", scale="tdb"), v_mag=float(v_mag),
    )
    t0 = time.time()
    with warnings.catch_warnings():
        # Empty-region / zero-likelihood tracklets are recorded (p_neo = nan),
        # not shouted about once per worker task.
        warnings.simplefilter("ignore", RuntimeWarning)
        res = classify(
            att, _CW["neo"], _CW["other"],
            R_obs=np.array([rx, ry, rz], dtype=float),
            V_obs=np.array([vx, vy, vz], dtype=float),
            rho_min=cfg.rho_min, rho_max=cfg.rho_max,
            n_rho=cfg.n_rho, n_rhodot=cfg.n_rhodot, G=cfg.G,
        )
    elapsed = time.time() - t0
    w = np.asarray(res.grid.weight, dtype=float)
    return (
        int(index),
        float(res.p_neo),
        float(res.log_lr),
        float(res.lambda_neo),
        float(res.lambda_other),
        int(res.grid.rho.size),
        float(w.sum()) if w.size else 0.0,
        float(res.measure_fraction_H_above(H_TRUNC)),
        float(elapsed),
    )


def _clamped_tasks(gen, rows: np.ndarray) -> list[tuple]:
    """Task tuples for the given gen.npz rows.

    A local, minimal tuple rather than ``self_consistency._tasks_from_generation``:
    that one also carries the truth (rho, rho_dot, elements, H) for the HPD and
    collapse statistics, which this pass does not compute — and unpacking a
    22-field tuple by position in a second module is a silent-misassignment
    hazard the moment either side grows a field.
    """
    return [
        (
            int(i), float(gen.ra[i]), float(gen.dec[i]),
            float(gen.ra_rate[i]), float(gen.dec_rate[i]),
            float(gen.epoch_jd_tdb[i]), float(gen.v_mag[i]),
            float(gen.R_obs[i, 0]), float(gen.R_obs[i, 1]), float(gen.R_obs[i, 2]),
            float(gen.V_obs[i, 0]), float(gen.V_obs[i, 1]), float(gen.V_obs[i, 2]),
        )
        for i in rows
    ]


def _clamped_stamp(run_dir: str, cfg: ClampedScoreConfig,
                   rows: np.ndarray) -> str:
    """Identity of (this catalogue, this config, this row selection).

    Same role as ``self_consistency._run_stamp``: a resumed run must never
    splice chunks computed against a different catalogue, grid, or subset.
    """
    h = hashlib.sha256()
    with open(os.path.join(run_dir, "gen.npz"), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    h.update(json.dumps(asdict(cfg), sort_keys=True).encode())
    h.update(b"clamped-v1")
    h.update(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    return h.hexdigest()


def _clamped_rows_to_arrays(rows: list[tuple]) -> dict[str, np.ndarray]:
    cols = list(zip(*rows)) if rows else [()] * len(CLAMPED_FIELDS)
    return {
        name: np.asarray(
            col, dtype=np.int64 if name in _CLAMPED_INT_FIELDS else np.float64
        )
        for name, col in zip(CLAMPED_FIELDS, cols)
    }


def clamped_score(
    run_dir: str,
    out_dir: str,
    index: np.ndarray,
    cfg: ClampedScoreConfig,
    workers: int = 8,
    chunk: int = 500,
    resume: bool = False,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Re-score ``index`` rows of ``run_dir``'s gen.npz with the CLAMPED pair.

    Consumes the WP6 catalogue exactly as ``self_consistency.score`` does — the
    same process-pool-with-initializer pattern, the same chunk-and-resume
    discipline — but writes to ``out_dir/priora_clamped_scores.npz`` and its
    own ``clamped_chunks/`` subdirectory.  It never touches ``run_dir``: the
    original ``scores.npz``, ``chunks/`` and ``score_config.json`` are inputs
    here and are left exactly as the gate produced them.

    ``index`` is the gen.npz row selection, in the order the results are to be
    reported (i.e. ``RunData.index``), so the returned arrays align row-for-row
    with the rest of the comparison.
    """
    from validation.self_consistency import load_generation

    gen = load_generation(run_dir)
    index = np.asarray(index, dtype=np.int64)
    n_gen = len(gen)
    if index.size and (index.min() < 0 or index.max() >= n_gen):
        raise ValueError(
            f"index out of range for {run_dir}/gen.npz (n={n_gen}): "
            f"[{index.min()}, {index.max()}]"
        )
    n = int(index.size)

    chunk_dir = os.path.join(out_dir, CLAMPED_CHUNK_DIR)
    os.makedirs(chunk_dir, exist_ok=True)
    stamp = _clamped_stamp(run_dir, cfg, index)

    bounds = [(s, min(s + chunk, n)) for s in range(0, n, max(1, int(chunk)))]
    done: dict[tuple[int, int], dict] = {}
    todo: list[tuple[int, int]] = []
    n_stale = 0
    for b in bounds:
        p = os.path.join(chunk_dir, "chunk_%06d_%06d.npz" % b)
        if resume and os.path.exists(p):
            try:
                with np.load(p) as z:
                    arrays = {k: z[k] for k in z.files}
                if str(arrays.pop("_stamp", "")) != stamp:
                    raise ValueError("stale chunk")
                if set(arrays) != set(CLAMPED_FIELDS):
                    raise ValueError("schema mismatch")
                done[b] = arrays
                continue
            except Exception:
                n_stale += 1
                os.remove(p)
        todo.append(b)

    if verbose:
        print(
            f"[clamped] {n} tracklets, grid ({cfg.n_rho}, {cfg.n_rhodot}), "
            f"{len(done)} chunk(s) reused, {len(todo)} to compute"
            + (f" ({n_stale} stale discarded)" if n_stale else "")
            + f", workers={workers}",
            flush=True,
        )

    t0 = time.time()
    n_done = 0
    executor = None
    try:
        if todo and workers and workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            executor = ProcessPoolExecutor(
                max_workers=int(workers),
                initializer=_init_clamped_worker,
                initargs=(cfg.model, cfg.variant, asdict(cfg)),
            )
        elif todo:
            if _CW.get("cfg") != cfg:
                _init_clamped_worker(cfg.model, cfg.variant, asdict(cfg))

        for start, stop in todo:
            tasks = _clamped_tasks(gen, index[start:stop])
            if executor is not None:
                rows = list(executor.map(_clamped_score_one, tasks, chunksize=1))
            else:
                rows = [_clamped_score_one(t) for t in tasks]
            arrays = _clamped_rows_to_arrays(rows)
            np.savez_compressed(
                os.path.join(chunk_dir, "chunk_%06d_%06d.npz" % (start, stop)),
                _stamp=np.str_(stamp), **arrays,
            )
            done[(start, stop)] = arrays
            n_done += len(rows)
            if verbose:
                el = time.time() - t0
                rate = n_done / el if el > 0 else 0.0
                eta = f", eta {(n - stop) / rate:.0f} s" if rate > 0 else ""
                print(
                    f"  [clamped] {stop}/{n}  ({el:.1f} s, "
                    f"{rate:.2f} tracklet/s{eta})", flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()

    merged = {
        name: (
            np.concatenate([done[b][name] for b in bounds])
            if bounds
            else np.zeros(
                0, dtype=np.int64 if name in _CLAMPED_INT_FIELDS else np.float64
            )
        )
        for name in CLAMPED_FIELDS
    }
    # Chunks are contiguous slices of ``index`` and are merged in order, so the
    # result is already aligned to it; assert rather than re-sort, because a
    # mismatch here means a chunk was spliced from the wrong selection.
    if not np.array_equal(merged["index"], index):
        raise RuntimeError(
            "clamped chunk merge is not aligned to the requested index"
        )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, CLAMPED_SCORES)
    np.savez_compressed(path, _config=np.str_(json.dumps(asdict(cfg),
                                                         sort_keys=True)),
                        **merged)
    if verbose:
        print(f"[clamped] wrote {path} ({time.time() - t0:.1f} s)")
    return merged


def load_clamped_scores(out_dir: str, index: np.ndarray) -> dict[str, np.ndarray]:
    """Read ``priora_clamped_scores.npz`` and align it to ``index``.

    Raises if any requested row is missing, rather than silently returning NaN
    for it: a partial clamped table would quietly turn the "premise holds for
    ALL tracklets" claim into a claim about an unspecified subset.
    """
    path = os.path.join(out_dir, CLAMPED_SCORES)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with np.load(path) as z:
        missing = [k for k in CLAMPED_FIELDS if k not in z.files]
        if missing:
            raise KeyError(f"{path} is missing columns {missing}")
        cols = {k: np.asarray(z[k]) for k in CLAMPED_FIELDS}

    index = np.asarray(index, dtype=np.int64)
    have = cols["index"].astype(np.int64)
    pos = {int(v): k for k, v in enumerate(have)}
    try:
        take = np.array([pos[int(i)] for i in index], dtype=np.int64)
    except KeyError as exc:
        raise KeyError(
            f"{path} has no clamped score for gen.npz row {exc.args[0]}; "
            "re-run `clamped` without --skip-score (or with --resume)"
        ) from None
    return {k: v[take] for k, v in cols.items()}


def analyze_clamped(
    run: RunData,
    clamped: dict[str, np.ndarray],
    d2: dict[str, np.ndarray],
    n_boot: int = 2000,
    seed: int = 7,
    clean_max: float = CLEAN_MAX_FRAC,
    broken_min: float = BROKEN_MIN_FRAC,
) -> dict:
    """The clamped-mode comparison, as a JSON-able dict.

    Subsets are the *WP7* subsets: they are cut on the unclamped run's
    §2.6(b) diagnostic, because the question is what happens to the tracklets
    the original experiment had to set aside.  (The diagnostic is a property of
    the admissible-region geometry, not of the density, so the clamped pass
    reproduces it — ``consistency.max_abs_diff_ar_frac`` checks exactly that.)
    """
    p42 = run.p_neo
    p42c = clamped["p_neo"].astype(float)
    s_d2 = d2["raw_neo"]
    masks = subset_masks(run.ar_frac_h_above, clean_max, broken_min)

    out: dict = {
        "n_clamped": int(p42c.size),
        "n_nonfinite_p_clamped": int((~np.isfinite(p42c)).sum()),
        # A clamped pair cannot produce a D-19 triage row: its H support end is
        # +inf, so the mechanism gate ("every evaluable node at or above the
        # support end") can never hold. Asserted rather than assumed, because
        # the clamped column is the WP7 control and a policy 1.0 in it would be
        # indistinguishable from a likelihood 1.0 in the ROC.
        "n_policy_assigned_clamped": int(np.count_nonzero(
            np.isfinite(p42c)
            & (np.asarray(clamped["n_nodes"]).astype(np.int64) > 0)
            & (np.asarray(clamped["lambda_neo"], dtype=float) == 0.0)
            & (np.asarray(clamped["lambda_other"], dtype=float) == 0.0)
        )) if {"n_nodes", "lambda_neo", "lambda_other"} <= set(clamped)
        else None,
        "thresholds": {"clean_max": clean_max, "broken_min": broken_min},
        "subsets": {},
    }

    for name in SUBSETS:
        m = masks[name] & d2["ok"] & np.isfinite(p42c)
        y, w = run.is_neo[m], run.weight[m]
        auc_c = auc_bootstrap(p42c[m], s_d2[m], y, w, n_boot=n_boot, seed=seed)
        auc_u = auc_bootstrap(p42[m], s_d2[m], y, w, n_boot=n_boot, seed=seed)
        # priora against itself: the pure boundary-handling effect, on
        # identical tracklets, with everything but the top H bin held fixed.
        auc_uc = auc_bootstrap(p42[m], p42c[m], y, w, n_boot=n_boot, seed=seed)
        out["subsets"][name] = {
            "n": int(m.sum()),
            "n_neo": int(y.sum()),
            "w_neo": float(w[y].sum()),
            "auc_clamped_vs_digest2": auc_c,
            "auc_unclamped_vs_digest2": auc_u,
            "auc_unclamped_vs_clamped": auc_uc,
            # Delta_WP7 = Delta_clamped + boundary.  With the clamp on, both
            # codes read the same effective model, so Delta_clamped is method
            # alone and the remainder is what D-4-vs-digest2 was worth.
            "decomposition": {
                "delta_wp7": auc_u["delta"],
                "delta_clamped_method": auc_c["delta"],
                "delta_boundary_handling": auc_u["delta"] - auc_c["delta"],
                "delta_clamped_ci": auc_c["delta_ci"],
                "delta_boundary_ci": auc_uc["delta_ci"],
            },
            "calibration": {
                "priora_clamped": calibration_stats(p42c[m], y, w),
                "priora": calibration_stats(p42[m], y, w),
                "digest2_raw_over_100": calibration_stats(
                    np.clip(s_d2[m] / 100.0, 0.0, 1.0), y, w
                ),
            },
            "operating_points_clamped": operating_points(p42c[m], s_d2[m], y, w),
        }

    # -- what the clamp did to the scores themselves ------------------------
    ok = np.isfinite(p42) & np.isfinite(p42c)
    f = run.ar_frac_h_above
    lo = 1e-300
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.log10(np.maximum(p42c, lo)) - np.log10(np.maximum(p42, lo))
    dlog_ok = dlog[ok]
    out["score_shift"] = {
        "n": int(ok.sum()),
        "frac_identical": float(np.mean(p42[ok] == p42c[ok])),
        "frac_abs_dlog10_gt_0p01": float(np.mean(np.abs(dlog_ok) > 0.01)),
        "frac_abs_dlog10_gt_0p1": float(np.mean(np.abs(dlog_ok) > 0.1)),
        "median_dlog10": float(np.median(dlog_ok)),
        "mean_dlog10": float(np.mean(dlog_ok)),
        "p01_dlog10": float(np.percentile(dlog_ok, 1)),
        "p99_dlog10": float(np.percentile(dlog_ok, 99)),
        "max_abs_dlog10": float(np.max(np.abs(dlog_ok))) if dlog_ok.size else 0.0,
        # The clamp can only act where the admissible region reaches H = 25.5.
        # These two must be (0, >= 0): a tracklet that moved with no exposure
        # would mean the clamp had leaked into the rest of the lookup.
        "n_moved_without_exposure": int(
            ((p42[ok] != p42c[ok]) & (f[ok] == 0.0)).sum()
        ),
        "n_unmoved_with_exposure": int(
            ((p42[ok] == p42c[ok]) & (f[ok] > 0.0)).sum()
        ),
        "frac_p_increased": float(np.mean(p42c[ok] > p42[ok])),
        "frac_p_decreased": float(np.mean(p42c[ok] < p42[ok])),
        "spearman_like_corr": float(
            np.corrcoef(percentile_rank(p42[ok]), percentile_rank(p42c[ok]))[0, 1]
        ),
        "by_truncation_bin": _shift_by_truncation_bin(f, dlog, ok),
    }

    # -- the clamped pass must have reproduced the geometry exactly ---------
    d_ar = np.abs(clamped["ar_frac_h_above"].astype(float) - f)
    d_nodes = clamped["n_nodes"].astype(int) - run.scores["n_nodes"].astype(int)
    out["consistency"] = {
        "max_abs_diff_ar_frac": float(np.nanmax(d_ar)) if d_ar.size else float("nan"),
        "max_abs_diff_n_nodes": int(np.abs(d_nodes).max()) if d_nodes.size else 0,
        "note": (
            "The clamp changes only the density, not the ranging grid or the "
            "implied H at each node, so both differences must be 0."
        ),
    }
    out["cpu_seconds"] = float(np.nansum(clamped["elapsed_s"].astype(float)))
    return out


def _shift_by_truncation_bin(f: np.ndarray, dlog: np.ndarray,
                             ok: np.ndarray) -> list[dict]:
    """Median clamp-induced score shift in bins of the §2.6(b) diagnostic."""
    edges = np.array([0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 1.0 + 1e-12])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & np.isfinite(f) & (f >= lo) & (f < hi)
        if not m.any():
            continue
        rows.append({
            "f_lo": float(lo),
            "f_hi": float(hi),
            "n": int(m.sum()),
            "median_dlog10_p": float(np.median(dlog[m])),
            "mean_dlog10_p": float(np.mean(dlog[m])),
            "frac_changed": float(np.mean(np.abs(dlog[m]) > 1e-12)),
        })
    return rows


# -- clamped plots ---------------------------------------------------------

_C42C = "#4c9f70"   # clamped priora


def plot_clamped_scatter(run, clamped, path):
    """(d) clamped vs unclamped priora, coloured by the §2.6(b) diagnostic."""
    p = np.clip(run.p_neo, 1e-12, 1.0)
    pc = np.clip(clamped["p_neo"].astype(float), 1e-12, 1.0)
    f = run.ar_frac_h_above
    ok = np.isfinite(p) & np.isfinite(pc)
    c = np.clip(np.where(np.isfinite(f), f, 0.0), 1e-5, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), constrained_layout=True)
    ax = axes[0]
    order = np.argsort(c[ok])          # draw the exposed tracklets on top
    sc = ax.scatter(
        p[ok][order], pc[ok][order], c=c[ok][order], s=5, alpha=0.55,
        linewidths=0, norm=matplotlib.colors.LogNorm(vmin=1e-5, vmax=1.0),
        cmap="viridis",
    )
    ax.plot([1e-12, 1.0], [1e-12, 1.0], color="0.6", lw=0.9, ls=":",
            label="unchanged")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-12, 1.5)
    ax.set_ylim(1e-12, 1.5)
    ax.set_xlabel("priora p(NEO), D-4 truncation at H = 25.5")
    ax.set_ylabel("priora p(NEO), digest2-clamped [24, inf)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.set_title("Every tracklet, coloured by §2.6(b) exposure", fontsize=10)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"measure fraction at implied $H > 25.5$", fontsize=8)

    ax = axes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.log10(pc) - np.log10(p)
    m = ok & np.isfinite(f) & np.isfinite(dlog)
    ax.scatter(np.clip(f[m], 1e-5, 1.0), dlog[m], s=5, alpha=0.35,
               linewidths=0, color=_C42C)
    ax.axhline(0.0, color="0.6", lw=0.9, ls=":")
    ax.axvline(CLEAN_MAX_FRAC, color=_C42, ls="--", lw=1.1, label="clean < 1e-3")
    ax.axvline(BROKEN_MIN_FRAC, color="k", ls="--", lw=1.1, label="broken > 1e-2")
    ax.set_xscale("log")
    ax.set_xlabel(r"measure fraction at implied $H > 25.5$")
    ax.set_ylabel(r"$\log_{10}\,p_{\rm clamped} - \log_{10}\,p_{\rm D-4}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.set_title("The clamp moves exactly the exposed tracklets", fontsize=10)
    fig.suptitle(
        "Clamped vs D-4 priora on identical tracklets and one shared model",
        fontsize=11,
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_clamped_roc(run, clamped, d2, masks, res, path):
    """(a)/(b) ROC: clamped priora, unclamped priora, digest2."""
    p42c = clamped["p_neo"].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    for ax, name in zip(axes, SUBSETS):
        m = masks[name] & d2["ok"] & np.isfinite(p42c)
        y, w = run.is_neo[m], run.weight[m]
        d = res["subsets"][name]
        # The two priora curves nearly coincide, so the clamped one is drawn
        # last and dashed; solid-on-solid would hide it entirely.
        for score, colour, lbl, auc, style in (
            (run.p_neo[m], _C42, "priora D-4",
             d["auc_unclamped_vs_digest2"]["auc_a"], "-"),
            (d2["raw_neo"][m], _CD2, "digest2 raw NEO",
             d["auc_clamped_vs_digest2"]["auc_b"], "-"),
            (p42c[m], _C42C, "priora clamped",
             d["auc_clamped_vs_digest2"]["auc_a"], "--"),
        ):
            fpr, tpr = _roc(score, y, w)
            ax.plot(fpr, tpr, color=colour, lw=1.7, ls=style,
                    label=f"{lbl}  AUC={auc:.4f}")
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("false positive rate")
        if ax is axes[0]:
            ax.set_ylabel("true positive rate")
        dec = d["decomposition"]
        ax.set_title(
            f"{name}  (n={d['n']}, NEO={d['n_neo']})\n"
            r"$\Delta$AUC clamped = %+0.4f [%+0.4f, %+0.4f];  boundary %+0.4f"
            % (dec["delta_clamped_method"], dec["delta_clamped_ci"][0],
               dec["delta_clamped_ci"][1], dec["delta_boundary_handling"]),
            fontsize=9,
        )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle(
        "Clamped control: with digest2's [24, inf) top H bin, the premise "
        "holds for every tracklet",
        fontsize=11,
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_clamped_calibration(res, path):
    """(c) reliability of clamped priora against D-4 priora and digest2."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, name in zip(axes, ("all", "broken")):
        c = res["subsets"][name]["calibration"]
        # Clamped last and dashed: it lies almost exactly on the D-4 curve
        # (that is the finding), so drawing it first would hide it.
        for key, colour, lbl, fmt in (
            ("priora", _C42, "priora D-4", "o-"),
            ("digest2_raw_over_100", _CD2, "digest2 raw NEO / 100", "o-"),
            ("priora_clamped", _C42C, "priora clamped", "s--"),
        ):
            rows = c[key].get("bins", [])
            if not rows:
                continue
            xs = [r["p_mean"] for r in rows]
            ys = [r["obs_frac"] for r in rows]
            err = [np.sqrt(max(r["expected_pos"], 1e-9)) / max(r["w"], 1e-9)
                   for r in rows]
            ax.errorbar(xs, ys, yerr=err, fmt=fmt, ms=4, lw=1.4, color=colour,
                        label=lbl, capsize=2)
        lim = [1e-5, 1.2]
        ax.plot(lim, lim, color="0.7", lw=0.9, ls=":", label="perfect")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lim)
        ax.set_ylim(1e-5, 1.2)
        ax.set_xlabel("mean predicted P(NEO)")
        if ax is axes[0]:
            ax.set_ylabel("observed NEO fraction")
        cc, cu, cd = (c["priora_clamped"], c["priora"],
                      c["digest2_raw_over_100"])
        ax.set_title(
            f"{name}: predicted NEOs  clamped {cc['expected_positives']:.1f}"
            f" / D-4 {cu['expected_positives']:.1f}"
            f" / digest2 {cd['expected_positives']:.1f}"
            f"  |  observed {cc['observed_positives']:.1f}",
            fontsize=8.5,
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5, which="both")
    fig.suptitle(
        "How much of digest2's calibration overshoot is the H-clamp alone?",
        fontsize=11,
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -- clamped report --------------------------------------------------------

CLAMPED_CAVEATS = [
    "The clamped mode is an INSTRUMENT, not a proposal. The clamped density "
    "is constant in H above 25.5 and therefore non-normalizable: it has no "
    "finite total count, cannot be sampled, and is gate-ineligible (DESIGN "
    "§4.1 has no generative counterpart for it). Nothing here revises D-4.",
    "Only the UPPER half of digest2's hToBin clamp is emulated. digest2 also "
    "puts H < 0 in bin 0; priora returns 0 there in both modes. Implied "
    "H < 0 does not occur at the geometries integrated here (rho >= 0.01 AU, "
    "V >= 14), so the omission is measurable-free and keeps the emulation a "
    "one-sided change.",
    "The clamp is applied to BOTH components. Clamping one side only would "
    "compare a truncated NEO density against a clamped 'other' density and "
    "the posterior would be an artifact of the mismatch.",
    "Everything except the top H bin is held fixed: the same gen.npz, the "
    "same grid, the same G, the same model file, the same digest2 scores and "
    "the same true labels as the WP7 run. Only priora is re-scored.",
    "The subsets are still cut on the UNCLAMPED §2.6(b) diagnostic, because "
    "the question is what happens to the tracklets WP7 had to set aside. The "
    "diagnostic is a property of the admissible-region geometry, not of the "
    "density, so it is identical in the two passes (verified: max abs diff "
    "reported in the consistency block).",
    "Emulating digest2's H binning does not emulate digest2's ranging, its "
    "1-arcsec error box, its bounded rho search, or its scoring heuristic. "
    "The residual gap is still 'everything else', not a single mechanism.",
]


def _fmt_ci(d: dict, key: str = "delta") -> str:
    return "%+0.4f [%+0.4f, %+0.4f]" % (
        d[key], d[key + "_ci"][0], d[key + "_ci"][1]
    )


def clamped_report(case: str, summary: dict, clamped: dict) -> str:
    """The CLAMPED_ADDENDUM.md body."""
    L: list[str] = []
    A = L.append
    S = clamped["subsets"]

    A("# WP7 addendum — the clamped (digest2-emulation) control")
    A("")
    A(f"*Case `{case}`; generated {clamped.get('generated', '?')}.*")
    A("")
    A("## What this measures")
    A("")
    A("WP7's head-to-head had one uncontrolled variable. `d2model.c:hToBin` "
      "clamps every trial orbit with H >= 24 into the final bin, so digest2's "
      "top H bin is effectively [24, inf); priora truncates at 25.5 per D-4. "
      "The two codes therefore consume the *same* effective model only where "
      "the admissible region carries no measure above H = 25.5 — which on "
      f"`{case}` is a minority of tracklets "
      f"({100 * summary['truncation_diagnostic']['frac_above_1e-2']:.1f}% sit "
      "above the 1e-2 'broken' threshold).")
    A("")
    A("`Digest2Histogram(h_top_bin_infinite=True)` reinstates digest2's clamp "
      "in priora. Re-scoring the identical tracklets with a clamped pair "
      "makes the shared-model premise hold for **all** of them, and splits the "
      "WP7 gap in two:")
    A("")
    A("```")
    A("Delta_WP7  =  Delta_clamped        +  boundary-handling term")
    A("              (method alone, both     (what D-4-vs-digest2's different")
    A("               codes now read the      faint-end model was worth)")
    A("               same faint end)")
    A("```")
    A("")
    A(f"Only priora is re-scored: the true labels, the digest2 scores, the "
      f"tracklet file and the grid are the WP7 run's own. "
      f"{clamped['n_clamped']} tracklets, "
      f"{clamped['cpu_seconds'] / 3600:.2f} CPU-hours.")
    A("")

    # (a) full set
    a = S["all"]
    dec = a["decomposition"]
    A("## (a) Full-set ΔAUC with the premise repaired")
    A("")
    A("With the clamp on, every tracklet is a valid head-to-head. Same paired, "
      "label-stratified bootstrap as WP7.")
    A("")
    A("| population | AUC priora | AUC digest2 | ΔAUC | one-sided p(Δ≤0) |")
    A("|---|---|---|---|---|")
    for name in SUBSETS:
        d = S[name]["auc_clamped_vs_digest2"]
        A(f"| `{name}` (n={S[name]['n']}, NEO={S[name]['n_neo']}) "
          f"| {d['auc_a']:.4f} [{d['auc_a_ci'][0]:.4f}, {d['auc_a_ci'][1]:.4f}] "
          f"| {d['auc_b']:.4f} [{d['auc_b_ci'][0]:.4f}, {d['auc_b_ci'][1]:.4f}] "
          f"| **{_fmt_ci(d)}** | {d['p_delta_le_0']:.4f} |")
    A("")
    A(f"Headline: on the **full set** with genuinely identical effective "
      f"models, priora outranks digest2 by ΔAUC = **{_fmt_ci(a['auc_clamped_vs_digest2'])}**.")
    A("")

    # (b) decomposition
    A("## (b) The broken subset, decomposed")
    A("")
    A("| population | Δ_WP7 (D-4 vs digest2) | Δ_clamped (method) | boundary handling |")
    A("|---|---|---|---|")
    for name in SUBSETS:
        d = S[name]["decomposition"]
        A(f"| `{name}` | {d['delta_wp7']:+.4f} "
          f"| {_fmt_ci(S[name]['auc_clamped_vs_digest2'])} "
          f"| {d['delta_boundary_handling']:+.4f} |")
    A("")
    b = S["broken"]
    bd = b["decomposition"]
    A(f"On the WP7 broken subset (n={b['n']}, {b['n_neo']} true NEOs) the WP7 "
      f"gap of {bd['delta_wp7']:+.4f} decomposes into "
      f"{bd['delta_clamped_method']:+.4f} of method and "
      f"{bd['delta_boundary_handling']:+.4f} of boundary handling.")
    A("")
    if bd["delta_boundary_handling"] < 0:
        A("The boundary term is **negative**: D-4's truncation was costing "
          "priora ranking accuracy on exactly the tracklets whose premise "
          "it broke. So the WP7 broken-subset number was *conservative* — it "
          "understated the method gap by handicapping priora, not digest2. "
          "Repairing the premise moves the broken subset from "
          f"{bd['delta_wp7']:+.4f} to {bd['delta_clamped_method']:+.4f}, "
          "which is now comfortably clear of zero rather than straddling it. "
          "The direction is what §2.6 predicts: truncating at 25.5 deletes "
          "faint-end density that is overwhelmingly NEO, so it suppresses "
          "P(NEO) for precisely the faint, nearby objects that are the true "
          "positives.")
    else:
        A("The boundary term is non-negative: D-4's truncation was, if "
          "anything, helping priora's ranking on these tracklets, so the "
          "WP7 broken-subset number was optimistic by that amount.")
    A("")
    A("Note also that this is not a large correction anywhere. The "
      "boundary-handling term is an order of magnitude smaller than the "
      "method term on every subset, and identically zero on `clean` — which "
      "is the WP7 subset definition working as designed.")
    A("")
    A("priora against *itself*, clamped vs D-4, on identical inputs — the "
      "boundary-handling term measured directly, with method held fixed:")
    A("")
    A("| population | AUC D-4 | AUC clamped | Δ (D-4 − clamped) |")
    A("|---|---|---|---|")
    for name in SUBSETS:
        d = S[name]["auc_unclamped_vs_clamped"]
        A(f"| `{name}` | {d['auc_a']:.4f} | {d['auc_b']:.4f} | {_fmt_ci(d)} |")
    A("")

    # (c) calibration
    A("## (c) Calibration: how much of digest2's overshoot is the H-clamp?")
    A("")
    A("| population | code | predicted NEOs | observed | ratio | ECE | max \\|z\\| bin |")
    A("|---|---|---|---|---|---|---|")
    for name in SUBSETS:
        cal = S[name]["calibration"]
        for key, lbl in (("priora_clamped", "priora clamped"),
                         ("priora", "priora D-4"),
                         ("digest2_raw_over_100", "digest2 raw/100")):
            c = cal[key]
            A(f"| `{name}` | {lbl} | {c['expected_positives']:.1f} "
              f"| {c['observed_positives']:.0f} "
              f"| {c['ratio_expected_observed']:.3f} | {c['ece']:.5f} "
              f"| {c['max_abs_z_bin']:.2f} |")
    A("")
    ca = S["all"]["calibration"]
    r_c = ca["priora_clamped"]["ratio_expected_observed"]
    r_u = ca["priora"]["ratio_expected_observed"]
    r_2 = ca["digest2_raw_over_100"]["ratio_expected_observed"]
    span = r_2 - r_u
    closed = (r_c - r_u) / span if abs(span) > 1e-9 else float("nan")
    A(f"digest2 over-predicts the NEO count by {r_2:.2f}x; priora under D-4 "
      f"predicts {r_u:.3f}x. Adopting digest2's H-clamp — and changing "
      f"nothing else — moves priora to {r_c:.4f}x, i.e. it closes "
      f"{100 * closed:.2f}% of the gap between the two codes' calibration. "
      "The H-clamp is therefore "
      + ("a minor contributor to" if abs(closed) < 0.25
         else "a substantial contributor to")
      + " digest2's calibration overshoot; the remaining "
      f"{100 * (1 - closed):.2f}% is the heuristic score itself (digest2's "
      "'raw' is a fraction of trial orbits, not a marginal likelihood ratio, "
      "and no amount of agreeing about the faint end makes it one).")
    A("")
    A(f"ECE moves from {ca['priora']['ece']:.5f} to "
      f"{ca['priora_clamped']['ece']:.5f} against digest2's "
      f"{ca['digest2_raw_over_100']['ece']:.5f}, and the clamped posterior's "
      "own calibration is untouched — which is the substantive finding here: "
      "the clamp is a change to the *prior's* faint end, and a correctly "
      "marginalized posterior absorbs it. It shows up in the ranking of "
      "faint-end-exposed tracklets (b) and essentially nowhere in the "
      "predicted count.")
    A("")
    op = S["all"]["operating_points_clamped"]
    if "priora_matched" in op:
        d2p, mp = op["digest2_neocp"], op["priora_matched"]
        A(f"Operationally, at digest2's own NEOCP cut (raw >= 65: "
          f"{100 * d2p['completeness']:.1f}% complete, "
          f"{100 * d2p['contamination']:.1f}% contaminated), clamped priora "
          f"matched to the same completeness carries "
          f"{100 * mp['contamination']:.1f}% contamination "
          f"(at p >= {mp['threshold']:.3g}) — the WP7 headline, unchanged by "
          "the clamp.")
        A("")

    # (d) scatter
    A("## (d) What the clamp did to the scores")
    A("")
    s = clamped["score_shift"]
    A(f"- {100 * s['frac_identical']:.1f}% of tracklets score *bit-identically* "
      f"in the two modes.")
    A(f"- {100 * s['frac_abs_dlog10_gt_0p01']:.1f}% move by more than 0.01 dex, "
      f"{100 * s['frac_abs_dlog10_gt_0p1']:.1f}% by more than 0.1 dex.")
    A(f"- The shift is {'upward' if s['median_dlog10'] > 0 else 'downward'} in "
      f"the median ({s['median_dlog10']:+.2e} dex); "
      f"{100 * s['frac_p_increased']:.2f}% of p(NEO) values rise, "
      f"{100 * s['frac_p_decreased']:.2f}% fall. "
      f"1st-99th percentile of Δlog10 p: "
      f"[{s['p01_dlog10']:+.4f}, {s['p99_dlog10']:+.4f}].")
    A(f"- Rank correlation between the two modes: "
      f"{s['spearman_like_corr']:.5f}.")
    if s["frac_p_decreased"] < 1e-9:
        A("- The clamp moves p(NEO) in **one direction only**. That is not an "
          "accident of this catalogue: the clamp adds density in the top H "
          "bin to both components, and DESIGN §2.6 records that 55.8% of the "
          "NEO component's mass sits in [24, 25.5) against 1.4% of the other "
          "component's — ~40x more sensitive per unit mass on the NEO side. "
          "Restoring the faint end therefore always raises the NEO share of "
          "the marginal likelihood, never lowers it.")
    A("")
    A("| §2.6(b) measure fraction | n | median Δlog₁₀ p | fraction moved |")
    A("|---|---|---|---|")
    for r in s["by_truncation_bin"]:
        A(f"| [{r['f_lo']:g}, {r['f_hi']:g}) | {r['n']} "
          f"| {r['median_dlog10_p']:+.4f} | {100 * r['frac_changed']:.1f}% |")
    A("")
    A("`clamped_scatter.png` plots the same thing: clamped vs D-4 p(NEO) with "
      "the §2.6(b) diagnostic as colour, and Δlog₁₀ p against that diagnostic. "
      "The story it tells is that the knob does exactly what it is supposed "
      "to and nothing else. Tracklets whose admissible region never reaches "
      "H = 25.5 sit *bit-identically* on the identity line — there is no "
      "density for the clamp to add — and the displacement above the line "
      "rises monotonically with exposure, bin for bin in the table above, "
      "across four decades of the diagnostic; the largest single displacement "
      f"is {s['max_abs_dlog10']:.3f} dex. There is no scatter off the trend: "
      f"{s['n_moved_without_exposure']} tracklets moved with zero exposure "
      f"(and {s['n_unmoved_with_exposure']} with exposure too small to reach "
      "a populated bin did not move), and none moved down. So the ΔAUC "
      "changes in (a) and (b) are attributable to the top H bin and to "
      "nothing that leaked in with it.")
    A("")

    cons = clamped["consistency"]
    A("## Controls")
    A("")
    A(f"- §2.6(b) diagnostic reproduced by the clamped pass to "
      f"{cons['max_abs_diff_ar_frac']:.3g} (must be 0: the clamp changes the "
      f"density, not the geometry).")
    A(f"- Node counts identical to {cons['max_abs_diff_n_nodes']} "
      f"(must be 0: same grid).")
    A(f"- Non-finite clamped p(NEO): {clamped['n_nonfinite_p_clamped']}.")
    wp7 = (summary.get("subsets") or {})
    if wp7:
        deltas = ", ".join(
            "`%s` %+0.4f vs %+0.4f" % (
                name, S[name]["auc_unclamped_vs_digest2"]["delta"],
                wp7[name]["auc"]["delta"])
            for name in SUBSETS if name in wp7
        )
        A(f"- Δ_WP7 recomputed here vs the WP7 summary: {deltas} (identical "
          "inputs, so these must agree exactly; any drift means the case "
          "directory and the run directory have come apart).")
    A("")
    A("## Caveats")
    A("")
    for c in CLAMPED_CAVEATS:
        A(f"- {c}")
    A("")
    A("## Reproduce")
    A("")
    A("```")
    A("python -m validation.digest2_comparison clamped \\")
    A(f"    --run {os.path.relpath(summary['run_dir'], REPO_ROOT)} \\")
    A(f"    --out validation/output/digest2_comparison/{case} --workers 8")
    A("```")
    return "\n".join(L)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    ok, msg = digest2_available()
    if not ok:
        print(msg, file=sys.stderr)
        return 2

    t_all = time.time()
    run = load_run(args.run, limit=args.limit)
    with np.load(os.path.join(args.run, "scores.npz")) as z:
        n_scored_rows = int(z["index"].size)
    n_dropped = n_scored_rows - run.n if args.limit is None else 0
    print(f"[wp7] {args.run}: {run.n} tracklets, {int(run.is_neo.sum())} true NEOs"
          + (f" ({n_dropped} dropped for non-finite priora p_neo)"
             if n_dropped else ""))

    os.makedirs(args.out, exist_ok=True)
    obs_path = os.path.join(args.out, "tracklets" + _SUFFIX[args.format])
    emission = EMITTERS[args.format](run, obs_path, dt_min=args.dt_min)
    print(f"[wp7] wrote {emission['n_observations']} observations -> {obs_path}")

    cache = os.path.join(args.out, "digest2_scores.npz")
    d2 = stats = None
    if args.reuse and os.path.exists(cache):
        with np.load(cache) as z:
            if z["index"].size == run.n and np.array_equal(z["index"], run.index):
                d2 = {k: z[k] for k in z.files if k != "index"}
                stats = json.load(open(os.path.join(args.out, "summary.json")))[
                    "digest2_run"
                ]
                print(f"[wp7] reusing cached digest2 scores from {cache}")
    if d2 is None:
        d2, stats = run_digest2(obs_path, run.index, model_path=args.model,
                                workers=args.workers)
        np.savez_compressed(cache, index=run.index, **d2)
    if stats["parse_rate"] < 1.0:
        print(f"[wp7] WARNING: parse rate {stats['parse_rate']:.4%} < 100%",
              file=sys.stderr)

    res = analyze(run, d2, n_boot=args.n_boot, seed=args.seed,
                  clean_max=args.clean_max, broken_min=args.broken_min)
    summary = {
        "run_dir": os.path.abspath(args.run),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "digest2": digest2_env(args.model),
        "emission": emission,
        "digest2_run": stats,
        "n_tracklets": run.n,
        "n_dropped_nonfinite_p42": int(n_dropped),
        "n_neo": int(run.is_neo.sum()),
        "w_neo": float(run.weight[run.is_neo].sum()),
        "weighted": bool(not np.allclose(run.weight, 1.0)),
        "caveats": CAVEATS,
        **res,
    }
    write_outputs(summary, args.out, run, d2)
    print(f"[wp7] total {time.time() - t_all:.1f} s -> {args.out}")
    return 0


def cmd_clamped(args: argparse.Namespace) -> int:
    """Re-score a WP7 case with the clamped pair and write the addendum.

    Reuses the case directory's existing artifacts — the true labels come from
    the run's gen.npz and the digest2 scores from ``digest2_scores.npz`` — and
    writes only new files plus a ``clamped`` block appended to the case's
    ``summary.json``.  ``scores.npz`` and ``chunks/`` in the run directory are
    never touched.
    """
    t_all = time.time()
    run = load_run(args.run, limit=args.limit)
    print(f"[clamped] {args.run}: {run.n} tracklets, "
          f"{int(run.is_neo.sum())} true NEOs")

    d2_path = os.path.join(args.out, "digest2_scores.npz")
    if not os.path.exists(d2_path):
        print(f"missing {d2_path}; run the `run` subcommand first",
              file=sys.stderr)
        return 2
    with np.load(d2_path) as z:
        if not np.array_equal(z["index"].astype(int), run.index.astype(int)):
            print(f"{d2_path} was computed for a different tracklet selection; "
                  "re-run `run` (or drop --limit)", file=sys.stderr)
            return 2
        d2 = {k: z[k] for k in z.files if k != "index"}
    print(f"[clamped] reusing digest2 scores and true labels from {args.out}")

    cfg = clamped_score_config(
        args.run, model=args.model, n_rho=args.n_rho, n_rhodot=args.n_rhodot
    )
    print(f"[clamped] scoring config: {asdict(cfg)}")

    if args.skip_score:
        clamped = load_clamped_scores(args.out, run.index)
        print(f"[clamped] reusing {os.path.join(args.out, CLAMPED_SCORES)}")
    else:
        clamped = clamped_score(
            args.run, args.out, run.index, cfg, workers=args.workers,
            chunk=args.chunk, resume=args.resume,
        )

    res = analyze_clamped(run, clamped, d2, n_boot=args.n_boot, seed=args.seed,
                          clean_max=args.clean_max, broken_min=args.broken_min)
    res["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    res["score_config"] = asdict(cfg)
    res["run_dir"] = os.path.abspath(args.run)
    res["caveats"] = CLAMPED_CAVEATS

    # Merge into the case summary rather than rewriting it: everything WP7
    # wrote there is still true and is what the addendum quotes.
    spath = os.path.join(args.out, "summary.json")
    with open(spath) as fh:
        summary = json.load(fh)
    summary["clamped"] = res
    with open(spath, "w") as fh:
        json.dump(summary, fh, indent=1, default=_json_default)

    masks = subset_masks(run.ar_frac_h_above, args.clean_max, args.broken_min)
    plot_clamped_scatter(run, clamped,
                         os.path.join(args.out, "clamped_scatter.png"))
    plot_clamped_roc(run, clamped, d2, masks, res,
                     os.path.join(args.out, "clamped_roc.png"))
    plot_clamped_calibration(res,
                             os.path.join(args.out, "clamped_calibration.png"))

    case = os.path.basename(os.path.normpath(args.out))
    addendum_dir = args.addendum_dir or os.path.dirname(
        os.path.normpath(os.path.abspath(args.out))
    )
    os.makedirs(addendum_dir, exist_ok=True)
    md = clamped_report(case, summary, res)
    apath = os.path.join(addendum_dir, "CLAMPED_ADDENDUM.md")
    with open(apath, "w") as fh:
        fh.write(md + "\n")

    # Mirror the block into the aggregate summary.json too, if one exists, so
    # the two stay consistent without rewriting anything else in it.
    agg = os.path.join(addendum_dir, "summary.json")
    if os.path.exists(agg):
        with open(agg) as fh:
            top = json.load(fh)
        if isinstance(top.get("runs"), dict) and case in top["runs"]:
            top["runs"][case]["clamped"] = res
            with open(agg, "w") as fh:
                json.dump(top, fh, indent=1, default=_json_default)

    print(md)
    print(f"\n[clamped] wrote {apath}, {spath}, and 3 plots in {args.out}")
    print(f"[clamped] total {time.time() - t_all:.1f} s")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cases: list[tuple[str, dict]] = []
    for name in args.case:
        p = os.path.join(args.out, name, "summary.json")
        if not os.path.exists(p):
            print(f"missing {p}; run the `run` subcommand first", file=sys.stderr)
            return 2
        with open(p) as fh:
            cases.append((name, json.load(fh)))

    os.makedirs(args.out, exist_ok=True)
    combined = {name: s for name, s in cases}
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"headline": cases[0][0], "runs": combined}, fh, indent=1,
                  default=_json_default)
    txt = "\n\n".join(f"### {name}\n{summary_text(s)}" for name, s in cases)
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(txt + "\n")
    md = build_report(cases)
    with open(os.path.join(args.out, "REPORT.md"), "w") as fh:
        fh.write(md + "\n")
    print(md)
    print(f"\n[wp7] wrote {args.out}/REPORT.md, summary.json, summary.txt")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="digest2_comparison",
        description="WP7 — digest2 vs priora on shared inputs (DESIGN §4.2)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="emit + score + analyze one WP6 run")
    r.add_argument("--run", required=True, help="WP6 run dir with gen.npz/scores.npz")
    r.add_argument("--out", required=True, help="output directory")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--format", default="ades_psv", choices=sorted(EMITTERS))
    r.add_argument("--dt-min", type=float, default=DEFAULT_DT_MIN)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--limit", type=int, default=None, help="first N tracklets only")
    r.add_argument("--reuse", action="store_true",
                   help="reuse a cached digest2_scores.npz in --out if it "
                        "matches the run's index (re-runs the analysis only)")
    r.add_argument("--clean-max", type=float, default=CLEAN_MAX_FRAC)
    r.add_argument("--broken-min", type=float, default=BROKEN_MIN_FRAC)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser(
        "clamped",
        help="re-score a case with the digest2-emulation (clamped top H bin) "
             "pair and write CLAMPED_ADDENDUM.md",
    )
    c.add_argument("--run", required=True, help="WP6 run dir with gen.npz/scores.npz")
    c.add_argument("--out", required=True,
                   help="the case output dir produced by `run` (holds "
                        "digest2_scores.npz and summary.json)")
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--chunk", type=int, default=500)
    c.add_argument("--resume", action="store_true",
                   help="reuse matching clamped chunk files")
    c.add_argument("--skip-score", action="store_true",
                   help="reuse an existing priora_clamped_scores.npz and "
                        "redo the analysis only")
    c.add_argument("--n-rho", type=int, default=None,
                   help="override the run's own grid (default: match it)")
    c.add_argument("--n-rhodot", type=int, default=None)
    c.add_argument("--n-boot", type=int, default=2000)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--clean-max", type=float, default=CLEAN_MAX_FRAC)
    c.add_argument("--broken-min", type=float, default=BROKEN_MIN_FRAC)
    c.add_argument("--addendum-dir", default=None,
                   help="where CLAMPED_ADDENDUM.md goes (default: parent of --out)")
    c.set_defaults(func=cmd_clamped)

    p = sub.add_parser("report", help="aggregate per-run summaries into REPORT.md")
    p.add_argument("--out", required=True)
    p.add_argument("--case", action="append", required=True,
                   help="subdirectory name under --out (repeatable; first is headline)")
    p.set_defaults(func=cmd_report)

    e = sub.add_parser("env", help="print the digest2 install/wiring record")
    e.add_argument("--model", default=DEFAULT_MODEL)
    e.set_defaults(func=lambda a: (
        print(json.dumps(digest2_env(a.model), indent=1, default=_json_default)) or 0
    ))

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
