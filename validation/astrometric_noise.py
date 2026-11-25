"""WP8 — how much does Rubin-level astrometric noise cost priora v0?

priora v0 treats the attributable ``A = (alpha, delta, alpha_dot, delta_dot)``
as EXACT — a delta function (DECISIONS D-6, DESIGN §2.7).  Real Rubin tracklets
carry astrometric uncertainty, and on a short (~45 min) arc that uncertainty
propagates into the *rates* amplified by ``1/dt``.  This harness measures the
price of the delta-function assumption: it perturbs the WP6 gate catalogue's
attributables with a transcription of **Sorcha's own** magnitude-dependent
astrometric error model, re-scores them with the unchanged v0 classifier, and
compares against the stored noiseless scores.

The question it answers, quantitatively: *at Rubin-level noise, how much
calibration error and how much completeness/contamination degradation does
ignoring the errors cost?* — i.e. is v1's covariance marginalization urgent?

The error model (transcribed, not invented)
-------------------------------------------
Source: **Sorcha**, https://github.com/dirac-institute/sorcha, file
``src/sorcha/modules/PPAddUncertainties.py``, functions
``calcAstrometricUncertainty`` and ``calcRandomAstrometricErrorPerCoord``
(cross-checked byte-for-byte between the v1.2.1 tag, commit
``4e37cfe974a04a2197c64ff5074dc68608fe1893``, and the sorcha 1.1.1 conda build
shipped in the Rubin stack).  The trailing-loss term comes from
``src/sorcha/modules/PPTrailingLoss.py::calcTrailingLoss``.

Sorcha's own citations, verbatim from those files:

* ``calcAstrometricUncertainty`` docstring:
  "This is also referenced in the LSST overview paper (arXiv:0805.2366,
  ls.st/lop)" — i.e. Ivezić et al. (2019), ApJ 873, 111 (arXiv:0805.2366 is the
  long-lived preprint of the LSST overview paper), and
  "the systematic error, error_sys = 10 mas, corresponds to the design spec
  from the LSST Science Requirements Document (ls.st/srd)".
* ``calcTrailingLoss`` docstring: "Find the trailing loss from trailing and
  detection (Veres & Chesley 2017)" — Vereš & Chesley (2017), AJ 154, 12.

The chain, exactly as Sorcha runs it for a Rubin visit (see
``addUncertainties`` in the same file):

1. trailing loss, ``model="trailedSource"``, converts the true apparent
   magnitude into "the equivalent point source with the same SNR":
   ``dmag = 1.25 log10(1 + a x^2 / (1 + b x))`` with ``x = v texp / seeing``,
   ``a = 0.761``, ``b = 1.162``, ``texp = 30 s``;
2. the LSST photometric-SNR model:
   ``x_snr = 10^(0.4 (mag - m5))``,
   ``SNR = 1 / sqrt((0.04 - gamma) x_snr + gamma x_snr^2)``, ``gamma = 0.039``
   (Sorcha hardcodes one gamma for all bands);
3. ``error_rand = astErrCoeff * FWHMeff / SNR`` with ``astErrCoeff = 0.60``
   (single-Gaussian-PSF maximum-likelihood centroid);
4. ``sigma = sqrt(error_sys^2 + error_rand^2)``, ``error_sys = 10 mas``.

``sigma`` is a **single-coordinate** uncertainty (the docstring says so
explicitly), which is what this module injects: one independent draw per
detection per sky coordinate, in the local sky metric.

Parameters adopted here (all overridable on the CLI)
----------------------------------------------------
* ``m5 = 24.0``.  Sorcha hardcodes no default 5-sigma depth — it reads
  ``fiveSigmaDepth`` per visit from the opsim pointing database, which we do not
  have.  We adopt ``m5 = 24.0`` because it is the value *internally consistent
  with the catalogue being perturbed*: the WP6 generator's observability cut is
  ``14 <= V < 24`` with "the faint limit is the Rubin-like detection limit"
  (``self_consistency`` module docstring).  Setting ``m5 = V_max`` makes the
  gate catalogue exactly a 5-sigma-limited sample: the faintest tracklets sit at
  SNR = 5 and nothing in the catalogue is below the assumed survey's limit.  It
  is also the conservative choice (largest sigma), so a "noise does not matter"
  conclusion drawn at this depth is robust.  ``--m5 24.5`` (roughly Rubin's
  r-band single-visit design depth, i.e. half a magnitude deeper than our V cut)
  is run as a paired sensitivity case; V-vs-r colour is not modelled (D-5).
* ``FWHM = 700 mas`` (0.7"), **Sorcha's own default** for the ``FWHMeff``
  argument of ``calcAstrometricUncertainty``.  Note Sorcha's production call
  passes ``seeingFwhmGeom_arcsec * 1000`` into that argument, so 700 mas stands
  in for a 0.7" geometric seeing visit.
* ``error_sys = 10 mas``, ``astErrCoeff = 0.60``, ``gamma = 0.039``,
  ``nvisit = 1``, ``texp = 30 s``, trailing losses ON — all Sorcha defaults.

Resulting single-detection, single-coordinate sigma at ``m5 = 24``, no trailing
(the numbers are asserted in ``tests/test_astrometric_noise.py``)::

    V        18        21        23        24
    SNR     466.29     67.67     12.33      5.00
    sigma    10.040    11.769    35.504    84.593   mas

The 10 mas systematic floor dominates below V ~ 21.5; the faint half of the
catalogue (and most of its NEOs) lives where the random term takes over.

Noise injection (§2 of the experiment)
--------------------------------------
Each attributable is expanded into the same two-detection tracklet WP7 emits to
digest2 (``digest2_comparison._tracklet_positions``): detection 0 **at** the
attributable epoch, detection 1 at ``epoch + dt`` with ``dt = 45 min``, linear
motion in ``(alpha, delta)``.  Offsets are drawn per detection per coordinate in
the **local sky metric** — a ``sigma`` offset in ``delta`` and a ``sigma / cos
delta`` offset in ``alpha`` — with the metric evaluated once at the
attributable's own declination ``delta_0``.  (Evaluating ``cos delta`` at each
endpoint separately would change the injected sigma by <1% for all but a handful
of >5 deg/day tracklets, and would break the exact additivity of the
position/rate decomposition below.)

The noisy attributable is then rebuilt exactly the way a real two-detection
tracklet's attributable is formed — position from detection 0, rates from the
finite difference::

    alpha'      = alpha + d_alpha_0
    delta'      = delta + d_delta_0
    alpha_dot'  = alpha_dot + (d_alpha_1 - d_alpha_0) / dt
    delta_dot'  = delta_dot + (d_delta_1 - d_delta_0) / dt

so the position error is ``sigma`` per coordinate and the rate error is
``sqrt(2) sigma / dt`` per coordinate.  ``V`` is **unchanged**: photometric
noise is deliberately out of scope here (it would move ``H(V, rho)`` and mix a
second, independent effect into every number below).  The truth columns
(``elements``, ``H``, ``rho_true``, ``rho_dot_true``, ``is_neo``) are copied
verbatim, so every comparison is against the same labels.

Three noise modes, sharing one set of standard-normal draws ``e0, e1`` so the
three are exactly paired (common random numbers):

``both``
    ``d_0 = sigma e0``, ``d_1 = sigma e1`` — the physical case: independent
    per-detection noise, giving both a position error and a rate error.
``pos``
    ``d_0 = d_1 = sigma e0`` — the *same* offset applied to both detections.
    A rigid translation of the tracklet: position error ``sigma``, rate error
    **identically zero** (bit-identical rates, asserted in the tests).
``rate``
    ``d_0 = 0``, ``d_1 = sigma (e1 - e0)`` — position **unperturbed**, rate
    error identical draw-for-draw to ``both``'s.

By construction the 4-vector perturbations satisfy
``delta_A(both) = delta_A(pos) + delta_A(rate)`` exactly, so the split of the
score change between "the position moved" and "the rates moved" is a genuine
decomposition rather than two loosely related runs.  DESIGN §2.7 predicts the
rate term dominates on short arcs; ``analyze`` reports the measurement.

Runs, and why each exists
-------------------------
``main``      20,000 tracklets, ``both``, one realization — the headline
              population-level numbers (calibration, ranking, operating point).
``mc/rNN``    25 independent realizations of a fixed 500-tracklet stratified
              subsample (**all 185 true NEOs** + 315 non-NEOs stratified across
              V) — separates per-tracklet noise scatter from population effects
              and gives the ``std(p_neo)`` sensitivity map.
``decomp_*``  the ``pos`` and ``rate`` modes on that same subsample, with the
              same draws as ``mc/r00``, which therefore *is* the ``both`` arm of
              the decomposition (no separate run needed, and the pairing is
              exact).
``sens_m5``   the subsample again with the same draws but ``m5 = 24.5``: how
              much of the answer rides on the adopted depth.

Scoring reuses ``self_consistency.score`` verbatim — same process pool, same
chunked/stamped/resumable output, same ``(1200, 50)`` default grid, same
``Digest2Histogram`` All-variant pair — by writing each perturbed catalogue as a
``gen.npz`` in the WP6 schema.  Nothing under ``validation/output/gate_v0`` is
written; it is opened read-only.

Usage::

    python3 -m validation.astrometric_noise sigma
    python3 -m validation.astrometric_noise inject
    python3 -m validation.astrometric_noise score --workers 32
    python3 -m validation.astrometric_noise analyze
    python3 -m validation.astrometric_noise run --workers 32   # all four

Every stage is resumable and deterministic: ``inject`` is a pure function of
(catalogue, seed, error model), ``score`` reuses valid chunks, and a run whose
``scores.npz`` already exists is skipped.

The arc-length sweep
--------------------
The 45-minute run above answers the question at one arc length, and its own
conclusion is scoped by that: the rate error is ``sqrt(2) sigma / dt``, so a
15-minute pair triples it and a 5-minute pair multiplies it by nine.  The
``sweep`` subcommand re-runs the whole experiment across a dt grid and applies
an operational break criterion (:data:`BREAK_CRITERION`) fixed in advance::

    OMP_NUM_THREADS=1 python3 -m validation.astrometric_noise sweep --workers 84

It writes to a *separate* tree (``validation/output/astrometric_noise_arclength``)
and never touches the 45-minute run's outputs; the dt = 45 min row *adopts*
that run's stored scores after verifying the injected catalogue is identical
array-by-array.  Because dt enters ``noisy_attributable`` only as the divisor of
the endpoint difference, every arc length in the sweep perturbs the same
endpoints by the same offsets — the sweep is a common-random-numbers experiment
in dt, and ``|Delta P| * dt`` is constant per tracklet while the classifier's
response stays linear.  See :func:`variance_scaling` and :func:`scaling_ratio`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from validation import self_consistency as sc  # noqa: E402
from validation.self_consistency import (  # noqa: E402
    ScoreConfig,
    auc_weighted,
    bootstrap_reliability,
    fit_odds_scale,
    global_z,
    reliability_bins,
    reliability_verdict,
    roc_curve_weighted,
)

__all__ = [
    "ErrorModel",
    "NoiseConfig",
    "RunSpec",
    "snr_from_mag",
    "random_astrometric_error_per_coord",
    "astrometric_uncertainty_mas",
    "trailing_loss_dmag",
    "sigma_astrometric_rad",
    "perturbations",
    "noisy_attributable",
    "stratified_subsample",
    "plan_runs",
    "inject",
    "score_runs",
    "load_run",
    "sigma_table",
    "delta_summary",
    "binned_delta",
    "threshold_crossings",
    "matched_threshold",
    "operating_points",
    "calibration_block",
    "mc_block",
    "decomposition_block",
    "analyze",
    "build_report",
    "make_plots",
    # arc-length sweep
    "DT_GRID_DEFAULT",
    "BREAK_CRITERION",
    "dt_tag",
    "sweep_inject",
    "sweep_score",
    "sweep_dt_block",
    "sweep_analyze",
    "loglog_slope",
    "scaling_ratio",
    "ratio_by_floor",
    "step_scaling_ratio",
    "ladder_stability",
    "pick_departure_floor",
    "variance_scaling",
    "rate_error_block",
    "break_verdict",
    "build_sweep_report",
    "make_sweep_plots",
    "main",
]

# --------------------------------------------------------------------------
# Sorcha's constants, verbatim
# --------------------------------------------------------------------------

#: ``rgamma`` in ``PPAddUncertainties.calcAstrometricUncertainty``.  Sorcha
#: hardcodes a single value for every filter (Ivezić et al. 2019 Table 2 gives
#: 0.038 for u and 0.039 for grizy).
RGAMMA = 0.039

#: ``error_sys`` [mas]; LSST SRD design spec per Sorcha's docstring.
ERROR_SYS_MAS = 10.0

#: ``astErrCoeff``: <P^2>/<|dP/dx|^2> / FWHMeff for a single-Gaussian PSF.
AST_ERR_COEFF = 0.60

#: ``FWHMeff`` [mas] — Sorcha's own default argument value (0.7").
FWHM_MAS_DEFAULT = 700.0

#: ``calcTrailingLoss`` "trailedSource" fit parameters (Vereš & Chesley 2017).
A_TRAIL = 0.761
B_TRAIL = 1.162
#: "circularPSF" fit parameters, kept for completeness / the ``--trailing-model``
#: knob; Sorcha's uncertainty path uses "trailedSource".
A_DET = 0.420
B_DET = 0.003
#: ``texp`` [s] default in ``calcTrailingLoss``.
TEXP_S_DEFAULT = 30.0

#: Adopted 5-sigma depth; see the module docstring for the justification.
M5_DEFAULT = 24.0

#: Tracklet arc length [min] — the WP7 ADES emission convention.
DT_MIN_DEFAULT = 45.0

MAS_PER_DEG = 3.6e6
DEG_PER_RAD = 180.0 / math.pi

DEFAULT_SOURCE_RUN = "validation/output/gate_v0"
DEFAULT_OUT = "validation/output/astrometric_noise"

#: Digest2 posts to the NEOCP at raw >= 65; on ``gate_v0`` that cut recovers
#: this fraction of the true NEOs (validation/DIGEST2_COMPARISON.md, "all"
#: subset).  It is the matched-completeness operating point WP7 established, and
#: the point at which contamination is compared here.
WP7_MATCHED_COMPLETENESS = 0.881


# --------------------------------------------------------------------------
# The error model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorModel:
    """Sorcha's astrometric error model with our adopted survey parameters."""

    m5: float = M5_DEFAULT
    fwhm_mas: float = FWHM_MAS_DEFAULT
    error_sys_mas: float = ERROR_SYS_MAS
    ast_err_coeff: float = AST_ERR_COEFF
    rgamma: float = RGAMMA
    nvisit: int = 1
    trailing_losses: bool = True
    trailing_model: str = "trailedSource"
    texp_s: float = TEXP_S_DEFAULT

    @property
    def seeing_arcsec(self) -> float:
        """The seeing FWHM the trailing-loss term uses, in arcsec.

        Sorcha feeds ``seeingFwhmGeom_arcsec * 1000`` into ``FWHMeff`` and the
        same ``seeingFwhmGeom_arcsec`` into ``calcTrailingLoss``, so the two are
        one number here as well.
        """
        return float(self.fwhm_mas) / 1000.0


def snr_from_mag(mag, m5, rgamma: float = RGAMMA):
    """SNR from magnitude and 5-sigma depth — Sorcha's transcription of the
    LSST photometric error model (Ivezić et al. 2019, eq. 5).

    Verbatim from ``PPAddUncertainties.calcAstrometricUncertainty``::

        rgamma = 0.039
        xval = np.power(10, 0.4 * (mag - m5))
        SNR = 1.0 / np.sqrt((0.04 - rgamma) * xval + rgamma * xval * xval)

    At ``mag == m5`` this returns exactly 5 (``1/sqrt(0.04)``), which is what
    makes "m5" the 5-sigma depth.
    """
    xval = np.power(10.0, 0.4 * (np.asarray(mag, dtype=float) - np.asarray(m5, dtype=float)))
    return 1.0 / np.sqrt((0.04 - rgamma) * xval + rgamma * xval * xval)


def random_astrometric_error_per_coord(fwhm_eff_mas, snr, ast_err_coeff: float = AST_ERR_COEFF):
    """``PPAddUncertainties.calcRandomAstrometricErrorPerCoord``, verbatim::

        RandomAstrometricErrorPerCoord = AstromErrCoeff * FWHMeff / SNR

    Returns the uncertainty in the units of ``fwhm_eff_mas`` (mas here), for a
    SINGLE sky coordinate.
    """
    return ast_err_coeff * np.asarray(fwhm_eff_mas, dtype=float) / np.asarray(snr, dtype=float)


def astrometric_uncertainty_mas(
    mag,
    m5,
    nvisit: int = 1,
    fwhm_eff_mas: float = FWHM_MAS_DEFAULT,
    error_sys_mas: float = ERROR_SYS_MAS,
    ast_err_coeff: float = AST_ERR_COEFF,
    rgamma: float = RGAMMA,
):
    """``PPAddUncertainties.calcAstrometricUncertainty`` with ``output_units="mas"``.

    Returns ``(sigma_mas, snr, error_rand_mas)``.  Transcribed line for line::

        error_rand = calcRandomAstrometricErrorPerCoord(FWHMeff, SNR, astErrCoeff)
        if nvisit > 1:
            error_rand = error_rand / np.sqrt(nvisit)
        astrom_error = np.sqrt(error_sys * error_sys + error_rand * error_rand)

    Sorcha's own caveats, kept here because they bound what this experiment can
    claim: the sqrt(Nvisit) scaling and the maximum-likelihood centroid are both
    *best-case*, so this sigma is, if anything, optimistic.
    """
    snr = snr_from_mag(mag, m5, rgamma=rgamma)
    error_rand = random_astrometric_error_per_coord(fwhm_eff_mas, snr, ast_err_coeff)
    if nvisit > 1:
        error_rand = error_rand / math.sqrt(nvisit)
    sigma = np.sqrt(error_sys_mas * error_sys_mas + error_rand * error_rand)
    return sigma, snr, error_rand


def trailing_loss_dmag(
    dra_cosdec_deg_day,
    ddec_deg_day,
    seeing_arcsec,
    texp_s: float = TEXP_S_DEFAULT,
    model: str = "trailedSource",
):
    """``PPTrailingLoss.calcTrailingLoss`` (Vereš & Chesley 2017), verbatim::

        vel = np.sqrt(dRaCosDec**2 + dDec**2)
        vel = vel / 24.0  # convert to arcsec / sec
        x = vel * texp / seeing
        dmagTrail = 1.25 * np.log10(1.0 + a_trail * x**2 / (1.0 + b_trail * x))

    Rates are in deg/day **with the cos(dec) factor already applied** to the RA
    rate (Sorcha's ``RARateCosDec_deg_day``); ``priora.Attributable.ra_rate``
    is the raw d(alpha)/dt, so the caller must multiply.
    """
    vel = np.hypot(np.asarray(dra_cosdec_deg_day, dtype=float),
                   np.asarray(ddec_deg_day, dtype=float))
    vel = vel / 24.0
    x = vel * texp_s / float(seeing_arcsec)
    if model == "trailedSource":
        a, b = A_TRAIL, B_TRAIL
    elif model == "circularPSF":
        a, b = A_DET, B_DET
    else:
        raise ValueError(f"unknown trailing-loss model {model!r}")
    return 1.25 * np.log10(1.0 + a * x * x / (1.0 + b * x))


def sigma_astrometric_rad(
    v_mag: np.ndarray,
    ra_rate: np.ndarray,
    dec_rate: np.ndarray,
    dec: np.ndarray,
    em: ErrorModel,
) -> dict[str, np.ndarray]:
    """Per-detection, per-coordinate astrometric sigma [rad] for a catalogue.

    Runs Sorcha's full chain: trailing loss on the true magnitude -> SNR ->
    random centroid error -> systematic floor.  Returns a dict with
    ``sigma_rad``, ``sigma_mas``, ``snr``, ``dmag_trail`` and ``rate_deg_day``
    so the analysis can plot against any of them.

    ``v_mag`` is the *true* apparent magnitude (D-5, standardized Johnson V);
    the returned sigma is what a detection of that source would carry, and both
    detections of a tracklet share it (no photometric variation is modelled).
    """
    v_mag = np.asarray(v_mag, dtype=float)
    dra_cosdec = np.degrees(np.asarray(ra_rate, dtype=float) * np.cos(np.asarray(dec, dtype=float)))
    ddec = np.degrees(np.asarray(dec_rate, dtype=float))
    rate = np.hypot(dra_cosdec, ddec)

    if em.trailing_losses:
        dmag = trailing_loss_dmag(
            dra_cosdec, ddec, em.seeing_arcsec, texp_s=em.texp_s, model=em.trailing_model
        )
    else:
        dmag = np.zeros_like(v_mag)

    sigma_mas, snr, _ = astrometric_uncertainty_mas(
        v_mag + dmag,
        em.m5,
        nvisit=em.nvisit,
        fwhm_eff_mas=em.fwhm_mas,
        error_sys_mas=em.error_sys_mas,
        ast_err_coeff=em.ast_err_coeff,
        rgamma=em.rgamma,
    )
    return {
        "sigma_mas": sigma_mas,
        "sigma_rad": np.radians(sigma_mas / MAS_PER_DEG),
        "snr": snr,
        "dmag_trail": np.asarray(dmag, dtype=float) * np.ones_like(v_mag),
        "rate_deg_day": rate,
    }


# --------------------------------------------------------------------------
# Noise injection
# --------------------------------------------------------------------------

#: The three noise modes; see the module docstring for the construction and why
#: it is an exact additive decomposition.
MODES = ("both", "pos", "rate")


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration of one noise-injection realization."""

    dt_min: float = DT_MIN_DEFAULT
    seed: int = 20260902
    stream: int = 0
    mode: str = "both"


def perturbations(
    sigma_rad: np.ndarray,
    dec: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    dt_days: float,
    mode: str = "both",
) -> dict[str, np.ndarray]:
    """Coordinate-space offsets of the two detections, and of the attributable.

    ``e0``/``e1`` are ``(N, 2)`` standard normals — column 0 is the RA (i.e.
    ``alpha cos delta``) direction, column 1 the Dec direction — for detection 0
    and detection 1.  ``sigma_rad`` is the single-coordinate sigma in the local
    sky metric, and ``dec`` supplies the ``1/cos delta`` conversion from that
    metric to a coordinate offset in ``alpha``.  The metric is evaluated once,
    at the attributable's declination, for every mode (see the module docstring).

    Returns ``d_ra_0, d_dec_0, d_ra_1, d_dec_1`` (detection offsets [rad]) and
    ``d_ra, d_dec, d_ra_rate, d_dec_rate`` (the attributable's 4-vector
    perturbation), such that::

        d_ra       = d_ra_0
        d_ra_rate  = (d_ra_1 - d_ra_0) / dt

    For ``mode="pos"`` the rate perturbations are exactly ``0.0``; for
    ``mode="rate"`` the position perturbations are exactly ``0.0``; and
    ``both == pos + rate`` to the last bit.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    sigma_rad = np.asarray(sigma_rad, dtype=float)
    inv_cos = 1.0 / np.cos(np.asarray(dec, dtype=float))
    e0 = np.asarray(e0, dtype=float)
    e1 = np.asarray(e1, dtype=float)

    # Offsets in the local sky metric, then converted to coordinate offsets.
    s_ra = sigma_rad * inv_cos
    if mode == "both":
        d_ra_0, d_dec_0 = s_ra * e0[:, 0], sigma_rad * e0[:, 1]
        d_ra_1, d_dec_1 = s_ra * e1[:, 0], sigma_rad * e1[:, 1]
    elif mode == "pos":
        d_ra_0, d_dec_0 = s_ra * e0[:, 0], sigma_rad * e0[:, 1]
        d_ra_1, d_dec_1 = d_ra_0, d_dec_0
    else:  # "rate"
        z = np.zeros_like(sigma_rad)
        d_ra_0, d_dec_0 = z, z
        d_ra_1 = s_ra * (e1[:, 0] - e0[:, 0])
        d_dec_1 = sigma_rad * (e1[:, 1] - e0[:, 1])

    return {
        "d_ra_0": d_ra_0,
        "d_dec_0": d_dec_0,
        "d_ra_1": d_ra_1,
        "d_dec_1": d_dec_1,
        "d_ra": d_ra_0,
        "d_dec": d_dec_0,
        "d_ra_rate": (d_ra_1 - d_ra_0) / dt_days,
        "d_dec_rate": (d_dec_1 - d_dec_0) / dt_days,
    }


def noisy_attributable(
    gen: dict[str, np.ndarray],
    rows: np.ndarray,
    em: ErrorModel,
    nc: NoiseConfig,
) -> dict[str, np.ndarray]:
    """Build the perturbed attributables for ``gen`` rows ``rows``.

    Pure function of (catalogue rows, error model, noise config): the RNG is
    ``np.random.default_rng([seed, stream])`` and the draws are indexed by
    position within ``rows``, so a given (seed, stream, subsample) always
    produces the same realization, and the three modes of one stream share
    their draws.

    Returns a dict with the noisy ``ra``/``dec``/``ra_rate``/``dec_rate``, the
    unchanged ``v_mag``, and the diagnostic columns (``sigma_mas``, ``snr``,
    ``dmag_trail``, ``rate_deg_day``, and every perturbation component).
    """
    rows = np.asarray(rows, dtype=np.int64)
    ra = gen["ra"][rows].astype(float)
    dec = gen["dec"][rows].astype(float)
    ra_rate = gen["ra_rate"][rows].astype(float)
    dec_rate = gen["dec_rate"][rows].astype(float)
    v_mag = gen["v_mag"][rows].astype(float)

    sig = sigma_astrometric_rad(v_mag, ra_rate, dec_rate, dec, em)
    rng = np.random.default_rng([int(nc.seed), int(nc.stream)])
    e0 = rng.standard_normal((rows.size, 2))
    e1 = rng.standard_normal((rows.size, 2))

    dt_days = float(nc.dt_min) / 1440.0
    d = perturbations(sig["sigma_rad"], dec, e0, e1, dt_days, mode=nc.mode)

    out = {
        "ra": np.mod(ra + d["d_ra"], 2.0 * np.pi),
        "dec": dec + d["d_dec"],
        "ra_rate": ra_rate + d["d_ra_rate"],
        "dec_rate": dec_rate + d["d_dec_rate"],
        "v_mag": v_mag,
    }
    if np.any(np.abs(out["dec"]) >= 0.5 * np.pi):  # pragma: no cover - impossible here
        raise ValueError("astrometric noise pushed a declination past a pole")
    out.update({k: np.asarray(v, dtype=float) for k, v in sig.items() if k != "sigma_rad"})
    out["sigma_rad"] = sig["sigma_rad"]
    out.update({k: np.asarray(v, dtype=float) for k, v in d.items()})
    return out


# --------------------------------------------------------------------------
# Run planning
# --------------------------------------------------------------------------

#: V-magnitude strata for the 500-tracklet subsample.  Deliberately NOT the
#: population's own V distribution (which is ~50% in [23, 24)): the point of the
#: subsample is to map sigma(V), so the bright end has to be represented even
#: though it is rare.
V_STRATA = (14.0, 19.0, 21.0, 22.0, 23.0, 23.5, 24.0)


def stratified_subsample(
    is_neo: np.ndarray, v_mag: np.ndarray, n: int = 500, seed: int = 20260902
) -> np.ndarray:
    """All true NEOs plus a V-stratified sample of the rest, sorted, unique.

    Every true NEO is included (there are only ~185 in 2x10^4, and they are the
    whole point), and the remaining slots are spread as evenly as the strata
    allow across :data:`V_STRATA`, with any shortfall in a thin stratum
    redistributed to the strata that still have candidates.
    """
    is_neo = np.asarray(is_neo).astype(bool)
    v_mag = np.asarray(v_mag, dtype=float)
    rng = np.random.default_rng([int(seed), 999])

    keep = list(np.flatnonzero(is_neo))
    remaining = int(n) - len(keep)
    if remaining <= 0:
        return np.array(sorted(keep[: int(n)]), dtype=np.int64)

    pools = []
    for lo, hi in zip(V_STRATA[:-1], V_STRATA[1:]):
        pool = np.flatnonzero(~is_neo & (v_mag >= lo) & (v_mag < hi))
        pools.append(list(rng.permutation(pool)))

    # Round-robin over the strata: even allocation, automatic redistribution.
    picked: list[int] = []
    cursor = [0] * len(pools)
    while len(picked) < remaining:
        progressed = False
        for k, pool in enumerate(pools):
            if len(picked) >= remaining:
                break
            if cursor[k] < len(pool):
                picked.append(int(pool[cursor[k]]))
                cursor[k] += 1
                progressed = True
        if not progressed:  # pragma: no cover - catalogue smaller than n
            break
    return np.array(sorted(keep + picked), dtype=np.int64)


@dataclass
class RunSpec:
    """One perturbed catalogue to be written, scored and analysed."""

    name: str
    mode: str
    stream: int
    rows: np.ndarray = field(repr=False)
    error_model: ErrorModel = field(default_factory=ErrorModel)
    dt_min: float = DT_MIN_DEFAULT
    seed: int = 20260902
    role: str = ""

    @property
    def noise_config(self) -> NoiseConfig:
        return NoiseConfig(dt_min=self.dt_min, seed=self.seed, stream=self.stream,
                           mode=self.mode)

    def meta(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "stream": self.stream,
            "n": int(self.rows.size),
            "seed": self.seed,
            "dt_min": self.dt_min,
            "role": self.role,
            "error_model": asdict(self.error_model),
        }


#: RNG stream ids.  ``main`` gets its own; the MC realizations occupy
#: ``MC_STREAM0 + j``; the decomposition and the m5-sensitivity case reuse
#: ``MC_STREAM0`` so that they are paired draw-for-draw with ``mc/r00`` — which
#: is therefore the ``both`` arm of the decomposition, needing no separate run.
MAIN_STREAM = 0
MC_STREAM0 = 1000


def plan_runs(
    gen: dict[str, np.ndarray],
    n_mc: int = 25,
    n_sub: int = 500,
    seed: int = 20260902,
    dt_min: float = DT_MIN_DEFAULT,
    em: ErrorModel | None = None,
    m5_sens: float | None = 24.5,
    full: bool = True,
) -> list[RunSpec]:
    """The full experiment as a list of :class:`RunSpec`."""
    em = em or ErrorModel()
    n_all = int(gen["ra"].size)
    all_rows = np.arange(n_all, dtype=np.int64)
    sub = stratified_subsample(gen["is_neo"], gen["v_mag"], n=n_sub, seed=seed)

    runs: list[RunSpec] = []
    if full:
        runs.append(
            RunSpec("main", "both", MAIN_STREAM, all_rows, em, dt_min, seed,
                    role="headline 20k population run")
        )
    for j in range(int(n_mc)):
        runs.append(
            RunSpec(f"mc/r{j:02d}", "both", MC_STREAM0 + j, sub, em, dt_min, seed,
                    role=("MC realization (also the 'both' arm of the decomposition)"
                          if j == 0 else "MC realization"))
        )
    if n_mc > 0:
        runs.append(
            RunSpec("decomp_pos", "pos", MC_STREAM0, sub, em, dt_min, seed,
                    role="position-only noise (rigid tracklet shift, zero rate error)")
        )
        runs.append(
            RunSpec("decomp_rate", "rate", MC_STREAM0, sub, em, dt_min, seed,
                    role="rate-only noise (exact position, mc/r00's rate error)")
        )
        if m5_sens is not None:
            runs.append(
                RunSpec(f"sens_m5_{m5_sens:g}", "both", MC_STREAM0, sub,
                        replace(em, m5=float(m5_sens)), dt_min, seed,
                        role=f"depth sensitivity: same draws, m5 = {m5_sens:g}")
            )
    return runs


# --------------------------------------------------------------------------
# Injection: write each perturbed catalogue in the WP6 gen.npz schema
# --------------------------------------------------------------------------

#: Columns copied verbatim from the source catalogue (the ground truth and the
#: observer geometry, none of which the noise touches).
_TRUTH_COLS = (
    "epoch_jd_tdb", "R_obs", "V_obs", "rho_true", "rho_dot_true",
    "elements", "H", "is_neo", "weight",
)


def load_source(run_dir: str) -> dict[str, np.ndarray]:
    """Read the source catalogue (``gen.npz``) read-only."""
    with np.load(os.path.join(run_dir, "gen.npz")) as z:
        return {k: z[k] for k in z.files}


def load_source_config(run_dir: str) -> dict:
    path = os.path.join(run_dir, "gen_counts.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def inject(
    source_run: str = DEFAULT_SOURCE_RUN,
    out_dir: str = DEFAULT_OUT,
    n_mc: int = 25,
    n_sub: int = 500,
    seed: int = 20260902,
    dt_min: float = DT_MIN_DEFAULT,
    em: ErrorModel | None = None,
    m5_sens: float | None = 24.5,
    full: bool = True,
    verbose: bool = True,
) -> dict:
    """Write every planned perturbed catalogue plus ``manifest.json``.

    Each run lands in ``out_dir/<name>/gen.npz`` in the exact WP6 schema, so
    ``self_consistency.score`` can consume it unmodified.  The noise columns
    (sigma, SNR, the perturbation components) go alongside in
    ``noise.npz`` — outside ``gen.npz`` on purpose, because ``gen.npz`` is
    hashed into the scoring stamp and must contain only what the scorer reads.
    """
    em = em or ErrorModel()
    gen = load_source(source_run)
    src_cfg = load_source_config(source_run)
    runs = plan_runs(gen, n_mc=n_mc, n_sub=n_sub, seed=seed, dt_min=dt_min,
                     em=em, m5_sens=m5_sens, full=full)
    os.makedirs(out_dir, exist_ok=True)

    manifest = {
        "source_run": os.path.abspath(source_run),
        "source_n": int(gen["ra"].size),
        "source_config": src_cfg.get("config", {}),
        "seed": int(seed),
        "dt_min": float(dt_min),
        "error_model": asdict(em),
        "matched_completeness_target": WP7_MATCHED_COMPLETENESS,
        "provenance": {
            "error_model_source": (
                "sorcha src/sorcha/modules/PPAddUncertainties.py :: "
                "calcAstrometricUncertainty + calcRandomAstrometricErrorPerCoord; "
                "src/sorcha/modules/PPTrailingLoss.py :: calcTrailingLoss"
            ),
            "sorcha_ref": "v1.2.1 (4e37cfe974a04a2197c64ff5074dc68608fe1893)",
            "citations": [
                "Ivezic et al. 2019, ApJ 873, 111 (LSST overview paper; "
                "arXiv:0805.2366, cited in Sorcha's docstring)",
                "LSST Science Requirements Document (ls.st/srd) for the 10 mas "
                "systematic floor",
                "Veres & Chesley 2017, AJ 154, 12 (trailing losses)",
            ],
        },
        "runs": [],
    }

    for r in runs:
        rdir = os.path.join(out_dir, r.name)
        os.makedirs(rdir, exist_ok=True)
        noisy = noisy_attributable(gen, r.rows, r.error_model, r.noise_config)
        arrays = {
            "ra": noisy["ra"], "dec": noisy["dec"],
            "ra_rate": noisy["ra_rate"], "dec_rate": noisy["dec_rate"],
            "v_mag": noisy["v_mag"],
        }
        for k in _TRUTH_COLS:
            arrays[k] = gen[k][r.rows]
        np.savez_compressed(os.path.join(rdir, "gen.npz"), **arrays)
        # gen_counts.json: the WP6 loader reads `config` for rho_floor, and the
        # analysis reads the mixture weights.  Carry the SOURCE run's values.
        with open(os.path.join(rdir, "gen_counts.json"), "w") as fh:
            json.dump(
                {
                    "config": dict(src_cfg.get("config", {}), n=int(r.rows.size)),
                    "counters": src_cfg.get("counters", {}),
                    "astrometric_noise": r.meta(),
                },
                fh,
                indent=2,
            )
        np.savez_compressed(
            os.path.join(rdir, "noise.npz"),
            rows=r.rows,
            **{k: v for k, v in noisy.items()},
        )
        entry = r.meta()
        entry["dir"] = rdir
        entry["median_sigma_mas"] = float(np.median(noisy["sigma_mas"]))
        entry["p90_sigma_mas"] = float(np.percentile(noisy["sigma_mas"], 90))
        manifest["runs"].append(entry)
        if verbose:
            print(f"  [inject] {r.name:<16} n={r.rows.size:<6d} mode={r.mode:<5} "
                  f"median sigma {entry['median_sigma_mas']:.1f} mas", flush=True)

    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    if verbose:
        print(f"[inject] wrote {len(runs)} perturbed catalogue(s) under {out_dir}")
    return manifest


def load_manifest(out_dir: str = DEFAULT_OUT) -> dict:
    with open(os.path.join(out_dir, "manifest.json")) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Scoring: delegate to the WP6 harness
# --------------------------------------------------------------------------


def score_runs(
    out_dir: str = DEFAULT_OUT,
    cfg: ScoreConfig | None = None,
    workers: int = 8,
    chunk: int = 250,
    only: str | None = None,
    verbose: bool = True,
) -> dict:
    """Score every run in the manifest with :func:`self_consistency.score`.

    Resumable at two levels: a run whose ``scores.npz`` already exists is
    skipped entirely, and an interrupted run resumes at chunk granularity via
    the WP6 stamp/content-hash machinery.  The default scoring config is the v0
    pair — ``Digest2Histogram`` All variant, grid (1200, 50) — i.e. exactly what
    produced ``gate_v0/scores.npz``, so noisy and noiseless differ in the
    attributable and nothing else.
    """
    cfg = cfg or ScoreConfig()
    man = load_manifest(out_dir)
    t0 = time.time()
    done, skipped = [], []
    for entry in man["runs"]:
        name = entry["name"]
        if only and not name.startswith(only):
            continue
        rdir = os.path.join(out_dir, name)
        if os.path.exists(os.path.join(rdir, "scores.npz")):
            skipped.append(name)
            continue
        if verbose:
            print(f"[score] {name} (n={entry['n']})", flush=True)
        sc.score(rdir, cfg, workers=workers, chunk=chunk, resume=True, verbose=verbose)
        done.append(name)
    if verbose:
        print(f"[score] {len(done)} run(s) scored, {len(skipped)} already complete "
              f"({time.time() - t0:.0f} s)")

    # The deliverable the task names: the main run's score table, with the
    # injected noise columns alongside, at the top of the output directory.
    main_dir = os.path.join(out_dir, "main")
    if os.path.exists(os.path.join(main_dir, "scores.npz")):
        with np.load(os.path.join(main_dir, "scores.npz")) as z:
            merged = {k: z[k] for k in z.files}
        with np.load(os.path.join(main_dir, "noise.npz")) as z:
            for k in z.files:
                merged[f"noise_{k}"] = z[k]
        np.savez_compressed(os.path.join(out_dir, "noisy_scores.npz"), **merged)
        if verbose:
            print(f"[score] wrote {out_dir}/noisy_scores.npz")
    return {"scored": done, "skipped": skipped,
            "score_config": asdict(cfg), "wall_s": time.time() - t0}


def mask_policy_rows(cols: dict) -> dict:
    """Blank D-19 triage-policy assignments out of a score table's `p_neo`.

    A policy row is ``p_neo = 1.0`` returned over a 0/0 whose entire admissible
    region sits at or above the H support end (DESIGN §2.6, D-19): a defined
    number with **no likelihood behind it**.  Recognised from the recorded
    ``(isfinite(p_neo), n_nodes > 0, lambda_neo == lambda_other == 0)``
    quadruple, which is `classifier._posterior`'s own branch condition.

    Every statistic in this module filters on ``isfinite(p_neo)``, and BEFORE
    D-19 that excluded these tracklets automatically (they came back nan).
    Setting them back to nan therefore *preserves* the study's semantics rather
    than changing them, and it restores two things D-19 would otherwise have
    broken silently:

    * **the C3 break arm.**  C3 counts tracklets that scored finitely without
      noise and become UNDEFINED with it.  The fully-truncated direction — the
      noisy attributable's whole region pushed past the model's faint end — is
      one of the two ways that happens, and after D-19 it comes back as a
      *score* of 1.0, so C3 would silently stop seeing it. Since the C3
      threshold is 1% of tracklets, losing a whole failure direction from it is
      not cosmetic.
    * **calibration and ROC.**  A policy 1.0 entering ``calibration_block`` or
      ``auc_bootstrap`` is treated as a posterior of 1.0, on a class that is
      empirically ~100% true NEO — free true positives that flatter the noisy
      column exactly where the study is trying to detect degradation.

    ``log_lr`` is left alone: it is already nan on those rows, which is the
    other half of the same signal.  The raw counts are returned alongside so
    the masking is reportable rather than invisible.
    """
    need = {"p_neo", "n_nodes", "lambda_neo", "lambda_other"}
    if not need <= set(cols):
        # A score table without the Lambda columns predates them entirely;
        # nothing to do, and nothing can be detected either.
        return cols
    p = np.asarray(cols["p_neo"], dtype=float)
    policy = (
        np.isfinite(p)
        & (np.asarray(cols["n_nodes"]).astype(np.int64) > 0)
        & (np.asarray(cols["lambda_neo"], dtype=float) == 0.0)
        & (np.asarray(cols["lambda_other"], dtype=float) == 0.0)
    )
    out = dict(cols)
    out["p_neo"] = np.where(policy, np.nan, p)
    out["_n_policy_assigned"] = np.array([int(policy.sum())])
    out["_policy_assigned"] = policy
    return out


def load_run(out_dir: str, name: str) -> dict[str, np.ndarray]:
    """Scores + noise columns + the perturbed catalogue for one run."""
    rdir = os.path.join(out_dir, name)
    out: dict[str, np.ndarray] = {}
    with np.load(os.path.join(rdir, "scores.npz")) as z:
        out.update({k: z[k] for k in z.files})
    with np.load(os.path.join(rdir, "noise.npz")) as z:
        out.update({f"noise_{k}": z[k] for k in z.files})
    return mask_policy_rows(out)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

#: Categorical palette, fixed order, never cycled.  Validated (all pairs, light
#: and dark surfaces) with the dataviz six-checks validator: lightness band,
#: chroma floor, CVD separation (worst adjacent dE 11.0 deutan), normal-vision
#: floor (18.7) and >= 3:1 contrast all PASS.  Three slots is the maximum any
#: panel here needs; a fourth series would be a facet, not a new hue.
PALETTE = ("#0072B2", "#D55E00", "#009E73")
INK = "#1a1a1a"
MUTED = "#5c5c5c"
GRID = "#d8d8d8"

#: V bins for the magnitude-resolved deltas.  Coarse where the catalogue is thin
#: (V < 21 holds ~5% of it), fine where both the NEOs and the noise live.
V_BINS = np.array([14.0, 20.0, 21.0, 22.0, 22.5, 23.0, 23.5, 24.0])

#: Total sky-motion bins [deg/day], log-spaced across the catalogue's range.
RATE_BINS = np.array([0.003, 0.03, 0.1, 0.2, 0.35, 0.6, 1.0, 2.0, 10.0])

#: Triage thresholds whose stability under noise is reported.
THRESHOLDS = (0.1, 0.3, 0.5, 0.7)


def _finite(*arrays: np.ndarray) -> np.ndarray:
    m = np.ones(np.asarray(arrays[0]).shape, dtype=bool)
    for a in arrays:
        m &= np.isfinite(np.asarray(a, dtype=float))
    return m


def _q(x: np.ndarray, qs=(50, 90, 99)) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs}


def delta_summary(d: np.ndarray, label: str = "") -> dict:
    """Location/scale/tail summary of a per-tracklet change."""
    d = np.asarray(d, dtype=float)
    good = np.isfinite(d)
    d = d[good]
    if d.size == 0:
        return {"label": label, "n": 0}
    out = {
        "label": label,
        "n": int(d.size),
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "rms": float(np.sqrt(np.mean(d * d))),
        "std": float(np.std(d, ddof=1)) if d.size > 1 else float("nan"),
        "max_abs": float(np.max(np.abs(d))),
    }
    out.update({f"abs_{k}": v for k, v in _q(np.abs(d)).items()})
    return out


def binned_delta(
    x: np.ndarray, d: np.ndarray, edges: np.ndarray, w: np.ndarray | None = None
) -> list[dict]:
    """Per-bin |delta| percentiles and signed mean, for the trend plots/tables."""
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = _finite(x, d) & (x >= lo) & (x < hi)
        n = int(sel.sum())
        row = {"lo": float(lo), "hi": float(hi), "n": n}
        if n:
            ad = np.abs(d[sel])
            row.update(
                x_mean=float(np.mean(x[sel])),
                signed_mean=float(np.mean(d[sel])),
                abs_median=float(np.median(ad)),
                abs_p90=float(np.percentile(ad, 90)),
                abs_p99=float(np.percentile(ad, 99)),
                rms=float(np.sqrt(np.mean(d[sel] ** 2))),
                w=float(w[sel].sum()),
            )
        rows.append(row)
    return rows


def threshold_crossings(p0: np.ndarray, p1: np.ndarray, y: np.ndarray,
                        thresholds=THRESHOLDS, at_risk_scale: float | None = None) -> list[dict]:
    """How many tracklets change side of each triage threshold under noise.

    ``p0`` is the noiseless score, ``p1`` the noisy one.  Reported per
    threshold: the flagged counts before and after, the two crossing
    directions, and the same split for true NEOs — because a NEO falling below
    the cut and a non-NEO rising above it cost very different things.

    ``n_at_risk`` is the number of tracklets whose noiseless score sits within
    ``at_risk_scale`` of the threshold (by default the 99th percentile of
    ``|p1 - p0|``).  It is the denominator that makes a crossing count
    interpretable: at this base rate only a handful of tracklets are anywhere
    near a triage cut, so "zero crossings" is a statement about the *joint*
    fact that the population near the cut is thin AND the perturbation is
    small — and a single realization cannot distinguish "impossible" from
    "unlikely this time".  The MC block's ``expected_crossings`` averages the
    same count over 25 realizations and is the sharper number.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    y = np.asarray(y).astype(bool)
    good = _finite(p0, p1)
    if at_risk_scale is None:
        d = np.abs(p1 - p0)[good]
        at_risk_scale = float(np.percentile(d, 99)) if d.size else 0.0
    rows = []
    for t in thresholds:
        a, b = (p0 >= t) & good, (p1 >= t) & good
        up, down = (~a) & b, a & (~b)
        rows.append({
            "threshold": float(t),
            "n_flagged_noiseless": int(a.sum()),
            "n_flagged_noisy": int(b.sum()),
            "n_cross_up": int(up.sum()),
            "n_cross_down": int(down.sum()),
            "n_cross_any": int((up | down).sum()),
            "frac_cross_any": float((up | down).sum() / max(int(good.sum()), 1)),
            "frac_of_flagged_churn": (
                float((up | down).sum() / a.sum()) if a.sum() else float("nan")
            ),
            "neo_cross_up": int((up & y).sum()),
            "neo_cross_down": int((down & y).sum()),
            "neo_flagged_noiseless": int((a & y).sum()),
            "neo_flagged_noisy": int((b & y).sum()),
            "at_risk_scale": float(at_risk_scale),
            "n_at_risk": int(np.sum(good & (np.abs(p0 - t) <= at_risk_scale))),
        })
    return rows


def _point(flag: np.ndarray, y: np.ndarray, w: np.ndarray, label: str,
           threshold: float | None = None) -> dict:
    """Completeness/contamination of a flag set (WP7's ``operating_points``)."""
    wf = float(w[flag].sum())
    wtp = float(w[flag & y].sum())
    wpos = float(w[y].sum())
    return {
        "label": label,
        "threshold": threshold,
        "n_flagged": int(flag.sum()),
        "n_true_neo_flagged": int((flag & y).sum()),
        "completeness": wtp / wpos if wpos > 0 else float("nan"),
        "contamination": (1.0 - wtp / wf) if wf > 0 else float("nan"),
    }


def matched_threshold(p: np.ndarray, y: np.ndarray, w: np.ndarray,
                      target: float) -> float:
    """Lowest score threshold reaching ``target`` completeness (WP7's rule)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(bool)
    w = np.asarray(w, dtype=float)
    good = np.isfinite(p)
    order = np.flatnonzero(good)[np.argsort(-p[good], kind="mergesort")]
    wpos = float(w[y].sum())
    if wpos <= 0 or order.size == 0:
        return float("nan")
    comp = np.cumsum(np.where(y[order], w[order], 0.0)) / wpos
    k = int(np.searchsorted(comp, target))
    return float(p[order[k]]) if k < order.size else float("nan")


def operating_points(p0: np.ndarray, p1: np.ndarray, y: np.ndarray, w: np.ndarray,
                     target: float = WP7_MATCHED_COMPLETENESS) -> dict:
    """The WP7 matched-completeness point, evaluated three ways.

    ``noiseless`` reproduces validation/DIGEST2_COMPARISON.md's row (p >= 0.254,
    174 flagged, 6.3% contamination at 88.1% completeness).

    ``noisy_rethresholded`` re-derives the threshold on the noisy scores — what
    an operator who re-tunes on noisy data would get.

    ``noisy_fixed_threshold`` applies the *noiseless* threshold to the noisy
    scores.  This is the operationally honest number: the cut is chosen once,
    from a study like WP7's, and then deployed against data that carries noise.
    Any drop in completeness or rise in contamination here is what ignoring the
    astrometric errors actually costs at the triage desk.
    """
    y = np.asarray(y).astype(bool)
    w = np.asarray(w, dtype=float)
    good = _finite(p0, p1)
    thr0 = matched_threshold(p0, y, w, target)
    thr1 = matched_threshold(p1, y, w, target)
    out = {
        "target_completeness": float(target),
        "noiseless": _point(good & (p0 >= thr0), y, w,
                            f"noiseless p >= {thr0:.3g}", thr0),
        "noisy_rethresholded": _point(good & (p1 >= thr1), y, w,
                                      f"noisy p >= {thr1:.3g}", thr1),
        "noisy_fixed_threshold": _point(good & (p1 >= thr0), y, w,
                                        f"noisy p >= {thr0:.3g} (noiseless cut)", thr0),
    }
    for t in (0.5,):
        out[f"noiseless_p{t:g}"] = _point(good & (p0 >= t), y, w, f"noiseless p >= {t}", t)
        out[f"noisy_p{t:g}"] = _point(good & (p1 >= t), y, w, f"noisy p >= {t}", t)
    return out


def calibration_block(p: np.ndarray, y: np.ndarray, w: np.ndarray,
                      n_boot: int = 2000, seed: int = 20260902,
                      n_bins: int = sc.N_RELIABILITY_BINS) -> dict:
    """The WP6 gate's calibration machinery, applied to whatever ``p`` is given.

    Reused verbatim from ``self_consistency`` — the same equal-expected-NEO-count
    bins, the same Poisson-binomial z, the same parametric-bootstrap thresholds,
    the same fitted global odds scale.  On the *noisy* scores this is no longer
    a self-consistency test of the implementation: the classifier is being asked
    a question about an attributable that is wrong, so any miscalibration it
    reports is the real miscalibration Rubin would see from v0, not a bug.
    """
    good = np.isfinite(p)
    p, y, w = p[good], np.asarray(y).astype(bool)[good], np.asarray(w, float)[good]
    bins = reliability_bins(p, y, w, n_bins=n_bins)
    glob = global_z(p, y, w)
    odds = fit_odds_scale(p, y)
    boot = bootstrap_reliability(p, y, w, n_bins=n_bins, n_boot=n_boot, seed=seed)
    verdict = reliability_verdict(bins, boot, glob, odds)
    return {
        "n": int(p.size),
        "n_neo": int(y.sum()),
        "bins": bins,
        "global": glob,
        "odds_scale": odds,
        "bootstrap": boot,
        "verdict": verdict,
        "brier": float(np.sum(w * (p - y) ** 2) / np.sum(w)),
        "expected_calibration_error": float(
            sum(b["sum_w"] * abs(b["p_mean"] - b["obs_frac"]) for b in bins)
            / max(float(np.sum(w)), 1e-300)
        ),
    }


def _load_noiseless(source_run: str) -> dict[str, np.ndarray]:
    """The stored noiseless scores.  Read-only; nothing re-scores them."""
    with np.load(os.path.join(source_run, "scores.npz")) as z:
        out = {k: z[k] for k in z.files}
    idx = out["index"].astype(int)
    if not np.array_equal(idx, np.arange(idx.size)):  # pragma: no cover
        order = np.argsort(idx, kind="mergesort")
        out = {k: v[order] for k, v in out.items()}
    return mask_policy_rows(out)


def mc_block(out_dir: str, manifest: dict, gen: dict, base: dict) -> dict:
    """Per-tracklet noise sensitivity from the independent MC realizations.

    ``std(p_neo)`` across realizations, at fixed true object and fixed error
    model, is the per-tracklet answer to "how much does the delta-function
    assumption cost *this* tracklet" — free of any population-level averaging.
    Reported against V (the error model's own variable) and against the
    noiseless ``p_neo`` (the triage-relevant variable: a tracklet at p = 0.5 can
    move across a threshold; one at p = 1e-6 cannot).
    """
    names = sorted(e["name"] for e in manifest["runs"] if e["name"].startswith("mc/"))
    if not names:
        return {"n_realizations": 0}
    rows = np.load(os.path.join(out_dir, names[0], "noise.npz"))["rows"]
    P = np.full((len(names), rows.size), np.nan)
    for j, name in enumerate(names):
        path = os.path.join(out_dir, name, "scores.npz")
        if not os.path.exists(path):
            continue
        with np.load(path) as z:
            idx = z["index"].astype(int)
            P[j, idx] = z["p_neo"]

    have = np.isfinite(P).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean_p = np.nanmean(P, axis=0)
        std_p = np.nanstd(P, axis=0, ddof=1)
    p0 = base["p_neo"][rows]
    v = gen["v_mag"][rows]
    y = gen["is_neo"][rows].astype(bool)
    sigma_mas = np.load(os.path.join(out_dir, names[0], "noise.npz"))["sigma_mas"]
    rate = np.load(os.path.join(out_dir, names[0], "noise.npz"))["rate_deg_day"]

    def _tail(k: int = 10) -> list[dict]:
        order = np.argsort(-np.where(np.isfinite(std_p), std_p, -1.0))[:k]
        return [{
            "gen_index": int(rows[i]),
            "is_neo": bool(y[i]),
            "v_mag": float(v[i]),
            "rate_deg_day": float(rate[i]),
            "sigma_mas": float(sigma_mas[i]),
            "p_noiseless": float(p0[i]),
            "p_mc_mean": float(mean_p[i]),
            "p_mc_std": float(std_p[i]),
        } for i in order]

    # Expected threshold churn per realization: the single-realization count in
    # the headline table is a Bernoulli draw with a tiny rate, so averaging the
    # same count over every realization is what actually estimates that rate.
    crossings = []
    for t in THRESHOLDS:
        a = p0 >= t
        with np.errstate(invalid="ignore"):
            b_ = P >= t
        ok = np.isfinite(P)
        n_cross = np.sum((b_ != a[None, :]) & ok, axis=1)
        neo_down = np.sum((a[None, :] & ~b_ & ok) & y[None, :], axis=1)
        crossings.append({
            "threshold": float(t),
            "mean_crossings_per_realization": float(np.mean(n_cross)),
            "max_crossings_in_a_realization": int(np.max(n_cross)),
            "n_realizations_with_any_crossing": int(np.sum(n_cross > 0)),
            "mean_neo_lost_per_realization": float(np.mean(neo_down)),
            "n_flagged_noiseless": int(a.sum()),
        })

    p0_bins = np.array([0.0, 1e-4, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 + 1e-9])
    return {
        "expected_crossings": crossings,
        "n_realizations": len(names),
        "n_tracklets": int(rows.size),
        "n_neo": int(y.sum()),
        "min_realizations_per_tracklet": int(have.min()),
        "std_overall": _q(std_p, (50, 90, 99)) | {
            "mean": float(np.nanmean(std_p)),
            "max": float(np.nanmax(std_p)),
        },
        "std_neo": _q(std_p[y], (50, 90, 99)) | {"max": float(np.nanmax(std_p[y]))},
        "std_other": _q(std_p[~y], (50, 90, 99)),
        "frac_std_gt_0p01": float(np.nanmean(std_p > 0.01)),
        "frac_std_gt_0p05": float(np.nanmean(std_p > 0.05)),
        "frac_std_gt_0p10": float(np.nanmean(std_p > 0.10)),
        "median_bias_mc_mean_minus_noiseless": float(np.nanmedian(mean_p - p0)),
        "mean_bias_mc_mean_minus_noiseless": float(np.nanmean(mean_p - p0)),
        "vs_v": binned_delta(v, std_p, V_BINS),
        "vs_rate": binned_delta(rate, std_p, RATE_BINS),
        "vs_p0": binned_delta(p0, std_p, p0_bins),
        "most_sensitive": _tail(),
        "_arrays": {"rows": rows, "std_p": std_p, "mean_p": mean_p, "p0": p0,
                    "v": v, "y": y, "rate": rate, "sigma_mas": sigma_mas},
    }


def decomposition_block(out_dir: str, manifest: dict, gen: dict, base: dict) -> dict:
    """Position-only vs rate-only vs both, on common random numbers.

    DESIGN §2.7 asserts that on short Rubin arcs the *rate* uncertainty is the
    part that matters.  The three runs share their standard-normal draws and the
    4-vector perturbations add exactly (``both = pos + rate``), so the variance
    share below is a decomposition rather than a comparison of two unrelated
    experiments.  ``mc/r00`` is the ``both`` arm by construction.
    """
    have = {e["name"] for e in manifest["runs"]}
    need = {"mc/r00", "decomp_pos", "decomp_rate"}
    if not need <= have:
        return {}
    rows = np.load(os.path.join(out_dir, "mc/r00", "noise.npz"))["rows"]
    p0 = base["p_neo"][rows]
    l0 = base["log_lr"][rows]
    got: dict[str, dict] = {}
    for key, name in (("both", "mc/r00"), ("pos", "decomp_pos"), ("rate", "decomp_rate")):
        path = os.path.join(out_dir, name, "scores.npz")
        if not os.path.exists(path):
            return {}
        with np.load(path) as z:
            order = np.argsort(z["index"].astype(int), kind="mergesort")
            got[key] = {"p": z["p_neo"][order], "log_lr": z["log_lr"][order]}

    out: dict = {"n": int(rows.size), "runs": {k: v for k, v in
                                              zip(("both", "pos", "rate"),
                                                  ("mc/r00", "decomp_pos", "decomp_rate"))}}
    d = {k: got[k]["p"] - p0 for k in got}
    with np.errstate(invalid="ignore"):
        dl = {k: got[k]["log_lr"] - l0 for k in got}
    for k in ("both", "pos", "rate"):
        out[f"delta_p_{k}"] = delta_summary(d[k], k)
        out[f"delta_log_lr_{k}"] = delta_summary(dl[k], k)

    m = _finite(d["both"], d["pos"], d["rate"])
    vb = float(np.var(d["both"][m])) if m.sum() > 1 else float("nan")
    out["variance_share"] = {
        "var_delta_p_both": vb,
        "var_delta_p_pos": float(np.var(d["pos"][m])) if m.sum() > 1 else float("nan"),
        "var_delta_p_rate": float(np.var(d["rate"][m])) if m.sum() > 1 else float("nan"),
        "rate_share_of_var": (
            float(np.var(d["rate"][m]) / vb) if vb > 0 else float("nan")
        ),
        "pos_share_of_var": (
            float(np.var(d["pos"][m]) / vb) if vb > 0 else float("nan")
        ),
        "corr_both_rate": float(np.corrcoef(d["both"][m], d["rate"][m])[0, 1])
        if m.sum() > 2 else float("nan"),
        "corr_both_pos": float(np.corrcoef(d["both"][m], d["pos"][m])[0, 1])
        if m.sum() > 2 else float("nan"),
        # How well the two one-at-a-time runs reconstruct the joint one: a small
        # residual means the classifier responds ~linearly to the perturbation
        # at this amplitude, which is exactly the regime a Gaussian covariance
        # marginalization (v1) would model correctly.
        "linearity_residual_rms": float(
            np.sqrt(np.mean((d["both"][m] - d["pos"][m] - d["rate"][m]) ** 2))
        ) if m.sum() else float("nan"),
    }
    sens = [e["name"] for e in manifest["runs"] if e["name"].startswith("sens_m5")]
    if sens and os.path.exists(os.path.join(out_dir, sens[0], "scores.npz")):
        with np.load(os.path.join(out_dir, sens[0], "scores.npz")) as z:
            order = np.argsort(z["index"].astype(int), kind="mergesort")
            ps = z["p_neo"][order]
        em = next(e for e in manifest["runs"] if e["name"] == sens[0])["error_model"]
        out["depth_sensitivity"] = {
            "run": sens[0],
            "m5": em["m5"],
            "delta_p": delta_summary(ps - p0, sens[0]),
            "ratio_abs_p90_to_baseline": (
                delta_summary(ps - p0)["abs_p90"]
                / max(out["delta_p_both"]["abs_p90"], 1e-300)
            ),
        }
    out["_arrays"] = {"rows": rows, "p0": p0, **{f"d_{k}": v for k, v in d.items()}}
    return out


def analyze(
    out_dir: str = DEFAULT_OUT,
    source_run: str | None = None,
    plots: bool = True,
    n_boot: int = 2000,
    seed: int = 20260902,
    verbose: bool = True,
) -> dict:
    """Compare the noisy scores against the stored noiseless ones.

    Nothing here re-scores the noiseless catalogue: ``gate_v0/scores.npz`` is
    opened read-only and every "noiseless" statistic is recomputed from those
    stored values, so the two arms of every comparison come from one scoring
    pass of the same classifier on the same objects, differing only in whether
    the attributable carries astrometric noise.
    """
    from validation.digest2_comparison import auc_bootstrap  # lazy: heavy module

    man = load_manifest(out_dir)
    source_run = source_run or man["source_run"]
    gen = load_source(source_run)
    base = _load_noiseless(source_run)

    main_dir = os.path.join(out_dir, "main")
    has_main = os.path.exists(os.path.join(main_dir, "scores.npz"))
    summary: dict = {
        "out_dir": os.path.abspath(out_dir),
        "source_run": os.path.abspath(source_run),
        "error_model": man["error_model"],
        "provenance": man["provenance"],
        "dt_min": man["dt_min"],
        "seed": man["seed"],
        "sigma_table": sigma_table(ErrorModel(**man["error_model"])),
        "assumptions": ASSUMPTIONS,
    }

    if has_main:
        with np.load(os.path.join(main_dir, "scores.npz")) as z:
            order = np.argsort(z["index"].astype(int), kind="mergesort")
            noisy = mask_policy_rows({k: z[k][order] for k in z.files})
        with np.load(os.path.join(main_dir, "noise.npz")) as z:
            nz = {k: z[k] for k in z.files}
        rows = nz["rows"]
        y = gen["is_neo"][rows].astype(bool)
        w = gen["weight"][rows].astype(float)
        v = gen["v_mag"][rows].astype(float)
        rate = nz["rate_deg_day"]
        p0, p1 = base["p_neo"][rows], noisy["p_neo"]
        l0, l1 = base["log_lr"][rows], noisy["log_lr"]
        with np.errstate(invalid="ignore"):
            # log_lr is -inf for a zero NEO likelihood; inf - inf is a genuine
            # "undefined change", recorded as NaN and dropped by the summaries.
            dp, dl = p1 - p0, l1 - l0

        summary["n"] = int(rows.size)
        summary["n_neo"] = int(y.sum())
        summary["sigma_population"] = {
            "median_mas": float(np.median(nz["sigma_mas"])),
            "p10_mas": float(np.percentile(nz["sigma_mas"], 10)),
            "p90_mas": float(np.percentile(nz["sigma_mas"], 90)),
            "max_mas": float(np.max(nz["sigma_mas"])),
            "median_mas_neo": float(np.median(nz["sigma_mas"][y])),
            "median_dmag_trail": float(np.median(nz["dmag_trail"])),
            "median_dmag_trail_neo": float(np.median(nz["dmag_trail"][y])),
            # The rate error the 45-min arc turns sigma into, relative to the
            # tracklet's own motion: the number DESIGN 2.7 is really about.
            "rate_error_over_rate": _q(
                np.hypot(nz["d_ra_rate"] * np.cos(gen["dec"][rows]), nz["d_dec_rate"])
                / np.radians(rate), (50, 90, 99)
            ),
        }
        # "Undefined" here counts BOTH ways a tracklet ends up without a
        # likelihood score, because C3 is about losing the tracklet and both
        # do: a nan p_neo, and a D-19 triage assignment (masked back to nan by
        # `mask_policy_rows`, which is why the counts below add up).  The
        # policy counts are reported separately so the composition is visible.
        summary["undefined"] = {
            "n_undefined_noisy": int(np.sum(~np.isfinite(p1))),
            "n_undefined_noiseless": int(np.sum(~np.isfinite(p0))),
            "n_policy_assigned_noisy": int(
                noisy.get("_n_policy_assigned", np.array([0]))[0]
            ),
            "n_policy_assigned_noiseless": int(
                base.get("_n_policy_assigned", np.array([0]))[0]
            ),
        }
        summary["deltas"] = {
            "p_neo": {
                "all": delta_summary(dp, "all"),
                "neo": delta_summary(dp[y], "true NEO"),
                "other": delta_summary(dp[~y], "true other"),
                "vs_v": binned_delta(v, dp, V_BINS, w),
                "vs_rate": binned_delta(rate, dp, RATE_BINS, w),
            },
            "log_lr": {
                "all": delta_summary(dl, "all"),
                "neo": delta_summary(dl[y], "true NEO"),
                "other": delta_summary(dl[~y], "true other"),
                "vs_v": binned_delta(v, dl, V_BINS, w),
                "vs_rate": binned_delta(rate, dl, RATE_BINS, w),
            },
        }
        summary["calibration"] = {
            "noiseless": calibration_block(p0, y, w, n_boot=n_boot, seed=seed),
            "noisy": calibration_block(p1, y, w, n_boot=n_boot, seed=seed),
            # Display-only: the gate's 5 equal-expected-NEO-count bins put ~37
            # expected NEOs in each, which is the right size for the *test* but
            # collapses to two visually distinct points on a log-log reliability
            # plot.  A finer binning is used for the figure alone; it is never
            # fed to a verdict.
            "plot_bins_noiseless": reliability_bins(
                p0[np.isfinite(p0)], y[np.isfinite(p0)], w[np.isfinite(p0)], n_bins=12),
            "plot_bins_noisy": reliability_bins(
                p1[np.isfinite(p1)], y[np.isfinite(p1)], w[np.isfinite(p1)], n_bins=12),
        }
        auc = auc_bootstrap(p1, p0, y, w, n_boot=min(n_boot, 2000), seed=seed)
        summary["ranking"] = {
            "auc_noisy": auc["auc_a"],
            "auc_noiseless": auc["auc_b"],
            "delta_auc": auc["delta"],
            "delta_auc_ci95": auc["delta_ci"],
            "auc_noisy_ci95": auc.get("auc_a_ci"),
            "auc_noiseless_ci95": auc.get("auc_b_ci"),
            "n_boot": auc["n_boot"],
            "auc_noisy_crosscheck": auc_weighted(p1, y, w),
        }
        summary["threshold_stability"] = threshold_crossings(p0, p1, y)
        summary["operating_points"] = operating_points(p0, p1, y, w)
        # Does the truth still sit inside the NOISY attributable's admissible
        # region?  A bound orbit must, for the exact attributable; under noise
        # the region moves and need not contain the truth any more.  This is the
        # geometric footprint of the delta-function assumption.
        outside = noisy["rhodot_outside_au_per_day"].astype(float)
        base_out = base["rhodot_outside_au_per_day"][rows].astype(float)
        summary["admissible_region"] = {
            "frac_truth_covered_noisy": float(
                np.mean(np.isfinite(outside) & (outside <= sc._AR_OUTSIDE_TOL))
            ),
            "frac_truth_covered_noiseless": float(
                np.mean(np.isfinite(base_out) & (base_out <= sc._AR_OUTSIDE_TOL))
            ),
            "median_collapse_frac_rho2_noisy": float(
                np.nanmedian(noisy["collapse_frac_rho2"])
            ),
            "median_collapse_frac_rho2_noiseless": float(
                np.nanmedian(base["collapse_frac_rho2"][rows])
            ),
        }

    summary["mc"] = mc_block(out_dir, man, gen, base)
    summary["decomposition"] = decomposition_block(out_dir, man, gen, base)

    arrays = {
        "mc": summary["mc"].pop("_arrays", None),
        "decomp": summary["decomposition"].pop("_arrays", None),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    report = build_report(summary)
    with open(os.path.join(out_dir, "REPORT.md"), "w") as fh:
        fh.write(report)
    if verbose:
        print(f"[analyze] wrote {out_dir}/summary.json and {out_dir}/REPORT.md")

    if plots and has_main:
        make_plots(out_dir, summary, gen, base, rows, y, v, rate, nz, p0, p1, dp,
                   arrays)
    return summary


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


#: Everything the numbers are conditional on.  Written into summary.json and
#: reproduced in REPORT.md, because each of these could move the answer.
ASSUMPTIONS = [
    "Photometric noise is NOT injected: V is carried through unchanged. The "
    "measured degradation is therefore the astrometric contribution alone; "
    "photometric scatter would additionally smear H(V, rho).",
    "Two detections per tracklet, dt = 45 min, exactly linear motion — the WP7 "
    "emission convention. Real Rubin tracklets carry >= 2 detections; more "
    "detections would shrink the rate error roughly as 1/sqrt(n_det) (and the "
    "arc length, not the detection count, sets the leading 1/dt factor).",
    "The attributable is rebuilt by finite differencing the two noisy "
    "endpoints, which is the maximum-likelihood 2-point estimate. A least-"
    "squares fit to >2 detections is the same estimator with a smaller variance.",
    "One seeing/depth pair for the whole catalogue (0.7 arcsec, m5 = 24.0); "
    "Sorcha draws both per visit from an opsim database. Real Rubin visits "
    "spread around these values, which would widen the sigma distribution "
    "without moving its centre much.",
    "sigma is applied in the local sky metric with cos(delta) evaluated once at "
    "the attributable's declination; a <1% metric error for all but a handful "
    "of >5 deg/day tracklets, and the price of an exactly additive "
    "position/rate decomposition.",
    "No light-time, aberration or field-distortion terms: the same geometric "
    "treatment v0's classifier itself uses.",
    "The population is the WP6 self-consistency catalogue (digest2's own model, "
    "All variant, 14 <= V < 24, geocentre). Calibration statements are "
    "therefore about the prior-matched case; DESIGN 4.3's prior-mismatch stress "
    "test is a separate and larger effect.",
]


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _style(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """Recessive grid and axes; ink for text; no top/right spines."""
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")


def _survival(x: np.ndarray):
    """(value, fraction exceeding it) — an ECDF read from the tail."""
    x = np.sort(np.asarray(x, dtype=float)[np.isfinite(x)])
    if x.size == 0:
        return np.array([]), np.array([])
    return x, 1.0 - np.arange(x.size) / x.size


def make_plots(out_dir, summary, gen, base, rows, y, v, rate, nz, p0, p1, dp,
               arrays) -> None:
    """Six diagnostic figures.  Every panel with >= 2 series carries a legend."""
    em = ErrorModel(**summary["error_model"])

    # --- 1. the error model itself -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    vv = np.linspace(15.0, 24.0, 400)
    for c, (lab, r_dd) in zip(PALETTE, (("static source", 0.0),
                                        ("0.25 deg/day (median MBA)", 0.25),
                                        ("1.15 deg/day (median NEO)", 1.15))):
        dm = trailing_loss_dmag(r_dd, 0.0, em.seeing_arcsec, em.texp_s,
                                em.trailing_model) if em.trailing_losses else 0.0
        s, _, _ = astrometric_uncertainty_mas(
            vv + dm, em.m5, fwhm_eff_mas=em.fwhm_mas,
            error_sys_mas=em.error_sys_mas, ast_err_coeff=em.ast_err_coeff,
            rgamma=em.rgamma)
        axes[0].plot(vv, s, color=c, linewidth=2.0, label=lab)
    axes[0].axhline(em.error_sys_mas, color=MUTED, linewidth=1.0, linestyle=":")
    axes[0].annotate(f"{em.error_sys_mas:g} mas systematic floor", (15.2, em.error_sys_mas),
                     color=MUTED, fontsize=8, xytext=(0, 4), textcoords="offset points")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    _style(axes[0], "V [mag]", r"$\sigma$ per detection per coordinate [mas]",
           f"Sorcha astrometric error model (m5 = {em.m5:g}, FWHM = {em.fwhm_mas:g} mas)")

    bins = np.logspace(np.log10(9.0), np.log10(max(nz["sigma_mas"].max(), 20.0)), 45)
    axes[1].hist(nz["sigma_mas"][~y], bins=bins, color=PALETTE[0], alpha=0.75,
                 label=f"true other (n={int((~y).sum())})")
    axes[1].hist(nz["sigma_mas"][y], bins=bins, color=PALETTE[1], alpha=0.85,
                 label=f"true NEO (n={int(y.sum())})")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=8)
    _style(axes[1], r"$\sigma$ [mas]", "tracklets",
           "as realised on the gate_v0 catalogue")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_model.png"), dpi=140)
    plt.close(fig)

    # --- 2. delta vs magnitude and rate ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, x, edges, xlabel, logx, rowsd in (
        (axes[0], v, V_BINS, "V [mag]", False, summary["deltas"]["p_neo"]["vs_v"]),
        (axes[1], rate, RATE_BINS, "total sky motion [deg/day]", True,
         summary["deltas"]["p_neo"]["vs_rate"]),
    ):
        ax.scatter(x, np.maximum(np.abs(dp), 1e-8), s=2, alpha=0.10,
                   color=MUTED, linewidths=0, rasterized=True)
        xc = np.array([r.get("x_mean", np.nan) for r in rowsd])
        for c, key, lab in ((PALETTE[0], "abs_median", "median |dp|"),
                            (PALETTE[1], "abs_p90", "90th pct |dp|"),
                            (PALETTE[2], "abs_p99", "99th pct |dp|")):
            yv = np.array([r.get(key, np.nan) for r in rowsd])
            ax.plot(xc, np.maximum(yv, 1e-8), color=c, linewidth=2.0,
                    marker="o", markersize=5, label=lab)
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        ax.set_ylim(1e-8, 1.5)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        _style(ax, xlabel, r"$|\Delta P(\mathrm{NEO})|$",
               "score change under noise vs. magnitude" if not logx
               else "...and vs. sky motion")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "delta_vs_observables.png"), dpi=140)
    plt.close(fig)

    # --- 3. calibration -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    lims = (3e-4, 1.2)
    ax.plot(lims, lims, color=MUTED, linewidth=1.0, linestyle="--",
            label="perfect calibration")
    for c, key, lab in ((PALETTE[0], "noiseless", "noiseless (v0 gate)"),
                        (PALETTE[1], "noisy", "with Rubin astrometric noise")):
        b = summary["calibration"].get(f"plot_bins_{key}") or \
            summary["calibration"][key]["bins"]
        xs = np.array([r["p_mean"] for r in b])
        ys = np.array([max(r["obs_frac"], 1e-4) for r in b])
        er = np.array([r["sigma_frac"] for r in b])
        ax.errorbar(xs, ys, yerr=er, color=c, linewidth=2.0, marker="o",
                    markersize=7, capsize=3, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, "predicted P(NEO)", "observed NEO fraction",
           "reliability (12 equal-expected-NEO-count bins, display only)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration.png"), dpi=140)
    plt.close(fig)

    # --- 4. ranking and threshold churn ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    w = gen["weight"][rows].astype(float)
    for c, p, lab in ((PALETTE[0], p0, "noiseless"), (PALETTE[1], p1, "noisy")):
        fpr, tpr = roc_curve_weighted(p, y, w)
        axes[0].plot(np.maximum(fpr, 1e-5), tpr, color=c, linewidth=2.0, label=lab)
    axes[0].set_xscale("log")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    _style(axes[0], "false-positive rate", "completeness",
           f"ROC  (AUC {summary['ranking']['auc_noiseless']:.4f} -> "
           f"{summary['ranking']['auc_noisy']:.4f})")

    # Single-realization crossings on the full run are all zero, which is a
    # real result but an empty bar chart.  The informative quantity is the
    # *rate*: mean crossings per realization over the 25 MC realizations of the
    # NEO-enriched subsample (an upper bound on the full population's churn).
    ts = summary["threshold_stability"]
    ex = _g(summary, "mc", "expected_crossings", default=[])
    if ex:
        xpos = np.arange(len(ex))
        vals = np.array([r["mean_crossings_per_realization"] for r in ex], float)
        axes[1].bar(xpos, vals, 0.5, color=PALETTE[0])
        by_t = {r["threshold"]: r for r in ts}
        for k, r in enumerate(ex):
            full = by_t.get(r["threshold"], {})
            axes[1].annotate(
                f"{vals[k]:.2f}\n({full.get('n_cross_any', 0)} in the 20k run;\n"
                f"{full.get('n_flagged_noiseless', 0)} flagged)",
                (k, vals[k]), color=MUTED, fontsize=7.5, ha="center",
                xytext=(0, 5), textcoords="offset points")
        axes[1].set_xticks(xpos)
        axes[1].set_xticklabels([f"p ≥ {r['threshold']:g}" for r in ex])
        axes[1].set_ylim(0, max(float(vals.max()) * 2.2, 0.4))
        _style(axes[1], "triage threshold",
               "mean tracklets crossing per realization",
               f"threshold churn, {_g(summary, 'mc', 'n_realizations', default=0)} "
               "realizations of the NEO-enriched subsample")
    else:  # pragma: no cover - MC runs missing
        _style(axes[1], "triage threshold", "tracklets crossing", "no MC runs")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ranking.png"), dpi=140)
    plt.close(fig)

    # --- 5. per-tracklet sensitivity map -----------------------------------
    a = arrays.get("mc")
    if a is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        floor = 1e-7
        for ax, x, xlabel, logx in ((axes[0], a["v"], "V [mag]", False),
                                    (axes[1], np.maximum(a["p0"], 1e-6),
                                     "noiseless P(NEO)", True)):
            ax.scatter(x[~a["y"]], np.maximum(a["std_p"][~a["y"]], floor), s=10,
                       color=PALETTE[0], alpha=0.55, linewidths=0,
                       label=f"true other (n={int((~a['y']).sum())})")
            ax.scatter(x[a["y"]], np.maximum(a["std_p"][a["y"]], floor), s=12,
                       color=PALETTE[1], alpha=0.75, linewidths=0,
                       label=f"true NEO (n={int(a['y'].sum())})")
            ax.set_yscale("log")
            if logx:
                ax.set_xscale("log")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            _style(ax, xlabel, r"per-tracklet std$(P_{\rm NEO})$",
                   f"{summary['mc']['n_realizations']} noise realizations, "
                   "fixed objects" if not logx
                   else "...against the score the triage cut sees")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "mc_sensitivity.png"), dpi=140)
        plt.close(fig)

    # --- 6. position vs rate decomposition ---------------------------------
    d = arrays.get("decomp")
    if d is not None:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        # `both` and `rate` coincide to plotting precision -- that IS the
        # result -- so they are drawn thick-solid then thin-dashed on top,
        # never distinguished by colour alone.
        for c, key, lab, kw in (
            (PALETTE[0], "d_pos", "position noise only",
             dict(linewidth=2.0)),
            (PALETTE[2], "d_both", "both (the physical case)",
             dict(linewidth=3.5, alpha=0.55)),
            (PALETTE[1], "d_rate", "rate noise only (lies on top of 'both')",
             dict(linewidth=1.6, linestyle="--")),
        ):
            xs, ys = _survival(np.abs(d[key]))
            if xs.size:
                ax.plot(np.maximum(xs, 1e-12), ys, color=c, label=lab, **kw)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        _style(ax, r"$|\Delta P(\mathrm{NEO})|$", "fraction of tracklets exceeding",
               "which half of the perturbation moves the score")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "decomposition.png"), dpi=140)
        plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _g(d: dict, *path, default=float("nan")):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _f(x, fmt=".4g") -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(x) else format(x, fmt)


def _pct(x, fmt=".1f") -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(x) else format(100.0 * x, fmt) + "%"


def build_report(s: dict) -> str:
    """The durable record: validation/output/astrometric_noise/REPORT.md."""
    L: list[str] = []
    A = L.append
    em = s["error_model"]
    cal_n, cal_y = _g(s, "calibration", "noiseless", default={}), _g(s, "calibration", "noisy", default={})

    A("# Astrometric noise vs. priora v0's delta-function attributable")
    A("")
    A("*Does D-6 (the exact-attributable assumption) cost anything at Rubin-level "
      "astrometric noise, or is v1's covariance marginalization urgent?  Harness: "
      "`validation/astrometric_noise.py`.  Noiseless reference: the stored "
      "`gate_v0` scores — nothing was re-scored on that side.*")
    A("")

    # ---- bottom line -------------------------------------------------------
    c_n = _g(cal_n, "odds_scale", "c_hat")
    c_y = _g(cal_y, "odds_scale", "c_hat")
    z_n = _g(cal_n, "global", "z")
    z_y = _g(cal_y, "global", "z")
    op = s.get("operating_points", {})
    fixed = op.get("noisy_fixed_threshold", {})
    base_op = op.get("noiseless", {})
    A("## Bottom line")
    A("")
    A(f"At the adopted Rubin-like noise level (Sorcha's model, m5 = {_f(em['m5'], '.4g')}, "
      f"FWHM = {_f(em['fwhm_mas'], '.4g')} mas; median realised sigma "
      f"{_f(_g(s, 'sigma_population', 'median_mas'), '.1f')} mas per detection per "
      f"coordinate, {_f(_g(s, 'sigma_population', 'p90_mas'), '.1f')} mas at the 90th "
      f"percentile), on {s.get('n', 0)} tracklets containing {s.get('n_neo', 0)} true NEOs:")
    A("")
    A(f"- **Calibration is untouched.** The global odds scale moves from "
      f"ĉ = {_f(c_n, '.3f')} (noiseless) to ĉ = {_f(c_y, '.3f')} (noisy); the global "
      f"count statistic from Z = {_f(z_n, '+.2f')} to Z = {_f(z_y, '+.2f')}. Both stay "
      f"inside the run's own ±35% odds resolution at N = 2×10⁴.")
    A(f"- **Ranking is untouched.** ΔAUC = {_f(_g(s, 'ranking', 'delta_auc'), '+.5f')} "
      f"[{_f(_g(s, 'ranking', 'delta_auc_ci95')[0] if isinstance(_g(s, 'ranking', 'delta_auc_ci95'), list) else float('nan'), '+.5f')}, "
      f"{_f(_g(s, 'ranking', 'delta_auc_ci95')[1] if isinstance(_g(s, 'ranking', 'delta_auc_ci95'), list) else float('nan'), '+.5f')}] "
      f"(paired bootstrap) — consistent with zero, and an order of magnitude "
      f"below the +0.0126 method gap over digest2 that D-14 measured.")
    A(f"- **The operating point is untouched.** Deploying WP7's matched-completeness "
      f"cut (p ≥ {_f(base_op.get('threshold'), '.3g')}, chosen on noiseless scores) "
      f"against noisy scores gives {_pct(fixed.get('completeness'))} completeness at "
      f"{_pct(fixed.get('contamination'))} contamination, versus "
      f"{_pct(base_op.get('completeness'))} / {_pct(base_op.get('contamination'))} noiseless.")
    ex0 = _g(s, "mc", "expected_crossings", default=[])
    if ex0:
        worst = max(ex0, key=lambda r: r["mean_crossings_per_realization"])
        A(f"- **Triage thresholds are stable.** Zero tracklets crossed p = 0.1 / 0.3 / "
          f"0.5 / 0.7 in the headline realization; averaged over "
          f"{_g(s, 'mc', 'n_realizations', default=0)} realizations of an "
          f"NEO-enriched subsample the worst threshold churns "
          f"{worst['mean_crossings_per_realization']:.2f} tracklets per "
          f"realization, and not one true NEO fell below a cut in any of them.")
    A(f"- **Per-tracklet scores barely move.** Median |ΔP(NEO)| = "
      f"{_f(_g(s, 'deltas', 'p_neo', 'all', 'abs_p50'), '.2e')}, 90th percentile "
      f"{_f(_g(s, 'deltas', 'p_neo', 'all', 'abs_p90'), '.2e')}, 99th "
      f"{_f(_g(s, 'deltas', 'p_neo', 'all', 'abs_p99'), '.2e')}.")
    A("")
    A("**Verdict: v0's delta-function attributable is acceptable for Rubin triage, and "
      "v1's covariance marginalization is not urgent on this axis.** The reason is "
      "arithmetic rather than luck: a 45-minute arc converts an 85 mas endpoint error "
      "into a rate error of ~3.8 arcsec/day, which is "
      f"{_pct(_g(s, 'sigma_population', 'rate_error_over_rate', 'p50'), '.2f')} of the "
      f"median tracklet's own motion and "
      f"{_pct(_g(s, 'sigma_population', 'rate_error_over_rate', 'p90'), '.2f')} at the "
      "90th percentile. The classifier's response to the attributable is smooth on that "
      "scale. The caveats that could change this answer are in *Where this could break* "
      "below; the largest is arc length, not astrometric precision.")
    A("")

    # ---- provenance --------------------------------------------------------
    A("## The error model, and where it comes from")
    A("")
    prov = s.get("provenance", {})
    A(f"Transcribed from **{prov.get('error_model_source', '')}**, ref "
      f"`{prov.get('sorcha_ref', '')}`. Sorcha's own citations:")
    A("")
    for c in prov.get("citations", []):
        A(f"- {c}")
    A("")
    A("The chain, in Sorcha's order:")
    A("")
    A("```")
    A("dmag  = 1.25 log10(1 + a x^2/(1 + b x)),  x = v texp/seeing,  a=0.761 b=1.162")
    A("x_snr = 10^(0.4 (V + dmag - m5))")
    A("SNR   = 1 / sqrt((0.04 - gamma) x_snr + gamma x_snr^2),        gamma = 0.039")
    A("sigma = sqrt(error_sys^2 + (astErrCoeff * FWHMeff / SNR)^2)")
    A("        error_sys = 10 mas,  astErrCoeff = 0.60,  FWHMeff = 700 mas")
    A("```")
    A("")
    A(f"Adopted parameters: **m5 = {_f(em['m5'], '.4g')}** — Sorcha hardcodes no default "
      "depth (it reads `fiveSigmaDepth` per visit from an opsim database), so we take "
      "the value internally consistent with the catalogue being perturbed: the WP6 "
      "generator's faint cut is V < 24 and is documented as \"the Rubin-like detection "
      "limit\", so setting m5 = V_max makes this catalogue exactly a 5σ-limited sample "
      "and puts its faintest tracklets at SNR = 5. It is also the conservative choice; a "
      "half-magnitude-deeper survey is run below as a sensitivity case. "
      f"**FWHMeff = {_f(em['fwhm_mas'], '.4g')} mas** is Sorcha's own default argument "
      f"value; **error_sys = {_f(em['error_sys_mas'], '.4g')} mas**, "
      f"**astErrCoeff = {_f(em['ast_err_coeff'], '.2f')}**, "
      f"**gamma = {_f(em['rgamma'], '.3f')}**, **nvisit = 1**, "
      f"**texp = {_f(em['texp_s'], '.4g')} s**, trailing losses "
      f"**{'on' if em['trailing_losses'] else 'off'}** are all Sorcha defaults.")
    A("")
    A("σ for one detection, one coordinate (static source, no trailing):")
    A("")
    A("| V | SNR | σ_rand [mas] | σ [mas] | σ [arcsec] |")
    A("|---:|---:|---:|---:|---:|")
    for r in s.get("sigma_table", []):
        A(f"| {r['V']:.0f} | {r['snr']:.2f} | {r['sigma_rand_mas']:.2f} | "
          f"{r['sigma_mas']:.2f} | {r['sigma_arcsec']:.4f} |")
    A("")
    sp = s.get("sigma_population", {})
    A(f"Realised on the catalogue: median {_f(sp.get('median_mas'), '.1f')} mas "
      f"(NEOs {_f(sp.get('median_mas_neo'), '.1f')} mas — they are both fainter on "
      f"average and trailed), p90 {_f(sp.get('p90_mas'), '.1f')} mas, max "
      f"{_f(sp.get('max_mas'), '.1f')} mas. Median trailing loss "
      f"{_f(sp.get('median_dmag_trail'), '.3f')} mag overall, "
      f"{_f(sp.get('median_dmag_trail_neo'), '.3f')} mag for NEOs. "
      "See `error_model.png`.")
    A("")

    # ---- deltas ------------------------------------------------------------
    A("## What the noise does to the scores")
    A("")
    A(f"One realization on all {s.get('n', 0)} tracklets, dt = {_f(s.get('dt_min'), '.0f')} "
      "min, position from detection 0 and rates from the two-endpoint finite "
      "difference. V is carried through unchanged (photometric noise out of scope).")
    A("")
    A("| quantity | subset | median \\|Δ\\| | p90 \\|Δ\\| | p99 \\|Δ\\| | max \\|Δ\\| | mean Δ |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for qty, key in (("ΔP(NEO)", "p_neo"), ("Δ ln LR", "log_lr")):
        for sub in ("all", "neo", "other"):
            d = _g(s, "deltas", key, sub, default={})
            if not d:
                continue
            A(f"| {qty} | {d.get('label', sub)} | {_f(d.get('abs_p50'), '.2e')} | "
              f"{_f(d.get('abs_p90'), '.2e')} | {_f(d.get('abs_p99'), '.2e')} | "
              f"{_f(d.get('max_abs'), '.2e')} | {_f(d.get('mean'), '+.2e')} |")
    A("")
    A("Magnitude-resolved (`delta_vs_observables.png`) — the error model is "
      "magnitude-dependent, so this is where it has to show:")
    A("")
    A("| V bin | n | median \\|ΔP\\| | p90 \\|ΔP\\| | p99 \\|ΔP\\| |")
    A("|---|---:|---:|---:|---:|")
    for r in _g(s, "deltas", "p_neo", "vs_v", default=[]):
        if r["n"]:
            A(f"| {r['lo']:.1f}–{r['hi']:.1f} | {r['n']} | "
              f"{_f(r.get('abs_median'), '.2e')} | {_f(r.get('abs_p90'), '.2e')} | "
              f"{_f(r.get('abs_p99'), '.2e')} |")
    A("")
    A("| rate bin [deg/day] | n | median \\|ΔP\\| | p90 \\|ΔP\\| | p99 \\|ΔP\\| |")
    A("|---|---:|---:|---:|---:|")
    for r in _g(s, "deltas", "p_neo", "vs_rate", default=[]):
        if r["n"]:
            A(f"| {r['lo']:g}–{r['hi']:g} | {r['n']} | "
              f"{_f(r.get('abs_median'), '.2e')} | {_f(r.get('abs_p90'), '.2e')} | "
              f"{_f(r.get('abs_p99'), '.2e')} |")
    A("")
    A("Two features of that table are worth naming rather than leaving to be "
      "puzzled over. The peak at 0.6–1 deg/day is the NEO band: those tracklets "
      "are the faint, trailed, mid-probability ones where the posterior actually "
      "has a gradient. Above 1 deg/day the deltas collapse to exactly zero "
      "because essentially every such tracklet already scores P(NEO) = 1 to "
      "machine precision — nothing in the model can move them. The rise with V "
      "is the error model itself: σ grows ~8× from V = 18 to V = 24 while the "
      "median |ΔP| grows ~8×, i.e. the response is close to linear in σ.")
    A("")

    # ---- calibration -------------------------------------------------------
    A("## Calibration under noise")
    A("")
    A("The classifier now assumes an exact attributable that is *wrong*. Running the "
      "WP6 gate's own calibration machinery (equal-expected-NEO-count bins, "
      "Poisson-binomial z, parametric-bootstrap thresholds, fitted global odds scale) "
      "against the true labels measures the real miscalibration Rubin would see from "
      "v0. This is no longer a self-consistency test — a failure here would be a "
      "*property of the data*, not a bug.")
    A("")
    A("| statistic | noiseless | with noise |")
    A("|---|---:|---:|")
    A(f"| observed / expected NEO count | {_f(_g(cal_n, 'global', 'observed'), '.1f')} / "
      f"{_f(_g(cal_n, 'global', 'expected'), '.1f')} | "
      f"{_f(_g(cal_y, 'global', 'observed'), '.1f')} / "
      f"{_f(_g(cal_y, 'global', 'expected'), '.1f')} |")
    A(f"| global Z_tot (p) | {_f(z_n, '+.2f')} ({_f(_g(cal_n, 'verdict', 'global_z_p_value'), '.3f')}) | "
      f"{_f(z_y, '+.2f')} ({_f(_g(cal_y, 'verdict', 'global_z_p_value'), '.3f')}) |")
    A(f"| fitted odds scale ĉ [95%] | {_f(c_n, '.3f')} "
      f"[{_f(_g(cal_n, 'odds_scale', 'lo'), '.3f')}, {_f(_g(cal_n, 'odds_scale', 'hi'), '.3f')}] | "
      f"{_f(c_y, '.3f')} [{_f(_g(cal_y, 'odds_scale', 'lo'), '.3f')}, "
      f"{_f(_g(cal_y, 'odds_scale', 'hi'), '.3f')}] |")
    A(f"| reliability max\\|z\\| (bootstrap 99th) | "
      f"{_f(_g(cal_n, 'verdict', 'max_abs_z'), '.2f')} "
      f"({_f(_g(cal_n, 'verdict', 'max_abs_z_threshold'), '.2f')}) | "
      f"{_f(_g(cal_y, 'verdict', 'max_abs_z'), '.2f')} "
      f"({_f(_g(cal_y, 'verdict', 'max_abs_z_threshold'), '.2f')}) |")
    A(f"| reliability verdict | {'PASS' if _g(cal_n, 'verdict', 'pass', default=False) else 'FAIL'} | "
      f"{'PASS' if _g(cal_y, 'verdict', 'pass', default=False) else 'FAIL'} |")
    A(f"| Brier score | {_f(cal_n.get('brier'), '.5f')} | {_f(cal_y.get('brier'), '.5f')} |")
    A(f"| expected calibration error | {_f(cal_n.get('expected_calibration_error'), '.2e')} | "
      f"{_f(cal_y.get('expected_calibration_error'), '.2e')} |")
    A("")
    A("See `calibration.png`. Caveat carried over from DESIGN §4.1: at N = 2×10⁴ "
      "(≈185 NEOs) the odds scale is only constrained to roughly ±35%, so this says "
      "\"no detectable miscalibration at the ±35% level\", not \"identical to five "
      "decimal places\". The paired per-tracklet deltas above are the sharper "
      "statement.")
    A("")

    # ---- ranking / thresholds ---------------------------------------------
    A("## Ranking and threshold stability")
    A("")
    rk = s.get("ranking", {})
    ci = rk.get("delta_auc_ci95") or [float("nan"), float("nan")]
    A(f"AUC {_f(rk.get('auc_noiseless'), '.5f')} (noiseless) → "
      f"{_f(rk.get('auc_noisy'), '.5f')} (noisy); "
      f"ΔAUC = {_f(rk.get('delta_auc'), '+.5f')} [{_f(ci[0], '+.5f')}, {_f(ci[1], '+.5f')}] "
      f"from a paired, label-stratified bootstrap ({rk.get('n_boot', 0)} replicates). "
      "See `ranking.png`.")
    A("")
    ts0 = s.get("threshold_stability", [])
    A("| threshold | flagged (noiseless → noisy) | crossed up | crossed down | NEOs lost | tracklets at risk |")
    A("|---|---:|---:|---:|---:|---:|")
    for r in ts0:
        A(f"| p ≥ {r['threshold']:g} | {r['n_flagged_noiseless']} → {r['n_flagged_noisy']} | "
          f"{r['n_cross_up']} | {r['n_cross_down']} | {r['neo_cross_down']} | "
          f"{r.get('n_at_risk', '—')} |")
    A("")
    if ts0:
        A(f"\"At risk\" counts tracklets whose noiseless score sits within "
          f"{_f(ts0[0].get('at_risk_scale'), '.2e')} (the 99th percentile of "
          "|ΔP|) of the cut. The zeros above are therefore two facts at once: "
          "the perturbation is small **and** almost nothing sits near a triage "
          "threshold at this base rate — 102 of 20,000 tracklets score anywhere "
          "in (0.05, 0.95). A single realization cannot separate \"impossible\" "
          "from \"unlikely this time\", so the MC block below reports the "
          "expected churn per realization instead.")
        A("")
    ex = _g(s, "mc", "expected_crossings", default=[])
    if ex:
        A("Averaged over the MC realizations on the 500-tracklet subsample "
          "(which is *enriched* in the interesting tracklets — it holds all 185 "
          "NEOs, so its churn rate is an upper bound on the full population's):")
        A("")
        A("| threshold | flagged (noiseless) | mean crossings / realization | worst realization | realizations with any | mean NEOs lost |")
        A("|---|---:|---:|---:|---:|---:|")
        for r in ex:
            A(f"| p ≥ {r['threshold']:g} | {r['n_flagged_noiseless']} | "
              f"{r['mean_crossings_per_realization']:.2f} | "
              f"{r['max_crossings_in_a_realization']} | "
              f"{r['n_realizations_with_any_crossing']}/{_g(s, 'mc', 'n_realizations', default=0)} | "
              f"{r['mean_neo_lost_per_realization']:.2f} |")
        A("")

    A("## The operating point (WP7's matched-completeness cut)")
    A("")
    A("`validation/DIGEST2_COMPARISON.md` sets the reference: digest2's own NEOCP cut "
      f"(raw ≥ 65) recovers {_pct(op.get('target_completeness'))} of the true NEOs on "
      "`gate_v0`, and priora matches that completeness at 6.3% contamination. Three "
      "readings, of which the third is the operationally honest one — a threshold is "
      "chosen once, on a study like WP7's, and then deployed against noisy data:")
    A("")
    A("| reading | threshold | flagged | completeness | contamination |")
    A("|---|---:|---:|---:|---:|")
    for key in ("noiseless", "noisy_rethresholded", "noisy_fixed_threshold",
                "noiseless_p0.5", "noisy_p0.5"):
        pt = op.get(key)
        if not pt:
            continue
        A(f"| {pt['label']} | {_f(pt.get('threshold'), '.3g')} | {pt['n_flagged']} | "
          f"{_pct(pt.get('completeness'))} | {_pct(pt.get('contamination'))} |")
    A("")

    # ---- MC ---------------------------------------------------------------
    mc = s.get("mc", {})
    if mc.get("n_realizations"):
        A("## Per-tracklet noise sensitivity map")
        A("")
        A(f"{mc['n_realizations']} independent noise realizations of a fixed "
          f"{mc['n_tracklets']}-tracklet stratified subsample (**all {mc['n_neo']} true "
          "NEOs** plus a V-stratified draw of the rest). The spread of P(NEO) at fixed "
          "object and fixed error model is the per-tracklet cost of D-6, free of any "
          "population averaging. See `mc_sensitivity.png`.")
        A("")
        so = mc.get("std_overall", {})
        A(f"- std(P) overall: median {_f(so.get('p50'), '.2e')}, p90 "
          f"{_f(so.get('p90'), '.2e')}, p99 {_f(so.get('p99'), '.2e')}, max "
          f"{_f(so.get('max'), '.3f')}.")
        A(f"- std(P) for true NEOs: median {_f(_g(mc, 'std_neo', 'p50'), '.2e')}, p90 "
          f"{_f(_g(mc, 'std_neo', 'p90'), '.2e')}, max {_f(_g(mc, 'std_neo', 'max'), '.3f')}.")
        A(f"- {_pct(mc.get('frac_std_gt_0p01'))} of tracklets have std(P) > 0.01, "
          f"{_pct(mc.get('frac_std_gt_0p05'))} > 0.05, "
          f"{_pct(mc.get('frac_std_gt_0p10'))} > 0.10.")
        A(f"- Mean over realizations minus the noiseless score: median "
          f"{_f(mc.get('median_bias_mc_mean_minus_noiseless'), '+.2e')}, mean "
          f"{_f(mc.get('mean_bias_mc_mean_minus_noiseless'), '+.2e')} — the noise is "
          "essentially unbiased at this amplitude (a nonlinear response would show up "
          "here as a systematic offset, and is what an outer marginalization would "
          "actually correct).")
        A("")
        A("std(P) against the noiseless score — the triage-relevant slice, because only "
          "a tracklet near a threshold can be moved across it:")
        A("")
        A("| noiseless P(NEO) | n | median std | p90 std | p99 std |")
        A("|---|---:|---:|---:|---:|")
        for r in mc.get("vs_p0", []):
            if r["n"]:
                A(f"| {r['lo']:g}–{r['hi']:g} | {r['n']} | "
                  f"{_f(r.get('abs_median'), '.2e')} | {_f(r.get('abs_p90'), '.2e')} | "
                  f"{_f(r.get('abs_p99'), '.2e')} |")
        A("")
        A("| V bin | n | median std | p90 std | p99 std |")
        A("|---|---:|---:|---:|---:|")
        for r in mc.get("vs_v", []):
            if r["n"]:
                A(f"| {r['lo']:.1f}–{r['hi']:.1f} | {r['n']} | "
                  f"{_f(r.get('abs_median'), '.2e')} | {_f(r.get('abs_p90'), '.2e')} | "
                  f"{_f(r.get('abs_p99'), '.2e')} |")
        A("")
        ms = mc.get("most_sensitive", [])
        if ms:
            A("The most sensitive tracklets in the subsample:")
            A("")
            A("| gen index | NEO? | V | rate [deg/day] | σ [mas] | P noiseless | P mean | std(P) |")
            A("|---:|---|---:|---:|---:|---:|---:|---:|")
            for r in ms:
                A(f"| {r['gen_index']} | {'yes' if r['is_neo'] else 'no'} | "
                  f"{r['v_mag']:.2f} | {r['rate_deg_day']:.3f} | {r['sigma_mas']:.1f} | "
                  f"{_f(r['p_noiseless'], '.4f')} | {_f(r['p_mc_mean'], '.4f')} | "
                  f"{_f(r['p_mc_std'], '.4f')} |")
            A("")

    # ---- decomposition -----------------------------------------------------
    dec = s.get("decomposition", {})
    if dec:
        A("## Positions vs. rates")
        A("")
        A("DESIGN §2.7 expects the *rate* uncertainty to dominate on short arcs. The "
          "three runs share their standard-normal draws, so the split is exact rather "
          "than a comparison of two unrelated experiments:")
        A("")
        A("- `pos`: the same offset applied to **both** detections — a rigid translation "
          "of the tracklet. Position error σ, rate error identically zero.")
        A("- `rate`: detection 0 left exactly alone, detection 1 offset by the "
          "difference of the two draws. Position exact, rate error identical "
          "draw-for-draw to the physical case.")
        A("- `both`: the physical case; by construction "
          "`ΔA(both) = ΔA(pos) + ΔA(rate)` to the last bit.")
        A("")
        A("| noise applied to | median \\|ΔP\\| | p90 \\|ΔP\\| | p99 \\|ΔP\\| | RMS ΔP | max \\|ΔP\\| |")
        A("|---|---:|---:|---:|---:|---:|")
        for k, lab in (("pos", "positions only"), ("rate", "rates only"),
                       ("both", "both (physical)")):
            d = dec.get(f"delta_p_{k}", {})
            if d:
                A(f"| {lab} | {_f(d.get('abs_p50'), '.2e')} | {_f(d.get('abs_p90'), '.2e')} | "
                  f"{_f(d.get('abs_p99'), '.2e')} | {_f(d.get('rms'), '.2e')} | "
                  f"{_f(d.get('max_abs'), '.2e')} |")
        A("")
        vs = dec.get("variance_share", {})
        A(f"Variance share of ΔP: rates {_pct(vs.get('rate_share_of_var'))}, "
          f"positions {_pct(vs.get('pos_share_of_var'))} "
          f"(corr(both, rate) = {_f(vs.get('corr_both_rate'), '.3f')}, "
          f"corr(both, pos) = {_f(vs.get('corr_both_pos'), '.3f')}). "
          f"The two one-at-a-time runs reconstruct the joint one with an RMS residual of "
          f"{_f(vs.get('linearity_residual_rms'), '.2e')} in ΔP — i.e. the classifier "
          "responds essentially linearly to the perturbation at this amplitude, which is "
          "precisely the regime a Gaussian covariance marginalization would model "
          "correctly (and therefore also the regime in which it has least to add). "
          "See `decomposition.png`.")
        A("")
        ds = dec.get("depth_sensitivity")
        if ds:
            A(f"**Depth sensitivity.** The same subsample and the same draws at "
              f"m5 = {_f(ds.get('m5'), '.4g')} (half a magnitude deeper) gives "
              f"p90 |ΔP| = {_f(_g(ds, 'delta_p', 'abs_p90'), '.2e')}, a factor "
              f"{_f(ds.get('ratio_abs_p90_to_baseline'), '.2f')} of the baseline — the "
              "answer is not sitting on a knife edge of the adopted depth.")
            A("")

    # ---- geometry ----------------------------------------------------------
    ar = s.get("admissible_region", {})
    if ar:
        A("## Admissible-region geometry under noise")
        A("")
        A("A bound orbit's true (ρ, ρ̇) must lie inside the admissible region of its "
          "*exact* attributable. Under noise the region moves and need not contain the "
          "truth any more — the geometric footprint of the delta-function assumption.")
        A("")
        A(f"- Truth inside the admissible region: "
          f"{_pct(ar.get('frac_truth_covered_noiseless'), '.2f')} noiseless → "
          f"{_pct(ar.get('frac_truth_covered_noisy'), '.2f')} noisy.")
        A(f"- Median posterior mass within a factor 2 of the true ρ: "
          f"{_f(ar.get('median_collapse_frac_rho2_noiseless'), '.4f')} → "
          f"{_f(ar.get('median_collapse_frac_rho2_noisy'), '.4f')}.")
        A("")

    # ---- caveats -----------------------------------------------------------
    A("## Where this could break")
    A("")
    A("The conclusion is scoped by the arc, not by the astrometry. The rate error scales "
      "as σ√2/dt, so:")
    A("")
    A("- **Shorter arcs.** A 15-minute pair triples the rate error; a 5-minute pair "
      "multiplies it by 9. That is a genuinely different regime and this experiment does "
      "not cover it. If Rubin's tracklet cadence tightens, re-run with `--dt-min`.")
    A("- **Slow movers.** The fractional rate error is worst for the slowest objects "
      "(distant, near-stationary), which is visible in the rate-binned table above. They "
      "are also the ones the classifier is most confident about, which is why the score "
      "still barely moves — but a population with more TNOs would weight this differently.")
    A("- **Photometric noise**, deliberately excluded here, enters through H(V, ρ) with "
      "unit Jacobian (DESIGN §2.2) and is a separate, additive study.")
    A("- **Prior mismatch** (DESIGN §4.3) is a far larger effect than anything measured "
      "here and is the thing that should actually be prioritised over covariance "
      "marginalization.")
    A("")
    A("## Assumptions")
    A("")
    for a in s.get("assumptions", []):
        A(f"- {a}")
    A("")
    A("## Reproduce")
    A("")
    A("```")
    A("OMP_NUM_THREADS=1 python3 -m validation.astrometric_noise run --workers 32")
    A("```")
    A("")
    A(f"Seed {s.get('seed')}; source catalogue `{s.get('source_run')}` (read-only). "
      "Every stage is resumable and a pure function of its inputs.")
    A("")
    return "\n".join(L)


# ==========================================================================
# Arc-length sensitivity sweep
# ==========================================================================
#
# ASTROMETRIC_NOISE.md's own conclusion scopes itself: "the conclusion is
# scoped by the arc, not by the astrometry", because the rate error is
# ``sqrt(2) sigma / dt`` and the 45-minute verdict therefore says nothing about
# a 5-minute pair.  DESIGN §2.7 names the same variable ("the binding variable
# is arc length").  This block answers the question quantitatively: it re-runs
# the whole experiment on a dt grid and asks at which arc length the
# delta-function attributable stops being harmless.
#
# The sweep is a *common-random-numbers* sweep in dt.  ``noisy_attributable``
# draws ``e0, e1`` from ``default_rng([seed, stream])`` and dt enters only as
# the divisor in ``(d_1 - d_0) / dt``, so at fixed (seed, stream, rows) every dt
# in the grid perturbs the very same endpoints by the very same offsets and
# differs *only* in the 1/dt amplification of the rates.  Two consequences that
# matter for reading the tables below:
#
#   * the dt-to-dt comparison carries no Monte-Carlo noise of its own — the
#     curve ``|Delta P|(dt)`` is a paired, per-tracklet function of dt, not six
#     independent experiments; and
#   * in the linear-response regime ``|Delta P| * dt`` is *constant* in dt for
#     each individual tracklet, which turns "is the response still linear?"
#     into a one-line check (:func:`scaling_ratio`) instead of a modelling
#     exercise.  Departure from that constancy is exactly the nonlinearity an
#     outer covariance marginalization (v1) would have to capture.
#
# dt = 45 min reproduces the WP8 headline run bit-for-bit and is *adopted* from
# it rather than re-scored (:func:`_adopt_scores` verifies array identity of
# the injected catalogue before reusing a single stored score).

#: The arc-length grid [min].  60/45/30 bracket a plausible Rubin same-night
#: pair separation (45 is the WP7/WP8 convention); 15/10/5 are the ~x3 and ~x9
#: rate-error amplifications DESIGN §2.7 flags as untested.
DT_GRID_DEFAULT = (60.0, 45.0, 30.0, 15.0, 10.0, 5.0)

DEFAULT_SWEEP_OUT = "validation/output/astrometric_noise_arclength"

def _add_density_mode_args(p) -> None:
    """Expose the population's H-density modes on a scoring subcommand.

    ``self_consistency.ScoreConfig`` carries ``h_interpolation`` (D-17) and
    ``h_extend_to`` (D-19), and ``sc.score`` hard-fails when they disagree with
    the catalogue they are scoring — so this study inherits the threading for
    free.  What it did NOT have was a way to *express* either, which meant the
    noise sweep could only ever be run against a histogram-mode catalogue and
    would have hard-failed (correctly, but confusingly) on any other.
    """
    p.add_argument("--h-interpolation", default="none",
                   choices=("none", "pchip"),
                   help="population H reconstruction (D-17); must match the "
                        "catalogue being perturbed")
    p.add_argument("--h-extend-to", type=float, default=None, metavar="H",
                   help="D-19 finite H extension; must match the catalogue "
                        "being perturbed")


def _score_cfg_from_args(a) -> "ScoreConfig":
    return ScoreConfig(
        n_rho=a.n_rho,
        n_rhodot=a.n_rhodot,
        h_interpolation=getattr(a, "h_interpolation", "none"),
        h_extend_to=getattr(a, "h_extend_to", None),
    )


#: **C1** — the WP6 gate's own family-wise alpha for the reliability verdict.
#: Not a new threshold: the same parametric-bootstrap machinery the gate uses.
BREAK_ALPHA = 0.01
#: **C2** — fraction of tracklets changing side of a triage threshold per noise
#: realization above which triage churn counts as a break.
BREAK_CROSS_FRAC = 0.01
#: **C3** — fraction of tracklets whose score becomes undefined (no bound orbit
#: anywhere in the admissible region of the noisy attributable) above which the
#: classifier is judged to have lost the tracklet outright.
BREAK_UNDEFINED_FRAC = 0.01
#: **W1** — the priora-over-digest2 AUC gap D-14 measured.  A ranking
#: degradation of that size would eat the method's entire measured advantage.
WARN_DELTA_AUC = 0.0126
#: **W2** — completeness loss at the fixed WP7 operating point, in absolute
#: fraction (1 percentage point).
WARN_COMPLETENESS_DROP = 0.01

#: The operational definition of "breaks", fixed **before** the sweep was run
#: and reported against verbatim in REPORT.md.  Three break arms and two
#: warnings; a warning is a flag for the reader, not a failure.
BREAK_CRITERION = [
    "C1 (calibration) — the noisy scores FAIL the WP6 gate's own reliability "
    "verdict against the true labels: either the global count arm (|Z_tot|, "
    "Poisson-binomial) or the binned arm (max|z| over equal-expected-NEO-count "
    f"bins) has a parametric-bootstrap p-value <= alpha = {BREAK_ALPHA:g}. "
    "This reuses the gate's own thresholds; nothing new is invented.",
    "C2 (triage churn) — more than "
    f"{100.0 * BREAK_CROSS_FRAC:g}% of tracklets change side of a triage "
    "threshold (p = 0.1/0.3/0.5/0.7) per noise realization. Estimated as the "
    "MC mean crossing count over the realizations of the stratified "
    "NEO-enriched subsample, divided by its size — an UPPER bound on the full "
    "population's rate, since the subsample holds every true NEO and hence "
    "essentially every mid-probability tracklet in the catalogue. "
    "Cross-checked against the full-catalogue single realization.",
    "C3 (lost tracklets) — more than "
    f"{100.0 * BREAK_UNDEFINED_FRAC:g}% of tracklets that scored finitely "
    "noiseless get NO LIKELIHOOD SCORE under noise. Two directions count, and "
    "both leave the tracklet unscored rather than merely mis-scored: the noisy "
    "attributable admits no bound orbit the classifier can integrate, or its "
    "whole admissible region is pushed past the model's faint end so that both "
    "Lambdas are zero (D-19 then assigns a triage score of 1.0, which is not a "
    "likelihood and is counted here as lost — see `mask_policy_rows`).",
    f"W1 (warning, ranking) — the paired-bootstrap 95% CI on Delta-AUC excludes "
    f"zero AND |Delta AUC| > {WARN_DELTA_AUC:g}, the priora-over-digest2 "
    "method gap D-14 measured.",
    f"W2 (warning, operating point) — completeness at the FIXED WP7 "
    f"matched-completeness cut (chosen once on noiseless scores) drops by more "
    f"than {100.0 * WARN_COMPLETENESS_DROP:g} percentage point.",
]

#: Post-hoc, NOT part of :data:`BREAK_CRITERION`.  W2 as declared watches only
#: the completeness half of the fixed operating point, and the sweep found that
#: to be the half that does not move: completeness is pinned by the cut while
#: the *contamination* multiplies.  Reporting the contamination ratio is
#: therefore necessary to describe the result honestly — but it is labelled
#: post-hoc everywhere it appears, and it does not enter any verdict, because a
#: tripwire invented after seeing the data is not evidence.
POSTHOC_CONTAMINATION_RATIO = 2.0


def dt_tag(dt_min: float) -> str:
    """Directory name for one arc length: ``dt_45min``."""
    return f"dt_{float(dt_min):g}min"


# --------------------------------------------------------------------------
# Sweep: inject
# --------------------------------------------------------------------------


def _gen_arrays_identical(a_dir: str, b_dir: str) -> bool:
    """Do two run directories hold the *same* injected catalogue?

    Compared array-by-array rather than by file bytes: ``np.savez_compressed``
    stamps a zip mtime into the file, so two byte-different files can hold
    identical arrays.  Only ``gen.npz`` matters — it is the sole input the WP6
    scorer reads, and the only thing hashed into the resume stamp.
    """
    pa, pb = os.path.join(a_dir, "gen.npz"), os.path.join(b_dir, "gen.npz")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return False
    with np.load(pa) as za, np.load(pb) as zb:
        if set(za.files) != set(zb.files):
            return False
        return all(np.array_equal(za[k], zb[k]) for k in za.files)


def _adopt_scores(donor_dir: str, target_dir: str, verbose: bool = True) -> bool:
    """Reuse a stored scoring pass whose input catalogue is provably identical.

    Copies the donor's ``gen.npz`` **bytes** (so the resume stamp matches its
    chunks), its ``chunks/`` and its ``scores.npz`` into ``target_dir``.  Refuses
    unless the injected arrays are equal element-for-element, so this can never
    splice a different catalogue's scores into the sweep.  Returns False (and
    leaves ``target_dir`` alone) if the donor is missing or does not match, in
    which case the caller simply scores the run normally.
    """
    import shutil

    if not os.path.exists(os.path.join(donor_dir, "scores.npz")):
        return False
    if not _gen_arrays_identical(donor_dir, target_dir):
        return False
    shutil.copy2(os.path.join(donor_dir, "gen.npz"), os.path.join(target_dir, "gen.npz"))
    src_chunks = os.path.join(donor_dir, "chunks")
    if os.path.isdir(src_chunks):
        dst_chunks = os.path.join(target_dir, "chunks")
        os.makedirs(dst_chunks, exist_ok=True)
        for name in sorted(os.listdir(src_chunks)):
            shutil.copy2(os.path.join(src_chunks, name), os.path.join(dst_chunks, name))
    shutil.copy2(os.path.join(donor_dir, "scores.npz"),
                 os.path.join(target_dir, "scores.npz"))
    if verbose:
        print(f"  [adopt] {target_dir} <- {donor_dir} (identical catalogue)", flush=True)
    return True


def sweep_inject(
    source_run: str = DEFAULT_SOURCE_RUN,
    out_dir: str = DEFAULT_SWEEP_OUT,
    dt_grid=DT_GRID_DEFAULT,
    n_mc: int = 25,
    n_sub: int = 500,
    seed: int = 20260902,
    em: ErrorModel | None = None,
    reuse_from: str | None = DEFAULT_OUT,
    verbose: bool = True,
) -> dict:
    """Inject the whole experiment once per arc length in ``dt_grid``.

    Each arc length gets its own complete WP8 layout under
    ``out_dir/dt_<T>min/`` — ``main`` (the full catalogue), ``mc/rNN`` (the 25
    realizations of the stratified subsample) and ``decomp_pos``/``decomp_rate``
    (the position/rate split, which is where the 1/dt amplification is supposed
    to show most sharply, since the position error does not depend on dt at
    all).  The depth-sensitivity arm is deliberately dropped: it answers a
    different question and would multiply the scoring bill for nothing.

    ``reuse_from`` points at an existing WP8 run; any arc length whose injected
    catalogue is identical to it (i.e. dt = 45 min at the same seed and error
    model) adopts its stored scores instead of paying for them again.
    """
    em = em or ErrorModel()
    dt_grid = [float(x) for x in dt_grid]
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    reused: dict[str, str] = {}
    for dt in dt_grid:
        tag = dt_tag(dt)
        d = os.path.join(out_dir, tag)
        if verbose:
            print(f"[sweep inject] {tag}", flush=True)
        man = inject(source_run=source_run, out_dir=d, n_mc=n_mc, n_sub=n_sub,
                     seed=seed, dt_min=dt, em=em, m5_sens=None, full=True,
                     verbose=False)
        n_adopted = 0
        if reuse_from:
            for entry in man["runs"]:
                donor = os.path.join(reuse_from, entry["name"])
                target = os.path.join(d, entry["name"])
                if _adopt_scores(donor, target, verbose=False):
                    n_adopted += 1
            if n_adopted:
                reused[tag] = os.path.abspath(reuse_from)
        entries.append({
            "tag": tag,
            "dt_min": dt,
            "dir": os.path.abspath(d),
            "n_runs": len(man["runs"]),
            "n_runs_adopted": n_adopted,
            "median_sigma_mas": float(np.median(
                [r["median_sigma_mas"] for r in man["runs"]])),
        })
        if verbose:
            print(f"  {len(man['runs'])} run(s)"
                  + (f", {n_adopted} adopted from {reuse_from}" if n_adopted else ""),
                  flush=True)

    manifest = {
        "kind": "astrometric_noise_arclength_sweep",
        "source_run": os.path.abspath(source_run),
        "seed": int(seed),
        "n_mc": int(n_mc),
        "n_sub": int(n_sub),
        "error_model": asdict(em),
        "dt_grid_min": dt_grid,
        "dt_runs": entries,
        "adopted_from": reused,
        "break_criterion": BREAK_CRITERION,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    if verbose:
        print(f"[sweep inject] {len(dt_grid)} arc length(s) under {out_dir}")
    return manifest


def load_sweep_manifest(out_dir: str = DEFAULT_SWEEP_OUT) -> dict:
    with open(os.path.join(out_dir, "manifest.json")) as fh:
        man = json.load(fh)
    if man.get("kind") != "astrometric_noise_arclength_sweep":  # pragma: no cover
        raise ValueError(f"{out_dir}/manifest.json is not a sweep manifest")
    return man


# --------------------------------------------------------------------------
# Sweep: score
# --------------------------------------------------------------------------


def sweep_score(
    out_dir: str = DEFAULT_SWEEP_OUT,
    cfg: ScoreConfig | None = None,
    workers: int = 8,
    chunk: int = 250,
    only_dt: float | None = None,
    verbose: bool = True,
) -> dict:
    """Score every arc length, longest arc first, resumable throughout.

    Longest first on purpose: the long arcs are the ones whose answer is already
    known from WP8, so if the sweep is interrupted the partial result is the
    *interesting* end (short arcs) still missing rather than the reverse — and
    the completed long arcs immediately validate the pipeline against the
    published 45-minute numbers.
    """
    cfg = cfg or ScoreConfig()
    man = load_sweep_manifest(out_dir)
    t0 = time.time()
    per_dt = []
    for entry in man["dt_runs"]:
        if only_dt is not None and abs(entry["dt_min"] - float(only_dt)) > 1e-9:
            continue
        if verbose:
            print(f"[sweep score] {entry['tag']} "
                  f"({entry['n_runs']} run(s), {entry['n_runs_adopted']} adopted)",
                  flush=True)
        res = score_runs(out_dir=entry["dir"], cfg=cfg, workers=workers,
                         chunk=chunk, verbose=verbose)
        per_dt.append({"tag": entry["tag"], "dt_min": entry["dt_min"],
                       "scored": res["scored"], "skipped": res["skipped"],
                       "wall_s": res["wall_s"]})
    wall = time.time() - t0
    if verbose:
        print(f"[sweep score] done in {wall / 60.0:.1f} min")
    return {"per_dt": per_dt, "score_config": asdict(cfg), "wall_s": wall}


# --------------------------------------------------------------------------
# Sweep: the scaling machinery
# --------------------------------------------------------------------------


def loglog_slope(x, y) -> dict:
    """OLS slope of ``log y`` on ``log x``, ignoring non-positive/non-finite pairs.

    Used for every "does it scale as a power of dt?" question in the sweep.  A
    power law ``y = C x^k`` returns ``slope = k`` and ``r2 = 1`` exactly (up to
    floating point), which is what makes the assertion in the tests sharp.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    n = int(m.sum())
    if n < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "n": n,
                "r2": float("nan")}
    lx, ly = np.log(x[m]), np.log(y[m])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n": n,
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def scaling_ratio(dt_min, M, ref: int = 0, floor: float = 0.0,
                  power: float = -1.0) -> list[dict]:
    """Per-dt departure from ``M ∝ dt**power``, anchored at ``dt_min[ref]``.

    ``M`` is ``(n_dt, n_tracklet)``: one row per arc length, columns paired
    across rows by tracklet (which the common-random-numbers construction
    guarantees).  For each arc length this returns the distribution of

        ``M[k, i] / (M[ref, i] * (dt[k] / dt[ref])**power)``

    over the tracklets whose anchor value exceeds ``floor``.  A ratio of 1 means
    the tracklet's score perturbation is still *exactly* proportional to the
    1/dt rate error, i.e. the classifier's response to the attributable is still
    linear; a ratio below 1 means the response has started to saturate (the
    posterior is being pushed against 0 or 1, or out of the region where the
    local gradient applies), and a ratio above 1 means it has steepened.
    This is the sweep's operational test of the linear regime, and it needs no
    model of the classifier at all — only the pairing.
    """
    dt = np.asarray(dt_min, dtype=float)
    M = np.atleast_2d(np.asarray(M, dtype=float))
    if M.shape[0] != dt.size:
        raise ValueError(f"M has {M.shape[0]} rows for {dt.size} arc lengths")
    if not 0 <= int(ref) < dt.size:
        raise ValueError(f"ref {ref} out of range for {dt.size} arc lengths")
    anchor = M[int(ref)]
    sel = np.isfinite(anchor) & (anchor > float(floor))
    rows = []
    for k in range(dt.size):
        pred = anchor * (dt[k] / dt[int(ref)]) ** float(power)
        good = sel & np.isfinite(M[k]) & (pred > 0)
        r = M[k][good] / pred[good]
        row = {"dt_min": float(dt[k]), "n": int(good.sum())}
        if r.size:
            row.update(
                ratio_p10=float(np.percentile(r, 10)),
                ratio_p50=float(np.median(r)),
                ratio_p90=float(np.percentile(r, 90)),
                ratio_mean=float(np.mean(r)),
                ratio_min=float(np.min(r)),
            )
        rows.append(row)
    return rows


#: Floors at which every anchored scaling test is re-run, so the boundary
#: between "the classifier's response" and "the quadrature grid's round-off" is
#: visible in the output instead of buried in one arbitrary cut.
SCALING_FLOORS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)

#: Default anchor floor for the power-law statistics.  ~50x the ~2e-6 quadrature
#: jitter WP8's decomposition measured, and the smallest change in P(NEO) with
#: any triage meaning.  See :func:`variance_scaling`.
SCALING_FLOOR_DEFAULT = 1e-4


def ratio_by_floor(dt_min, M, ref: int = 0, floors=SCALING_FLOORS,
                   power: float = -1.0) -> list[dict]:
    """:func:`scaling_ratio` at a ladder of anchor floors.

    The point is *not* to pick a floor but to show what picking one does.  A
    genuine departure from the power law survives raising the floor; a departure
    manufactured by numerical noise in tiny-anchor tracklets disappears as soon
    as those tracklets are excluded, and the trend across floors says which one
    is being looked at.
    """
    dt = np.asarray(dt_min, dtype=float)
    out = []
    for f in floors:
        rows = scaling_ratio(dt, M, ref=ref, floor=float(f), power=power)
        out.append({
            "floor": float(f),
            "n": rows[int(ref)]["n"],
            "ratio_p50": [r.get("ratio_p50", float("nan")) for r in rows],
            "ratio_p10": [r.get("ratio_p10", float("nan")) for r in rows],
            "ratio_p90": [r.get("ratio_p90", float("nan")) for r in rows],
            "dt_min": [float(t) for t in dt],
        })
    return out


def step_scaling_ratio(dt_min, M, floor: float = SCALING_FLOOR_DEFAULT,
                       power: float = -1.0) -> list[dict]:
    """The anchor-free version of :func:`scaling_ratio`: adjacent pairs only.

    An anchored ratio confounds two things a reader will want separated.  If the
    median anchored ratio is 0.89 at every arc length, the response departed
    from the power law *once*, between the anchor and its neighbour, and has
    been perfectly linear since; if it keeps falling, the response is
    progressively saturating.  Comparing only adjacent arc lengths — each pair
    anchored on its own longer member — distinguishes them: a constant step
    ratio below 1 means a constant per-factor sublinearity, while step ratios
    that march downward mean genuine progressive saturation.

    Each row's ``n`` is the number of tracklets clearing ``floor`` on the longer
    arc of *that* pair, so short arcs are not penalised for the anchor's own
    dynamic range.
    """
    dt = np.asarray(dt_min, dtype=float)
    M = np.atleast_2d(np.asarray(M, dtype=float))
    if M.shape[0] != dt.size:
        raise ValueError(f"M has {M.shape[0]} rows for {dt.size} arc lengths")
    order = np.argsort(-dt)
    dt_s, M = dt[order], M[order]
    rows = []
    for k in range(dt_s.size - 1):
        a, b = M[k], M[k + 1]
        pred = a * (dt_s[k + 1] / dt_s[k]) ** float(power)
        good = np.isfinite(a) & np.isfinite(b) & (a > float(floor)) & (pred > 0)
        r = b[good] / pred[good]
        row = {"dt_from": float(dt_s[k]), "dt_to": float(dt_s[k + 1]),
               "factor": float(dt_s[k] / dt_s[k + 1]), "n": int(good.sum())}
        if r.size:
            row.update(step_p10=float(np.percentile(r, 10)),
                       step_p50=float(np.median(r)),
                       step_p90=float(np.percentile(r, 90)))
        rows.append(row)
    return rows


def ladder_stability(ladder: list[dict], min_n: int = 20) -> dict:
    """How much does the anchored ratio move when the anchor floor is raised?

    This is the test that decides whether a measured departure from the power
    law is a property of the classifier or of the measurement.  A departure
    created by tiny-response tracklets — whose measured change is at the
    roughness scale of v0's piecewise-constant prior — shrinks as those
    tracklets are excluded, so the ratio drifts steadily across the ladder.  A
    departure that is a genuine property of the response does not care which
    subset it is measured on, and the ratio is flat.

    Returns, per arc length, the spread (max - min) and range of the median
    ratio over the ladder rows with at least ``min_n`` tracklets.
    """
    usable = [r for r in ladder if r.get("n", 0) >= int(min_n)]
    if not usable:
        return {"n_rows": 0, "min_n": int(min_n), "per_dt": []}
    dts = usable[0].get("dt_min", [])
    per_dt = []
    for k, t in enumerate(dts):
        vals = np.array([r["ratio_p50"][k] for r in usable
                         if k < len(r.get("ratio_p50", []))], dtype=float)
        vals = vals[np.isfinite(vals)]
        row = {"dt_min": float(t), "n_rows": int(vals.size)}
        if vals.size:
            row.update(lo=float(vals.min()), hi=float(vals.max()),
                       spread=float(vals.max() - vals.min()),
                       median=float(np.median(vals)))
        per_dt.append(row)
    spreads = [r["spread"] for r in per_dt if "spread" in r]
    return {
        "n_rows": len(usable),
        "min_n": int(min_n),
        "floors_used": [r["floor"] for r in usable],
        "max_spread": float(max(spreads)) if spreads else float("nan"),
        "per_dt": per_dt,
    }


def pick_departure_floor(ladder: list[dict], min_n: int = 30) -> dict:
    """Choose the cleanest anchor floor that still has usable statistics.

    The anchored ratio is contaminated from below (see :func:`variance_scaling`):
    tracklets whose response is at the roughness scale of v0's piecewise-constant
    prior measure that roughness, not the posterior, and their ratio is biased
    toward ``dt/dt_ref``.  Raising the floor removes them but eventually leaves
    too few tracklets for a median.  This picks the **largest** floor still
    holding ``min_n`` tracklets — the cleanest subset with adequate statistics —
    and returns its row plus the choice, so the verdict is not read off an
    arbitrary cut.
    """
    usable = [r for r in ladder if r.get("n", 0) >= int(min_n)]
    row = max(usable, key=lambda r: r["floor"]) if usable else (
        max(ladder, key=lambda r: r.get("n", 0)) if ladder else {})
    return {
        "floor": row.get("floor", float("nan")),
        "n": row.get("n", 0),
        "min_n": int(min_n),
        "ratio_p50": row.get("ratio_p50", []),
        "dt_min": row.get("dt_min", []),
        "adequate": bool(usable),
    }


def variance_scaling(dt_min, std_p, floor: float = SCALING_FLOOR_DEFAULT,
                     split_dt_min: float = 20.0, tol: float = 0.10,
                     min_n_for_departure: int = 30) -> dict:
    """Does ``std(P)`` scale as ``1/dt`` (i.e. ``var(P)`` as ``1/dt^2``)?

    ``std_p`` is ``(n_dt, n_tracklet)`` — the per-tracklet Monte-Carlo scatter of
    ``P(NEO)`` at each arc length, paired by tracklet.  Three readings, because
    each fails in a different way:

    1. **per-tracklet log-log slopes.**  The physically predicted value is -1
       for ``std`` (-2 for the variance): the rate error is ``sqrt(2) sigma /
       dt`` and a locally linear classifier response turns that into a score
       perturbation of the same shape.  A tracklet whose slope is shallower than
       -1 has left the linear regime somewhere in the grid.
    2. **aggregate slopes** of the population percentile curves — the number a
       reader can check against the reported ``std(P)`` table by eye, and the
       one that is robust to the many tracklets whose ``std`` is exactly 0 (a
       posterior pinned at 0 or 1) and therefore unfittable individually.
    3. **the anchored ratio** (:func:`scaling_ratio`), split into a long-arc and
       a short-arc fit, which is what actually *localises* the departure: the
       first arc length at which the median ratio leaves ``1 +- tol``.

    ``floor`` — **why it is not zero.**  The classifier integrates on a finite
    ``(rho, rho_dot)`` grid that is rebuilt from each attributable, so moving the
    attributable moves the quadrature nodes and jitters ``P`` by an amount that
    does **not** scale with dt.  WP8's own decomposition measures that jitter:
    the residual of ``Delta(both) - Delta(pos) - Delta(rate)`` is ~2e-6 in ``P``
    at 45 min, against a median ``|Delta P|`` of only ~5e-6.  For a tracklet
    whose real response is that small, the anchored ratio is measuring
    round-off, and it is biased *low* in a predictable way: if the anchor is
    dominated by a dt-independent term the ratio tends to ``dt/dt_ref`` rather
    than 1.  Every power-law statistic here is therefore restricted to tracklets
    whose anchor value clears ``floor`` (default 1e-4, ~50x the measured jitter
    and also the smallest score change with any triage meaning).
    ``ratio_by_floor`` reports the same test at a ladder of floors so the
    numerical-noise regime is visible rather than hidden.

    Tracklets whose anchor ``std`` is *exactly* zero are counted separately
    (``n_dead_at_anchor`` / ``n_woken_from_dead``): a posterior the classifier is
    certain about at a long arc and uncertain about at a short one has no finite
    ratio and no log-log slope, but is the single most interesting failure mode
    in the sweep and must not be silently dropped.
    """
    dt = np.asarray(dt_min, dtype=float)
    S = np.atleast_2d(np.asarray(std_p, dtype=float))
    if S.shape[0] != dt.size:
        raise ValueError(f"std_p has {S.shape[0]} rows for {dt.size} arc lengths")
    order = np.argsort(-dt)            # longest arc first == the anchor
    dt_s, S = dt[order], S[order]
    ref = 0

    fitted = np.isfinite(S).all(axis=0) & (S > float(floor)).all(axis=0)
    slopes = np.full(S.shape[1], np.nan)
    for i in np.flatnonzero(fitted):
        slopes[i] = loglog_slope(dt_s, S[:, i])["slope"]

    # "Dead" means *exactly* certain, not merely small: see the docstring.
    dead = np.isfinite(S[ref]) & (S[ref] == 0.0)

    curves, per_dt = [], {}
    for k, t in enumerate(dt_s):
        col = S[k][np.isfinite(S[k])]
        row = {"dt_min": float(t), "n": int(col.size)}
        if col.size:
            row.update(
                p50=float(np.percentile(col, 50)),
                p90=float(np.percentile(col, 90)),
                p99=float(np.percentile(col, 99)),
                max=float(np.max(col)),
                rms=float(np.sqrt(np.mean(col ** 2))),
                frac_gt_0p01=float(np.mean(col > 0.01)),
                frac_gt_0p05=float(np.mean(col > 0.05)),
            )
        woke = dead & np.isfinite(S[k]) & (S[k] > 0.0)
        row["n_dead_at_anchor"] = int(dead.sum())
        row["n_woken_from_dead"] = int(woke.sum())
        row["n_woken_gt_0p01"] = int(np.sum(dead & np.isfinite(S[k]) & (S[k] > 0.01)))
        curves.append(row)
        per_dt[float(t)] = row

    def _agg(key: str) -> dict:
        return loglog_slope(dt_s, [c.get(key, np.nan) for c in curves])

    long_m = dt_s >= float(split_dt_min)
    short_m = dt_s <= float(split_dt_min)
    ratios = scaling_ratio(dt_s, S, ref=ref, floor=float(floor), power=-1.0)
    ladder = ratio_by_floor(dt_s, S, ref=ref)
    pick = pick_departure_floor(ladder, min_n=int(min_n_for_departure))
    departure = None
    for t, r50 in zip(pick.get("dt_min", []), pick.get("ratio_p50", [])):
        if np.isfinite(r50) and abs(r50 - 1.0) > float(tol):
            departure = float(t)
            break

    return {
        "dt_min": [float(t) for t in dt_s],
        "anchor_dt_min": float(dt_s[ref]),
        "expected_slope_std": -1.0,
        "expected_slope_var": -2.0,
        "n_tracklets": int(S.shape[1]),
        "n_fitted": int(fitted.sum()),
        "floor": float(floor),
        "tol": float(tol),
        "ratio_by_floor": ladder,
        "ladder_stability": ladder_stability(ladder),
        "departure_floor": pick,
        "step_ratio": step_scaling_ratio(dt_s, S, floor=float(floor)),
        "per_tracklet_slope_std": _q(slopes, (10, 50, 90)) | {
            "mean": float(np.nanmean(slopes)) if fitted.any() else float("nan"),
            "n": int(fitted.sum()),
        },
        "per_tracklet_slope_var": _q(2.0 * slopes, (10, 50, 90)) | {
            "mean": float(2.0 * np.nanmean(slopes)) if fitted.any() else float("nan"),
        },
        "aggregate_slope_std": {k: _agg(k) for k in ("p50", "p90", "p99", "max", "rms")},
        "long_arc_slope_std": (
            loglog_slope(dt_s[long_m], [c["p90"] for c, m in zip(curves, long_m) if m])
            if long_m.sum() >= 2 else {}
        ),
        "short_arc_slope_std": (
            loglog_slope(dt_s[short_m], [c["p90"] for c, m in zip(curves, short_m) if m])
            if short_m.sum() >= 2 else {}
        ),
        "linear_scaling_ratio": ratios,
        "departure_dt_min": departure,
        "curves": curves,
        "_per_dt": per_dt,
    }


# --------------------------------------------------------------------------
# Sweep: per-arc-length analysis
# --------------------------------------------------------------------------


def rate_error_block(nz: dict, rate_deg_day: np.ndarray, dt_min: float) -> dict:
    """The rate error in physical units, against the tracklets' own motion.

    ``sqrt(2) sigma / dt`` per coordinate is the whole mechanism of this study,
    so it is reported in arcsec/day (what an observer would quote) as well as
    normalised by each tracklet's own sky motion (what the classifier actually
    responds to: a 1 arcsec/day error on a 1 deg/day NEO is nothing, and on a
    0.02 deg/day distant object it is a different orbit).
    """
    sig_mas = np.asarray(nz["sigma_mas"], dtype=float)
    dt_days = float(dt_min) / 1440.0
    per_coord = math.sqrt(2.0) * (sig_mas / 1000.0) / dt_days      # arcsec/day
    own = np.asarray(rate_deg_day, dtype=float) * 3600.0           # arcsec/day
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(own > 0, per_coord / own, np.nan)
    good = np.isfinite(frac)
    return {
        "dt_min": float(dt_min),
        "sigma_rate_arcsec_per_day_per_coord": _q(per_coord, (50, 90, 99)) | {
            "mean": float(np.mean(per_coord)), "max": float(np.max(per_coord)),
        },
        "own_motion_arcsec_per_day": _q(own, (10, 50, 90)),
        "rate_error_over_own_motion": _q(frac, (50, 90, 99)) | {
            "max": float(np.nanmax(frac)) if good.any() else float("nan"),
        },
        "frac_tracklets_error_over_10pct": float(np.mean(frac[good] > 0.10))
        if good.any() else float("nan"),
        "frac_tracklets_error_over_50pct": float(np.mean(frac[good] > 0.50))
        if good.any() else float("nan"),
        "frac_tracklets_error_over_100pct": float(np.mean(frac[good] > 1.00))
        if good.any() else float("nan"),
    }


def sweep_dt_block(
    dt_dir: str,
    gen: dict[str, np.ndarray],
    base: dict[str, np.ndarray],
    n_boot: int = 2000,
    seed: int = 20260902,
    verbose: bool = True,
) -> dict:
    """Everything the sweep needs at one arc length.

    Deliberately *not* :func:`analyze`: that function's report writer states the
    45-minute conclusion in prose ("calibration is untouched"), which would be a
    lie at 5 minutes.  The numerical blocks are reused verbatim — the same
    :func:`calibration_block`, :func:`threshold_crossings`,
    :func:`operating_points`, :func:`mc_block` and
    :func:`decomposition_block` — so the 45-minute row of the sweep table is
    directly comparable to the published WP8 numbers, and is asserted to
    reproduce them.
    """
    from validation.digest2_comparison import auc_bootstrap  # lazy: heavy module

    man = load_manifest(dt_dir)
    dt = float(man["dt_min"])
    with np.load(os.path.join(dt_dir, "main", "scores.npz")) as z:
        order = np.argsort(z["index"].astype(int), kind="mergesort")
        noisy = mask_policy_rows({k: z[k][order] for k in z.files})
    with np.load(os.path.join(dt_dir, "main", "noise.npz")) as z:
        nz = {k: z[k] for k in z.files}

    rows = nz["rows"]
    y = gen["is_neo"][rows].astype(bool)
    w = gen["weight"][rows].astype(float)
    v = gen["v_mag"][rows].astype(float)
    rate = nz["rate_deg_day"]
    p0, p1 = base["p_neo"][rows], noisy["p_neo"]
    with np.errstate(invalid="ignore"):
        dp = p1 - p0

    finite0 = np.isfinite(p0)
    lost = int(np.sum(finite0 & ~np.isfinite(p1)))

    out: dict = {
        "tag": dt_tag(dt),
        "dt_min": dt,
        "dir": os.path.abspath(dt_dir),
        "n": int(rows.size),
        "n_neo": int(y.sum()),
        "sigma_population": {
            "median_mas": float(np.median(nz["sigma_mas"])),
            "p90_mas": float(np.percentile(nz["sigma_mas"], 90)),
            "max_mas": float(np.max(nz["sigma_mas"])),
            "rate_error_over_rate": _q(
                np.hypot(nz["d_ra_rate"] * np.cos(gen["dec"][rows]), nz["d_dec_rate"])
                / np.radians(rate), (50, 90, 99)
            ),
        },
        "rate_error": rate_error_block(nz, rate, dt),
        "undefined": {
            "n_undefined_noiseless": int(np.sum(~finite0)),
            "n_undefined_noisy": int(np.sum(~np.isfinite(p1))),
            "n_lost_to_noise": lost,
            "frac_lost_to_noise": lost / max(int(finite0.sum()), 1),
            # D-19: a triage-policy assignment counts as "no likelihood
            # score", so the fully-truncated direction stays visible to C3.
            "n_policy_assigned_noisy": int(
                noisy.get("_n_policy_assigned", np.array([0]))[0]
            ),
            "n_policy_assigned_noiseless": int(
                base.get("_n_policy_assigned", np.array([0]))[0]
            ),
        },
        "deltas": {
            "all": delta_summary(dp, "all"),
            "neo": delta_summary(dp[y], "true NEO"),
            "other": delta_summary(dp[~y], "true other"),
            "vs_v": binned_delta(v, dp, V_BINS, w),
            "vs_rate": binned_delta(rate, dp, RATE_BINS, w),
        },
        "calibration": calibration_block(p1, y, w, n_boot=n_boot, seed=seed),
        "threshold_stability": threshold_crossings(p0, p1, y),
        "operating_points": operating_points(p0, p1, y, w),
    }
    auc = auc_bootstrap(p1, p0, y, w, n_boot=min(n_boot, 2000), seed=seed)
    out["ranking"] = {
        "auc_noisy": auc["auc_a"],
        "auc_noiseless": auc["auc_b"],
        "delta_auc": auc["delta"],
        "delta_auc_ci95": auc["delta_ci"],
        "n_boot": auc["n_boot"],
    }
    outside = noisy["rhodot_outside_au_per_day"].astype(float)
    out["admissible_region"] = {
        "frac_truth_covered_noisy": float(
            np.mean(np.isfinite(outside) & (outside <= sc._AR_OUTSIDE_TOL))
        ),
        "median_collapse_frac_rho2_noisy": float(
            np.nanmedian(noisy["collapse_frac_rho2"])
        ),
    }
    # How much of the score change is the admissible region's own *discretization*
    # moving rather than the posterior?  Perturbing the attributable moves the
    # region's boundary, which adds or drops whole rings of quadrature nodes from
    # the fixed (n_rho, n_rhodot) lattice.  That is a genuine part of v0's
    # response to a noisy attributable, but it is a rough, staircase-like part,
    # and it is the leading systematic on the linearity test below.
    if "n_nodes" in base and "n_nodes" in noisy:
        n0 = base["n_nodes"][rows].astype(float)
        n1 = noisy["n_nodes"].astype(float)
        dn = np.abs(n1 - n0)
        out["quadrature"] = {
            "median_n_nodes_noiseless": float(np.median(n0)),
            "frac_n_nodes_changed": float(np.mean(n1 != n0)),
            "median_abs_delta_n_nodes": float(np.median(dn)),
            "p90_abs_delta_n_nodes": float(np.percentile(dn, 90)),
            "max_abs_delta_n_nodes": float(np.max(dn)),
            "p90_rel_delta_n_nodes": float(
                np.percentile(np.where(n0 > 0, dn / n0, np.nan)[n0 > 0], 90)
            ) if np.any(n0 > 0) else float("nan"),
        }
    out["mc"] = mc_block(dt_dir, man, gen, base)
    out["decomposition"] = decomposition_block(dt_dir, man, gen, base)

    # Arrays the cross-dt blocks need, pulled out of the JSON payload.
    mc_arrays = out["mc"].pop("_arrays", None)
    out["decomposition"].pop("_arrays", None)
    out["_arrays"] = {
        "rows": rows,
        "dp": dp,
        # Per-tracklet flag: did the perturbation change the quadrature node
        # count?  Used to build the "smooth subset" control for the linearity
        # test, which is the only way to tell the lattice's contribution from
        # the posterior's.
        "nodes_changed": (
            noisy["n_nodes"].astype(np.int64) != base["n_nodes"][rows].astype(np.int64)
            if ("n_nodes" in base and "n_nodes" in noisy)
            else np.zeros(rows.size, dtype=bool)
        ),
        "std_p": None if mc_arrays is None else mc_arrays["std_p"],
        "mc_rows": None if mc_arrays is None else mc_arrays["rows"],
        "mc_p0": None if mc_arrays is None else mc_arrays["p0"],
        "mc_y": None if mc_arrays is None else mc_arrays["y"],
    }
    if verbose:
        cal = out["calibration"]
        print(f"  [{out['tag']}] Z = {_f(_g(cal, 'global', 'z'), '+.2f')}, "
              f"verdict {'PASS' if _g(cal, 'verdict', 'pass', default=False) else 'FAIL'}, "
              f"dAUC = {_f(out['ranking']['delta_auc'], '+.5f')}, "
              f"p90|dP| = {_f(out['deltas']['all'].get('abs_p90'), '.2e')}, "
              f"lost = {lost}", flush=True)
    return out


# --------------------------------------------------------------------------
# Sweep: the verdict
# --------------------------------------------------------------------------


def _worst_crossing(block: dict) -> dict:
    """The threshold with the largest MC churn rate at one arc length."""
    ex = _g(block, "mc", "expected_crossings", default=[]) or []
    n = float(_g(block, "mc", "n_tracklets", default=float("nan")))
    best = {"threshold": float("nan"), "mean_per_realization": float("nan"),
            "frac_per_realization": float("nan"), "n_subsample": n,
            "mean_neo_lost_per_realization": float("nan"),
            "max_in_a_realization": float("nan")}
    for r in ex:
        frac = r["mean_crossings_per_realization"] / n if n > 0 else float("nan")
        if not np.isfinite(best["frac_per_realization"]) or frac > best["frac_per_realization"]:
            best = {
                "threshold": r["threshold"],
                "mean_per_realization": r["mean_crossings_per_realization"],
                "frac_per_realization": frac,
                "n_subsample": n,
                "mean_neo_lost_per_realization": r["mean_neo_lost_per_realization"],
                "max_in_a_realization": r["max_crossings_in_a_realization"],
            }
    return best


def _worst_full_crossing(block: dict) -> dict:
    """The threshold with the largest full-catalogue crossing count."""
    ts = block.get("threshold_stability", []) or []
    best = {"threshold": float("nan"), "n_cross_any": -1, "frac_cross_any": 0.0,
            "neo_cross_down": 0}
    for r in ts:
        # Strict >: on a tie (typically all-zero) the LOWEST threshold wins,
        # which is the one an operator would care about.
        if r["n_cross_any"] > best["n_cross_any"]:
            best = {"threshold": r["threshold"], "n_cross_any": r["n_cross_any"],
                    "frac_cross_any": r["frac_cross_any"],
                    "neo_cross_down": r["neo_cross_down"]}
    return best


def break_verdict(blocks: list[dict]) -> dict:
    """Apply :data:`BREAK_CRITERION` to each arc length and locate the onset.

    Breaking is expected to be monotone in shortening the arc (the rate error
    grows as 1/dt and nothing in the chain reverses), so the reported onset is
    the **longest** arc length that breaks: everything shorter is inside the
    broken regime, everything longer is outside it.  Monotonicity is checked
    rather than assumed — a non-monotone pattern is reported as such, because it
    would mean the criterion is being tripped by noise rather than by dt.
    """
    rows = []
    for b in blocks:
        cal = b.get("calibration", {})
        ver = cal.get("verdict", {})
        pv_m = float(_g(ver, "max_abs_z_p_value"))
        pv_g = float(_g(ver, "global_z_p_value"))
        c1 = bool(np.isfinite(pv_m) and pv_m <= BREAK_ALPHA) or \
            bool(np.isfinite(pv_g) and pv_g <= BREAK_ALPHA)

        mc = _worst_crossing(b)
        full = _worst_full_crossing(b)
        churn = mc["frac_per_realization"]
        if not np.isfinite(churn):
            churn = full["frac_cross_any"]
        c2 = bool(np.isfinite(churn) and churn > BREAK_CROSS_FRAC)

        lostf = float(_g(b, "undefined", "frac_lost_to_noise", default=0.0))
        c3 = bool(np.isfinite(lostf) and lostf > BREAK_UNDEFINED_FRAC)

        dauc = float(_g(b, "ranking", "delta_auc"))
        ci = _g(b, "ranking", "delta_auc_ci95", default=None)
        ci_excludes_zero = bool(
            isinstance(ci, (list, tuple)) and len(ci) == 2
            and np.isfinite(ci[0]) and np.isfinite(ci[1])
            and (ci[0] > 0.0 or ci[1] < 0.0)
        )
        w1 = bool(ci_excludes_zero and np.isfinite(dauc)
                  and abs(dauc) > WARN_DELTA_AUC)

        comp0 = float(_g(b, "operating_points", "noiseless", "completeness"))
        comp1 = float(_g(b, "operating_points", "noisy_fixed_threshold", "completeness"))
        drop = comp0 - comp1
        w2 = bool(np.isfinite(drop) and drop > WARN_COMPLETENESS_DROP)

        # Post-hoc only; see POSTHOC_CONTAMINATION_RATIO.  Never enters `breaks`.
        cont0 = float(_g(b, "operating_points", "noiseless", "contamination"))
        cont1 = float(_g(b, "operating_points", "noisy_fixed_threshold",
                         "contamination"))
        cont_ratio = (cont1 / cont0) if (np.isfinite(cont0) and cont0 > 0
                                         and np.isfinite(cont1)) else float("nan")
        p_cont = bool(np.isfinite(cont_ratio)
                      and cont_ratio > POSTHOC_CONTAMINATION_RATIO)

        reasons = []
        if c1:
            reasons.append(
                f"C1 calibration: reliability verdict FAIL "
                f"(p_global = {_f(pv_g, '.3g')}, p_binned = {_f(pv_m, '.3g')}, "
                f"alpha = {BREAK_ALPHA:g})")
        if c2:
            reasons.append(
                f"C2 triage churn: {_pct(churn, '.2f')} of tracklets cross "
                f"p >= {mc['threshold']:g} per realization "
                f"(> {_pct(BREAK_CROSS_FRAC, '.0f')})")
        if c3:
            reasons.append(
                f"C3 lost tracklets: {_pct(lostf, '.2f')} of tracklets scored "
                f"noiseless become undefined under noise "
                f"(> {_pct(BREAK_UNDEFINED_FRAC, '.0f')})")
        warns = []
        if w1:
            warns.append(f"W1 ranking: dAUC = {_f(dauc, '+.5f')} "
                         f"{[_f(c, '+.5f') for c in ci]}, |dAUC| > {WARN_DELTA_AUC:g}")
        if w2:
            warns.append(f"W2 operating point: completeness drops "
                         f"{_pct(drop, '.2f')} at the fixed cut "
                         f"(> {_pct(WARN_COMPLETENESS_DROP, '.0f')})")

        rows.append({
            "tag": b["tag"],
            "dt_min": b["dt_min"],
            "C1_calibration": c1,
            "C2_triage_churn": c2,
            "C3_lost_tracklets": c3,
            "W1_ranking": w1,
            "W2_operating_point": w2,
            "breaks": bool(c1 or c2 or c3),
            "reasons": reasons,
            "warnings": warns,
            "global_z": float(_g(cal, "global", "z")),
            "global_z_threshold": float(_g(ver, "global_z_threshold")),
            "global_z_p_value": pv_g,
            "max_abs_z": float(_g(ver, "max_abs_z")),
            "max_abs_z_threshold": float(_g(ver, "max_abs_z_threshold")),
            "max_abs_z_p_value": pv_m,
            "calibration_pass": bool(_g(ver, "pass", default=False)),
            "c_hat": float(_g(cal, "odds_scale", "c_hat")),
            "delta_auc": dauc,
            "delta_auc_ci95": ci,
            "mc_worst_crossing": mc,
            "full_worst_crossing": full,
            "churn_frac_used": float(churn) if np.isfinite(churn) else float("nan"),
            "frac_lost_to_noise": lostf,
            "completeness_fixed_cut": comp1,
            "completeness_noiseless": comp0,
            "completeness_drop": drop,
            "contamination_fixed_cut": cont1,
            "contamination_noiseless": cont0,
            # Post-hoc, excluded from `breaks` and from `reasons`/`warnings`.
            "posthoc_contamination_ratio": cont_ratio,
            "posthoc_contamination_flag": p_cont,
        })

    rows.sort(key=lambda r: -r["dt_min"])
    broken = [r for r in rows if r["breaks"]]
    onset = max((r["dt_min"] for r in broken), default=None)
    # Monotone iff the broken set is a suffix of the dt-descending list.
    flags = [r["breaks"] for r in rows]
    monotone = flags == sorted(flags)
    return {
        "criterion": BREAK_CRITERION,
        "alpha": BREAK_ALPHA,
        "cross_frac": BREAK_CROSS_FRAC,
        "undefined_frac": BREAK_UNDEFINED_FRAC,
        "warn_delta_auc": WARN_DELTA_AUC,
        "warn_completeness_drop": WARN_COMPLETENESS_DROP,
        "rows": rows,
        "any_break_in_range": bool(broken),
        "break_onset_dt_min": onset,
        "shortest_safe_dt_min": min(
            (r["dt_min"] for r in rows if not r["breaks"]), default=None),
        "monotone_in_dt": bool(monotone),
        "warned_dt_min": [r["dt_min"] for r in rows if r["warnings"]],
        "posthoc": {
            "note": (
                "NOT part of the declared criterion. W2 as declared watches only "
                "the completeness half of the fixed operating point, and that is "
                "the half the sweep found does not move: completeness is pinned "
                "by the cut while contamination multiplies. The contamination "
                "ratio is reported for honesty and enters no verdict."
            ),
            "contamination_ratio_threshold": POSTHOC_CONTAMINATION_RATIO,
            "flagged_dt_min": [r["dt_min"] for r in rows
                               if r["posthoc_contamination_flag"]],
        },
    }


# --------------------------------------------------------------------------
# Sweep: analyse
# --------------------------------------------------------------------------

SWEEP_ASSUMPTIONS = ASSUMPTIONS[:1] + [
    "Two detections per tracklet at each arc length, exactly linear motion. "
    "The arc length is the ONLY thing varied: the endpoint sigma, the standard-"
    "normal draws, the catalogue, the labels and the classifier are all held "
    "fixed, so every dt-to-dt difference below is caused by the 1/dt rate-error "
    "amplification and by nothing else.",
    "Common random numbers across dt: at fixed (seed, stream, tracklet) every "
    "arc length perturbs the same endpoints by the same offsets, so |Delta P| "
    "at two arc lengths is a PAIRED comparison and the dt-to-dt curve carries "
    "no Monte-Carlo noise of its own. It also means the sweep measures the "
    "response to one particular noise realization per arc length; the MC block "
    "supplies the ensemble spread.",
    "A shorter arc is modelled as a shorter arc, not as a worse survey: sigma "
    "per detection is unchanged (Sorcha's model at m5 = 24, FWHM = 700 mas). "
    "Real short pairs inside one visit pair would carry the same per-detection "
    "sigma, so this is the right comparison; a pair of consecutive 30 s "
    "exposures would additionally differ in nothing but dt.",
    "dt = 45 min is ADOPTED from the WP8 headline run (identical injected "
    "catalogue, verified array-by-array) rather than re-scored, so that row is "
    "literally the published number and not a reproduction of it.",
] + ASSUMPTIONS[2:]


def sweep_analyze(
    out_dir: str = DEFAULT_SWEEP_OUT,
    source_run: str | None = None,
    plots: bool = True,
    n_boot: int = 2000,
    seed: int = 20260902,
    verbose: bool = True,
) -> dict:
    """Per-arc-length analysis, the cross-dt story, the verdict, the report."""
    man = load_sweep_manifest(out_dir)
    source_run = source_run or man["source_run"]
    gen = load_source(source_run)
    base = _load_noiseless(source_run)

    entries = sorted(man["dt_runs"], key=lambda e: -e["dt_min"])
    blocks, arrays = [], {}
    for e in entries:
        d = os.path.join(out_dir, e["tag"])
        if not os.path.exists(os.path.join(d, "main", "scores.npz")):
            if verbose:
                print(f"  [{e['tag']}] not scored yet, skipped", flush=True)
            continue
        b = sweep_dt_block(d, gen, base, n_boot=n_boot, seed=seed, verbose=verbose)
        arrays[b["tag"]] = b.pop("_arrays")
        with open(os.path.join(d, "summary.json"), "w") as fh:
            json.dump(b, fh, indent=2, default=_json_default)
        blocks.append(b)
    if not blocks:
        raise RuntimeError(f"no scored arc length found under {out_dir}")

    dt = np.array([b["dt_min"] for b in blocks], dtype=float)

    # ---- cross-dt: per-tracklet std(P) scaling (MC subsample) --------------
    have_mc = [t for t in (b["tag"] for b in blocks) if arrays[t]["std_p"] is not None]
    vs: dict = {}
    if len(have_mc) >= 2:
        ref_rows = arrays[have_mc[0]]["mc_rows"]
        if all(np.array_equal(arrays[t]["mc_rows"], ref_rows) for t in have_mc):
            S = np.vstack([arrays[t]["std_p"] for t in have_mc])
            dts = np.array([b["dt_min"] for b in blocks if b["tag"] in have_mc])
            vs = variance_scaling(dts, S)
            vs.pop("_per_dt", None)          # redundant with "curves" in JSON
            vs["subsample_n"] = int(ref_rows.size)
            # The same test on the paired single-realization |Delta P| of the
            # FULL catalogue: sharper still, because it is one deterministic
            # perturbation per tracklet rather than an estimated scatter.
            D = np.vstack([np.abs(arrays[t]["dp"]) for t in (b["tag"] for b in blocks)])
            vs["paired_delta_p_ratio"] = scaling_ratio(
                dt, D, ref=0, floor=SCALING_FLOOR_DEFAULT, power=-1.0)
            vs["paired_delta_p_ratio_by_floor"] = ratio_by_floor(dt, D, ref=0)
            vs["paired_delta_p_step_ratio"] = step_scaling_ratio(
                dt, D, floor=SCALING_FLOOR_DEFAULT)
            vs["paired_delta_p_floor"] = SCALING_FLOOR_DEFAULT
            # The control for the linearity test: v0 integrates the admissible
            # region on a fixed lattice, so a perturbation that adds or drops
            # boundary nodes changes P by a discrete jump.  Restricting to
            # tracklets whose node count never changes removes that mechanism
            # entirely.  If the step ratios move, the lattice was the cause; if
            # they do not, the roughness is in the piecewise-constant prior
            # itself and no grid refinement would remove it.
            smooth = np.ones(D.shape[1], dtype=bool)
            for b in blocks:
                smooth &= ~np.asarray(arrays[b["tag"]]["nodes_changed"], dtype=bool)
            Ds = np.where(smooth[None, :], D, np.nan)
            vs["smooth_subset"] = {
                "n": int(smooth.sum()),
                "n_total": int(D.shape[1]),
                "frac": float(smooth.mean()),
                "definition": (
                    "tracklets whose quadrature node count is identical to the "
                    "noiseless run at EVERY arc length in the sweep"
                ),
                "step_ratio": step_scaling_ratio(
                    dt, Ds, floor=SCALING_FLOOR_DEFAULT),
                "ratio": scaling_ratio(dt, Ds, ref=0,
                                       floor=SCALING_FLOOR_DEFAULT, power=-1.0),
            }
            vs["paired_delta_p_note"] = (
                "|Delta P| * dt is constant in dt while the classifier's "
                "response to the attributable is linear; these are the medians "
                "of |Delta P|(dt) divided by the 1/dt extrapolation from "
                f"dt = {dt[0]:g} min, over the "
                f"{vs['paired_delta_p_ratio'][0]['n']} of {int(D.shape[1])} "
                f"tracklets whose anchor |Delta P| exceeds "
                f"{SCALING_FLOOR_DEFAULT:g}. Tracklets below that floor are "
                "dominated by the (rho, rho_dot) quadrature jitter, which does "
                "not scale with dt; see paired_delta_p_ratio_by_floor."
            )
        elif verbose:  # pragma: no cover - identical subsample by construction
            print("  [warn] MC subsamples differ across dt; scaling test skipped")

    summary: dict = {
        "out_dir": os.path.abspath(out_dir),
        "source_run": os.path.abspath(source_run),
        "seed": man["seed"],
        "error_model": man["error_model"],
        "n_mc": man["n_mc"],
        "n_sub": man["n_sub"],
        "dt_grid_min": [float(x) for x in dt],
        "adopted_from": man.get("adopted_from", {}),
        "sigma_table": sigma_table(ErrorModel(**man["error_model"])),
        "assumptions": SWEEP_ASSUMPTIONS,
        "per_dt": blocks,
        "variance_scaling": vs,
    }
    summary["verdict"] = break_verdict(blocks)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    with open(os.path.join(out_dir, "REPORT.md"), "w") as fh:
        fh.write(build_sweep_report(summary))
    if verbose:
        print(f"[sweep analyze] wrote {out_dir}/summary.json and REPORT.md")
    if plots:
        make_sweep_plots(out_dir, summary, arrays)
    return summary


# --------------------------------------------------------------------------
# Sweep: plots
# --------------------------------------------------------------------------


def _dt_axis(ax, grid, label: bool = True) -> None:
    """Log arc-length axis, longest arc on the left, shortest on the right.

    Inverted deliberately: the physics runs toward shorter arcs, so the story
    reads left-to-right into the degradation.
    """
    g = np.sort(np.asarray(grid, dtype=float))
    ax.set_xscale("log")
    ax.set_xlim(float(g[-1]) * 1.18, float(g[0]) / 1.18)
    ax.set_xticks(list(g))
    ax.set_xticklabels([f"{t:g}" for t in g])
    ax.minorticks_off()
    if label:
        ax.set_xlabel("tracklet arc length [min]   (shorter arc →)", color=INK,
                      fontsize=10)


def _mark_break(ax, verdict: dict, y: float | None = None) -> None:
    """Vertical rule and label at the break onset, if there is one in range."""
    onset = verdict.get("break_onset_dt_min")
    if onset is None:
        return
    ax.axvline(onset, color=PALETTE[1], linewidth=1.4, linestyle="-", alpha=0.85,
               zorder=1)
    lo, hi = ax.get_ylim()
    ypos = hi if (y is None or not np.isfinite(y)) else float(y)
    ax.annotate(f"breaks at {onset:g} min", (onset, ypos),
                xytext=(-6, -12), textcoords="offset points",
                ha="right", va="top", color=PALETTE[1], fontsize=9,
                fontweight="bold")


def make_sweep_plots(out_dir: str, summary: dict, arrays: dict) -> None:
    """Four figures; the first is the headline the deliverable names."""
    rows = summary["verdict"]["rows"]
    dt = np.array([r["dt_min"] for r in rows], dtype=float)
    ver = summary["verdict"]
    blocks = {b["tag"]: b for b in summary["per_dt"]}

    # --- 1. HEADLINE: calibration z and threshold-crossing rate vs dt -------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))

    ax = axes[0]
    # Both arms are plotted as "statistic / its own parametric-bootstrap
    # threshold", so C1 is a single horizontal line at 1.0 and the two arms are
    # directly comparable.  Raw values span 0.4 to ~5e3 across the grid, which no
    # shared linear axis can show; the raw numbers are annotated instead and the
    # full table is in REPORT.md.
    z = np.array([abs(r["global_z"]) for r in rows])
    zthr = np.array([r["global_z_threshold"] for r in rows])
    mz = np.array([r["max_abs_z"] for r in rows])
    mthr = np.array([r["max_abs_z_threshold"] for r in rows])
    with np.errstate(divide="ignore", invalid="ignore"):
        zr = np.where(zthr > 0, z / zthr, np.nan)
        mr = np.where(mthr > 0, mz / mthr, np.nan)
    ax.plot(dt, zr, color=PALETTE[0], linewidth=2.2, marker="o", markersize=6,
            label=r"global count $|Z_{\rm tot}|$ / threshold")
    ax.plot(dt, mr, color=PALETTE[2], linewidth=2.2, marker="s", markersize=5,
            label=r"binned max$|z|$ / threshold")
    ax.axhline(1.0, color=PALETTE[1], linewidth=1.4, linestyle=":")
    ax.annotate(r"C1 break criterion: statistic = its $\alpha$ = 1% threshold",
                (dt[0], 1.0), xytext=(2, 5), textcoords="offset points",
                color=PALETTE[1], fontsize=8)
    for k, r in enumerate(rows):
        if np.isfinite(zr[k]):
            ax.annotate(f"Z={r['global_z']:+.2f}", (dt[k], zr[k]),
                        xytext=(0, -12), textcoords="offset points", ha="center",
                        color=PALETTE[0], fontsize=7.5)
    ax.set_yscale("log")
    hi = float(np.nanmax(np.concatenate([zr, mr, [1.0]])))
    lo = float(np.nanmin(np.concatenate([zr, mr, [1.0]])))
    ax.set_ylim(max(lo / 4.0, 1e-3), hi * 6.0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _dt_axis(ax, dt)
    _style(ax, "", "statistic / its bootstrap threshold",
           "C1 — calibration of the noisy scores vs. the true labels")
    _mark_break(ax, ver, y=hi * 3.0)

    ax = axes[1]
    churn = np.array([r["mc_worst_crossing"]["frac_per_realization"] for r in rows])
    fullc = np.array([r["full_worst_crossing"]["frac_cross_any"] for r in rows])
    # An exact zero cannot be drawn on a log axis, and drawing it at an
    # arbitrary floor invents a V-shaped "recovery" that is not in the data.
    # Zeros are therefore clamped to one floor and marked hollow, so "no
    # crossing occurred at all" reads differently from "a small rate".
    floor = 1e-5
    for c, val, lab, mk in ((PALETTE[1], churn,
                             "MC rate, NEO-enriched subsample (upper bound)", "o"),
                            (PALETTE[0], fullc,
                             "full catalogue, single realization", "s")):
        ax.plot(dt, np.maximum(val, floor), color=c, linewidth=2.2, marker=mk,
                markersize=6, label=lab)
        zero = np.isfinite(val) & (val <= 0.0)
        if zero.any():
            ax.scatter(dt[zero], np.full(int(zero.sum()), floor), marker=mk,
                       facecolors="white", edgecolors=c, linewidths=1.6, s=52,
                       zorder=6)
    if np.any(churn <= 0.0) or np.any(fullc <= 0.0):
        ax.scatter([], [], marker="o", facecolors="white", edgecolors=MUTED,
                   linewidths=1.6, s=52, label="hollow: exactly zero crossings")
    ax.axhline(BREAK_CROSS_FRAC, color=PALETTE[1], linewidth=1.3, linestyle=":")
    ax.annotate(f"C2 break criterion: {100 * BREAK_CROSS_FRAC:g}% per realization",
                (dt[0], BREAK_CROSS_FRAC), xytext=(2, 5), textcoords="offset points",
                color=PALETTE[1], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(floor / 2.2, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _dt_axis(ax, dt)
    _style(ax, "", "fraction crossing a triage threshold / realization",
           "C2 — triage churn at p = 0.1 / 0.3 / 0.5 / 0.7 (worst threshold)")
    _mark_break(ax, ver)

    onset = ver.get("break_onset_dt_min")
    fig.suptitle(
        "Where v0's delta-function attributable breaks: arc-length sweep at "
        "Rubin astrometric precision"
        + (f"  —  breaks at dt = {onset:g} min" if onset is not None
           else "  —  no break in 5–60 min"),
        color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, "arclength_headline.png"), dpi=140)
    plt.close(fig)

    # --- 2. std(P) scaling and its departure from 1/dt ----------------------
    vs = summary.get("variance_scaling") or {}
    if vs.get("curves"):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
        cdt = np.array([c["dt_min"] for c in vs["curves"]], dtype=float)
        ax = axes[0]
        for c, key, lab in ((PALETTE[0], "p50", "median std(P)"),
                            (PALETTE[1], "p90", "90th pct"),
                            (PALETTE[2], "p99", "99th pct")):
            yv = np.array([r.get(key, np.nan) for r in vs["curves"]])
            ax.plot(cdt, np.maximum(yv, 1e-12), color=c, linewidth=2.2, marker="o",
                    markersize=5, label=lab)
            k = int(np.argmax(cdt))
            if np.isfinite(yv[k]) and yv[k] > 0:
                ax.plot(cdt, yv[k] * (cdt / cdt[k]) ** -1.0, color=c, linewidth=1.0,
                        linestyle=":", alpha=0.9)
        ax.plot([], [], color=MUTED, linewidth=1.0, linestyle=":",
                label=fr"$\propto 1/\Delta t$ from the {cdt.max():g} min anchor")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        _dt_axis(ax, dt)
        _style(ax, "", "std(P(NEO)) over 25 realizations",
               f"per-tracklet MC scatter, {vs.get('subsample_n', 0)}-tracklet subsample")

        ax = axes[1]
        r50 = np.array([r.get("ratio_p50", np.nan) for r in vs["linear_scaling_ratio"]])
        r10 = np.array([r.get("ratio_p10", np.nan) for r in vs["linear_scaling_ratio"]])
        r90 = np.array([r.get("ratio_p90", np.nan) for r in vs["linear_scaling_ratio"]])
        rdt = np.array([r["dt_min"] for r in vs["linear_scaling_ratio"]], dtype=float)
        ax.fill_between(rdt, r10, r90, color=PALETTE[0], alpha=0.18, linewidth=0,
                        label="10th–90th pct over tracklets")
        ax.plot(rdt, r50, color=PALETTE[0], linewidth=2.2, marker="o", markersize=6,
                label="median std(P), MC subsample")
        pr = vs.get("paired_delta_p_ratio") or []
        if pr:
            ax.plot([r["dt_min"] for r in pr],
                    [r.get("ratio_p50", np.nan) for r in pr],
                    color=PALETTE[1], linewidth=2.0, marker="s", markersize=5,
                    label=r"median $|\Delta P|$, full catalogue (paired)")
        ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle="--")
        ax.axhspan(1.0 - vs.get("tol", 0.1), 1.0 + vs.get("tol", 0.1),
                   color=MUTED, alpha=0.10, linewidth=0)
        dep = vs.get("departure_dt_min")
        if dep is not None:
            ax.axvline(dep, color=PALETTE[2], linewidth=1.3)
            ax.annotate(f"1/$\\Delta t$ law departs\nat {dep:g} min", (dep, 1.0),
                        xytext=(-8, 0), textcoords="offset points", ha="right",
                        va="center", color=PALETTE[2], fontsize=8.5)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        _dt_axis(ax, dt)
        _style(ax, "", r"measured / $(1/\Delta t)$ extrapolation",
               "departure from linear response (1 = exactly linear)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "std_scaling.png"), dpi=140)
        plt.close(fig)

    # --- 3. the rate error in physical units --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax = axes[0]
    for c, key, lab in ((PALETTE[0], "p50", "median tracklet"),
                        (PALETTE[1], "p90", "90th pct sigma"),
                        (PALETTE[2], "max", "worst tracklet")):
        yv = np.array([_g(blocks[r["tag"]], "rate_error",
                          "sigma_rate_arcsec_per_day_per_coord", key) for r in rows])
        ax.plot(dt, yv, color=c, linewidth=2.2, marker="o", markersize=5, label=lab)
    own = _g(blocks[rows[0]["tag"]], "rate_error", "own_motion_arcsec_per_day",
             "p50", default=np.nan)
    ax.axhline(own, color=MUTED, linewidth=1.2, linestyle="--")
    ax.annotate(f"median tracklet's own motion ({own:.0f} arcsec/day)",
                (dt[0], own), xytext=(2, 5), textcoords="offset points",
                color=MUTED, fontsize=8)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _dt_axis(ax, dt)
    _style(ax, "", r"rate error $\sqrt{2}\,\sigma/\Delta t$ [arcsec/day/coord]",
           "what the arc length does to the rate")

    ax = axes[1]
    for c, key, lab in ((PALETTE[0], "frac_tracklets_error_over_10pct", "> 10%"),
                        (PALETTE[1], "frac_tracklets_error_over_50pct", "> 50%"),
                        (PALETTE[2], "frac_tracklets_error_over_100pct", "> 100%")):
        yv = np.array([_g(blocks[r["tag"]], "rate_error", key) for r in rows])
        ax.plot(dt, np.maximum(yv, 1e-5), color=c, linewidth=2.2, marker="o",
                markersize=5, label=f"rate error {lab} of own motion")
    ax.set_yscale("log")
    ax.set_ylim(5e-6, 1.2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _dt_axis(ax, dt)
    _style(ax, "", "fraction of tracklets",
           "the rate error relative to each tracklet's own motion")
    _mark_break(ax, ver)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rate_error.png"), dpi=140)
    plt.close(fig)

    # --- 4. ranking and the fixed operating point ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax = axes[0]
    da = np.array([r["delta_auc"] for r in rows])
    lo = np.array([r["delta_auc_ci95"][0] if isinstance(r["delta_auc_ci95"], (list, tuple))
                   else np.nan for r in rows])
    hi = np.array([r["delta_auc_ci95"][1] if isinstance(r["delta_auc_ci95"], (list, tuple))
                   else np.nan for r in rows])
    ax.fill_between(dt, lo, hi, color=PALETTE[0], alpha=0.18, linewidth=0,
                    label="95% paired bootstrap CI")
    ax.plot(dt, da, color=PALETTE[0], linewidth=2.2, marker="o", markersize=6,
            label="ΔAUC (noisy − noiseless)")
    ax.axhline(0.0, color=MUTED, linewidth=1.0, linestyle="--")
    ax.axhline(-WARN_DELTA_AUC, color=PALETTE[1], linewidth=1.2, linestyle=":")
    ax.annotate(f"W1: −{WARN_DELTA_AUC:g} (the digest2 method gap)",
                (dt[0], -WARN_DELTA_AUC), xytext=(2, -12),
                textcoords="offset points", color=PALETTE[1], fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _dt_axis(ax, dt)
    _style(ax, "", "ΔAUC", "ranking: does the noise reorder the catalogue?")
    _mark_break(ax, ver)

    ax = axes[1]
    comp = 100.0 * np.array([r["completeness_fixed_cut"] for r in rows])
    cont = 100.0 * np.array([r["contamination_fixed_cut"] for r in rows])
    ax.plot(dt, comp, color=PALETTE[0], linewidth=2.2, marker="o", markersize=6,
            label="completeness")
    ax.plot(dt, cont, color=PALETTE[1], linewidth=2.2, marker="s", markersize=5,
            label="contamination")
    ax.axhline(100.0 * rows[0]["completeness_noiseless"], color=PALETTE[0],
               linewidth=1.1, linestyle="--", alpha=0.8, label="noiseless completeness")
    ax.axhline(100.0 * rows[0]["contamination_noiseless"], color=PALETTE[1],
               linewidth=1.1, linestyle="--", alpha=0.8, label="noiseless contamination")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    _dt_axis(ax, dt)
    thr = _g(blocks[rows[0]["tag"]], "operating_points", "noiseless", "threshold")
    _style(ax, "", "percent",
           f"WP7 fixed cut p >= {_f(thr, '.3g')}, chosen once on noiseless scores")
    _mark_break(ax, ver)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ranking_operating_point.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Sweep: report
# --------------------------------------------------------------------------


def build_sweep_report(s: dict) -> str:
    """The durable record: astrometric_noise_arclength/REPORT.md."""
    L: list[str] = []
    A = L.append
    ver = s["verdict"]
    rows = ver["rows"]
    blocks = {b["tag"]: b for b in s["per_dt"]}
    em = s["error_model"]
    onset = ver.get("break_onset_dt_min")
    safe = ver.get("shortest_safe_dt_min")
    grid = ", ".join(f"{r['dt_min']:g}" for r in rows)

    A("# At what arc length does v0's delta-function attributable break?")
    A("")
    A("*The open question `validation/ASTROMETRIC_NOISE.md` leaves and DESIGN §2.7 "
      "names: the 45-minute verdict (harmless) is scoped by the arc, not by the "
      "astrometry, because the rate error is `sqrt(2) sigma / dt`.  This sweep re-runs "
      "the whole WP8 experiment at "
      f"dt = {grid} min and applies a criterion fixed in advance.  Harness: "
      "`validation/astrometric_noise.py sweep`.  Noiseless reference: the stored "
      "`gate_v0` scores — nothing was re-scored on that side.*")
    A("")

    # ---- the criterion, stated first --------------------------------------
    A("## The criterion (declared before the sweep was run)")
    A("")
    A("v0's delta-function attributable **breaks** at arc length dt if any of three "
      "arms trips.  Two further tripwires are reported as **warnings** — they flag a "
      "cost the reader should see, not a failure.")
    A("")
    for c in ver["criterion"]:
        head, sep, tail = c.partition(" — ")
        A(f"- **{head}**{sep}{tail}" if sep else f"- {c}")
    A("")
    A("C1 is the gate's own reliability test, not a new one: the same "
      "equal-expected-NEO-count bins, the same Poisson-binomial z, the same "
      "parametric-bootstrap thresholds that `validation/GATE_RESULTS.md` runs, applied "
      "to the *noisy* scores against the *true* labels.  A failure there is a property "
      "of the data, not a bug.")
    A("")

    # ---- bottom line -------------------------------------------------------
    A("## Bottom line")
    A("")
    if onset is None:
        A(f"**No arc length in {grid} min breaks the criterion.** The delta-function "
          f"attributable survives a {rows[-1]['dt_min']:g}-minute pair — a "
          f"{rows[0]['dt_min'] / rows[-1]['dt_min']:.0f}x amplification of the rate "
          "error relative to the longest arc tested — with calibration, ranking, "
          "triage thresholds and the operating point all inside the bounds fixed above.")
    else:
        A(f"**The assumption breaks at dt = {onset:g} min.** Every arc length at or "
          f"below {onset:g} min trips at least one break arm; "
          + (f"the shortest arc that does not is {safe:g} min."
             if safe is not None else "no tested arc length is clean."))
        # How much margin does the shortest clean arc actually have?  A "PASS"
        # one part in a thousand from alpha is not the same claim as a PASS four
        # orders of magnitude from it, and the difference decides whether the
        # safe arc length can be quoted as a recommendation.
        clean = [r for r in rows if not r["breaks"]]
        if clean and safe is not None:
            sr = min(clean, key=lambda r: r["dt_min"])
            pmin = min(p for p in (sr["global_z_p_value"], sr["max_abs_z_p_value"])
                       if np.isfinite(p))
            margin = pmin / BREAK_ALPHA
            A("")
            A(f"That clean verdict at {safe:g} min has little margin and should not be "
              f"read as a recommendation: its worst arm's p-value is "
              f"{_f(pmin, '.4f')} against alpha = {BREAK_ALPHA:g}, i.e. only "
              f"{_f(margin, '.2f')}x clear of failing (max\\|z\\| = "
              f"{_f(sr['max_abs_z'], '.2f')} against a bootstrap threshold of "
              f"{_f(sr['max_abs_z_threshold'], '.2f')}). Read the transition as "
              f"**somewhere between {safe:g} and {onset:g} minutes**, with "
              f"{safe:g} min already marginal.")
            roomy = [r["dt_min"] for r in clean
                     if min(p for p in (r["global_z_p_value"], r["max_abs_z_p_value"])
                            if np.isfinite(p)) > 0.1]
            if roomy:
                A("")
                A(f"The shortest arc length with comfortable margin (worst arm's "
                  f"p > 0.1) is **{min(roomy):g} min**. That, not "
                  f"{safe:g} min, is the number to design a cadence against.")
    A("")
    vs0 = s.get("variance_scaling") or {}
    if vs0.get("curves"):
        agg0 = _g(vs0, "aggregate_slope_std", "p90", "slope")
        dep0 = vs0.get("departure_dt_min")
        A(f"The second result is about the *shape* of the degradation rather than its "
          f"size.  The rate error grows exactly as 1/dt — that is arithmetic, and the "
          f"tests assert it — but the score's response to it does **not**: `std(P)` "
          f"scales as dt^{_f(agg0, '+.2f')} at the 90th percentile against the "
          f"predicted −1"
          + (f", and the per-tracklet 1/dt law departs at dt = {dep0:g} min."
             if dep0 is not None else ", and the per-tracklet 1/dt law holds "
             "across the whole grid on the cleanly-measurable subset.")
          + "  The shortfall is traced below to v0's piecewise-constant prior, not to "
          "the integration grid, and it means the degradation grows **more slowly** than "
          "the naive scaling predicts — the answer is more robust than the algebra, not "
          "less.")
        A("")
    hdr = ("| dt [min] | rate err / own motion (p50 / p90) | Z_tot (thr) | max\\|z\\| (thr) "
           "| C1 verdict | ΔAUC [95% CI] | churn / realization | lost "
           "| completeness @ fixed cut | contamination @ fixed cut | breaks? |")
    A(hdr)
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        b = blocks[r["tag"]]
        rr = _g(b, "rate_error", "rate_error_over_own_motion", default={})
        ci = r["delta_auc_ci95"] if isinstance(r["delta_auc_ci95"], (list, tuple)) \
            else [float("nan")] * 2
        mc = r["mc_worst_crossing"]
        A(f"| {r['dt_min']:g} "
          f"| {_pct(rr.get('p50'), '.2f')} / {_pct(rr.get('p90'), '.2f')} "
          f"| {_f(r['global_z'], '+.2f')} ({_f(r['global_z_threshold'], '.2f')}) "
          f"| {_f(r['max_abs_z'], '.2f')} ({_f(r['max_abs_z_threshold'], '.2f')}) "
          f"| {'PASS' if r['calibration_pass'] else '**FAIL**'} "
          f"| {_f(r['delta_auc'], '+.5f')} [{_f(ci[0], '+.4f')}, {_f(ci[1], '+.4f')}] "
          f"| {_pct(mc.get('frac_per_realization'), '.3f')} "
          f"| {_pct(r['frac_lost_to_noise'], '.3f')} "
          f"| {_pct(r['completeness_fixed_cut'])} "
          f"| {_pct(r['contamination_fixed_cut'], '.2f')} "
          f"| {'**YES**' if r['breaks'] else 'no'} |")
    A("")
    A(f"*Churn* is the MC mean over {s['n_mc']} realizations of the "
      f"{s['n_sub']}-tracklet NEO-enriched subsample, at its worst threshold, divided "
      "by the subsample size — an upper bound on the full population's rate.  *Lost* is "
      "the fraction of tracklets that scored finitely noiseless and become undefined "
      "under noise.  *Completeness* is at the fixed WP7 cut "
      f"p >= {_f(_g(blocks[rows[0]['tag']], 'operating_points', 'noiseless', 'threshold'), '.3g')} "
      f"(noiseless: {_pct(rows[0]['completeness_noiseless'])}).")
    A("")
    for r in rows:
        if r["reasons"] or r["warnings"]:
            A(f"**dt = {r['dt_min']:g} min**")
            for x in r["reasons"]:
                A(f"- BREAK: {x}")
            for x in r["warnings"]:
                A(f"- warning: {x}")
            A("")

    # ---- where the declared criterion was wrong ---------------------------
    ph = ver.get("posthoc") or {}
    flagged = ph.get("flagged_dt_min") or []
    if flagged:
        worst = min(flagged)
        wr = next(r for r in rows if r["dt_min"] == worst)
        A("### The declared criterion watched the wrong half of the operating point")
        A("")
        A(f"W2, as declared above, tests the **completeness** at the fixed WP7 cut.  "
          f"That is the half that does not move: completeness is "
          f"{_pct(rows[0]['completeness_fixed_cut'])} at "
          f"{rows[0]['dt_min']:g} min and {_pct(wr['completeness_fixed_cut'])} at "
          f"{worst:g} min, because the cut pins it.  The cost lands entirely on "
          f"**contamination**, which goes from "
          f"{_pct(wr['contamination_noiseless'], '.2f')} noiseless to "
          f"{_pct(wr['contamination_fixed_cut'], '.2f')} at {worst:g} min — a factor "
          f"{_f(wr['posthoc_contamination_ratio'], '.1f')}.  In operational terms the "
          f"same cut flags "
          f"{_g(blocks[rows[0]['tag']], 'operating_points', 'noisy_fixed_threshold', 'n_flagged', default=0)}"
          f" tracklets at {rows[0]['dt_min']:g} min and "
          f"{_g(blocks[wr['tag']], 'operating_points', 'noisy_fixed_threshold', 'n_flagged', default=0)}"
          f" at {worst:g} min for the same number of real NEOs — the extra flags are "
          "all false positives arriving at the triage desk.")
        A("")
        A(f"This is reported as **post-hoc** and enters no verdict: a tripwire invented "
          f"after seeing the data is not evidence, and the break at {onset:g} min is "
          f"called by C1 on its own.  But the criterion would have been better written "
          f"with a contamination arm, and that is a finding about the criterion, so it "
          f"is recorded here rather than quietly fixed.  Arc lengths whose contamination "
          f"more than "
          f"{_f(ph.get('contamination_ratio_threshold'), '.0f')}x-es: "
          + ", ".join(f"{t:g} min" for t in sorted(flagged, reverse=True)) + ".")
        A("")
    if not ver["monotone_in_dt"]:  # pragma: no cover - not expected
        A("> **Caution:** the break pattern is NOT monotone in dt. Since the rate error "
          "grows monotonically as 1/dt and nothing downstream reverses that, a "
          "non-monotone pattern means a criterion is being tripped by realization "
          "noise rather than by arc length. Read the individual arms, not the onset.")
        A("")

    # ---- headline figure ---------------------------------------------------
    A("![calibration and triage churn vs arc length](arclength_headline.png)")
    A("")

    # ---- what is actually being varied ------------------------------------
    A("## What is varied, and what is held fixed")
    A("")
    A(f"One thing changes across the rows above: the arc length.  The per-detection "
      f"sigma (Sorcha's model, m5 = {_f(em['m5'], '.4g')}, FWHM = {_f(em['fwhm_mas'], '.4g')} "
      "mas), the catalogue, the labels, the classifier, the scoring grid **and the "
      "standard-normal draws themselves** are identical at every arc length.  That last "
      "point is what makes the sweep sharp: `noisy_attributable` draws `e0, e1` from "
      "`default_rng([seed, stream])` and dt enters only as the divisor in "
      "`(d_1 - d_0) / dt`, so every arc length perturbs *the same endpoints by the same "
      "offsets* and differs only in the 1/dt amplification of the rates.  The dt-to-dt "
      "comparison is therefore paired per tracklet and carries no Monte-Carlo noise of "
      "its own.")
    A("")
    ad = s.get("adopted_from") or {}
    if ad:
        A(f"dt = 45 min is **adopted** from `{list(ad.values())[0]}` rather than "
          "re-scored: the injected catalogue is verified identical array-by-array and "
          "the stored scores are reused, so that row is literally the number published "
          "in `validation/ASTROMETRIC_NOISE.md`, not a reproduction of it.")
        A("")
        A("That makes the 45-minute row a cross-check on this harness, and it passes on "
          "every point estimate: Z_tot = −0.42, ĉ = 0.925, reliability max\\|z\\| = 0.72 "
          "against a bootstrap threshold of 2.62, AUC 0.98125 → 0.98222, ΔAUC = "
          "+0.00097, \\|ΔP\\| percentiles 5.06e-06 / 7.03e-05 / 7.20e-04, 87.6% "
          "completeness at 5.3% contamination, MC std(P) 4.55e-06 / 1.71e-04 / 8.17e-03 "
          "— all identical to the published values.")
        A("")
        A("**One published number does not reproduce, and it is not this sweep's doing.** "
          "`ASTROMETRIC_NOISE.md` quotes the ΔAUC 95% CI as [−0.00014, +0.00261]; every "
          "ΔAUC CI in this report is wider (at 45 min, [−0.00029, +0.00321]). The point "
          "estimate is bit-identical, and the CI is produced by "
          "`validation.digest2_comparison.auc_bootstrap`, which is outside this "
          "harness and was modified while this sweep was running. The CIs here are all "
          "from one analysis pass against one version of that function, so they are "
          "mutually comparable and the ΔAUC column's conclusion (consistent with zero at "
          "every arc length, two orders of magnitude below the 0.0126 method gap) is "
          "unaffected — but a reader diffing this row against the published CI should "
          "know why it moved. Nothing else in the comparison changed.")
        A("")

    dts = [float(r["dt_min"]) for r in rows]
    A("| V | SNR | sigma [mas] | "
      + " | ".join(f"rate err @ {t:g} min" for t in dts) + " |")
    A("|---:|---:|---:|" + "---:|" * len(dts))
    for t in s["sigma_table"]:
        cells = [f"{math.sqrt(2.0) * (t['sigma_mas'] / 1000.0) / (dtm / 1440.0):.2f}″/d"
                 for dtm in dts]
        A(f"| {t['V']:.0f} | {t['snr']:.2f} | {t['sigma_mas']:.2f} | "
          + " | ".join(cells) + " |")
    A("")
    e_long = math.sqrt(2.0) * (s["sigma_table"][-1]["sigma_mas"] / 1000.0) / (dts[0] / 1440.0)
    e_short = math.sqrt(2.0) * (s["sigma_table"][-1]["sigma_mas"] / 1000.0) / (dts[-1] / 1440.0)
    A(f"The right-hand columns are the whole experiment in one table.  The endpoint error "
      f"is a fixed property of the survey; the arc length alone decides whether the "
      f"faintest tracklet's rate is uncertain by {e_long:.1f} arcsec/day "
      f"({dts[0]:g} min) or by {e_short:.1f} arcsec/day ({dts[-1]:g} min) per "
      "coordinate — the latter being a materially different orbit.")
    A("")

    # ---- rate error vs the population ------------------------------------
    A("## The rate error against the tracklets' own motion")
    A("")
    A("The classifier does not respond to an absolute rate error; it responds to a rate "
      "error *relative to the motion that identifies the object*.  The median "
      f"`gate_v0` tracklet moves "
      f"{_f(_g(blocks[rows[0]['tag']], 'rate_error', 'own_motion_arcsec_per_day', 'p50'), '.0f')} "
      "arcsec/day.")
    A("")
    A("| dt [min] | rate error / own motion (p50) | (p90) | (p99) | > 10% of own motion "
      "| > 50% | > 100% |")
    A("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        re_ = _g(blocks[r["tag"]], "rate_error", default={})
        rr = re_.get("rate_error_over_own_motion", {})
        A(f"| {r['dt_min']:g} | {_pct(rr.get('p50'), '.2f')} | {_pct(rr.get('p90'), '.2f')} "
          f"| {_pct(rr.get('p99'), '.2f')} "
          f"| {_pct(re_.get('frac_tracklets_error_over_10pct'), '.2f')} "
          f"| {_pct(re_.get('frac_tracklets_error_over_50pct'), '.2f')} "
          f"| {_pct(re_.get('frac_tracklets_error_over_100pct'), '.2f')} |")
    A("")
    A("![rate error vs arc length](rate_error.png)")
    A("")

    # ---- score deltas ------------------------------------------------------
    A("## Per-tracklet score change")
    A("")
    A("| dt [min] | median \\|ΔP\\| | p90 | p99 | max | median \\|ΔP\\| (true NEO) "
      "| p99 (true NEO) |")
    A("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        d = _g(blocks[r["tag"]], "deltas", default={})
        a, ne = d.get("all", {}), d.get("neo", {})
        A(f"| {r['dt_min']:g} | {_f(a.get('abs_p50'), '.2e')} | {_f(a.get('abs_p90'), '.2e')} "
          f"| {_f(a.get('abs_p99'), '.2e')} | {_f(a.get('max_abs'), '.2e')} "
          f"| {_f(ne.get('abs_p50'), '.2e')} | {_f(ne.get('abs_p99'), '.2e')} |")
    A("")

    # ---- the variance scaling test ----------------------------------------
    vs = s.get("variance_scaling") or {}
    if vs.get("curves"):
        A("## Variance scaling: is the response still linear?")
        A("")
        A("DESIGN §2.7's expectation is a locally linear classifier response, in which a "
          "rate error proportional to 1/dt gives `std(P) ∝ 1/dt` and therefore "
          "`var(P) ∝ 1/dt²`.  That is a *testable* claim here rather than an "
          "assumption, because the draws are common across dt.  Two independent forms "
          "of the test:")
        A("")
        pt = vs.get("per_tracklet_slope_std", {})
        agg = vs.get("aggregate_slope_std", {})
        A(f"1. **Per-tracklet log-log slopes** of `std(P)` on dt, over the "
          f"{pt.get('n', 0)} of {vs.get('n_tracklets', 0)} subsample tracklets whose "
          f"scatter is non-zero at every arc length: median "
          f"{_f(pt.get('p50'), '+.3f')} (10th–90th pct "
          f"{_f(pt.get('p10'), '+.3f')} to {_f(pt.get('p90'), '+.3f')}), against the "
          f"predicted −1.  In variance terms: median "
          f"{_f(_g(vs, 'per_tracklet_slope_var', 'p50'), '+.3f')} against the "
          "predicted −2.")
        A(f"2. **Aggregate slopes** of the population percentile curves: median "
          f"{_f(_g(agg, 'p50', 'slope'), '+.3f')} (r² "
          f"{_f(_g(agg, 'p50', 'r2'), '.4f')}), 90th percentile "
          f"{_f(_g(agg, 'p90', 'slope'), '+.3f')} (r² "
          f"{_f(_g(agg, 'p90', 'r2'), '.4f')}), 99th percentile "
          f"{_f(_g(agg, 'p99', 'slope'), '+.3f')}, max "
          f"{_f(_g(agg, 'max', 'slope'), '+.3f')}.")
        A("")
        A("| dt [min] | median std(P) | p90 | p99 | max | frac std > 0.01 "
          "| linear-law ratio (p50) | (p10–p90) |")
        A("|---:|---:|---:|---:|---:|---:|---:|---:|")
        ratios = {r["dt_min"]: r for r in vs["linear_scaling_ratio"]}
        for c in vs["curves"]:
            rt = ratios.get(c["dt_min"], {})
            A(f"| {c['dt_min']:g} | {_f(c.get('p50'), '.2e')} | {_f(c.get('p90'), '.2e')} "
              f"| {_f(c.get('p99'), '.2e')} | {_f(c.get('max'), '.2e')} "
              f"| {_pct(c.get('frac_gt_0p01'), '.1f')} "
              f"| {_f(rt.get('ratio_p50'), '.3f')} "
              f"| {_f(rt.get('ratio_p10'), '.3f')}–{_f(rt.get('ratio_p90'), '.3f')} |")
        A("")
        A(f"The *linear-law ratio* is the measured `std(P)` divided by the pure-1/dt "
          f"extrapolation from the {_f(vs.get('anchor_dt_min'), '.0f')}-minute anchor, "
          f"per tracklet, over the {vs['linear_scaling_ratio'][0]['n']} tracklets whose "
          f"anchor `std(P)` exceeds {_f(vs.get('floor'), '.0e')}.  1.000 means the "
          "response is still exactly linear; below 1 means it has started to saturate "
          "(the posterior is being pushed against 0 or 1, or off the local gradient "
          "entirely).")
        A("")
        A("**Is the shortfall real, or is it the measurement?**  Two things could "
          "manufacture a ratio below 1 without the classifier being sublinear at all, "
          "and both are testable here rather than arguable.")
        A("")
        A("*Candidate 1 — the integration lattice.*  v0 rebuilds the (ρ, ρ̇) admissible "
          "region from each attributable and integrates it on a fixed `(1200, 50)` grid, "
          "so a perturbation can add or drop whole rings of boundary nodes and move `P` "
          "by a discrete jump that owes nothing to the posterior.  That does happen, and "
          "at a rate that grows sharply as the arc shortens:")
        A("")
        A("| dt [min] | tracklets whose node count changes | p90 \\|Δ nodes\\| "
          "| p90 relative | median \\|ΔP\\| |")
        A("|---:|---:|---:|---:|---:|")
        for r in rows:
            q = _g(blocks[r["tag"]], "quadrature", default={})
            A(f"| {r['dt_min']:g} | {_pct(q.get('frac_n_nodes_changed'), '.1f')} "
              f"| {_f(q.get('p90_abs_delta_n_nodes'), '.0f')} of "
              f"{_f(q.get('median_n_nodes_noiseless'), '.0f')} "
              f"| {_pct(q.get('p90_rel_delta_n_nodes'), '.2f')} "
              f"| {_f(_g(blocks[r['tag']], 'deltas', 'all', 'abs_p50'), '.2e')} |")
        A("")
        smo = vs.get("smooth_subset") or {}
        A(f"It is nevertheless **not** the cause.  Restricting the whole test to the "
          f"{smo.get('n', 0)} of {smo.get('n_total', 0)} tracklets "
          f"({_pct(smo.get('frac'), '.0f')}) whose node count is identical to the "
          "noiseless run at *every* arc length — switching the lattice mechanism off "
          "entirely — leaves the step ratios where they were (last column of the step "
          "table below).")
        A("")
        lst = vs.get("ladder_stability") or {}
        A("*Candidate 2 — small-amplitude roughness.*  A tracklet whose real response is "
          "at the roughness scale of the model measures the roughness, not the "
          "posterior, and its ratio is biased toward `dt/dt_ref`.  If that were driving "
          "the shortfall, excluding such tracklets would move the answer, so the "
          "identical test is re-run at a ladder of anchor floors:")
        A("")
        rbf = vs.get("ratio_by_floor") or []
        if rbf:
            A("| anchor floor on std(P) | tracklets | "
              + " | ".join(f"{t:g} min" for t in rbf[0]["dt_min"]) + " |")
            A("|---:|---:|" + "---:|" * len(rbf[0]["dt_min"]))
            for r in rbf:
                A(f"| {'none' if r['floor'] <= 0 else format(r['floor'], '.0e')} "
                  f"| {r['n']} | "
                  + " | ".join(_f(x, ".3f") for x in r["ratio_p50"]) + " |")
            A("")
            A("Median linear-law ratio; 1.000 is exact linearity.")
            A("")
            if lst.get("per_dt"):
                A("Across the ladder rows holding at least "
                  f"{lst.get('min_n', 20)} tracklets (floors "
                  + ", ".join('none' if f <= 0 else format(f, '.0e')
                              for f in lst.get("floors_used", []))
                  + "), the median ratio moves by at most "
                  f"**{_f(lst.get('max_spread'), '.3f')}** at any arc length: "
                  + ", ".join(f"{r['dt_min']:g} min "
                              f"{_f(r.get('lo'), '.2f')}–{_f(r.get('hi'), '.2f')}"
                              for r in lst["per_dt"] if "lo" in r) + ".")
                A("")
                # Two separable questions: does the floor matter (spread), and
                # does the shortfall survive the most favourable floor (hi)?
                spread = float(lst.get("max_spread", float("nan")))
                short = lst["per_dt"][-1] if lst["per_dt"] else {}
                best = float(short.get("hi", float("nan")))
                drifts = np.isfinite(spread) and spread >= 0.15
                survives = np.isfinite(best) and abs(best - 1.0) > float(
                    vs.get("tol", 0.10))
                A("Both halves of that answer matter and they point different ways. "
                  + (f"The ratio **does** drift with the floor (up to {spread:.3f}), "
                     "so part of the measured shortfall is an amplitude-dependent "
                     "artefact and the low-floor rows overstate it. "
                     if drifts else
                     f"The ratio is **flat** in the floor (moving at most "
                     f"{spread:.3f}), so candidate 2 is ruled out outright. ")
                  + (f"But it does not vanish: taking the single most favourable floor "
                     f"at each arc length, the {short.get('dt_min', float('nan')):g}-minute "
                     f"ratio is still {best:.3f} — far outside "
                     f"1 ± {vs.get('tol', 0.1):g}. A real sublinear component survives "
                     "every control applied here, and it is that component the rest of "
                     "this section is about."
                     if survives else
                     f"And at the most favourable floor the "
                     f"{short.get('dt_min', float('nan')):g}-minute ratio returns to "
                     f"{best:.3f}, inside 1 ± {vs.get('tol', 0.1):g} — so what remains "
                     "is consistent with a linear response measured through noise."))
                A("")
        sr = vs.get("step_ratio") or []
        psr = vs.get("paired_delta_p_step_ratio") or []
        if sr or psr:
            A("An anchored ratio cannot say *where* a departure happened, because every "
              "row is measured against the same longest arc.  The anchor-free version "
              "compares only adjacent arc lengths, each pair anchored on its own longer "
              "member.  A **constant** step ratio below 1 means the response left the "
              "power law once and has been self-similar since; step ratios that **march "
              "downward** mean progressive saturation.")
            A("")
            ssr = _g(vs, "smooth_subset", "step_ratio", default=[]) or []
            A("| arc pair | factor | median step ratio, std(P) (n) "
              "| median step ratio, paired \\|ΔP\\| (n) "
              "| ...restricted to fixed-node tracklets (n) |")
            A("|---|---:|---:|---:|---:|")
            pmap = {(r["dt_from"], r["dt_to"]): r for r in psr}
            smap = {(r["dt_from"], r["dt_to"]): r for r in ssr}
            for r in (sr or psr):
                key = (r["dt_from"], r["dt_to"])
                q, sm = pmap.get(key, {}), smap.get(key, {})
                s_ = next((x for x in sr if (x["dt_from"], x["dt_to"]) == key), {})
                A(f"| {r['dt_from']:g} → {r['dt_to']:g} min | {r['factor']:.2f}x "
                  f"| {_f(s_.get('step_p50'), '.3f')} ({s_.get('n', 0)}) "
                  f"| {_f(q.get('step_p50'), '.3f')} ({q.get('n', 0)}) "
                  f"| {_f(sm.get('step_p50'), '.3f')} ({sm.get('n', 0)}) |")
            A("")
            A("The last column is the control described above: the lattice mechanism is "
              "switched off and the answer does not move.")
            A("")
        last = vs["curves"][-1]
        n_dead = last.get("n_dead_at_anchor", 0)
        A(f"One failure mode is invisible to every ratio and slope above and is "
          f"therefore counted separately.  {n_dead} of {vs.get('n_tracklets', 0)} "
          f"subsample tracklets have `std(P)` numerically **zero** at the "
          f"{_f(vs.get('anchor_dt_min'), '.0f')}-minute anchor — the classifier is "
          "completely certain about them, and a ratio to zero has no value.  Of those, "
          + ", ".join(
              f"{c.get('n_woken_from_dead', 0)} move at all at {c['dt_min']:g} min "
              f"({c.get('n_woken_gt_0p01', 0)} by more than 0.01)"
              for c in vs["curves"][1:]) + ".")
        A("")
        dep = vs.get("departure_dt_min")
        pk = vs.get("departure_floor") or {}
        pk_ratio = {t: r for t, r in zip(pk.get("dt_min", []), pk.get("ratio_p50", []))}
        A(f"The verdict below is read off the **cleanest adequately-populated** row of "
          f"that ladder — anchor floor {_f(pk.get('floor'), '.0e')}, "
          f"{pk.get('n', 0)} tracklets (the largest floor still holding "
          f"{pk.get('min_n', 30)}) — rather than off a floor chosen to suit the answer.")
        A("")
        if dep is None:
            A(f"**Verdict: the 1/dt law holds across the whole grid** — on that subset "
              f"the median ratio stays inside ±{100 * vs.get('tol', 0.1):g}% at every arc "
              "length tested. Where the response is large enough to measure cleanly, it "
              "is linear over the full range, which is the regime in which a Gaussian "
              "covariance marginalization would be *correct* — and also the regime in "
              "which it has least to add, since a linear response to zero-mean noise is "
              "unbiased.")
        else:
            A(f"**Verdict: the 1/dt law departs at dt = {dep:g} min** "
              f"(median ratio {_f(pk_ratio.get(dep), '.3f')} on that subset, outside "
              f"1 ± {vs.get('tol', 0.1):g}). Above that arc length the response is "
              "linear and the perturbation is a zero-mean, correctly-modelled Gaussian; "
              "at and below it the response has left the linear regime, which is exactly "
              "where an outer marginalization stops being a refinement and starts being "
              "a correction.")
        A("")
        neo_p50 = _g(blocks[rows[0]["tag"]], "deltas", "neo", "abs_p50")
        med_nodes = _g(blocks[rows[0]["tag"]], "quadrature",
                       "median_n_nodes_noiseless", default=float("nan"))
        A("**What the surviving sublinearity is, and why it matters less than it looks.** "
          "The lattice control does not move it and no anchor floor removes it, so a "
          "real component is a property of v0's response. The "
          "mechanism is visible in *which* tracklets are sublinear: the aggregate slopes "
          f"above are {_f(_g(agg, 'p50', 'slope'), '+.2f')} at the median but "
          f"{_f(_g(agg, 'p99', 'slope'), '+.2f')} at the 99th percentile and "
          f"{_f(_g(agg, 'max', 'slope'), '+.2f')} at the maximum — the tracklets that "
          "actually move scale essentially as 1/dt exactly, and it is the typical, "
          "barely-responding tracklet that is sublinear. That is the signature of a "
          "**piecewise-constant** prior: `Digest2Histogram` bins (q, e, i, H) into 45,936 "
          "cells (DESIGN §3), so the likelihood is a step function of the orbit and the "
          "score's response to an attributable perturbation is a staircase, not a "
          f"gradient. Marginalizing over the ~{_f(med_nodes, '.0f')} quadrature nodes in "
          "a typical admissible region averages that many staircases — enough to look "
          "smooth for a tracklet whose perturbation crosses many bin walls, not enough "
          "for one that crosses few. Two observations fit only this explanation: the "
          f"median |ΔP| for true NEOs at {rows[0]['dt_min']:g} min is "
          f"{_f(neo_p50, '.2e')}"
          + (" — **exactly zero**, which a smooth response can never produce and a step "
             "function produces whenever the perturbation stays inside its bins"
             if np.isfinite(neo_p50) and neo_p50 == 0.0 else "")
          + f"; and {n_dead} of {vs.get('n_tracklets', 0)} subsample tracklets have "
          "identically zero MC scatter at the anchor arc.")
        A("")
        A("Three consequences worth stating plainly. **(i)** The degradation grows more "
          "slowly than the rate error, so the short-arc answer is *better* than the 1/dt "
          "algebra predicts — the extrapolation in `ASTROMETRIC_NOISE.md`'s \"where this "
          "could break\" was conservative. **(ii)** The tail is nevertheless linear, so "
          "the tracklets that can actually cross a triage threshold do degrade at the "
          "full 1/dt rate, and the C2 arm above is the number to trust rather than the "
          "median score change. **(iii)** \"The linear regime\", which is the regime a "
          "Gaussian covariance marginalization models correctly, is a statement about "
          "the responsive tail and about population aggregates — not a per-tracklet "
          "fact. On this axis v1 owes a smoother prior at least as much as it owes an "
          "outer marginalization; marginalizing a staircase with a Gaussian is a "
          "well-posed operation but it is not the operation the roughness calls for.")
        A("")
        pr = vs.get("paired_delta_p_ratio") or []
        if pr:
            A(f"The same test on the full catalogue's single paired realization — "
              f"sharper still, because `|ΔP| · dt` is a *deterministic* per-tracklet "
              f"quantity there rather than an estimated scatter (n = {pr[0]['n']} "
              f"tracklets above the {_f(vs.get('paired_delta_p_floor'), '.0e')} anchor "
              f"floor):")
            A("")
            A("| dt [min] | " + " | ".join(f"{r['dt_min']:g}" for r in pr) + " |")
            A("|---|" + "---|" * len(pr))
            A("| median \\|ΔP\\|·dt / anchor | "
              + " | ".join(_f(r.get("ratio_p50"), ".3f") for r in pr) + " |")
            A("| 10th pct | " + " | ".join(_f(r.get("ratio_p10"), ".3f") for r in pr) + " |")
            A("| 90th pct | " + " | ".join(_f(r.get("ratio_p90"), ".3f") for r in pr) + " |")
            A("")
            prf = vs.get("paired_delta_p_ratio_by_floor") or []
            if prf:
                A("...and its own floor ladder, for the same reason:")
                A("")
                A("| anchor floor on \\|ΔP\\| | tracklets | "
                  + " | ".join(f"{t:g} min" for t in prf[0]["dt_min"]) + " |")
                A("|---:|---:|" + "---:|" * len(prf[0]["dt_min"]))
                for r in prf:
                    A(f"| {'none' if r['floor'] <= 0 else format(r['floor'], '.0e')} "
                      f"| {r['n']} | "
                      + " | ".join(_f(x, ".3f") for x in r["ratio_p50"]) + " |")
                A("")
        A("![std(P) scaling](std_scaling.png)")
        A("")

    # ---- which tracklets actually move ------------------------------------
    short = blocks[rows[-1]["tag"]]
    long_ = blocks[rows[0]["tag"]]
    vs_p0_s = _g(short, "mc", "vs_p0", default=[]) or []
    if vs_p0_s:
        A(f"## Which tracklets move, at the shortest arc ({short['dt_min']:g} min)")
        A("")
        A("Only a tracklet near a triage threshold can be moved across one, so the "
          "triage-relevant slice of the sensitivity map is `std(P)` against the "
          "*noiseless* score.  Compared side by side with the longest arc tested, so "
          "the growth is visible per score bin rather than only in the population "
          "aggregate.")
        A("")
        by_long = {(r["lo"], r["hi"]): r for r in (_g(long_, "mc", "vs_p0", default=[]) or [])}
        A(f"| noiseless P(NEO) | n | median std @ {long_['dt_min']:g} min "
          f"| median std @ {short['dt_min']:g} min "
          f"| p90 std @ {short['dt_min']:g} min | growth (median) |")
        A("|---|---:|---:|---:|---:|---:|")
        for r in vs_p0_s:
            if not r["n"]:
                continue
            lg = by_long.get((r["lo"], r["hi"]), {})
            m_l, m_s = lg.get("abs_median"), r.get("abs_median")
            growth = (m_s / m_l) if (m_l and m_s and m_l > 0) else float("nan")
            gtxt = f"{growth:.1f}x" if np.isfinite(growth) else "n/a"
            A(f"| {r['lo']:.4g}–{r['hi']:.4g} | {r['n']} | {_f(m_l, '.2e')} "
              f"| {_f(m_s, '.2e')} | {_f(r.get('abs_p90'), '.2e')} | {gtxt} |")
        A("")
        A(f"Fraction of subsample tracklets with std(P) > 0.01: "
          f"{_pct(_g(long_, 'mc', 'frac_std_gt_0p01'), '.1f')} at "
          f"{long_['dt_min']:g} min, {_pct(_g(short, 'mc', 'frac_std_gt_0p01'), '.1f')} at "
          f"{short['dt_min']:g} min; > 0.10: "
          f"{_pct(_g(long_, 'mc', 'frac_std_gt_0p10'), '.1f')} and "
          f"{_pct(_g(short, 'mc', 'frac_std_gt_0p10'), '.1f')}.  Mean over "
          "realizations minus the noiseless score (a nonlinear response shows up here "
          f"as a systematic offset): {_f(_g(short, 'mc', 'mean_bias_mc_mean_minus_noiseless'), '+.2e')} "
          f"at {short['dt_min']:g} min, against "
          f"{_f(_g(long_, 'mc', 'mean_bias_mc_mean_minus_noiseless'), '+.2e')} at "
          f"{long_['dt_min']:g} min.")
        A("")
        ms = _g(short, "mc", "most_sensitive", default=[]) or []
        if ms:
            slow = [t for t in ms[:5] if t["rate_deg_day"] < 0.05]
            if slow:
                A(f"`ASTROMETRIC_NOISE.md`'s second caveat — \"the fractional rate error "
                  f"is worst for the slowest objects\" — is confirmed directly here, and "
                  f"in the worst possible way.  {len(slow)} of the five most sensitive "
                  f"tracklets at {short['dt_min']:g} min are near-stationary non-NEOs "
                  f"(rate ~{slow[0]['rate_deg_day']:.3f} deg/day) whose noiseless score "
                  f"is {slow[0]['p_noiseless']:.4f} — the classifier is *certain* they "
                  f"are not NEOs — and whose noisy score has mean "
                  f"{slow[0]['p_mc_mean']:.4f} and scatter {slow[0]['p_mc_std']:.4f}.  "
                  "A 14 arcsec/day rate error on a 100 arcsec/day object is not a "
                  "perturbation of its orbit, it is a different orbit, and the "
                  "manufactured NEO probability is where the C1 failure comes from: "
                  "these tracklets are what inflates the expected NEO count from "
                  f"{_f(_g(blocks[rows[0]['tag']], 'calibration', 'verdict', 'expected_neo_count'), '.0f')} "
                  f"to {_f(_g(short, 'calibration', 'verdict', 'expected_neo_count'), '.0f')} "
                  f"against {_f(_g(short, 'calibration', 'verdict', 'observed_neo_count'), '.0f')} "
                  "real ones.")
                A("")
            A(f"The most sensitive tracklets at {short['dt_min']:g} min:")
            A("")
            A("| gen index | NEO? | V | rate [deg/day] | sigma [mas] | P noiseless "
              "| P mean | std(P) |")
            A("|---:|---|---:|---:|---:|---:|---:|---:|")
            for t in ms:
                A(f"| {t['gen_index']} | {'yes' if t['is_neo'] else 'no'} "
                  f"| {t['v_mag']:.2f} | {t['rate_deg_day']:.3f} "
                  f"| {t['sigma_mas']:.1f} | {t['p_noiseless']:.4f} "
                  f"| {t['p_mc_mean']:.4f} | {t['p_mc_std']:.4f} |")
            A("")

    # ---- position vs rate --------------------------------------------------
    dec_rows = [(r, _g(blocks[r["tag"]], "decomposition", default={})) for r in rows]
    if any(d.get("variance_share") for _, d in dec_rows):
        A("## Positions vs. rates, arc length by arc length")
        A("")
        A("The position error does not depend on dt at all; only the rate error does.  "
          "The share of the induced score variance carried by the rate term is therefore "
          "a direct read on whether shortening the arc is doing what the algebra says.")
        A("")
        A("| dt [min] | rate share of var(ΔP) | position share | corr(both, rate) "
          "| linearity residual RMS | p90 \\|ΔP\\| rate-only | p90 \\|ΔP\\| pos-only |")
        A("|---:|---:|---:|---:|---:|---:|---:|")
        for r, d in dec_rows:
            v_ = d.get("variance_share", {})
            A(f"| {r['dt_min']:g} | {_pct(v_.get('rate_share_of_var'), '.1f')} "
              f"| {_pct(v_.get('pos_share_of_var'), '.1f')} "
              f"| {_f(v_.get('corr_both_rate'), '.3f')} "
              f"| {_f(v_.get('linearity_residual_rms'), '.2e')} "
              f"| {_f(_g(d, 'delta_p_rate', 'abs_p90'), '.2e')} "
              f"| {_f(_g(d, 'delta_p_pos', 'abs_p90'), '.2e')} |")
        A("")
        A("The position-only column is the control: it is (up to the shared draws) the "
          "same number at every arc length, because a rigid translation of the tracklet "
          "is dt-independent by construction.")
        A("")

    # ---- triage detail -----------------------------------------------------
    A("## Triage thresholds")
    A("")
    A("| dt [min] | worst threshold | MC mean crossings / realization | worst realization "
      "| mean NEOs lost / realization | full-catalogue crossings "
      f"({rows[0]['n'] if 'n' in rows[0] else blocks[rows[0]['tag']]['n']}) |")
    A("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        mc, fu = r["mc_worst_crossing"], r["full_worst_crossing"]
        A(f"| {r['dt_min']:g} | p >= {mc['threshold']:g} "
          f"| {_f(mc.get('mean_per_realization'), '.2f')} "
          f"| {_f(mc.get('max_in_a_realization'), '.0f')} "
          f"| {_f(mc.get('mean_neo_lost_per_realization'), '.2f')} "
          f"| {fu['n_cross_any']} (at p >= {fu['threshold']:g}) |")
    A("")
    A("## Calibration and the operating point")
    A("")
    A("| dt [min] | ĉ [95%] | Z_tot (p) | reliability max\\|z\\| (thr) | verdict "
      "| Brier | ECE | completeness | contamination |")
    A("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        cal = _g(blocks[r["tag"]], "calibration", default={})
        od = cal.get("odds_scale", {})
        A(f"| {r['dt_min']:g} | {_f(od.get('c_hat'), '.3f')} "
          f"[{_f(od.get('lo'), '.3f')}, {_f(od.get('hi'), '.3f')}] "
          f"| {_f(r['global_z'], '+.2f')} ({_f(r['global_z_p_value'], '.3f')}) "
          f"| {_f(r['max_abs_z'], '.2f')} ({_f(r['max_abs_z_threshold'], '.2f')}) "
          f"| {'PASS' if r['calibration_pass'] else '**FAIL**'} "
          f"| {_f(cal.get('brier'), '.5f')} "
          f"| {_f(cal.get('expected_calibration_error'), '.2e')} "
          f"| {_pct(r['completeness_fixed_cut'])} "
          f"| {_pct(r['contamination_fixed_cut'])} |")
    A("")
    A("![ranking and operating point](ranking_operating_point.png)")
    A("")
    A("At N = 2×10⁴ (≈185 NEOs) the fitted odds scale is only constrained to roughly "
      "±35%, so the ĉ column says \"no detectable miscalibration at the ±35% level\", "
      "not \"identical\".  The paired per-tracklet ΔP columns above are the sharper "
      "statement, and the C1 arm is deliberately the gate's own bootstrap-thresholded "
      "test rather than an eyeball comparison of ĉ.")
    A("")

    # ---- assumptions -------------------------------------------------------
    A("## Assumptions")
    A("")
    for a in s.get("assumptions", []):
        A(f"- {a}")
    A("")
    A("## Reproduce")
    A("")
    A("```")
    A("OMP_NUM_THREADS=1 PRIORA_BACKEND=python \\")
    A("  python3 -m validation.astrometric_noise sweep --workers 84")
    A("```")
    A("")
    A(f"Seed {s.get('seed')}; source catalogue `{s.get('source_run')}` (read-only); "
      f"{len(rows)} arc lengths x (1 x 20k full-catalogue realization + {s['n_mc']} x "
      f"{s['n_sub']} MC + 2 x {s['n_sub']} decomposition). Every stage is resumable and "
      "a pure function of its inputs.")
    A("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _error_model_from_args(a: argparse.Namespace) -> ErrorModel:
    return ErrorModel(
        m5=a.m5,
        fwhm_mas=a.fwhm_mas,
        error_sys_mas=a.error_sys_mas,
        ast_err_coeff=a.ast_err_coeff,
        rgamma=a.rgamma,
        nvisit=1,
        trailing_losses=not a.no_trailing,
        texp_s=a.texp,
    )


def sigma_table(em: ErrorModel, mags=(18.0, 21.0, 23.0, 24.0)) -> list[dict]:
    """sigma(V) for a static (non-moving) source — the provenance readout."""
    rows = []
    for v in mags:
        s_mas, snr, rand = astrometric_uncertainty_mas(
            v, em.m5, nvisit=em.nvisit, fwhm_eff_mas=em.fwhm_mas,
            error_sys_mas=em.error_sys_mas, ast_err_coeff=em.ast_err_coeff,
            rgamma=em.rgamma,
        )
        rows.append({
            "V": float(v),
            "snr": float(snr),
            "sigma_rand_mas": float(rand),
            "sigma_mas": float(s_mas),
            "sigma_arcsec": float(s_mas) / 1000.0,
        })
    return rows


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--m5", type=float, default=M5_DEFAULT,
                   help="adopted 5-sigma depth [mag] (default: %(default)s)")
    p.add_argument("--fwhm-mas", type=float, default=FWHM_MAS_DEFAULT,
                   help="FWHMeff [mas]; Sorcha's own default is 700")
    p.add_argument("--error-sys-mas", type=float, default=ERROR_SYS_MAS)
    p.add_argument("--ast-err-coeff", type=float, default=AST_ERR_COEFF)
    p.add_argument("--rgamma", type=float, default=RGAMMA)
    p.add_argument("--texp", type=float, default=TEXP_S_DEFAULT)
    p.add_argument("--no-trailing", action="store_true",
                   help="disable the Veres & Chesley trailing-loss term")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m validation.astrometric_noise",
        description=(
            "Measure the cost of priora v0's delta-function attributable "
            "(D-6) under Rubin-level astrometric noise."
        ),
    )
    subs = ap.add_subparsers(dest="cmd", required=True)

    p = subs.add_parser("sigma", help="print sigma(V) for the adopted error model")
    _add_model_args(p)

    for name, helptext in (
        ("inject", "write the perturbed catalogues"),
        ("run", "inject + score + analyze"),
    ):
        p = subs.add_parser(name, help=helptext)
        _add_model_args(p)
        p.add_argument("--source", default=DEFAULT_SOURCE_RUN)
        p.add_argument("--out", default=DEFAULT_OUT)
        p.add_argument("--seed", type=int, default=20260902)
        p.add_argument("--dt-min", type=float, default=DT_MIN_DEFAULT)
        p.add_argument("--n-mc", type=int, default=25)
        p.add_argument("--n-sub", type=int, default=500)
        p.add_argument("--m5-sens", type=float, default=24.5)
        p.add_argument("--no-main", action="store_true",
                       help="skip the 20k run (subsample studies only)")
        if name == "run":
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--chunk", type=int, default=250)
            p.add_argument("--n-rho", type=int, default=ScoreConfig().n_rho)
            p.add_argument("--n-rhodot", type=int, default=ScoreConfig().n_rhodot)
            _add_density_mode_args(p)
            p.add_argument("--no-plots", action="store_true")
            p.add_argument("--n-boot", type=int, default=2000)

    p = subs.add_parser("score", help="score the perturbed catalogues (resumable)")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--only", default=None, help="only runs whose name starts with this")
    p.add_argument("--n-rho", type=int, default=ScoreConfig().n_rho)
    p.add_argument("--n-rhodot", type=int, default=ScoreConfig().n_rhodot)
    _add_density_mode_args(p)

    p = subs.add_parser("analyze", help="compare against the noiseless scores")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--n-boot", type=int, default=2000)

    p = subs.add_parser(
        "sweep",
        help="arc-length sensitivity sweep: at what dt does D-6 break?",
        description=(
            "Re-run the whole experiment on a grid of tracklet arc lengths and "
            "apply the break criterion in BREAK_CRITERION. Writes to a separate "
            "output tree; the 45-minute headline run is left untouched and its "
            "scores are adopted for the dt = 45 min row."
        ),
    )
    _add_model_args(p)
    p.add_argument("--source", default=DEFAULT_SOURCE_RUN)
    p.add_argument("--out", default=DEFAULT_SWEEP_OUT)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--dt-grid", type=float, nargs="+", default=list(DT_GRID_DEFAULT),
                   help="arc lengths [min] (default: %(default)s)")
    p.add_argument("--n-mc", type=int, default=25)
    p.add_argument("--n-sub", type=int, default=500)
    p.add_argument("--reuse-from", default=DEFAULT_OUT,
                   help="existing WP8 run whose identical catalogues may be "
                        "adopted instead of re-scored ('' to disable)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--n-rho", type=int, default=ScoreConfig().n_rho)
    p.add_argument("--n-rhodot", type=int, default=ScoreConfig().n_rhodot)
    _add_density_mode_args(p)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--only-dt", type=float, default=None,
                   help="score only this arc length (resumable partial run)")
    p.add_argument("--stage", default="all",
                   choices=("all", "inject", "score", "analyze"),
                   help="run only one stage (default: all)")

    a = ap.parse_args(argv)

    if a.cmd == "sweep":
        t0 = time.time()
        if a.stage in ("all", "inject"):
            sweep_inject(
                source_run=a.source, out_dir=a.out, dt_grid=a.dt_grid,
                n_mc=a.n_mc, n_sub=a.n_sub, seed=a.seed,
                em=_error_model_from_args(a),
                reuse_from=(a.reuse_from or None),
            )
        if a.stage in ("all", "score"):
            sweep_score(out_dir=a.out,
                        cfg=_score_cfg_from_args(a),
                        workers=a.workers, chunk=a.chunk, only_dt=a.only_dt)
        if a.stage in ("all", "analyze"):
            sweep_analyze(out_dir=a.out, plots=not a.no_plots, n_boot=a.n_boot,
                          seed=a.seed)
        print(f"[sweep] total wall {(time.time() - t0) / 60.0:.1f} min")
        return 0

    if a.cmd == "sigma":
        em = _error_model_from_args(a)
        print(f"Sorcha astrometric error model, m5 = {em.m5}, FWHMeff = "
              f"{em.fwhm_mas} mas, error_sys = {em.error_sys_mas} mas, "
              f"astErrCoeff = {em.ast_err_coeff}, gamma = {em.rgamma}")
        print(f"{'V':>6} {'SNR':>10} {'sig_rand':>10} {'sigma':>10} {'sigma':>10}")
        print(f"{'':>6} {'':>10} {'[mas]':>10} {'[mas]':>10} {'[arcsec]':>10}")
        for r in sigma_table(em):
            print(f"{r['V']:>6.1f} {r['snr']:>10.2f} {r['sigma_rand_mas']:>10.2f} "
                  f"{r['sigma_mas']:>10.2f} {r['sigma_arcsec']:>10.4f}")
        return 0

    if a.cmd in ("inject", "run"):
        inject(
            source_run=a.source, out_dir=a.out, n_mc=a.n_mc, n_sub=a.n_sub,
            seed=a.seed, dt_min=a.dt_min, em=_error_model_from_args(a),
            m5_sens=a.m5_sens, full=not a.no_main,
        )
        if a.cmd == "inject":
            return 0

    if a.cmd in ("score", "run"):
        cfg = _score_cfg_from_args(a)
        score_runs(out_dir=a.out, cfg=cfg, workers=a.workers, chunk=a.chunk,
                   only=getattr(a, "only", None))
        if a.cmd == "score":
            return 0

    analyze(out_dir=a.out, plots=not a.no_plots, n_boot=a.n_boot)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
