"""WP6 — self-consistency validation harness (the v0 gate).

Implements DESIGN.md §4.1: generate a test population *from the classifier's own
prior*, score it, and check that the posterior is calibrated.  Because the test
population and the prior are the same distribution by construction, prior
mismatch is exactly zero and **any** deviation from the diagonal in the
reliability diagram is a bug (Jacobian, measure, admissible-region bounds, or
quadrature).

Three decoupled stages, each its own subcommand, so a long gate run can be
generated once, scored incrementally/resumably, and re-analysed for free::

    python3 -m validation.self_consistency generate --n 10000 --seed 42 \
        --out validation/output/gate
    python3 -m validation.self_consistency score --run validation/output/gate \
        --workers 8 [--resume]
    python3 -m validation.self_consistency analyze --run validation/output/gate --gate

``run`` does all three in sequence with one set of flags; ``rate-scan`` and
``ar-edge`` are the DESIGN §4.1 targeted slices.

The scoring grid defaults to ``classifier.N_RHO_DEFAULT`` x
``N_RHODOT_DEFAULT`` = (1200, 50), which is both cheaper and more accurate than
the (400, 200) this harness originally used: H(V, rho) turns every H-bin and
q-bin edge of a binned model into a rho-aligned discontinuity, so essentially
all quadrature error lives in the rho direction while the integrand is smooth
across rho_dot.  Measured at N = 1e4: HPD KS D 0.0499/0.0581 on (1200, 50)
versus 0.0679/0.0753 on (400, 200), at ~30% less CPU.

Generation (DESIGN §4.1)
------------------------
Per draw:

1. the *class* is drawn from the mixture, P(NEO) = total_count_NEO /
   (total_count_NEO + total_count_other) = 1.908% for the All variant — so the
   reliability diagram tests the calibration of the actual posterior, base rate
   included.  ``--h-extend-to`` changes this number, and by a lot: the D-19
   finite H extension multiplies the NEO component's total count by ~2.7 and
   the other component's by ~1.04, so at ``--h-extend-to 30`` the mixture
   P(NEO) is 4.754%.  That is not a bug to be normalized away — it is the
   extended model's own base rate, and the gate is testing the extended model
   as a population;
2. (elements, H) are drawn from that component's histogram (uniform angles, so
   the elements are already "at the observation epoch" and the epoch may be
   picked freely per tracklet);
3. the epoch is uniform in a configurable window (default: calendar year 2026);
4. the observer state is the geocentre (configurable via ``--obscode``);
5. ``observables_from_elements`` gives (ra, dec, ra_rate, dec_rate, V) plus the
   ground-truth (rho, rho_dot).

Rejections (all of them **joint** — a rejected draw is replaced by a completely
new draw, class included, never by a redraw within the same class; see
"Why rejection must be joint" below):

* **unbound draws.**  The model's top eccentricity bin is [0.9, 1.1), so
  ``sample()`` yields formally unbound objects: ~1.9% of All/NEO draws versus
  ~5e-6 of "other" (DESIGN §4.1, ``Digest2Histogram.sample`` docstring).  They
  are discarded (kept: ``a > 0`` and ``e < 1``) and the discarded fraction is
  reported per component.  Calibration survives because the classifier's
  admissible region admits only bound orbits (D-8), so prior and test
  population are both the bound-restricted model.
* **the observability cut**, ``v_min <= V < v_max`` (default 14 <= V < 24: the
  faint limit is the Rubin-like detection limit, the bright limit stands in for
  saturation).  DESIGN §4.1: selection applied on the *observables* does not
  break (A, V)-conditional calibration, because eps(A, V) multiplies both
  Lambda_C and cancels in the ratio.
* **pathological geometries**: non-finite observables, or rho below
  ``--rho-floor`` (object essentially at the observer, where the rates diverge).

Why rejection must be joint
---------------------------
The unbound fraction differs by five orders of magnitude between the two
components, and the observability cut has a strongly class-dependent acceptance
(~7% NEO vs ~17% other).  Redrawing *within* the rejected draw's class would
therefore reproduce the raw ``total_count`` mixture instead of the mixture the
classifier's likelihood ratio actually describes, and would show up as a real
calibration offset.  Discarding the whole draw and starting over from the class
choice is plain rejection sampling on the joint (class, E, H, epoch) and yields
exactly the bound-restricted, selection-conditioned mixture.

Scoring
-------
``classify`` at the configured resolution with the *same* (neo, other) pair and
the observer state recorded at generation time (``ClassificationResult`` does
not carry R_obs/V_obs, so the harness keeps them itself).  Per-node implied
(q, H) are read from ``ClassificationResult.node_q``/``node_H`` — the very
arrays ``classify`` integrated — rather than recomputed here, so the
diagnostics cannot drift from the likelihood they describe.

Per-tracklet diagnostics recorded alongside the score:

* **DESIGN §2.6 (a)** — posterior mass fraction at nodes with q > 1.3 AU and
  implied H > H_max^model.  H_max^model is taken from the components
  themselves (``classifier.h_support_end``), NOT assumed to be 25.5: it is
  25.5 for the D-4 pair and ``h_extend_to`` with the D-19 extension on.
  Because both components share it in v0 the density is identically zero above
  it and this flag is identically 0; it is recorded anyway because it becomes
  the operative flag the moment a component with a deeper H support is swapped
  in on one side only (DESIGN §3).
* **DESIGN §2.6 (b)** — the fraction of the admissible-region *measure*
  (sum of quadrature weights) at nodes with implied H above that same
  H_max^model, i.e. the part of the integration domain that both components
  zero out purely by model boundary.  This is the operative v0 flag, and the
  quantity WP7 needs in order to exclude or separately discuss tracklets where
  digest2 (which clamps H >= 24 into an open top bin) and ds42 (which
  truncates at its declared support end, D-4) consume different effective
  models.  With ``--h-extend-to 30`` the boundary moves to 30 and the
  measured fractions drop accordingly, so a number from this line is only
  comparable across runs in the same mode — which is why the summary prints
  the boundary next to it.
* the same measure fraction restricted to q > 1.3 AU (the Mars-crosser
  truncation corner of §2.6).

Collapse statistic (DESIGN §4.1)
--------------------------------
Posterior mass per node is ``(node_density_neo + node_density_other) * weight``
(``node_density_*`` is the integrand n_C |J| *without* the cell measure, per the
``ClassificationResult`` contract).  Two statistics:

* **concentration** — the mass fraction at nodes with rho within a factor of two
  of the true rho;
* **HPD rank** — the mass fraction at nodes whose mass *density* exceeds the
  density at the truth, with a mid-rank correction for ties.  Under exact
  calibration this is uniform on [0, 1]: the truth is a draw from the very
  conditional p(rho, rho_dot | A, V) whose (unnormalised) density the integrand
  is.  Two estimators of "the density at the truth" are recorded because they
  err in opposite directions at finite grid resolution:

  - ``hpd_rank`` uses the density at the *node nearest* the truth.  A coarse
    grid under-samples the peaks of a piecewise-constant density, so the truth
    tends to look lower-density than it is and the rank drifts towards 1.
  - ``hpd_rank_exact`` evaluates the integrand *exactly* at the truth (the
    generator knows the true elements and H).  That fully resolves the peaks
    while the comparison set stays coarse, so the rank drifts towards 0.

  Both converge to uniform under refinement, **both are gated** (tolerance
  0.075 at the (1200, 50) default grid), and the gap between them is reported
  as the grid-adequacy readout.  Measured at N = 1e4: 0.0499 / 0.0581 on
  (1200, 50) versus 0.0679 / 0.0753 on (400, 200) — slow convergence, exactly
  as the DESIGN §2.5 integrable-singularity caveat warns.
  **Consequence:** the criterion is on the KS *statistic*, not its p-value; at
  gate N any finite grid fails a p-value gate on discretisation bias alone.
  The tolerance is a quadrature floor and must be tightened with the grid.

A third, sharper quantity falls out for free: the integrand evaluated at the
truth must be strictly **positive** for every generated tracklet, because every
object was drawn from a non-empty histogram bin.  A zero there is not
miscalibration — it means the sampler and the density lookup disagree about the
model, i.e. a broken round-trip in the transforms.  It is gated
(``TRUTH_DENSITY_POSITIVE``).

Determinism and resumability
----------------------------
Generation batch *b* draws from ``default_rng([seed, b])``, so a given (seed,
batch) always produces the same draws regardless of N or of how many draws were
accepted before it: growing N extends the catalogue rather than changing it.
Scoring is a pure function of the generated catalogue (``classify`` uses no
randomness), so ``--resume`` reproduces a killed run exactly.  Each chunk file
carries a stamp of (catalogue bytes, scoring config); a chunk whose stamp,
schema, or integrity does not match is recomputed rather than spliced in, so a
resume across a changed grid or a regenerated catalogue cannot silently produce
a mixed results table.

Files written under ``--out``/``--run`` (all gitignored via validation/output/)::

    gen.npz          the generated catalogue (one row per accepted tracklet)
    gen_counts.json  generation configuration + per-class rejection counters
    chunks/chunk_%06d_%06d.npz   incremental scoring output (resume unit)
    scores.npz       all chunks concatenated (written when scoring completes)
    summary.json     machine-readable analysis result (PASS/FAIL per criterion)
    summary.txt      the printed summary block
    *.png            reliability / scores / ROC / collapse / diagnostics plots

Reliability criterion
---------------------
Rebuilt after review; the original (equal-tracklet-count bins, "every bin
within 2 sigma", plus a chi-square at a nominal 1%) was broken in both
directions.  Measured on real score vectors, its family-wise false-alarm rate
was 0.43 (ten chances at a 2-sigma excursion, with a per-bin z nowhere near
normal), while its power against a real odds distortion was 0.00 — nine of ten
equal-count bins held 0.03-1.3 expected NEOs and could neither pass nor fail.

The replacement has two bootstrap-thresholded arms:

* **binned** max|z| over **equal-EXPECTED-NEO-COUNT** bins (5 by default);
* **global** ``Z_tot = sum w(y-p) / sqrt(w_neo w_other sum p(1-p))`` — observed
  versus expected NEO count over the whole run, which is what actually detects
  a uniform odds bias (power ~0.8 against p' = p^1.3 where the binned arm gets
  ~0.0), reported at the *top* of the summary as the cheapest tripwire.

Both thresholds come from a parametric bootstrap: labels are re-simulated from
the generator's own null on the observed p vector, B = 1000 times, and the 99th
percentile is the FAIL line.  That is exact Type-I by construction for any
binning, base rate, or weighting.  The chi-square is kept as display only.
The fitted global odds scale ``c_hat`` (with a 95% interval) is reported as the
effect size behind ``Z_tot``.

Enrichment (NOT the gate mode, and NOT a calibration booster)
------------------------------------------------------------
``--neo-fraction P`` draws the class with probability P instead of the mixture
rate and records per-tracklet importance weights (p_mix/P for NEOs,
(1-p_mix)/(1-P) for the rest); every analysis statistic is weighted, so it
still estimates the calibration of the *mixture* posterior.

It is a **low-p debugging amplifier only**.  Measured here as the
per-tracklet-scored noncentrality for a c = 1.1 odds distortion confined to a
band of score space, enrichment at f = 0.5 buys 5.5x where NEOs are rare in the
mixture (p < 0.02) and **nothing at all** — ratio 0.99 — in the band where the
triage decision actually lives, p in [0.2, 0.8).  (The WP6 review measured that
middle band as 3.4x *worse* rather than neutral; the two measurements agree on
the conclusion and disagree on its size, and both are small-sample.)  Either
way it is not a calibration booster: it moves sensitivity to the tail where
NEOs are cheap, not to the region a threshold would be set in.

The bare flag therefore defaults to 0.1, not 0.5.  The gate must be run without
it, and a run that used it is marked non-gate-eligible in the summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np

# Headless by default: the harness runs on batch nodes.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from astropy.time import Time  # noqa: E402

from ds42.attributable import Attributable, observer_state  # noqa: E402
from ds42.classifier import (  # noqa: E402
    N_RHO_DEFAULT,
    N_RHODOT_DEFAULT,
    classify,
    h_support_end,
)
from ds42.constants import G_SLOPE_DEFAULT  # noqa: E402
from ds42.population import (  # noqa: E402
    H_EDGES,
    H_INTERPOLATIONS,
    Digest2Histogram,
)
from ds42.ranging import admissible_rhodot_interval  # noqa: E402
from ds42.transforms import (  # noqa: E402
    observables_from_elements,
    qei_from_elements,
    ranging_jacobian_det,
)

__all__ = [
    "GenConfig",
    "ScoreConfig",
    "generate",
    "save_generation",
    "load_generation",
    "score",
    "analyze",
    "reliability_bins",
    "reliability_verdict",
    "global_z",
    "fit_odds_scale",
    "bootstrap_reliability",
    "null_label_probs",
    "ar_edge_slice",
    "rate_scan",
    "auc_weighted",
    "ks_uniform",
    "chi2_sf",
    "main",
]

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: q [AU] boundary separating digest2's NEO class from the rest (D-2).
Q_NEO_BOUNDARY = 1.3

DEFAULT_MODEL = "data/digest2.model.csv"

#: Fields stored per scored tracklet (the scores.npz / chunk schema).
RECORD_FIELDS: tuple[str, ...] = (
    "index",
    "p_neo",
    "log_lr",
    "lambda_neo",
    "lambda_other",
    "n_nodes",
    "ar_measure",
    "ar_frac_h_above",
    "ar_frac_h_above_q_above",
    "mass_frac_q_above_h_above",
    "collapse_frac_rho2",
    "hpd_rank",
    "hpd_rank_exact",
    "truth_density",
    "truth_node_density",
    "max_node_density",
    "rho_map",
    "rhodot_map",
    "d_log10_rho_truth",
    "rhodot_outside_au_per_day",
    "rhodot_margin_frac",
    "truth_rho_in_range",
    "elapsed_s",
)

_INT_FIELDS = frozenset({"index", "n_nodes", "truth_rho_in_range"})


# --------------------------------------------------------------------------
# Small self-contained statistics (scipy is not a declared dependency)
# --------------------------------------------------------------------------


def _gammap_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) by series (x < a + 1)."""
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammaq_cf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) by continued fraction."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(x: float, dof: int) -> float:
    """Upper tail P(chi^2_dof > x). Pure-Python; matches scipy to ~1e-12."""
    x = float(x)
    dof = int(dof)
    if dof <= 0:
        raise ValueError("dof must be >= 1")
    if not np.isfinite(x):
        return 0.0 if x > 0 else 1.0
    if x <= 0.0:
        return 1.0
    a, y = 0.5 * dof, 0.5 * x
    if y < a + 1.0:
        return max(0.0, min(1.0, 1.0 - _gammap_series(a, y)))
    return max(0.0, min(1.0, _gammaq_cf(a, y)))


def normal_sf(z: float) -> float:
    """Upper tail of the standard normal."""
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def _kolmogorov_sf(lam: float) -> float:
    """Q_KS(lambda) = 2 sum_{j>=1} (-1)^(j-1) exp(-2 j^2 lambda^2)."""
    lam = float(lam)
    if lam <= 0.0:
        return 1.0
    if lam < 0.2:  # series is useless here; the tail is 1 to machine precision
        return 1.0
    total = 0.0
    for j in range(1, 200):
        term = 2.0 * (-1.0) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam)
        total += term
        if abs(term) < 1e-14:
            break
    return max(0.0, min(1.0, total))


def ks_uniform(u: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float, float]:
    """One-sample KS test of ``u`` against Uniform(0, 1).

    Returns (D, p_value, n_eff).  With weights, the ECDF is the weighted one and
    the asymptotic p-value uses the Kish effective sample size
    ``(sum w)^2 / sum w^2`` — an approximation, adequate for a diagnostic.
    """
    u = np.asarray(u, dtype=float)
    good = np.isfinite(u)
    u = u[good]
    if weights is None:
        w = np.ones_like(u)
    else:
        w = np.asarray(weights, dtype=float)[good]
    if u.size == 0 or w.sum() <= 0:
        return float("nan"), float("nan"), 0.0
    order = np.argsort(u, kind="mergesort")
    us = u[order]
    ws = w[order]
    cw = np.cumsum(ws) / ws.sum()
    below = np.concatenate(([0.0], cw[:-1]))
    d = float(max(np.max(cw - us), np.max(us - below)))
    n_eff = float(w.sum() ** 2 / np.sum(w * w))
    sn = math.sqrt(n_eff)
    p = _kolmogorov_sf((sn + 0.12 + 0.11 / sn) * d)
    return d, p, n_eff


def auc_weighted(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Weighted ROC AUC = P(score(pos) > score(neg)) + 0.5 P(tie).

    O(n log n): ties get the average rank, exactly as the Mann-Whitney U
    identity requires.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    w = np.ones_like(scores) if weights is None else np.asarray(weights, dtype=float)
    good = np.isfinite(scores)
    scores, labels, w = scores[good], labels[good], w[good]
    wp, wn = w[labels].sum(), w[~labels].sum()
    if wp <= 0 or wn <= 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s, lab, ww = scores[order], labels[order], w[order]
    # cumulative negative weight strictly below / at each tie group
    n_below = np.zeros(s.size)
    acc = 0.0
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        grp = slice(i, j + 1)
        neg_in_grp = ww[grp][~lab[grp]].sum()
        n_below[grp] = acc + 0.5 * neg_in_grp
        acc += neg_in_grp
        i = j + 1
    return float(np.sum(ww[lab] * n_below[lab]) / (wp * wn))


def roc_curve_weighted(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted ROC (fpr, tpr), thresholded at every distinct score."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    w = np.ones_like(scores) if weights is None else np.asarray(weights, dtype=float)
    good = np.isfinite(scores)
    scores, labels, w = scores[good], labels[good], w[good]
    if scores.size == 0 or not labels.any() or labels.all():
        # No finite scores, or only one class present: there is no ROC to draw.
        # `np.concatenate(([0.0], s[1:] != s[:-1]))` on an empty array raises,
        # so this must be caught here rather than falling through.
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-scores, kind="mergesort")
    s, lab, ww = scores[order], labels[order], w[order]
    tp = np.cumsum(np.where(lab, ww, 0.0))
    fp = np.cumsum(np.where(lab, 0.0, ww))
    # One vertex per *distinct threshold*: emitting one per sample would draw a
    # staircase through tie groups, so the trapezoidal area would disagree with
    # the Mann-Whitney AUC.
    last = np.concatenate((s[1:] != s[:-1], [True]))
    tpr = np.concatenate(([0.0], tp[last] / max(tp[-1], 1e-300)))
    fpr = np.concatenate(([0.0], fp[last] / max(fp[-1], 1e-300)))
    return fpr, tpr


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


@dataclass
class GenConfig:
    """Configuration of the test-population generator (DESIGN §4.1)."""

    n: int = 1000
    seed: int = 42
    model: str = DEFAULT_MODEL
    variant: str = "All"
    v_max: float = 24.0
    v_min: float = 14.0
    epoch_start: str = "2026-01-01T00:00:00"
    epoch_days: float = 365.0
    rho_floor: float = 0.01
    obscode: str | None = None
    G: float = G_SLOPE_DEFAULT
    batch: int = 4096
    neo_fraction: float | None = None  # None -> the true mixture (gate mode)
    #: Population H reconstruction, passed to ``neo_other_pair``.  In gate mode
    #: the generator and the classifier MUST use the same value — that is the
    #: whole premise of DESIGN §4.1 — so the flag sets both.
    h_interpolation: str = "none"
    #: Finite H extension [D-19], passed to ``neo_other_pair``.  Same
    #: generator/classifier discipline as ``h_interpolation``, and for the same
    #: reason: it changes the density the population is drawn from.  Unlike the
    #: D-14 clamp the extended pair IS gate-eligible — it is normalizable and
    #: samplable — so prior = generator = extended model and the gate tests the
    #: extended model as a legitimate population.
    h_extend_to: float | None = None


@dataclass
class GenCounters:
    """Per-class draw bookkeeping. ``accepted + every rejection == drawn``."""

    drawn: int = 0
    unbound: int = 0
    nonfinite: int = 0
    rho_below_floor: int = 0
    too_bright: int = 0
    too_faint: int = 0
    accepted: int = 0

    def rejected(self) -> int:
        return (
            self.unbound
            + self.nonfinite
            + self.rho_below_floor
            + self.too_bright
            + self.too_faint
        )

    def consistent(self) -> bool:
        return self.drawn == self.accepted + self.rejected()


@dataclass
class Generation:
    """A generated test population (one row per accepted tracklet)."""

    ra: np.ndarray
    dec: np.ndarray
    ra_rate: np.ndarray
    dec_rate: np.ndarray
    v_mag: np.ndarray
    epoch_jd_tdb: np.ndarray
    R_obs: np.ndarray  # (N, 3) heliocentric equatorial [AU]
    V_obs: np.ndarray  # (N, 3) heliocentric equatorial [AU/day]
    rho_true: np.ndarray
    rho_dot_true: np.ndarray
    elements: np.ndarray  # (N, 6) (a, e, i, Omega, omega, M) ecliptic
    H: np.ndarray
    is_neo: np.ndarray  # bool, the ground-truth class label
    weight: np.ndarray  # importance weight (all 1.0 in gate mode)
    config: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.ra.size)


def _mixture_neo_fraction(neo: Digest2Histogram, other: Digest2Histogram) -> float:
    tot = neo.total_count + other.total_count
    if tot <= 0:
        raise ValueError("both population components are empty")
    return float(neo.total_count / tot)


def generate(cfg: GenConfig, verbose: bool = True) -> Generation:
    """Draw ``cfg.n`` accepted tracklets from the mixture (DESIGN §4.1).

    Rejection sampling on the joint (class, elements, H, epoch): unbound draws,
    pathological geometries and observability-cut failures are discarded whole
    and replaced by a fresh draw *including a fresh class choice* (see the
    module docstring for why that is required).
    """
    if cfg.n <= 0:
        raise ValueError("n must be positive")
    if not (cfg.v_max > cfg.v_min):
        raise ValueError("v_max must exceed v_min")
    neo, other = Digest2Histogram.neo_other_pair(
        cfg.model, cfg.variant, h_interpolation=cfg.h_interpolation,
        h_extend_to=cfg.h_extend_to,
    )
    p_mix = _mixture_neo_fraction(neo, other)
    p_draw = p_mix if cfg.neo_fraction is None else float(cfg.neo_fraction)
    if not (0.0 < p_draw < 1.0):
        raise ValueError("neo-fraction must lie strictly between 0 and 1")
    w_neo = p_mix / p_draw
    w_other = (1.0 - p_mix) / (1.0 - p_draw)

    t_start = Time(cfg.epoch_start, scale="tdb")
    jd0 = float(t_start.tdb.jd)

    counters = {"NEO": GenCounters(), "other": GenCounters()}
    cols: dict[str, list] = {
        k: []
        for k in (
            "ra", "dec", "ra_rate", "dec_rate", "v_mag", "epoch_jd_tdb",
            "R_obs", "V_obs", "rho_true", "rho_dot_true", "elements", "H",
            "is_neo", "weight",
        )
    }
    n_have = 0
    batch_index = 0
    t0 = time.time()

    while n_have < cfg.n:
        rng = np.random.default_rng([int(cfg.seed), int(batch_index)])
        batch_index += 1
        m = int(cfg.batch)

        cls_neo = rng.random(m) < p_draw
        n_n = int(cls_neo.sum())
        el = np.empty((m, 6), dtype=float)
        hh = np.empty(m, dtype=float)
        if n_n:
            el_n, h_n = neo.sample(n_n, rng)
            el[cls_neo] = el_n
            hh[cls_neo] = h_n
        if m - n_n:
            el_o, h_o = other.sample(m - n_n, rng)
            el[~cls_neo] = el_o
            hh[~cls_neo] = h_o
        jd = jd0 + rng.random(m) * float(cfg.epoch_days)

        # One outcome code per batch row, so the counters can be tallied over
        # exactly the rows that were consumed (see the truncation below).
        # 0 accepted, 1 unbound, 2 non-finite observables, 3 rho below floor,
        # 4 saturated (V < v_min), 5 too faint (V >= v_max).
        outcome = np.full(m, 5, dtype=np.int8)
        with np.errstate(invalid="ignore"):
            bound = (el[:, 1] < 1.0) & (el[:, 0] > 0.0) & np.all(np.isfinite(el), axis=1)
        outcome[~bound] = 1
        idx = np.flatnonzero(bound)

        obs = None
        R = V = None
        if idx.size:
            epochs = Time(jd[idx], format="jd", scale="tdb")
            R, V = observer_state(epochs, obscode=cfg.obscode)
            R = np.atleast_2d(np.asarray(R, dtype=float))
            V = np.atleast_2d(np.asarray(V, dtype=float))
            with np.errstate(invalid="ignore", divide="ignore"):
                obs = observables_from_elements(el[idx], hh[idx], R, V, G=cfg.G)

            finite = np.ones(idx.size, dtype=bool)
            for key in ("ra", "dec", "ra_rate", "dec_rate", "v_mag", "rho", "rho_dot"):
                finite &= np.isfinite(obs[key])
            with np.errstate(invalid="ignore"):
                rho_ok = obs["rho"] > float(cfg.rho_floor)
                bright = obs["v_mag"] < float(cfg.v_min)
                faint = obs["v_mag"] >= float(cfg.v_max)

            # Priority order: exactly one reason is attributed per rejection.
            code = np.where(
                ~finite, 2,
                np.where(~rho_ok, 3, np.where(bright, 4, np.where(faint, 5, 0))),
            ).astype(np.int8)
            outcome[idx] = code

        # Truncation: stop at exactly cfg.n accepted tracklets.  Rows *after*
        # the last accepted row of this batch were drawn but never used, so
        # they must not enter the counters either — otherwise the reported
        # acceptance rates would be diluted by an arbitrary tail.
        acc_pos = np.flatnonzero(outcome == 0)
        room = cfg.n - n_have
        if acc_pos.size > room:
            n_used = int(acc_pos[room - 1]) + 1
            acc_pos = acc_pos[:room]
        else:
            n_used = m

        used_outcome = outcome[:n_used]
        used_neo = cls_neo[:n_used]
        for label, cmask in (("NEO", used_neo), ("other", ~used_neo)):
            c = counters[label]
            c.drawn += int(cmask.sum())
            c.accepted += int(np.sum((used_outcome == 0) & cmask))
            c.unbound += int(np.sum((used_outcome == 1) & cmask))
            c.nonfinite += int(np.sum((used_outcome == 2) & cmask))
            c.rho_below_floor += int(np.sum((used_outcome == 3) & cmask))
            c.too_bright += int(np.sum((used_outcome == 4) & cmask))
            c.too_faint += int(np.sum((used_outcome == 5) & cmask))

        if acc_pos.size == 0:
            continue

        # Map accepted batch rows back into the `idx` subset the observables
        # were computed for.
        pos_in_idx = np.full(m, -1, dtype=np.int64)
        pos_in_idx[idx] = np.arange(idx.size)
        take = pos_in_idx[acc_pos]
        keep_neo = cls_neo[acc_pos]

        cols["ra"].append(obs["ra"][take])
        cols["dec"].append(obs["dec"][take])
        cols["ra_rate"].append(obs["ra_rate"][take])
        cols["dec_rate"].append(obs["dec_rate"][take])
        cols["v_mag"].append(obs["v_mag"][take])
        cols["epoch_jd_tdb"].append(jd[acc_pos])
        cols["R_obs"].append(R[take])
        cols["V_obs"].append(V[take])
        cols["rho_true"].append(obs["rho"][take])
        cols["rho_dot_true"].append(obs["rho_dot"][take])
        cols["elements"].append(el[acc_pos])
        cols["H"].append(hh[acc_pos])
        cols["is_neo"].append(keep_neo)
        cols["weight"].append(np.where(keep_neo, w_neo, w_other))
        n_have += acc_pos.size

        if verbose:
            print(
                f"  [generate] batch {batch_index:4d}: "
                f"{n_have}/{cfg.n} accepted ({time.time() - t0:.1f} s)",
                flush=True,
            )

    counts_json = {
        "p_mixture_neo": p_mix,
        "p_draw_neo": p_draw,
        "weight_neo": w_neo,
        "weight_other": w_other,
        "total_count_neo": float(neo.total_count),
        "total_count_other": float(other.total_count),
        "batches": batch_index,
        "wall_s": time.time() - t0,
        "per_class": {k: asdict(v) for k, v in counters.items()},
    }
    for k, c in counters.items():
        d = counts_json["per_class"][k]
        d["rejected"] = c.rejected()
        d["counters_consistent"] = c.consistent()
        d["unbound_fraction"] = (c.unbound / c.drawn) if c.drawn else float("nan")
        bound_n = c.drawn - c.unbound
        d["acceptance_rate_after_bound_cut"] = (
            c.accepted / bound_n if bound_n > 0 else float("nan")
        )
        d["acceptance_rate_overall"] = (
            c.accepted / c.drawn if c.drawn > 0 else float("nan")
        )
        if not c.consistent():  # pragma: no cover - accounting invariant
            raise AssertionError(
                f"generation counters for {k!r} do not balance: drawn={c.drawn}, "
                f"accepted={c.accepted}, rejected={c.rejected()}"
            )

    gen = Generation(
        ra=np.concatenate(cols["ra"]),
        dec=np.concatenate(cols["dec"]),
        ra_rate=np.concatenate(cols["ra_rate"]),
        dec_rate=np.concatenate(cols["dec_rate"]),
        v_mag=np.concatenate(cols["v_mag"]),
        epoch_jd_tdb=np.concatenate(cols["epoch_jd_tdb"]),
        R_obs=np.concatenate(cols["R_obs"]),
        V_obs=np.concatenate(cols["V_obs"]),
        rho_true=np.concatenate(cols["rho_true"]),
        rho_dot_true=np.concatenate(cols["rho_dot_true"]),
        elements=np.concatenate(cols["elements"]),
        H=np.concatenate(cols["H"]),
        is_neo=np.concatenate(cols["is_neo"]),
        weight=np.concatenate(cols["weight"]),
        config=asdict(cfg),
        counters=counts_json,
    )
    return gen


def save_generation(gen: Generation, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "gen.npz")
    np.savez_compressed(
        path,
        ra=gen.ra, dec=gen.dec, ra_rate=gen.ra_rate, dec_rate=gen.dec_rate,
        v_mag=gen.v_mag, epoch_jd_tdb=gen.epoch_jd_tdb, R_obs=gen.R_obs,
        V_obs=gen.V_obs, rho_true=gen.rho_true, rho_dot_true=gen.rho_dot_true,
        elements=gen.elements, H=gen.H, is_neo=gen.is_neo, weight=gen.weight,
    )
    with open(os.path.join(out_dir, "gen_counts.json"), "w") as fh:
        json.dump({"config": gen.config, "counters": gen.counters}, fh, indent=2)
    return path


def load_generation(run_dir: str) -> Generation:
    with np.load(os.path.join(run_dir, "gen.npz")) as z:
        arrays = {k: z[k] for k in z.files}
    meta_path = os.path.join(run_dir, "gen_counts.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
    return Generation(
        config=meta.get("config", {}), counters=meta.get("counters", {}), **arrays
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass
class ScoreConfig:
    """Configuration of the scoring pass."""

    n_rho: int = N_RHO_DEFAULT
    n_rhodot: int = N_RHODOT_DEFAULT
    rho_min: float = 0.01
    rho_max: float = 100.0
    G: float = G_SLOPE_DEFAULT
    model: str = DEFAULT_MODEL
    variant: str = "All"
    #: Population H reconstruction ("none" | "pchip").  Part of the config that
    #: ``_run_stamp`` hashes, so switching it invalidates cached chunks — it
    #: changes the density the classifier consumes while leaving the catalogue
    #: and the grid alone, which is exactly the silent-splice hazard the stamp
    #: exists for.
    h_interpolation: str = "none"
    #: Finite H extension [D-19]; ``None`` = the D-4 support ends at 25.5.
    #: Hashed into the stamp for the same reason as ``h_interpolation`` (it
    #: swaps the density while leaving catalogue and grid alone), and omitted
    #: from the payload at its default so historical stamps are preserved.
    h_extend_to: float | None = None


# Per-worker state, populated by the ProcessPoolExecutor initializer so the
# 45,936-bin model is parsed once per worker rather than once per tracklet.
_W: dict = {}


def _init_worker(model: str, variant: str, score_cfg: dict) -> None:
    _ext = score_cfg.get("h_extend_to", None)
    neo, other = Digest2Histogram.neo_other_pair(
        model, variant,
        h_interpolation=str(score_cfg.get("h_interpolation", "none")),
        h_extend_to=None if _ext is None else float(_ext),
    )
    _W["neo"] = neo
    _W["other"] = other
    _W["cfg"] = ScoreConfig(**score_cfg)
    # H_max^model: the H value at and above which BOTH components are zero.
    # Single-sourced from the classifier so this diagnostic cannot describe a
    # different boundary than the one the likelihood actually truncated at —
    # 25.5 for the D-4 pair, `h_extend_to` under D-19, inf under the D-14 clamp
    # (which the harness never builds, since a clamped pair cannot be sampled).
    # Previously read off the "other" component alone; identical in v0, but
    # `h_support_end` is the pair-level definition and takes the max.
    _W["h_max"] = h_support_end(neo, other)


def _nearest_node(grid, rho_true: float, rho_dot_true: float):
    """Index of the node nearest the truth, plus the log-rho offset.

    The grid is a set of log-spaced rho columns, each carrying ``n_rhodot``
    midpoint nodes across that column's admissible rho_dot interval.  So: pick
    the nearest rho column in log space, then the nearest rho_dot node inside
    it.  Used only for the nearest-node HPD rank; the admissible-region
    coverage alarm is measured against the exact interval instead (see
    :func:`_ar_margin`), because a node-offset criterion moves with n_rhodot.
    """
    rho = grid.rho
    if rho.size == 0 or not (np.isfinite(rho_true) and rho_true > 0.0) \
            or not np.isfinite(rho_dot_true):
        return -1, float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.abs(np.log10(rho) - np.log10(rho_true))
    col_val = rho[np.nanargmin(dlog)]
    col = np.flatnonzero(rho == col_val)
    rd = grid.rho_dot[col]
    j = int(np.argmin(np.abs(rd - rho_dot_true)))
    return int(col[j]), float(np.nanmin(dlog))


def _ar_margin(att, R_obs, V_obs, rho_true: float, rho_dot_true: float):
    """How the truth sits relative to the admissible rho_dot interval at rho_true.

    Returns ``(outside_au_per_day, margin_frac)``:

    * ``outside_au_per_day`` — 0.0 when the truth is inside the interval, else
      the distance beyond the nearer edge, in AU/day.  A bound orbit must be
      inside, so anything above :data:`_AR_OUTSIDE_TOL` is a real alarm about
      the admissible-region construction (DESIGN §2.5) — and being physical, it
      does not weaken when ``n_rhodot`` is reduced, the way a "half a cell
      width" criterion does.
    * ``margin_frac`` — distance to the nearer edge as a fraction of the
      interval width; 0 at the edge, 0.5 dead centre.  Small values select the
      DESIGN §4.1 admissible-region-edge slice.

    The interval is evaluated EXACTLY at ``rho_true`` via
    :func:`ds42.ranging.admissible_rhodot_interval`, not reconstructed from
    the grid nodes, so both numbers are independent of the quadrature.
    """
    nan = float("nan")
    if not (np.isfinite(rho_true) and rho_true > 0.0) or not np.isfinite(rho_dot_true):
        return nan, nan
    lo, hi = admissible_rhodot_interval(att, R_obs, V_obs, np.array([rho_true]))
    lo_v, hi_v = float(lo[0]), float(hi[0])
    if not (np.isfinite(lo_v) and np.isfinite(hi_v)) or hi_v <= lo_v:
        # No bound solutions at all at rho_true: the truth cannot be covered.
        return float("inf"), 0.0
    outside = max(0.0, lo_v - rho_dot_true, rho_dot_true - hi_v)
    width = hi_v - lo_v
    margin = min(rho_dot_true - lo_v, hi_v - rho_dot_true) / width
    return float(outside), float(margin)


def _integrand_at_truth(att, R_obs, V_obs, rho_true, rho_dot_true,
                        elements_true, H_true) -> float:
    """The summed integrand (n_NEO + n_other) |J| evaluated EXACTLY at the truth.

    The HPD rank needs the posterior density at the true (rho, rho_dot).
    Reading it off the node nearest the truth is a poor estimator: the density
    is piecewise constant with sharp bin walls and carries the 1/(e sin i)
    integrable singularities of DESIGN §2.5, so a half-cell offset can miss a
    peak by orders of magnitude — and it misses *downward* far more often than
    upward, which biases the rank towards 1.  Evaluating at the truth removes
    that bias entirely: the generator knows the true elements and H exactly,
    and (no light-time in v0) the ranging map at (rho_true, rho_dot_true)
    reproduces them, so this is the same integrand ``classify`` sums, just
    sampled at a point instead of at a node.

    Only the *mass* sum on the right-hand side of the rank comparison is then
    a quadrature approximation, and that converges as an integral rather than
    pointwise.
    """
    rho = np.array([float(rho_true)], dtype=float)
    rhod = np.array([float(rho_dot_true)], dtype=float)
    elements_true = np.asarray(elements_true, dtype=float).reshape(1, 6)
    h = np.array([float(H_true)], dtype=float)

    total = 0.0
    jac: dict[str, float] = {}
    for density in (_W["neo"], _W["other"]):
        param = density.parameterization
        if param not in jac:
            det = ranging_jacobian_det(
                att.ra, att.dec, att.ra_rate, att.dec_rate, rho, rhod,
                R_obs, V_obs, parameterization=param,
            )
            jac[param] = float(np.abs(det[0]))
        el3 = (
            qei_from_elements(elements_true)
            if param == "qei"
            else elements_true[..., 0:3]
        )
        n_c = float(np.asarray(density.density(el3, h), dtype=float).ravel()[0])
        if np.isfinite(n_c) and np.isfinite(jac[param]):
            total += n_c * jac[param]
    return total


def _emit(rec: dict) -> tuple:
    """Flatten a by-name record into RECORD_FIELDS order (the wire format)."""
    missing = set(RECORD_FIELDS) - set(rec)
    if missing:  # pragma: no cover - programming error
        raise KeyError(f"score record is missing {sorted(missing)}")
    return tuple(rec[name] for name in RECORD_FIELDS)


def _score_one(task: tuple) -> tuple:
    """Score one tracklet. Module-level and picklable (ProcessPoolExecutor)."""
    (
        index, ra, dec, ra_rate, dec_rate, jd, v_mag,
        rx, ry, rz, vx, vy, vz, rho_true, rho_dot_true,
        el0, el1, el2, el3, el4, el5, h_true,
    ) = task
    elements_true = np.array([el0, el1, el2, el3, el4, el5], dtype=float)
    cfg: ScoreConfig = _W["cfg"]
    R_obs = np.array([rx, ry, rz], dtype=float)
    V_obs = np.array([vx, vy, vz], dtype=float)
    att = Attributable(
        ra=float(ra), dec=float(dec), ra_rate=float(ra_rate),
        dec_rate=float(dec_rate),
        epoch=Time(float(jd), format="jd", scale="tdb"), v_mag=float(v_mag),
    )

    t0 = time.time()
    with warnings.catch_warnings():
        # Empty-region / zero-likelihood tracklets are *recorded* (p_neo = nan),
        # not shouted about once per worker task.
        warnings.simplefilter("ignore", RuntimeWarning)
        res = classify(
            att, _W["neo"], _W["other"], R_obs=R_obs, V_obs=V_obs,
            rho_min=cfg.rho_min, rho_max=cfg.rho_max,
            n_rho=cfg.n_rho, n_rhodot=cfg.n_rhodot, G=cfg.G,
        )
    elapsed = time.time() - t0

    grid = res.grid
    n_nodes = int(grid.rho.size)
    nan = float("nan")
    # Built by name, then emitted in RECORD_FIELDS order: a positional tuple
    # literal silently mis-assigns every field after an inserted one.
    out: dict[str, float] = dict.fromkeys(RECORD_FIELDS, nan)
    out["index"] = int(index)
    out["p_neo"] = float(res.p_neo)
    out["log_lr"] = float(res.log_lr)
    out["lambda_neo"] = float(res.lambda_neo)
    out["lambda_other"] = float(res.lambda_other)
    out["n_nodes"] = n_nodes
    out["elapsed_s"] = elapsed
    out["truth_rho_in_range"] = int(
        np.isfinite(rho_true) and cfg.rho_min <= rho_true <= cfg.rho_max
    )

    # Physical, quadrature-independent admissible-region coverage check.
    outside, margin = _ar_margin(att, R_obs, V_obs, rho_true, rho_dot_true)
    out["rhodot_outside_au_per_day"] = outside
    out["rhodot_margin_frac"] = margin

    if n_nodes == 0:
        out["ar_measure"] = 0.0
        return _emit(out)

    weight = np.asarray(grid.weight, dtype=float)
    dens = np.asarray(res.node_density_neo, dtype=float) + np.asarray(
        res.node_density_other, dtype=float
    )
    dens = np.where(np.isfinite(dens), dens, 0.0)
    mass = dens * weight
    mass_tot = float(mass.sum())

    # Per-node implied (q, H) come straight from the classifier now — they are
    # public fields of ClassificationResult, so recomputing them here would
    # only add ~9% CPU and a chance to drift from what classify() actually
    # integrated.
    h_max = _W["h_max"]
    ar_measure = float(weight.sum())
    ar_frac_h = res.measure_fraction_H_above(h_max)
    with np.errstate(invalid="ignore"):
        q_node = np.asarray(res.node_q, dtype=float)
        h_node = np.asarray(res.node_H, dtype=float)
        corner = (q_node > Q_NEO_BOUNDARY) & (h_node > h_max)
        corner = np.where(np.isfinite(corner), corner, False)
    ar_frac_hq = (
        float(weight[corner].sum() / ar_measure) if ar_measure > 0 else nan
    )
    # DESIGN §2.6 (a): identically zero whenever both components share one H
    # support end (every v0 configuration, extended or not), because the
    # density is exactly zero above `h_max` on both sides by construction.
    mass_frac_hq = res.posterior_mass_where(corner, component="both")

    # --- collapse statistics ------------------------------------------------
    if mass_tot > 0 and np.isfinite(rho_true) and rho_true > 0:
        near = (grid.rho >= 0.5 * rho_true) & (grid.rho <= 2.0 * rho_true)
        # Clipped: partial sums of the same array need not add up to the total
        # to the last ULP, and a "fraction" of 1 + 2e-16 is just noise.
        collapse2 = float(min(1.0, max(0.0, mass[near].sum() / mass_tot)))
    else:
        collapse2 = nan

    k, d_log_rho = _nearest_node(grid, float(rho_true), float(rho_dot_true))
    try:
        d_truth = _integrand_at_truth(
            att, R_obs, V_obs, rho_true, rho_dot_true, elements_true, h_true
        )
    except Exception:  # pragma: no cover - degenerate geometry
        d_truth = nan
    d_node = float(dens[k]) if k >= 0 else nan

    def _rank(level: float) -> float:
        if not (mass_tot > 0) or not np.isfinite(level):
            return nan
        above = mass[dens > level].sum()
        tied = mass[dens == level].sum()
        return float(min(1.0, max(0.0, (above + 0.5 * tied) / mass_tot)))

    hpd = _rank(d_node)
    hpd_exact = _rank(d_truth)
    d_max = float(dens.max()) if dens.size else nan
    imap = int(np.argmax(mass)) if mass_tot > 0 else -1

    out.update(
        ar_measure=ar_measure,
        ar_frac_h_above=ar_frac_h,
        ar_frac_h_above_q_above=ar_frac_hq,
        mass_frac_q_above_h_above=mass_frac_hq,
        collapse_frac_rho2=collapse2,
        hpd_rank=hpd,
        hpd_rank_exact=hpd_exact,
        truth_density=d_truth,
        truth_node_density=d_node,
        max_node_density=d_max,
        rho_map=float(grid.rho[imap]) if imap >= 0 else nan,
        rhodot_map=float(grid.rho_dot[imap]) if imap >= 0 else nan,
        d_log10_rho_truth=d_log_rho,
    )
    return _emit(out)


def _tasks_from_generation(gen: Generation, indices: np.ndarray) -> list[tuple]:
    return [
        (
            int(i), gen.ra[i], gen.dec[i], gen.ra_rate[i], gen.dec_rate[i],
            gen.epoch_jd_tdb[i], gen.v_mag[i],
            gen.R_obs[i, 0], gen.R_obs[i, 1], gen.R_obs[i, 2],
            gen.V_obs[i, 0], gen.V_obs[i, 1], gen.V_obs[i, 2],
            gen.rho_true[i], gen.rho_dot_true[i],
            gen.elements[i, 0], gen.elements[i, 1], gen.elements[i, 2],
            gen.elements[i, 3], gen.elements[i, 4], gen.elements[i, 5],
            gen.H[i],
        )
        for i in indices
    ]


def _chunk_path(run_dir: str, start: int, stop: int) -> str:
    return os.path.join(run_dir, "chunks", f"chunk_{start:06d}_{stop:06d}.npz")


def _run_stamp(run_dir: str, cfg: ScoreConfig) -> str:
    """Identity of (this catalogue, this scoring configuration).

    Written into every chunk and checked on ``--resume``.  Without it a resumed
    run would happily splice chunks computed at a different grid resolution, or
    against a different generated catalogue, into one results table — which
    would be invisible in the output and fatal to the gate.

    ``h_interpolation`` and ``h_extend_to`` are omitted from the payload when
    they hold their defaults (``"none"`` / ``None``).  That is a deliberate,
    narrow exception: the fields were added
    after chunks existed on disk, and hashing it unconditionally would have
    changed the stamp of every run ever computed — forcing a full
    recomputation of every cached chunk in the repository even though nothing
    about those runs' scoring changed.  With the exception, a histogram-mode
    run reproduces its historical stamp exactly, while
    ``h_interpolation="pchip"`` changes it — which is the case the stamp
    exists for, since the mode swaps the density the classifier consumes while
    leaving the catalogue and the grid untouched.  Any *future* field must
    either be added to this exception or accept invalidating the cache.

    ``h_extend_to`` [D-19] follows exactly that rule: added later still, so it
    is popped at its ``None`` default (every histogram-mode and pchip-mode run
    on disk keeps its stamp), and hashed whenever it is set — a
    ``--h-extend-to 30`` run can never splice a chunk scored against the
    unextended support, nor one extended to a different reach.
    """
    import hashlib

    payload = asdict(cfg)
    if payload.get("h_interpolation", "none") == "none":
        payload.pop("h_interpolation", None)
    if payload.get("h_extend_to", None) is None:
        payload.pop("h_extend_to", None)

    h = hashlib.sha256()
    with open(os.path.join(run_dir, "gen.npz"), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    h.update(json.dumps(payload, sort_keys=True).encode())
    return h.hexdigest()


def _content_hash(arrays: dict[str, np.ndarray]) -> str:
    """SHA-256 over the record arrays themselves.

    The provenance stamp says "this chunk was produced from that catalogue with
    that config"; it says nothing about whether the *values* on disk are still
    the ones that were written.  A truncated write, a bit flip, or an edited
    file passes the stamp check and silently poisons the merged table.  Hashing
    the arrays closes that hole: on resume the values are re-hashed and a
    mismatch forces recomputation.
    """
    import hashlib

    h = hashlib.sha256()
    for name in RECORD_FIELDS:
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _rows_to_arrays(rows: list[tuple]) -> dict[str, np.ndarray]:
    cols = list(zip(*rows)) if rows else [()] * len(RECORD_FIELDS)
    out = {}
    for name, col in zip(RECORD_FIELDS, cols):
        dtype = np.int64 if name in _INT_FIELDS else np.float64
        out[name] = np.asarray(col, dtype=dtype)
    return out


def score(
    run_dir: str,
    cfg: ScoreConfig,
    workers: int = 8,
    chunk: int = 500,
    resume: bool = False,
    limit: int | None = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Score the generated catalogue in ``run_dir``; write chunks + scores.npz.

    Chunks are contiguous index ranges written as soon as they complete, so a
    killed run resumes at chunk granularity with ``--resume``.  Scoring is a
    pure function of the catalogue, so a resumed run is bit-identical; chunks
    carry a stamp of (catalogue bytes, scoring config) and any that do not
    match are recomputed rather than reused.

    Throughput note: at the default (1200, 50) grid one tracklet costs ~0.55 s
    of CPU -- 30% cheaper than (400, 200) and more accurate, because the
    quadrature error lives almost entirely in the rho direction (see
    ``classifier.N_RHO_DEFAULT``).  The work is numpy-elementwise rather than
    BLAS-bound.  Historical note: at (400, 200) one tracklet costs ~0.8 s, and the
    work is numpy-elementwise rather than BLAS-bound.  Exporting
    ``OMP_NUM_THREADS=1`` before launching avoids the mild thread
    oversubscription that ``workers`` processes x an unbounded BLAS pool
    otherwise produces (measured: ~1.5x wasted CPU, little effect on wall time).
    """
    gen = load_generation(run_dir)
    n = len(gen) if limit is None else min(int(limit), len(gen))
    chunk_dir = os.path.join(run_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    stamp = _run_stamp(run_dir, cfg)

    # The generator's rho floor must not admit truths the classifier will never
    # integrate over: rho_floor < rho_min would put the true (rho, rho_dot)
    # outside the integration domain by construction, which is a generator/
    # classifier mismatch, not a result.
    gen_floor = float(gen.config.get("rho_floor", cfg.rho_min))
    if gen_floor < cfg.rho_min:
        raise ValueError(
            f"generator rho_floor = {gen_floor:g} AU is below the scoring "
            f"rho_min = {cfg.rho_min:g} AU: the test population is not "
            "contained in the classifier's integration domain. Regenerate with "
            f"--rho-floor {cfg.rho_min:g} (or score with a smaller --rho-min)."
        )

    # The generator and the classifier must consume the SAME population model.
    # DESIGN §4.1's entire calibration argument is that the test set is drawn
    # from the classifier's own prior; a run that generated from the histogram
    # and scored with the reconstruction (or the reverse) is not a
    # self-consistency test at all, it is the §4.3 prior-misspecification
    # experiment wearing the gate's clothes — and it would still produce a
    # PASS/FAIL verdict, which is the dangerous part.  Same reasoning as the
    # rho_floor check above, and the same treatment: hard failure, not a
    # warning.
    gen_interp = str(gen.config.get("h_interpolation", "none"))
    if gen_interp != cfg.h_interpolation:
        raise ValueError(
            f"generator h_interpolation = {gen_interp!r} but scoring "
            f"h_interpolation = {cfg.h_interpolation!r}: the test population "
            "was not drawn from the density the classifier is about to use, so "
            "DESIGN §4.1's self-consistency argument does not hold and the "
            "verdict would be meaningless. Score with "
            f"--h-interpolation {gen_interp} to match the catalogue, or "
            f"regenerate with --h-interpolation {cfg.h_interpolation}."
        )

    # Same discipline for the D-19 finite H extension, and it matters more
    # here: extending the support multiplies the NEO component's total count by
    # ~2.7, so a generator/scorer mismatch would not merely reshape the faint
    # end, it would change the mixture fraction the calibration argument is
    # about.  Compared exactly (both are either None or a float read back from
    # the same JSON), not with a tolerance: there is no legitimate reason for
    # the two to differ by a little.
    gen_ext = gen.config.get("h_extend_to", None)
    gen_ext = None if gen_ext is None else float(gen_ext)
    if gen_ext != cfg.h_extend_to:
        def _flag(v):
            return "omit --h-extend-to" if v is None else f"--h-extend-to {v:g}"
        raise ValueError(
            f"generator h_extend_to = {gen_ext!r} but scoring h_extend_to = "
            f"{cfg.h_extend_to!r}: the test population was not drawn from the "
            "density the classifier is about to use, so DESIGN §4.1's "
            "self-consistency argument does not hold and the verdict would be "
            "meaningless (the extension changes the NEO component's total "
            f"count by ~2.7x, i.e. the mixture fraction itself). Score with "
            f"{_flag(gen_ext)} to match the catalogue, or regenerate with "
            f"{_flag(cfg.h_extend_to)}."
        )

    bounds = [(s, min(s + chunk, n)) for s in range(0, n, max(1, int(chunk)))]
    keep_names = {os.path.basename(_chunk_path(run_dir, *b)) for b in bounds}
    done: dict[tuple[int, int], dict] = {}
    todo: list[tuple[int, int]] = []
    n_stale = 0
    for b in bounds:
        p = _chunk_path(run_dir, *b)
        if resume and os.path.exists(p):
            try:
                with np.load(p) as z:
                    arrays = {k: z[k] for k in z.files}
                content = str(arrays.pop("_content", ""))
                if str(arrays.pop("_stamp", "")) != stamp:
                    raise ValueError("stale chunk (different catalogue or config)")
                if set(arrays) != set(RECORD_FIELDS):
                    raise ValueError("schema mismatch")
                if content != _content_hash(arrays):
                    raise ValueError("content hash mismatch (corrupt or edited)")
                done[b] = arrays
                continue
            except Exception:
                # Corrupt, truncated, tampered, stale (different catalogue or
                # grid), or an older schema: recompute rather than splice.
                n_stale += 1
                os.remove(p)
        todo.append(b)

    # Chunk files left over from a run with a different --chunk size would
    # never be read (their bounds do not appear above) but would be silently
    # left behind to confuse the next resume; drop them.
    n_orphan = 0
    for name in sorted(os.listdir(chunk_dir)):
        if name.startswith("chunk_") and name.endswith(".npz") and name not in keep_names:
            os.remove(os.path.join(chunk_dir, name))
            n_orphan += 1

    if verbose:
        print(
            f"[score] {n} tracklets, grid ({cfg.n_rho}, {cfg.n_rhodot}), "
            f"{len(done)} chunk(s) reused, {len(todo)} to compute"
            + (f" ({n_stale} stale/corrupt discarded)" if n_stale else "")
            + (f" ({n_orphan} orphaned chunk file(s) removed)" if n_orphan else "")
            + f", workers={workers}",
            flush=True,
        )

    t0 = time.time()
    n_done = 0
    executor = None
    try:
        if todo and workers and workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            # One pool for the whole run: the initializer parses the 45,936-bin
            # model once per worker, which would otherwise be paid per chunk.
            executor = ProcessPoolExecutor(
                max_workers=int(workers),
                initializer=_init_worker,
                initargs=(cfg.model, cfg.variant, asdict(cfg)),
            )
        elif todo:
            if _W.get("cfg") != cfg:
                _init_worker(cfg.model, cfg.variant, asdict(cfg))

        for start, stop in todo:
            tasks = _tasks_from_generation(gen, np.arange(start, stop))
            if executor is not None:
                rows = list(executor.map(_score_one, tasks, chunksize=1))
            else:
                rows = [_score_one(t) for t in tasks]
            arrays = _rows_to_arrays(rows)
            np.savez_compressed(
                _chunk_path(run_dir, start, stop),
                _stamp=np.str_(stamp),
                _content=np.str_(_content_hash(arrays)),
                **arrays,
            )
            done[(start, stop)] = arrays
            n_done += len(rows)
            if verbose:
                el = time.time() - t0
                rate = n_done / el if el > 0 else 0.0
                eta = f", eta {(n - stop) / rate:.0f} s" if rate > 0 else ""
                print(
                    f"  [score] {stop}/{n}  ({el:.1f} s, {rate:.2f} tracklet/s{eta})",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()

    merged = {
        name: (
            np.concatenate([done[b][name] for b in bounds])
            if bounds
            else np.zeros(0, dtype=np.int64 if name in _INT_FIELDS else np.float64)
        )
        for name in RECORD_FIELDS
    }
    order = np.argsort(merged["index"], kind="mergesort")
    merged = {k: v[order] for k, v in merged.items()}
    np.savez_compressed(os.path.join(run_dir, "scores.npz"), **merged)
    with open(os.path.join(run_dir, "score_config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    if verbose:
        print(f"[score] wrote {run_dir}/scores.npz ({time.time() - t0:.1f} s)")
    return merged


def load_scores(run_dir: str) -> dict[str, np.ndarray]:
    with np.load(os.path.join(run_dir, "scores.npz")) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

#: Variance floor for the reliability z-statistic.  A tracklet scored exactly
#: p = 0 or 1 that turns out to be the other class is an *impossible* event
#: under the model; clipping keeps z finite (and enormous) instead of inf, so
#: the diagram still renders while the criterion still fails.
_P_VAR_FLOOR = 1e-9

#: Default number of reliability bins.  Five, not ten: the bins are equal
#: EXPECTED-NEO-COUNT, and at the mixture base rate a gate-scale run only has
#: ~65 expected NEOs to spread over them.  Ten bins put 0.03-1.3 expected NEOs
#: in nine of them, which is no test at all.
N_RELIABILITY_BINS = 5

#: Parametric-bootstrap replicates used to threshold the reliability statistics.
N_BOOTSTRAP = 2000


def null_label_probs(
    p: np.ndarray, w_neo: float = 1.0, w_other: float = 1.0
) -> np.ndarray:
    """P(y = 1 | observables) under the *generator*, given the model's ``p``.

    In gate mode (``w_neo == w_other == 1``) this is just ``p``: the test
    population is drawn from the model, so the label of a tracklet scored p is
    a Bernoulli(p) draw and that is exactly the null being tested.

    Under ``--neo-fraction`` enrichment the generator draws the class with
    probability P instead of the mixture rate, which multiplies the *generating*
    density of the NEO branch by ``1/w_neo`` and of the other branch by
    ``1/w_other``.  Conditioned on the observables the label odds are therefore
    tilted by ``w_other / w_neo``::

        odds_generator = (w_other / w_neo) * odds_model

    Simulating the null with this — rather than with ``p`` — is what makes the
    parametric bootstrap exact in enriched mode too.
    """
    p = np.asarray(p, dtype=float)
    t = float(w_other) / float(w_neo)
    if t == 1.0:
        return p.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        odds = t * p / (1.0 - p)
        out = odds / (1.0 + odds)
    # p == 1 -> odds inf -> nan; the tilt cannot move a certainty.
    return np.where(p >= 1.0, 1.0, np.where(p <= 0.0, 0.0, out))


def _bin_edges_equal_expected(p_sorted: np.ndarray, n_bins: int) -> list[slice]:
    """Split a p-sorted vector into ``n_bins`` slices of equal expected NEO count.

    "Expected NEO count" is ``sum(p)`` over the slice.  Binning uses ``p`` only —
    never the realised labels or the realised importance weights (which depend
    on the label and would leak it into the bin definition).
    """
    n = p_sorted.size
    if n == 0:
        return []
    n_bins = max(1, min(int(n_bins), n))
    cum = np.cumsum(p_sorted)
    total = float(cum[-1])
    if not (total > 0.0):  # degenerate: fall back to equal counts
        idx = np.array_split(np.arange(n), n_bins)
        return [slice(int(a[0]), int(a[-1]) + 1) for a in idx if a.size]
    targets = total * (np.arange(1, n_bins) / n_bins)
    cuts = np.searchsorted(cum, targets, side="left") + 1
    cuts = np.unique(np.clip(cuts, 1, n - 1))
    bounds = np.concatenate(([0], cuts, [n]))
    return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def reliability_bins(
    p: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    n_bins: int = N_RELIABILITY_BINS,
    w_neo: float = 1.0,
    w_other: float = 1.0,
) -> list[dict]:
    """Equal-expected-NEO-count reliability bins with a Poisson-binomial z.

    Binning
    -------
    Equal *expected NEO count*, not equal tracklet count.  At the mixture base
    rate (~0.5% of accepted tracklets are NEOs) equal-count bins put 0.03-1.3
    expected NEOs into nine of ten bins: those bins cannot fail and cannot pass,
    they merely inflate the family-wise false-alarm rate of a max|z| criterion.

    The z statistic
    ---------------
    Per bin, under "the posterior is calibrated",

        D   = sum_i w_i (y_i - p_i)
        Var = w_neo * w_other * sum_i p_i (1 - p_i)
        z   = D / sqrt(Var)

    The variance is the **product** of the two class weights times the
    Poisson-binomial variance — NOT ``sum_i w_i^2 p_i(1-p_i)``.  The realised
    weight w_i is a function of the realised label, so it is not a constant to
    be pulled through a variance: writing T_i = w(c_i)(c_i - p_i) with the
    generator's tilted label probability (:func:`null_label_probs`) and
    evaluating E[T_i] and E[T_i^2] exactly gives mean 0 and
    Var = w_neo w_other p(1-p).  In gate mode (w_neo = w_other = 1) this
    reduces to sum p(1-p); at f = 0.5 enrichment the two differ by up to ~50x.

    The p_i need not be equal within a bin, which is why the Poisson-binomial
    form is used rather than a single-p binomial.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    good = np.isfinite(p)
    p, y, w = p[good], y[good], w[good]
    if p.size == 0:
        return []

    order = np.argsort(p, kind="mergesort")
    ps, ys, ws = p[order], y[order], w[order]
    ww = float(w_neo) * float(w_other)

    bins = []
    for sel in _bin_edges_equal_expected(ps, n_bins):
        pi, yi, wi = ps[sel], ys[sel], ws[sel]
        if pi.size == 0:
            continue
        sw = float(wi.sum())
        pv = np.clip(pi, _P_VAR_FLOOR, 1.0 - _P_VAR_FLOOR)
        d = float(np.sum(wi * (yi - pi)))
        var = ww * float(np.sum(pv * (1.0 - pv)))
        sigma = math.sqrt(var) if var > 0 else float("nan")
        bins.append(
            {
                "n": int(pi.size),
                "sum_w": sw,
                "p_lo": float(pi.min()),
                "p_hi": float(pi.max()),
                "p_mean": float(np.sum(wi * pi) / sw) if sw > 0 else float("nan"),
                "obs_frac": float(np.sum(wi * yi) / sw) if sw > 0 else float("nan"),
                "sigma_frac": (sigma / sw) if (sw > 0 and np.isfinite(sigma)) else float("nan"),
                # Weighted, so that observed - expected is exactly D and the
                # two columns of the printed table are on the same scale in
                # enriched mode as well.  The *unweighted* sum is what the
                # binning uses (no label leakage) and is kept alongside.
                "expected_events": float(np.sum(wi * pi)),
                "expected_events_unweighted": float(np.sum(pi)),
                "observed_events": float(np.sum(wi * yi)),
                "z": (d / sigma) if (np.isfinite(sigma) and sigma > 0) else float("nan"),
                "poisson_binomial_var": var,
            }
        )
    return bins


def global_z(
    p: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    w_neo: float = 1.0,
    w_other: float = 1.0,
) -> dict:
    """The single global calibration statistic: observed vs expected NEO count.

        Z_tot = sum_i w_i (y_i - p_i) / sqrt(w_neo w_other sum_i p_i (1 - p_i))

    One number, all N tracklets, no binning.  It is the cheapest tripwire for a
    global odds bias — the exact failure mode a binned max|z| test is worst at
    detecting, because a uniform odds shift moves every bin a little and no bin
    a lot.  Measured power against p' = p^1.3 at gate scale: ~0.8 here versus
    ~0.0 for the binned test at matched size.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    good = np.isfinite(p)
    p, y, w = p[good], y[good], w[good]
    if p.size == 0:
        return {"z": float("nan"), "observed": float("nan"),
                "expected": float("nan"), "sigma": float("nan"), "n": 0}
    pv = np.clip(p, _P_VAR_FLOOR, 1.0 - _P_VAR_FLOOR)
    var = float(w_neo) * float(w_other) * float(np.sum(pv * (1.0 - pv)))
    sigma = math.sqrt(var) if var > 0 else float("nan")
    obs = float(np.sum(w * y))
    exp = float(np.sum(w * p))
    d = float(np.sum(w * (y - p)))
    return {
        "z": (d / sigma) if (np.isfinite(sigma) and sigma > 0) else float("nan"),
        "observed": obs,
        "expected": exp,
        "sigma": sigma,
        "n": int(p.size),
    }


def fit_odds_scale(
    p: np.ndarray,
    y: np.ndarray,
    w_neo: float = 1.0,
    w_other: float = 1.0,
    ci: float = 0.95,
) -> dict:
    """Maximum-likelihood global odds-scale ``c`` with a Wald interval.

    Alternative hypothesis: the model's odds are off by one constant factor,

        odds_true(x) = c * odds_model(x),

    so ``c = 1`` is perfect calibration, ``c > 1`` means the model
    under-states P(NEO) everywhere and ``c < 1`` that it over-states it.  This
    is the *effect size* behind :func:`global_z`: a significant Z_tot says
    "something is off", ``c_hat`` says by how much and in which direction.

    Fitted on the generator's tilted label probabilities (so the estimate is of
    the model's own odds error, not of the enrichment tilt).  ``c_hat`` is
    found by bisection on the score in ``log c``; the interval is Wald on
    ``log c`` from the observed Fisher information.  With no NEOs observed the
    likelihood is monotone and ``c_hat`` rails at the search bound, which is
    reported via ``railed``.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(bool)
    good = np.isfinite(p)
    p, y = p[good], y[good]
    nan = float("nan")

    # Tracklets scored exactly 0 or 1 carry NO information about a
    # multiplicative odds scale — a certainty stays a certainty under any
    # finite c — but if they are left in, their clipped odds (1e9) dominate the
    # likelihood and drag c_hat to whatever value makes the clip cancel.  On a
    # 500-tracklet shakedown with two p = 1 tracklets that produced a spurious
    # c_hat = 4e-5.  Drop them and say how many were dropped.
    certain = (p <= _P_VAR_FLOOR) | (p >= 1.0 - _P_VAR_FLOOR)
    n_certain = int(np.sum(certain))
    p, y = p[~certain], y[~certain]
    if p.size == 0:
        return {"c_hat": nan, "log_c": nan, "lo": nan, "hi": nan,
                "log_c_se": nan, "railed": True, "n_excluded_certain": n_certain,
                "n_used": 0}

    t = float(w_other) / float(w_neo)
    pv = np.clip(p, _P_VAR_FLOOR, 1.0 - _P_VAR_FLOOR)
    odds = pv / (1.0 - pv)

    def q(logc):
        o = t * math.exp(logc) * odds
        return o / (1.0 + o)

    def score(logc):  # d(loglik)/d(log c)
        return float(np.sum(y.astype(float) - q(logc)))

    lo_b, hi_b = -12.0, 12.0
    s_lo, s_hi = score(lo_b), score(hi_b)
    railed = not (s_lo > 0.0 > s_hi)
    if railed:
        logc = lo_b if s_hi > 0.0 else hi_b
        if s_lo <= 0.0:
            logc = lo_b
        elif s_hi >= 0.0:
            logc = hi_b
    else:
        a, b = lo_b, hi_b
        for _ in range(200):
            m = 0.5 * (a + b)
            if score(m) > 0.0:
                a = m
            else:
                b = m
        logc = 0.5 * (a + b)

    qi = q(logc)
    info = float(np.sum(qi * (1.0 - qi)))
    se = 1.0 / math.sqrt(info) if info > 0 else nan

    def _exp(x: float) -> float:
        """exp clamped to the search range; a railed fit has no finite bound."""
        return float(math.exp(min(max(x, lo_b), hi_b)))

    zc = 1.959963984540054 if abs(ci - 0.95) < 1e-9 else _normal_quantile(0.5 + ci / 2)
    return {
        "c_hat": _exp(logc),
        "log_c": float(logc),
        "log_c_se": float(se),
        "lo": _exp(logc - zc * se) if np.isfinite(se) else nan,
        "hi": _exp(logc + zc * se) if np.isfinite(se) else nan,
        "railed": bool(railed),
        "n_excluded_certain": n_certain,
        "n_used": int(p.size),
    }


def _normal_quantile(u: float) -> float:
    """Inverse standard-normal CDF by bisection on erfc (no scipy)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if 1.0 - normal_sf(m) < u:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def poisson_binomial_pmf(p: np.ndarray, k_max: int | None = None) -> np.ndarray:
    """Exact pmf of ``S = sum_i Bernoulli(p_i)`` over ``0..k_max``.

    Direct DP (repeated convolution with ``[1-p_i, p_i]``), truncated at
    ``k_max`` — chosen ~10 sigma above the mean by default, where the neglected
    tail is far below any p-value we would report.  O(N * k_max), which at gate
    scale (N = 1e4, k_max ~ 170) is milliseconds.
    """
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    mean = float(p.sum())
    if k_max is None:
        k_max = int(min(p.size, math.ceil(mean + 10.0 * math.sqrt(mean + 1.0) + 10.0)))
    k_max = int(max(1, min(k_max, p.size)))
    pmf = np.zeros(k_max + 1)
    pmf[0] = 1.0
    for pi in p:
        pi = min(max(float(pi), 0.0), 1.0)
        if pi == 0.0:
            continue
        pmf[1:] = pmf[1:] * (1.0 - pi) + pmf[:-1] * pi
        pmf[0] *= 1.0 - pi
    return pmf


def global_exact_p(p: np.ndarray, n_obs: int) -> float:
    """Exact two-sided p-value for the observed NEO *count* (gate mode only).

    In gate mode every weight is 1, so the global statistic's numerator is
    ``sum(y) - sum(p)`` and ``sum(y)`` is exactly Poisson-binomial: no
    simulation is needed, and none should be used.  ``sum(y)`` is an integer
    while its standard deviation is ~8 at gate scale, so the statistic lives on
    a coarse lattice; a bootstrap threshold then lands on a lattice point and
    the achieved level swings between ~0.7% and ~1.6% depending on which side
    of a tie it falls.  The exact tail removes that entirely.

    Convention: twice the smaller tail, capped at 1 — the standard (slightly
    conservative) two-sided rule for a discrete statistic.

    Under ``--neo-fraction`` enrichment the weighted numerator depends on
    *which* tracklets are NEOs, not merely how many, so it is not
    Poisson-binomial; there the bootstrap p-value is used instead.
    """
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    pmf = poisson_binomial_pmf(p)
    n_obs = int(n_obs)
    if n_obs < 0:
        return float("nan")
    if n_obs >= pmf.size:
        lower = 1.0
        upper = float(pmf[min(n_obs, pmf.size - 1):].sum())
    else:
        lower = float(pmf[: n_obs + 1].sum())
        upper = float(pmf[n_obs:].sum())
    return float(min(1.0, 2.0 * min(lower, upper)))


def bootstrap_reliability(
    p: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    n_bins: int = N_RELIABILITY_BINS,
    w_neo: float = 1.0,
    w_other: float = 1.0,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 20260901,
    alpha: float = 0.01,
) -> dict:
    """Parametric-bootstrap thresholds and p-values for the reliability tests.

    The observed p vector is held fixed and labels are re-simulated from the
    generator's own null, ``y* ~ Bernoulli(null_label_probs(p))``, ``n_boot``
    times.  Two statistics are tracked per replicate: max|z| over the bins and
    the global |Z_tot|.

    Why not an analytic level.  Under the mixture base rate most bins carry
    O(1) expected NEOs, so the per-bin z is nowhere near normal: an
    "all bins within 2 sigma" rule has a measured family-wise false-alarm rate
    of ~0.43, and the chi-square arm's nominal 1% level is really ~6%.  The
    bootstrap gives exact Type-I error by construction, for any binning, any
    base rate, and any weighting — including the p == 0/1 bins where the
    analytic z is meaningless.

    Multiplicity.  ``min_p`` and its bootstrap-calibrated threshold are
    computed and reported, but the verdict does not use them: see
    :func:`reliability_verdict` for why the plain union of the two arms already
    lands at the nominal level here, and why min-p costs a third of the power.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    good = np.isfinite(p)
    p, y, w = p[good], y[good], w[good]
    nan = float("nan")
    empty = {
        "n_boot": 0, "alpha": alpha,
        "max_abs_z_threshold": nan, "max_abs_z_p_value": nan,
        "global_z_threshold": nan, "global_z_p_value": nan,
        "min_p": nan, "min_p_threshold": nan, "family_wise_pass": False,
    }
    if p.size == 0 or n_boot <= 0:
        return empty

    obs_max = _max_abs_z(reliability_bins(p, y, w, n_bins, w_neo, w_other))
    obs_glob = abs(global_z(p, y, w, w_neo, w_other)["z"])

    p_null = null_label_probs(p, w_neo, w_other)
    rng = np.random.default_rng(seed)
    sims_max = np.empty(n_boot)
    sims_glob = np.empty(n_boot)
    sims_ny = np.empty(n_boot)
    # Gate mode: the global arm's null is exactly Poisson-binomial, so its
    # p-value is computed in closed form and the simulation is used only to
    # supply the joint (two-arm) null for the multiplicity correction.
    exact_global = (float(w_neo) == 1.0 and float(w_other) == 1.0)
    pmf = poisson_binomial_pmf(p) if exact_global else None
    for b in range(n_boot):
        ystar = (rng.random(p.size) < p_null).astype(float)
        # The importance weight follows the simulated label, exactly as it
        # does in a real run.
        wstar = np.where(ystar > 0.5, w_neo, w_other)
        sims_max[b] = _max_abs_z(reliability_bins(p, ystar, wstar, n_bins, w_neo, w_other))
        sims_glob[b] = abs(global_z(p, ystar, wstar, w_neo, w_other)["z"])
        sims_ny[b] = float(ystar.sum())

    fin = np.isfinite(sims_max) & np.isfinite(sims_glob)
    sims_max, sims_glob, sims_ny = sims_max[fin], sims_glob[fin], sims_ny[fin]
    B = sims_max.size
    if B == 0:
        return empty

    def _pv(sims, stat):
        # +1 smoothing: a p-value of exactly 0 is not supportable from B draws.
        if not np.isfinite(stat):
            return nan
        return float((np.sum(sims >= stat) + 1) / (B + 1))

    def _thr(sims):
        return float(np.quantile(sims, 1.0 - alpha))

    def _pv_exact(counts):
        """Two-sided exact p-value of a NEO count from the precomputed pmf."""
        c = np.clip(np.asarray(counts, dtype=int), 0, pmf.size - 1)
        cdf = np.cumsum(pmf)
        lower = cdf[c]
        upper = 1.0 - np.concatenate(([0.0], cdf))[c]
        return np.minimum(1.0, 2.0 * np.minimum(lower, upper))

    pv_max = _pv(sims_max, obs_max)
    if exact_global:
        obs_ny = int(round(float(np.sum(y))))
        pv_glob = float(_pv_exact(np.array([obs_ny]))[0])
        pv_glob_null = _pv_exact(sims_ny)
    else:
        pv_glob = _pv(sims_glob, obs_glob)
        pv_glob_null = None
    obs_minp = float(np.nanmin([pv_max, pv_glob]))

    # Null distribution of min-p: rank every replicate through the same two-arm
    # machinery.  Vectorised via sorted-array searches rather than B^2 loops.
    #
    # LEAVE-ONE-OUT is essential.  A replicate ranked against a set that
    # includes itself can never score better than 2/(B+1), while a fresh
    # observation above all B replicates scores 1/(B+1); the null min-p is then
    # stochastically too large, the alpha-quantile too lenient, and the measured
    # family-wise rate comes out at ~2% instead of 1%.  Excluding self makes the
    # replicate's p-value the same function of the same-sized reference set that
    # the observed statistic's is.
    srt_m = np.sort(sims_max)
    srt_g = np.sort(sims_glob)

    def _pv_loo(sims, srt):
        ge_incl = B - np.searchsorted(srt, sims, side="left")  # includes self
        return (1.0 + (ge_incl - 1.0)) / B

    minp_null = np.minimum(
        _pv_loo(sims_max, srt_m),
        pv_glob_null if pv_glob_null is not None else _pv_loo(sims_glob, srt_g),
    )
    # Discrete distribution: use the largest attainable value whose left tail
    # mass does not exceed alpha, so the test is conservative rather than
    # anti-conservative under ties.
    cand = np.unique(minp_null)
    minp_thr = 0.0
    for v in cand:
        if float(np.mean(minp_null <= v)) <= alpha:
            minp_thr = float(v)
        else:
            break
    family_pass = bool(np.isfinite(obs_minp) and obs_minp > minp_thr)

    return {
        "n_boot": int(B),
        "alpha": float(alpha),
        "max_abs_z_observed": obs_max,
        "max_abs_z_threshold": _thr(sims_max),
        "max_abs_z_p_value": pv_max,
        "global_z_observed": obs_glob,
        "global_z_threshold": _thr(sims_glob),
        "global_z_p_value": pv_glob,
        "global_p_is_exact": bool(exact_global),
        "min_p": obs_minp,
        "min_p_threshold": minp_thr,
        "family_wise_pass": family_pass,
    }


def _max_abs_z(bins: list[dict]) -> float:
    zs = np.array([b["z"] for b in bins], dtype=float)
    zs = zs[np.isfinite(zs)]
    return float(np.max(np.abs(zs))) if zs.size else float("nan")


def reliability_verdict(
    bins: list[dict],
    boot: dict,
    glob: dict,
    odds: dict | None = None,
) -> dict:
    """PASS iff neither bootstrap-thresholded statistic is exceeded.

    Two arms, combined by the bootstrap min-p rule at a family-wise alpha = 1%:

    * ``max|z|`` over the equal-expected-count bins — sensitive to a
      *localised* miscalibration (one region of score space wrong);
    * ``|Z_tot|`` — sensitive to a *global* odds bias, which the binned arm is
      nearly blind to.

    FAIL if either arm's p-value falls at or below alpha.  Measured family-wise
    false-alarm rate of that union: 0.0098 against a nominal 0.01.

    The chi-square over bins is still computed and displayed, but it is NOT in
    the verdict: its nominal 1% level is really ~6% because the per-bin z is
    far from normal in the low-expected-count bins.
    """
    zs = np.array([b["z"] for b in bins], dtype=float)
    finite = np.isfinite(zs)
    chi2 = float(np.sum(zs[finite] ** 2))
    dof = int(finite.sum())
    worst = _max_abs_z(bins)

    thr_m = boot.get("max_abs_z_threshold", float("nan"))
    thr_g = boot.get("global_z_threshold", float("nan"))
    gz = abs(glob.get("z", float("nan")))
    alpha = float(boot.get("alpha", 0.01))
    pv_m = boot.get("max_abs_z_p_value", float("nan"))
    pv_g = boot.get("global_z_p_value", float("nan"))

    # Per-arm flags are taken on the p-values, not the thresholds: the global
    # arm's p-value is exact (Poisson-binomial) in gate mode, and a threshold
    # comparison on a lattice-valued statistic is ambiguous at ties.
    binned_ok = bool(np.isfinite(pv_m) and pv_m > alpha)
    global_ok = bool(np.isfinite(pv_g) and pv_g > alpha)
    # Verdict = FAIL if EITHER arm trips at alpha.  No multiplicity correction
    # is applied, and none is needed: the two arms are strongly positively
    # correlated (same labels) and the global arm's exact discrete p-value is
    # itself conservative, so the measured family-wise false-alarm rate of the
    # union is 0.0098 against a nominal 0.01 (6000 MC null trials, gate-scale
    # score vector).  A min-p correction on top of that measured 0.0052 —
    # correctly levelled, but it threw away a third of the power against a real
    # odds distortion (0.33 vs 0.55).  min_p is still reported for reference.
    family_ok = bool(binned_ok and global_ok)
    return {
        "max_abs_z": worst,
        "max_abs_z_threshold": thr_m,
        "max_abs_z_p_value": boot.get("max_abs_z_p_value", float("nan")),
        "binned_pass": binned_ok,
        "global_z": glob.get("z", float("nan")),
        "global_z_threshold": thr_g,
        "global_z_p_value": boot.get("global_z_p_value", float("nan")),
        "global_pass": global_ok,
        "observed_neo_count": glob.get("observed", float("nan")),
        "expected_neo_count": glob.get("expected", float("nan")),
        "odds_scale": odds or {},
        "chi2_display_only": chi2,
        "chi2_dof": dof,
        "chi2_p_value_display_only": chi2_sf(chi2, dof) if dof > 0 else float("nan"),
        "min_p": boot.get("min_p", float("nan")),
        "min_p_threshold": boot.get("min_p_threshold", float("nan")),
        "n_bins_low_variance": int(
            sum(1 for b in bins if b["poisson_binomial_var"] < 5.0)
        ),
        "pass": family_ok,
    }


def _fmt_frac(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.4g}"


def analyze(
    run_dir: str,
    n_bins: int = N_RELIABILITY_BINS,
    plots: bool = True,
    hpd_ks_d_max: float = 0.075,
    coverage_min: float = 0.99,
    undefined_max: float = 0.01,
    out_of_domain_max: float = 1e-3,
    n_boot: int = N_BOOTSTRAP,
    boot_seed: int = 20260901,
    gate: bool = False,
    gate_min_n: int = 10000,
    verbose: bool = True,
) -> dict:
    """Read the saved catalogue + scores; produce plots, summary, PASS/FAIL.

    RELIABILITY
    -----------
    Two bootstrap-thresholded arms (see :func:`reliability_verdict`): max|z|
    over equal-expected-NEO-count bins, and the global |Z_tot|.  No analytic
    "within 2 sigma" or chi-square level is used in the verdict; both are
    unusable at the mixture base rate (measured family-wise false alarm 0.43
    and 0.06 respectively against a nominal 0.01).

    HPD
    ---
    Gates on the KS *statistic* D, not the KS p-value, and on BOTH rank
    estimators (nearest-node and exact-at-truth).  The null "HPD rank ~
    Uniform(0,1)" is only approximately true at finite grid resolution, so at
    gate N any finite grid fails a p-value gate on discretisation bias alone.
    Measured D at N = 1e4: 0.0499 (nearest) / 0.0581 (exact) on the (1200, 50)
    grid, versus 0.0679 / 0.0753 on (400, 200).  The default tolerance 0.075
    sits just above the (1200, 50) pair; it is a floor set by the quadrature,
    **not a licence, and must be tightened whenever the grid is refined**.  The
    gap between the two estimators is reported as the grid-adequacy readout:
    they bias in opposite directions and converge under refinement.
    """
    gen = load_generation(run_dir)
    sc = load_scores(run_dir)
    idx = sc["index"].astype(int)
    y = gen.is_neo[idx].astype(bool)
    w = gen.weight[idx].astype(float)
    p = sc["p_neo"].astype(float)

    counters = gen.counters or {}
    w_neo = float(counters.get("weight_neo", 1.0))
    w_other = float(counters.get("weight_other", 1.0))

    n_total = int(p.size)

    # --- D-19 triage-policy accounting --------------------------------------
    # A fully-truncated tracklet (non-empty admissible region, both Lambdas
    # exactly zero, every evaluable node at/above the H support end) is now
    # scored p_neo = 1.0 by policy rather than left as nan.  It is a *defined
    # number* but NOT a likelihood-based score: both Lambdas were exactly
    # zero, so it carries no information at all about whether the posterior is
    # calibrated.  This harness therefore treats it as its own third category,
    # everywhere:
    #
    #   * EXCLUDED from the reliability diagram and from the global-Z inputs.
    #     Not diluted — excluded.  Measured by the adversarial review: ~34
    #     policy rows (the 0.17% the real-data rate would give at N = 2e4)
    #     already make one of five equal-expected-NEO-count bins vacuous
    #     (every member p = 1 exactly), and the real-data rate of 2.2% (~440
    #     rows) makes four of five vacuous; worse, a policy row on a
    #     true non-NEO contributes (1 - p) = 0 predicted with an observed
    #     failure, which pushes max|z| arbitrarily. Neither is a measurement
    #     of the classifier.
    #   * NOT counted as likelihood-defined by SCORES_DEFINED.  Counting them
    #     as defined would let a run pass the undefined-fraction criterion by
    #     assigning scores it cannot justify — at the real-data 2.2% rate that
    #     is exactly the margin between FAIL and PASS on the default
    #     tolerance.
    #   * reported as their own line, with the three counts kept distinct.
    #
    # Identified from the recorded fields rather than from a new column: the
    # combination (n_nodes > 0, lambda_neo == 0, lambda_other == 0) is exactly
    # `classifier._posterior`'s "fully_truncated_flagged" branch, so no
    # RECORD_FIELDS change is needed and every cached chunk in the repository
    # stays valid.  (A schema change here would force a full re-score of every
    # stored run to add a column that is already derivable.)
    #
    # The `isfinite(p)` term is NOT redundant and closes a double-count on
    # cached chunks written BEFORE D-19: back then a fully-truncated tracklet
    # was stored with the same zero Lambdas and a *nan* p_neo, so the triple
    # alone would tag it both "undefined" and "policy-assigned" (verified on
    # `enriched_f01`'s one such row).  A policy row by definition has a finite
    # p, so intersecting with `isfinite` makes the three categories a genuine
    # partition of the run and keeps a pre-D-19 run classified the way it was
    # actually scored.
    n_nodes_col = sc["n_nodes"].astype(np.int64)
    lam_n_col = sc["lambda_neo"].astype(float)
    lam_o_col = sc["lambda_other"].astype(float)
    finite_p = np.isfinite(p)
    policy_assigned = (
        finite_p & (n_nodes_col > 0) & (lam_n_col == 0.0) & (lam_o_col == 0.0)
    )
    n_policy = int(np.sum(policy_assigned))
    frac_policy = n_policy / n_total if n_total else float("nan")

    # `defined` == "carries likelihood information", which is what every
    # statistic below wants.  The three categories partition the run:
    #   n_defined + n_policy + n_undef == n_total.
    defined = finite_p & ~policy_assigned
    n_defined = int(np.sum(defined))
    n_undef = n_total - n_defined - n_policy
    # SCORES_DEFINED's criterion is on tracklets that produced no usable
    # likelihood score, which now includes the policy-assigned ones.
    frac_undef = (n_undef + n_policy) / n_total if n_total else float("nan")
    assert n_defined + n_policy + n_undef == n_total  # the partition

    # --- gate eligibility ---------------------------------------------------
    n_gen = len(gen)
    gate_reasons = []
    if n_total != n_gen:
        gate_reasons.append(
            f"only {n_total} of {n_gen} generated tracklets scored "
            "(a --limit run is not a gate run)"
        )
    if n_total < int(gate_min_n):
        gate_reasons.append(f"N = {n_total} < {int(gate_min_n)}")
    if gen.config.get("neo_fraction") is not None:
        gate_reasons.append(
            f"enriched generation (--neo-fraction "
            f"{gen.config.get('neo_fraction')}); the gate must run the mixture"
        )
    gate_eligible = not gate_reasons

    # --- reliability --------------------------------------------------------
    # Every statistic below is computed on `defined` = finite AND not
    # policy-assigned, i.e. on likelihood-based scores only.  That includes the
    # reliability bins, the global count statistic, the odds-scale fit, the
    # bootstrap thresholds, the ROC/AUC and the impossible-event counters.  A
    # D-19 policy row is a defined number with zero likelihood content behind
    # it, so pooling it into any of these measures the *policy* and reports the
    # answer as a property of the classifier.  The count is reported on its own
    # line instead (and is 0 on every run in this repository, so no published
    # number moves).
    pd_, yd_, wd_ = p[defined], y[defined], w[defined]
    bins = reliability_bins(pd_, yd_, wd_, n_bins=n_bins,
                            w_neo=w_neo, w_other=w_other)
    glob = global_z(pd_, yd_, wd_, w_neo=w_neo, w_other=w_other)
    odds = fit_odds_scale(pd_, yd_, w_neo=w_neo, w_other=w_other)
    boot = bootstrap_reliability(
        pd_, yd_, wd_, n_bins=n_bins, w_neo=w_neo, w_other=w_other,
        n_boot=n_boot, seed=boot_seed,
    )
    rel = reliability_verdict(bins, boot, glob, odds)

    auc = auc_weighted(pd_, yd_, wd_)
    fpr, tpr = roc_curve_weighted(pd_, yd_, wd_)

    # --- HPD ranks ----------------------------------------------------------
    hpd = sc["hpd_rank"].astype(float)
    hpd_ok = np.isfinite(hpd)
    ks_d, ks_p, ks_neff = ks_uniform(hpd[hpd_ok], w[hpd_ok])
    dtn = sc["truth_node_density"].astype(float)
    zero_truth = np.isfinite(dtn) & (dtn <= 0.0)
    frac_zero_truth = (
        float(np.sum(w[zero_truth]) / np.sum(w)) if np.sum(w) > 0 else float("nan")
    )
    sub = hpd_ok & ~zero_truth
    ks_d_nz, ks_p_nz, ks_neff_nz = ks_uniform(hpd[sub], w[sub])

    hpe = sc["hpd_rank_exact"].astype(float)
    hpe_ok = np.isfinite(hpe)
    ks_d_ex, ks_p_ex, _ = ks_uniform(hpe[hpe_ok], w[hpe_ok])
    hpd_gap = (
        abs(ks_d_ex - ks_d) if (np.isfinite(ks_d) and np.isfinite(ks_d_ex))
        else float("nan")
    )

    # --- truth-density invariant (NaN counts as a FAILURE) ------------------
    dt = sc["truth_density"].astype(float)
    n_truth_density_nan = int(np.sum(~np.isfinite(dt)))
    with np.errstate(invalid="ignore"):
        truth_pos = np.isfinite(dt) & (dt > 0.0)
    frac_truth_density_positive = (
        float(np.mean(truth_pos)) if dt.size else float("nan")
    )

    # --- collapse -----------------------------------------------------------
    coll = sc["collapse_frac_rho2"].astype(float)
    # Coverage alarm in PHYSICAL units: how far outside the admissible rho_dot
    # interval (evaluated exactly at rho_true, not reconstructed from nodes)
    # the truth falls, in AU/day.  A cell-width measure would weaken by 4x
    # simply from dropping n_rhodot 200 -> 50.
    outside = sc["rhodot_outside_au_per_day"].astype(float)
    rho_true = gen.rho_true[idx].astype(float)
    scfg = _score_config_of(run_dir)
    default = ScoreConfig()
    rho_min_s = scfg.rho_min if scfg is not None else default.rho_min
    rho_max_s = scfg.rho_max if scfg is not None else default.rho_max
    above_max = rho_true > rho_max_s
    below_min = rho_true < rho_min_s
    out_of_domain = above_max | below_min
    sw = float(np.sum(w)) if np.sum(w) > 0 else float("nan")
    frac_above_max = float(np.sum(w[above_max]) / sw) if np.isfinite(sw) else float("nan")
    frac_below_min = float(np.sum(w[below_min]) / sw) if np.isfinite(sw) else float("nan")
    frac_out_of_domain = (
        float(np.sum(w[out_of_domain]) / sw) if np.isfinite(sw) else float("nan")
    )
    # Coverage is now conditioned on the truth being inside the integration
    # domain; the out-of-domain rate is its own reported line and its own
    # criterion, rather than being silently folded in here.
    in_dom = ~out_of_domain
    cov_den = float(np.sum(w[in_dom]))
    covered = in_dom & np.isfinite(outside) & (outside <= _AR_OUTSIDE_TOL)
    cov_frac = (
        float(np.sum(w[covered]) / cov_den) if cov_den > 0 else float("nan")
    )

    # --- DESIGN §4.1 targeted slice: admissible-region edge ------------------
    edge = ar_edge_slice(gen, sc, margin_max=AR_EDGE_MARGIN)

    ar_h = sc["ar_frac_h_above"].astype(float)
    ar_hq = sc["ar_frac_h_above_q_above"].astype(float)
    mass_hq = sc["mass_frac_q_above_h_above"].astype(float)

    verdicts = {
        "RELIABILITY": rel["pass"],
        "HPD_RANK_UNIFORM": bool(
            np.isfinite(ks_d) and ks_d <= hpd_ks_d_max
            and np.isfinite(ks_d_ex) and ks_d_ex <= hpd_ks_d_max
        ),
        "COLLAPSE_COVERAGE": bool(np.isfinite(cov_frac) and cov_frac >= coverage_min),
        "TRUTH_IN_DOMAIN": bool(
            np.isfinite(frac_out_of_domain) and frac_out_of_domain <= out_of_domain_max
        ),
        "SCORES_DEFINED": bool(np.isfinite(frac_undef) and frac_undef <= undefined_max),
        "TRUTH_DENSITY_POSITIVE": bool(
            frac_truth_density_positive == 1.0 and n_truth_density_nan == 0
        ),
    }
    if gate:
        verdicts["GATE_SCALE"] = bool(gate_eligible)
    overall = all(verdicts.values())

    summary = {
        "run_dir": os.path.abspath(run_dir),
        "n_scored": n_total,
        "n_generated": n_gen,
        "gate_requested": bool(gate),
        "gate_eligible": bool(gate_eligible),
        "gate_ineligible_reasons": gate_reasons,
        "n_neo_true": int(np.sum(y)),
        "weighted_neo_fraction": float(np.sum(w * y) / np.sum(w)) if n_total else float("nan"),
        "importance_weights": {"neo": w_neo, "other": w_other},
        # Three DISJOINT counts summing to n_scored (see the partition assert):
        #   n_likelihood_defined_p  finite p from a real Lambda ratio
        #   n_policy_assigned_p     finite p = 1.0 from the D-19 triage policy
        #   n_undefined_p           nan p (empty region, or a 0/0 whose
        #                           mechanism is not the faint-end wall)
        # `frac_undefined_p` is the SCORES_DEFINED criterion's input and is
        # (n_undefined + n_policy) / n_scored: neither category produced a
        # usable likelihood score.  It is therefore NOT n_undefined_p /
        # n_scored, and the summary text spells that out.
        "n_undefined_p": n_undef,
        "n_policy_assigned_p": n_policy,
        "n_likelihood_defined_p": n_defined,
        "frac_undefined_p": frac_undef,
        "frac_policy_assigned_p": frac_policy,
        "frac_likelihood_defined_p": (
            n_defined / n_total if n_total else float("nan")
        ),
        # A policy assignment on a true non-NEO is a policy false positive, not
        # a model failure, so it is deliberately kept out of
        # `impossible_events` below (which is about the likelihood scores) and
        # reported here instead.
        "n_policy_assigned_on_true_non_neo": int(
            np.sum(policy_assigned & ~y)
        ),
        "generation": gen.counters,
        "generation_config": gen.config,
        "scoring_config": asdict(scfg) if scfg is not None else {},
        "reliability_bins": bins,
        "reliability": rel,
        "reliability_bootstrap": boot,
        "auc": auc,
        "impossible_events": {
            "n_true_neo_with_p_zero": int(np.sum(y & defined & (p <= _P_VAR_FLOOR))),
            "n_true_other_with_p_one": int(
                np.sum(~y & defined & (p >= 1.0 - _P_VAR_FLOOR))
            ),
        },
        "collapse": {
            "hpd_ks_D": ks_d,
            "hpd_ks_p": ks_p,
            "hpd_n_eff": ks_neff,
            "hpd_ks_D_max_allowed": hpd_ks_d_max,
            "hpd_exact_ks_D": ks_d_ex,
            "hpd_exact_ks_p": ks_p_ex,
            "hpd_estimator_gap_D": hpd_gap,
            "frac_truth_density_positive": frac_truth_density_positive,
            "n_truth_density_nan": n_truth_density_nan,
            "frac_truth_density_zero": frac_zero_truth,
            "hpd_ks_D_excluding_zero_density_truth": ks_d_nz,
            "hpd_ks_p_excluding_zero_density_truth": ks_p_nz,
            "hpd_n_eff_excluding_zero_density_truth": ks_neff_nz,
            "coverage_frac_truth_in_AR": cov_frac,
            "coverage_tolerance_au_per_day": _AR_OUTSIDE_TOL,
            "max_outside_au_per_day": _stat(outside, np.max),
            "median_mass_within_factor2_rho": _stat(coll, np.median),
            "mean_mass_within_factor2_rho": _stat(coll, np.mean),
            "frac_tracklets_mass_within_factor2_gt_0p5": float(
                np.sum(w[np.isfinite(coll)] * (coll[np.isfinite(coll)] > 0.5))
                / np.sum(w[np.isfinite(coll)])
            ) if np.any(np.isfinite(coll)) else float("nan"),
        },
        "integration_domain": {
            "rho_min": rho_min_s,
            "rho_max": rho_max_s,
            "generator_rho_floor": gen.config.get("rho_floor"),
            "frac_truth_rho_above_rho_max": frac_above_max,
            "frac_truth_rho_below_rho_min": frac_below_min,
            "frac_truth_outside_domain": frac_out_of_domain,
            "max_allowed_outside_domain": out_of_domain_max,
        },
        "ar_edge_slice": edge,
        "diagnostics_2_6": {
            "median_ar_frac_H_above_Hmax": _stat(ar_h, np.median),
            "mean_ar_frac_H_above_Hmax": _stat(ar_h, np.mean),
            "frac_tracklets_ar_frac_H_above_gt_1e-3": _frac_gt(ar_h, 1e-3),
            "frac_tracklets_ar_frac_H_above_gt_1e-2": _frac_gt(ar_h, 1e-2),
            "frac_tracklets_ar_frac_H_above_gt_1e-1": _frac_gt(ar_h, 1e-1),
            "median_ar_frac_H_above_and_q_above": _stat(ar_hq, np.median),
            "frac_tracklets_ar_frac_H_above_q_above_gt_1e-2": _frac_gt(ar_hq, 1e-2),
            "max_posterior_mass_q_above_H_above": _stat(mass_hq, np.max),
            # The boundary these fractions are measured against, so a saved
            # summary is self-describing: 25.5 for the D-4 pair, h_extend_to
            # with the D-19 extension on.  A number from a run at 25.5 is not
            # comparable with one from a run at 30.
            "h_max_model": _h_max_of_config(scfg),
        },
        "timing": {
            "mean_s_per_tracklet": _stat(sc["elapsed_s"], np.mean),
            "total_cpu_s": _stat(sc["elapsed_s"], np.sum),
            "mean_nodes": _stat(sc["n_nodes"], np.mean),
        },
        "verdicts": verdicts,
        "overall_pass": bool(overall),
    }

    text = _summary_text(summary)
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(os.path.join(run_dir, "summary.txt"), "w") as fh:
        fh.write(text)
    if verbose:
        print(text)

    if plots and n_defined > 0:
        _make_plots(run_dir, gen, sc, bins, fpr, tpr, auc, summary)
    elif plots and verbose:
        print("[analyze] no tracklet has a finite P(NEO); plots skipped.")

    return summary


def _h_max_of_config(scfg) -> float | None:
    """H_max^model implied by a scoring config, without reparsing the model.

    Mirrors ``classifier.h_support_end`` for the configurations this harness
    can build: ``h_extend_to`` when set (D-19), else the digest2 support end
    25.5 (D-4).  The harness never builds a D-14 clamped pair — a clamped
    component cannot be sampled, so it has no §4.1 generator — so the ``inf``
    case does not arise here.  Returns ``None`` when the config is unknown, so
    the summary text can fall back to the unlabelled wording.
    """
    if scfg is None:
        return None
    ext = getattr(scfg, "h_extend_to", None)
    if ext is not None:
        return float(ext)
    return float(H_EDGES[-1])


#: A truth that is a bound orbit MUST lie inside the admissible rho_dot
#: interval at its own rho; anything beyond this many AU/day outside is a real
#: alarm, not a rounding artifact.  Physical, so it does not move with
#: n_rhodot (a cell-width criterion weakens 4x going 200 -> 50 columns).
_AR_OUTSIDE_TOL = 1e-9

#: Tracklets whose truth sits within this fraction of the admissible interval's
#: width of either edge form the DESIGN §4.1 "admissible-region edge" slice.
AR_EDGE_MARGIN = 0.02


def ar_edge_slice(gen: "Generation", sc: dict, margin_max: float = AR_EDGE_MARGIN) -> dict:
    """DESIGN §4.1 targeted slice: tracklets whose truth sits near the AR edge.

    The admissible region's boundary is where the implied orbit is marginally
    bound, i.e. where the energy constraint of DESIGN §2.5 is about to bite.
    It is also where the quadrature is least comfortable: the rho_dot interval
    is being clipped, the Jacobian is worst-conditioned, and any error in the
    boundary itself shows up here first.  Reporting coverage/HPD/reliability
    for this subset separately keeps a boundary bug from being diluted by the
    99% of tracklets that sit comfortably inside.
    """
    margin = np.asarray(sc["rhodot_margin_frac"], dtype=float)
    idx = sc["index"].astype(int)
    w = gen.weight[idx].astype(float)
    y = gen.is_neo[idx].astype(bool)
    p = np.asarray(sc["p_neo"], dtype=float)
    sel = np.isfinite(margin) & (margin <= float(margin_max))
    n_sel = int(np.sum(sel))
    out = {
        "margin_max": float(margin_max),
        "n": n_sel,
        "frac_of_run": float(n_sel / margin.size) if margin.size else float("nan"),
        "n_neo": int(np.sum(y & sel)),
    }
    if n_sel == 0:
        return out
    outside = np.asarray(sc["rhodot_outside_au_per_day"], dtype=float)[sel]
    hp = np.asarray(sc["hpd_rank"], dtype=float)[sel]
    ok = np.isfinite(hp)
    d, pv, neff = ks_uniform(hp[ok], w[sel][ok])
    out.update(
        coverage=float(
            np.sum(w[sel][np.isfinite(outside) & (outside <= _AR_OUTSIDE_TOL)])
            / np.sum(w[sel])
        ),
        hpd_ks_D=d,
        hpd_ks_p=pv,
        hpd_n_eff=neff,
        median_mass_within_factor2_rho=_stat(
            np.asarray(sc["collapse_frac_rho2"], dtype=float)[sel], np.median
        ),
        mean_p_neo=_stat(p[sel], np.mean),
        median_margin=_stat(margin[sel], np.median),
    )
    return out


def _score_config_of(run_dir: str) -> "ScoreConfig | None":
    """The ScoreConfig a run was scored with, from any chunk's stamp sidecar."""
    path = os.path.join(run_dir, "score_config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return ScoreConfig(**json.load(fh))
    except Exception:  # pragma: no cover - corrupt sidecar
        return None


def _frac_gt(x: np.ndarray, thr: float) -> float:
    good = np.isfinite(x)
    if not np.any(good):
        return float("nan")
    return float(np.mean(x[good] > thr))


def _stat(x: np.ndarray, fn) -> float:
    """nan-safe reducer that never warns on an all-NaN / empty input."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(fn(x)) if x.size else float("nan")


def _summary_text(s: dict) -> str:
    L = []
    A = L.append
    A("=" * 78)
    A("ds42 WP6 — self-consistency validation (DESIGN §4.1, v0 gate)")
    A("=" * 78)
    cfgn = s.get("generation_config", {})
    cfgs = s.get("scoring_config", {})
    A(f"run                : {s['run_dir']}")
    A(
        f"generator          : N={cfgn.get('n')} seed={cfgn.get('seed')} "
        f"variant={cfgn.get('variant')} V in [{cfgn.get('v_min')}, {cfgn.get('v_max')}) "
        f"epoch {cfgn.get('epoch_start')} +{cfgn.get('epoch_days')} d "
        f"rho_floor={cfgn.get('rho_floor')} "
        f"h_interpolation={cfgn.get('h_interpolation', 'none')} "
        f"h_extend_to={cfgn.get('h_extend_to', None)}"
    )
    if cfgs:
        # h_interpolation and h_extend_to are printed on BOTH lines on purpose.
        # score() hard fails when either disagrees, so they always match here —
        # but a reader of a saved summary.txt has to be able to see *which*
        # model the run used without opening the JSON, and printing them once
        # would leave it ambiguous which side they described.
        A(
            f"scorer             : grid ({cfgs.get('n_rho')}, {cfgs.get('n_rhodot')}) "
            f"rho in [{cfgs.get('rho_min')}, {cfgs.get('rho_max')}] AU "
            f"h_interpolation={cfgs.get('h_interpolation', 'none')} "
            f"h_extend_to={cfgs.get('h_extend_to', None)}"
        )
    A(f"scored             : {s['n_scored']} of {s.get('n_generated')} generated  "
      f"(true NEOs: {s['n_neo_true']}, weighted NEO fraction "
      f"{_fmt_frac(s['weighted_neo_fraction'])})")
    iw = s.get("importance_weights", {})
    if iw and (iw.get("neo") != 1.0 or iw.get("other") != 1.0):
        A(f"importance weights : NEO {iw.get('neo'):.4g}, other {iw.get('other'):.4g} "
          f"(ENRICHED run — not a gate run)")
    # Three DISJOINT categories, always printed (including the 0 cases): a
    # reader of a saved summary must be able to tell "no policy assignments
    # happened" from "this summary predates the policy".
    _np_ = s.get("n_policy_assigned_p")
    if _np_ is None:
        A(f"undefined P(NEO)   : {s['n_undefined_p']} "
          f"({_fmt_frac(s['frac_undefined_p'])})")
        A("policy-assigned    : (not recorded — summary predates D-19)")
    else:
        A(f"likelihood-defined : {s.get('n_likelihood_defined_p')} "
          f"({_fmt_frac(s.get('frac_likelihood_defined_p', float('nan')))})"
          "   <- every statistic below is computed on THESE only")
        A(f"policy-assigned    : {_np_} "
          f"({_fmt_frac(s.get('frac_policy_assigned_p', float('nan')))})"
          "   [D-19: p=1.0 by triage policy; both Lambdas exactly 0 and the "
          "whole region at/above the H support end; log_lr = nan — a defined "
          "number, NOT a posterior]")
        if _np_:
            A("                     ^ EXCLUDED from the reliability diagram, "
              "the global-Z, the odds fit and the ROC: they carry no "
              "likelihood information at all.")
            _fp = s.get("n_policy_assigned_on_true_non_neo")
            if _fp is not None:
                A(f"                     of which on a true NON-NEO (policy "
                  f"false positives): {_fp}")
        A(f"undefined P(NEO)   : {s['n_undefined_p']} "
          f"({_fmt_frac(s['n_undefined_p'] / s['n_scored']) if s['n_scored'] else '-'})"
          "   [nan: empty region, or a 0/0 that is not the faint-end wall]")
        A(f"  SCORES_DEFINED reads (undefined + policy-assigned) / scored = "
          f"{_fmt_frac(s['frac_undefined_p'])}: neither category produced a "
          "usable likelihood score.")
    if s.get("gate_eligible"):
        A("gate eligibility   : ELIGIBLE (full mixture catalogue at gate scale)")
    else:
        A("gate eligibility   : NOT a gate run — "
          + "; ".join(s.get("gate_ineligible_reasons", ["(unknown)"])))
    A("")

    # ---- the headline tripwire, first ------------------------------------
    r = s["reliability"]
    A("-- GLOBAL NEO COUNT (the cheapest tripwire for an odds bias) ---------------")
    A(f"observed {_fmt_frac(r['observed_neo_count'])}  vs  expected "
      f"{_fmt_frac(r['expected_neo_count'])}   ->   Z_tot = "
      f"{_fmt_frac(r['global_z'])}")
    A(f"  bootstrap 99th pct = {_fmt_frac(r['global_z_threshold'])}, "
      f"p = {_fmt_frac(r['global_z_p_value'])}  -> "
      f"{'PASS' if r['global_pass'] else 'FAIL'}")
    o = r.get("odds_scale", {})
    if o:
        rail = "  [RAILED — likelihood monotone, read as a one-sided bound]" \
            if o.get("railed") else ""
        A(f"  fitted global odds scale c_hat = {_fmt_frac(o.get('c_hat', float('nan')))} "
          f"[{_fmt_frac(o.get('lo', float('nan')))}, "
          f"{_fmt_frac(o.get('hi', float('nan')))}] (95%){rail}")
        A(f"  (fitted on {o.get('n_used', 0)} tracklets; "
          f"{o.get('n_excluded_certain', 0)} scored exactly 0 or 1 and carry no "
          "information about a multiplicative odds scale)")
        A("  (c = 1 is perfect; c > 1 means P(NEO) is understated everywhere.)")
    A("")

    A("-- generation bookkeeping (DESIGN §4.1 unbound-draw filtering) --------------")
    per = s.get("generation", {}).get("per_class", {})
    A(f"{'class':>6}  {'drawn':>9} {'unbound':>9} {'unb.frac':>9} "
      f"{'V-cut acc':>10} {'accepted':>9} {'consistent':>10}")
    for k in ("NEO", "other"):
        d = per.get(k, {})
        if not d:
            continue
        cons = d.get("drawn", 0) == (d.get("accepted", 0) + d.get("rejected", 0))
        A(
            f"{k:>6}  {d.get('drawn', 0):>9d} {d.get('unbound', 0):>9d} "
            f"{_fmt_frac(d.get('unbound_fraction', float('nan'))):>9} "
            f"{_fmt_frac(d.get('acceptance_rate_after_bound_cut', float('nan'))):>10} "
            f"{d.get('accepted', 0):>9d} {str(cons):>10}"
        )
    A(f"mixture P(NEO) = {_fmt_frac(s.get('generation', {}).get('p_mixture_neo', float('nan')))}"
      f"   drawn with P = {_fmt_frac(s.get('generation', {}).get('p_draw_neo', float('nan')))}")
    A("")

    A("-- reliability diagram (equal EXPECTED-NEO-COUNT bins) ---------------------")
    A(f"{'bin':>3} {'n':>7} {'p range':>21} {'<p>':>10} {'exp NEO':>9} "
      f"{'obs NEO':>9} {'z':>8}")
    for i, b in enumerate(s["reliability_bins"]):
        A(
            f"{i:>3} {b['n']:>7d} "
            f"[{b['p_lo']:.3e},{b['p_hi']:.3e}] "
            f"{b['p_mean']:>10.4g} {b['expected_events']:>9.3f} "
            f"{b['observed_events']:>9.3f} {b['z']:>8.2f}"
        )
    boot = s.get("reliability_bootstrap", {})
    A(f"max|z| = {_fmt_frac(r['max_abs_z'])}  vs bootstrap 99th pct "
      f"{_fmt_frac(r['max_abs_z_threshold'])} "
      f"(p = {_fmt_frac(r['max_abs_z_p_value'])}, B = {boot.get('n_boot', 0)})"
      f"  -> {'PASS' if r['binned_pass'] else 'FAIL'}")
    A(f"[display only, NOT in the verdict] chi2 = "
      f"{r['chi2_display_only']:.2f} on {r['chi2_dof']} bins, nominal p = "
      f"{_fmt_frac(r['chi2_p_value_display_only'])}")
    A("  Both arms are thresholded by parametric bootstrap (labels re-simulated")
    A("  from the generator's own null on the observed p vector).  Analytic levels")
    A("  are not usable here: a '2 sigma in any of 10 bins' rule false-alarms at")
    A("  ~43% and the chi2's nominal 1% is really ~6% at this base rate.")
    if r["n_bins_low_variance"]:
        A(f"  NOTE: {r['n_bins_low_variance']} bin(s) have Poisson-binomial variance "
          f"< 5 — their z is far from normal, which is exactly why the threshold")
        A("  is bootstrapped rather than taken from a table.")
    A("")

    A("-- ranking ------------------------------------------------------------------")
    A(f"ROC AUC            : {_fmt_frac(s['auc'])}")
    ie = s["impossible_events"]
    A(f"impossible events  : true NEO scored P=0: {ie['n_true_neo_with_p_zero']}; "
      f"true other scored P=1: {ie['n_true_other_with_p_one']}")
    A("")

    A("-- collapse tests (DESIGN §4.1) --------------------------------------------")
    c = s["collapse"]
    A(f"truth inside the admissible region : "
      f"{_fmt_frac(c['coverage_frac_truth_in_AR'])} "
      f"(tolerance {c['coverage_tolerance_au_per_day']:g} AU/day; worst excursion "
      f"{_fmt_frac(c['max_outside_au_per_day'])} AU/day)")
    A(f"posterior mass within factor 2 of true rho: median "
      f"{_fmt_frac(c['median_mass_within_factor2_rho'])}, mean "
      f"{_fmt_frac(c['mean_mass_within_factor2_rho'])}, "
      f"frac(>0.5) {_fmt_frac(c['frac_tracklets_mass_within_factor2_gt_0p5'])}")
    A(f"HPD rank vs Uniform(0,1)  [criterion: BOTH estimators D <= "
      f"{c['hpd_ks_D_max_allowed']:g}]")
    A(f"  nearest-node    D = {_fmt_frac(c['hpd_ks_D'])}  "
      f"(p = {_fmt_frac(c['hpd_ks_p'])}, n_eff = {c['hpd_n_eff']:.0f})")
    A(f"  exact-at-truth  D = {_fmt_frac(c['hpd_exact_ks_D'])}  "
      f"(p = {_fmt_frac(c['hpd_exact_ks_p'])})")
    A(f"  estimator gap   = {_fmt_frac(c['hpd_estimator_gap_D'])}  "
      "<- the grid-adequacy readout: the two bias in OPPOSITE directions and")
    A("     converge under refinement, so a large gap means the grid, not the math.")
    A(f"  nearest node in an EMPTY histogram bin for "
      f"{_fmt_frac(c['frac_truth_density_zero'])} of tracklets; excluding those, "
      f"D = {_fmt_frac(c['hpd_ks_D_excluding_zero_density_truth'])}")
    A("  The criterion is on the KS statistic D, not its p-value — at gate N any")
    A("  finite grid fails a p-value gate on discretisation bias alone.")
    A(f"  integrand > 0 at the truth for "
      f"{_fmt_frac(c['frac_truth_density_positive'])} of tracklets "
      f"(must be exactly 1; NaN returns count as failures: "
      f"{c['n_truth_density_nan']})")
    A("")

    A("-- integration domain ------------------------------------------------------")
    dm = s["integration_domain"]
    A(f"rho in [{dm['rho_min']:g}, {dm['rho_max']:g}] AU; generator floor "
      f"{dm['generator_rho_floor']}")
    A(f"truth rho > rho_max : {_fmt_frac(dm['frac_truth_rho_above_rho_max'])}   "
      f"truth rho < rho_min : {_fmt_frac(dm['frac_truth_rho_below_rho_min'])}")
    A(f"total outside the integration domain: "
      f"{_fmt_frac(dm['frac_truth_outside_domain'])} "
      f"(max allowed {dm['max_allowed_outside_domain']:g}) — these tracklets have a")
    A("  truth the classifier never integrates over; reported here rather than")
    A("  absorbed into the coverage number above.")
    A("")

    A("-- targeted slice: admissible-region edge (DESIGN §4.1) --------------------")
    e = s.get("ar_edge_slice", {})
    A(f"tracklets with truth within {e.get('margin_max', float('nan')):g} of the "
      f"rho_dot interval edge: {e.get('n', 0)} "
      f"({_fmt_frac(e.get('frac_of_run', float('nan')))} of the run, "
      f"{e.get('n_neo', 0)} NEO)")
    if e.get("n", 0):
        A(f"  coverage {_fmt_frac(e.get('coverage', float('nan')))}, "
          f"HPD D = {_fmt_frac(e.get('hpd_ks_D', float('nan')))} "
          f"(p = {_fmt_frac(e.get('hpd_ks_p', float('nan')))}), "
          f"median mass within factor 2 = "
          f"{_fmt_frac(e.get('median_mass_within_factor2_rho', float('nan')))}")
    A("")

    A("-- support-truncation diagnostics (DESIGN §2.6) ----------------------------")
    d = s["diagnostics_2_6"]
    _hm = d.get("h_max_model")
    _hm_s = "H_max^model" if _hm is None else f"H_max^model = {_hm:g}"
    A(f"(b) admissible-region measure fraction with implied H > {_hm_s}:")
    A(f"    median {_fmt_frac(d['median_ar_frac_H_above_Hmax'])}, "
      f"mean {_fmt_frac(d['mean_ar_frac_H_above_Hmax'])}; "
      f"frac of tracklets > 1e-3 / 1e-2 / 1e-1 = "
      f"{_fmt_frac(d['frac_tracklets_ar_frac_H_above_gt_1e-3'])} / "
      f"{_fmt_frac(d['frac_tracklets_ar_frac_H_above_gt_1e-2'])} / "
      f"{_fmt_frac(d['frac_tracklets_ar_frac_H_above_gt_1e-1'])}")
    A(f"    restricted to q > 1.3 AU (Mars-crosser corner): median "
      f"{_fmt_frac(d['median_ar_frac_H_above_and_q_above'])}, "
      f"frac > 1e-2 = {_fmt_frac(d['frac_tracklets_ar_frac_H_above_q_above_gt_1e-2'])}")
    A(f"(a) posterior mass fraction in {{q > 1.3, H > H_max^model}}: max over tracklets "
      f"= {_fmt_frac(d['max_posterior_mass_q_above_H_above'])}")
    A("    (identically 0 whenever both components share one H support end — every")
    A("     v0 configuration — since the density is zero above it by construction;")
    A("     flag (b) is the operative one.)")
    A("")

    A("-- timing -------------------------------------------------------------------")
    t = s["timing"]
    A(f"mean {t['mean_s_per_tracklet']:.3f} s/tracklet, {t['mean_nodes']:.0f} nodes, "
      f"total CPU {t['total_cpu_s']:.0f} s")
    A("")
    A("-- VERDICT ------------------------------------------------------------------")
    for k, v in s["verdicts"].items():
        A(f"  {k:<24} {'PASS' if v else 'FAIL'}")
    A(f"  {'OVERALL':<24} {'PASS' if s['overall_pass'] else 'FAIL'}")
    if not s.get("gate_eligible"):
        A("  (this run is NOT gate-eligible; the verdict above is indicative only)")
    A("=" * 78)
    return "\n".join(L) + "\n"


def _make_plots(run_dir, gen, sc, bins, fpr, tpr, auc, summary) -> None:
    idx = sc["index"].astype(int)
    y = gen.is_neo[idx].astype(bool)
    p = sc["p_neo"].astype(float)

    # 1. reliability -------------------------------------------------------
    pm = np.array([b["p_mean"] for b in bins])
    ob = np.array([b["obs_frac"] for b in bins])
    er = np.array([b["sigma_frac"] for b in bins])
    zz = np.array([b["z"] for b in bins])
    exp_n = np.array([b["expected_events"] for b in bins])
    obs_n = np.array([b["obs_frac"] * b["sum_w"] for b in bins])
    sig_n = np.array([math.sqrt(max(b["poisson_binomial_var"], 0.0)) for b in bins])

    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.1])
    ax = fig.add_subplot(gs[0, 0])
    axc = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.errorbar(pm, ob, yerr=er, fmt="o", ms=5, capsize=3, label="equal-expected-NEO-count bins")
    ax.set_ylabel("observed NEO fraction")
    ax.set_xlabel("mean P(NEO) in bin")
    ax.set_title("reliability (linear)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # In pure-mixture mode the NEO base rate is ~0.5%, so every bin but the
    # last sits in the bottom-left corner of the linear panel and the diagram
    # reads as a single dot.  The same information as expected-vs-observed
    # NEO counts, on log axes, is legible in that rare-event regime.
    pos_exp = exp_n[exp_n > 0]
    lo = float(pos_exp.min()) / 3.0 if pos_exp.size else 1e-3
    hi = float(max(exp_n.max(), obs_n.max(), 1.0)) * 3.0
    floor = lo  # where a "0 observed" bin is drawn
    axc.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect calibration")
    seen = obs_n > 0
    if np.any(seen):
        axc.errorbar(
            exp_n[seen], obs_n[seen], yerr=sig_n[seen], fmt="o", ms=5, capsize=3,
            label="bins with a NEO",
        )
    if np.any(~seen):
        axc.plot(
            exp_n[~seen], np.full(int((~seen).sum()), floor), "v", ms=6,
            color="C1", ls="none", label="bins with none observed",
        )
    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.set_xlim(lo, hi)
    axc.set_ylim(lo, hi)
    axc.legend(fontsize=7, loc="upper left")
    axc.set_xlabel("expected NEO count in bin  (sum w p)")
    axc.set_ylabel("observed NEO count  (sum w y)")
    axc.set_title("reliability (counts, log-log; v = zero observed)")

    ax2.axhline(0, color="k", lw=1)
    for lev in (2, -2):
        ax2.axhline(lev, color="r", ls="--", lw=0.8)
    ax2.plot(np.arange(len(bins)), zz, "o-", ms=5, lw=0.8)
    thr = summary["reliability"].get("max_abs_z_threshold", float("nan"))
    if np.isfinite(thr):
        for lev in (thr, -thr):
            ax2.axhline(lev, color="C2", ls=":", lw=1.2)
        ax2.text(0.01, 0.9, f"bootstrap 99th pct = {thr:.2f} (dotted)",
                 transform=ax2.transAxes, fontsize=7, color="C2")
    ax2.set_xlabel("bin index (ordered by P(NEO))")
    ax2.set_ylabel("z")
    lim = max(2.6, float(np.nanmax(np.abs(zz))) * 1.25 if np.any(np.isfinite(zz)) else 2.6,
              (thr * 1.25 if np.isfinite(thr) else 0.0))
    ax2.set_ylim(-lim, lim)
    r = summary["reliability"]
    fig.suptitle(
        f"Reliability — max|z| = {r['max_abs_z']:.2f} "
        f"(bootstrap 99th pct {r['max_abs_z_threshold']:.2f}, "
        f"p = {r['max_abs_z_p_value']:.3g});  "
        f"Z_tot = {r['global_z']:.2f} "
        f"(thr {r['global_z_threshold']:.2f});  N = {summary['n_scored']}"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "reliability.png"), dpi=130)
    plt.close(fig)

    # 2. score distributions ----------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    edges = np.linspace(0, 1, 41)
    for mask, lab in ((~y & np.isfinite(p), "true other"), (y & np.isfinite(p), "true NEO")):
        if np.any(mask):
            a1.hist(p[mask], bins=edges, histtype="step", label=lab, log=True)
    a1.set_xlabel("P(NEO)")
    a1.set_ylabel("tracklets")
    a1.legend(fontsize=8)
    ll = sc["log_lr"].astype(float)
    fin = np.isfinite(ll)
    if np.any(fin):
        lo, hi = np.percentile(ll[fin], [0.5, 99.5])
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0
        be = np.linspace(lo, hi, 60)
        for mask, lab in ((fin & ~y, "true other"), (fin & y, "true NEO")):
            if np.any(mask):
                a2.hist(ll[mask], bins=be, histtype="step", label=lab, log=True)
        a2.legend(fontsize=8)
    a2.set_xlabel("ln(Lambda_NEO / Lambda_other)")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "scores.png"), dpi=130)
    plt.close(fig)

    # 3. ROC ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(fpr, tpr, lw=1.5)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC — AUC = {auc:.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "roc.png"), dpi=130)
    plt.close(fig)

    # 4. collapse ----------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    coll = sc["collapse_frac_rho2"]
    coll = coll[np.isfinite(coll)]
    if coll.size:
        a1.hist(coll, bins=np.linspace(0, 1, 41), histtype="stepfilled", alpha=0.7)
    a1.set_xlabel("posterior mass within factor 2 of true rho")
    a1.set_ylabel("tracklets")
    redges = np.linspace(0, 1, 21)
    n_ref = 0
    for key, lab in (("hpd_rank", "nearest node"), ("hpd_rank_exact", "exact at truth")):
        v = np.asarray(sc[key], dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            a2.hist(v, bins=redges, histtype="step", label=lab)
            n_ref = max(n_ref, v.size)
    if n_ref:
        a2.axhline(n_ref / 20.0, color="r", ls="--", lw=1, label="uniform")
        a2.legend(fontsize=8)
    a2.set_xlabel("HPD rank of the truth")
    c = summary["collapse"]
    a2.set_title(
        f"KS D = {c['hpd_ks_D']:.3g} (nearest) / {c['hpd_exact_ks_D']:.3g} (exact)"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "collapse.png"), dpi=130)
    plt.close(fig)

    # 5. §2.6 diagnostics ---------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for arr, lab in (
        (sc["ar_frac_h_above"], "H > H_max (all q)"),
        (sc["ar_frac_h_above_q_above"], "H > H_max and q > 1.3"),
    ):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            a1.hist(
                np.log10(arr), bins=40, histtype="step", label=f"{lab} (n={arr.size})"
            )
    a1.set_xlabel("log10 admissible-region measure fraction")
    a1.set_ylabel("tracklets")
    a1.set_title("DESIGN §2.6 (b) truncation diagnostic")
    a1.legend(fontsize=8)
    mg = np.asarray(sc["rhodot_margin_frac"], dtype=float)
    mg = mg[np.isfinite(mg)]
    if mg.size:
        a2.hist(np.clip(mg, 0.0, 0.5), bins=np.linspace(0, 0.5, 51),
                histtype="stepfilled", alpha=0.7)
    a2.axvline(AR_EDGE_MARGIN, color="r", ls="--", lw=1,
               label=f"edge slice (<= {AR_EDGE_MARGIN:g})")
    a2.set_xlabel("truth margin to the admissible rho_dot edge  [fraction of width]")
    a2.set_title("admissible-region coverage / edge proximity")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "diagnostics.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# DESIGN §4.1 targeted slice: opposition rate scan
# --------------------------------------------------------------------------


def rate_scan(
    model: str = DEFAULT_MODEL,
    variant: str = "All",
    v_mag: float = 21.0,
    epoch: str = "2026-06-15T00:00:00",
    rate_min_deg_day: float = 0.02,
    rate_max_deg_day: float = 5.0,
    n_rate: int = 60,
    n_rho: int = N_RHO_DEFAULT,
    n_rhodot: int = N_RHODOT_DEFAULT,
    rho_min: float = 0.01,
    rho_max: float = 100.0,
    out_dir: str | None = None,
    verbose: bool = True,
) -> dict:
    """Sweep apparent sky rate at fixed V, at opposition (DESIGN §4.1).

    The controlled experiment behind the whole method: at opposition an
    object's apparent rate is a nearly monotonic proxy for its distance, so
    P(NEO) versus rate is the NEO/MBA separation curve with every other
    variable held fixed.  It is the localisation tool — when a self-consistency
    run goes wrong, this says *where* in observable space the posterior moved,
    which a scalar reliability statistic never can.

    Geometry: geocentric observer at ``epoch``; the field is placed at the
    anti-solar point (exact opposition, phase angle 0, so implied H is at its
    maximum for a given V — the DESIGN §2.6 truncation corner is at its widest
    here); motion is along ecliptic longitude, i.e. the retrograde-to-prograde
    axis that distance actually maps onto.

    Reference marks: the rate a circular, coplanar orbit at heliocentric
    distance r would show at opposition,
    ``rate = (n_earth - n_obj) * (1 AU) / rho`` with ``rho = r - 1``, which is
    where a main-belt object (r ~ 2.5-3.2 AU) and a 1.3 AU NEO respectively
    land.
    """
    neo, other = Digest2Histogram.neo_other_pair(model, variant)
    t = Time(epoch, scale="tdb")
    R_obs, V_obs = observer_state(t)
    R_obs = np.asarray(R_obs, dtype=float).reshape(3)
    V_obs = np.asarray(V_obs, dtype=float).reshape(3)

    # Anti-solar direction = -R_obs (the observer's heliocentric position),
    # expressed as (ra, dec) in the equatorial frame the transforms use.
    u = -R_obs / np.linalg.norm(R_obs)
    dec = float(np.arcsin(np.clip(u[2], -1.0, 1.0)))
    ra = float(np.arctan2(u[1], u[0]) % (2.0 * np.pi))

    rates_deg = np.geomspace(float(rate_min_deg_day), float(rate_max_deg_day),
                             int(n_rate))
    rates = np.deg2rad(rates_deg)

    rows = []
    for rate in rates:
        # Motion along ecliptic longitude: at opposition the anti-solar point
        # lies on the ecliptic, so d(ra)/dt with the cos(dec) factor removed is
        # the natural parameterisation. att.ra_rate is RAW d(ra)/dt.
        att = Attributable(
            ra=ra, dec=dec, ra_rate=float(rate / max(np.cos(dec), 1e-12)),
            dec_rate=0.0, epoch=t, v_mag=float(v_mag),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = classify(
                att, neo, other, R_obs=R_obs, V_obs=V_obs,
                rho_min=rho_min, rho_max=rho_max,
                n_rho=n_rho, n_rhodot=n_rhodot,
            )
        mass = np.asarray(res.node_density_neo, dtype=float) + np.asarray(
            res.node_density_other, dtype=float
        )
        mass = np.where(np.isfinite(mass), mass, 0.0) * np.asarray(
            res.grid.weight, dtype=float
        )
        tot = float(mass.sum())
        rho_map = (
            float(res.grid.rho[int(np.argmax(mass))]) if tot > 0 else float("nan")
        )
        rows.append(
            {
                "rate_deg_per_day": float(np.rad2deg(rate)),
                "p_neo": float(res.p_neo),
                "log_lr": float(res.log_lr),
                "rho_map": rho_map,
                "n_nodes": int(res.n_nodes),
                "ar_frac_H_above": float(res.measure_fraction_H_above(
                    float(other.support["H"][1])
                )),
            }
        )
        if verbose:
            print(f"  [rate-scan] {np.rad2deg(rate):7.3f} deg/day -> "
                  f"P(NEO) = {res.p_neo:.4g}", flush=True)

    out = {
        "config": {
            "model": model, "variant": variant, "v_mag": v_mag, "epoch": epoch,
            "n_rho": n_rho, "n_rhodot": n_rhodot,
            "rho_min": rho_min, "rho_max": rho_max,
            "ra": ra, "dec": dec,
        },
        "reference_rates_deg_per_day": _circular_opposition_rates(),
        # P(NEO) is undefined where BOTH components' support is exhausted
        # (DESIGN §2.6): at high rate and fixed V every admissible node implies
        # H > 25.5.  The scan is the tool that shows where in observable space
        # that happens, so the count is reported rather than hidden.
        "n_undefined_p_neo": int(sum(1 for r in rows if not np.isfinite(r["p_neo"]))),
        "scan": rows,
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "rate_scan.json"), "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        _plot_rate_scan(out, out_dir)
    return out


def _circular_opposition_rates() -> dict:
    """Apparent opposition rate of a circular coplanar orbit at r AU [deg/day].

    Relative angular speed of Earth and the object, projected at the
    topocentric distance rho = r - 1:  ``(n_E - n_obj) * 1 AU / rho`` where
    ``n = 0.9856 / r^1.5`` deg/day.  Retrograde (positive here) at opposition.
    """
    out = {}
    for label, r in (("NEO q=1.3", 1.3), ("Mars-crosser 1.7", 1.7),
                     ("inner belt 2.2", 2.2), ("outer belt 3.2", 3.2),
                     ("Jupiter Trojan 5.2", 5.2)):
        n_e, n_o = 0.98560912, 0.98560912 / r ** 1.5
        rho = r - 1.0
        out[label] = float((n_e - n_o) / rho)
    return out


def _plot_rate_scan(scan: dict, out_dir: str) -> None:
    rows = scan["scan"]
    x = np.array([r["rate_deg_per_day"] for r in rows])
    y = np.array([r["p_neo"] for r in rows])
    rho = np.array([r["rho_map"] for r in rows])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    a1.semilogx(x, y, "o-", ms=3, lw=1.2)
    a1.set_ylabel("P(NEO)")
    a1.set_ylim(-0.03, 1.03)
    a1.set_title(
        f"Opposition rate scan — V = {scan['config']['v_mag']}, "
        f"{scan['config']['epoch'][:10]}, geocentric"
    )
    for label, r in scan["reference_rates_deg_per_day"].items():
        for ax in (a1, a2):
            ax.axvline(r, color="0.7", ls=":", lw=1)
        a1.text(r, 1.01, label, rotation=90, fontsize=6, ha="right", va="top",
                color="0.4")
    a2.loglog(x, rho, "o-", ms=3, lw=1.2)
    a2.set_ylabel(r"MAP $\rho$  [AU]")
    a2.set_xlabel("apparent sky rate  [deg/day]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rate_scan.png"), dpi=130)
    plt.close(fig)


def _add_gen_args(ap):
    ap.add_argument("--n", type=int, default=1000, help="number of ACCEPTED tracklets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--variant", default="All", choices=("All", "Unk"))
    ap.add_argument("--v-max", type=float, default=24.0, help="faint observability limit")
    ap.add_argument("--v-min", type=float, default=14.0, help="bright (saturation) limit")
    ap.add_argument("--epoch-start", default="2026-01-01T00:00:00")
    ap.add_argument("--epoch-days", type=float, default=365.0)
    ap.add_argument("--rho-floor", type=float, default=0.01,
                    help="reject geometries with rho below this [AU]; defaults to "
                         "the scoring rho_min so the test population is contained "
                         "in the classifier's integration domain")
    ap.add_argument("--obscode", default=None, help="MPC code; default geocentre")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--neo-fraction", type=float, nargs="?", const=0.1, default=None,
                    help="DIAGNOSTIC ONLY (low-p amplifier): draw NEOs with this "
                         "probability and carry importance weights. Bare flag = 0.1. "
                         "The gate must NOT use it — and note it makes calibration "
                         "sensitivity WORSE at p in [0.2, 0.8)")


def _add_h_interpolation_arg(ap):
    """The population H reconstruction, shared by the generator and the scorer.

    ONE flag deliberately drives both sides: DESIGN §4.1's calibration argument
    is that the test population is drawn from the classifier's own prior, so a
    run that generated from the histogram and scored with the reconstruction
    (or the reverse) would not be a self-consistency test at all — it would be
    the §4.3 prior-misspecification experiment wearing the gate's clothes.
    ``ScoreConfig`` carries it into the chunk provenance stamp, so switching it
    invalidates cached chunks.
    """
    ap.add_argument(
        "--h-interpolation", default="none", choices=H_INTERPOLATIONS,
        help="population density in H: 'none' = piecewise constant (D-10 "
             "default), 'pchip' = mass-conserving monotone-cubic "
             "reconstruction. Applied to BOTH components and to BOTH the "
             "generator and the classifier.",
    )
    _add_h_extend_to_arg(ap)


def _add_h_extend_to_arg(ap):
    """The D-19 finite H extension, shared by the generator and the scorer.

    Same one-flag-drives-both discipline as ``--h-interpolation``, same
    provenance-stamp treatment, and the same hard failure in :func:`score` when
    a catalogue and a scoring config disagree.  What is *different* — and is the
    reason a gate run in this mode is meaningful at all — is that the extended
    pair is normalizable and samplable, so prior = generator = extended model
    holds exactly.  The D-14 clamp could never be run here (``sample()``
    raises); this mode can, and the gate then tests the extended model as a
    legitimate population rather than as an emulation device.

    Note that switching it on changes the *mixture fraction* the catalogue is
    drawn at (the NEO component's total count grows ~2.7x), so an extended
    catalogue is not comparable tracklet-for-tracklet with an unextended one —
    it is a different population.
    """
    ap.add_argument(
        "--h-extend-to", type=float, default=None, metavar="H",
        help="append a synthetic H bin [25.5, H) to every (q, e, i) cell "
             "continuing the top bin's per-unit-H density flatly [D-19]; "
             "density is hard zero above H. Canonical value 30. Applied to "
             "BOTH components and to BOTH the generator and the classifier. "
             "Mutually exclusive with the D-14 clamp; composes with "
             "--h-interpolation. Unlike the clamp this mode IS gate-eligible.",
    )


def _add_score_args(ap):
    ap.add_argument("--n-rho", type=int, default=N_RHO_DEFAULT)
    ap.add_argument("--n-rhodot", type=int, default=N_RHODOT_DEFAULT)
    ap.add_argument("--rho-min", type=float, default=0.01)
    ap.add_argument("--rho-max", type=float, default=100.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N (marks the run NON-GATE)")
    _add_h_interpolation_arg(ap)


def _add_analyze_args(ap):
    ap.add_argument("--bins", type=int, default=N_RELIABILITY_BINS,
                    help="equal-EXPECTED-NEO-COUNT reliability bins")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--hpd-ks-d-max", type=float, default=0.075)
    ap.add_argument("--boot", type=int, default=N_BOOTSTRAP,
                    help="parametric-bootstrap replicates for the reliability "
                         "thresholds")
    ap.add_argument("--boot-seed", type=int, default=20260901)
    ap.add_argument("--gate", action="store_true",
                    help="enforce gate eligibility (full mixture catalogue, "
                         "N >= --gate-min-n) as a verdict")
    ap.add_argument("--gate-min-n", type=int, default=10000)


def _gen_cfg(a) -> GenConfig:
    return GenConfig(
        n=a.n, seed=a.seed, model=a.model, variant=a.variant, v_max=a.v_max,
        v_min=a.v_min, epoch_start=a.epoch_start, epoch_days=a.epoch_days,
        rho_floor=a.rho_floor, obscode=a.obscode, batch=a.batch,
        neo_fraction=a.neo_fraction,
        h_interpolation=getattr(a, "h_interpolation", "none"),
        h_extend_to=getattr(a, "h_extend_to", None),
    )


def _score_cfg(a) -> ScoreConfig:
    return ScoreConfig(
        n_rho=a.n_rho, n_rhodot=a.n_rhodot, rho_min=a.rho_min,
        rho_max=a.rho_max, model=a.model, variant=a.variant,
        h_interpolation=getattr(a, "h_interpolation", "none"),
        h_extend_to=getattr(a, "h_extend_to", None),
    )


def _analyze_kwargs(a) -> dict:
    return dict(
        n_bins=a.bins, plots=not a.no_plots, hpd_ks_d_max=a.hpd_ks_d_max,
        n_boot=a.boot, boot_seed=a.boot_seed, gate=a.gate,
        gate_min_n=a.gate_min_n,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m validation.self_consistency",
        description="ds42 WP6 self-consistency validation harness (DESIGN §4.1).",
        epilog=(
            "The v0 gate is: generate --n 10000 (mixture mode, no --neo-fraction), "
            "score (defaults are the (1200, 50) grid), analyze --gate.  Export "
            "OMP_NUM_THREADS=1 first: the work is numpy-elementwise, and an "
            "unbounded BLAS pool per worker only wastes CPU.  Runs are resumable "
            "at chunk granularity (--resume); chunks carry both a provenance "
            "stamp and a content hash, so stale or corrupted ones are recomputed "
            "rather than reused."
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="draw a test population and save it")
    _add_gen_args(g)
    _add_h_interpolation_arg(g)   # `run` gets it from _add_score_args instead
    g.add_argument("--out", required=True)

    sc_ = sub.add_parser("score", help="classify a saved test population")
    sc_.add_argument("--run", required=True)
    sc_.add_argument("--model", default=DEFAULT_MODEL)
    sc_.add_argument("--variant", default="All", choices=("All", "Unk"))
    _add_score_args(sc_)

    a = sub.add_parser("analyze", help="analyse saved scores; plots + PASS/FAIL")
    a.add_argument("--run", required=True)
    _add_analyze_args(a)

    r = sub.add_parser("run", help="generate + score + analyze")
    _add_gen_args(r)
    _add_score_args(r)
    _add_analyze_args(r)
    r.add_argument("--out", required=True)

    rs = sub.add_parser(
        "rate-scan",
        help="DESIGN §4.1 targeted slice: P(NEO) vs apparent rate at opposition",
    )
    rs.add_argument("--out", required=True)
    rs.add_argument("--model", default=DEFAULT_MODEL)
    rs.add_argument("--variant", default="All", choices=("All", "Unk"))
    rs.add_argument("--v-mag", type=float, default=21.0)
    rs.add_argument("--epoch", default="2026-06-15T00:00:00")
    rs.add_argument("--rate-min", type=float, default=0.02, help="deg/day")
    rs.add_argument("--rate-max", type=float, default=5.0, help="deg/day")
    rs.add_argument("--n-rate", type=int, default=60)
    rs.add_argument("--n-rho", type=int, default=N_RHO_DEFAULT)
    rs.add_argument("--n-rhodot", type=int, default=N_RHODOT_DEFAULT)
    rs.add_argument("--rho-min", type=float, default=0.01)
    rs.add_argument("--rho-max", type=float, default=100.0)

    ae = sub.add_parser(
        "ar-edge",
        help="DESIGN §4.1 targeted slice: tracklets whose truth sits near the "
             "admissible-region boundary",
    )
    ae.add_argument("--run", required=True)
    ae.add_argument("--margin", type=float, default=AR_EDGE_MARGIN)

    args = ap.parse_args(argv)

    if args.cmd == "generate":
        gen = generate(_gen_cfg(args))
        save_generation(gen, args.out)
        print(json.dumps(gen.counters, indent=2, default=float))
        return 0

    if args.cmd == "score":
        score(args.run, _score_cfg(args), workers=args.workers, chunk=args.chunk,
              resume=args.resume, limit=args.limit)
        return 0

    if args.cmd == "analyze":
        summary = analyze(args.run, **_analyze_kwargs(args))
        return 0 if summary["overall_pass"] else 1

    if args.cmd == "run":
        gen = generate(_gen_cfg(args))
        save_generation(gen, args.out)
        score(args.out, _score_cfg(args), workers=args.workers, chunk=args.chunk,
              resume=args.resume, limit=args.limit)
        summary = analyze(args.out, **_analyze_kwargs(args))
        return 0 if summary["overall_pass"] else 1

    if args.cmd == "rate-scan":
        rate_scan(
            model=args.model, variant=args.variant, v_mag=args.v_mag,
            epoch=args.epoch, rate_min_deg_day=args.rate_min,
            rate_max_deg_day=args.rate_max, n_rate=args.n_rate,
            n_rho=args.n_rho, n_rhodot=args.n_rhodot,
            rho_min=args.rho_min, rho_max=args.rho_max, out_dir=args.out,
        )
        print(f"[rate-scan] wrote {args.out}/rate_scan.json and rate_scan.png")
        return 0

    if args.cmd == "ar-edge":
        gen = load_generation(args.run)
        sc_data = load_scores(args.run)
        print(json.dumps(
            ar_edge_slice(gen, sc_data, margin_max=args.margin),
            indent=2, default=float,
        ))
        return 0

    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
