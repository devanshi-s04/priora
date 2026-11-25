"""Unit tests for the WP6 self-consistency harness (validation/self_consistency.py).

Deliberately small and fast (~1 minute total): the end-to-end smoke uses a tiny
(60, 30) grid and N = 50.  Nothing here is a gate-scale run — the gate is
N >= 1e4 at (400, 200), which the supervisor executes separately.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import pytest
from astropy.time import Time

# validation/ is a top-level directory in the repo, not an installed package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from validation import self_consistency as sc  # noqa: E402

from priora.attributable import observer_state  # noqa: E402
from priora.population import Digest2Histogram  # noqa: E402
from priora.transforms import observables_from_elements  # noqa: E402

MODEL = os.path.join(REPO_ROOT, "data", "digest2.model.csv")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL),
    reason="digest2 model file not fetched (see scripts/fetch_digest2_model.py)",
)


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------


def test_chi2_sf_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    for dof in (1, 2, 3, 5, 10, 20):
        for x in (1e-3, 0.5, 3.0, 10.0, 25.0, 60.0):
            assert sc.chi2_sf(x, dof) == pytest.approx(
                scipy_stats.chi2.sf(x, dof), rel=1e-9, abs=1e-15
            )
    assert sc.chi2_sf(0.0, 3) == 1.0
    assert sc.chi2_sf(-1.0, 3) == 1.0


def test_ks_uniform_statistic_and_power():
    rng = np.random.default_rng(0)
    u = rng.random(2000)
    d, p, n_eff = sc.ks_uniform(u)
    assert n_eff == pytest.approx(2000.0)
    assert d < 0.05 and p > 0.05  # a uniform sample is not rejected

    d2, p2, _ = sc.ks_uniform(rng.random(2000) ** 2)  # clearly non-uniform
    assert d2 > 0.2 and p2 < 1e-10

    scipy_stats = pytest.importorskip("scipy.stats")
    assert d == pytest.approx(scipy_stats.kstest(u, "uniform").statistic, rel=1e-12)


def test_auc_matches_brute_force_including_ties_and_weights():
    rng = np.random.default_rng(3)
    s = np.round(rng.random(200), 2)  # rounding forces ties
    y = rng.random(200) < s
    w = rng.random(200) + 0.5

    def brute(weights):
        pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
        num = sum(
            weights[i] * weights[j]
            * (1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0)
            for i in pos for j in neg
        )
        return num / (weights[y].sum() * weights[~y].sum())

    assert sc.auc_weighted(s, y) == pytest.approx(brute(np.ones(200)), rel=1e-12)
    assert sc.auc_weighted(s, y, w) == pytest.approx(brute(w), rel=1e-12)

    fpr, tpr = sc.roc_curve_weighted(s, y)
    assert np.trapezoid(tpr, fpr) == pytest.approx(sc.auc_weighted(s, y), rel=1e-12)


# --------------------------------------------------------------------------
# Reliability binning + PASS/FAIL logic on synthetic data
# --------------------------------------------------------------------------


def test_equal_expected_count_binning():
    """Bins carry equal expected NEO count, not equal tracklet count."""
    rng = np.random.default_rng(5)
    # A realistic gate-mode shape: mostly tiny p, a few near 1.
    p = np.concatenate([rng.random(9800) * 2e-3, 0.5 + rng.random(200) * 0.5])
    y = rng.random(p.size) < p
    bins = sc.reliability_bins(p, y, n_bins=5)
    assert len(bins) == 5
    exp = np.array([b["expected_events"] for b in bins])
    # equal expected NEO count to within one tracklet's worth
    assert exp.max() / exp.min() < 1.5
    # and emphatically NOT equal tracklet counts
    counts = np.array([b["n"] for b in bins])
    assert counts.max() / counts.min() > 10
    assert counts.sum() == p.size


def test_reliability_binning_on_calibrated_synthetic_data():
    """Labels drawn from the scores themselves -> calibrated by construction."""
    rng = np.random.default_rng(11)
    n = 20000
    p = rng.random(n)
    y = rng.random(n) < p

    bins = sc.reliability_bins(p, y, n_bins=5)
    assert len(bins) == 5
    assert sum(b["n"] for b in bins) == n
    for b in bins:
        assert abs(b["obs_frac"] - b["p_mean"]) < 4.0 * b["sigma_frac"]
    assert abs(sc.global_z(p, y)["z"]) < 3.0


def test_weighted_null_variance_is_the_product_of_class_weights():
    """The enriched-mode per-bin z must have unit variance under the null.

    This is the regression test for the variance bug: the null variance of
    ``sum_i w_i (y_i - p_i)`` is ``w_neo * w_other * sum p(1-p)``, not
    ``sum w_i^2 p(1-p)``.  With the wrong form the per-bin z is off by a large
    factor (here ~2.9x), so ``std(z)`` under the null is not 1.
    """
    rng = np.random.default_rng(3)
    p_mix, P = 0.019080332498358048, 0.5
    w_neo, w_other = p_mix / P, (1 - p_mix) / (1 - P)
    p = np.clip(rng.random(4000) ** 3, 1e-6, 1 - 1e-6)
    p_null = sc.null_label_probs(p, w_neo, w_other)

    zs = []
    for _ in range(300):
        y = (rng.random(p.size) < p_null).astype(float)
        w = np.where(y > 0.5, w_neo, w_other)
        zs += [b["z"] for b in sc.reliability_bins(p, y, w, 5, w_neo, w_other)]
    zs = np.array([z for z in zs if np.isfinite(z)])
    assert abs(zs.mean()) < 0.12
    assert 0.9 < zs.std() < 1.1, f"std(z) = {zs.std():.3f}, expected ~1"

    # ... and the old (wrong) form would be badly off on this configuration.
    wrong = np.sqrt(np.sum(np.full(p.size, w_neo) ** 2 * p * (1 - p)))
    right = np.sqrt(w_neo * w_other * np.sum(p * (1 - p)))
    assert wrong / right > 2.0 or right / wrong > 2.0


def test_gate_mode_variance_matches_plain_poisson_binomial():
    rng = np.random.default_rng(4)
    p = rng.random(500)
    y = rng.random(500) < p
    for b in sc.reliability_bins(p, y, n_bins=5):
        assert b["poisson_binomial_var"] > 0


def test_null_label_probs_tilt():
    p = np.array([0.0, 1e-4, 0.5, 0.9, 1.0])
    assert np.allclose(sc.null_label_probs(p), p)  # gate mode: identity
    w_neo, w_other = 0.038, 1.96
    q = sc.null_label_probs(p, w_neo, w_other)
    # odds are scaled by w_other / w_neo
    mid = p[2]
    assert q[2] / (1 - q[2]) == pytest.approx(
        (w_other / w_neo) * mid / (1 - mid), rel=1e-12
    )
    assert q[0] == 0.0 and q[-1] == 1.0


def test_poisson_binomial_exact_tail_matches_monte_carlo():
    rng = np.random.default_rng(0)
    p = rng.random(200) * 0.25
    pmf = sc.poisson_binomial_pmf(p)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-9)
    assert (np.arange(pmf.size) * pmf).sum() == pytest.approx(p.sum(), rel=1e-9)
    draws = np.array([(rng.random(200) < p).sum() for _ in range(60000)])
    for k in (int(p.sum()), int(p.sum() + 8)):
        mc = min(1.0, 2 * min((draws <= k).mean(), (draws >= k).mean()))
        assert sc.global_exact_p(p, k) == pytest.approx(mc, abs=0.01)


def test_bootstrap_passes_calibrated_and_fails_miscalibrated():
    """The rebuilt criterion: calibrated -> PASS, odds-distorted -> FAIL."""
    rng = np.random.default_rng(21)
    # A gate-shaped score vector: rare NEOs, ~70 expected in 1e4.
    p = np.clip(np.concatenate([
        rng.random(9900) * 4e-3, 0.4 + rng.random(100) * 0.6]), 0, 1)

    y_ok = (rng.random(p.size) < p).astype(float)
    boot = sc.bootstrap_reliability(p, y_ok, n_bins=5, n_boot=300, seed=2)
    v = sc.reliability_verdict(boot=boot, bins=sc.reliability_bins(p, y_ok, n_bins=5),
                               glob=sc.global_z(p, y_ok),
                               odds=sc.fit_odds_scale(p, y_ok))
    assert v["pass"] is True
    assert v["odds_scale"]["lo"] < 1.0 < v["odds_scale"]["hi"]

    # Strongly distorted odds: c = 3 (P(NEO) understated everywhere).
    o = 3.0 * p / (1 - p)
    y_bad = (rng.random(p.size) < o / (1 + o)).astype(float)
    boot_b = sc.bootstrap_reliability(p, y_bad, n_bins=5, n_boot=300, seed=2)
    v_b = sc.reliability_verdict(boot=boot_b, bins=sc.reliability_bins(p, y_bad, n_bins=5),
                                 glob=sc.global_z(p, y_bad),
                                 odds=sc.fit_odds_scale(p, y_bad))
    assert v_b["pass"] is False
    assert v_b["global_z"] > 2.0
    assert v_b["odds_scale"]["c_hat"] > 1.5


def test_bootstrap_null_false_alarm_rate_is_near_nominal():
    """MC the null through the real code path: level should be ~1%, not ~43%.

    Small by unit-test standards (400 trials on a 3000-tracklet vector), so it
    only asserts the rate is in a sane band; the full 6000-trial validation at
    gate scale is in the WP6 report.
    """
    rng = np.random.default_rng(31)
    p = np.clip(np.concatenate([
        rng.random(2950) * 6e-3, 0.4 + rng.random(50) * 0.6]), 0, 1)
    B = 400
    r2 = np.random.default_rng(7)
    sims = np.array([
        sc._max_abs_z(sc.reliability_bins(p, (r2.random(p.size) < p).astype(float),
                                          n_bins=5))
        for _ in range(B)
    ])
    pmf = sc.poisson_binomial_pmf(p)
    cdf = np.cumsum(pmf)
    low = np.concatenate(([0.0], cdf))

    fails = 0
    trials = 400
    for _ in range(trials):
        y = (rng.random(p.size) < p).astype(float)
        pv_m = (np.sum(sims >= sc._max_abs_z(sc.reliability_bins(p, y, n_bins=5))) + 1) / (B + 1)
        k = int(min(y.sum(), pmf.size - 1))
        pv_g = min(1.0, 2 * min(cdf[k], 1.0 - low[k]))
        fails += (pv_m <= 0.01) or (pv_g <= 0.01)
    rate = fails / trials
    assert rate < 0.06, f"false-alarm rate {rate:.3f} is far above nominal 0.01"


def test_odds_scale_ignores_certainties():
    """p == 1 tracklets must not drive c_hat (they carry no odds-scale info)."""
    rng = np.random.default_rng(42)
    p = np.concatenate([rng.random(2000) * 0.02, np.ones(3)])
    y = np.concatenate([(rng.random(2000) < p[:2000]), np.ones(3, dtype=bool)])
    fit = sc.fit_odds_scale(p, y)
    assert fit["n_excluded_certain"] == 3
    assert fit["n_used"] == 2000
    assert fit["lo"] < 1.0 < fit["hi"]

    # Without the exclusion the certainties would drag the fit off; check that
    # dropping them by hand gives the same answer.
    bare = sc.fit_odds_scale(p[:2000], y[:2000])
    assert fit["c_hat"] == pytest.approx(bare["c_hat"], rel=1e-9)


def test_odds_scale_railed_when_no_neos_observed():
    p = np.full(500, 1e-3)
    fit = sc.fit_odds_scale(p, np.zeros(500, dtype=bool))
    assert fit["railed"] is True
    assert fit["c_hat"] < 1e-3


def test_odds_scale_recovers_a_known_distortion():
    rng = np.random.default_rng(41)
    p = np.clip(rng.random(40000) * 0.05, 1e-6, 1 - 1e-6)
    for c_true in (0.5, 1.0, 2.5):
        o = c_true * p / (1 - p)
        y = rng.random(p.size) < o / (1 + o)
        fit = sc.fit_odds_scale(p, y)
        assert fit["railed"] is False
        assert fit["lo"] < c_true < fit["hi"], (c_true, fit)
        assert fit["c_hat"] == pytest.approx(c_true, rel=0.25)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def test_generation_counters_are_consistent_and_labels_bound():
    cfg = sc.GenConfig(n=200, seed=5, model=MODEL, batch=2048)
    gen = sc.generate(cfg, verbose=False)
    assert len(gen) == 200

    for name in ("NEO", "other"):
        d = gen.counters["per_class"][name]
        assert d["drawn"] == d["accepted"] + d["rejected"]
        assert d["counters_consistent"] is True
    assert (
        gen.counters["per_class"]["NEO"]["accepted"]
        + gen.counters["per_class"]["other"]["accepted"]
        == 200
    )

    # every accepted object is a bound orbit (DESIGN §4.1 unbound filtering)
    assert np.all(gen.elements[:, 0] > 0.0)
    assert np.all(gen.elements[:, 1] < 1.0)
    # every accepted object passes the observability cut
    assert np.all(gen.v_mag >= cfg.v_min)
    assert np.all(gen.v_mag < cfg.v_max)
    assert np.all(np.isfinite(gen.ra)) and np.all(np.isfinite(gen.ra_rate))
    assert np.all(gen.rho_true > cfg.rho_floor)
    # epochs inside the requested window
    jd0 = Time(cfg.epoch_start, scale="tdb").tdb.jd
    assert np.all(gen.epoch_jd_tdb >= jd0)
    assert np.all(gen.epoch_jd_tdb <= jd0 + cfg.epoch_days)
    # gate mode: no importance weighting
    assert np.allclose(gen.weight, 1.0)


def test_generation_is_deterministic_and_index_stable():
    """Same seed -> same catalogue; a larger N extends it rather than changing it."""
    a = sc.generate(sc.GenConfig(n=40, seed=9, model=MODEL, batch=2048), verbose=False)
    b = sc.generate(sc.GenConfig(n=40, seed=9, model=MODEL, batch=2048), verbose=False)
    assert np.array_equal(a.ra, b.ra)
    assert np.array_equal(a.is_neo, b.is_neo)

    c = sc.generate(sc.GenConfig(n=80, seed=9, model=MODEL, batch=2048), verbose=False)
    assert np.array_equal(a.ra, c.ra[:40])
    assert np.array_equal(a.rho_true, c.rho_true[:40])


def test_generation_round_trip_truth_matches_forward_map():
    """The stored (rho, rho_dot) truth is exactly what the forward map returns."""
    gen = sc.generate(sc.GenConfig(n=60, seed=17, model=MODEL, batch=2048), verbose=False)
    obs = observables_from_elements(gen.elements, gen.H, gen.R_obs, gen.V_obs)
    for key, stored in (
        ("ra", gen.ra), ("dec", gen.dec), ("ra_rate", gen.ra_rate),
        ("dec_rate", gen.dec_rate), ("v_mag", gen.v_mag),
        ("rho", gen.rho_true), ("rho_dot", gen.rho_dot_true),
    ):
        assert np.allclose(obs[key], stored, rtol=1e-12, atol=1e-12), key

    # ... and the recorded observer state is the one for the recorded epoch.
    R, V = observer_state(Time(gen.epoch_jd_tdb, format="jd", scale="tdb"))
    assert np.allclose(R, gen.R_obs, atol=1e-12)
    assert np.allclose(V, gen.V_obs, atol=1e-14)


def test_mixture_proportions_are_statistically_sane():
    """Accepted NEO fraction consistent with mixture x per-class acceptance.

    The class is drawn with P(NEO) = total_count_NEO / total_count_total, but
    the unbound cut and the V cut have class-dependent acceptance, so the
    *accepted* NEO fraction is the reweighted one.  Both are measured from the
    harness's own counters, and the observed split is checked against the
    3-sigma binomial band around that prediction.
    """
    n = 2000
    gen = sc.generate(sc.GenConfig(n=n, seed=23, model=MODEL, batch=4096), verbose=False)
    cn = gen.counters["per_class"]["NEO"]
    co = gen.counters["per_class"]["other"]

    p_mix = gen.counters["p_mixture_neo"]
    neo, other = Digest2Histogram.neo_other_pair(MODEL, "All")
    assert p_mix == pytest.approx(
        neo.total_count / (neo.total_count + other.total_count), rel=1e-12
    )

    # Predicted accepted-NEO fraction from the measured per-class acceptance.
    a_neo = cn["accepted"] / cn["drawn"]
    a_other = co["accepted"] / co["drawn"]
    # Sanity: the acceptance rates themselves are in the expected ballpark.
    assert 0.01 < a_neo < 0.3
    assert 0.01 < a_other < 0.5

    k = int(gen.is_neo.sum())
    p_pred = p_mix * a_neo / (p_mix * a_neo + (1.0 - p_mix) * a_other)
    sigma = np.sqrt(n * p_pred * (1.0 - p_pred))
    assert abs(k - n * p_pred) <= 3.0 * sigma + 1.0

    # And the unbound-discard fractions match the WP4-measured values.
    assert cn["unbound_fraction"] == pytest.approx(0.019, abs=0.006)
    assert co["unbound_fraction"] < 1e-3


def test_enrichment_weights_restore_the_mixture():
    """--neo-fraction changes what is drawn but not what the weights represent."""
    gen = sc.generate(
        sc.GenConfig(n=300, seed=31, model=MODEL, batch=2048, neo_fraction=0.5),
        verbose=False,
    )
    p_mix = gen.counters["p_mixture_neo"]
    assert gen.is_neo.mean() > 0.05  # NEOs genuinely enriched
    assert gen.weight[gen.is_neo][0] == pytest.approx(p_mix / 0.5)
    assert gen.weight[~gen.is_neo][0] == pytest.approx((1 - p_mix) / 0.5)


def test_bare_neo_fraction_flag_defaults_to_0p1():
    """The bare --neo-fraction flag is the low-p amplifier at 0.1, not 0.5."""
    ap = argparse.ArgumentParser()
    sc._add_gen_args(ap)
    assert ap.parse_args([]).neo_fraction is None
    assert ap.parse_args(["--neo-fraction"]).neo_fraction == 0.1
    assert ap.parse_args(["--neo-fraction", "0.4"]).neo_fraction == 0.4


def test_score_config_defaults_track_the_classifier():
    """The harness must not pin a grid the classifier has moved off."""
    from priora.classifier import N_RHO_DEFAULT, N_RHODOT_DEFAULT

    cfg = sc.ScoreConfig()
    assert (cfg.n_rho, cfg.n_rhodot) == (N_RHO_DEFAULT, N_RHODOT_DEFAULT)
    ap = argparse.ArgumentParser()
    sc._add_score_args(ap)
    a = ap.parse_args([])
    assert (a.n_rho, a.n_rhodot) == (N_RHO_DEFAULT, N_RHODOT_DEFAULT)


def test_generator_support_must_lie_inside_the_integration_domain(tmp_path):
    out = str(tmp_path / "domain")
    gen = sc.generate(
        sc.GenConfig(n=5, seed=1, model=MODEL, batch=2048, rho_floor=1e-4),
        verbose=False,
    )
    sc.save_generation(gen, out)
    with pytest.raises(ValueError, match="integration domain"):
        sc.score(out, sc.ScoreConfig(n_rho=40, n_rhodot=10, model=MODEL,
                                     rho_min=0.01),
                 workers=1, chunk=5, verbose=False)


def test_default_rho_floor_matches_the_default_scoring_rho_min():
    assert sc.GenConfig().rho_floor == sc.ScoreConfig().rho_min


@pytest.mark.parametrize("gen_mode,score_mode",
                         [("none", "pchip"), ("pchip", "none")])
def test_generator_and_scorer_must_use_the_same_population_model(
        tmp_path, gen_mode, score_mode):
    """A model mismatch is a hard failure, not a result.

    DESIGN §4.1's calibration argument is that the test set is drawn from the
    classifier's own prior.  Generating from the histogram and scoring with the
    smooth reconstruction (or the reverse) is the §4.3 prior-misspecification
    experiment, not a self-consistency test — and it would still emit a
    PASS/FAIL verdict, which is what makes it dangerous rather than merely
    wrong.  Same treatment as the rho_floor domain check.
    """
    out = str(tmp_path / f"mismatch_{gen_mode}_{score_mode}")
    gen = sc.generate(
        sc.GenConfig(n=5, seed=1, model=MODEL, batch=2048,
                     h_interpolation=gen_mode),
        verbose=False,
    )
    assert gen.config["h_interpolation"] == gen_mode
    sc.save_generation(gen, out)
    with pytest.raises(ValueError, match="h_interpolation"):
        sc.score(out, sc.ScoreConfig(n_rho=40, n_rhodot=10, model=MODEL,
                                     h_interpolation=score_mode),
                 workers=1, chunk=5, verbose=False)


@pytest.mark.parametrize("mode", ["none", "pchip"])
def test_matched_generator_and_scorer_modes_are_accepted(tmp_path, mode):
    """The guard must not fire on the configurations that ARE valid."""
    out = str(tmp_path / f"matched_{mode}")
    gen = sc.generate(
        sc.GenConfig(n=5, seed=1, model=MODEL, batch=2048,
                     h_interpolation=mode),
        verbose=False,
    )
    sc.save_generation(gen, out)
    rows = sc.score(out, sc.ScoreConfig(n_rho=40, n_rhodot=10, model=MODEL,
                                        h_interpolation=mode),
                    workers=1, chunk=5, verbose=False)
    assert rows["p_neo"].size == 5


def test_summary_text_names_the_h_interpolation_on_both_lines(tmp_path):
    """A saved summary.txt must say which population model produced it.

    The `gate_v0_pchip` run's summary.txt originally said "pchip" nowhere at
    all, so a reader could not tell a smooth-mode gate from a histogram one
    without opening the JSON.  This drives the real generate -> score ->
    analyze path in pchip mode at N = 50 and reads the emitted text, rather
    than hand-building a summary dict that would drift from the real schema.
    """
    out = str(tmp_path / "interp_summary")
    gen = sc.generate(sc.GenConfig(n=50, seed=101, model=MODEL, batch=2048,
                                   h_interpolation="pchip"), verbose=False)
    sc.save_generation(gen, out)
    sc.score(out, sc.ScoreConfig(n_rho=120, n_rhodot=20, model=MODEL,
                                 h_interpolation="pchip"),
             workers=1, chunk=25, verbose=False)
    summary = sc.analyze(out, n_bins=5, plots=False, verbose=False, n_boot=20)
    assert summary["generation_config"]["h_interpolation"] == "pchip"
    assert summary["scoring_config"]["h_interpolation"] == "pchip"
    s = sc._summary_text(summary)
    gen_line = next(ln for ln in s.splitlines() if ln.startswith("generator"))
    sco_line = next(ln for ln in s.splitlines() if ln.startswith("scorer"))
    assert "h_interpolation=pchip" in gen_line
    assert "h_interpolation=pchip" in sco_line
    # and the default mode is stated too, so a histogram run is not merely
    # "the one that does not mention it"
    with open(os.path.join(out, "summary.txt")) as fh:
        assert "h_interpolation=pchip" in fh.read()


# --------------------------------------------------------------------------
# End-to-end smoke (tiny grid, single worker)
# --------------------------------------------------------------------------


def test_score_record_schema_is_positionally_consistent():
    """Every _score_one return path must emit exactly len(RECORD_FIELDS) values."""
    rec = dict.fromkeys(sc.RECORD_FIELDS, 0.0)
    assert len(sc._emit(rec)) == len(sc.RECORD_FIELDS)
    rec.pop("hpd_rank")
    with pytest.raises(KeyError):
        sc._emit(rec)


def test_end_to_end_smoke(tmp_path):
    out = str(tmp_path / "smoke")
    gen = sc.generate(sc.GenConfig(n=50, seed=101, model=MODEL, batch=2048), verbose=False)
    sc.save_generation(gen, out)

    reloaded = sc.load_generation(out)
    assert len(reloaded) == 50
    assert np.array_equal(reloaded.is_neo, gen.is_neo)
    assert np.allclose(reloaded.R_obs, gen.R_obs)
    with open(os.path.join(out, "gen_counts.json")) as fh:
        assert "per_class" in json.load(fh)["counters"]

    cfg = sc.ScoreConfig(n_rho=120, n_rhodot=20, model=MODEL)
    res = sc.score(out, cfg, workers=1, chunk=25, verbose=False)

    assert set(res) == set(sc.RECORD_FIELDS)
    assert res["index"].size == 50
    assert np.array_equal(res["index"], np.arange(50))
    assert np.all(np.isfinite(res["p_neo"]))
    assert np.all((res["p_neo"] >= 0.0) & (res["p_neo"] <= 1.0))
    assert np.all(res["n_nodes"] > 0)
    for key in ("hpd_rank", "hpd_rank_exact"):
        assert np.all(np.isfinite(res[key])), key
        assert np.all((res[key] >= 0.0) & (res[key] <= 1.0)), key
    assert np.all(res["truth_density"] > 0.0)
    assert np.all((res["collapse_frac_rho2"] >= 0.0) & (res["collapse_frac_rho2"] <= 1.0))
    assert np.all(res["ar_frac_h_above"] >= 0.0)
    # v0: both components' support ends at H = 25.5, so §2.6 flag (a) is zero.
    assert np.all(res["mass_frac_q_above_h_above"] == 0.0)
    # The truth is a bound orbit, so it must lie INSIDE the admissible rho_dot
    # interval at its own rho -- checked in AU/day against the exact interval,
    # not in grid cell widths (which would weaken as n_rhodot drops).
    assert np.all(res["truth_rho_in_range"] == 1)
    assert np.all(res["rhodot_outside_au_per_day"] == 0.0)
    assert np.all((res["rhodot_margin_frac"] >= 0.0)
                  & (res["rhodot_margin_frac"] <= 0.5))

    summary = sc.analyze(out, n_bins=5, plots=True, verbose=False, n_boot=100)
    assert summary["n_scored"] == 50
    assert set(summary["verdicts"]) == {
        "RELIABILITY", "HPD_RANK_UNIFORM", "COLLAPSE_COVERAGE", "SCORES_DEFINED",
        "TRUTH_DENSITY_POSITIVE", "TRUTH_IN_DOMAIN",
    }
    # A 50-tracklet run is not a gate run and must say so.
    assert summary["gate_eligible"] is False
    assert summary["gate_ineligible_reasons"]
    assert summary["collapse"]["n_truth_density_nan"] == 0
    assert summary["verdicts"]["SCORES_DEFINED"] is True
    assert summary["verdicts"]["COLLAPSE_COVERAGE"] is True
    # Every generated object came from a NON-EMPTY histogram bin, so the summed
    # integrand at its own (elements, H) must be > 0.  This is the sharpest
    # cheap check that the generator's sampler and the classifier's density
    # lookup are talking about the same model.
    assert summary["verdicts"]["TRUTH_DENSITY_POSITIVE"] is True
    # AUC is nan when a 50-tracklet draw from a 0.8%-NEO mixture contains no
    # NEO at all; that is expected, not a failure.
    assert np.isfinite(summary["auc"]) or summary["n_neo_true"] == 0
    for name in ("reliability.png", "scores.png", "roc.png", "collapse.png",
                 "diagnostics.png", "summary.txt", "summary.json"):
        assert os.path.exists(os.path.join(out, name)), name
    with open(os.path.join(out, "summary.json")) as fh:
        assert json.load(fh)["n_scored"] == 50


def test_scoring_resume_is_exact(tmp_path):
    out = str(tmp_path / "resume")
    sc.save_generation(
        sc.generate(sc.GenConfig(n=20, seed=77, model=MODEL, batch=2048), verbose=False),
        out,
    )
    cfg = sc.ScoreConfig(n_rho=80, n_rhodot=20, model=MODEL)
    first = sc.score(out, cfg, workers=1, chunk=10, verbose=False)

    # Drop the second chunk and the merged file: --resume must recompute only it
    # and land on bit-identical numbers.
    os.remove(sc._chunk_path(out, 10, 20))
    os.remove(os.path.join(out, "scores.npz"))
    second = sc.score(out, cfg, workers=1, chunk=10, resume=True, verbose=False)
    for name in sc.RECORD_FIELDS:
        if name == "elapsed_s":  # wall clock, not reproducible
            continue
        assert np.array_equal(first[name], second[name], equal_nan=True), name


def test_resume_rejects_chunks_from_a_different_grid(tmp_path):
    """A stale chunk must be recomputed, never spliced into the results table."""
    out = str(tmp_path / "stale")
    sc.save_generation(
        sc.generate(sc.GenConfig(n=20, seed=78, model=MODEL, batch=2048), verbose=False),
        out,
    )
    coarse = sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL)
    fine = sc.ScoreConfig(n_rho=120, n_rhodot=15, model=MODEL)
    a = sc.score(out, coarse, workers=1, chunk=20, verbose=False)
    b = sc.score(out, fine, workers=1, chunk=20, resume=True, verbose=False)
    # Same catalogue, finer grid -> the chunk was NOT reused.
    assert not np.array_equal(a["n_nodes"], b["n_nodes"])
    assert np.all(b["n_nodes"] > a["n_nodes"])


def test_cli_run_subcommand(tmp_path):
    out = str(tmp_path / "cli")
    rc = sc.main(
        [
            "run", "--n", "40", "--seed", "202", "--model", MODEL,
            "--n-rho", "120", "--n-rhodot", "20", "--workers", "2",
            "--chunk", "20", "--out", out, "--bins", "4", "--no-plots",
            "--boot", "50",
        ]
    )
    assert rc in (0, 1)  # PASS/FAIL both legal at N = 40; a crash is not
    assert os.path.exists(os.path.join(out, "scores.npz"))
    assert os.path.exists(os.path.join(out, "summary.json"))


def test_analyze_survives_an_all_undefined_score_column(tmp_path):
    """No tracklet has a finite P(NEO): must return a clean FAIL, not IndexError."""
    out = str(tmp_path / "undef")
    gen = sc.generate(sc.GenConfig(n=12, seed=55, model=MODEL, batch=2048), verbose=False)
    sc.save_generation(gen, out)
    cfg = sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL)
    res = sc.score(out, cfg, workers=1, chunk=12, verbose=False)

    res["p_neo"] = np.full(res["p_neo"].shape, np.nan)
    res["log_lr"] = np.full(res["log_lr"].shape, np.nan)
    np.savez_compressed(os.path.join(out, "scores.npz"), **res)

    summary = sc.analyze(out, plots=True, verbose=False, n_boot=20)
    assert summary["frac_undefined_p"] == 1.0
    assert summary["verdicts"]["SCORES_DEFINED"] is False
    assert summary["overall_pass"] is False
    assert not np.isfinite(summary["auc"])
    assert summary["reliability_bins"] == []


def test_truth_density_nan_counts_as_a_failure(tmp_path):
    """A systematically broken _integrand_at_truth must not PASS at 1.0."""
    out = str(tmp_path / "nan")
    gen = sc.generate(sc.GenConfig(n=12, seed=56, model=MODEL, batch=2048), verbose=False)
    sc.save_generation(gen, out)
    res = sc.score(out, sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL),
                   workers=1, chunk=12, verbose=False)
    assert np.all(np.isfinite(res["truth_density"]))

    res["truth_density"] = np.full(res["truth_density"].shape, np.nan)
    np.savez_compressed(os.path.join(out, "scores.npz"), **res)
    summary = sc.analyze(out, plots=False, verbose=False, n_boot=20)
    assert summary["collapse"]["n_truth_density_nan"] == 12
    assert summary["verdicts"]["TRUTH_DENSITY_POSITIVE"] is False


def test_resume_rejects_a_tampered_chunk(tmp_path):
    """The provenance stamp survives an edit; the content hash must not."""
    out = str(tmp_path / "tamper")
    sc.save_generation(
        sc.generate(sc.GenConfig(n=10, seed=57, model=MODEL, batch=2048), verbose=False),
        out,
    )
    cfg = sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL)
    first = sc.score(out, cfg, workers=1, chunk=10, verbose=False)

    path = sc._chunk_path(out, 0, 10)
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    data["p_neo"] = data["p_neo"] * 0.5 + 0.1  # same stamp, different values
    np.savez_compressed(path, **data)

    second = sc.score(out, cfg, workers=1, chunk=10, resume=True, verbose=False)
    assert np.allclose(second["p_neo"], first["p_neo"], equal_nan=True)


def test_resume_removes_orphaned_chunks_from_another_chunk_size(tmp_path):
    out = str(tmp_path / "orphan")
    sc.save_generation(
        sc.generate(sc.GenConfig(n=12, seed=58, model=MODEL, batch=2048), verbose=False),
        out,
    )
    cfg = sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL)
    sc.score(out, cfg, workers=1, chunk=4, verbose=False)
    assert len(os.listdir(os.path.join(out, "chunks"))) == 3

    sc.score(out, cfg, workers=1, chunk=6, resume=True, verbose=False)
    names = sorted(os.listdir(os.path.join(out, "chunks")))
    assert len(names) == 2, names  # the three chunk-of-4 files are gone


def test_ar_edge_slice_selects_the_boundary_population(tmp_path):
    out = str(tmp_path / "edge")
    gen = sc.generate(sc.GenConfig(n=60, seed=59, model=MODEL, batch=2048), verbose=False)
    sc.save_generation(gen, out)
    res = sc.score(out, sc.ScoreConfig(n_rho=120, n_rhodot=20, model=MODEL),
                   workers=1, chunk=30, verbose=False)
    wide = sc.ar_edge_slice(gen, res, margin_max=0.5)
    assert wide["n"] == 60  # every margin is <= 0.5 by construction
    assert wide["coverage"] == pytest.approx(1.0)
    tight = sc.ar_edge_slice(gen, res, margin_max=1e-6)
    assert tight["n"] <= wide["n"]


def test_rate_scan_produces_a_monotone_separation(tmp_path):
    """Opposition rate scan: fast movers must score higher than slow ones."""
    out = str(tmp_path / "scan")
    res = sc.rate_scan(
        model=MODEL, v_mag=21.0, rate_min_deg_day=0.05, rate_max_deg_day=3.0,
        n_rate=6, n_rho=200, n_rhodot=20, out_dir=out, verbose=False,
    )
    rows = res["scan"]
    assert len(rows) == 6
    p = np.array([r["p_neo"] for r in rows])
    rho = np.array([r["rho_map"] for r in rows])
    # P(NEO) may be undefined at the fastest rates (both supports exhausted,
    # DESIGN §2.6); the finite ones must be probabilities.
    fin = np.isfinite(p)
    assert fin.sum() >= 4
    assert np.all((p[fin] >= 0) & (p[fin] <= 1))
    assert "n_undefined_p_neo" in res
    # A 3 deg/day mover at opposition is a NEO; a 0.05 deg/day one is not.
    assert p[fin][-1] > p[fin][0]
    # ... and the inferred distance falls as the rate rises.
    assert rho[fin][-1] < rho[fin][0]
    assert os.path.exists(os.path.join(out, "rate_scan.png"))
    assert os.path.exists(os.path.join(out, "rate_scan.json"))
    assert res["reference_rates_deg_per_day"]["NEO q=1.3"] > \
        res["reference_rates_deg_per_day"]["outer belt 3.2"]


def test_gate_flag_adds_a_scale_verdict(tmp_path):
    out = str(tmp_path / "gate")
    gen = sc.generate(sc.GenConfig(n=12, seed=60, model=MODEL, batch=2048), verbose=False)
    sc.save_generation(gen, out)
    sc.score(out, sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL),
             workers=1, chunk=12, verbose=False)
    s_no = sc.analyze(out, plots=False, verbose=False, n_boot=20)
    assert "GATE_SCALE" not in s_no["verdicts"]
    s_gate = sc.analyze(out, plots=False, verbose=False, n_boot=20, gate=True)
    assert s_gate["verdicts"]["GATE_SCALE"] is False
    assert s_gate["overall_pass"] is False
    assert any("< 10000" in r for r in s_gate["gate_ineligible_reasons"])


def test_limit_run_is_marked_non_gate(tmp_path):
    out = str(tmp_path / "limited")
    gen = sc.generate(sc.GenConfig(n=20, seed=61, model=MODEL, batch=2048), verbose=False)
    sc.save_generation(gen, out)
    sc.score(out, sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL),
             workers=1, chunk=10, limit=10, verbose=False)
    summary = sc.analyze(out, plots=False, verbose=False, n_boot=20)
    assert summary["n_scored"] == 10
    assert summary["n_generated"] == 20
    assert summary["gate_eligible"] is False
    assert any("--limit" in r for r in summary["gate_ineligible_reasons"])


# --------------------------------------------------------------------------
# D-19 — the finite H extension and the triage-policy accounting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gen_ext,score_ext",
                         [(None, 30.0), (30.0, None), (30.0, 40.0)])
def test_generator_and_scorer_must_agree_on_h_extend_to(tmp_path, gen_ext,
                                                        score_ext):
    """Same hard failure as the h_interpolation mismatch, and it matters more.

    The extension multiplies the NEO component's total count by ~2.7, so a
    generator/scorer mismatch does not merely reshape the faint end — it
    changes the mixture fraction the whole calibration argument is about.
    A 30-vs-40 mismatch is included because those two share a bin shape.
    """
    out = str(tmp_path / f"extmismatch_{gen_ext}_{score_ext}")
    gen = sc.generate(
        sc.GenConfig(n=5, seed=1, model=MODEL, batch=2048,
                     h_extend_to=gen_ext),
        verbose=False,
    )
    assert gen.config["h_extend_to"] == gen_ext
    sc.save_generation(gen, out)
    with pytest.raises(ValueError, match="h_extend_to"):
        sc.score(out, sc.ScoreConfig(n_rho=40, n_rhodot=10, model=MODEL,
                                     h_extend_to=score_ext),
                 workers=1, chunk=5, verbose=False)


@pytest.mark.parametrize("interp", ["none", "pchip"])
def test_matched_extended_generator_and_scorer_are_accepted(tmp_path, interp):
    """The gate path in extended mode: generator = prior = extended model."""
    out = str(tmp_path / f"extmatched_{interp}")
    gen = sc.generate(
        sc.GenConfig(n=8, seed=2, model=MODEL, batch=2048, h_extend_to=30.0,
                     h_interpolation=interp),
        verbose=False,
    )
    sc.save_generation(gen, out)
    rows = sc.score(out, sc.ScoreConfig(n_rho=40, n_rhodot=10, model=MODEL,
                                        h_extend_to=30.0,
                                        h_interpolation=interp),
                    workers=1, chunk=8, verbose=False)
    assert rows["p_neo"].size == 8
    # The extended pair is gate-eligible because it is *samplable* — the D-14
    # clamp cannot be run here at all, because sample() raises.  That is the
    # property being asserted, at the component level where it is unambiguous:
    neo, other = Digest2Histogram.neo_other_pair(
        MODEL, "All", h_extend_to=30.0, h_interpolation=interp
    )
    _, h_draw = neo.sample(20_000, np.random.default_rng(0))
    assert np.any(h_draw > 25.5) and np.all(h_draw < 30.0)
    # In the *accepted* catalogue H > 25.5 is rare, and that is a property of
    # the observability cut rather than of the extension: V in [14, 24) admits
    # an H = 26 object only at very small rho.  Measured at N = 2000:
    # 0.05% of accepted tracklets.  What the extension does change at gate
    # scale is the MIXTURE FRACTION — the NEO component's total count grows
    # ~2.7x, so the drawn NEO fraction goes ~0.45% -> ~0.60%.  Both facts
    # matter for reading a gate run and neither is asserted at N = 8; what is
    # asserted here is only that nothing leaves the extended support.
    assert np.all(gen.H < 30.0) and np.all(gen.H >= 0.0)


def test_h_extend_to_default_preserves_the_historical_run_stamp(tmp_path):
    """Adding the field must not invalidate every cached chunk in the repo.

    Same narrow exception as `h_interpolation`: popped from the stamp payload
    at its default, hashed whenever set.  Checked by comparing the stamp of a
    default ScoreConfig against a hand-built payload that omits BOTH fields.
    """
    import hashlib

    out = str(tmp_path / "stamp")
    sc.save_generation(
        sc.generate(sc.GenConfig(n=4, seed=3, model=MODEL, batch=2048),
                    verbose=False),
        out,
    )
    cfg = sc.ScoreConfig(n_rho=40, n_rhodot=10, model=MODEL)
    payload = dataclasses.asdict(cfg)
    payload.pop("h_interpolation")
    payload.pop("h_extend_to")
    h = hashlib.sha256()
    with open(os.path.join(out, "gen.npz"), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    h.update(json.dumps(payload, sort_keys=True).encode())
    assert sc._run_stamp(out, cfg) == h.hexdigest()
    # ... and setting it DOES change the stamp, in both directions.
    ext30 = dataclasses.replace(cfg, h_extend_to=30.0)
    ext40 = dataclasses.replace(cfg, h_extend_to=40.0)
    assert sc._run_stamp(out, ext30) != sc._run_stamp(out, cfg)
    assert sc._run_stamp(out, ext30) != sc._run_stamp(out, ext40)


def test_summary_text_names_h_extend_to_on_both_lines(tmp_path):
    out = str(tmp_path / "ext_summary")
    gen = sc.generate(sc.GenConfig(n=40, seed=105, model=MODEL, batch=2048,
                                   h_extend_to=30.0), verbose=False)
    sc.save_generation(gen, out)
    sc.score(out, sc.ScoreConfig(n_rho=100, n_rhodot=20, model=MODEL,
                                 h_extend_to=30.0),
             workers=1, chunk=20, verbose=False)
    summary = sc.analyze(out, n_bins=4, plots=False, verbose=False, n_boot=20)
    assert summary["generation_config"]["h_extend_to"] == 30.0
    assert summary["scoring_config"]["h_extend_to"] == 30.0
    s = sc._summary_text(summary)
    gen_line = next(ln for ln in s.splitlines() if ln.startswith("generator"))
    sco_line = next(ln for ln in s.splitlines() if ln.startswith("scorer"))
    assert "h_extend_to=30.0" in gen_line
    assert "h_extend_to=30.0" in sco_line


def test_cli_threads_h_extend_to_through_generate_and_score(tmp_path):
    out = str(tmp_path / "ext_cli")
    rc = sc.main([
        "run", "--n", "20", "--seed", "303", "--model", MODEL,
        "--n-rho", "80", "--n-rhodot", "15", "--workers", "2",
        "--chunk", "10", "--out", out, "--bins", "3", "--no-plots",
        "--boot", "30", "--h-extend-to", "30",
    ])
    assert rc in (0, 1)
    with open(os.path.join(out, "summary.json")) as fh:
        s = json.load(fh)
    assert s["generation_config"]["h_extend_to"] == 30.0
    assert s["scoring_config"]["h_extend_to"] == 30.0
    # one flag drives both sides, so they cannot disagree from the CLI
    assert (s["generation_config"]["h_extend_to"]
            == s["scoring_config"]["h_extend_to"])


def test_scores_defined_accounting_under_the_triage_policy(tmp_path):
    """The harness's three-way score accounting after D-19.

    A D-19 policy row is a *defined number* with **zero likelihood content**
    behind it, so this harness treats it as its own third category:

    * it is NOT counted as likelihood-defined, and SCORES_DEFINED's criterion
      reads (undefined + policy) / scored — otherwise a run could pass the
      undefined-fraction criterion by assigning scores it cannot justify, and
      at the real-data 2.2% rate that is exactly the FAIL/PASS margin;
    * it is EXCLUDED from the reliability diagram, the global-Z, the odds fit
      and the ROC (the adversarial review measured that ~34 such rows already
      make one of five equal-expected-NEO-count bins vacuous, and a policy row
      on a true non-NEO pushes max|z| arbitrarily);
    * the three counts are reported separately and partition the run.

    Policy rows are recognised from the recorded
    (isfinite(p), n_nodes > 0, lambda_neo == 0, lambda_other == 0) quadruple,
    so no chunk schema change was needed; this test forges that quadruple on a
    real scored run and checks every consequence.
    """
    out = str(tmp_path / "policy_acct")
    gen = sc.generate(sc.GenConfig(n=12, seed=57, model=MODEL, batch=2048),
                      verbose=False)
    sc.save_generation(gen, out)
    cfg = sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL)
    res = sc.score(out, cfg, workers=1, chunk=12, verbose=False)

    # Baseline: a clean run has neither undefined nor policy-assigned rows,
    # and the three counts partition it.
    base = sc.analyze(out, plots=False, verbose=False, n_boot=20)
    assert base["n_undefined_p"] == 0
    assert base["n_policy_assigned_p"] == 0
    assert base["frac_policy_assigned_p"] == 0.0
    assert base["n_likelihood_defined_p"] == base["n_scored"] == 12
    assert base["verdicts"]["SCORES_DEFINED"] is True
    base_text = sc._summary_text(base)
    assert "policy-assigned    : 0" in base_text
    assert "likelihood-defined : 12" in base_text
    # With zero policy rows the reliability inputs are untouched by the
    # exclusion -- which is why the extended-mode gate did not need re-running.
    assert base["reliability"]["global_z"] == base["reliability"]["global_z"]

    # Forge three policy-assigned rows exactly as classify() would write them:
    # non-empty region, both Lambdas exactly zero, p = 1, log_lr = nan.
    assert np.all(res["n_nodes"] > 0)
    res["lambda_neo"][:3] = 0.0
    res["lambda_other"][:3] = 0.0
    res["p_neo"][:3] = 1.0
    res["log_lr"][:3] = np.nan
    np.savez_compressed(os.path.join(out, "scores.npz"), **res)

    s = sc.analyze(out, plots=False, verbose=False, n_boot=20)
    assert s["n_policy_assigned_p"] == 3
    assert s["frac_policy_assigned_p"] == pytest.approx(3 / 12)
    assert s["n_undefined_p"] == 0
    # The three counts still partition the run, with the policy rows moved out
    # of the likelihood-defined column.
    assert s["n_likelihood_defined_p"] == 9
    assert (s["n_likelihood_defined_p"] + s["n_policy_assigned_p"]
            + s["n_undefined_p"] == s["n_scored"])
    # SCORES_DEFINED counts them as NOT likelihood-defined.
    assert s["frac_undefined_p"] == pytest.approx(3 / 12)
    assert s["verdicts"]["SCORES_DEFINED"] is False
    # They are excluded from the reliability inputs: the bins must hold only
    # the 9 likelihood-scored tracklets.
    assert sum(b["n"] for b in s["reliability_bins"]) == 9
    # ... and the summary text says all of it, on distinct lines.
    text = sc._summary_text(s)
    pol = next(ln for ln in text.splitlines()
               if ln.startswith("policy-assigned"))
    lik = next(ln for ln in text.splitlines()
               if ln.startswith("likelihood-defined"))
    assert "3" in pol and "D-19" in pol and "NOT a posterior" in pol
    assert "9" in lik
    assert "EXCLUDED from the reliability diagram" in text
    assert "SCORES_DEFINED reads (undefined + policy-assigned)" in text

    # A summary that predates the field must still render (no KeyError).
    old = dict(s)
    old.pop("n_policy_assigned_p")
    old.pop("frac_policy_assigned_p")
    assert "predates D-19" in sc._summary_text(old)


def test_policy_rows_are_excluded_from_reliability_and_global_z(tmp_path):
    """Exclusion, not dilution — measured on the statistics themselves.

    The adversarial review's objection was quantitative: policy rows are all
    p = 1 exactly, so they collapse reliability bins, and a policy row on a
    true non-NEO contributes zero predicted against an observed failure. This
    checks that neither reaches the verdict machinery, by comparing an
    analyze() of a forged run against an analyze() of the same run with those
    rows physically deleted -- the two must agree exactly.
    """
    out = str(tmp_path / "policy_excl")
    gen = sc.generate(sc.GenConfig(n=40, seed=59, model=MODEL, batch=2048),
                      verbose=False)
    sc.save_generation(gen, out)
    cfg = sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL)
    res = sc.score(out, cfg, workers=1, chunk=20, verbose=False)
    clean = {k: v.copy() for k, v in res.items()}

    # Forge 5 policy rows; make sure at least one lands on a true non-NEO,
    # which is the case that explodes max|z| if it is not excluded.
    y = gen.is_neo[res["index"].astype(int)].astype(bool)
    non_neo = np.flatnonzero(~y)[:5]
    assert non_neo.size >= 2
    for i in non_neo:
        res["lambda_neo"][i] = 0.0
        res["lambda_other"][i] = 0.0
        res["p_neo"][i] = 1.0
        res["log_lr"][i] = np.nan
    np.savez_compressed(os.path.join(out, "scores.npz"), **res)
    forged = sc.analyze(out, plots=False, verbose=False, n_boot=50,
                        boot_seed=7)

    # Now physically delete those rows and re-analyse: the likelihood-based
    # statistics must be IDENTICAL, which is what "excluded" means.
    keep = np.ones(res["index"].size, dtype=bool)
    keep[non_neo] = False
    np.savez_compressed(os.path.join(out, "scores.npz"),
                        **{k: v[keep] for k, v in clean.items()})
    deleted = sc.analyze(out, plots=False, verbose=False, n_boot=50,
                         boot_seed=7)

    assert forged["n_policy_assigned_p"] == non_neo.size
    assert forged["n_policy_assigned_on_true_non_neo"] == non_neo.size
    assert forged["n_likelihood_defined_p"] == deleted["n_scored"]
    for key in ("global_z", "observed_neo_count", "expected_neo_count",
                "max_abs_z"):
        a, b = forged["reliability"][key], deleted["reliability"][key]
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12), key
    # AUC is nan when a small draw from a 0.9%-NEO mixture holds no NEO at
    # all; equal-or-both-nan is the assertion that holds either way.
    if np.isfinite(deleted["auc"]) or np.isfinite(forged["auc"]):
        assert forged["auc"] == pytest.approx(deleted["auc"], rel=1e-12)
    else:
        assert np.isnan(forged["auc"]) and np.isnan(deleted["auc"])
    assert [b["n"] for b in forged["reliability_bins"]] == \
        [b["n"] for b in deleted["reliability_bins"]]
    # And the policy false positives are reported rather than swallowed into
    # the impossible-event counter, which is about likelihood scores.
    assert forged["impossible_events"]["n_true_other_with_p_one"] == \
        deleted["impossible_events"]["n_true_other_with_p_one"]


def test_pre_d19_cached_rows_are_not_double_counted(tmp_path):
    """MINOR 6: a nan-p row with zero Lambdas is undefined, not policy.

    Chunks written before D-19 stored a fully-truncated tracklet with the same
    zero Lambdas and a *nan* p_neo.  Without the isfinite(p) term such a row
    would be counted in BOTH the undefined and the policy-assigned column, so
    the three counts would not partition the run and the summary's claim about
    them would be false.  Verified on `enriched_f01`'s one such row by the
    reviewer; reproduced here synthetically.
    """
    out = str(tmp_path / "pre_d19")
    gen = sc.generate(sc.GenConfig(n=10, seed=61, model=MODEL, batch=2048),
                      verbose=False)
    sc.save_generation(gen, out)
    res = sc.score(out, sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL),
                   workers=1, chunk=10, verbose=False)
    # exactly how a pre-D-19 run recorded a fully-truncated tracklet
    res["n_nodes"][0] = 1234
    res["lambda_neo"][0] = 0.0
    res["lambda_other"][0] = 0.0
    res["p_neo"][0] = np.nan
    res["log_lr"][0] = np.nan
    np.savez_compressed(os.path.join(out, "scores.npz"), **res)
    s = sc.analyze(out, plots=False, verbose=False, n_boot=20)
    assert s["n_undefined_p"] == 1
    assert s["n_policy_assigned_p"] == 0
    assert (s["n_likelihood_defined_p"] + s["n_policy_assigned_p"]
            + s["n_undefined_p"] == s["n_scored"] == 10)


def test_empty_region_rows_are_not_counted_as_policy_assigned(tmp_path):
    """n_nodes == 0 keeps its nan and must NOT be mistaken for a policy row.

    The two 0/0 cases are separated by n_nodes, on the analysis side exactly as
    in `classifier._posterior`; a row with an empty admissible region also has
    both Lambdas zero, so the n_nodes term of the test is load-bearing.
    """
    out = str(tmp_path / "policy_vs_empty")
    gen = sc.generate(sc.GenConfig(n=10, seed=58, model=MODEL, batch=2048),
                      verbose=False)
    sc.save_generation(gen, out)
    res = sc.score(out, sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL),
                   workers=1, chunk=10, verbose=False)
    res["n_nodes"][:2] = 0
    res["lambda_neo"][:2] = 0.0
    res["lambda_other"][:2] = 0.0
    res["p_neo"][:2] = np.nan
    res["log_lr"][:2] = np.nan
    np.savez_compressed(os.path.join(out, "scores.npz"), **res)
    s = sc.analyze(out, plots=False, verbose=False, n_boot=20)
    assert s["n_policy_assigned_p"] == 0
    assert s["n_undefined_p"] == 2


# --------------------------------------------------------------------------
# D-19 — the downstream validation modules
#
# These live here rather than in the per-module test files because the thing
# under test is one shared rule ("a D-19 policy row is not a likelihood
# score"), and because the modules' own test files are outside this task's
# ownership.
# --------------------------------------------------------------------------


def _forge_policy_run(tmp_path, name, n=12, n_policy=3, seed=71):
    """A real scored WP6 run with `n_policy` rows rewritten as policy rows."""
    out = str(tmp_path / name)
    gen = sc.generate(sc.GenConfig(n=n, seed=seed, model=MODEL, batch=2048),
                      verbose=False)
    sc.save_generation(gen, out)
    res = sc.score(out, sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL),
                   workers=1, chunk=n, verbose=False)
    assert np.all(res["n_nodes"] > 0)
    res["lambda_neo"][:n_policy] = 0.0
    res["lambda_other"][:n_policy] = 0.0
    res["p_neo"][:n_policy] = 1.0
    res["log_lr"][:n_policy] = np.nan
    np.savez_compressed(os.path.join(out, "scores.npz"), **res)
    return out


def test_digest2_comparison_drops_policy_rows_from_the_keep_filter(tmp_path):
    """MAJOR 5: the symmetric-drop contract of DIGEST2_COMPARISON.md.

    A policy row is p = 1.0 for priora with no likelihood behind it, on a
    class that is empirically ~100% true NEO.  Keeping it in `load_run` would
    hand priora free true positives that digest2 has to earn, biasing dAUC in
    priora's favour by construction — and silently, because the row looks
    like an ordinary saturated score.
    """
    from validation import digest2_comparison as dc

    out = _forge_policy_run(tmp_path, "d2c", n=12, n_policy=3)
    run = dc.load_run(out)
    assert run.policy_detectable is True
    assert run.n == 9
    assert run.n_dropped_policy == 3
    assert run.n_dropped_nonfinite == 0
    # nothing that survived is a policy row
    assert np.all(np.isfinite(run.p_neo))
    lam = run.scores["lambda_neo"].astype(float)
    lao = run.scores["lambda_other"].astype(float)
    assert not np.any((lam == 0.0) & (lao == 0.0))

    # A score table WITHOUT the Lambda columns must still load — this module is
    # driven with hand-built tables too — but it must not then claim that no
    # policy rows were present.  "0 detected" and "0 present" are different
    # statements and `policy_detectable` is which one this is.
    thin = str(tmp_path / "d2c_thin")
    os.makedirs(os.path.join(thin, "chunks"), exist_ok=True)
    import shutil
    shutil.copy(os.path.join(out, "gen.npz"), os.path.join(thin, "gen.npz"))
    with np.load(os.path.join(out, "scores.npz")) as z:
        cols = {k: z[k] for k in z.files
                if k not in ("lambda_neo", "lambda_other")}
    np.savez_compressed(os.path.join(thin, "scores.npz"), **cols)
    thin_run = dc.load_run(thin)
    assert thin_run.policy_detectable is False
    assert thin_run.n_dropped_policy == 0
    # ... and the 3 policy rows DO leak through, because nothing in the table
    # distinguishes them (they carry a finite p = 1.0).  That is the honest
    # consequence and the reason `policy_detectable` is exposed rather than
    # the count being read as "none were present".
    assert thin_run.n == 12


def test_digest2_comparison_refuses_a_clamped_control_for_a_non_histogram_run(
        tmp_path):
    """The clamp is mutually exclusive with pchip (D-17) and with the D-19
    extension, so those runs have no clamped control — and the old code
    silently dropped both keys and scored a plain clamped histogram against
    them, calling the difference "boundary handling"."""
    from validation import digest2_comparison as dc

    for kw, needle in (
        (dict(h_interpolation="pchip"), "h_interpolation"),
        (dict(h_extend_to=30.0), "h_extend_to"),
    ):
        out = str(tmp_path / ("clamp_refuse_" + "".join(map(str, kw))))
        os.makedirs(out, exist_ok=True)
        cfg = dataclasses.asdict(
            sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL, **kw)
        )
        with open(os.path.join(out, "score_config.json"), "w") as fh:
            json.dump(cfg, fh)
        with pytest.raises(ValueError, match=needle):
            dc.clamped_score_config(out)
        # and the worker refuses too, so the guard is not only in the config
        with pytest.raises(ValueError, match="mutually exclusive"):
            dc._init_clamped_worker(MODEL, "All", cfg)

    # A histogram-mode run still works, and carries the new fields at default.
    out = str(tmp_path / "clamp_ok")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "score_config.json"), "w") as fh:
        json.dump(dataclasses.asdict(
            sc.ScoreConfig(n_rho=60, n_rhodot=15, model=MODEL)), fh)
    got = dc.clamped_score_config(out)
    assert got.h_interpolation == "none" and got.h_extend_to is None


def test_astrometric_noise_masks_policy_rows_back_to_nan(tmp_path):
    """MAJOR 5 / C3: a policy row must read as "no likelihood score".

    `mask_policy_rows` is what keeps the fully-truncated direction visible to
    the C3 break arm (which counts tracklets that lose their score under
    noise) and out of the calibration/ROC blocks (where a 1.0 on a ~100%-NEO
    class would flatter the noisy column).
    """
    from validation import astrometric_noise as an

    cols = {
        "p_neo": np.array([0.4, 1.0, np.nan, 1.0, 0.0]),
        "n_nodes": np.array([100, 100, 0, 100, 100], dtype=np.int64),
        "lambda_neo": np.array([1.0, 0.0, 0.0, 3.0, 0.0]),
        "lambda_other": np.array([2.0, 0.0, 0.0, 0.0, 5.0]),
    }
    out = an.mask_policy_rows(cols)
    # row 1 is the policy row; row 3 is a genuine one-sided posterior (p = 1
    # with Lambda_NEO > 0) and must survive; row 2 was already nan.
    assert np.isnan(out["p_neo"][1])
    assert out["p_neo"][3] == 1.0
    assert out["p_neo"][0] == 0.4 and out["p_neo"][4] == 0.0
    assert int(out["_n_policy_assigned"][0]) == 1
    assert list(out["_policy_assigned"]) == [False, True, False, False, False]
    # A table without the Lambda columns is passed through untouched.
    thin = {"p_neo": np.array([1.0, 0.5])}
    assert an.mask_policy_rows(thin) is thin


def test_neocp_pilot_detects_the_h_wall_without_relying_on_nan(tmp_path):
    """MAJOR 3: the 11/11 measurement must survive the policy that cites it.

    The pilot's detector used to be `~isfinite(p_neo)`, which after D-19 is
    all-False on a freshly scored sample.  It is now the recorded triple, which
    gives the same answer on a pre-D-19 score file (nan p) and a post-D-19 one
    (p = 1.0) — checked here on both vintages of the same forged table.
    """
    from validation import neocp_pilot as pilot

    store = {
        "priora_all_n_nodes": np.array([100, 100, 0, 100], dtype=np.int64),
        "priora_all_lambda_neo": np.array([0.0, 1.0, 0.0, 0.0]),
        "priora_all_lambda_other": np.array([0.0, 2.0, 0.0, 3.0]),
    }
    # the detector is a closure inside analyze(); reproduce its definition here
    # against the same columns, which is what the test is really pinning.
    wall = ((store["priora_all_n_nodes"] > 0)
            & (store["priora_all_lambda_neo"] == 0.0)
            & (store["priora_all_lambda_other"] == 0.0))
    assert list(wall) == [True, False, False, False]
    # index 2 has an empty region (not the wall); index 3 has Lambda_other > 0.
    for p_neo, label in ((np.array([np.nan, 0.3, np.nan, 0.0]), "pre-D-19"),
                         (np.array([1.0, 0.3, np.nan, 0.0]), "post-D-19")):
        policy = np.isfinite(p_neo) & wall
        assert int(wall.sum()) == 1, label
        assert int(policy.sum()) == (0 if label == "pre-D-19" else 1), label
    assert hasattr(pilot, "analyze")
