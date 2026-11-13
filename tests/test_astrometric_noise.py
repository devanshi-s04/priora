"""Tests for validation/astrometric_noise.py (WP8).

Three jobs, in order of what would hurt most if it were wrong:

1. **The error-model transcription.**  The whole experiment rests on Sorcha's
   formula having been copied correctly, so every constant and every branch is
   pinned against values worked out by hand from the published expressions —
   not against a previous run of this code.
2. **The noisy-attributable reconstruction.**  Zero noise must reproduce the
   input attributable bit-for-bit, in every mode; the position/rate
   decomposition must be exactly additive; ``pos`` must leave the rates
   *identical* and ``rate`` the positions.
3. **A small end-to-end smoke run** on 20 tracklets at a tiny grid, to catch
   wiring breakage between inject -> score -> analyze.  Kept well under a minute.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from validation import astrometric_noise as an

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_V0 = os.path.join(REPO, "validation", "output", "gate_v0")


# ==========================================================================
# 1. The error model, against hand-computed values
# ==========================================================================


def test_snr_at_m5_is_exactly_five():
    """``m5`` is the 5-sigma depth: at mag == m5, x = 1 and SNR = 1/sqrt(0.04)."""
    assert an.snr_from_mag(24.0, 24.0) == pytest.approx(5.0, rel=1e-14)
    assert an.snr_from_mag(21.5, 21.5) == pytest.approx(5.0, rel=1e-14)


@pytest.mark.parametrize(
    "mag, m5",
    [(18.0, 24.0), (21.0, 24.0), (23.0, 24.0), (24.0, 24.0), (25.0, 24.0),
     (22.0, 24.5)],
)
def test_snr_matches_hand_computation(mag, m5):
    """SNR = 1/sqrt((0.04 - g) x + g x^2), x = 10^(0.4 (m - m5)), g = 0.039.

    Recomputed here from the closed form with plain ``math``, independently of
    the module's numpy path.
    """
    g = 0.039
    x = math.pow(10.0, 0.4 * (mag - m5))
    want = 1.0 / math.sqrt((0.04 - g) * x + g * x * x)
    assert float(an.snr_from_mag(mag, m5)) == pytest.approx(want, rel=1e-14)


def test_random_error_is_coeff_times_fwhm_over_snr():
    """``astErrCoeff * FWHMeff / SNR`` — Sorcha's one-liner, verbatim."""
    assert an.random_astrometric_error_per_coord(700.0, 5.0) == pytest.approx(84.0, rel=1e-14)
    assert an.random_astrometric_error_per_coord(700.0, 5.0, 1.0) == pytest.approx(140.0, rel=1e-14)
    assert an.random_astrometric_error_per_coord(500.0, 10.0, 0.55) == pytest.approx(27.5, rel=1e-14)


def test_sigma_at_the_five_sigma_limit_is_hand_computable():
    """At V = m5: SNR = 5 exactly, sigma_rand = 0.6*700/5 = 84 mas exactly,
    sigma = sqrt(10^2 + 84^2) = sqrt(7156) mas."""
    sigma, snr, rand = an.astrometric_uncertainty_mas(24.0, 24.0)
    assert float(snr) == pytest.approx(5.0, rel=1e-14)
    assert float(rand) == pytest.approx(84.0, rel=1e-14)
    assert float(sigma) == pytest.approx(math.sqrt(7156.0), rel=1e-14)
    assert float(sigma) == pytest.approx(84.593144, abs=1e-6)


@pytest.mark.parametrize(
    "V, want_snr, want_sigma_mas",
    [
        # Hand-computed from the closed form (see the module docstring table).
        # (recomputed in exact decimal arithmetic; see the docstring table)
        (18.0, 466.293964, 10.040483),
        (21.0, 67.673075, 11.769378),
        (23.0, 12.328600, 35.504496),
        (24.0, 5.000000, 84.593144),
    ],
)
def test_sigma_v_table_matches_the_documented_values(V, want_snr, want_sigma_mas):
    """The sigma(V) table quoted in the module docstring and in REPORT.md."""
    sigma, snr, _ = an.astrometric_uncertainty_mas(V, 24.0)
    assert float(snr) == pytest.approx(want_snr, rel=1e-6)
    assert float(sigma) == pytest.approx(want_sigma_mas, rel=1e-6)


def test_systematic_floor_dominates_at_the_bright_end():
    """error_sys = 10 mas is a floor, so sigma -> 10 mas as SNR -> infinity."""
    sigma, _, _ = an.astrometric_uncertainty_mas(10.0, 24.0)
    assert float(sigma) == pytest.approx(10.0, abs=1e-4)
    assert float(sigma) >= 10.0
    # ...and it is a floor, not an asymptote reached from below.
    assert float(an.astrometric_uncertainty_mas(24.0, 24.0)[0]) > 10.0


def test_nvisit_scaling_is_sqrt_n_on_the_random_term_only():
    """Sorcha divides only ``error_rand`` by sqrt(nvisit); the floor stays."""
    s1, _, r1 = an.astrometric_uncertainty_mas(24.0, 24.0, nvisit=1)
    s4, _, r4 = an.astrometric_uncertainty_mas(24.0, 24.0, nvisit=4)
    assert float(r4) == pytest.approx(float(r1) / 2.0, rel=1e-14)
    assert float(s4) == pytest.approx(math.sqrt(100.0 + (84.0 / 2.0) ** 2), rel=1e-14)


def test_trailing_loss_matches_veres_chesley_by_hand():
    """dmag = 1.25 log10(1 + a x^2/(1 + b x)), x = (v/24) texp / seeing."""
    v_deg_day, seeing, texp = 1.15, 0.7, 30.0
    x = (v_deg_day / 24.0) * texp / seeing
    want = 1.25 * math.log10(1.0 + 0.761 * x * x / (1.0 + 1.162 * x))
    got = float(an.trailing_loss_dmag(v_deg_day, 0.0, seeing, texp, "trailedSource"))
    assert got == pytest.approx(want, rel=1e-13)
    # The RA/Dec components combine in quadrature.
    diag = float(an.trailing_loss_dmag(v_deg_day / math.sqrt(2.0),
                                       v_deg_day / math.sqrt(2.0), seeing, texp))
    assert diag == pytest.approx(want, rel=1e-13)


def test_trailing_loss_is_zero_for_a_static_source_and_monotone_in_rate():
    assert float(an.trailing_loss_dmag(0.0, 0.0, 0.7)) == 0.0
    rates = np.array([0.0, 0.1, 0.5, 1.0, 5.0])
    dm = an.trailing_loss_dmag(rates, np.zeros_like(rates), 0.7)
    assert np.all(np.diff(dm) > 0)


def test_circular_psf_model_uses_its_own_coefficients():
    x = (1.0 / 24.0) * 30.0 / 0.7
    want = 1.25 * math.log10(1.0 + 0.420 * x * x / (1.0 + 0.003 * x))
    assert float(an.trailing_loss_dmag(1.0, 0.0, 0.7, 30.0, "circularPSF")) == pytest.approx(
        want, rel=1e-13
    )
    with pytest.raises(ValueError):
        an.trailing_loss_dmag(1.0, 0.0, 0.7, 30.0, "nope")


def test_sigma_pipeline_units_and_cos_dec_metric():
    """``sigma_astrometric_rad`` converts mas -> rad and uses RA*cos(dec)."""
    v = np.array([24.0])
    dec = np.array([np.radians(60.0)])
    # ra_rate chosen so that ra_rate*cos(dec) is exactly 1 deg/day.
    ra_rate = np.radians(np.array([1.0 / math.cos(math.radians(60.0))]))
    dec_rate = np.zeros(1)
    em = an.ErrorModel()
    out = an.sigma_astrometric_rad(v, ra_rate, dec_rate, dec, em)
    assert out["rate_deg_day"][0] == pytest.approx(1.0, rel=1e-12)
    want_dmag = float(an.trailing_loss_dmag(1.0, 0.0, 0.7, 30.0, "trailedSource"))
    assert out["dmag_trail"][0] == pytest.approx(want_dmag, rel=1e-12)
    want_sig, _, _ = an.astrometric_uncertainty_mas(24.0 + want_dmag, 24.0)
    assert out["sigma_mas"][0] == pytest.approx(float(want_sig), rel=1e-12)
    assert out["sigma_rad"][0] == pytest.approx(
        math.radians(float(want_sig) / 3.6e6), rel=1e-14
    )


def test_trailing_can_be_switched_off():
    v = np.array([23.0])
    kw = dict(ra_rate=np.array([0.02]), dec_rate=np.array([0.01]),
              dec=np.array([0.1]))
    on = an.sigma_astrometric_rad(v, em=an.ErrorModel(trailing_losses=True), **kw)
    off = an.sigma_astrometric_rad(v, em=an.ErrorModel(trailing_losses=False), **kw)
    assert off["dmag_trail"][0] == 0.0
    assert on["sigma_mas"][0] > off["sigma_mas"][0]


# ==========================================================================
# 2. The noisy-attributable reconstruction
# ==========================================================================


def _toy(n: int = 64, seed: int = 3) -> dict[str, np.ndarray]:
    """A synthetic catalogue in the gen.npz schema (no gate_v0 needed)."""
    rng = np.random.default_rng(seed)
    return {
        "ra": rng.uniform(0.0, 2.0 * np.pi, n),
        "dec": rng.uniform(-1.3, 1.3, n),
        "ra_rate": rng.normal(0.0, 0.02, n),
        "dec_rate": rng.normal(0.0, 0.01, n),
        "v_mag": rng.uniform(16.0, 24.0, n),
        "is_neo": rng.random(n) < 0.2,
        "H": rng.uniform(15.0, 25.0, n),
    }


@pytest.mark.parametrize("mode", an.MODES)
def test_zero_noise_round_trip_is_bit_identical(mode):
    """With sigma identically zero the perturbed attributable IS the input.

    ``error_sys = 0`` and ``astErrCoeff = 0`` kill both terms of Sorcha's
    sigma, so every offset is exactly 0.0 and the reconstruction must return
    the original arrays to the last bit -- not merely to within a tolerance.
    """
    gen = _toy()
    rows = np.arange(gen["ra"].size)
    em0 = an.ErrorModel(error_sys_mas=0.0, ast_err_coeff=0.0)
    out = an.noisy_attributable(gen, rows, em0, an.NoiseConfig(mode=mode))
    assert np.all(out["sigma_mas"] == 0.0)
    np.testing.assert_array_equal(out["ra"], np.mod(gen["ra"], 2.0 * np.pi))
    for key in ("dec", "ra_rate", "dec_rate", "v_mag"):
        np.testing.assert_array_equal(out[key], gen[key])


def test_v_mag_is_never_perturbed():
    """Photometric noise is out of scope: V passes through untouched."""
    gen = _toy()
    rows = np.arange(gen["ra"].size)
    out = an.noisy_attributable(gen, rows, an.ErrorModel(), an.NoiseConfig())
    np.testing.assert_array_equal(out["v_mag"], gen["v_mag"])


def test_position_only_mode_leaves_the_rates_exactly_alone():
    gen = _toy()
    rows = np.arange(gen["ra"].size)
    out = an.noisy_attributable(gen, rows, an.ErrorModel(), an.NoiseConfig(mode="pos"))
    assert np.all(out["d_ra_rate"] == 0.0)
    assert np.all(out["d_dec_rate"] == 0.0)
    np.testing.assert_array_equal(out["ra_rate"], gen["ra_rate"])
    np.testing.assert_array_equal(out["dec_rate"], gen["dec_rate"])
    assert np.any(out["d_ra"] != 0.0)  # ...but the position really did move


def test_rate_only_mode_leaves_the_position_exactly_alone():
    gen = _toy()
    rows = np.arange(gen["ra"].size)
    out = an.noisy_attributable(gen, rows, an.ErrorModel(), an.NoiseConfig(mode="rate"))
    assert np.all(out["d_ra"] == 0.0)
    assert np.all(out["d_dec"] == 0.0)
    np.testing.assert_array_equal(out["ra"], np.mod(gen["ra"], 2.0 * np.pi))
    np.testing.assert_array_equal(out["dec"], gen["dec"])
    assert np.any(out["d_ra_rate"] != 0.0)


def test_decomposition_is_additive():
    """``both = pos + rate`` on the 4-vector perturbation, to rounding."""
    gen = _toy()
    rows = np.arange(gen["ra"].size)
    em = an.ErrorModel()
    got = {m: an.noisy_attributable(gen, rows, em, an.NoiseConfig(stream=7, mode=m))
           for m in an.MODES}
    for key in ("d_ra", "d_dec", "d_ra_rate", "d_dec_rate"):
        np.testing.assert_allclose(
            got["both"][key], got["pos"][key] + got["rate"][key],
            rtol=1e-12, atol=1e-18,
        )
    # The position halves are *identical*, not merely close: `pos` reuses the
    # very same offsets `both` applies to detection 0.
    for key in ("d_ra", "d_dec"):
        np.testing.assert_array_equal(got["both"][key], got["pos"][key])
    # The rate halves are the same quantity computed by a different association
    # -- (s e1 - s e0) versus s (e1 - e0) -- so they agree to rounding, not to
    # the bit.
    for key in ("d_ra_rate", "d_dec_rate"):
        np.testing.assert_allclose(got["both"][key], got["rate"][key],
                                   rtol=1e-12, atol=0.0)


def test_rate_perturbation_is_the_endpoint_difference_over_dt():
    gen = _toy()
    rows = np.arange(gen["ra"].size)
    nc = an.NoiseConfig(dt_min=45.0, stream=11)
    out = an.noisy_attributable(gen, rows, an.ErrorModel(), nc)
    dt = 45.0 / 1440.0
    np.testing.assert_allclose(
        out["d_ra_rate"], (out["d_ra_1"] - out["d_ra_0"]) / dt, rtol=1e-13
    )
    np.testing.assert_allclose(
        out["d_dec_rate"], (out["d_dec_1"] - out["d_dec_0"]) / dt, rtol=1e-13
    )
    np.testing.assert_array_equal(out["d_ra"], out["d_ra_0"])


def test_injected_scatter_has_the_requested_sigma():
    """Statistical check: the Dec offsets are N(0, sigma) and the RA offsets
    N(0, sigma/cos dec) — the local-sky-metric convention."""
    n = 20000
    rng = np.random.default_rng(0)
    gen = {
        "ra": np.zeros(n), "dec": np.full(n, np.radians(60.0)),
        "ra_rate": np.zeros(n), "dec_rate": np.zeros(n),
        "v_mag": np.full(n, 24.0),
    }
    rows = np.arange(n)
    out = an.noisy_attributable(gen, rows, an.ErrorModel(trailing_losses=False),
                               an.NoiseConfig(stream=1))
    sig = out["sigma_rad"][0]
    assert np.std(out["d_dec_0"]) == pytest.approx(sig, rel=0.03)
    assert np.std(out["d_ra_0"]) == pytest.approx(sig / math.cos(math.radians(60.0)),
                                                  rel=0.03)
    # Rate error per coordinate is sqrt(2) sigma / dt.
    dt = an.DT_MIN_DEFAULT / 1440.0
    assert np.std(out["d_dec_rate"]) == pytest.approx(math.sqrt(2.0) * sig / dt, rel=0.03)


def test_injection_is_deterministic_and_stream_separated():
    gen = _toy()
    rows = np.arange(gen["ra"].size)
    em = an.ErrorModel()
    a1 = an.noisy_attributable(gen, rows, em, an.NoiseConfig(seed=5, stream=2))
    a2 = an.noisy_attributable(gen, rows, em, an.NoiseConfig(seed=5, stream=2))
    b = an.noisy_attributable(gen, rows, em, an.NoiseConfig(seed=5, stream=3))
    np.testing.assert_array_equal(a1["ra"], a2["ra"])
    np.testing.assert_array_equal(a1["dec_rate"], a2["dec_rate"])
    assert not np.array_equal(a1["ra"], b["ra"])


def test_bad_mode_is_rejected():
    with pytest.raises(ValueError):
        an.perturbations(np.zeros(2), np.zeros(2), np.zeros((2, 2)),
                         np.zeros((2, 2)), 0.03, mode="sideways")


def test_stratified_subsample_keeps_every_neo_and_spans_v():
    rng = np.random.default_rng(1)
    n = 5000
    is_neo = rng.random(n) < 0.01
    v = rng.uniform(14.0, 24.0, n)
    sub = an.stratified_subsample(is_neo, v, n=500, seed=7)
    assert sub.size == 500
    assert np.array_equal(np.unique(sub), sub)  # sorted, unique
    assert np.all(is_neo[np.flatnonzero(is_neo)])  # sanity
    assert set(np.flatnonzero(is_neo)) <= set(sub.tolist())
    counts = np.histogram(v[sub], bins=list(an.V_STRATA))[0]
    assert np.all(counts > 0)  # every stratum represented


# ==========================================================================
# 3. Pure-numpy analysis helpers
# ==========================================================================


def test_threshold_crossings_counts_both_directions():
    p0 = np.array([0.2, 0.6, 0.6, 0.2, np.nan])
    p1 = np.array([0.6, 0.2, 0.6, 0.2, 0.9])
    y = np.array([True, True, False, False, True])
    (row,) = an.threshold_crossings(p0, p1, y, thresholds=(0.5,))
    assert row["n_flagged_noiseless"] == 2
    assert row["n_flagged_noisy"] == 2
    assert row["n_cross_up"] == 1
    assert row["n_cross_down"] == 1
    assert row["neo_cross_down"] == 1
    assert row["n_cross_any"] == 2


def test_matched_threshold_and_operating_point():
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.05])
    y = np.array([True, False, True, False, True, False])
    w = np.ones(6)
    thr = an.matched_threshold(p, y, w, 2.0 / 3.0)
    assert thr == pytest.approx(0.7)
    op = an.operating_points(p, p, y, w, target=2.0 / 3.0)
    assert op["noiseless"]["n_flagged"] == 3
    assert op["noiseless"]["completeness"] == pytest.approx(2.0 / 3.0)
    assert op["noiseless"]["contamination"] == pytest.approx(1.0 / 3.0)
    # Identical scores in both arms -> the fixed-threshold reading is identical.
    assert op["noisy_fixed_threshold"]["contamination"] == pytest.approx(1.0 / 3.0)


def test_delta_summary_and_binning():
    d = np.array([0.0, 0.1, -0.2, np.nan, 0.3])
    s = an.delta_summary(d, "x")
    assert s["n"] == 4
    assert s["max_abs"] == pytest.approx(0.3)
    assert s["mean"] == pytest.approx(0.05)
    rows = an.binned_delta(np.array([1.0, 2.0, 3.0, 4.0]),
                           np.array([0.1, 0.2, 0.3, 0.4]),
                           np.array([0.0, 2.5, 5.0]))
    assert [r["n"] for r in rows] == [2, 2]
    assert rows[0]["abs_median"] == pytest.approx(0.15)


def test_calibration_block_runs_and_reports_a_verdict():
    rng = np.random.default_rng(4)
    p = rng.beta(0.3, 8.0, 3000)
    y = rng.random(3000) < p
    out = an.calibration_block(p, y, np.ones(3000), n_boot=60, seed=1)
    assert out["n"] == 3000
    assert "pass" in out["verdict"]
    assert np.isfinite(out["odds_scale"]["c_hat"])
    assert out["odds_scale"]["c_hat"] == pytest.approx(1.0, abs=0.5)


# ==========================================================================
# 4. End-to-end smoke: inject -> score -> analyze on 20 tracklets
# ==========================================================================


@pytest.mark.skipif(
    not os.path.exists(os.path.join(GATE_V0, "scores.npz")),
    reason="needs the gate_v0 run (validation/output/ is gitignored)",
)
def test_end_to_end_smoke(tmp_path):
    """20 tracklets, a deliberately tiny (40 x 6) grid, 3 MC realizations.

    Not a science run — the grid is far too coarse for the scores to mean
    anything.  It exists to catch wiring breakage across the full pipeline:
    the gen.npz schema the WP6 scorer consumes, the chunk/stamp round trip, the
    manifest, the analysis joins and the report/plot writers.
    """
    src = tmp_path / "src"
    src.mkdir()
    with np.load(os.path.join(GATE_V0, "gen.npz")) as z:
        gen = {k: z[k] for k in z.files}
    with np.load(os.path.join(GATE_V0, "scores.npz")) as z:
        scores = {k: z[k] for k in z.files}
    # Take 20 rows including a few NEOs, so the analysis has both classes.
    neo = np.flatnonzero(gen["is_neo"])[:4]
    other = np.flatnonzero(~gen["is_neo"])[:16]
    take = np.sort(np.concatenate([neo, other]))
    np.savez_compressed(src / "gen.npz", **{k: v[take] for k, v in gen.items()})
    order = np.argsort(scores["index"].astype(int), kind="mergesort")
    sub = {k: v[order][take] for k, v in scores.items()}
    sub["index"] = np.arange(take.size, dtype=np.int64)
    np.savez_compressed(src / "scores.npz", **sub)
    with open(src / "gen_counts.json", "w") as fh:
        json.dump({"config": {"rho_floor": 0.01}, "counters": {}}, fh)

    out = tmp_path / "out"
    man = an.inject(source_run=str(src), out_dir=str(out), n_mc=3, n_sub=12,
                    m5_sens=24.5, verbose=False)
    assert len(man["runs"]) == 3 + 1 + 2 + 1  # main + 3 mc + 2 decomp + 1 sens
    assert os.path.exists(out / "manifest.json")

    from validation.self_consistency import ScoreConfig

    cfg = ScoreConfig(n_rho=40, n_rhodot=6)
    an.score_runs(out_dir=str(out), cfg=cfg, workers=1, chunk=10, verbose=False)
    assert os.path.exists(out / "noisy_scores.npz")
    for entry in man["runs"]:
        assert os.path.exists(out / entry["name"] / "scores.npz")

    # Resuming must be a no-op, not a recompute.
    again = an.score_runs(out_dir=str(out), cfg=cfg, workers=1, chunk=10, verbose=False)
    assert again["scored"] == []

    summary = an.analyze(out_dir=str(out), plots=True, n_boot=40, verbose=False)
    assert summary["n"] == 20
    assert np.isfinite(summary["ranking"]["auc_noisy"])
    assert summary["mc"]["n_realizations"] == 3
    assert summary["decomposition"]["n"] == 12
    assert os.path.exists(out / "REPORT.md")
    assert os.path.exists(out / "summary.json")
    for png in ("error_model.png", "delta_vs_observables.png", "calibration.png",
                "ranking.png", "mc_sensitivity.png", "decomposition.png"):
        assert os.path.exists(out / png), png
    text = (out / "REPORT.md").read_text()
    assert "Sorcha" in text and "Bottom line" in text
    # The report must not silently print stale/NaN-free nonsense: the sigma
    # table it quotes has to be the one the injection actually used.
    assert f"{summary['sigma_table'][-1]['sigma_mas']:.2f}" in text


@pytest.mark.skipif(
    not os.path.exists(os.path.join(GATE_V0, "gen.npz")),
    reason="needs the gate_v0 catalogue",
)
def test_inject_never_writes_into_the_source_run(tmp_path):
    """gate_v0 is read-only for this harness."""
    before = sorted(os.listdir(GATE_V0))
    stamps = {f: os.stat(os.path.join(GATE_V0, f)).st_mtime for f in before}
    an.inject(source_run=GATE_V0, out_dir=str(tmp_path / "o"), n_mc=1, n_sub=8,
              m5_sens=None, full=False, verbose=False)
    assert sorted(os.listdir(GATE_V0)) == before
    assert {f: os.stat(os.path.join(GATE_V0, f)).st_mtime for f in before} == stamps


# ==========================================================================
# 5. The arc-length sweep
# ==========================================================================
#
# The sweep's whole claim to sharpness is that dt is the ONLY thing that
# changes: same draws, same endpoints, only the 1/dt divisor.  Section 5a pins
# that construction at every arc length; 5b pins the scaling machinery that
# reads the result; 5c is a wiring smoke test through the shortest arc.


# --- 5a. dt parameterization of the endpoint construction -----------------


@pytest.mark.parametrize("dt_min", list(an.DT_GRID_DEFAULT) + [1.0, 240.0])
@pytest.mark.parametrize("mode", an.MODES)
def test_zero_noise_round_trip_is_bit_identical_at_any_dt(dt_min, mode):
    """Zero sigma must reproduce the input attributable at EVERY arc length.

    dt appears only as a divisor of a perturbation that is identically 0.0, so
    the reconstruction must be exact, not approximate — including at dt = 1 min,
    where a 1/dt of 1440/day would amplify any stray epsilon by three orders of
    magnitude and show up immediately.
    """
    gen = _toy()
    rows = np.arange(gen["ra"].size)
    em0 = an.ErrorModel(error_sys_mas=0.0, ast_err_coeff=0.0)
    out = an.noisy_attributable(gen, rows, em0,
                                an.NoiseConfig(dt_min=dt_min, mode=mode))
    assert np.all(out["sigma_mas"] == 0.0)
    np.testing.assert_array_equal(out["ra"], np.mod(gen["ra"], 2.0 * np.pi))
    for key in ("dec", "ra_rate", "dec_rate", "v_mag"):
        np.testing.assert_array_equal(out[key], gen[key])
    for key in ("d_ra_rate", "d_dec_rate", "d_ra", "d_dec"):
        assert np.all(out[key] == 0.0)


def test_endpoint_offsets_are_bit_identical_across_dt():
    """Common random numbers in dt: the *endpoints* do not know about dt.

    This is the property the whole sweep rests on — at fixed (seed, stream,
    rows) every arc length perturbs the same two detections by the same
    offsets, so |Delta P| at two arc lengths is a paired comparison.  It holds
    because ``noisy_attributable`` draws e0/e1 before dt is used at all.
    """
    gen = _toy()
    rows = np.arange(gen["ra"].size)
    em = an.ErrorModel()
    ref = an.noisy_attributable(gen, rows, em, an.NoiseConfig(dt_min=60.0, stream=4))
    for dt in (45.0, 30.0, 15.0, 10.0, 5.0):
        got = an.noisy_attributable(gen, rows, em, an.NoiseConfig(dt_min=dt, stream=4))
        for key in ("d_ra_0", "d_dec_0", "d_ra_1", "d_dec_1", "sigma_mas",
                    "d_ra", "d_dec"):
            np.testing.assert_array_equal(got[key], ref[key], err_msg=key)
        # ...and the position is therefore also identical.
        np.testing.assert_array_equal(got["ra"], ref["ra"])
        np.testing.assert_array_equal(got["dec"], ref["dec"])


@pytest.mark.parametrize("dt_min", [45.0, 15.0, 5.0])
def test_rate_perturbation_scales_exactly_as_one_over_dt(dt_min):
    """The rate error is ``(d_1 - d_0)/dt``, so ``d_rate * dt`` is dt-invariant."""
    gen = _toy()
    rows = np.arange(gen["ra"].size)
    em = an.ErrorModel()
    ref = an.noisy_attributable(gen, rows, em, an.NoiseConfig(dt_min=60.0, stream=9))
    got = an.noisy_attributable(gen, rows, em, an.NoiseConfig(dt_min=dt_min, stream=9))
    for key in ("d_ra_rate", "d_dec_rate"):
        np.testing.assert_allclose(got[key] * dt_min, ref[key] * 60.0,
                                   rtol=1e-13, atol=0.0, err_msg=key)
        # and the amplification really is 60/dt, not 1
        np.testing.assert_allclose(
            got[key], ref[key] * (60.0 / dt_min), rtol=1e-13, atol=0.0
        )


def test_position_only_noise_is_completely_dt_independent():
    """``pos`` mode is the sweep's control: identical at every arc length.

    A rigid translation of the tracklet cannot depend on how far apart its two
    detections are, so the whole perturbed catalogue must be bit-identical
    across dt.  (The sweep exploits exactly this to adopt one scoring pass of
    ``decomp_pos`` for every arc length.)
    """
    gen = _toy()
    rows = np.arange(gen["ra"].size)
    em = an.ErrorModel()
    ref = an.noisy_attributable(gen, rows, em,
                                an.NoiseConfig(dt_min=60.0, stream=2, mode="pos"))
    for dt in (45.0, 5.0):
        got = an.noisy_attributable(gen, rows, em,
                                    an.NoiseConfig(dt_min=dt, stream=2, mode="pos"))
        for key in ("ra", "dec", "ra_rate", "dec_rate"):
            np.testing.assert_array_equal(got[key], ref[key], err_msg=key)


def test_rate_error_block_scales_as_one_over_dt():
    """``sqrt(2) sigma / dt`` in arcsec/day, and its ratio to the own motion."""
    n = 200
    nz = {"sigma_mas": np.full(n, 84.593144)}
    rate = np.full(n, 0.24)                       # deg/day
    a = an.rate_error_block(nz, rate, 60.0)
    b = an.rate_error_block(nz, rate, 15.0)
    want_60 = math.sqrt(2.0) * 0.084593144 / (60.0 / 1440.0)
    assert a["sigma_rate_arcsec_per_day_per_coord"]["p50"] == pytest.approx(
        want_60, rel=1e-12)
    assert b["sigma_rate_arcsec_per_day_per_coord"]["p50"] == pytest.approx(
        4.0 * want_60, rel=1e-12)
    assert b["rate_error_over_own_motion"]["p50"] == pytest.approx(
        4.0 * a["rate_error_over_own_motion"]["p50"], rel=1e-12)
    assert a["rate_error_over_own_motion"]["p50"] == pytest.approx(
        want_60 / (0.24 * 3600.0), rel=1e-12)


# --- 5b. the scaling machinery, on synthetic inputs -----------------------


@pytest.mark.parametrize("k", [-2.0, -1.0, -0.5, 0.0, 1.5])
def test_loglog_slope_recovers_an_exact_power_law(k):
    x = np.array([60.0, 45.0, 30.0, 15.0, 10.0, 5.0])
    y = 3.7 * x ** k
    got = an.loglog_slope(x, y)
    assert got["slope"] == pytest.approx(k, abs=1e-10)
    assert got["intercept"] == pytest.approx(math.log(3.7), abs=1e-9)
    assert got["n"] == 6
    if k != 0.0:
        assert got["r2"] == pytest.approx(1.0, abs=1e-12)


def test_loglog_slope_ignores_nonpositive_and_nonfinite():
    x = np.array([1.0, 2.0, 4.0, -1.0, 8.0, 16.0])
    y = np.array([1.0, 0.5, 0.25, 5.0, np.nan, 0.0625])
    got = an.loglog_slope(x, y)
    assert got["n"] == 4                      # (1,1) (2,.5) (4,.25) (16,.0625)
    assert got["slope"] == pytest.approx(-1.0, abs=1e-12)
    assert not np.isfinite(an.loglog_slope([1.0], [1.0])["slope"])


def _exact_one_over_dt(dt, amp):
    """``M[k, i] = amp[i] / dt[k]`` — the exactly-linear-response synthetic."""
    return np.asarray(amp)[None, :] / np.asarray(dt, dtype=float)[:, None]


def test_scaling_ratio_is_exactly_one_for_a_pure_one_over_dt_law():
    dt = np.array([60.0, 45.0, 30.0, 15.0, 10.0, 5.0])
    amp = np.array([1e-6, 3e-4, 2e-2, 0.1])
    rows = an.scaling_ratio(dt, _exact_one_over_dt(dt, amp), ref=0)
    assert [r["dt_min"] for r in rows] == list(dt)
    for r in rows:
        assert r["n"] == 4
        assert r["ratio_p50"] == pytest.approx(1.0, rel=1e-12)
        assert r["ratio_p10"] == pytest.approx(1.0, rel=1e-12)
        assert r["ratio_p90"] == pytest.approx(1.0, rel=1e-12)


def test_scaling_ratio_detects_a_saturating_response():
    """Halve the response below 20 min: the ratio must read 0.5 there and 1 above."""
    dt = np.array([60.0, 45.0, 30.0, 15.0, 10.0, 5.0])
    amp = np.array([1e-3, 1e-2, 1e-1])
    M = _exact_one_over_dt(dt, amp)
    M[dt < 20.0] *= 0.5
    rows = an.scaling_ratio(dt, M, ref=0)
    got = {r["dt_min"]: r["ratio_p50"] for r in rows}
    for t in (60.0, 45.0, 30.0):
        assert got[t] == pytest.approx(1.0, rel=1e-12)
    for t in (15.0, 10.0, 5.0):
        assert got[t] == pytest.approx(0.5, rel=1e-12)


def test_scaling_ratio_floor_and_argument_validation():
    dt = np.array([60.0, 30.0])
    M = _exact_one_over_dt(dt, np.array([1e-12, 1.0]))
    assert an.scaling_ratio(dt, M, ref=0, floor=1e-9)[0]["n"] == 1
    assert an.scaling_ratio(dt, M, ref=0, floor=0.0)[0]["n"] == 2
    with pytest.raises(ValueError):
        an.scaling_ratio(dt, M[:1], ref=0)
    with pytest.raises(ValueError):
        an.scaling_ratio(dt, M, ref=7)


def test_variance_scaling_on_an_exactly_linear_response():
    """The assertion the sweep actually makes: std ∝ 1/dt, var ∝ 1/dt²."""
    dt = np.array([15.0, 60.0, 5.0, 30.0, 45.0, 10.0])      # deliberately unsorted
    amp = np.geomspace(1e-6, 1e-1, 40)
    out = an.variance_scaling(dt, _exact_one_over_dt(dt, amp), floor=1e-12)
    # It sorts to longest-arc-first and anchors there.
    assert out["dt_min"] == [60.0, 45.0, 30.0, 15.0, 10.0, 5.0]
    assert out["anchor_dt_min"] == 60.0
    assert out["n_fitted"] == 40
    assert out["per_tracklet_slope_std"]["p50"] == pytest.approx(-1.0, abs=1e-9)
    assert out["per_tracklet_slope_std"]["p10"] == pytest.approx(-1.0, abs=1e-9)
    assert out["per_tracklet_slope_std"]["p90"] == pytest.approx(-1.0, abs=1e-9)
    assert out["per_tracklet_slope_var"]["p50"] == pytest.approx(-2.0, abs=1e-9)
    for key in ("p50", "p90", "p99", "max", "rms"):
        assert out["aggregate_slope_std"][key]["slope"] == pytest.approx(-1.0, abs=1e-9)
        assert out["aggregate_slope_std"][key]["r2"] == pytest.approx(1.0, abs=1e-12)
    assert out["long_arc_slope_std"]["slope"] == pytest.approx(-1.0, abs=1e-9)
    assert out["short_arc_slope_std"]["slope"] == pytest.approx(-1.0, abs=1e-9)
    # A pure power law never departs from itself.
    assert out["departure_dt_min"] is None
    assert all(r["ratio_p50"] == pytest.approx(1.0, rel=1e-12)
               for r in out["linear_scaling_ratio"])
    # ...and the percentile curves really do rise by 12x from 60 to 5 min.
    c = {r["dt_min"]: r for r in out["curves"]}
    assert c[5.0]["p90"] / c[60.0]["p90"] == pytest.approx(12.0, rel=1e-10)


def test_variance_scaling_localises_a_departure_from_the_one_over_dt_law():
    dt = np.array([60.0, 45.0, 30.0, 15.0, 10.0, 5.0])
    amp = np.geomspace(1e-5, 1e-2, 30)
    S = _exact_one_over_dt(dt, amp)
    S[dt <= 15.0] *= 0.4                     # response saturates on short arcs
    out = an.variance_scaling(dt, S, floor=1e-12, tol=0.10)
    assert out["departure_dt_min"] == 15.0
    # Shallower than -1 overall, still exactly -1 on the long-arc half.
    assert -1.0 < out["per_tracklet_slope_std"]["p50"] < -0.4
    assert out["long_arc_slope_std"]["slope"] == pytest.approx(-1.0, abs=1e-9)
    assert out["short_arc_slope_std"]["slope"] == pytest.approx(-1.0, abs=1e-9)
    got = {r["dt_min"]: r["ratio_p50"] for r in out["linear_scaling_ratio"]}
    assert got[30.0] == pytest.approx(1.0, rel=1e-12)
    assert got[5.0] == pytest.approx(0.4, rel=1e-12)


def test_variance_scaling_tolerance_is_respected_and_shapes_are_checked():
    dt = np.array([60.0, 30.0, 15.0])
    S = _exact_one_over_dt(dt, np.geomspace(1e-4, 1e-2, 12))
    S[2] *= 1.05                              # a 5% wobble
    assert an.variance_scaling(dt, S, floor=1e-12, tol=0.10)["departure_dt_min"] is None
    assert an.variance_scaling(dt, S, floor=1e-12, tol=0.01)["departure_dt_min"] == 15.0
    with pytest.raises(ValueError):
        an.variance_scaling(dt, S[:2], floor=1e-12)


def test_variance_scaling_survives_all_zero_tracklets():
    """Posteriors pinned at 0 or 1 have std == 0 at every dt and no power law.

    They must be excluded from the per-tracklet fit rather than poisoning it,
    while still counting in the population percentile curves.
    """
    dt = np.array([60.0, 15.0, 5.0])
    S = _exact_one_over_dt(dt, np.array([0.0, 0.0, 1e-3, 1e-2]))
    out = an.variance_scaling(dt, S, floor=1e-12)
    assert out["n_tracklets"] == 4 and out["n_fitted"] == 2
    assert out["per_tracklet_slope_std"]["p50"] == pytest.approx(-1.0, abs=1e-9)
    assert out["curves"][0]["n"] == 4
    assert out["curves"][0]["p50"] == pytest.approx(
        0.5 * (0.0 + 1e-3 / 60.0), rel=1e-12)


def test_ratio_by_floor_separates_real_departure_from_numerical_noise():
    """The floor ladder is the sweep's defence against measuring round-off.

    Two synthetic populations, both departing from the 1/dt law at low floors:
    one where the departure is a genuine property of large-response tracklets
    (survives raising the floor) and one where it is a fixed dt-independent
    jitter added to tiny-response tracklets (vanishes when they are excluded).
    Distinguishing these is the entire reason the ladder exists.
    """
    dt = np.array([60.0, 15.0])
    # (a) genuine: every tracklet saturates to half the linear prediction.
    amp = np.geomspace(1e-6, 1e-1, 400)
    real = _exact_one_over_dt(dt, amp)
    real[1] *= 0.5
    for r in an.ratio_by_floor(dt, real):
        if r["n"]:
            assert r["ratio_p50"][1] == pytest.approx(0.5, rel=1e-9)
    # (b) spurious: exact 1/dt plus a dt-INDEPENDENT 2e-6 jitter.  Tracklets
    # whose anchor is dominated by the jitter pull the ratio toward dt/dt_ref.
    fake = _exact_one_over_dt(dt, amp) + 2e-6
    ladder = {r["floor"]: r for r in an.ratio_by_floor(dt, fake)}
    assert ladder[0.0]["ratio_p50"][1] < 0.95        # contaminated
    assert ladder[1e-3]["ratio_p50"][1] == pytest.approx(1.0, abs=0.01)
    assert ladder[1e-3]["n"] < ladder[0.0]["n"]
    # The ladder is monotone in n and reports the requested floors in order.
    ns = [r["n"] for r in an.ratio_by_floor(dt, fake)]
    assert ns == sorted(ns, reverse=True)
    assert [r["floor"] for r in an.ratio_by_floor(dt, fake)] == \
        [float(f) for f in an.SCALING_FLOORS]


def test_step_scaling_ratio_separates_a_one_off_kink_from_progressive_saturation():
    """The anchor-free diagnostic: is the departure once, or compounding?

    Both synthetics have an *anchored* ratio that sits below 1 at every short
    arc, and they are indistinguishable that way.  The step ratios tell them
    apart: a one-off kink shows a single step below 1 and 1.000 thereafter,
    while progressive saturation shows every step below 1.
    """
    dt = np.array([60.0, 30.0, 15.0, 5.0])
    amp = np.geomspace(1e-2, 1.0, 200)          # anchors 1.7e-4 .. 1.7e-2
    kink = _exact_one_over_dt(dt, amp)
    kink[1:] *= 0.8                             # one 20% kink at 60 -> 30
    rows = an.step_scaling_ratio(dt, kink)
    assert [(r["dt_from"], r["dt_to"]) for r in rows] == \
        [(60.0, 30.0), (30.0, 15.0), (15.0, 5.0)]
    assert [r["factor"] for r in rows] == [2.0, 2.0, 3.0]
    assert rows[0]["step_p50"] == pytest.approx(0.8, rel=1e-9)
    assert rows[1]["step_p50"] == pytest.approx(1.0, rel=1e-9)
    assert rows[2]["step_p50"] == pytest.approx(1.0, rel=1e-9)
    # ...whereas the anchored ratio is 0.8 at all three short arcs and so
    # cannot distinguish the two cases at all.
    anch = an.scaling_ratio(dt, kink, ref=0, floor=an.SCALING_FLOOR_DEFAULT)
    assert [round(r["ratio_p50"], 6) for r in anch[1:]] == [0.8, 0.8, 0.8]

    creep = _exact_one_over_dt(dt, amp) * np.array([1.0, 0.8, 0.64, 0.512])[:, None]
    rows = an.step_scaling_ratio(dt, creep)
    assert all(r["step_p50"] == pytest.approx(0.8, rel=1e-9) for r in rows)
    with pytest.raises(ValueError):
        an.step_scaling_ratio(dt, creep[:2])


def test_ladder_stability_tells_a_real_departure_from_a_measurement_artefact():
    """Flat across floors = real; drifting across floors = artefact.

    This is the discriminator the sweep's interpretation turns on, so it is
    pinned on both cases explicitly.
    """
    flat = [
        {"floor": 0.0, "n": 400, "ratio_p50": [1.0, 0.79], "dt_min": [60.0, 5.0]},
        {"floor": 1e-4, "n": 100, "ratio_p50": [1.0, 0.81], "dt_min": [60.0, 5.0]},
        {"floor": 1e-3, "n": 30, "ratio_p50": [1.0, 0.78], "dt_min": [60.0, 5.0]},
        {"floor": 1e-2, "n": 2, "ratio_p50": [1.0, 0.20], "dt_min": [60.0, 5.0]},
    ]
    got = an.ladder_stability(flat, min_n=20)
    assert got["n_rows"] == 3                      # the n=2 row is excluded
    assert got["floors_used"] == [0.0, 1e-4, 1e-3]
    assert got["max_spread"] == pytest.approx(0.03, abs=1e-12)
    by = {r["dt_min"]: r for r in got["per_dt"]}
    assert by[5.0]["lo"] == pytest.approx(0.78) and by[5.0]["hi"] == pytest.approx(0.81)
    assert by[60.0]["spread"] == pytest.approx(0.0)

    drift = [
        {"floor": 0.0, "n": 400, "ratio_p50": [1.0, 0.40], "dt_min": [60.0, 5.0]},
        {"floor": 1e-4, "n": 100, "ratio_p50": [1.0, 0.70], "dt_min": [60.0, 5.0]},
        {"floor": 1e-3, "n": 30, "ratio_p50": [1.0, 0.97], "dt_min": [60.0, 5.0]},
    ]
    assert an.ladder_stability(drift, min_n=20)["max_spread"] == pytest.approx(0.57)
    assert an.ladder_stability([], min_n=1)["n_rows"] == 0
    assert an.ladder_stability(flat, min_n=10_000)["per_dt"] == []


def test_pick_departure_floor_takes_the_cleanest_adequate_row():
    ladder = [
        {"floor": 0.0, "n": 5000, "ratio_p50": [1.0, 0.8], "dt_min": [60.0, 15.0]},
        {"floor": 1e-4, "n": 200, "ratio_p50": [1.0, 0.95], "dt_min": [60.0, 15.0]},
        {"floor": 1e-3, "n": 40, "ratio_p50": [1.0, 0.99], "dt_min": [60.0, 15.0]},
        {"floor": 1e-2, "n": 3, "ratio_p50": [1.0, 0.5], "dt_min": [60.0, 15.0]},
    ]
    pick = an.pick_departure_floor(ladder, min_n=30)
    assert pick["floor"] == 1e-3 and pick["n"] == 40 and pick["adequate"] is True
    # Nothing adequate: fall back to the best-populated row, flagged as such.
    thin = an.pick_departure_floor(ladder, min_n=10000)
    assert thin["adequate"] is False and thin["n"] == 5000
    assert an.pick_departure_floor([], min_n=1)["n"] == 0


def test_variance_scaling_departure_uses_the_clean_subset_not_the_noisy_one():
    """A departure visible only in low-amplitude tracklets must not be reported.

    Small-anchor tracklets are contaminated by the roughness of v0's
    piecewise-constant prior, so a departure that exists only there is a
    property of the measurement, not of the classifier.  The block must read its
    verdict off the cleanest adequately-populated floor.
    """
    dt = np.array([60.0, 15.0])
    n_small, n_big = 4000, 60
    amp = np.concatenate([np.full(n_small, 1.2e-2),    # anchor 2e-4
                          np.full(n_big, 1.2)])        # anchor 2e-2
    S = _exact_one_over_dt(dt, amp)
    S[1, :n_small] *= 0.5                              # only the small ones depart
    out = an.variance_scaling(dt, S, min_n_for_departure=30)
    assert out["departure_floor"]["floor"] == 1e-2
    assert out["departure_floor"]["n"] == n_big
    assert out["departure_dt_min"] is None
    # ...and the contaminated low floor really would have flagged it.
    low = {r["floor"]: r for r in out["ratio_by_floor"]}
    assert low[0.0]["ratio_p50"][1] < 0.9
    # Make the big-amplitude tracklets depart too and it is reported.
    S[1, n_small:] *= 0.5
    out2 = an.variance_scaling(dt, S, min_n_for_departure=30)
    assert out2["departure_dt_min"] == 15.0


def test_variance_scaling_default_floor_is_above_the_quadrature_jitter():
    """The adopted default must sit well clear of the ~2e-6 measured jitter."""
    assert an.SCALING_FLOOR_DEFAULT == 1e-4
    assert an.SCALING_FLOOR_DEFAULT > 20 * 2e-6
    assert 0.0 in an.SCALING_FLOORS and an.SCALING_FLOOR_DEFAULT in an.SCALING_FLOORS
    dt = np.array([60.0, 5.0])
    # Anchors straddling the default: only the ones above it are fitted.
    S = _exact_one_over_dt(dt, np.array([1e-6, 6e-3, 1.0]))   # /60 -> 1.7e-8, 1e-4, 1.7e-2
    out = an.variance_scaling(dt, S)
    assert out["floor"] == 1e-4
    assert out["n_fitted"] == 1                              # only the 1.7e-2 anchor
    assert out["linear_scaling_ratio"][0]["n"] == 1


def test_variance_scaling_counts_tracklets_that_only_wake_up_on_short_arcs():
    """A tracklet dead at the anchor arc has no ratio and no slope — count it.

    This is the failure mode every power-law statistic in the block is blind to
    by construction: the classifier is perfectly certain at a long arc (std
    exactly 0, no finite ratio, no log) and stops being certain at a short one.
    """
    dt = np.array([60.0, 15.0, 5.0])
    S = np.zeros((3, 5))
    S[:, 0] = [1e-3, 4e-3, 1.2e-2]          # a normal 1/dt responder
    S[:, 1] = [0.0, 0.0, 0.02]              # dead until 5 min, then big
    S[:, 2] = [0.0, 1e-6, 3e-6]             # dead until 15 min, then tiny
    S[:, 3] = [0.0, 0.0, 0.0]               # dead everywhere
    S[:, 4] = [2e-4, 8e-4, 2.4e-3]          # another normal responder
    out = an.variance_scaling(dt, S, floor=1e-12)
    c = {r["dt_min"]: r for r in out["curves"]}
    assert c[60.0]["n_dead_at_anchor"] == 3
    assert c[60.0]["n_woken_from_dead"] == 0
    assert c[15.0]["n_woken_from_dead"] == 1 and c[15.0]["n_woken_gt_0p01"] == 0
    assert c[5.0]["n_woken_from_dead"] == 2 and c[5.0]["n_woken_gt_0p01"] == 1
    # The two live tracklets scale as 1/dt and are the only ones fitted.
    assert out["n_fitted"] == 2
    assert out["per_tracklet_slope_std"]["p50"] == pytest.approx(-1.0, abs=1e-9)


def _fake_block(dt, *, pv=0.5, churn_frac=0.0, lost=0.0, dauc=0.0,
                dauc_ci=(-0.001, 0.001), comp=0.881, n_sub=500):
    """A minimal per-dt block, only the fields ``break_verdict`` reads."""
    return {
        "tag": an.dt_tag(dt),
        "dt_min": float(dt),
        "calibration": {
            "global": {"z": -0.4},
            "odds_scale": {"c_hat": 0.92},
            "verdict": {"pass": pv > an.BREAK_ALPHA, "max_abs_z": 0.7,
                        "max_abs_z_threshold": 2.6, "max_abs_z_p_value": pv,
                        "global_z_threshold": 2.5, "global_z_p_value": pv},
        },
        "ranking": {"delta_auc": dauc, "delta_auc_ci95": list(dauc_ci)},
        "threshold_stability": [
            {"threshold": t, "n_cross_any": 0, "frac_cross_any": 0.0,
             "neo_cross_down": 0} for t in an.THRESHOLDS],
        "operating_points": {
            "noiseless": {"completeness": 0.881, "contamination": 0.063,
                          "threshold": 0.254},
            "noisy_fixed_threshold": {"completeness": comp, "contamination": 0.05},
        },
        "undefined": {"frac_lost_to_noise": lost},
        "mc": {"n_tracklets": n_sub, "expected_crossings": [
            {"threshold": t, "mean_crossings_per_realization": churn_frac * n_sub,
             "max_crossings_in_a_realization": 1,
             "mean_neo_lost_per_realization": 0.0, "n_flagged_noiseless": 170}
            for t in an.THRESHOLDS]},
    }


def test_break_verdict_finds_no_break_when_everything_is_clean():
    ver = an.break_verdict([_fake_block(t) for t in (60.0, 15.0, 5.0)])
    assert ver["any_break_in_range"] is False
    assert ver["break_onset_dt_min"] is None
    assert ver["shortest_safe_dt_min"] == 5.0
    assert ver["monotone_in_dt"] is True
    assert [r["dt_min"] for r in ver["rows"]] == [60.0, 15.0, 5.0]   # dt-descending


def test_break_verdict_arms_fire_independently_and_locate_the_onset():
    blocks = [
        _fake_block(60.0),
        _fake_block(15.0, churn_frac=0.02),                  # C2
        _fake_block(5.0, pv=0.001, lost=0.05),               # C1 + C3
    ]
    ver = an.break_verdict(blocks)
    by = {r["dt_min"]: r for r in ver["rows"]}
    assert by[60.0]["breaks"] is False
    assert by[15.0]["breaks"] and by[15.0]["C2_triage_churn"]
    assert not by[15.0]["C1_calibration"] and not by[15.0]["C3_lost_tracklets"]
    assert by[5.0]["C1_calibration"] and by[5.0]["C3_lost_tracklets"]
    # The onset is the LONGEST broken arc: everything shorter is inside it.
    assert ver["break_onset_dt_min"] == 15.0
    assert ver["shortest_safe_dt_min"] == 60.0
    assert ver["monotone_in_dt"] is True
    assert any("C2" in x for x in by[15.0]["reasons"])


def test_break_verdict_warnings_are_not_breaks():
    blocks = [
        _fake_block(60.0),
        _fake_block(5.0, dauc=-0.05, dauc_ci=(-0.06, -0.04), comp=0.85),
    ]
    ver = an.break_verdict(blocks)
    r = ver["rows"][-1]
    assert r["W1_ranking"] and r["W2_operating_point"]
    assert r["breaks"] is False and ver["any_break_in_range"] is False
    assert len(r["warnings"]) == 2
    # A CI that straddles zero is not a W1, however large the point estimate.
    ok = an.break_verdict([_fake_block(5.0, dauc=-0.05, dauc_ci=(-0.2, 0.1))])
    assert ok["rows"][0]["W1_ranking"] is False


def test_break_verdict_flags_a_non_monotone_pattern():
    blocks = [_fake_block(60.0, churn_frac=0.02), _fake_block(5.0)]
    ver = an.break_verdict(blocks)
    assert ver["monotone_in_dt"] is False


def test_dt_tag_is_stable_and_readable():
    assert an.dt_tag(45.0) == "dt_45min"
    assert an.dt_tag(5) == "dt_5min"
    assert an.dt_tag(7.5) == "dt_7.5min"
    assert len({an.dt_tag(t) for t in an.DT_GRID_DEFAULT}) == len(an.DT_GRID_DEFAULT)


# --- 5c. end-to-end smoke through the shortest arc ------------------------


def _tiny_source(tmp_path, n_neo: int = 3, n_other: int = 7):
    """A 10-row stand-in for ``gate_v0``: catalogue + stored noiseless scores."""
    src = tmp_path / "src"
    src.mkdir()
    with np.load(os.path.join(GATE_V0, "gen.npz")) as z:
        gen = {k: z[k] for k in z.files}
    with np.load(os.path.join(GATE_V0, "scores.npz")) as z:
        scores = {k: z[k] for k in z.files}
    take = np.sort(np.concatenate([np.flatnonzero(gen["is_neo"])[:n_neo],
                                   np.flatnonzero(~gen["is_neo"])[:n_other]]))
    np.savez_compressed(src / "gen.npz", **{k: v[take] for k, v in gen.items()})
    order = np.argsort(scores["index"].astype(int), kind="mergesort")
    sub = {k: v[order][take] for k, v in scores.items()}
    sub["index"] = np.arange(take.size, dtype=np.int64)
    np.savez_compressed(src / "scores.npz", **sub)
    with open(src / "gen_counts.json", "w") as fh:
        json.dump({"config": {"rho_floor": 0.01}, "counters": {}}, fh)
    return src


@pytest.mark.skipif(
    not os.path.exists(os.path.join(GATE_V0, "scores.npz")),
    reason="needs the gate_v0 run (validation/output/ is gitignored)",
)
def test_sweep_end_to_end_smoke_at_five_minutes(tmp_path):
    """10 tracklets, dt = 45 and 5 min, a deliberately tiny (30 x 5) grid.

    Not a science run — the grid is far too coarse for the scores to mean
    anything.  It exists to catch wiring breakage across inject -> score ->
    analyze for the *sweep* specifically: the per-dt directory layout, the
    cross-dt array pairing, the scaling block, the verdict, the report and the
    four figures.  dt = 5 min is included because that is the arc length the
    sweep exists to reach.
    """
    from validation.self_consistency import ScoreConfig

    src = _tiny_source(tmp_path)
    out = tmp_path / "sweep"
    man = an.sweep_inject(source_run=str(src), out_dir=str(out),
                          dt_grid=(45.0, 5.0), n_mc=2, n_sub=6,
                          reuse_from=None, verbose=False)
    assert [e["tag"] for e in man["dt_runs"]] == ["dt_45min", "dt_5min"]
    assert man["break_criterion"] == an.BREAK_CRITERION
    # main + 2 mc + 2 decomp, per arc length.
    assert all(e["n_runs"] == 5 for e in man["dt_runs"])

    cfg = ScoreConfig(n_rho=30, n_rhodot=5)
    an.sweep_score(out_dir=str(out), cfg=cfg, workers=2, chunk=10, verbose=False)
    for e in man["dt_runs"]:
        assert os.path.exists(out / e["tag"] / "main" / "scores.npz")
        assert os.path.exists(out / e["tag"] / "noisy_scores.npz")
    # Rescoring must be a no-op.
    again = an.sweep_score(out_dir=str(out), cfg=cfg, workers=2, chunk=10,
                           verbose=False)
    assert all(p["scored"] == [] for p in again["per_dt"])

    s = an.sweep_analyze(out_dir=str(out), plots=True, n_boot=30, verbose=False)
    assert s["dt_grid_min"] == [45.0, 5.0]
    assert [b["dt_min"] for b in s["per_dt"]] == [45.0, 5.0]
    assert all(b["n"] == 10 for b in s["per_dt"])
    assert all(np.isfinite(b["ranking"]["auc_noisy"]) for b in s["per_dt"])
    assert all(b["mc"]["n_realizations"] == 2 for b in s["per_dt"])
    assert all(b["decomposition"]["n"] == 6 for b in s["per_dt"])
    # The rate error must be exactly 9x larger at 5 min than at 45 min.
    r45, r5 = (b["rate_error"]["sigma_rate_arcsec_per_day_per_coord"]["p50"]
               for b in s["per_dt"])
    assert r5 == pytest.approx(9.0 * r45, rel=1e-12)
    # Cross-dt machinery ran and the verdict is well formed.
    assert s["variance_scaling"]["dt_min"] == [45.0, 5.0]
    assert s["variance_scaling"]["linear_scaling_ratio"][0]["ratio_p50"] == \
        pytest.approx(1.0, rel=1e-12)
    ver = s["verdict"]
    assert set(r["tag"] for r in ver["rows"]) == {"dt_45min", "dt_5min"}
    assert isinstance(ver["any_break_in_range"], bool)
    assert ver["criterion"] == an.BREAK_CRITERION

    for f in ("REPORT.md", "summary.json", "arclength_headline.png",
              "std_scaling.png", "rate_error.png",
              "ranking_operating_point.png"):
        assert os.path.exists(out / f), f
    for e in man["dt_runs"]:
        assert os.path.exists(out / e["tag"] / "summary.json")
    text = (out / "REPORT.md").read_text()
    assert "The criterion (declared before the sweep was run)" in text
    assert "C1 (calibration)" in text and "C2 (triage churn)" in text
    assert "5 min" in text


@pytest.mark.skipif(
    not os.path.exists(os.path.join(GATE_V0, "gen.npz")),
    reason="needs the gate_v0 catalogue",
)
def test_adopt_scores_refuses_a_different_catalogue(tmp_path):
    """The reuse shortcut must never splice another catalogue's scores in.

    ``_adopt_scores`` is the only place in the sweep where a scoring pass is
    taken on trust, so it verifies array identity of ``gen.npz`` and declines
    otherwise — including when the donor simply has no scores yet.
    """
    src = _tiny_source(tmp_path)
    a = an.inject(source_run=str(src), out_dir=str(tmp_path / "a"), n_mc=0,
                  n_sub=0, m5_sens=None, dt_min=45.0, verbose=False)
    an.inject(source_run=str(src), out_dir=str(tmp_path / "b"), n_mc=0,
              n_sub=0, m5_sens=None, dt_min=5.0, verbose=False)
    assert a["runs"][0]["name"] == "main"
    da, db = str(tmp_path / "a" / "main"), str(tmp_path / "b" / "main")
    # Same catalogue, but the donor has no scores -> declined.
    assert an._adopt_scores(da, da, verbose=False) is False
    np.savez_compressed(os.path.join(da, "scores.npz"), index=np.arange(10))
    # Different dt means different rates means a different catalogue.
    assert an._gen_arrays_identical(da, db) is False
    assert an._adopt_scores(da, db, verbose=False) is False
    assert not os.path.exists(os.path.join(db, "scores.npz"))
    # Identical catalogue -> adopted.
    an.inject(source_run=str(src), out_dir=str(tmp_path / "c"), n_mc=0,
              n_sub=0, m5_sens=None, dt_min=45.0, verbose=False)
    dc = str(tmp_path / "c" / "main")
    assert an._gen_arrays_identical(da, dc) is True
    assert an._adopt_scores(da, dc, verbose=False) is True
    assert os.path.exists(os.path.join(dc, "scores.npz"))
