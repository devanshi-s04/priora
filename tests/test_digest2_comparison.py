"""Unit tests for the WP7 digest2 head-to-head harness.

Deliberately small and fast (< 2 min): the digest2 round trip uses 10 synthetic
tracklets.  The full 2x10^4-tracklet comparison is a CLI, not a test --
``python -m validation.digest2_comparison run --run validation/output/gate_v0``.

Tests that need the digest2 package skip cleanly (with the install command in
the skip reason) when it is not present; everything else -- the emitted format,
the subset split, the AUC/bootstrap/calibration maths, the score-file join --
runs with numpy alone.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from validation import digest2_comparison as d2c  # noqa: E402
from validation import self_consistency as sc  # noqa: E402

MODEL = os.path.join(REPO_ROOT, "data", "digest2.model.csv")

_D2_OK, _D2_MSG = d2c.digest2_available()
needs_digest2 = pytest.mark.skipif(not _D2_OK, reason=_D2_MSG)
needs_model = pytest.mark.skipif(
    not os.path.exists(MODEL),
    reason="digest2 model file not fetched (see scripts/fetch_digest2_model.py)",
)


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------


def _synth_run(tmp_path, n=10, seed=3):
    """Write a minimal WP6-shaped run directory and load it.

    The attributables are plausible main-belt-ish tracklets at moderate
    ecliptic latitude; the point is format/plumbing, not physics.
    """
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.2, 6.0, n)
    dec = rng.uniform(-0.4, 0.4, n)
    ra_rate = rng.uniform(-0.008, 0.008, n)
    dec_rate = rng.uniform(-0.003, 0.003, n)
    v_mag = rng.uniform(18.0, 23.0, n)
    jd = 2461041.5 + rng.uniform(0.0, 300.0, n)
    is_neo = np.zeros(n, dtype=bool)
    is_neo[: max(1, n // 5)] = True
    a = np.where(is_neo, 1.6, 2.7)
    e = np.where(is_neo, 0.5, 0.13)
    inc = np.radians(np.where(is_neo, 12.0, 6.0))

    gen = dict(
        ra=ra,
        dec=dec,
        ra_rate=ra_rate,
        dec_rate=dec_rate,
        v_mag=v_mag,
        epoch_jd_tdb=jd,
        R_obs=np.zeros((n, 3)),
        V_obs=np.zeros((n, 3)),
        rho_true=rng.uniform(0.5, 3.0, n),
        rho_dot_true=rng.uniform(-0.01, 0.01, n),
        elements=np.column_stack(
            [a, e, inc, np.zeros(n), np.zeros(n), np.zeros(n)]
        ),
        H=np.where(is_neo, 22.0, 17.0),
        is_neo=is_neo,
        weight=np.ones(n),
    )
    p = np.where(is_neo, rng.uniform(0.4, 0.95, n), rng.uniform(1e-5, 0.05, n))
    scores = dict(
        index=np.arange(n),
        p_neo=p,
        log_lr=np.log(p / (1 - p)),
        n_nodes=np.full(n, 137),
        ar_measure=np.full(n, 1.0),
        # deterministic spread across the clean / middle / broken subsets
        ar_frac_h_above=np.tile([1e-6, 5e-3, 5e-2, 2e-1, 0.0], n)[:n],
        mass_frac_q_above_h_above=np.zeros(n),
    )
    run_dir = os.path.join(str(tmp_path), "run")
    os.makedirs(run_dir, exist_ok=True)
    np.savez_compressed(os.path.join(run_dir, "gen.npz"), **gen)
    np.savez_compressed(os.path.join(run_dir, "scores.npz"), **scores)
    return run_dir


# --------------------------------------------------------------------------
# Score-file consumption
# --------------------------------------------------------------------------


def test_load_run_joins_on_score_index(tmp_path):
    """scores.npz is chunk-concatenated, so the join must use its own index."""
    run_dir = _synth_run(tmp_path, n=10)
    # Shuffle scores.npz rows and drop one: load_run must follow 'index'.
    with np.load(os.path.join(run_dir, "scores.npz")) as z:
        s = {k: z[k] for k in z.files}
    perm = np.array([7, 2, 9, 0, 5, 1, 8, 3, 6])  # index 4 dropped
    s = {k: v[perm] for k, v in s.items()}
    np.savez_compressed(os.path.join(run_dir, "scores.npz"), **s)

    run = d2c.load_run(run_dir)
    assert run.n == 9
    np.testing.assert_array_equal(run.index, perm)
    with np.load(os.path.join(run_dir, "gen.npz")) as z:
        np.testing.assert_allclose(run.gen["ra"], z["ra"][perm])
        np.testing.assert_array_equal(run.is_neo, z["is_neo"][perm])
    np.testing.assert_allclose(run.p_neo, s["p_neo"])


def test_load_run_drops_nonfinite_p_and_honours_limit(tmp_path):
    run_dir = _synth_run(tmp_path, n=10)
    with np.load(os.path.join(run_dir, "scores.npz")) as z:
        s = {k: z[k] for k in z.files}
    s["p_neo"] = s["p_neo"].copy()
    s["p_neo"][3] = np.nan
    np.savez_compressed(os.path.join(run_dir, "scores.npz"), **s)

    run = d2c.load_run(run_dir)
    assert run.n == 9
    assert 3 not in set(run.index.tolist())
    assert np.isfinite(run.p_neo).all()

    assert d2c.load_run(run_dir, limit=4).n == 4


def test_load_run_rejects_bad_index_and_missing_columns(tmp_path):
    run_dir = _synth_run(tmp_path, n=6)
    with np.load(os.path.join(run_dir, "scores.npz")) as z:
        s = {k: z[k] for k in z.files}
    bad = dict(s)
    bad["index"] = s["index"].copy()
    bad["index"][0] = 999
    np.savez_compressed(os.path.join(run_dir, "scores.npz"), **bad)
    with pytest.raises(ValueError, match="out of range"):
        d2c.load_run(run_dir)

    trimmed = {k: v for k, v in s.items() if k != "ar_frac_h_above"}
    np.savez_compressed(os.path.join(run_dir, "scores.npz"), **trimmed)
    with pytest.raises(KeyError, match="ar_frac_h_above"):
        d2c.load_run(run_dir)

    np.savez_compressed(os.path.join(run_dir, "scores.npz"), **s)
    os.remove(os.path.join(run_dir, "gen.npz"))
    with pytest.raises(FileNotFoundError):
        d2c.load_run(run_dir)


# --------------------------------------------------------------------------
# Emission — format correctness independent of digest2
# --------------------------------------------------------------------------


def test_emit_ades_psv_shape_and_linear_propagation(tmp_path):
    run_dir = _synth_run(tmp_path, n=10)
    run = d2c.load_run(run_dir)
    path = os.path.join(str(tmp_path), "t.psv")
    man = d2c.emit_ades_psv(run, path, dt_min=45.0)

    assert man["n_tracklets"] == 10 and man["n_observations"] == 20
    lines = [ln for ln in open(path).read().splitlines() if not ln.startswith("#")]
    header = [c.strip() for c in lines[0].split("|")]
    assert header[:6] == ["permID", "provID", "trkSub", "mode", "stn", "obsTime"]
    assert {"ra", "dec", "mag", "band"} <= set(header)
    rows = [[c.strip() for c in ln.split("|")] for ln in lines[1:]]
    assert len(rows) == 20
    assert all(len(r) == len(header) for r in rows)
    assert {r[header.index("stn")] for r in rows} == {"500"}

    # Two rows per designation, and the offset row is the linear propagation.
    by = {}
    for r in rows:
        by.setdefault(r[header.index("trkSub")], []).append(r)
    assert set(by) == {d2c.designation(i) for i in range(10)}
    dt = 45.0 / 1440.0
    for i in range(10):
        r0, r1 = by[d2c.designation(i)]
        ra0, ra1 = float(r0[header.index("ra")]), float(r1[header.index("ra")])
        dec0, dec1 = float(r0[header.index("dec")]), float(r1[header.index("dec")])
        exp_ra = np.degrees(run.gen["ra"][i] + run.gen["ra_rate"][i] * dt) % 360.0
        exp_dec = np.degrees(run.gen["dec"][i] + run.gen["dec_rate"][i] * dt)
        assert ra0 == pytest.approx(np.degrees(run.gen["ra"][i]) % 360.0, abs=1e-7)
        assert ra1 == pytest.approx(exp_ra, abs=1e-7)
        assert dec0 == pytest.approx(np.degrees(run.gen["dec"][i]), abs=1e-7)
        assert dec1 == pytest.approx(exp_dec, abs=1e-7)


def test_emit_mpc80_is_exactly_80_columns(tmp_path):
    run_dir = _synth_run(tmp_path, n=10)
    run = d2c.load_run(run_dir)
    path = os.path.join(str(tmp_path), "t.obs")
    man = d2c.emit_mpc80(run, path, dt_min=30.0)
    lines = open(path).read().splitlines()
    assert man["n_observations"] == 20 and len(lines) == 20
    assert {len(ln) for ln in lines} == {80}
    assert {ln[14] for ln in lines} == {"C"}
    assert {ln[77:80] for ln in lines} == {"500"}


def test_emit_rejects_pole_crossing(tmp_path):
    run_dir = _synth_run(tmp_path, n=3)
    with np.load(os.path.join(run_dir, "gen.npz")) as z:
        g = {k: z[k] for k in z.files}
    g["dec"] = np.full(3, np.radians(89.999))
    g["dec_rate"] = np.full(3, 1.0)  # rad/day: steps over the pole in minutes
    np.savez_compressed(os.path.join(run_dir, "gen.npz"), **g)
    run = d2c.load_run(run_dir)
    with pytest.raises(ValueError, match="pole"):
        d2c.emit_ades_psv(run, os.path.join(str(tmp_path), "p.psv"), dt_min=45.0)


# --------------------------------------------------------------------------
# Round trip through digest2's own parser + scorer
# --------------------------------------------------------------------------


@needs_digest2
@needs_model
@pytest.mark.parametrize("fmt", ["ades_psv", "mpc80"])
def test_roundtrip_through_digest2_parser(tmp_path, fmt):
    """digest2 must parse 100% of what we emit, in both supported formats."""
    from digest2.observation import parse_ades_psv, parse_mpc80_file

    run_dir = _synth_run(tmp_path, n=10)
    run = d2c.load_run(run_dir)
    path = os.path.join(str(tmp_path), "t" + d2c._SUFFIX[fmt])
    d2c.EMITTERS[fmt](run, path, dt_min=45.0)

    parser = parse_ades_psv if fmt == "ades_psv" else parse_mpc80_file
    parsed = parser(path)
    assert len(parsed) == 10, "digest2's parser dropped tracklets"
    assert all(len(v) == 2 for v in parsed.values())
    # MPC 80-col keys carry the 12-column designation field's padding.
    assert {k.strip() for k in parsed} == {d2c.designation(i) for i in range(10)}
    parsed = {k.strip(): v for k, v in parsed.items()}

    # Round-trip fidelity of the ADES path (full float precision).
    if fmt == "ades_psv":
        for i in range(10):
            obs = parsed[d2c.designation(i)]
            assert obs[0].mjd < obs[1].mjd
            assert obs[1].mjd - obs[0].mjd == pytest.approx(45.0 / 1440.0, abs=1e-6)
            assert obs[0].ra == pytest.approx(
                np.degrees(run.gen["ra"][i]) % 360.0, abs=1e-7
            )
            assert obs[0].dec == pytest.approx(np.degrees(run.gen["dec"][i]), abs=1e-7)
            # band V => no magnitude correction
            assert obs[0].mag == pytest.approx(run.gen["v_mag"][i], abs=0.005)
            assert obs[0].obscode == "500"


@needs_digest2
@needs_model
def test_run_digest2_scores_every_tracklet(tmp_path):
    run_dir = _synth_run(tmp_path, n=10)
    run = d2c.load_run(run_dir)
    path = os.path.join(str(tmp_path), "t.psv")
    d2c.emit_ades_psv(run, path, dt_min=45.0)

    cols, stats = d2c.run_digest2(path, run.index, model_path=MODEL,
                                  workers=2, verbose=False)
    assert stats["parse_rate"] == 1.0
    assert stats["score_rate"] == 1.0
    assert stats["n_matched"] == 10
    assert cols["ok"].all()
    assert np.all((cols["raw_neo"] >= 0.0) & (cols["raw_neo"] <= 100.0))
    assert np.all((cols["noid_neo"] >= 0.0) & (cols["noid_neo"] <= 100.0))
    # Two exactly-collinear detections: digest2's great-circle RMS is 0.
    assert np.allclose(cols["rms"], 0.0, atol=1e-6)


@needs_digest2
@needs_model
def test_digest2_env_records_the_shared_model(tmp_path):
    env = d2c.digest2_env(MODEL)
    assert env["available"] is True
    assert env["version"]
    assert env["source_commit"].startswith("c9e3f89")
    assert len(env["model_sha256"]) == 64
    # The premise of DESIGN 4.2: the package's own bundled model is the same
    # file (it ships as an unresolved LFS pointer, so we can check its oid).
    if env.get("bundled_model_is_lfs_pointer"):
        assert env["bundled_model_lfs_oid"] == env["model_sha256"]
    # digest2 renders a binary cache beside the CSV on first load; if it is
    # there, the recorded path must be the real file, not model.model.
    cache = env.get("binary_model_cache")
    if cache is not None:
        assert cache.endswith("digest2.model") and os.path.exists(cache)


@needs_digest2
@needs_model
def test_analyze_end_to_end_on_ten_tracklets(tmp_path):
    run_dir = _synth_run(tmp_path, n=10)
    run = d2c.load_run(run_dir)
    path = os.path.join(str(tmp_path), "t.psv")
    d2c.emit_ades_psv(run, path, dt_min=45.0)
    cols, _ = d2c.run_digest2(path, run.index, model_path=MODEL, workers=2,
                              verbose=False)
    res = d2c.analyze(run, cols, n_boot=25, seed=1)
    assert set(res["subsets"]) == set(d2c.SUBSETS)
    for name in d2c.SUBSETS:
        s = res["subsets"][name]
        assert s["n"] >= 0
        assert set(s["calibration"]) == {
            "priora", "digest2_raw_over_100", "digest2_noid_over_100"
        }
    # summary_text must render (it is the committed artifact's body)
    summary = {
        "run_dir": run_dir, "digest2": d2c.digest2_env(MODEL),
        "emission": {"format": "ades_psv", "n_tracklets": 10, "dt_min": 45.0,
                     "obscode": "500"},
        "digest2_run": {"n_emitted": 10, "n_parsed_with_pair": 10, "n_matched": 10,
                        "parse_rate": 1.0, "score_ms_per_tracklet": 1.0,
                        "workers": 2},
        "caveats": d2c.CAVEATS, **res,
    }
    txt = d2c.summary_text(summary)
    assert "RANKING" in txt and "CALIBRATION" in txt and "TRUNCATION" in txt
    # and it must be JSON-serialisable for summary.json
    json.loads(json.dumps(summary, default=d2c._json_default))


# --------------------------------------------------------------------------
# Subset split (DESIGN 4.2)
# --------------------------------------------------------------------------


def test_subset_masks_split_the_truncation_diagnostic():
    f = np.array([0.0, 1e-6, 9.99e-4, 1e-3, 5e-3, 1e-2, 1.01e-2, 0.2, np.nan])
    m = d2c.subset_masks(f)
    assert m["all"].all()
    np.testing.assert_array_equal(
        m["clean"], [True, True, True, False, False, False, False, False, False]
    )
    np.testing.assert_array_equal(
        m["broken"], [False, False, False, False, False, False, True, True, False]
    )
    # Boundary values belong to neither subset; NaN never enters either.
    assert not m["clean"][-1] and not m["broken"][-1]
    assert not (m["clean"] & m["broken"]).any()


def test_subset_masks_custom_thresholds_and_validation():
    f = np.array([1e-4, 1e-2, 0.5])
    m = d2c.subset_masks(f, clean_max=1e-2, broken_min=1e-1)
    np.testing.assert_array_equal(m["clean"], [True, False, False])
    np.testing.assert_array_equal(m["broken"], [False, False, True])
    with pytest.raises(ValueError, match="clean_max"):
        d2c.subset_masks(f, clean_max=0.1, broken_min=0.01)


def test_subset_masks_on_a_real_shaped_distribution():
    rng = np.random.default_rng(0)
    f = np.concatenate([np.zeros(200), rng.uniform(0, 0.3, 800)])
    m = d2c.subset_masks(f)
    assert m["clean"].sum() >= 200
    assert m["broken"].sum() > 0
    assert m["clean"].sum() + m["broken"].sum() <= f.size


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_auc_fast_matches_the_wp6_reference():
    rng = np.random.default_rng(11)
    for _ in range(5):
        n = 400
        y = rng.random(n) < 0.15
        s = rng.normal(size=n) + 1.2 * y
        s[rng.random(n) < 0.2] = 0.0  # force ties
        w = rng.uniform(0.5, 2.0, n)
        assert d2c.auc_fast(s, y, w) == pytest.approx(sc.auc_weighted(s, y, w))
        assert d2c.auc_fast(s, y) == pytest.approx(sc.auc_weighted(s, y))


def test_auc_fast_edge_cases():
    y = np.array([True, False, True, False])
    assert d2c.auc_fast(np.array([2.0, 1.0, 2.0, 1.0]), y) == 1.0
    assert d2c.auc_fast(np.array([1.0, 1.0, 1.0, 1.0]), y) == 0.5
    assert d2c.auc_fast(np.array([0.0, 1.0, 0.0, 1.0]), y) == 0.0
    assert np.isnan(d2c.auc_fast(np.array([1.0, 2.0]), np.array([True, True])))
    assert np.isnan(d2c.auc_fast(np.array([]), np.array([], dtype=bool)))


def test_auc_bootstrap_recovers_a_known_ordering_gap():
    rng = np.random.default_rng(5)
    n = 3000
    y = rng.random(n) < 0.1
    good = rng.normal(size=n) + 2.0 * y      # separates well
    poor = rng.normal(size=n) + 0.6 * y      # separates weakly
    out = d2c.auc_bootstrap(good, poor, y, n_boot=300, seed=2)
    assert out["auc_a"] > out["auc_b"] > 0.5
    assert out["delta"] > 0.1
    assert out["delta_ci"][0] < out["delta"] < out["delta_ci"][1]
    assert out["delta_ci"][0] > 0.0          # gap is resolved
    assert out["p_delta_le_0"] == 0.0
    assert out["n_pos"] == int(y.sum())

    # Same score twice: delta identically zero, CI degenerate.
    same = d2c.auc_bootstrap(good, good, y, n_boot=100, seed=2)
    assert same["delta"] == 0.0
    assert same["delta_ci"] == [0.0, 0.0]


def test_auc_bootstrap_degenerates_gracefully():
    y = np.array([True, False, False])
    out = d2c.auc_bootstrap(np.arange(3.0), np.arange(3.0), y, n_boot=10)
    assert np.isnan(out["delta_ci"][0])


def test_calibration_detects_a_well_and_a_badly_calibrated_score():
    rng = np.random.default_rng(17)
    n = 40000
    p = rng.beta(0.4, 12.0, n)
    y = rng.random(n) < p                    # p is exactly calibrated
    good = d2c.calibration_stats(p, y)
    bad = d2c.calibration_stats(np.clip(5.0 * p, 0, 1), y)

    assert good["ratio_expected_observed"] == pytest.approx(1.0, abs=0.05)
    assert abs(good["z_total"]) < 3.0
    assert bad["ratio_expected_observed"] > 3.0
    # Over-prediction: far fewer positives observed than the score claims.
    assert bad["z_total"] < -10.0
    assert bad["brier"] > good["brier"]
    assert bad["ece"] > good["ece"]
    assert good["observed_positives"] == pytest.approx(bad["observed_positives"])


def test_reliability_table_bins_carry_comparable_expectation():
    rng = np.random.default_rng(19)
    n = 20000
    p = rng.beta(0.4, 12.0, n)
    y = rng.random(n) < p
    rows = d2c.reliability_table(p, y, n_bins=5)
    assert 2 <= len(rows) <= 5
    exp = np.array([r["expected_pos"] for r in rows])
    assert exp.min() > 0
    assert exp.max() / exp.min() < 3.0       # roughly equal expected counts
    assert sum(r["n"] for r in rows) == n
    assert all(r["p_lo"] <= r["p_mean"] <= r["p_hi"] for r in rows)
    assert d2c.reliability_table(np.array([]), np.array([])) == []
    assert d2c.reliability_table(np.zeros(10), np.zeros(10)) == []


def test_operating_points_match_completeness_and_score_contamination():
    rng = np.random.default_rng(21)
    n = 4000
    y = rng.random(n) < 0.05
    p42 = np.clip(rng.beta(0.4, 12.0, n) + 0.75 * y, 0, 1)
    # digest2 stand-in: same ordering, but a uniform 6x over-statement.
    s_d2 = np.clip(600.0 * p42, 0, 100)
    w = np.ones(n)
    op = d2c.operating_points(p42, s_d2, y, w)

    a, m, p50 = op["digest2_neocp"], op["priora_matched"], op["priora_p50"]
    assert a["label"].startswith("digest2 raw >= 65")
    # The matched point reproduces digest2's completeness (to one tracklet).
    assert m["completeness"] >= a["completeness"]
    assert m["completeness"] - a["completeness"] < 2.0 / max(y.sum(), 1)
    # An over-stated score flags far more for the same recall.
    assert a["n_flagged"] > m["n_flagged"]
    assert a["contamination"] > m["contamination"]
    for d in (a, m, p50):
        assert 0.0 <= d["completeness"] <= 1.0
        assert 0.0 <= d["contamination"] <= 1.0
        assert d["n_true_neo"] <= d["n_flagged"]
    assert 0.0 < m["threshold"] < 1.0


def test_operating_points_respect_importance_weights():
    y = np.array([True, True, False, False, False, False])
    p42 = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1])
    s_d2 = np.array([90.0, 20.0, 80.0, 10.0, 10.0, 10.0])
    heavy = np.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0])
    flat = d2c.operating_points(p42, s_d2, y, np.ones(6))
    wt = d2c.operating_points(p42, s_d2, y, heavy)
    # Same flags, but up-weighted positives change completeness/contamination.
    assert flat["digest2_neocp"]["n_flagged"] == wt["digest2_neocp"]["n_flagged"]
    assert wt["digest2_neocp"]["contamination"] < \
        flat["digest2_neocp"]["contamination"]
    assert wt["digest2_neocp"]["completeness"] == pytest.approx(10.0 / 20.0)


def test_percentile_rank_handles_ties():
    r = d2c.percentile_rank(np.array([3.0, 1.0, 2.0, 1.0]))
    assert r[1] == r[3]                       # tied values share a rank
    assert r[0] > r[2] > r[1]
    assert np.all((r > 0) & (r < 1))
    assert d2c.percentile_rank(np.array([])).size == 0
    mono = d2c.percentile_rank(np.arange(10.0))
    assert np.all(np.diff(mono) > 0)


def test_disagreement_census_finds_the_planted_swap(tmp_path):
    run_dir = _synth_run(tmp_path, n=40)
    run = d2c.load_run(run_dir)
    n = run.n
    # digest2 = a reversed ranking, so the census must be maximally populated.
    d2 = {
        "raw_neo": 100.0 * (1.0 - d2c.percentile_rank(run.p_neo)),
        "noid_neo": np.zeros(n),
        "rms": np.zeros(n),
        "ok": np.ones(n, dtype=bool),
    }
    cen = d2c.disagreement_census(run, d2, np.ones(n, dtype=bool), top_k=3)
    assert cen["n"] == n
    assert cen["rank_spearman"] < -0.99
    assert len(cen["priora_higher"]) == 3
    assert len(cen["digest2_higher"]) == 3
    assert cen["priora_higher"][0]["delta_rank"] > 0
    assert cen["digest2_higher"][0]["delta_rank"] < 0
    assert cen["frac_delta_rank_gt_0p25"] > 0.5
    for r in cen["priora_higher"]:
        assert set(r) >= {"index", "is_neo", "p_neo_42", "d2_raw_neo",
                          "delta_rank", "v_mag", "H_true", "q_true",
                          "ar_frac_h_above"}
    # Rows join back to gen.npz correctly.
    top = cen["priora_higher"][0]
    k = int(np.flatnonzero(run.index == top["index"])[0])
    assert top["p_neo_42"] == pytest.approx(run.p_neo[k])
    assert top["is_neo"] == bool(run.is_neo[k])

    # Tail diagnostics describe the two ends, not the whole population.
    assert cen["tail_priora_higher"]["median_delta_rank"] > 0
    assert cen["tail_digest2_higher"]["median_delta_rank"] < 0
    for t in (cen["tail_priora_higher"], cen["tail_digest2_higher"]):
        assert t["n"] >= 3
        assert set(t) >= {"median_rho_true_au", "median_q_true_au",
                          "frac_q_gt_5au", "frac_rho_gt_10au", "frac_true_neo"}
        assert 0.0 <= t["frac_true_neo"] <= 1.0

    # Empty selection is handled, not crashed on.
    empty = d2c.disagreement_census(run, d2, np.zeros(n, dtype=bool))
    assert empty["n"] == 0 and empty["priora_higher"] == []


def test_analyze_without_digest2_uses_injected_scores(tmp_path):
    """The analysis layer is pure numpy; exercise it with a stand-in scorer."""
    run_dir = _synth_run(tmp_path, n=600, seed=8)
    run = d2c.load_run(run_dir)
    rng = np.random.default_rng(4)
    # A deliberately noisier version of the truth signal.
    raw = 100.0 * np.clip(
        run.p_neo * 0.4 + rng.uniform(0, 0.25, run.n), 0, 1
    )
    d2 = {"raw_neo": raw, "noid_neo": raw * 0.7, "rms": np.zeros(run.n),
          "ok": np.ones(run.n, dtype=bool)}
    res = d2c.analyze(run, d2, n_boot=50, seed=1)
    for name in d2c.SUBSETS:
        s = res["subsets"][name]
        assert s["n"] == int(d2c.subset_masks(run.ar_frac_h_above)[name].sum())
        assert 0.0 <= s["auc"]["auc_a"] <= 1.0
        assert 0.0 <= s["auc"]["auc_b"] <= 1.0
    assert res["thresholds"]["clean_max"] == d2c.CLEAN_MAX_FRAC
    assert res["subsets"]["all"]["auc"]["auc_a"] > res["subsets"]["all"]["auc"]["auc_b"]
    json.loads(json.dumps(res, default=d2c._json_default))


def test_analyze_respects_masked_out_digest2_failures(tmp_path):
    run_dir = _synth_run(tmp_path, n=100, seed=9)
    run = d2c.load_run(run_dir)
    raw = 100.0 * run.p_neo
    ok = np.ones(run.n, dtype=bool)
    ok[::5] = False
    raw[~ok] = np.nan
    d2 = {"raw_neo": raw, "noid_neo": raw, "rms": np.zeros(run.n), "ok": ok}
    res = d2c.analyze(run, d2, n_boot=20, seed=1)
    assert res["subsets"]["all"]["n"] == int(ok.sum())


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def test_cli_run_requires_digest2_and_reports_cleanly(tmp_path, capsys):
    if _D2_OK:
        pytest.skip("digest2 is installed; the missing-install path cannot be hit")
    rc = d2c.main(["run", "--run", str(tmp_path), "--out", str(tmp_path)])
    assert rc == 2
    assert "pip install" in capsys.readouterr().err


def test_cli_report_errors_on_missing_case(tmp_path, capsys):
    rc = d2c.main(["report", "--out", str(tmp_path), "--case", "nope"])
    assert rc == 2
    assert "missing" in capsys.readouterr().err


def test_cli_parses_the_documented_invocation():
    ap_ok = d2c.main
    assert callable(ap_ok)
    with pytest.raises(SystemExit):
        d2c.main(["run"])  # --run/--out are required


def test_designation_roundtrip():
    for i in (0, 7, 19999, 1234567):
        d = d2c.designation(i)
        assert len(d) == 8
        assert d2c._designation_index(d) == i


# --------------------------------------------------------------------------
# WP7 addendum — the clamped (digest2-emulation) control
#
# The knob itself is tested in tests/test_population.py; here we test the
# harness around it: that the clamped score file is consumed and aligned
# correctly, that the subset/decomposition logic says what it claims to say,
# and that the re-score path actually runs end to end on real geometry.
# --------------------------------------------------------------------------


def _synth_clamped(run, shift=0.0, seed=11):
    """A clamped score table aligned to ``run``, with a controllable shift.

    ``shift`` multiplies the odds, so shift = 0 reproduces the unclamped
    scores exactly (what the clean subset must do) and shift > 0 pushes
    p(NEO) up (what the clamp does to exposed tracklets).
    """
    p = run.p_neo
    f = run.ar_frac_h_above
    odds = p / (1.0 - p)
    boost = 1.0 + shift * np.where(np.isfinite(f), f, 0.0)
    pc = odds * boost / (1.0 + odds * boost)
    # Where there is no exposure the clamp is a no-op *exactly*, as it is in
    # the real thing; the odds round trip alone would leave 1-ULP noise.
    pc = np.where(boost == 1.0, p, pc)
    return {
        "index": run.index.astype(np.int64),
        "p_neo": pc,
        "log_lr": np.log(odds * boost),
        "lambda_neo": np.full(run.n, 1e-6),
        "lambda_other": np.full(run.n, 1e-5),
        "n_nodes": run.scores["n_nodes"].astype(np.int64),
        "ar_measure": run.scores["ar_measure"].astype(float),
        "ar_frac_h_above": f.astype(float),
        "elapsed_s": np.full(run.n, 0.5),
    }


def test_load_clamped_scores_aligns_on_index_and_refuses_gaps(tmp_path):
    """The table is written in run order, but must still be *joined*, and a
    missing row must raise rather than silently become NaN."""
    run_dir = _synth_run(tmp_path, n=12)
    run = d2c.load_run(run_dir)
    cols = _synth_clamped(run, shift=3.0)

    out = os.path.join(str(tmp_path), "case")
    os.makedirs(out, exist_ok=True)
    # Shuffle the rows on disk: load_clamped_scores must follow 'index'.
    perm = np.array([5, 0, 11, 2, 9, 1, 7, 3, 10, 4, 8, 6])
    np.savez_compressed(
        os.path.join(out, d2c.CLAMPED_SCORES),
        **{k: v[perm] for k, v in cols.items()},
    )
    got = d2c.load_clamped_scores(out, run.index)
    assert np.array_equal(got["index"], run.index)
    for k in d2c.CLAMPED_FIELDS:
        assert np.allclose(got[k], cols[k]), k

    # A dropped row is an error, not a NaN.
    np.savez_compressed(
        os.path.join(out, d2c.CLAMPED_SCORES),
        **{k: v[:-3] for k, v in cols.items()},
    )
    with pytest.raises(KeyError, match="no clamped score"):
        d2c.load_clamped_scores(out, run.index)

    # A missing column is an error too.
    np.savez_compressed(
        os.path.join(out, d2c.CLAMPED_SCORES),
        **{k: v for k, v in cols.items() if k != "log_lr"},
    )
    with pytest.raises(KeyError, match="missing columns"):
        d2c.load_clamped_scores(out, run.index)

    with pytest.raises(FileNotFoundError):
        d2c.load_clamped_scores(str(tmp_path), run.index)


def test_clamped_score_config_follows_the_run(tmp_path):
    """The clamped pass must inherit the run's own grid, not a default: a
    different grid would confound the clamp with quadrature."""
    run_dir = _synth_run(tmp_path, n=4)
    # No score_config.json -> the documented defaults.
    cfg = d2c.clamped_score_config(run_dir)
    assert (cfg.n_rho, cfg.n_rhodot) == (1200, 50)

    with open(os.path.join(run_dir, "score_config.json"), "w") as fh:
        json.dump({"n_rho": 37, "n_rhodot": 11, "G": 0.25,
                   "model": "data/digest2.model.csv", "variant": "Unk",
                   "some_future_field": 1}, fh)
    cfg = d2c.clamped_score_config(run_dir)
    assert (cfg.n_rho, cfg.n_rhodot, cfg.G, cfg.variant) == (37, 11, 0.25, "Unk")
    assert os.path.isabs(cfg.model)          # relative paths are anchored
    # Explicit overrides win; None means "don't override".
    cfg = d2c.clamped_score_config(run_dir, n_rho=5, n_rhodot=None)
    assert (cfg.n_rho, cfg.n_rhodot) == (5, 11)


def test_analyze_clamped_subsets_and_decomposition(tmp_path):
    """Subsets are cut on the UNCLAMPED diagnostic, and the decomposition is
    an identity: Delta_WP7 == Delta_clamped + boundary."""
    run_dir = _synth_run(tmp_path, n=600, seed=8)
    run = d2c.load_run(run_dir)
    rng = np.random.default_rng(4)
    raw = 100.0 * np.clip(run.p_neo * 0.4 + rng.uniform(0, 0.25, run.n), 0, 1)
    d2 = {"raw_neo": raw, "noid_neo": raw * 0.7, "rms": np.zeros(run.n),
          "ok": np.ones(run.n, dtype=bool)}
    clamped = _synth_clamped(run, shift=6.0)

    res = d2c.analyze_clamped(run, clamped, d2, n_boot=50, seed=1)
    masks = d2c.subset_masks(run.ar_frac_h_above)
    for name in d2c.SUBSETS:
        s = res["subsets"][name]
        assert s["n"] == int(masks[name].sum())          # unclamped cut
        dec = s["decomposition"]
        assert dec["delta_wp7"] == pytest.approx(
            dec["delta_clamped_method"] + dec["delta_boundary_handling"]
        )
        assert set(s["calibration"]) == {
            "priora_clamped", "priora", "digest2_raw_over_100"
        }
        assert s["auc_clamped_vs_digest2"]["auc_b"] == pytest.approx(
            s["auc_unclamped_vs_digest2"]["auc_b"]
        )  # digest2's AUC does not depend on which priora it is paired with

    # On the clean subset the shift is ~0, so clamped == unclamped ranking.
    clean = res["subsets"]["clean"]
    assert clean["decomposition"]["delta_boundary_handling"] == pytest.approx(
        0.0, abs=2e-3
    )
    # The score shift is monotone in the diagnostic, and zero where it is zero.
    rows = res["score_shift"]["by_truncation_bin"]
    assert [r["median_dlog10_p"] for r in rows] == sorted(
        r["median_dlog10_p"] for r in rows
    )
    # The clamp can only act where the diagnostic is non-zero: a tracklet that
    # moved at f == 0 would mean it had leaked into the rest of the lookup.
    # (_synth_clamped is built to have exactly that property.)
    assert res["score_shift"]["n_moved_without_exposure"] == 0
    assert res["score_shift"]["max_abs_dlog10"] > 0.0
    assert res["consistency"]["max_abs_diff_ar_frac"] == 0.0
    assert res["consistency"]["max_abs_diff_n_nodes"] == 0
    assert res["n_nonfinite_p_clamped"] == 0
    json.loads(json.dumps(res, default=d2c._json_default))

    # The report renders from the dict alone.
    summary = {"run_dir": run_dir,
               "truncation_diagnostic": {"frac_above_1e-2": 0.5}}
    res["generated"] = "2026-09-01T00:00:00"
    md = d2c.clamped_report("unit", summary, res)
    assert "Delta_WP7" in md and "gate-ineligible" in md
    assert "## (a)" in md and "## (b)" in md and "## (c)" in md and "## (d)" in md


def test_analyze_clamped_drops_nonfinite_clamped_scores(tmp_path):
    run_dir = _synth_run(tmp_path, n=120, seed=9)
    run = d2c.load_run(run_dir)
    clamped = _synth_clamped(run, shift=2.0)
    clamped["p_neo"] = clamped["p_neo"].copy()
    clamped["p_neo"][::6] = np.nan
    raw = 100.0 * run.p_neo
    d2 = {"raw_neo": raw, "noid_neo": raw, "rms": np.zeros(run.n),
          "ok": np.ones(run.n, dtype=bool)}
    res = d2c.analyze_clamped(run, clamped, d2, n_boot=20, seed=1)
    n_bad = int(np.isnan(clamped["p_neo"]).sum())
    assert res["n_nonfinite_p_clamped"] == n_bad
    assert res["subsets"]["all"]["n"] == run.n - n_bad


def test_clamped_plots_render(tmp_path):
    run_dir = _synth_run(tmp_path, n=200, seed=12)
    run = d2c.load_run(run_dir)
    clamped = _synth_clamped(run, shift=4.0)
    raw = 100.0 * run.p_neo
    d2 = {"raw_neo": raw, "noid_neo": raw, "rms": np.zeros(run.n),
          "ok": np.ones(run.n, dtype=bool)}
    res = d2c.analyze_clamped(run, clamped, d2, n_boot=20, seed=1)
    masks = d2c.subset_masks(run.ar_frac_h_above)
    for fn, args in (
        (d2c.plot_clamped_scatter, (run, clamped)),
        (d2c.plot_clamped_roc, (run, clamped, d2, masks, res)),
        (d2c.plot_clamped_calibration, (res,)),
    ):
        p = os.path.join(str(tmp_path), fn.__name__ + ".png")
        fn(*args, p)
        assert os.path.getsize(p) > 5000


# --------------------------------------------------------------------------
# End-to-end: the clamped re-score path on real geometry
# --------------------------------------------------------------------------


@needs_model
def test_clamped_rescore_end_to_end_on_ten_tracklets(tmp_path):
    """Ten real WP6 tracklets through the clamped scorer, tiny grid, 1 worker.

    Checks the thing the whole addendum rests on: the clamped pass reproduces
    the *geometry* (node count, §2.6(b) diagnostic) exactly and changes only
    the density, so any score difference is the top H bin and nothing else.
    """
    gen = sc.generate(sc.GenConfig(n=10, seed=1), verbose=False)
    run_dir = os.path.join(str(tmp_path), "run")
    sc.save_generation(gen, run_dir)

    cfg = d2c.ClampedScoreConfig(n_rho=40, n_rhodot=8)
    index = np.arange(len(gen))
    out = os.path.join(str(tmp_path), "case")
    got = d2c.clamped_score(run_dir, out, index, cfg, workers=1, chunk=4,
                            verbose=False)

    assert np.array_equal(got["index"], index)
    assert set(got) == set(d2c.CLAMPED_FIELDS)
    assert os.path.exists(os.path.join(out, d2c.CLAMPED_SCORES))
    assert np.all(got["n_nodes"] > 0)
    assert np.all(np.isfinite(got["p_neo"]))
    assert np.all((got["p_neo"] >= 0.0) & (got["p_neo"] <= 1.0))
    assert np.all(got["ar_frac_h_above"] >= 0.0)

    # Round-trip through the reader, including the join.
    back = d2c.load_clamped_scores(out, index[::-1])
    assert np.array_equal(back["index"], index[::-1])
    assert np.allclose(back["p_neo"], got["p_neo"][::-1])

    # Resume must reuse the chunks and return the identical table.
    again = d2c.clamped_score(run_dir, out, index, cfg, workers=1, chunk=4,
                              resume=True, verbose=False)
    for k in d2c.CLAMPED_FIELDS:
        if k != "elapsed_s":
            assert np.array_equal(again[k], got[k]), k

    # A different grid must invalidate the chunks, not splice them.
    other = d2c.ClampedScoreConfig(n_rho=41, n_rhodot=8)
    d2c.clamped_score(run_dir, out, index, other, workers=1, chunk=4,
                      resume=True, verbose=False)

    # The unclamped pass on the same grid: same geometry, density-only change.
    from priora import Attributable, classify
    from priora.population import Digest2Histogram
    from astropy.time import Time

    neo, oth = Digest2Histogram.neo_other_pair(MODEL)
    n_diff = 0
    for i in range(len(gen)):
        att = Attributable(
            ra=float(gen.ra[i]), dec=float(gen.dec[i]),
            ra_rate=float(gen.ra_rate[i]), dec_rate=float(gen.dec_rate[i]),
            epoch=Time(float(gen.epoch_jd_tdb[i]), format="jd", scale="tdb"),
            v_mag=float(gen.v_mag[i]),
        )
        res = classify(att, neo, oth, R_obs=gen.R_obs[i], V_obs=gen.V_obs[i],
                       n_rho=40, n_rhodot=8, rho_min=cfg.rho_min,
                       rho_max=cfg.rho_max, G=cfg.G)
        assert res.grid.rho.size == got["n_nodes"][i]
        assert res.measure_fraction_H_above(25.5) == pytest.approx(
            got["ar_frac_h_above"][i], abs=1e-12
        )
        # Clamping can only ADD density, so each component's Lambda can only
        # grow -- a strong, direction-aware check on the whole plumbing.
        assert got["lambda_neo"][i] >= res.lambda_neo * (1 - 1e-12)
        assert got["lambda_other"][i] >= res.lambda_other * (1 - 1e-12)
        if abs(got["p_neo"][i] - res.p_neo) > 1e-12:
            n_diff += 1
    # ...and on a V ~ 14-24 catalogue at least one tracklet must actually be
    # exposed to the top bin, or the test is vacuous.
    assert n_diff > 0
    assert (got["ar_frac_h_above"] > 0).any()


def test_cli_clamped_errors_without_the_wp7_case(tmp_path, capsys):
    run_dir = _synth_run(tmp_path, n=5)
    out = os.path.join(str(tmp_path), "case")
    os.makedirs(out, exist_ok=True)
    rc = d2c.main(["clamped", "--run", run_dir, "--out", out])
    assert rc == 2
    assert "digest2_scores.npz" in capsys.readouterr().err

    # A digest2 score file for a *different* selection is refused, not joined.
    run = d2c.load_run(run_dir)
    np.savez_compressed(
        os.path.join(out, "digest2_scores.npz"),
        index=run.index[:3], raw_neo=np.zeros(3), noid_neo=np.zeros(3),
        rms=np.zeros(3), ok=np.ones(3, dtype=bool),
    )
    rc = d2c.main(["clamped", "--run", run_dir, "--out", out])
    assert rc == 2
    assert "different tracklet selection" in capsys.readouterr().err


def test_auc_bootstrap_strata_stay_mixed_when_positives_dominate():
    """Regression for the NEOCP-pilot finding: with n_pos >= n_neg the old
    resample indexed the concatenated (pos, neg) array with per-stratum
    indices, making every resample all-positive (AUC = nan, CI = [nan, nan],
    p_delta_le_0 silently 0.0)."""
    import numpy as np

    from validation.digest2_comparison import auc_bootstrap

    rng = np.random.default_rng(0)
    n_pos, n_neg = 300, 190  # positives dominate, as in the NEOCP sample
    labels = np.concatenate((np.ones(n_pos, bool), np.zeros(n_neg, bool)))
    s_a = np.where(labels, 0.8, 0.2) + rng.normal(0, 0.05, labels.size)
    s_b = np.where(labels, 0.7, 0.3) + rng.normal(0, 0.05, labels.size)
    out = auc_bootstrap(s_a, s_b, labels, n_boot=200, seed=3)
    assert np.isfinite(out["delta_ci"]).all()
    assert np.isfinite(out["auc_a_ci"]).all()
    assert 0.0 <= out["p_delta_le_0"] <= 1.0
    # both AUCs are high and the CI brackets the point estimate
    assert out["auc_a_ci"][0] <= out["auc_a"] <= out["auc_a_ci"][1]
