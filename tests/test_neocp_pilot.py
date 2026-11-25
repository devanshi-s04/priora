"""Unit tests for the WP8 real-data NEOCP pilot (``validation/neocp_pilot.py``).

Fast and **entirely offline**.  Nothing here touches the network: the fetch
stage is a CLI, and the tests that exercise the cache-reading path build the
cache from small inline fixtures.  Two of them additionally monkeypatch
``urllib.request.urlopen`` to raise, so an accidental future network call in
``build``/``analyze`` fails the suite instead of silently working on the
developer's laptop and hanging on a batch node.

The substantive test is
:func:`test_fit_attributable_roundtrip_against_observables_from_elements`: a
Keplerian tracklet is generated with ``transforms.observables_from_elements``,
fitted back with :func:`neocp_pilot.fit_attributable`, and the residual is
shown to be *curvature* (position error ~ arc^2, rate error ~ arc) rather than
a bug.
"""

from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from astropy.time import Time  # noqa: E402

from priora import transforms as T  # noqa: E402
from priora.attributable import (  # noqa: E402
    MPC_OBSCODES,
    observatory_location,
    observer_state,
)
from priora.constants import K_GAUSS  # noqa: E402
from validation import neocp_pilot as NP  # noqa: E402

ARCSEC = 206264.80624709636


# --------------------------------------------------------------------------
# Band -> V conversion (D-5; digest2's own table)
# --------------------------------------------------------------------------


def test_band_table_matches_digest2_verbatim():
    """Our table must BE digest2's, not merely resemble it.

    The whole point of the pilot's photometry is that both codes standardise
    identically; if digest2 ever changes an offset this test is how we find
    out, rather than by a silent 0.2 mag bias in one code.
    """
    d2obs = pytest.importorskip(
        "digest2.observation", reason=NP.__name__ + " comparison needs digest2"
    )
    assert NP.BAND_TO_V == d2obs._BAND_CORRECTIONS
    # ... including the fallback for an unrecognised band.
    assert NP.BAND_TO_V_DEFAULT == d2obs._BAND_CORRECTIONS.get("no-such-band", -0.8)


@pytest.mark.parametrize(
    "band, mag, expect",
    [
        ("V", 20.0, 20.0),
        ("G", 22.29, 22.53),  # Gaia G: +0.24 (Gaia's single-char code)
        ("r", 19.0, 19.23),
        ("w", 21.0, 20.84),  # Pan-STARRS w: -0.16
        ("o", 18.0, 18.33),  # ATLAS orange
        ("u", 20.0, 22.5),
        (None, 20.0, 20.0),  # missing band -> V
        ("", 20.0, 20.0),
        ("Q", 20.0, 19.2),  # unknown band -> the -0.8 fallback
    ],
)
def test_band_to_v_values(band, mag, expect):
    assert NP.band_to_v(band, mag) == pytest.approx(expect, abs=1e-12)


def test_band_to_v_leaves_nonpositive_magnitude_alone():
    """digest2 applies the offset only when ``mag > 0`` (its no-photometry
    sentinel is 0.0), and so must we."""
    assert NP.band_to_v("G", 0.0) == 0.0
    assert NP.band_to_v("G", -3.0) == -3.0
    assert np.isnan(NP.band_to_v("G", float("nan")))


def test_band_alias_targets_are_all_real_digest2_bands():
    """Every alias must land on a single character digest2's table knows.

    A typo here would be invisible: the alias would miss, the band would fall
    through to the -0.8 unknown-band offset, and the photometry would be
    quietly wrong by ~0.7 mag on a third of the sample.
    """
    for code, target in NP.MPC_BAND_ALIASES.items():
        assert len(code) == 2, code
        assert len(target) == 1, (code, target)
        assert target in NP.BAND_TO_V, (code, target)
    # 'UNK' is a *present* band digest2 does not know; it must NOT be aliased,
    # so that it keeps taking digest2's own fallback.
    assert "UNK" not in NP.MPC_BAND_ALIASES
    assert NP.normalize_band("UNK") == "UNK"
    assert NP.band_to_v("UNK", 20.0) == pytest.approx(20.0 + NP.BAND_TO_V_DEFAULT)


@pytest.mark.parametrize(
    "code, expect_band, expect_offset",
    [
        ("Pw", "w", -0.16),   # Pan-STARRS broad w
        ("Pi", "i", 0.39),
        ("Ao", "o", 0.33),    # ATLAS orange
        ("Ac", "c", -0.05),   # ATLAS cyan
        ("Lr", "r", 0.23),    # Rubin r
        ("Ly", "y", 0.36),
        # Johnson/Cousins put the BAND first: 'Vj' is Johnson V, not J-band.
        ("Vj", "V", 0.0),
        ("Rc", "R", 0.4),
    ],
)
def test_mpc_two_character_band_codes_resolve(code, expect_band, expect_offset):
    assert NP.normalize_band(code) == expect_band
    assert NP.band_to_v(code, 20.0) == pytest.approx(20.0 + expect_offset, abs=1e-12)
    # The whole point: the alias must not fall through to the -0.8 fallback.
    assert NP.band_to_v(code, 20.0) != pytest.approx(20.0 + NP.BAND_TO_V_DEFAULT)


def test_normalize_band_handles_missing_and_padded_values():
    assert NP.normalize_band(None) == "V"
    assert NP.normalize_band("") == "V"
    assert NP.normalize_band("  ") == "V"
    assert NP.normalize_band(" Pw ") == "w"
    assert NP.normalize_band("G") == "G"


def test_band_to_v_agrees_with_digest2_on_every_band():
    d2obs = pytest.importorskip("digest2.observation")
    for band in list(NP.BAND_TO_V) + ["Q", "?"]:
        assert NP.band_to_v(band, 20.0) == pytest.approx(
            d2obs._update_magnitude(band, 20.0), abs=1e-12
        )


# --------------------------------------------------------------------------
# Observatory codes
# --------------------------------------------------------------------------

_OBSCODES_FIXTURE = """<html><body><pre>
Code  Long.    cos       sin     Name
500   0.0000   0.000000 +0.000000 Geocentric
309   289.59569 0.909943 -0.414336 Cerro Paranal
G96   249.21128 0.845107 +0.533611 <a href="x">Mt. Lemmon Survey</a>
002   0.62     0.622    +0.781    Rayleigh
258                              Gaia
247                              Roving Observer
</pre></body></html>
"""


def test_parse_obscodes_fixture(tmp_path):
    path = tmp_path / "ObsCodesF.html"
    path.write_text(_OBSCODES_FIXTURE)
    table = NP.parse_obscodes(str(path))
    assert set(table) == {"500", "309", "G96", "002"}, "space/roving codes must drop"
    assert table["G96"]["lon_deg"] == pytest.approx(249.21128)
    assert table["G96"]["rho_cos_phi"] == pytest.approx(0.845107)
    assert table["G96"]["rho_sin_phi"] == pytest.approx(0.533611)
    assert "Mt. Lemmon" in table["G96"]["name"], "HTML links must be stripped"
    # Low-precision rows (the MPC publishes some to 3 digits) still parse.
    assert table["002"]["rho_sin_phi"] == pytest.approx(0.781)


def test_obscode_geodetic_reproduces_builtin_table():
    """MPC parallax constants -> WGS84 -> ITRS must land back on the same site.

    This is the correctness guarantee for scoring at arbitrary obscodes: the
    pilot resolves stations outside priora's 14-entry built-in table by this
    route, so it has to agree with the built-in table wherever both exist.
    """
    import astropy.units as u

    worst = 0.0
    for code, (lon, rho_cos, rho_sin) in MPC_OBSCODES.items():
        if code == "500":
            continue
        row = {"lon_deg": lon, "rho_cos_phi": rho_cos, "rho_sin_phi": rho_sin}
        lon_d, lat_d, height = NP.obscode_geodetic(row)
        a = observatory_location(obscode=code)
        b = observatory_location(lon_deg=lon_d, lat_deg=lat_d, height_m=height)
        d = np.linalg.norm(
            a.get_itrs().cartesian.xyz.to_value(u.m)
            - b.get_itrs().cartesian.xyz.to_value(u.m)
        )
        worst = max(worst, float(d))
    assert worst < 1e-3, f"geodetic round trip off by {worst} m"


def test_obscode_geodetic_of_the_geocentre_is_degenerate():
    """Code 500 has zero parallax constants; the geodetic form is meaningless.

    Documented rather than special-cased: a geocentric "discovery station" is
    not a thing in the NEOCP archive, and the eligible-pool filter would have
    to invent an answer for it. The height comes back as minus the polar
    radius, which is the tell if it ever shows up.
    """
    _lon, _lat, height = NP.obscode_geodetic(
        {"lon_deg": 0.0, "rho_cos_phi": 0.0, "rho_sin_phi": 0.0}
    )
    assert height == pytest.approx(-6356752.3142, abs=1.0)


# --------------------------------------------------------------------------
# The attributable fit
# --------------------------------------------------------------------------


def test_fit_attributable_recovers_an_exact_line():
    jd0 = 2459200.5
    t = jd0 + np.array([0.0, 0.007, 0.015, 0.021])
    ra0, dec0, ra_dot, dec_dot = 123.456, -7.89, 3.5, -1.25  # deg, deg/day
    ra = ra0 + ra_dot * (t - jd0)
    dec = dec0 + dec_dot * (t - jd0)
    fit = NP.fit_attributable(t, ra, dec)
    assert fit["epoch_jd_utc"] == pytest.approx(jd0, abs=1e-12)
    assert fit["ra"] == pytest.approx(np.radians(ra0), abs=1e-12)
    assert fit["dec"] == pytest.approx(np.radians(dec0), abs=1e-12)
    assert fit["ra_rate"] == pytest.approx(np.radians(ra_dot), rel=1e-12)
    assert fit["dec_rate"] == pytest.approx(np.radians(dec_dot), rel=1e-12)
    assert fit["resid_rms_arcsec"] == pytest.approx(0.0, abs=1e-6)
    assert fit["arc_hours"] == pytest.approx(0.021 * 24.0)
    assert fit["n_det"] == 4


def test_fit_attributable_unwraps_ra_across_zero():
    """A tracklet straddling alpha = 0 must not fit a 360 deg/day rate."""
    jd0 = 2459200.5
    t = jd0 + np.array([0.0, 0.01, 0.02])
    ra_dot = 12.0
    ra_true = (359.9 + ra_dot * (t - jd0)) % 360.0
    assert ra_true[0] > 300.0 and ra_true[-1] < 60.0, "fixture must wrap"
    fit = NP.fit_attributable(t, ra_true, np.array([1.0, 1.1, 1.2]))
    assert fit["ra_rate"] == pytest.approx(np.radians(ra_dot), rel=1e-10)
    assert fit["ra"] == pytest.approx(np.radians(359.9), abs=1e-10)


def test_fit_attributable_is_order_insensitive():
    jd0 = 2459200.5
    t = jd0 + np.array([0.02, 0.0, 0.01])
    ra = 10.0 + 2.0 * (t - jd0)
    dec = 5.0 - 1.0 * (t - jd0)
    fit = NP.fit_attributable(t, ra, dec)
    # The epoch is the EARLIEST detection whatever order the rows arrived in.
    assert fit["epoch_jd_utc"] == pytest.approx(jd0)
    assert fit["ra"] == pytest.approx(np.radians(10.0), abs=1e-12)


def test_fit_attributable_rejects_single_detection():
    with pytest.raises(ValueError, match="at least 2"):
        NP.fit_attributable(np.array([2459200.5]), np.array([1.0]), np.array([2.0]))


def test_fit_attributable_zero_length_arc_gives_nan_rates():
    """Two detections at the identical timestamp: no rate is determined."""
    t = np.full(3, 2459200.5)
    fit = NP.fit_attributable(t, np.array([10.0, 10.0, 10.0]), np.zeros(3))
    assert np.isnan(fit["ra_rate"]) and np.isnan(fit["dec_rate"])
    assert fit["arc_hours"] == 0.0
    assert fit["ra"] == pytest.approx(np.radians(10.0))


def test_fit_attributable_residual_flags_a_bad_detection():
    jd0 = 2459200.5
    t = jd0 + np.array([0.0, 0.005, 0.010, 0.015])
    ra = 100.0 + 1.0 * (t - jd0)
    dec = np.zeros_like(t)
    dec[2] += 2.0 / 3600.0  # one detection 2 arcsec off the line
    fit = NP.fit_attributable(t, ra, dec)
    assert 0.3 < fit["resid_rms_arcsec"] < 3.0


def _kepler_tracklet(arc_min, n=5, a=1.6, e=0.5, inc=0.12, Om=1.0, om=2.0,
                     M0=0.3, H=20.0, jd0=2459200.5, obscode="G96"):
    """A real two-body tracklet: elements propagated, observables generated.

    Mean anomaly advances at the two-body mean motion; every other element is
    fixed (osculating, no perturbations), which is exactly the model
    ``observables_from_elements`` inverts.
    """
    mean_motion = K_GAUSS / a**1.5
    dt = np.linspace(0.0, arc_min / 1440.0, n)
    jd = jd0 + dt
    el = np.column_stack(
        [
            np.full(n, a),
            np.full(n, e),
            np.full(n, inc),
            np.full(n, Om),
            np.full(n, om),
            M0 + mean_motion * dt,
        ]
    )
    epochs = Time(jd, format="jd", scale="utc")
    R, V = observer_state(epochs, obscode=obscode)
    obs = T.observables_from_elements(el, np.full(n, H), R, V)
    # Truth at the first detection's epoch, the fit's reference epoch.
    R0, V0 = observer_state(epochs[0], obscode=obscode)
    truth = T.observables_from_elements(
        el[:1], np.array([H]), R0[None, :], V0[None, :]
    )
    return jd, obs, {k: float(v[0]) for k, v in truth.items()}


def test_fit_attributable_roundtrip_against_observables_from_elements():
    """Synthetic Keplerian tracklet -> fit -> compare with the generator.

    Tolerances are set by *curvature*, not by numerics: a straight-line fit of
    a curved arc cannot do better.  At a 60-minute arc (already longer than
    the real NEOCP discovery tracklets this harness sees) the fitted position
    at the reference epoch is within 20 mas and the rates within 0.2%.
    """
    jd, obs, truth = _kepler_tracklet(60.0, n=5)
    fit = NP.fit_attributable(jd, np.degrees(obs["ra"]), np.degrees(obs["dec"]))

    dra = (fit["ra"] - truth["ra"]) * np.cos(truth["dec"]) * ARCSEC
    ddec = (fit["dec"] - truth["dec"]) * ARCSEC
    assert abs(dra) < 0.02, f"RA off by {dra} arcsec"
    assert abs(ddec) < 0.02, f"Dec off by {ddec} arcsec"
    assert fit["ra_rate"] == pytest.approx(truth["ra_rate"], rel=2e-3)
    assert fit["dec_rate"] == pytest.approx(truth["dec_rate"], rel=2e-3)
    # A pure two-body arc has no scatter about its own smooth curve, so the
    # residual RMS measures only the curvature the straight line cannot follow.
    assert fit["resid_rms_arcsec"] < 0.05


def test_fit_attributable_roundtrip_error_is_curvature():
    """Halving the arc must quarter the position error and halve the rate error.

    That scaling is the signature of a correct linear estimator applied to a
    quadratic truth; any indexing, epoch or unit error would break the powers.
    """
    pos_err, rate_err = [], []
    for arc in (60.0, 30.0, 15.0):
        jd, obs, truth = _kepler_tracklet(arc, n=5)
        fit = NP.fit_attributable(jd, np.degrees(obs["ra"]), np.degrees(obs["dec"]))
        pos_err.append(
            abs(fit["ra"] - truth["ra"]) * np.cos(truth["dec"]) * ARCSEC
        )
        rate_err.append(abs(fit["ra_rate"] / truth["ra_rate"] - 1.0))
    for k in range(2):
        assert pos_err[k] / pos_err[k + 1] == pytest.approx(4.0, rel=0.25)
        assert rate_err[k] / rate_err[k + 1] == pytest.approx(2.0, rel=0.15)


def test_fit_attributable_roundtrip_for_a_main_belt_arc():
    """Same check for a slow, distant tracklet (the non-NEO half of the sample)."""
    jd, obs, truth = _kepler_tracklet(45.0, n=4, a=2.7, e=0.13, inc=0.05)
    fit = NP.fit_attributable(jd, np.degrees(obs["ra"]), np.degrees(obs["dec"]))
    assert fit["ra_rate"] == pytest.approx(truth["ra_rate"], rel=3e-3)
    assert fit["dec_rate"] == pytest.approx(truth["dec_rate"], rel=1e-2)
    assert abs(fit["ra"] - truth["ra"]) * ARCSEC < 0.02


# --------------------------------------------------------------------------
# Night clustering, designations, archive parsing
# --------------------------------------------------------------------------


def test_cluster_nights_splits_on_the_gap_only():
    t = np.array([0.0, 0.01, 0.02, 1.0, 1.01, 5.0])
    groups = NP.cluster_nights(t, gap_hours=3.0)
    assert [sorted(g.tolist()) for g in groups] == [[0, 1, 2], [3, 4], [5]]


def test_cluster_nights_sorts_unsorted_input():
    t = np.array([1.01, 0.0, 1.0, 0.02])
    groups = NP.cluster_nights(t, gap_hours=3.0)
    assert [sorted(g.tolist()) for g in groups] == [[1, 3], [0, 2]]
    assert NP.cluster_nights(np.array([])) == []


@pytest.mark.parametrize(
    "desig, expect",
    [
        ("2021 XS6", False),
        ("557271", False),
        ("(125368)", False),
        ("C/2019 E3", True),
        ("P/2021 R8", True),
        ("A/2021 X2", True),
        ("141P", True),
        ("(108P)", True),
        ("304P/Ory", True),
        ("1I/2017 U1", True),
    ],
)
def test_is_comet_designation(desig, expect):
    assert NP.is_comet_designation(desig) is expect


def test_designation_key_strips_parentheses_and_whitespace():
    assert NP.designation_key("  (557271) ") == "557271"
    assert NP.designation_key("2021  XS6") == "2021 XS6"


def test_timestamp_to_jd_handles_unpadded_dates():
    """``elsewhere.csv`` writes "2019-3-21T19:57:17"; astropy's isot rejects it."""
    a = NP._timestamp_to_jd_utc("2019-3-21T19:57:17")
    b = float(Time("2019-03-21T19:57:17", format="isot", scale="utc").jd)
    assert a == pytest.approx(b, abs=1e-9)
    assert np.isnan(NP._timestamp_to_jd_utc("not a date"))


_ELSEWHERE_FIXTURE = """trksub, original_disposition, reference, current_disposition, current_designation, discovery_obscode, timestamp
A0aaaa1,2021 AA1,MPEC 2021-A01,published,2021 AA1,G96,2021-1-05T10:00:00
A0aaaa2,was not confirmed,None,ITF,None,G96,2021-1-06T10:00:00
A0aaaa3,C/2021 B1,MPEC 2021-B01,published,C/2021 B1,F51,2021-1-07T10:00:00
A0aaaa4,2021 AA1,None,published,2021 AA1,F51,2021-1-08T10:00:00
A0aaaa5,2021 CC3,None,published,2021 CC3,C51,2021-1-09T10:00:00
A0aaaa6,2021 DD4,None,published,(125368),309,2021-1-10T10:00:00
"""


def test_load_elsewhere_and_eligible_pool(tmp_path):
    path = tmp_path / "elsewhere.csv"
    path.write_text(_ELSEWHERE_FIXTURE)
    cands = NP.load_elsewhere(str(path))
    assert len(cands) == 6
    assert cands[0].trksub == "A0aaaa1"
    assert cands[1].designation == "None"

    oc_path = tmp_path / "ObsCodesF.html"
    oc_path.write_text(
        _OBSCODES_FIXTURE.replace("309   289", "F51   203.74409 0.936241 "
                                  "+0.351543 Pan-STARRS 1\n309   289")
    )
    table = NP.parse_obscodes(str(oc_path))
    pool, counts = NP.eligible_pool(cands, table)

    assert counts["not_published"] == 1          # the ITF row
    assert counts["comet"] == 1                  # C/2021 B1
    assert counts["obscode_not_ground_based"] == 1  # C51 (NEOWISE)
    assert counts["duplicate_designation"] == 1  # 2021 AA1 twice
    assert set(pool) == {"2021 AA1", "125368"}
    assert counts["eligible"] == 2
    # Deduplication keeps the EARLIEST posting of a repeated object.
    assert pool["2021 AA1"].trksub == "A0aaaa1"


def test_select_sample_is_deterministic_and_label_blind(tmp_path):
    pool = {f"2021 A{i}": NP.Candidate(
        f"t{i}", "", "", "published", f"2021 A{i}", "G96", "2021-1-01T00:00:00"
    ) for i in range(50)}
    a = [c.key for c in NP.select_sample(pool, 10, seed=5)]
    b = [c.key for c in NP.select_sample(pool, 10, seed=5)]
    c = [c.key for c in NP.select_sample(pool, 10, seed=6)]
    assert a == b and a != c and len(set(a)) == 10
    # Asking for more than exists returns everything, not an error.
    assert len(NP.select_sample(pool, 999, seed=5)) == 50


def test_load_nea_designations_collects_every_alias(tmp_path):
    path = tmp_path / "nea_extended.json.gz"
    with gzip.open(path, "wt") as fh:
        json.dump(
            [
                {
                    "Number": "(433)",
                    "Principal_desig": "A898 PA",
                    "Other_desigs": ["1956 PC"],
                    "Perihelion_dist": 1.1332,
                },
                {"Principal_desig": "2021 AA1", "Perihelion_dist": 0.9},
            ],
            fh,
        )
    names, q = NP.load_nea_designations(str(path))
    assert {"433", "A898 PA", "1956 PC", "2021 AA1"} <= names
    assert q["1956 PC"] == pytest.approx(1.1332)
    assert q["2021 AA1"] == pytest.approx(0.9)


# --------------------------------------------------------------------------
# The offline build stage, end to end on a synthetic cache
# --------------------------------------------------------------------------


def _write_cache(root, *, objects):
    """Lay out a raw/ cache the way ``fetch`` would, from inline fixtures."""
    raw = os.path.join(root, "raw")
    os.makedirs(os.path.join(raw, "obs"), exist_ok=True)
    os.makedirs(os.path.join(raw, "sbdb"), exist_ok=True)
    with open(os.path.join(raw, "ObsCodesF.html"), "w") as fh:
        fh.write(_OBSCODES_FIXTURE)
    with gzip.open(os.path.join(raw, "nea_extended.json.gz"), "wt") as fh:
        json.dump(
            [{"Principal_desig": d, "Perihelion_dist": q}
             for d, q in objects.get("_nea", [])],
            fh,
        )
    sample = []
    for spec in objects["objects"]:
        name = NP._safe_name(spec["designation"]) + ".json.gz"
        with gzip.open(os.path.join(raw, "obs", name), "wt") as fh:
            json.dump([{"ADES_DF": spec["obs"], "OBS80": None}, 200], fh)
        with gzip.open(os.path.join(raw, "sbdb", name), "wt") as fh:
            json.dump(spec["sbdb"], fh)
        sample.append(
            {
                "trksub": spec.get("trksub", "T00000"),
                "original": "",
                "reference": "",
                "disposition": "published",
                "designation": spec["designation"],
                "discovery_obscode": spec["obscode"],
                "timestamp": spec["timestamp"],
            }
        )
    with open(os.path.join(root, "sample.json"), "w") as fh:
        json.dump({"sample": sample, "selection_counts": {"eligible": len(sample)}},
                  fh)
    return root


def _ades(stn, iso, ra, dec, mag="20.50", band="G", trksub="T00000"):
    return {
        "stn": stn,
        "obstime": iso,
        "ra": f"{ra:.7f}",
        "dec": f"{dec:.7f}",
        "mag": mag,
        "band": band,
        "trksub": trksub,
    }


def _sbdb(q, e=0.4, i=10.0, a=2.0, H=20.0, code="APO", kind="au", neo=True):
    return {
        "object": {"kind": kind, "orbit_class": {"code": code}, "neo": neo},
        "orbit": {
            "n_obs_used": 42,
            "elements": [
                {"name": "q", "value": str(q)},
                {"name": "e", "value": str(e)},
                {"name": "i", "value": str(i)},
                {"name": "a", "value": str(a)},
                {"name": "H", "value": str(H)},
            ],
        },
    }


def _standard_cache(root):
    """One good NEO, one good non-NEO, and four objects that each break a rule."""
    good_neo = [
        _ades("G96", f"2021-01-05T10:0{k}:00.000Z", 100.0 + 0.02 * k,
              -5.0 + 0.01 * k)
        for k in range(4)
    ]
    # Same station, but the earliest night is a later-linked precovery two
    # months before the NEOCP posting: the window rule must skip it.
    precovery = [
        _ades("G96", f"2020-10-01T10:0{k}:00.000Z", 80.0 + 0.02 * k, 1.0)
        for k in range(3)
    ]
    # Band "Pw" is the MPC ADES-dataframe spelling of Pan-STARRS w; the build
    # must alias it rather than let it fall through to the -0.8 fallback.
    slow = [
        _ades("G96", f"2021-01-06T09:0{k}:00.000Z", 200.0 + 0.001 * k, 12.0,
              mag="21.80", band="Pw")
        for k in range(3)
    ]
    return _write_cache(
        root,
        objects={
            "_nea": [("2021 AA1", 0.95)],
            "objects": [
                {
                    "designation": "2021 AA1",
                    "obscode": "G96",
                    "timestamp": "2021-1-07T10:00:00",
                    "obs": precovery + good_neo,
                    "sbdb": _sbdb(0.95, code="APO", neo=True),
                },
                {
                    "designation": "2021 BB2",
                    "obscode": "G96",
                    "timestamp": "2021-1-08T10:00:00",
                    "obs": slow,
                    "sbdb": _sbdb(2.1, e=0.1, a=2.4, code="MBA", neo=False),
                },
                {   # only one detection at the discovery station
                    "designation": "2021 CC3",
                    "obscode": "G96",
                    "timestamp": "2021-1-09T10:00:00",
                    "obs": [_ades("G96", "2021-01-09T10:00:00.000Z", 5.0, 5.0),
                            _ades("F51", "2021-01-09T12:00:00.000Z", 5.1, 5.1)],
                    "sbdb": _sbdb(1.0),
                },
                {   # SBDB says comet
                    "designation": "2021 DD4",
                    "obscode": "G96",
                    "timestamp": "2021-1-10T10:00:00",
                    "obs": good_neo,
                    "sbdb": _sbdb(1.0, kind="cn"),
                },
                {   # no photometry at all
                    "designation": "2021 EE5",
                    "obscode": "G96",
                    "timestamp": "2021-1-11T10:00:00",
                    "obs": [
                        _ades("G96", f"2021-01-11T10:0{k}:00.000Z", 9.0 + 0.01 * k,
                              9.0, mag="")
                        for k in range(3)
                    ],
                    "sbdb": _sbdb(1.0),
                },
                {   # station has no parallax constants in the fixture table
                    "designation": "2021 FF6",
                    "obscode": "703",
                    "timestamp": "2021-1-12T10:00:00",
                    "obs": [
                        _ades("703", f"2021-01-12T10:0{k}:00.000Z", 3.0 + 0.01 * k,
                              3.0)
                        for k in range(3)
                    ],
                    "sbdb": _sbdb(1.0),
                },
            ],
        },
    )


def test_build_tracklets_offline_census_and_geometry(tmp_path, monkeypatch):
    """The whole build stage on a synthetic cache, with the network disabled."""

    def _no_network(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("build() must not touch the network")

    monkeypatch.setattr("urllib.request.urlopen", _no_network)

    root = _standard_cache(str(tmp_path))
    tracklets, census = NP.build_tracklets(out_dir=root, verbose=False)

    assert census["built"] == 2
    assert census["no_obs_at_discovery_obscode"] == 1   # 2021 CC3
    assert census["cometary_by_sbdb_kind"] == 1         # 2021 DD4
    assert census["no_usable_magnitude"] == 1           # 2021 EE5
    assert census["obscode_unresolvable"] == 1          # 2021 FF6 at code 703
    assert census["n_neo"] == 1 and census["n_non_neo"] == 1

    by_desig = {t.designation: t for t in tracklets}
    neo = by_desig["2021 AA1"]
    assert neo.is_neo is True and neo.q_au == pytest.approx(0.95)
    assert neo.obscode == "G96"
    assert neo.n_det == 4
    # The window rule skipped the October precovery ...
    assert neo.epoch_jd_utc == pytest.approx(
        float(Time("2021-01-05T10:00:00", format="isot", scale="utc").jd)
    )
    # ... and the precovery is still *recorded* as a diagnostic.
    assert neo.precovery_exists is True
    assert neo.window_fallback is False
    assert neo.days_before_neocp_exit == pytest.approx(2.0, abs=1e-6)
    # Band G -> V is +0.24 on every detection, so the median moves with it.
    assert neo.v_mag == pytest.approx(20.50 + 0.24, abs=1e-9)
    assert neo.bands == ["G"]
    assert len(neo.obs_times_jd) == len(neo.obs_mag) == 4
    assert neo.lat_deg == pytest.approx(32.44, abs=0.05)  # Mt Lemmon

    other = by_desig["2021 BB2"]
    assert other.is_neo is False and other.orbit_class == "MBA"
    # Band "Pw" -> "w" -> -0.16, NOT the -0.8 unknown-band fallback.
    assert other.v_mag == pytest.approx(21.80 - 0.16, abs=1e-9)
    assert census["band_aliased"] == 3
    assert census["band_unknown_to_digest2"] == 0
    # Label sources agree: 2021 BB2 is absent from the fixture NEA list.
    assert other.label_source_agree is True
    # q for a non-NEA comes from SBDB; for the NEA it comes from the MPC file.
    assert other.q_source == "sbdb" and other.q_au == pytest.approx(2.1)
    assert neo.q_source == "mpc_nea" and neo.q_au == pytest.approx(0.95)


def test_build_tracklets_prefers_full_precision_q_on_the_class_boundary(tmp_path):
    """SBDB rounds q to 3 significant figures; a boundary Amor comes back as
    exactly "1.3" and an SBDB-only ``q < 1.3`` test calls it a non-NEO.

    Three objects in the real 500-object sample sit exactly there, so this is a
    measured failure mode, not a hypothetical one.
    """
    root = _write_cache(
        str(tmp_path),
        objects={
            "_nea": [("2020 KT5", 1.29987)],
            "objects": [
                {
                    "designation": "2020 KT5",
                    "obscode": "G96",
                    "timestamp": "2020-6-01T10:00:00",
                    "obs": [
                        _ades("G96", f"2020-05-30T10:0{k}:00.000Z",
                              40.0 + 0.02 * k, 3.0)
                        for k in range(3)
                    ],
                    "sbdb": _sbdb(1.3, code="AMO", neo=True),
                },
            ],
        },
    )
    tracklets, census = NP.build_tracklets(out_dir=root, verbose=False)
    assert census["built"] == 1
    t = tracklets[0]
    assert t.is_neo is True, "the MPC's 7-digit q must win over SBDB's '1.3'"
    assert t.q_source == "mpc_nea"
    assert t.q_au == pytest.approx(1.29987)
    assert t.q_sbdb == pytest.approx(1.3)
    # The two sources genuinely disagree here, and that is recorded.
    assert census["label_source_disagree"] == 1
    assert t.label_source_agree is False


def test_build_tracklets_truncates_an_overlong_night(tmp_path):
    root = _write_cache(
        str(tmp_path),
        objects={
            "_nea": [],
            "objects": [
                {
                    "designation": "2021 GG7",
                    "obscode": "G96",
                    "timestamp": "2021-1-06T10:00:00",
                    "obs": [
                        _ades("G96", f"2021-01-05T{10 + 2 * k:02d}:00:00.000Z",
                              50.0 + 0.5 * k, 0.0)
                        for k in range(3)
                    ],
                    "sbdb": _sbdb(1.9, code="MBA", neo=False),
                },
            ],
        },
    )
    # Detections 2 h apart: the night clusters into three singletons at the
    # default 3 h gap, so widen the gap and let the span rule do the cutting.
    tracklets, census = NP.build_tracklets(
        out_dir=root, gap_hours=6.0, max_span_hours=2.0, verbose=False
    )
    assert census["truncated_span"] == 1
    assert tracklets[0].n_det == 2
    assert tracklets[0].arc_hours == pytest.approx(2.0, abs=1e-6)


def test_build_tracklets_window_fallback_is_flagged(tmp_path):
    """If the only night at the discovery station predates the window, use it
    anyway — but say so."""
    root = _write_cache(
        str(tmp_path),
        objects={
            "_nea": [],
            "objects": [
                {
                    "designation": "2021 HH8",
                    "obscode": "G96",
                    "timestamp": "2021-6-01T10:00:00",
                    "obs": [
                        _ades("G96", f"2020-01-05T10:0{k}:00.000Z",
                              50.0 + 0.01 * k, 0.0)
                        for k in range(3)
                    ],
                    "sbdb": _sbdb(1.5, code="MCA", neo=False),
                },
            ],
        },
    )
    tracklets, census = NP.build_tracklets(out_dir=root, verbose=False)
    assert census["built"] == 1 and census["window_fallback"] == 1
    assert tracklets[0].window_fallback is True


def test_build_tracklets_survives_an_api_error_payload(tmp_path):
    """A cached ``{"error": ...}`` reply must be counted, not crash the run."""
    root = _write_cache(
        str(tmp_path),
        objects={"_nea": [], "objects": [
            {
                "designation": "2021 JJ9",
                "obscode": "G96",
                "timestamp": "2021-1-06T10:00:00",
                "obs": [],
                "sbdb": _sbdb(1.0),
            }
        ]},
    )
    raw = os.path.join(root, "raw")
    name = NP._safe_name("2021 JJ9") + ".json.gz"
    with gzip.open(os.path.join(raw, "obs", name), "wt") as fh:
        json.dump({"error": "input_error", "message": "nope"}, fh)
    with gzip.open(os.path.join(raw, "sbdb", name), "wt") as fh:
        json.dump({"message": "specified object was not found"}, fh)
    tracklets, census = NP.build_tracklets(out_dir=root, verbose=False)
    assert tracklets == []
    assert census["obs_empty_or_error"] == 1


def test_build_writes_a_reloadable_tracklet_file(tmp_path):
    root = _standard_cache(str(tmp_path))
    census = NP.build(out_dir=root, verbose=False)
    assert census["built"] == 2
    tracklets, census2 = NP.load_tracklets(root)
    assert census2 == census
    assert [t.designation for t in tracklets] == ["2021 AA1", "2021 BB2"]
    assert isinstance(tracklets[0], NP.Tracklet)


# --------------------------------------------------------------------------
# The digest2 hand-off
# --------------------------------------------------------------------------


def _synthetic_tracklets():
    jd0 = float(Time("2021-01-05T10:00:00", format="isot", scale="utc").jd)
    out = []
    for i in range(3):
        jd = [jd0 + i + k * 0.005 for k in range(3)]
        out.append(
            NP.Tracklet(
                designation=f"2021 A{i}",
                trksub=f"T{i}",
                obscode="G96",
                epoch_jd_utc=jd[0],
                ra=np.radians(100.0 + i),
                dec=np.radians(-5.0),
                ra_rate=0.01,
                dec_rate=0.002,
                v_mag=20.0 + i,
                n_det=3,
                arc_hours=0.24,
                resid_rms_arcsec=0.2,
                obs_times_jd=jd,
                obs_ra_deg=[100.0 + i + 0.01 * k for k in range(3)],
                obs_dec_deg=[-5.0 + 0.004 * k for k in range(3)],
                obs_mag=[20.0 + i, float("nan"), 20.2 + i],
            )
        )
    return out


def test_emit_ades_psv_is_parsed_back_by_digest2(tmp_path):
    d2obs = pytest.importorskip(
        "digest2.observation",
        reason='install with CFLAGS="-D_GNU_SOURCE" pip install --user '
        '"git+https://github.com/Smithsonian/mpc-public.git#subdirectory=digest2"',
    )
    tracklets = _synthetic_tracklets()
    path = str(tmp_path / "t.psv")
    manifest = NP.emit_ades_psv(tracklets, path)
    assert manifest["n_tracklets"] == 3 and manifest["n_observations"] == 9

    parsed = d2obs.parse_ades_psv(path)
    assert set(parsed) == {"N00000", "N00001", "N00002"}
    obs = parsed["N00001"]
    assert len(obs) == 3
    assert [o.obscode for o in obs] == ["G96"] * 3
    assert obs[0].ra == pytest.approx(101.0, abs=1e-7)
    assert obs[0].dec == pytest.approx(-5.0, abs=1e-7)
    # Magnitudes are pre-converted to V and the band is 'V', so digest2's own
    # conversion is the identity -- that is what makes the inputs identical.
    assert obs[0].mag == pytest.approx(21.0, abs=1e-6)
    # The NaN magnitude became a blank field, which digest2 reads as its
    # no-photometry sentinel rather than as a bright object.
    assert obs[1].mag == pytest.approx(0.0, abs=1e-9)
    # Times survive to better than a millisecond.
    mjd0 = tracklets[1].obs_times_jd[0] - 2400000.5
    assert obs[0].mjd == pytest.approx(mjd0, abs=1e-8)


def test_emit_ades_psv_lines_have_the_declared_columns(tmp_path):
    """No digest2 needed: check the header and field count of the emitted PSV."""
    path = str(tmp_path / "t.psv")
    NP.emit_ades_psv(_synthetic_tracklets(), path)
    lines = [ln for ln in open(path).read().splitlines() if ln and ln[0] != "#"]
    header = [c.strip() for c in lines[0].split("|")]
    assert header[:9] == ["permID", "provID", "trkSub", "mode", "stn",
                          "obsTime", "ra", "dec", "mag"]
    for ln in lines[1:]:
        fields = ln.split("|")
        assert len(fields) == len(header), ln
        assert fields[4].strip() == "G96"


# --------------------------------------------------------------------------
# Caching / offline guarantees
# --------------------------------------------------------------------------


def test_http_get_returns_the_cached_file_without_a_request(tmp_path, monkeypatch):
    dest = tmp_path / "cached.txt"
    dest.write_text("already here")

    def _no_network(*a, **k):  # pragma: no cover
        raise AssertionError("http_get must not re-fetch an existing cache entry")

    monkeypatch.setattr("urllib.request.urlopen", _no_network)
    assert NP.http_get("https://example.invalid/x", str(dest)) == str(dest)
    assert dest.read_text() == "already here"


def test_http_get_does_not_retry_a_client_error(tmp_path, monkeypatch):
    import urllib.error

    calls = []

    def _raise(*a, **k):
        calls.append(1)
        raise urllib.error.HTTPError("u", 400, "Bad Request", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    monkeypatch.setattr(NP, "RATE_LIMIT_S", 0.0)
    with pytest.raises(NP.FetchError, match="HTTP 400"):
        NP.http_get("https://example.invalid/y", str(tmp_path / "out"), retries=3)
    assert len(calls) == 1, "a 400 is deterministic; retrying wastes the service"


def test_http_get_retries_a_transient_error(tmp_path, monkeypatch):
    import urllib.error

    calls = []

    def _raise(*a, **k):
        calls.append(1)
        raise urllib.error.URLError("temporary")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    monkeypatch.setattr(NP, "RATE_LIMIT_S", 0.0)
    monkeypatch.setattr(NP.time, "sleep", lambda *_: None)
    with pytest.raises(NP.FetchError, match="after 3 tries"):
        NP.http_get("https://example.invalid/z", str(tmp_path / "out"), retries=3)
    assert len(calls) == 3


def test_user_agent_identifies_the_client_and_carries_no_address():
    assert "priora" in NP.USER_AGENT
    assert "@" not in NP.USER_AGENT, "no e-mail address in the UA"


# --------------------------------------------------------------------------
# The corrected paired bootstrap
# --------------------------------------------------------------------------


def _bootstrap_fixture(n_pos, n_neg, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n_pos + n_neg, bool)
    y[:n_pos] = True
    a = rng.normal(0, 1, y.size) + y * 1.5
    b = rng.normal(0, 1, y.size) + y * 1.0
    return a, b, y


def test_paired_bootstrap_works_when_positives_outnumber_negatives():
    """The regression the pilot was written into: a NEOCP-filtered sample has
    more positives than negatives, which is where the shared helper breaks."""
    a, b, y = _bootstrap_fixture(300, 200)
    out = NP.paired_auc_bootstrap(a, b, y, n_boot=300, seed=1)
    assert out["n_pos"] == 300 and out["n_neg"] == 200
    lo, hi = out["delta_ci"]
    assert np.isfinite(lo) and np.isfinite(hi), "CI must not be NaN"
    assert lo <= out["delta"] <= hi
    assert 0.0 <= out["p_delta_le_0"] <= 1.0
    assert np.isfinite(out["delta_boot_sd"]) and out["delta_boot_sd"] > 0


def test_paired_bootstrap_matches_the_shared_helper_where_that_one_is_sound():
    """With n_pos << n_neg the buggy indexing is only ~n_pos/n_neg
    contaminating, so the two implementations must agree to well inside the
    interval width — a guard that this module's rewrite did not change the
    estimator, only the resampling."""
    from validation.digest2_comparison import auc_bootstrap

    a, b, y = _bootstrap_fixture(60, 6000, seed=3)
    mine = NP.paired_auc_bootstrap(a, b, y, n_boot=400, seed=2)
    theirs = auc_bootstrap(a, b, y, n_boot=400, seed=2)
    # The point estimates are computed the same way and must be identical.
    assert mine["auc_a"] == pytest.approx(theirs["auc_a"], abs=1e-12)
    assert mine["delta"] == pytest.approx(theirs["delta"], abs=1e-12)
    width = mine["delta_ci"][1] - mine["delta_ci"][0]
    assert abs(mine["delta_ci"][0] - theirs["delta_ci"][0]) < 0.5 * width
    assert abs(mine["delta_ci"][1] - theirs["delta_ci"][1]) < 0.5 * width


def test_paired_bootstrap_resample_keeps_both_strata():
    """Directly assert the property the shared helper violates: a
    label-stratified resample must preserve the class counts exactly."""
    rng = np.random.default_rng(7)
    y = np.zeros(500, bool)
    y[:300] = True
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    for _ in range(20):
        sel = np.concatenate(
            (
                pos[rng.integers(0, pos.size, pos.size)],
                neg[rng.integers(0, neg.size, neg.size)],
            )
        )
        assert y[sel].sum() == 300 and (~y[sel]).sum() == 200


def test_paired_bootstrap_degenerates_gracefully():
    a, b, y = _bootstrap_fixture(1, 50)
    out = NP.paired_auc_bootstrap(a, b, y, n_boot=50, seed=1)
    assert np.isnan(out["delta_ci"][0]) and np.isnan(out["p_delta_le_0"])
    # Non-finite scores are dropped from BOTH sides, symmetrically, and counted.
    a2, b2, y2 = _bootstrap_fixture(40, 40, seed=9)
    a2[:5] = np.nan
    out2 = NP.paired_auc_bootstrap(a2, b2, y2, n_boot=50, seed=1)
    assert out2["n"] == 75 and out2["n_dropped_nonfinite"] == 5


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------


def _minimal_summary():
    return {
        "n_tracklets": 2,
        "n_neo": 1,
        "n_non_neo": 1,
        "neo_fraction": 0.5,
        "archive_timestamp_range": ["2019-03-21", "2021-12-21"],
        "pool_neo_fraction": 0.63,
        "selection": {
            "rows": 100, "not_published": 10, "no_designation": 1, "comet": 2,
            "no_obscode": 0, "obscode_not_ground_based": 3,
            "duplicate_designation": 4, "eligible": 80,
        },
        "census": {"n_sampled": 3, "built": 2, "fit_failed": 1,
                   "n_neo": 1, "n_non_neo": 1},
        "score_meta": {"n_tracklets": 3},
        "sources": {}, "sha256": {},
        "robustness": {
            "p_neo_nonfinite_all": 0, "p_neo_nonfinite_unk": 0,
            "empty_admissible_region": 0, "n_masked_det_total": 5,
            "ar_frac_h_above_median": 0.02, "ar_frac_h_above_gt_1e2": 0.6,
            "ar_frac_h_above_gt_1e1": 0.1, "n_clean": 0, "n_broken": 1,
            "priora_seconds_median": 3.4,
        },
        "tracklets": {
            "n_det": {"p10": 3, "median": 4, "p90": 4},
            "arc_hours": {"p10": 0.3, "median": 0.5, "p90": 1.0},
            "fit_resid_rms_arcsec": {"p10": 0.1, "median": 0.3, "p90": 0.6},
            "total_rate_deg_per_day": {"p10": 0.2, "median": 0.7, "p90": 16.0},
            "v_mag": {"p10": 19.0, "median": 20.3, "p90": 21.8},
            "q_au": {"p10": 0.9, "median": 1.1, "p90": 1.7},
            "n_precovery_exists": 1, "n_window_fallback": 0,
            "n_truncated_span": 0, "n_label_source_disagree": 0,
            "obscodes": {"G96": 2}, "bands": {"G": 2},
        },
        "score_distributions": {
            "priora All": {"neo": {"median": 1.0, "mean": 0.9, "p10": 0.5,
                                     "p90": 1.0},
                             "non_neo": {"median": 0.01, "mean": 0.02,
                                         "p10": 0.0, "p90": 0.05}},
        },
        "auc": {"all": {"n": 2, "n_neo": 1,
                        "single": {"priora All": 0.9},
                        "paired": {"a_vs_b": {"delta": 0.01,
                                              "delta_ci": [0.0, 0.02],
                                              "p_delta_le_0": 0.03}}}},
        "calibration": {"all": {"priora All": {
            "expected_positives": 1.1, "observed_positives": 1.0,
            "ratio_expected_observed": 1.1, "brier": 0.1, "ece": 0.05,
            "bins": [{"p_mean": 0.5, "obs_frac": 0.5}],
        }}},
        "by_q_bin": {"NEO (q<1.3)": {"n": 1, "mean_p_all": 0.9,
                                     "mean_p_unk": 0.8,
                                     "frac_p_all_ge_0.5": 1.0,
                                     "mean_d2_raw": 99.0,
                                     "frac_d2_ge_65": 1.0}},
        "plots": {},
    }


def test_build_report_renders_and_states_the_caveats():
    text = NP.build_report(_minimal_summary())
    assert text.startswith("# WP8")
    # The censoring caveat is the single most important sentence in the file.
    assert "conditioned on digest2 saying yes" in text
    assert "censored against digest2" in text
    assert "prior-misspecification stress test" in text
    # Dynamic bits actually rendered.
    assert "2019-03-21 to 2021-12-21" in text
    assert "`fit_failed` (1)" in text
    assert "| eligible | 80 |" in text
    # No unrendered placeholders or NaN leakage into the prose.
    assert "n/a" not in text.split("## Ranking")[0]


def test_build_report_survives_missing_optional_blocks():
    s = _minimal_summary()
    for key in ("auc", "calibration", "by_q_bin"):
        s[key] = {}
    s.pop("disagreement", None)
    text = NP.build_report(s)
    assert "## Ranking" in text and "## Assumptions" in text
