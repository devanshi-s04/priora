"""WP8 — real-data NEOCP pilot: priora v0 against the real sky (DESIGN §4.3).

What this is
------------
Everything priora has been measured on so far was drawn from its own prior:
the §4.1 self-consistency gate and the §4.2 digest2 head-to-head both sample
the test population *from* ``data/digest2.model.csv``, the file both codes read.
This harness is the first contact with real observations and real labels.  It
is therefore explicitly a DESIGN §4.3 **prior-misspecification stress test**,
not a correctness test: v0's prior is the 2011 S3M-derived digest2 histogram,
the real sky is not, and any miscalibration measured here is a sum of prior
mismatch, selection, and method — in that order of expected size.

The labelled sample
-------------------
The MPC publishes ``iau/NEO/elsewhere.csv`` — "previously posted NEOCP
candidates", one row per NEOCP posting since 2019-03-21, with the observer
tracklet id (``trksub``), the discovery observatory code, the disposition, and
the final designation where one was assigned.  That file is the label source:
a designated object's mature orbit gives an unambiguous q, and D-2 defines the
NEO class by exactly that boundary (q < 1.3 AU).  Perihelion distances come
from JPL's SBDB (per object) and are cross-checked against the MPC's own
``nea_extended.json.gz`` NEA list.

**Selection caveats — read these before reading any number below.**

1. *The sample is conditioned on digest2 saying yes.*  The MPC posts a tracklet
   to the NEOCP when digest2's score clears its policy threshold (score >= 65
   on the NEO class, plus operator judgement).  Every object here therefore
   already passed a digest2 cut.  A head-to-head ROC on this sample is
   **censored against digest2**: digest2's own false negatives are absent by
   construction, so its measured completeness is not its true completeness, and
   the comparison cannot be read as "which code would triage better".  What it
   *can* measure is the discrimination each code retains *within* the set
   digest2 already accepted — i.e. how much of the residual non-NEO
   contamination each code could remove.
2. *Only designated objects carry a label.*  Roughly 15% of NEOCP postings are
   never designated ("was not confirmed", "does not exist", ITF) and are
   dropped: their truth is genuinely unknown, and they are plausibly enriched
   in both hard NEOs (fast, lost) and noise.
3. *Comets are excluded* — the digest2 population model has no cometary
   component (DESIGN §5), so a comet is neither a NEO nor an "other" in the
   sense the two components partition.  They are counted, not scored.
4. *Space-based discovery stations are excluded* (C51/NEOWISE, 258/Gaia, ...):
   the MPC obscode table carries no parallax constants for them and priora's
   observer model has no spacecraft ephemeris.
5. *The archive spans 2019-03-21 to 2021-12-21* and stops there; the rolling
   ``ToConfirm_PrevDes.html`` page (~4 months, no years) is fetched as a
   contemporary cross-check on the disposition vocabulary only.
6. *The base rate is the sample's own.*  Objects are drawn by uniform random
   sample from the eligible pool, **not** balanced by label, precisely so that
   the observed NEO fraction is the real NEOCP-conditioned one and the
   calibration question ("does mean p match the observed NEO fraction?") has a
   meaningful answer.

The discovery tracklet
----------------------
For each object the MPC observations API (``data.minorplanetcenter.net/api/
get-obs``, ADES dataframe form) returns the full observation history.  The
NEOCP-triggering tracklet is reconstructed as: observations at the row's
``discovery_obscode``, grouped into nights (gap > 3 h), taking the earliest
night that falls within ``--window-days`` before the NEOCP-exit timestamp.
That window matters: the MPC back-fills the NEOCP ``trksub`` onto *every*
observation of the object once linked, including self-precoveries months
earlier that were not available at triage time, so "earliest observation
overall" is the wrong tracklet.  Nights longer than ``--max-span-hours`` are
truncated to the first observations inside that span.

The attributable is an ordinary (unweighted) least-squares straight-line fit of
alpha(t) and delta(t) over the night's detections, evaluated **at the first
detection's epoch** — the WP6/WP7 convention, and also what digest2 anchors on
(``obsPair[0]``).  RA is unwrapped about the first detection before fitting.
The magnitude is the median over the tracklet of the reported magnitudes after
band -> V conversion using digest2's own offset table (see :data:`BAND_TO_V`),
so both codes standardise photometry identically.

Both codes, same inputs
-----------------------
digest2 2.8.2 (the WP7 pip build) is run locally on the *same* detections, from
an ADES PSV file carrying the real station, time and (already V-standardised)
magnitude — so the comparison is same-inputs, not against archived NEOCP scores
of unknown digest2 vintage or unknown model file.  Both codes read
``data/digest2.model.csv``.

priora is scored **three** times on identical inputs:

* ``All`` — the v0 default, the counterpart of digest2's ``raw`` score;
* ``Unk`` — the "yet undiscovered" tables, the counterpart of digest2's default
  ``noid`` score.  This is the D-10 open question the pilot is meant to inform:
  a NEOCP candidate is by definition an undiscovered object, so ``Unk`` is
  *a priori* the right prior for this use case.  The data disagrees; see the
  report.
* ``All`` with ``h_top_bin_infinite=True`` — digest2's own [24, inf) top H bin
  instead of priora's hard truncation at H = 25.5 (D-4).  Not a cosmetic
  control on real data: for close, fast NEOs the *entire* admissible region can
  imply H > 25.5, at which point both truncated components are zero everywhere
  and the posterior is 0/0.  ~2% of this sample, all of them true NEOs.

Two things real data broke that no synthetic test could have caught, both
handled and both reported: the MPC's ADES dataframe uses two-character band
codes that digest2's offset table does not know (:data:`MPC_BAND_ALIASES`), and
JPL's SBDB returns display-rounded elements, so boundary Amors arrive as exactly
``q = 1.3`` (see the labelling in :func:`build_tracklets`).  A third finding is
a genuine bug in ``digest2_comparison.auc_bootstrap``, which this sample's
label balance exposes — reported, not patched; see
:func:`paired_auc_bootstrap`.

CLI
---
::

    python -m validation.neocp_pilot fetch   --n 500 --seed 42
    python -m validation.neocp_pilot build
    python -m validation.neocp_pilot score   --workers 90
    python -m validation.neocp_pilot analyze
    python -m validation.neocp_pilot all     --n 500 --workers 90

Every raw download is cached under ``validation/output/neocp_pilot/raw/``, so
``build``/``score``/``analyze`` are fully offline and re-runnable.  ``fetch``
is the only stage that touches the network; it rate-limits itself, sends a
plain identifying User-Agent, and asks for one object at a time because the
MPC API's designation limit is 1.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np

# Headless: this runs on batch nodes.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

__all__ = [
    "BAND_TO_V",
    "BAND_TO_V_DEFAULT",
    "MPC_BAND_ALIASES",
    "DEFAULT_OUT",
    "H_TRUNC",
    "Q_NEO",
    "Candidate",
    "FetchError",
    "Tracklet",
    "analyze",
    "band_to_v",
    "build",
    "build_report",
    "build_tracklets",
    "calibration_quantile",
    "cluster_nights",
    "designation_key",
    "eligible_pool",
    "emit_ades_psv",
    "fetch",
    "fetch_object",
    "fetch_static",
    "fit_attributable",
    "http_get",
    "is_comet_designation",
    "load_elsewhere",
    "load_nea_designations",
    "load_tracklets",
    "normalize_band",
    "obscode_geodetic",
    "paired_auc_bootstrap",
    "parse_obscodes",
    "reliability_quantile",
    "score",
    "score_digest2",
    "score_priora",
    "select_sample",
    "main",
]

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DEFAULT_OUT = os.path.join(REPO_ROOT, "validation", "output", "neocp_pilot")
DEFAULT_MODEL = os.path.join(REPO_ROOT, "data", "digest2.model.csv")

#: DESIGN §2.6 truncation diagnostic threshold: the digest2 model's H support.
H_TRUNC = 25.5

#: The D-19 trigger's measure threshold, imported from the classifier rather
#: than restated, so this module's H-wall detector cannot drift from the branch
#: it is reconstructing.
from priora.classifier import H_TRUNCATION_MEASURE_MIN  # noqa: E402

#: D-2 / DESIGN §1: the NEO class boundary in perihelion distance [AU].
Q_NEO = 1.3

#: MPC data sources.
URL_ELSEWHERE = "https://www.minorplanetcenter.net/iau/NEO/elsewhere.csv"
URL_ELSEWHERE_README = "https://www.minorplanetcenter.net/iau/NEO/elsewhere.README"
URL_PREVDES = "https://www.minorplanetcenter.net/iau/NEO/ToConfirm_PrevDes.html"
URL_OBSCODES = "https://www.minorplanetcenter.net/iau/lists/ObsCodesF.html"
URL_NEA = "https://minorplanetcenter.net/Extended_Files/nea_extended.json.gz"
URL_GETOBS = "https://data.minorplanetcenter.net/api/get-obs"
URL_SBDB = "https://ssd-api.jpl.nasa.gov/sbdb.api"

#: Sent on every request.  Plain, identifying, no address of the user's.
USER_AGENT = (
    "priora-validation/0.1 (NEOCP pilot, DESIGN 4.3; "
    "+https://github.com/devanshi-s04/priora)"
)

#: Minimum seconds between two requests to the same host (politeness).
RATE_LIMIT_S = 0.5

#: Band -> Johnson V offsets.  **Verbatim** from digest2 2.8.2's own table,
#: ``digest2/observation.py:_BAND_CORRECTIONS`` ("same as common.c
#: updateMagnitude"), so the two codes standardise photometry identically.
#: digest2 applies the offset only when the reported magnitude is > 0 and uses
#: :data:`BAND_TO_V_DEFAULT` for any band not listed; :func:`band_to_v`
#: reproduces both rules.
BAND_TO_V: dict[str, float] = {
    "V": 0.0,
    "B": -0.8,
    "U": -1.3,
    "g": -0.28,
    "r": 0.23,
    "R": 0.4,
    "C": 0.4,
    "W": 0.4,
    "i": 0.39,
    "z": 0.37,
    "I": 0.8,
    "J": 1.2,
    "w": -0.16,
    "y": 0.36,
    "L": 0.2,
    "H": 1.4,
    "K": 1.7,
    "Y": 0.7,
    "G": 0.24,
    "v": 0.0,
    "c": -0.05,
    "o": 0.33,
    "u": 2.5,
}

#: digest2's fallback offset for an unrecognised band (``.get(band, -0.8)``).
BAND_TO_V_DEFAULT = -0.8

#: **Real-data finding.** The MPC observations API's ADES dataframe reports the
#: photometric band as a *two-character* code for most modern surveys — `Pw`,
#: `Ao`, `Lr`, `Vj` — where every other input format digest2 can read (MPC
#: 80-column, ADES PSV/XML) carries a single character.  digest2's table is
#: keyed on the single character, so a naive lookup sends 2-char codes to the
#: -0.8 "unknown band" fallback: a 0.6-1.0 mag photometric error on a third of
#: this sample, silently.
#:
#: The mapping below was **verified against the MPC itself**: requesting
#: ``output_format: ["ADES_DF", "OBS80"]`` for the same object returns both
#: forms, and the 80-column band character equals the second character of the
#: survey codes (`Pw`->`w`: 105 of each for 2017 UQ124; `Ac`->`c` 38/38 and
#: `Ao`->`o` 32/32 for 2006 UW157; `Lr`->`r`, `Lz`->`z` 6/6 each).  Note the
#: order is **not** uniform: the survey codes are (system, band) with an
#: uppercase system prefix, while the Johnson/Cousins codes are (band, system)
#: with a lowercase system suffix -- `Vj` is Johnson V, and mechanically taking
#: the second character would read it as J-band (+1.2 mag).  Hence an explicit
#: table rather than a slicing rule.
#:
#: ``UNK`` is deliberately absent: it is a *present* band value that digest2
#: does not know, so it must keep taking the -0.8 fallback, exactly as digest2
#: would on the same ADES input.
MPC_BAND_ALIASES: dict[str, str] = {
    # Pan-STARRS (F51, F52, and the O18/W* mirrors): grizy + the broad w
    "Pg": "g", "Pr": "r", "Pi": "i", "Pz": "z", "Py": "y", "Pw": "w",
    # ATLAS (T05, T08, M22, W68): cyan and orange
    "Ac": "c", "Ao": "o",
    # Rubin / LSST (X05, Simonyi Survey Telescope): ugrizy
    "Lu": "u", "Lg": "g", "Lr": "r", "Li": "i", "Lz": "z", "Ly": "y",
    # SDSS-like ugriz reported with a 'G' system prefix
    "Gu": "u", "Gg": "g", "Gr": "r", "Gi": "i", "Gz": "z",
    # Johnson (j) / Cousins (c) systems: band first, system second
    "Uj": "U", "Bj": "B", "Vj": "V", "Rj": "R", "Ij": "I",
    "Rc": "R", "Ic": "I",
}

#: Equatorial Earth radius used by the MPC to normalise its parallax constants.
#: Same value as ``priora.attributable.MPC_EARTH_RADIUS_M``; repeated here so
#: the obscode parser has no import-time dependency on the package.
MPC_EARTH_RADIUS_M = 6378137.0

#: Comet-style designations, excluded from the labelled sample (DESIGN §5: the
#: digest2 model has no cometary component).  Matches "C/2019 E3", "141P",
#: "304P/…", "P/2021 A1", "1I/…".
_COMET_RE = re.compile(r"^(?:[CPDXAI]/|\d+[PDICX]\b|\d+[PDICX]/)")

#: Night-splitting gap and maximum tracklet span.
DEFAULT_GAP_HOURS = 3.0
DEFAULT_MAX_SPAN_HOURS = 2.0

#: How far before the NEOCP-exit timestamp the discovery night may lie.
DEFAULT_WINDOW_DAYS = 90.0

#: digest2's NEOCP posting threshold (its own policy number).
D2_NEOCP_THRESHOLD = 65.0

SUBSETS = ("all", "clean", "broken")


# --------------------------------------------------------------------------
# Polite, cached HTTP
# --------------------------------------------------------------------------

_last_request: dict[str, float] = {}


def _throttle(url: str, min_interval: float = RATE_LIMIT_S) -> None:
    host = urllib.parse.urlsplit(url).netloc
    now = time.time()
    prev = _last_request.get(host)
    if prev is not None:
        wait = min_interval - (now - prev)
        if wait > 0:
            time.sleep(wait)
    _last_request[host] = time.time()


class FetchError(RuntimeError):
    """A cached download could not be produced (network or HTTP error)."""


def http_get(
    url: str,
    dest: str,
    *,
    json_body: dict | None = None,
    params: dict | None = None,
    force: bool = False,
    timeout: float = 180.0,
    gzip_out: bool = False,
    retries: int = 3,
) -> str:
    """GET ``url`` into ``dest``, once.  Returns ``dest``.

    A file that already exists and is non-empty is *never* re-fetched unless
    ``force`` — that is what makes ``build``/``score``/``analyze`` offline.

    ``json_body`` is sent as a JSON request body **on a GET**, which is how the
    MPC's ``data.minorplanetcenter.net`` API is designed (it answers 405 to
    POST and refuses a query string with "use application/json").
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not force:
        return dest
    full = url if not params else url + "?" + urllib.parse.urlencode(params)
    data = json.dumps(json_body).encode() if json_body is not None else None
    headers = {"User-Agent": USER_AGENT}
    if data is not None:
        headers["Content-Type"] = "application/json"

    last: Exception | None = None
    for attempt in range(retries):
        _throttle(full)
        req = urllib.request.Request(full, data=data, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
            break
        except urllib.error.HTTPError as exc:
            # A 4xx is the server telling us the request itself is wrong
            # (SBDB answers 400 to an unresolvable designation).  Retrying it
            # only wastes the service's time and ours.
            if 400 <= exc.code < 500:
                raise FetchError(f"GET {full} -> HTTP {exc.code}") from None
            last = exc
            time.sleep(1.5 * (attempt + 1))
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    else:
        raise FetchError(f"GET {full} failed after {retries} tries: {last}")

    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    tmp = dest + ".part"
    opener = gzip.open if gzip_out else open
    with opener(tmp, "wb") as fh:
        fh.write(payload)
    os.replace(tmp, dest)
    return dest


def _read_maybe_gzip(path: str) -> bytes:
    with open(path, "rb") as fh:
        head = fh.read(2)
    if head == b"\x1f\x8b":
        with gzip.open(path, "rb") as fh:
            return fh.read()
    with open(path, "rb") as fh:
        return fh.read()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Photometry: band -> V, digest2's table
# --------------------------------------------------------------------------


def normalize_band(band: str | None) -> str:
    """MPC ADES-dataframe band code -> the single character digest2 knows.

    Missing/empty becomes ``"V"``, which is exactly what digest2's own ADES
    parser does (``row.get("band", "V") or "V"``).  Two-character survey codes
    are resolved through :data:`MPC_BAND_ALIASES`; anything else is passed
    through unchanged so that a genuinely unknown band still takes digest2's
    -0.8 fallback rather than being silently invented.
    """
    if band is None:
        return "V"
    key = str(band).strip()
    if key == "":
        return "V"
    return MPC_BAND_ALIASES.get(key, key)


def band_to_v(band: str | None, mag: float) -> float:
    """Convert a reported magnitude in ``band`` to Johnson V, digest2's way.

    Reproduces ``digest2/observation.py:_update_magnitude`` exactly: the offset
    from :data:`BAND_TO_V` (default :data:`BAND_TO_V_DEFAULT` for an unknown
    band) is added only when ``mag > 0``; a missing/zero magnitude passes
    through untouched, which is digest2's "no photometry" sentinel.  The band
    is put through :func:`normalize_band` first — see
    :data:`MPC_BAND_ALIASES` for why that step is not optional on real MPC
    ADES output.
    """
    m = float(mag)
    if not np.isfinite(m) or m <= 0.0:
        return m
    return m + BAND_TO_V.get(normalize_band(band), BAND_TO_V_DEFAULT)


# --------------------------------------------------------------------------
# Observatory codes: MPC parallax constants -> WGS84 geodetic
# --------------------------------------------------------------------------

#: One MPC obscode row that yielded parallax constants.
_OBSCODE_RE = re.compile(
    r"^(\S{3})\s+(-?\d+(?:\.\d*)?)\s+([-+]?\d+(?:\.\d*)?)\s+"
    r"([-+]?\d+(?:\.\d*)?)\s*(.*)$"
)


def parse_obscodes(path: str) -> dict[str, dict]:
    """Parse the MPC "List Of Observatory Codes" page.

    Returns ``{code: {"lon_deg", "rho_cos_phi", "rho_sin_phi", "name"}}`` for
    every code that publishes parallax constants.  Space-based and roving
    codes (Gaia, TESS, "Roving Observer", ...) leave those columns blank and
    are **omitted**: they define no fixed ITRS site, and priora's observer
    model has no spacecraft ephemeris.
    """
    raw = _read_maybe_gzip(path).decode("utf-8", errors="replace")
    lo = raw.find("<pre>")
    hi = raw.find("</pre>", lo + 1)
    body = raw[lo + 5 : hi] if lo >= 0 and hi > lo else raw
    body = re.sub(r"<[^>]+>", "", body)
    import html as _html

    body = _html.unescape(body)

    out: dict[str, dict] = {}
    for line in body.split("\n"):
        if not line.strip() or line.lstrip().startswith("Code"):
            continue
        m = _OBSCODE_RE.match(line)
        if not m:
            continue
        code, lon, rcos, rsin, name = m.groups()
        out[code] = {
            "lon_deg": float(lon),
            "rho_cos_phi": float(rcos),
            "rho_sin_phi": float(rsin),
            "name": name.strip(),
        }
    return out


def obscode_geodetic(row: dict) -> tuple[float, float, float]:
    """MPC parallax constants -> WGS84 ``(lon_deg, lat_deg, height_m)``.

    The MPC publishes each site as geocentric cylindrical coordinates in units
    of the equatorial Earth radius: ``x = rho cos(phi') cos(lon)``,
    ``y = rho cos(phi') sin(lon)``, ``z = rho sin(phi')``.  Those define an
    ITRS position with no ellipsoid involved, so we build the position and let
    astropy invert to geodetic — which is what
    :func:`priora.attributable.observatory_location` consumes for any code
    outside its 14-entry built-in table.  The round trip is exact to
    sub-millimetre; ``tests/test_neocp_pilot.py`` checks it against every entry
    of that built-in table.
    """
    import astropy.units as u
    from astropy.coordinates import EarthLocation

    lam = np.radians(row["lon_deg"])
    rc, rs = float(row["rho_cos_phi"]), float(row["rho_sin_phi"])
    xyz = np.array([rc * np.cos(lam), rc * np.sin(lam), rs]) * MPC_EARTH_RADIUS_M
    loc = EarthLocation.from_geocentric(*xyz, unit=u.m)
    geo = loc.to_geodetic(ellipsoid="WGS84")
    return (
        float(geo.lon.to_value(u.deg)),
        float(geo.lat.to_value(u.deg)),
        float(geo.height.to_value(u.m)),
    )


# --------------------------------------------------------------------------
# The labelled sample
# --------------------------------------------------------------------------


def designation_key(desig: str) -> str:
    """Canonical form of a designation, for cross-source matching.

    Collapses whitespace and **strips parentheses**.  ``elsewhere.csv`` writes
    numbered objects as ``"(557271)"``; the MPC NEA list writes the same object
    as ``Number = "(557271)"`` but exposes it stripped, and JPL's SBDB rejects
    the parenthesised form outright (HTTP 400).  One canonical form fixes all
    three, and it also routes parenthesised comets ``"(108P)"`` into
    :func:`is_comet_designation`.
    """
    return re.sub(r"\s+", " ", re.sub(r"[()]", "", str(desig)).strip())


def is_comet_designation(desig: str) -> bool:
    """True for cometary/interstellar designations (excluded; DESIGN §5)."""
    return bool(_COMET_RE.match(designation_key(desig)))


@dataclass
class Candidate:
    """One row of the MPC's NEOCP archive (``elsewhere.csv``)."""

    trksub: str
    original: str
    reference: str
    disposition: str
    designation: str
    discovery_obscode: str
    timestamp: str

    @property
    def key(self) -> str:
        return designation_key(self.designation)


def load_elsewhere(path: str) -> list[Candidate]:
    """Parse ``elsewhere.csv``.

    The published header names 7 columns; the second holds either a
    disposition phrase ("was not confirmed") or the object's original
    designation, depending on the row.  We keep it verbatim as
    ``Candidate.original`` and never key on it.
    """
    text = _read_maybe_gzip(path).decode("utf-8", errors="replace")
    rows = list(csv.reader(text.splitlines()))
    out: list[Candidate] = []
    for row in rows[1:]:
        if len(row) < 7:
            continue
        f = [x.strip() for x in row[:7]]
        out.append(Candidate(*f))
    return out


def load_nea_designations(path: str) -> tuple[set[str], dict[str, float]]:
    """MPC ``nea_extended.json.gz`` -> (designation set, designation -> q).

    Every alias the MPC lists (principal designation, other designations, and
    the packed-free number) maps to the same object, so membership can be
    tested against whatever designation ``elsewhere.csv`` recorded.
    """
    data = json.loads(_read_maybe_gzip(path).decode("utf-8", errors="replace"))
    names: set[str] = set()
    q: dict[str, float] = {}
    for obj in data:
        aliases = [obj.get("Principal_desig")]
        aliases += list(obj.get("Other_desigs") or [])
        num = obj.get("Number")
        if num:
            aliases.append(str(num).strip("()"))
        peri = obj.get("Perihelion_dist")
        for alias in aliases:
            if not alias:
                continue
            key = designation_key(alias)
            names.add(key)
            if peri is not None:
                q[key] = float(peri)
    return names, q


def eligible_pool(
    candidates: list[Candidate], obscodes: dict[str, dict]
) -> tuple[dict[str, Candidate], dict[str, int]]:
    """Filter ``elsewhere.csv`` to objects that can carry a label AND a fit.

    Returns ``({designation: earliest Candidate}, rejection counts)``.  The
    counts are the honest accounting of the selection function and go straight
    into the report.
    """
    counts = {
        "rows": len(candidates),
        "no_designation": 0,
        "not_published": 0,
        "comet": 0,
        "no_obscode": 0,
        "obscode_not_ground_based": 0,
        "duplicate_designation": 0,
    }
    pool: dict[str, Candidate] = {}
    for cand in candidates:
        if cand.disposition != "published":
            counts["not_published"] += 1
            continue
        if cand.designation in ("", "None"):
            counts["no_designation"] += 1
            continue
        if is_comet_designation(cand.designation):
            counts["comet"] += 1
            continue
        code = cand.discovery_obscode
        if code in ("", "None"):
            counts["no_obscode"] += 1
            continue
        if code not in obscodes:
            counts["obscode_not_ground_based"] += 1
            continue
        key = cand.key
        prev = pool.get(key)
        if prev is None:
            pool[key] = cand
        else:
            counts["duplicate_designation"] += 1
            if cand.timestamp < prev.timestamp:
                pool[key] = cand
    counts["eligible"] = len(pool)
    return pool, counts


def select_sample(
    pool: dict[str, Candidate], n: int, seed: int
) -> list[Candidate]:
    """Uniform random sample of ``n`` objects, label-blind (see module docstring).

    Deliberately *not* stratified by label: the observed NEO fraction of the
    sample is then the real NEOCP-conditioned base rate, which is the only
    thing that makes the calibration question answerable.  Sorted keys plus a
    seeded Generator make the draw reproducible.
    """
    keys = sorted(pool)
    rng = np.random.default_rng(seed)
    if n >= len(keys):
        chosen = keys
    else:
        idx = rng.choice(len(keys), size=int(n), replace=False)
        chosen = [keys[i] for i in sorted(idx)]
    return [pool[k] for k in chosen]


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------


def _safe_name(desig: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", designation_key(desig))


def _archive_range(candidates: list[Candidate]) -> list[str]:
    """First and last NEOCP-exit date in the archive, as ``YYYY-MM-DD``."""
    from astropy.time import Time

    jd = np.array([_timestamp_to_jd_utc(c.timestamp) for c in candidates])
    jd = jd[np.isfinite(jd)]
    if jd.size == 0:
        return ["", ""]
    return [
        str(Time(float(jd.min()), format="jd", scale="utc").isot)[:10],
        str(Time(float(jd.max()), format="jd", scale="utc").isot)[:10],
    ]


def fetch_static(raw_dir: str, force: bool = False) -> dict[str, str]:
    """Download (once) the four bulk sources.  Returns their paths."""
    os.makedirs(raw_dir, exist_ok=True)
    paths = {
        "elsewhere": os.path.join(raw_dir, "elsewhere.csv"),
        "elsewhere_readme": os.path.join(raw_dir, "elsewhere.README"),
        "prevdes": os.path.join(raw_dir, "ToConfirm_PrevDes.html"),
        "obscodes": os.path.join(raw_dir, "ObsCodesF.html"),
        "nea": os.path.join(raw_dir, "nea_extended.json.gz"),
    }
    http_get(URL_ELSEWHERE, paths["elsewhere"], force=force)
    http_get(URL_ELSEWHERE_README, paths["elsewhere_readme"], force=force)
    http_get(URL_PREVDES, paths["prevdes"], force=force)
    http_get(URL_OBSCODES, paths["obscodes"], force=force)
    http_get(URL_NEA, paths["nea"], force=force)
    return paths


def fetch_object(desig: str, raw_dir: str, force: bool = False) -> list[str]:
    """Cache one object's observations (MPC) and orbit (JPL SBDB).

    One request each; the MPC API refuses more than one designation per call
    ("Too many designations requested. Limit is 1"), which is also why this
    stage is rate-limited rather than batched.  Returns a list of error
    strings (empty on full success): the two services fail independently and a
    missing orbit should not cost us the cached astrometry, or vice versa.
    """
    key = designation_key(desig)
    name = _safe_name(desig) + ".json.gz"
    errors: list[str] = []
    for url, sub, kw in (
        (
            URL_GETOBS,
            "obs",
            {"json_body": {"desigs": [key], "output_format": ["ADES_DF"]}},
        ),
        (URL_SBDB, "sbdb", {"params": {"sstr": key, "phys-par": "0"}}),
    ):
        try:
            http_get(
                url,
                os.path.join(raw_dir, sub, name),
                force=force,
                gzip_out=True,
                **kw,
            )
        except FetchError as exc:
            errors.append(f"{sub}: {exc}")
    return errors


def fetch(
    out_dir: str = DEFAULT_OUT,
    n: int = 500,
    seed: int = 42,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Stage 1: cache the archive, choose the sample, cache per-object data."""
    raw_dir = os.path.join(out_dir, "raw")
    paths = fetch_static(raw_dir, force=force)
    obscodes = parse_obscodes(paths["obscodes"])
    candidates = load_elsewhere(paths["elsewhere"])
    nea_names, nea_q = load_nea_designations(paths["nea"])
    pool, counts = eligible_pool(candidates, obscodes)

    pool_neo = sum(1 for k in pool if k in nea_names)
    sample = select_sample(pool, n, seed)

    manifest = {
        "sources": {
            "elsewhere_csv": URL_ELSEWHERE,
            "prevdes_html": URL_PREVDES,
            "obscodes_html": URL_OBSCODES,
            "nea_extended": URL_NEA,
            "mpc_get_obs": URL_GETOBS,
            "jpl_sbdb": URL_SBDB,
        },
        "sha256": {k: _sha256(v) for k, v in paths.items()},
        "n_obscodes_ground_based": len(obscodes),
        "selection_counts": counts,
        "pool_neo_by_mpc_nea_list": pool_neo,
        "pool_non_neo_by_mpc_nea_list": len(pool) - pool_neo,
        # elsewhere.csv timestamps are not zero-padded ("2019-3-21T..."), so a
        # lexical min/max lies; order on the parsed JD instead.
        "archive_timestamp_range": _archive_range(candidates),
        "n_requested": int(n),
        "seed": int(seed),
        "sample": [asdict(c) for c in sample],
        "user_agent": USER_AGENT,
    }

    failures = []
    for i, cand in enumerate(sample, 1):
        errors = fetch_object(cand.designation, raw_dir, force=force)
        if errors:
            failures.append({"designation": cand.key, "errors": errors})
        if verbose and (i % 25 == 0 or i == len(sample)):
            print(
                f"[fetch] {i}/{len(sample)} objects cached "
                f"({len(failures)} partial failures)",
                flush=True,
            )
    manifest["fetch_failures"] = failures

    path = os.path.join(out_dir, "sample.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=1, default=_json_default)
    if verbose:
        print(
            f"[fetch] pool={len(pool)} eligible, sampled {len(sample)}, "
            f"{len(failures)} fetch failures -> {path}"
        )
    return manifest


# --------------------------------------------------------------------------
# build: observations -> attributables
# --------------------------------------------------------------------------


def cluster_nights(t_days: np.ndarray, gap_hours: float = DEFAULT_GAP_HOURS
                   ) -> list[np.ndarray]:
    """Split time-sorted epochs into groups separated by more than ``gap_hours``.

    A "night" for tracklet purposes: real discovery tracklets are minutes to
    an hour or two long, and consecutive nights are ~a day apart, so any gap
    of a few hours is a night boundary.  Returns index arrays into the input.
    """
    t = np.asarray(t_days, dtype=float)
    if t.size == 0:
        return []
    order = np.argsort(t, kind="mergesort")
    ts = t[order]
    breaks = np.nonzero(np.diff(ts) > gap_hours / 24.0)[0] + 1
    return [order[sl] for sl in np.split(np.arange(t.size), breaks)]


def fit_attributable(
    t_days: np.ndarray, ra_deg: np.ndarray, dec_deg: np.ndarray
) -> dict:
    """Least-squares straight-line fit of alpha(t), delta(t); epoch = t[0].

    Ordinary (unweighted) least squares in each coordinate independently — the
    ADES per-observation uncertainties are frequently absent in this archive
    and weighting a 3-point fit by sporadic sigmas would make the estimator
    depend on which observers happened to report them.

    RA is unwrapped about the first detection before fitting, so a tracklet
    straddling alpha = 0 fits correctly.  The reported epoch is the **first**
    detection's, matching the WP6/WP7 convention and digest2's ``obsPair[0]``
    anchor.

    Returns radians and rad/day, plus the fit residual RMS in arcsec (the
    honest per-tracklet quality flag) and the arc length in hours.  A
    zero-length arc (all epochs equal) yields NaN rates.
    """
    t = np.asarray(t_days, dtype=float)
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    if t.size < 2:
        raise ValueError("need at least 2 detections to fit an attributable")
    order = np.argsort(t, kind="mergesort")
    t, ra, dec = t[order], ra[order], dec[order]

    t0 = t[0]
    dt = t - t0
    ra0 = ra[0]
    ra_un = ra0 + ((ra - ra0 + 180.0) % 360.0 - 180.0)

    design = np.column_stack([np.ones_like(dt), dt])
    if not np.any(dt != 0.0):
        return {
            "epoch_jd_utc": float(t0),
            "ra": float(np.radians(ra0 % 360.0)),
            "dec": float(np.radians(dec[0])),
            "ra_rate": float("nan"),
            "dec_rate": float("nan"),
            "resid_rms_arcsec": float("nan"),
            "arc_hours": 0.0,
            "n_det": int(t.size),
        }

    (a0, a1), *_ = np.linalg.lstsq(design, ra_un, rcond=None)
    (d0, d1), *_ = np.linalg.lstsq(design, dec, rcond=None)

    res_ra = (ra_un - (a0 + a1 * dt)) * np.cos(np.radians(d0))
    res_dec = dec - (d0 + d1 * dt)
    n_free = max(t.size - 2, 1)
    rms = np.sqrt((np.sum(res_ra**2) + np.sum(res_dec**2)) / (2 * n_free)) * 3600.0

    return {
        "epoch_jd_utc": float(t0),
        "ra": float(np.radians(a0 % 360.0)),
        "dec": float(np.radians(d0)),
        "ra_rate": float(np.radians(a1)),
        "dec_rate": float(np.radians(d1)),
        "resid_rms_arcsec": float(rms),
        "arc_hours": float((t[-1] - t[0]) * 24.0),
        "n_det": int(t.size),
    }


@dataclass
class Tracklet:
    """One real discovery tracklet, ready to score."""

    designation: str
    trksub: str
    obscode: str
    epoch_jd_utc: float
    ra: float
    dec: float
    ra_rate: float
    dec_rate: float
    v_mag: float
    n_det: int
    arc_hours: float
    resid_rms_arcsec: float
    # provenance / diagnostics
    bands: list[str] = field(default_factory=list)
    mags_v: list[float] = field(default_factory=list)
    lon_deg: float = 0.0
    lat_deg: float = 0.0
    height_m: float = 0.0
    is_neo: bool = False
    q_au: float = float("nan")
    q_sbdb: float = float("nan")
    q_source: str = ""
    e: float = float("nan")
    incl_deg: float = float("nan")
    a_au: float = float("nan")
    H_sbdb: float = float("nan")
    orbit_class: str = ""
    n_obs_used: int = 0
    label_source_agree: bool = True
    days_before_neocp_exit: float = float("nan")
    precovery_exists: bool = False
    window_fallback: bool = False
    truncated_span: bool = False
    obs_times_jd: list[float] = field(default_factory=list)
    obs_ra_deg: list[float] = field(default_factory=list)
    obs_dec_deg: list[float] = field(default_factory=list)
    obs_mag: list[float] = field(default_factory=list)


def _iso_to_jd_utc(iso: str) -> float:
    """ADES ``obsTime`` (ISO-8601 UTC, trailing Z) -> Julian Date (UTC)."""
    from astropy.time import Time

    return float(Time(iso.strip().rstrip("Z"), format="isot", scale="utc").jd)


_TS_RE = re.compile(
    r"(\d{4})-(\d{1,2})-(\d{1,2})[T ](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?"
)


def _timestamp_to_jd_utc(ts: str) -> float:
    """``elsewhere.csv`` timestamp -> JD (UTC).

    Those fields are *not* zero-padded ("2019-3-21T19:57:17"), so astropy's
    ``isot`` parser rejects them; parse the parts and rebuild.
    """
    from datetime import datetime

    from astropy.time import Time

    m = _TS_RE.match(str(ts).strip())
    if not m:
        return float("nan")
    y, mo, d, hh, mm, ss = (int(g) if g else 0 for g in m.groups())
    return float(Time(datetime(y, mo, d, hh, mm, ss), scale="utc").jd)


def _load_obs(path: str) -> list[dict]:
    """Read a cached ``get-obs`` reply; ``[]`` if it carries no ADES rows."""
    payload = json.loads(_read_maybe_gzip(path).decode("utf-8", errors="replace"))
    if isinstance(payload, dict):  # {"error": ..., "message": ...}
        return []
    if not payload or not isinstance(payload[0], dict):
        return []
    rows = payload[0].get("ADES_DF")
    return list(rows) if rows else []


def _load_orbit(path: str) -> dict:
    """Read a cached SBDB reply -> {q, e, i, a, H, class, n_obs_used, kind}."""
    payload = json.loads(_read_maybe_gzip(path).decode("utf-8", errors="replace"))
    orbit = payload.get("orbit") if isinstance(payload, dict) else None
    if not orbit:
        return {}
    els = {e["name"]: e.get("value") for e in orbit.get("elements", [])}

    def num(key):
        try:
            return float(els[key])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    obj = payload.get("object") or {}
    return {
        "q": num("q"),
        "e": num("e"),
        "i": num("i"),
        "a": num("a"),
        "H": num("H"),
        "class": (obj.get("orbit_class") or {}).get("code", ""),
        "kind": obj.get("kind", ""),
        "neo_flag": bool(obj.get("neo")),
        "n_obs_used": int(orbit.get("n_obs_used") or 0),
    }


def build_tracklets(
    out_dir: str = DEFAULT_OUT,
    gap_hours: float = DEFAULT_GAP_HOURS,
    max_span_hours: float = DEFAULT_MAX_SPAN_HOURS,
    window_days: float = DEFAULT_WINDOW_DAYS,
    verbose: bool = True,
) -> tuple[list[Tracklet], dict]:
    """Stage 2: cached observations -> attributables + labels, purely offline.

    Every rejection is counted.  Real data breaks things, and the census of
    *how* is a deliverable in its own right.
    """
    raw_dir = os.path.join(out_dir, "raw")
    with open(os.path.join(out_dir, "sample.json")) as fh:
        manifest = json.load(fh)
    obscodes = parse_obscodes(os.path.join(raw_dir, "ObsCodesF.html"))
    nea_names, nea_q = load_nea_designations(
        os.path.join(raw_dir, "nea_extended.json.gz")
    )

    census = {
        "n_sampled": len(manifest["sample"]),
        "obs_file_missing": 0,
        "obs_empty_or_error": 0,
        "no_obs_at_discovery_obscode": 0,
        "discovery_night_single_detection": 0,
        "orbit_file_missing": 0,
        "orbit_unavailable": 0,
        "orbit_q_nonfinite": 0,
        "cometary_by_sbdb_kind": 0,
        "no_usable_magnitude": 0,
        # Photometry bookkeeping (per DETECTION, not per tracklet).
        "band_aliased": 0,
        "band_unknown_to_digest2": 0,
        "fit_failed": 0,
        "obscode_unresolvable": 0,
        "time_parse_failed": 0,
        "built": 0,
        "window_fallback": 0,
        "truncated_span": 0,
        "precovery_exists": 0,
        "label_source_disagree": 0,
    }
    tracklets: list[Tracklet] = []

    for entry in manifest["sample"]:
        cand = Candidate(**{k: entry[k] for k in Candidate.__dataclass_fields__})
        name = _safe_name(cand.designation) + ".json.gz"
        obs_path = os.path.join(raw_dir, "obs", name)
        orb_path = os.path.join(raw_dir, "sbdb", name)
        if not os.path.exists(obs_path):
            census["obs_file_missing"] += 1
            continue
        if not os.path.exists(orb_path):
            census["orbit_file_missing"] += 1
            continue

        rows = _load_obs(obs_path)
        if not rows:
            census["obs_empty_or_error"] += 1
            continue
        orbit = _load_orbit(orb_path)
        if not orbit:
            census["orbit_unavailable"] += 1
            continue
        if str(orbit.get("kind", "")).startswith("c"):
            census["cometary_by_sbdb_kind"] += 1
            continue
        if not np.isfinite(orbit["q"]):
            census["orbit_q_nonfinite"] += 1
            continue

        # -- restrict to the discovery station -----------------------------
        at_station = [
            r
            for r in rows
            if (r.get("stn") or "").strip() == cand.discovery_obscode
        ]
        if len(at_station) < 2:
            census["no_obs_at_discovery_obscode"] += 1
            continue

        try:
            jd_all = np.array([_iso_to_jd_utc(r["obstime"]) for r in at_station])
            jd_any = np.array([_iso_to_jd_utc(r["obstime"]) for r in rows])
        except Exception:
            census["time_parse_failed"] += 1
            continue

        # -- pick the NEOCP-triggering night -------------------------------
        exit_jd = _timestamp_to_jd_utc(cand.timestamp)
        nights = cluster_nights(jd_all, gap_hours)
        nights = [g for g in nights if g.size >= 2]
        if not nights:
            census["discovery_night_single_detection"] += 1
            continue
        window_fallback = False
        chosen = None
        if np.isfinite(exit_jd):
            for g in nights:
                if jd_all[g].min() >= exit_jd - window_days:
                    chosen = g
                    break
        if chosen is None:
            chosen = nights[0]
            window_fallback = np.isfinite(exit_jd)
        if window_fallback:
            census["window_fallback"] += 1

        sel = [at_station[i] for i in chosen]
        jd_sel = jd_all[chosen]
        order = np.argsort(jd_sel, kind="mergesort")
        sel = [sel[i] for i in order]
        jd_sel = jd_sel[order]

        truncated = False
        if jd_sel[-1] - jd_sel[0] > max_span_hours / 24.0:
            keep = jd_sel - jd_sel[0] <= max_span_hours / 24.0
            if keep.sum() < 2:
                keep = np.zeros(jd_sel.size, bool)
                keep[:2] = True
            sel = [s for s, k in zip(sel, keep) if k]
            jd_sel = jd_sel[keep]
            truncated = True
            census["truncated_span"] += 1

        # -- photometry ----------------------------------------------------
        # One entry per detection, NaN where the MPC published no magnitude,
        # so obs_mag stays index-aligned with obs_times_jd (the PSV emitter
        # writes a blank magnitude for a NaN, which is digest2's own "no
        # photometry" sentinel).
        bands: list[str] = []
        mags_v = np.full(len(sel), np.nan)
        for j, r in enumerate(sel):
            raw_mag = r.get("mag")
            if raw_mag in (None, ""):
                continue
            try:
                m = float(raw_mag)
            except (TypeError, ValueError):
                continue
            native = (r.get("band") or "").strip()
            norm = normalize_band(native)
            if native and norm != native:
                census["band_aliased"] += 1
            if norm not in BAND_TO_V:
                census["band_unknown_to_digest2"] += 1
            v = band_to_v(native, m)
            if np.isfinite(v) and v > 0:
                bands.append(native or "V")
                mags_v[j] = v
        if not np.any(np.isfinite(mags_v)):
            census["no_usable_magnitude"] += 1
            continue
        v_mag = float(np.nanmedian(mags_v))

        # -- the fit -------------------------------------------------------
        ra_deg = np.array([float(r["ra"]) for r in sel])
        dec_deg = np.array([float(r["dec"]) for r in sel])
        try:
            fit = fit_attributable(jd_sel, ra_deg, dec_deg)
        except Exception:
            census["fit_failed"] += 1
            continue
        if not np.isfinite(fit["ra_rate"]) or not np.isfinite(fit["dec_rate"]):
            census["fit_failed"] += 1
            continue

        row = obscodes.get(cand.discovery_obscode)
        if row is None:
            census["obscode_unresolvable"] += 1
            continue
        lon, lat, height = obscode_geodetic(row)

        # -- the label -----------------------------------------------------
        # JPL's SBDB returns elements as display-rounded strings (q = "1.3"
        # for a boundary Amor), so an SBDB-only q < 1.3 test mislabels objects
        # sitting exactly on the class boundary.  The MPC's nea_extended file
        # carries q to 7 digits and its membership IS the MPC's q < 1.3
        # definition, so prefer that where the object appears in it, and keep
        # the SBDB value as the cross-check.
        by_nea = cand.key in nea_names
        q_mpc = nea_q.get(cand.key)
        if q_mpc is not None and np.isfinite(q_mpc):
            q_used, q_source = float(q_mpc), "mpc_nea"
        else:
            q_used, q_source = float(orbit["q"]), "sbdb"
        is_neo = bool(q_used < Q_NEO)
        if by_nea != bool(orbit["q"] < Q_NEO):
            census["label_source_disagree"] += 1

        precovery = bool(np.any(jd_any < jd_sel[0] - 1e-9))
        if precovery:
            census["precovery_exists"] += 1

        tracklets.append(
            Tracklet(
                designation=cand.key,
                trksub=cand.trksub,
                obscode=cand.discovery_obscode,
                epoch_jd_utc=fit["epoch_jd_utc"],
                ra=fit["ra"],
                dec=fit["dec"],
                ra_rate=fit["ra_rate"],
                dec_rate=fit["dec_rate"],
                v_mag=v_mag,
                n_det=fit["n_det"],
                arc_hours=fit["arc_hours"],
                resid_rms_arcsec=fit["resid_rms_arcsec"],
                bands=sorted(set(bands)),
                mags_v=[float(m) for m in mags_v[np.isfinite(mags_v)]],
                lon_deg=lon,
                lat_deg=lat,
                height_m=height,
                is_neo=is_neo,
                q_au=q_used,
                q_sbdb=float(orbit["q"]),
                q_source=q_source,
                e=orbit["e"],
                incl_deg=orbit["i"],
                a_au=orbit["a"],
                H_sbdb=orbit["H"],
                orbit_class=orbit["class"],
                n_obs_used=orbit["n_obs_used"],
                label_source_agree=(by_nea == bool(orbit["q"] < Q_NEO)),
                days_before_neocp_exit=(
                    float(exit_jd - fit["epoch_jd_utc"])
                    if np.isfinite(exit_jd)
                    else float("nan")
                ),
                precovery_exists=precovery,
                window_fallback=bool(window_fallback),
                truncated_span=truncated,
                obs_times_jd=[float(x) for x in jd_sel],
                obs_ra_deg=[float(x) for x in ra_deg],
                obs_dec_deg=[float(x) for x in dec_deg],
                obs_mag=[float(m) for m in mags_v],  # NaN-padded, index-aligned
            )
        )
        census["built"] += 1

    census["n_neo"] = int(sum(t.is_neo for t in tracklets))
    census["n_non_neo"] = int(len(tracklets) - census["n_neo"])
    if verbose:
        print(
            f"[build] {census['built']}/{census['n_sampled']} tracklets "
            f"({census['n_neo']} NEO, {census['n_non_neo']} non-NEO)"
        )
    return tracklets, census


def build(out_dir: str = DEFAULT_OUT, verbose: bool = True, **kw) -> dict:
    tracklets, census = build_tracklets(out_dir=out_dir, verbose=verbose, **kw)
    path = os.path.join(out_dir, "tracklets.json")
    with open(path, "w") as fh:
        json.dump(
            {"census": census, "tracklets": [asdict(t) for t in tracklets]},
            fh,
            indent=1,
            default=_json_default,
        )
    if verbose:
        print(f"[build] wrote {path}")
    return census


def load_tracklets(out_dir: str = DEFAULT_OUT) -> tuple[list[Tracklet], dict]:
    with open(os.path.join(out_dir, "tracklets.json")) as fh:
        blob = json.load(fh)
    return [Tracklet(**t) for t in blob["tracklets"]], blob["census"]


# --------------------------------------------------------------------------
# score: priora (both variants) and digest2, same detections
# --------------------------------------------------------------------------

_W: dict = {}


def _init_worker(model: str, variant: str, cfg: dict, clamp: bool = False) -> None:
    """Parse the 45,936-bin model once per worker, not once per tracklet.

    ``clamp`` builds the pair with ``h_top_bin_infinite=True``, i.e. with
    digest2's own [24, inf) top H bin instead of priora's hard truncation at
    H = 25.5 (D-4).  On this real sample that is not a cosmetic control: see
    the report's "did anything break" section.
    """
    from priora.population import Digest2Histogram

    neo, other = Digest2Histogram.neo_other_pair(
        model, variant, h_top_bin_infinite=bool(clamp)
    )
    _W["neo"], _W["other"], _W["cfg"] = neo, other, cfg


def _score_one(task: tuple) -> tuple:
    """Score one real tracklet.  Module level and picklable (multiprocessing)."""
    from astropy.time import Time

    from priora import Attributable, classify

    (idx, ra, dec, ra_rate, dec_rate, jd, v_mag, lon, lat, height) = task
    cfg = _W["cfg"]
    att = Attributable(
        ra=float(ra),
        dec=float(dec),
        ra_rate=float(ra_rate),
        dec_rate=float(dec_rate),
        epoch=Time(float(jd), format="jd", scale="utc"),
        v_mag=float(v_mag),
    )
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = classify(
            att,
            _W["neo"],
            _W["other"],
            lon_deg=float(lon),
            lat_deg=float(lat),
            height_m=float(height),
            rho_min=cfg["rho_min"],
            rho_max=cfg["rho_max"],
            n_rho=cfg["n_rho"],
            n_rhodot=cfg["n_rhodot"],
            G=cfg["G"],
            backend="python",
        )
    w = np.asarray(res.grid.weight, dtype=float)
    return (
        int(idx),
        float(res.p_neo),
        float(res.log_lr),
        float(res.lambda_neo),
        float(res.lambda_other),
        int(res.grid.rho.size),
        float(w.sum()) if w.size else 0.0,
        float(res.measure_fraction_H_above(H_TRUNC)),
        int(res.n_masked_nonfinite),
        int(res.n_masked_det),
        float(time.time() - t0),
    )


SCORE_FIELDS = (
    "index",
    "p_neo",
    "log_lr",
    "lambda_neo",
    "lambda_other",
    "n_nodes",
    "ar_weight",
    "ar_frac_h_above",
    "n_masked_nonfinite",
    "n_masked_det",
    "seconds",
)
_INT_FIELDS = {"index", "n_nodes", "n_masked_nonfinite", "n_masked_det"}


def score_priora(
    tracklets: list[Tracklet],
    variant: str = "All",
    model: str = DEFAULT_MODEL,
    workers: int = 8,
    n_rho: int = 1200,
    n_rhodot: int = 50,
    rho_min: float = 0.01,
    rho_max: float = 100.0,
    G: float = 0.15,
    clamp: bool = False,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Score every tracklet with priora v0 in one model variant."""
    import multiprocessing as mp

    cfg = {
        "n_rho": int(n_rho),
        "n_rhodot": int(n_rhodot),
        "rho_min": float(rho_min),
        "rho_max": float(rho_max),
        "G": float(G),
    }
    tasks = [
        (
            i,
            t.ra,
            t.dec,
            t.ra_rate,
            t.dec_rate,
            t.epoch_jd_utc,
            t.v_mag,
            t.lon_deg,
            t.lat_deg,
            t.height_m,
        )
        for i, t in enumerate(tracklets)
    ]
    t0 = time.time()
    if workers <= 1:
        _init_worker(model, variant, cfg, clamp)
        rows = [_score_one(task) for task in tasks]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=min(workers, max(len(tasks), 1)),
            initializer=_init_worker,
            initargs=(model, variant, cfg, clamp),
        ) as pool:
            rows = pool.map(_score_one, tasks, chunksize=1)
    elapsed = time.time() - t0
    rows.sort(key=lambda r: r[0])
    cols = list(zip(*rows)) if rows else [()] * len(SCORE_FIELDS)
    out = {
        name: np.asarray(
            col, dtype=np.int64 if name in _INT_FIELDS else np.float64
        )
        for name, col in zip(SCORE_FIELDS, cols)
    }
    if verbose:
        print(
            f"[priora/{variant}{'+clamp' if clamp else ''}] "
            f"{len(rows)} tracklets in {elapsed:.1f} s "
            f"({1e3 * elapsed / max(len(rows), 1):.0f} ms/tracklet wall, "
            f"{workers} workers)"
        )
    return out


_PSV_HEADER = (
    "permID |provID |trkSub  |mode |stn  |obsTime                     |"
    "ra          |dec         |mag   |band |rmsRA |rmsDec |astCat"
)


def emit_ades_psv(tracklets: list[Tracklet], path: str) -> dict:
    """Write the *real* detections as an ADES PSV file digest2 can parse.

    ``trkSub`` is a synthetic key ``Nnnnnn`` (the tracklet's index) rather than
    the observer's, because digest2 groups by ``trkSub`` and the archive's own
    ids are not guaranteed unique across the sample.  Everything else —
    station, time, RA/Dec, magnitude, band — is exactly what the MPC published,
    and the band is left native so digest2 applies its own conversion (the same
    table :data:`BAND_TO_V` gives priora).
    """
    from astropy.time import Time

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n_obs = 0
    with open(path, "w") as fh:
        fh.write("# priora WP8 NEOCP pilot: real MPC discovery-tracklet astrometry\n")
        fh.write(_PSV_HEADER + "\n")
        for i, t in enumerate(tracklets):
            # Native bands are unknown per-detection after the median, so the
            # PSV carries the V-converted magnitude with band 'V': that makes
            # digest2's own conversion a no-op on an already-converted value,
            # i.e. both codes consume the identical V.
            iso = np.atleast_1d(
                Time(np.asarray(t.obs_times_jd), format="jd", scale="utc").isot
            )
            for k in range(len(t.obs_times_jd)):
                mag = t.obs_mag[k]
                mag_field = "%6.2f" % mag if np.isfinite(mag) else " " * 6
                fh.write(
                    "       |       |%-8s|CCD  |%-4s |%-27s |%12.8f |%+12.8f "
                    "|%s |V    |      |       |Gaia3\n"
                    % (
                        f"N{i:05d}",
                        t.obscode,
                        str(iso[k]) + "Z",
                        t.obs_ra_deg[k],
                        t.obs_dec_deg[k],
                        mag_field,
                    )
                )
                n_obs += 1
    return {
        "path": os.path.abspath(path),
        "format": "ades_psv",
        "n_tracklets": len(tracklets),
        "n_observations": n_obs,
        "band": "V (pre-converted with digest2's own offsets)",
    }


def score_digest2(
    tracklets: list[Tracklet],
    obs_path: str,
    model: str = DEFAULT_MODEL,
    workers: int = 8,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run the local digest2 2.8.2 on the same PSV file."""
    from validation.digest2_comparison import digest2_available

    ok, msg = digest2_available()
    if not ok:
        raise RuntimeError(msg)
    from digest2 import Digest2
    from digest2.model import ensure_obscodes
    from digest2.observation import parse_ades_psv

    t0 = time.time()
    parsed = parse_ades_psv(obs_path)
    n_pairs = sum(1 for v in parsed.values() if len(v) >= 2)
    t_parse = time.time() - t0

    t0 = time.time()
    with Digest2(
        model_path=model, obscodes_path=ensure_obscodes(), repeatable=True
    ) as d2:
        results = d2.classify_file(obs_path, max_workers=workers)
    t_score = time.time() - t0

    n = len(tracklets)
    cols = {
        "raw_neo": np.full(n, np.nan),
        "noid_neo": np.full(n, np.nan),
        "rms": np.full(n, np.nan),
    }
    matched = 0
    for r in results:
        m = re.match(r"^N(\d+)$", str(r.designation).strip())
        if not m:
            continue
        k = int(m.group(1))
        if not (0 <= k < n):
            continue
        cols["raw_neo"][k] = float(r.raw.NEO)
        cols["noid_neo"][k] = float(r.noid.NEO)
        cols["rms"][k] = float(r.rms)
        matched += 1
    cols["ok"] = np.isfinite(cols["raw_neo"])
    stats = {
        "n_emitted": n,
        "n_parsed_with_pair": int(n_pairs),
        "n_scored": len(results),
        "n_matched": matched,
        "parse_rate": n_pairs / n if n else float("nan"),
        "score_rate": matched / n if n else float("nan"),
        "parse_seconds": t_parse,
        "score_seconds": t_score,
        "workers": int(workers),
    }
    if verbose:
        print(
            f"[digest2] parsed {n_pairs}/{n}, scored {matched}/{n} "
            f"in {t_score:.1f} s ({workers} threads)"
        )
    return cols, stats


def score(
    out_dir: str = DEFAULT_OUT,
    workers: int = 8,
    model: str = DEFAULT_MODEL,
    n_rho: int = 1200,
    n_rhodot: int = 50,
    skip_digest2: bool = False,
    verbose: bool = True,
) -> dict:
    """Stage 3: score with priora (All + Unk) and digest2, same detections."""
    tracklets, _ = load_tracklets(out_dir)
    if not tracklets:
        raise RuntimeError("no tracklets built; run build first")

    store: dict[str, np.ndarray] = {}
    # Three priora passes: the two D-10 model variants, plus the "All"
    # variant with digest2's [24, inf) top H bin.  The clamped pass is not a
    # curiosity on real data: for close fast NEOs the entire admissible region
    # can imply H > 25.5, at which point both truncated components are zero
    # everywhere and the posterior is 0/0.
    for prefix, variant, clamp in (
        ("all", "All", False),
        ("unk", "Unk", False),
        ("allclamp", "All", True),
    ):
        cols = score_priora(
            tracklets,
            variant=variant,
            model=model,
            workers=workers,
            n_rho=n_rho,
            n_rhodot=n_rhodot,
            clamp=clamp,
            verbose=verbose,
        )
        for k, v in cols.items():
            store[f"priora_{prefix}_{k}"] = v

    d2_stats: dict = {"skipped": True}
    if not skip_digest2:
        psv = os.path.join(out_dir, "tracklets.psv")
        manifest = emit_ades_psv(tracklets, psv)
        d2_cols, d2_stats = score_digest2(
            tracklets, psv, model=model, workers=workers, verbose=verbose
        )
        d2_stats["obs_file"] = manifest
        for k, v in d2_cols.items():
            store[f"d2_{k}"] = np.asarray(v)

    store["is_neo"] = np.array([t.is_neo for t in tracklets], dtype=bool)
    store["q_au"] = np.array([t.q_au for t in tracklets], dtype=float)
    store["v_mag"] = np.array([t.v_mag for t in tracklets], dtype=float)

    path = os.path.join(out_dir, "scores.npz")
    np.savez_compressed(path, **store)
    meta = {
        "model": model,
        "model_sha256": _sha256(model),
        "n_rho": int(n_rho),
        "n_rhodot": int(n_rhodot),
        "workers": int(workers),
        "backend": os.environ.get("PRIORA_BACKEND", "auto"),
        "digest2": d2_stats,
        "n_tracklets": len(tracklets),
    }
    with open(os.path.join(out_dir, "score_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=_json_default)
    if verbose:
        print(f"[score] wrote {path}")
    return meta


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------


def paired_auc_bootstrap(
    s_a: np.ndarray,
    s_b: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray | None = None,
    n_boot: int = 4000,
    seed: int = 7,
    alpha: float = 0.05,
) -> dict:
    """Paired, label-stratified bootstrap for AUC(a), AUC(b) and AUC(a)-AUC(b).

    Same contract and output keys as
    ``validation.digest2_comparison.auc_bootstrap``, and it is *not* reusing
    that function on purpose.

    **Bug found by this pilot (2026-09-02), reported, not patched here.**
    ``digest2_comparison.auc_bootstrap`` builds its resample as::

        take = concatenate((rng.integers(0, pos.size, pos.size),
                            rng.integers(0, neg.size, neg.size)))
        sel  = concatenate((pos, neg))[take]

    The negative half's indices are never offset by ``pos.size``, so they
    address the *positive* block of the concatenation.  Consequences:

    * when ``n_pos >= n_neg`` — which is this sample (305 NEOs vs 194
      non-NEOs, because the NEOCP is already digest2-filtered) — every
      resample is 100% positives, every bootstrap AUC is NaN, the CIs come
      back ``[nan, nan]`` and ``p_delta_le_0`` silently reports ``0.0``
      because ``NaN <= 0`` is ``False``;
    * when ``n_pos < n_neg`` (WP6/WP7's regime, 185 NEOs in 2x10^4) the
      negative half is drawn from a block that is ``n_pos / n_neg`` ~ 0.9%
      positives, so the published intervals come from a slightly wrong
      resampling distribution — small there, but wrong.

    Fixing that file is outside this task's ownership, so this module carries
    its own correct implementation and the finding is written up in the report.
    """
    from validation.digest2_comparison import auc_fast

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
        "n_dropped_nonfinite": int((~good).sum()),
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
        # Resample the two label strata *independently*, each from its own
        # index array -- no concatenation offset to get wrong.
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
    out["auc_a_ci"] = [float(np.percentile(da, lo)), float(np.percentile(da, hi))]
    out["auc_b_ci"] = [float(np.percentile(db, lo)), float(np.percentile(db, hi))]
    out["delta_ci"] = [float(np.percentile(d, lo)), float(np.percentile(d, hi))]
    out["delta_boot_mean"] = float(d.mean())
    out["delta_boot_sd"] = float(d.std(ddof=1))
    out["p_delta_le_0"] = float(np.mean(d <= 0.0))
    return out


def reliability_quantile(
    p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, n_bins: int = 8
) -> list[dict]:
    """Equal-count (quantile) reliability bins.

    ``digest2_comparison.reliability_table`` cuts bins on equal *predicted-
    positive mass*, which is the right choice at WP6's ~1% NEO base rate
    (equal-count bins there hold far less than one expected NEO each and can
    neither pass nor fail).  It is the wrong choice here: this sample's base
    rate is ~61% and the posteriors pile up against p = 1, so equal-mass
    cutting collapses to two or three usable bins and the reliability curve
    cannot be read.  Equal-count bins carry ~n/n_bins objects and tens of
    expected NEOs each at this base rate, which is what resolves the curve.
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
    pieces = [b for b in np.array_split(np.arange(p.size), n_bins) if b.size]

    rows = []
    for sl in pieces:
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
                # Binomial 1-sigma on the observed fraction, for the plot.
                "obs_frac_err": float(
                    np.sqrt(max(obs_pos, 0.0) * (1.0 - obs_pos / max(ww.sum(), 1e-12)))
                    / max(ww.sum(), 1e-12)
                ),
                "z": (obs_pos - exp_pos) / np.sqrt(var) if var > 0 else float("nan"),
            }
        )
    return rows


def calibration_quantile(
    p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, n_bins: int = 8
) -> dict:
    """Global calibration summary with :func:`reliability_quantile` bins."""
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
    rows = reliability_quantile(p, y, w, n_bins=n_bins)
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


def _q_bins(q: np.ndarray) -> dict[str, np.ndarray]:
    q = np.asarray(q, float)
    return {
        "NEO (q<1.3)": q < 1.3,
        "Mars-crosser (1.3-1.67)": (q >= 1.3) & (q < 1.667),
        "inner belt (1.67-2.5)": (q >= 1.667) & (q < 2.5),
        "outer belt+ (>=2.5)": q >= 2.5,
    }


def analyze(out_dir: str = DEFAULT_OUT, n_boot: int = 4000, seed: int = 11,
            verbose: bool = True) -> dict:
    """Stage 4: distributions, ROC, calibration, disagreements, plots."""
    from validation.digest2_comparison import (
        auc_fast,
        operating_points,
        percentile_rank,
        subset_masks,
    )

    tracklets, census = load_tracklets(out_dir)
    with open(os.path.join(out_dir, "sample.json")) as fh:
        manifest = json.load(fh)
    with open(os.path.join(out_dir, "score_meta.json")) as fh:
        meta = json.load(fh)
    npz = np.load(os.path.join(out_dir, "scores.npz"))
    store = {k: npz[k] for k in npz.files}

    y = store["is_neo"].astype(bool)
    w = np.ones(y.size)
    frac_h = store["priora_all_ar_frac_h_above"]
    masks = subset_masks(frac_h)

    has_d2 = "d2_raw_neo" in store
    d2_raw = store.get("d2_raw_neo", np.full(y.size, np.nan))
    d2_noid = store.get("d2_noid_neo", np.full(y.size, np.nan))

    # D-19: a triage-policy assignment (p = 1.0 over a 0/0 whose whole region
    # sits at/above the H support end) is a *defined number* with no likelihood
    # content behind it.  Every statistic in this module filters on
    # `isfinite(p)`, which BEFORE D-19 excluded those tracklets automatically
    # (which is exactly why the published pred/obs for the All column is
    # 295.6 / 294.0 over 488 tracklets, not 306.6 / 305 over 499).  Masking
    # them back out to NaN for the ranking and calibration passes therefore
    # *preserves* the published semantics rather than changing them — the
    # alternative would silently fold 11 policy scores, all of them on true
    # NEOs, into an ROC and an ECE as if they were posteriors.
    #
    # The raw arrays are kept separately: the robustness block below needs the
    # actual recorded values to report how many 1.0s came from which route.
    def _policy_mask(prefix: str) -> np.ndarray:
        return (
            np.isfinite(store[f"{prefix}_p_neo"])
            & (store[f"{prefix}_n_nodes"] > 0)
            & (store[f"{prefix}_lambda_neo"] == 0.0)
            & (store[f"{prefix}_lambda_other"] == 0.0)
        )

    def _likelihood_only(prefix: str):
        raw = np.asarray(store[f"{prefix}_p_neo"], dtype=float)
        pol = _policy_mask(prefix)
        return np.where(pol, np.nan, raw), pol

    p_all_raw = np.asarray(store["priora_all_p_neo"], dtype=float)
    p_all, policy_all_col = _likelihood_only("priora_all")
    p_unk, policy_unk_col = _likelihood_only("priora_unk")

    p_clamp = None
    policy_clamp_col = np.zeros(y.size, dtype=bool)
    if "priora_allclamp_p_neo" in store:
        p_clamp, policy_clamp_col = _likelihood_only("priora_allclamp")
    has_clamp = p_clamp is not None

    scores = {
        "priora All": p_all,
        "priora Unk": p_unk,
    }
    if has_clamp:
        scores["priora All H-clamped"] = p_clamp
    if has_d2:
        scores["digest2 raw (All)"] = d2_raw
        scores["digest2 noid (Unk)"] = d2_noid

    summary: dict = {
        "n_tracklets": int(y.size),
        "n_neo": int(y.sum()),
        "n_non_neo": int((~y).sum()),
        "neo_fraction": float(y.mean()) if y.size else float("nan"),
        "census": census,
        "selection": manifest["selection_counts"],
        "pool_neo_fraction": (
            manifest["pool_neo_by_mpc_nea_list"]
            / max(manifest["selection_counts"]["eligible"], 1)
        ),
        "archive_timestamp_range": manifest["archive_timestamp_range"],
        "score_meta": meta,
        "sources": manifest["sources"],
        "sha256": manifest["sha256"],
    }

    # -- robustness: what real data did to v0 --------------------------------
    #
    # D-19 REWORK.  This pilot's headline finding — 11 of 499 tracklets (2.2%),
    # 100% of them true NEOs, whose ENTIRE admissible region implies H > 25.5
    # and which therefore score 0/0 under D-4 — used to be detected as
    # `~isfinite(p_neo)`.  Since D-19 those tracklets are *scored*, by triage
    # policy, at p = 1.0 with log_lr = nan, so `~isfinite(p_neo)` is
    # all-False and that detector would silently erase the very measurement the
    # policy cites as its empirical basis.
    #
    # The detector is now the recorded triple — non-empty admissible region and
    # both Lambdas exactly zero — which is `classifier._posterior`'s own branch
    # condition and works on both pre- and post-D-19 score files.  Two masks are
    # kept, and they are not the same question:
    #
    #   `wall_all`     the H-support wall: 0/0 over a non-empty region.  This is
    #                  the pilot's finding, and it is invariant to D-19.
    #   `unscored_all` no usable score of any kind (nan p_neo): after D-19 that
    #                  is only an empty admissible region or a 0/0 whose
    #                  mechanism is not the faint-end wall.
    #
    # `frac_h == 1` is NOT used as the detector, and cannot be: it is a
    # measure-weighted fraction that counts NaN-H nodes as "not above", so it
    # cannot distinguish a fully truncated tracklet from one whose nodes were
    # all masked (NOTE 13 of the D-19 review).
    def _wall_mask(prefix: str) -> np.ndarray:
        """The DESIGN §2.6 H wall: 0/0 caused by the H support running out.

        Exactly `classifier._posterior`'s own trigger, reconstructed from the
        stored columns: a non-empty admissible region with both Lambdas
        exactly zero, AND at least `H_TRUNCATION_MEASURE_MIN` of the region's
        measure at or above the H support end.  The measure term is not
        decoration — without it this would also catch the two 0/0 mechanisms
        that are NOT the wall (a hole in the model inside the support, and an
        implied H below zero), which the adversarial review constructed and
        which carry no faint-end information at all.

        `ar_frac_h_above` is stored against `neocp_pilot.H_TRUNC` = 25.5, which
        IS the support end for the All and Unk passes (D-4).  For the clamped
        pass the support end is +inf, so the wall can never be hit there and
        the extra term only makes the mask more conservative — correct either
        way.

        Works on both score-file vintages: pre-D-19 these rows carry a nan
        p_neo, post-D-19 a policy 1.0, and neither the Lambdas nor the measure
        fraction depend on which.
        """
        return (
            (store[f"{prefix}_n_nodes"] > 0)
            & (store[f"{prefix}_lambda_neo"] == 0.0)
            & (store[f"{prefix}_lambda_other"] == 0.0)
            & (store[f"{prefix}_ar_frac_h_above"] >= H_TRUNCATION_MEASURE_MIN)
        )

    wall_all = _wall_mask("priora_all")
    wall_unk = _wall_mask("priora_unk")
    # RAW (unmasked) columns in this block only: it reports what the run
    # actually recorded, policy assignments included.  `p_all`/`p_unk`/
    # `p_clamp` stay the likelihood-only views everywhere else, so no statistic
    # below can pick up a policy score by accident.
    p_unk_raw = np.asarray(store["priora_unk_p_neo"], dtype=float)
    p_clamp_raw = (
        np.asarray(store["priora_allclamp_p_neo"], dtype=float)
        if has_clamp else None
    )
    unscored_all = ~np.isfinite(p_all_raw)
    unscored_unk = ~np.isfinite(p_unk_raw)
    # `nan_all` / `nan_unk` retain their old meaning — "this tracklet has no
    # usable score at all" — measured on the RAW columns; the *reporting* of
    # the H wall uses `wall_all` instead, and every likelihood statistic uses
    # the masked `p_all`/`p_unk` views (see above).
    nan_all = unscored_all
    nan_unk = unscored_unk
    policy_all = policy_all_col
    summary["robustness"] = {
        "p_neo_nonfinite_all": int(unscored_all.sum()),
        "p_neo_nonfinite_unk": int(unscored_unk.sum()),
        # D-19: the H-wall count, independent of whether the run assigned a
        # triage score to them.  THIS is the pilot's 11/499 measurement.
        "h_wall_all": int(wall_all.sum()),
        "h_wall_unk": int(wall_unk.sum()),
        "h_wall_frac_true_neo": (
            float(y[wall_all].mean()) if wall_all.any() else float("nan")
        ),
        "policy_assigned_all": int(policy_all.sum()),
        "policy_assigned_frac_true_neo": (
            float(y[policy_all].mean()) if policy_all.any() else float("nan")
        ),
        "empty_admissible_region": int((store["priora_all_n_nodes"] == 0).sum()),
        "n_masked_nonfinite_median": float(
            np.median(store["priora_all_n_masked_nonfinite"])
        ),
        "n_masked_det_total": int(store["priora_all_n_masked_det"].sum()),
        "ar_frac_h_above_median": float(np.median(frac_h)),
        "ar_frac_h_above_gt_1e2": float(np.mean(frac_h > 1e-2)),
        "ar_frac_h_above_gt_1e1": float(np.mean(frac_h > 1e-1)),
        "n_clean": int(masks["clean"].sum()),
        "n_broken": int(masks["broken"].sum()),
        "priora_seconds_median": float(np.median(store["priora_all_seconds"])),
        # Saturation.  p = 0 exactly means Lambda_NEO is exactly zero while
        # Lambda_other is not.  p = 1 exactly now has TWO causes and they must
        # not be conflated (D-19):
        #   (a) Lambda_other exactly zero with Lambda_NEO > 0 — a genuine
        #       one-sided posterior, log_lr = +inf;
        #   (b) both Lambdas exactly zero over a non-empty region — the D-19
        #       triage POLICY assignment, log_lr = nan, no posterior at all.
        # "Underflowed" was the wrong word for either: the digest2 histogram's
        # smallest non-zero density is ~3e-2 / 3.6e-5 in the interface's units,
        # some 300 orders of magnitude above the denormal floor, so an exact
        # zero means "no node had any density", never "the mass rounded away".
        # Both are the H-support boundary talking rather than the physics
        # (DESIGN §2.6), which is the point the pilot makes.
        "p_neo_exactly_one": int(np.sum(p_all_raw == 1.0)),
        "p_neo_exactly_one_policy_assigned": int(policy_all.sum()),
        "p_neo_exactly_one_one_sided_posterior": int(
            np.sum((p_all_raw == 1.0) & ~policy_all)
        ),
        "p_neo_exactly_zero": int(np.sum(p_all_raw == 0.0)),
    }
    # The DESIGN §2.6 truncation corner, on real data.  Keyed on the H wall,
    # not on non-finiteness, so the measurement survives D-19; the historical
    # key `nonfinite_cases` is kept (the report and its readers use it) and now
    # means "the H-wall cases", which is what it always described.
    wall = wall_all
    if wall.any():
        summary["robustness"]["nonfinite_cases"] = {
            "n": int(wall.sum()),
            "detector": "n_nodes > 0 and lambda_neo == lambda_other == 0.0",
            "n_unscored_under_d4": int(unscored_all.sum()),
            "n_policy_assigned": int(policy_all.sum()),
            "frac_of_sample": float(wall.mean()),
            "frac_true_neo": float(y[wall].mean()),
            "ar_frac_h_above_min": float(np.min(frac_h[wall])),
            "median_rate_deg_per_day": float(
                np.median(
                    np.degrees(
                        np.hypot(
                            np.array([t.ra_rate for t in tracklets])[wall]
                            * np.cos(np.array([t.dec for t in tracklets])[wall]),
                            np.array([t.dec_rate for t in tracklets])[wall],
                        )
                    )
                )
            ),
            "median_v_mag": float(np.median(store["v_mag"][wall])),
            "median_q_au": float(np.median(store["q_au"][wall])),
            "designations": [t.designation for t, m in zip(tracklets, wall) if m],
        }
    if has_clamp:
        # "Recovered" means: the D-4 pair hit the H wall (0/0 over a non-empty
        # region) and the clamped pair produced a real likelihood-based score
        # there.  Measured against the wall rather than against non-finiteness,
        # so the number is the same before and after D-19 — under D-19 the D-4
        # pair also returns a *number* for these, but a policy assignment is
        # not a recovery and must not be counted as one.
        wall_clamp = _wall_mask("priora_allclamp")
        nan_clamp = ~np.isfinite(p_clamp_raw)
        recovered = wall_all & ~wall_clamp
        summary["robustness"]["clamp"] = {
            "p_neo_nonfinite_clamped": int(nan_clamp.sum()),
            "h_wall_clamped": int(wall_clamp.sum()),
            "recovered_by_clamp": int(recovered.sum()),
            "recovered_are_true_neo": (
                float(y[recovered].mean()) if recovered.any() else float("nan")
            ),
            # "elsewhere": both scores are real likelihood-based posteriors,
            # so the wall cases are excluded from BOTH sides (a policy 1.0 vs
            # a clamped likelihood score is not a |dp| worth a median).
            "median_abs_dp_where_both_finite": float(
                np.median(
                    np.abs(p_clamp_raw - p_all_raw)[
                        np.isfinite(p_all_raw) & np.isfinite(p_clamp_raw)
                        & ~wall_all & ~wall_clamp
                    ]
                )
            ),
            "p_neo_exactly_one_clamped": int(np.sum(p_clamp_raw == 1.0)),
        }
    if has_d2:
        summary["robustness"]["digest2_unscored"] = int((~store["d2_ok"]).sum())
        # Does the locally re-run digest2 reproduce the cut that put these
        # objects on the NEOCP in the first place?  It should mostly say yes;
        # where it does not, the tracklet reconstruction and/or the archived
        # digest2 vintage differ from ours, and that is worth knowing before
        # any of the numbers below are read as a head-to-head.
        fin = np.isfinite(d2_raw)
        summary["selection_check"] = {
            "n_with_digest2_score": int(fin.sum()),
            "frac_raw_ge_65": float(np.mean(d2_raw[fin] >= D2_NEOCP_THRESHOLD)),
            "frac_noid_ge_65": float(
                np.mean(d2_noid[np.isfinite(d2_noid)] >= D2_NEOCP_THRESHOLD)
            ),
            "frac_raw_ge_65_neo": float(
                np.mean(d2_raw[fin & y] >= D2_NEOCP_THRESHOLD)
            ) if (fin & y).any() else float("nan"),
            "frac_raw_ge_65_non_neo": float(
                np.mean(d2_raw[fin & ~y] >= D2_NEOCP_THRESHOLD)
            ) if (fin & ~y).any() else float("nan"),
        }

    # -- tracklet-geometry descriptives -------------------------------------
    arc = np.array([t.arc_hours for t in tracklets])
    ndet = np.array([t.n_det for t in tracklets])
    rms = np.array([t.resid_rms_arcsec for t in tracklets])
    rate = np.degrees(
        np.hypot(
            np.array([t.ra_rate for t in tracklets])
            * np.cos(np.array([t.dec for t in tracklets])),
            np.array([t.dec_rate for t in tracklets]),
        )
    )
    summary["tracklets"] = {
        "n_det": _describe(ndet),
        "arc_hours": _describe(arc),
        "fit_resid_rms_arcsec": _describe(rms),
        "total_rate_deg_per_day": _describe(rate),
        "v_mag": _describe(store["v_mag"]),
        "q_au": _describe(store["q_au"]),
        "days_before_neocp_exit": _describe(
            np.array([t.days_before_neocp_exit for t in tracklets])
        ),
        # digest2's own great-circle RMS.  In WP7 this was 0 by construction
        # (two-point exactly linear tracklets); on real astrometry it is not,
        # so digest2's RMS-driven machinery is exercised here for the first time.
        "digest2_rms_arcsec": (
            _describe(store["d2_rms"]) if "d2_rms" in store else {"n": 0}
        ),
        "n_precovery_exists": int(sum(t.precovery_exists for t in tracklets)),
        "n_window_fallback": int(sum(t.window_fallback for t in tracklets)),
        "n_truncated_span": int(sum(t.truncated_span for t in tracklets)),
        "n_label_source_disagree": int(
            sum(not t.label_source_agree for t in tracklets)
        ),
        "obscodes": _counter([t.obscode for t in tracklets]),
        "bands": _counter([b for t in tracklets for b in t.bands]),
        "orbit_classes_non_neo": _counter(
            [t.orbit_class for t in tracklets if not t.is_neo]
        ),
        "orbit_classes_neo": _counter(
            [t.orbit_class for t in tracklets if t.is_neo]
        ),
    }

    # -- score distributions by label ---------------------------------------
    summary["score_distributions"] = {
        name: {
            "neo": _describe(s[y]),
            "non_neo": _describe(s[~y]),
        }
        for name, s in scores.items()
    }

    # -- ROC / AUC ----------------------------------------------------------
    summary["auc"] = {}
    for subset in SUBSETS:
        m = masks[subset]
        if m.sum() < 10 or not y[m].any() or not (~y[m]).any():
            continue
        entry = {
            "n": int(m.sum()),
            "n_neo": int(y[m].sum()),
            "single": {
                name: float(auc_fast(s[m], y[m], w[m])) for name, s in scores.items()
            },
        }
        entry["paired"] = {
            "priora_All_vs_priora_Unk": paired_auc_bootstrap(
                p_all[m], p_unk[m], y[m], w[m], n_boot=n_boot, seed=seed
            )
        }
        if has_clamp:
            entry["paired"]["priora_All_clamped_vs_digest2_raw"] = (
                paired_auc_bootstrap(
                    p_clamp[m], d2_raw[m], y[m], w[m], n_boot=n_boot, seed=seed
                )
                if has_d2
                else {}
            )
            entry["paired"]["priora_All_clamped_vs_All"] = paired_auc_bootstrap(
                p_clamp[m], p_all[m], y[m], w[m], n_boot=n_boot, seed=seed
            )
        if has_d2:
            entry["paired"]["priora_All_vs_digest2_raw"] = paired_auc_bootstrap(
                p_all[m], d2_raw[m], y[m], w[m], n_boot=n_boot, seed=seed
            )
            entry["paired"]["priora_Unk_vs_digest2_noid"] = paired_auc_bootstrap(
                p_unk[m], d2_noid[m], y[m], w[m], n_boot=n_boot, seed=seed
            )
        summary["auc"][subset] = entry

    # -- calibration --------------------------------------------------------
    cal_scores = {"priora All": p_all, "priora Unk": p_unk}
    if has_clamp:
        cal_scores["priora All H-clamped"] = p_clamp
    if has_d2:
        cal_scores["digest2 raw/100"] = d2_raw / 100.0
        cal_scores["digest2 noid/100"] = d2_noid / 100.0
    summary["calibration"] = {}
    for subset in ("all", "clean"):
        m = masks[subset]
        if m.sum() < 10:
            continue
        summary["calibration"][subset] = {
            name: calibration_quantile(sc[m], y[m].astype(float), w[m], n_bins=8)
            for name, sc in cal_scores.items()
        }

    # -- operating points ---------------------------------------------------
    # ``operating_points`` thresholds and cumsums directly, so it needs both
    # scores finite on every row it sees.
    if has_d2:
        def _ops(p, d2s):
            out = {}
            for subset in ("all", "clean"):
                m = masks[subset] & np.isfinite(p) & np.isfinite(d2s)
                if m.sum() < 10 or not y[m].any():
                    continue
                out[subset] = operating_points(
                    p[m], d2s[m], y[m], w[m], d2_threshold=D2_NEOCP_THRESHOLD
                )
            return out

        summary["operating_points"] = _ops(p_all, d2_raw)
        # And the same for the Unk pair, since that is the D-10 question.
        summary["operating_points_unk"] = _ops(p_unk, d2_noid)
        if has_clamp:
            # The clamped pair is the like-for-like comparison with digest2:
            # same top H bin on both sides, so no tracklet is scored against a
            # different effective model, and none returns 0/0.
            summary["operating_points_clamped"] = _ops(p_clamp, d2_raw)

    # -- disagreement census ------------------------------------------------
    if has_d2:
        summary["disagreement"] = _disagreement(tracklets, p_all, d2_raw, y,
                                                percentile_rank)

    # -- by-q-bin behaviour -------------------------------------------------
    summary["by_q_bin"] = {}
    for name, m in _q_bins(store["q_au"]).items():
        if m.sum() == 0:
            continue
        summary["by_q_bin"][name] = {
            "n": int(m.sum()),
            "mean_p_all": float(np.nanmean(p_all[m])),
            "mean_p_unk": float(np.nanmean(p_unk[m])),
            "median_p_all": float(np.nanmedian(p_all[m])),
            "frac_p_all_ge_0.5": float(np.nanmean(p_all[m] >= 0.5)),
            "mean_d2_raw": float(np.nanmean(d2_raw[m])) if has_d2 else None,
            "frac_d2_ge_65": (
                float(np.nanmean(d2_raw[m] >= D2_NEOCP_THRESHOLD))
                if has_d2
                else None
            ),
        }

    # -- plots --------------------------------------------------------------
    plots = {}
    plots["score_hist"] = _plot_score_hist(scores, y, out_dir)
    plots["roc"] = _plot_roc(scores, y, w, out_dir)
    plots["calibration"] = _plot_calibration(summary, out_dir)
    plots["variant"] = _plot_variant_scatter(p_all, p_unk, y, out_dir)
    if has_d2:
        plots["scatter"] = _plot_scatter(p_all, d2_raw, y, out_dir)
    plots["truncation"] = _plot_truncation(frac_h, p_all, y, out_dir)
    summary["plots"] = plots

    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=_json_default)
    report = build_report(summary)
    with open(os.path.join(out_dir, "REPORT.md"), "w") as fh:
        fh.write(report)
    if verbose:
        print(f"[analyze] wrote {out_dir}/summary.json and REPORT.md")
    return summary


def _describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    good = x[np.isfinite(x)]
    if good.size == 0:
        return {"n": 0}
    return {
        "n": int(good.size),
        "mean": float(good.mean()),
        "median": float(np.median(good)),
        "p10": float(np.percentile(good, 10)),
        "p90": float(np.percentile(good, 90)),
        "min": float(good.min()),
        "max": float(good.max()),
    }


def _counter(items) -> dict:
    out: dict[str, int] = {}
    for it in items:
        out[str(it)] = out.get(str(it), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _disagreement(tracklets, p42, d2, y, percentile_rank, k: int = 10) -> dict:
    good = np.isfinite(p42) & np.isfinite(d2)
    idx = np.flatnonzero(good)
    if idx.size < 4:
        return {}
    r42 = percentile_rank(p42[idx])
    rd2 = percentile_rank(d2[idx])
    delta = r42 - rd2
    order = np.argsort(delta)

    def row(j: int) -> dict:
        t = tracklets[int(idx[j])]
        return {
            "designation": t.designation,
            "true_neo": bool(t.is_neo),
            "p_neo_42": float(p42[idx[j]]),
            "d2_raw": float(d2[idx[j]]),
            "delta_rank": float(delta[j]),
            "V": float(t.v_mag),
            "q": float(t.q_au),
            "e": float(t.e),
            "i_deg": float(t.incl_deg),
            "a": float(t.a_au),
            "class": t.orbit_class,
            "rate_deg_day": float(
                np.degrees(np.hypot(t.ra_rate * np.cos(t.dec), t.dec_rate))
            ),
            "arc_hours": float(t.arc_hours),
            "n_det": int(t.n_det),
            "obscode": t.obscode,
        }

    lo = [row(int(j)) for j in order[:k]]
    hi = [row(int(j)) for j in order[::-1][:k]]
    return {
        "rank_correlation": float(np.corrcoef(r42, rd2)[0, 1]),
        "frac_rank_gap_gt_0.25": float(np.mean(np.abs(delta) > 0.25)),
        "digest2_ranks_above": lo,
        "priora_ranks_above": hi,
        "digest2_above_true_neo_frac": float(
            np.mean([r["true_neo"] for r in lo])
        ),
        "priora_above_true_neo_frac": float(
            np.mean([r["true_neo"] for r in hi])
        ),
    }


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _save(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def _plot_score_hist(scores: dict, y: np.ndarray, out_dir: str) -> str:
    names = list(scores)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.4),
                             squeeze=False)
    for ax, name in zip(axes[0], names):
        s = np.asarray(scores[name], float)
        good = s[np.isfinite(s)]
        lo, hi = (0.0, 1.0) if good.size == 0 or good.max() <= 1.0 else (0.0, 100.0)
        bins = np.linspace(lo, hi, 26)
        ax.hist(s[y & np.isfinite(s)], bins=bins, histtype="step", lw=1.8,
                color="crimson", label=f"NEO (n={int(y.sum())})", density=True)
        ax.hist(s[~y & np.isfinite(s)], bins=bins, histtype="step", lw=1.8,
                color="steelblue", label=f"non-NEO (n={int((~y).sum())})",
                density=True)
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
    fig.suptitle("NEOCP pilot: real-data score distributions by true label", y=1.02)
    return _save(fig, out_dir, "score_hist.png")


def _roc_curve(s, y, w):
    s = np.asarray(s, float)
    good = np.isfinite(s)
    s, yy, ww = s[good], np.asarray(y)[good].astype(bool), np.asarray(w)[good]
    order = np.argsort(-s, kind="mergesort")
    tp = np.cumsum(np.where(yy[order], ww[order], 0.0))
    fp = np.cumsum(np.where(~yy[order], ww[order], 0.0))
    P, N = ww[yy].sum(), ww[~yy].sum()
    if P <= 0 or N <= 0:
        return np.array([0, 1.0]), np.array([0, 1.0])
    return np.concatenate(([0], fp / N)), np.concatenate(([0], tp / P))


def _plot_roc(scores: dict, y: np.ndarray, w: np.ndarray, out_dir: str) -> str:
    from validation.digest2_comparison import auc_fast

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    for name, s in scores.items():
        fpr, tpr = _roc_curve(s, y, w)
        ax.plot(fpr, tpr, lw=1.7, label=f"{name}  AUC={auc_fast(s, y, w):.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("false-positive rate (non-NEOs flagged)")
    ax.set_ylabel("true-positive rate (NEOs flagged)")
    ax.set_title("ROC on real NEOCP objects\n(sample already digest2-filtered)",
                 fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out_dir, "roc.png")


def _plot_calibration(summary: dict, out_dir: str) -> str:
    """Reliability curve on equal-count bins, linear axes.

    Linear rather than log-log: unlike WP6's 1%-base-rate catalogue, this
    sample's posteriors occupy the whole [0, 1] range and the interesting
    deviation is absolute, not multiplicative.
    """
    cal = summary.get("calibration", {}).get("all", {})
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    ax.plot([0, 1], [0, 1], "k:", lw=0.9, label="perfect")
    for name, stats in cal.items():
        bins = stats.get("bins") or []
        if not bins:
            continue
        pm = np.array([b["p_mean"] for b in bins])
        of = np.array([b["obs_frac"] for b in bins])
        err = np.array([b.get("obs_frac_err", 0.0) for b in bins])
        ax.errorbar(pm, of, yerr=err, fmt="o-", ms=4, lw=1.4, capsize=2,
                    elinewidth=0.8, label=name)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("mean predicted P(NEO) in bin")
    ax.set_ylabel("observed NEO fraction in bin")
    ax.set_title("Reliability on real NEOCP objects\n"
                 "(equal-count bins; error bars binomial 1-sigma)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, out_dir, "calibration.png")


def _plot_variant_scatter(p_all, p_unk, y, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    both = np.concatenate([p_all, p_unk])
    pos = both[np.isfinite(both) & (both > 0)]
    eps = float(pos.min()) * 0.7 if pos.size else 1e-8
    for m, c, lab in ((y, "crimson", "NEO"), (~y, "steelblue", "non-NEO")):
        ax.scatter(np.clip(p_all[m], eps, 1), np.clip(p_unk[m], eps, 1),
                   s=9, alpha=0.55, color=c, label=lab, edgecolors="none")
    ax.plot([eps, 1], [eps, 1], "k:", lw=0.8)
    ax.set_xlim(eps, 1.6)
    ax.set_ylim(eps, 1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("priora p_neo, All variant")
    ax.set_ylabel("priora p_neo, Unk variant")
    ax.set_title("All vs Unk model variant (D-10 open question)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, out_dir, "variant_scatter.png")


def _plot_scatter(p42, d2, y, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    pos = p42[np.isfinite(p42) & (p42 > 0)]
    eps = float(pos.min()) * 0.7 if pos.size else 1e-8
    for m, c, lab in ((y, "crimson", "NEO"), (~y, "steelblue", "non-NEO")):
        ax.scatter(np.clip(p42[m], eps, 1), d2[m], s=10, alpha=0.55, color=c,
                   label=lab, edgecolors="none")
    ax.set_xlim(eps, 1.6)
    ax.axhline(D2_NEOCP_THRESHOLD, color="0.4", ls="--", lw=0.9,
               label="digest2 NEOCP cut = 65")
    ax.axvline(0.5, color="0.4", ls=":", lw=0.9, label="priora p = 0.5")
    ax.set_xscale("log")
    ax.set_xlabel("priora p_neo (All)")
    ax.set_ylabel("digest2 raw NEO score")
    ax.set_title("Same detections, same model file", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    return _save(fig, out_dir, "scatter.png")


def _plot_truncation(frac_h, p42, y, out_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    f = np.asarray(frac_h, float)
    axes[0].hist(np.clip(f, 1e-5, 1), bins=np.logspace(-5, 0, 30),
                 color="0.5", edgecolor="k", lw=0.4)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(f"admissible-region measure fraction at H > {H_TRUNC}")
    axes[0].set_ylabel("tracklets")
    axes[0].set_title("DESIGN §2.6(b) diagnostic", fontsize=9)
    eps = 1e-8
    for m, c, lab in ((y, "crimson", "NEO"), (~y, "steelblue", "non-NEO")):
        axes[1].scatter(np.clip(f[m], 1e-5, 1), np.clip(p42[m], eps, 1),
                        s=9, alpha=0.55, color=c, label=lab, edgecolors="none")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(f"measure fraction at H > {H_TRUNC}")
    axes[1].set_ylabel("priora p_neo (All)")
    axes[1].legend(fontsize=8)
    return _save(fig, out_dir, "truncation.png")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _fmt(x, spec=".3f"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return format(v, spec)


def _ci(d: dict, key: str = "delta") -> str:
    """``auc_bootstrap`` stores the point estimate in ``key`` and the interval
    in ``key + "_ci"``; render both."""
    if not d or key not in d:
        return "n/a"
    text = _fmt(d[key], "+.4f")
    ci = d.get(key + "_ci")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        text += f" [{_fmt(ci[0], '+.4f')}, {_fmt(ci[1], '+.4f')}]"
    return text


def _headline(s: dict, A) -> None:
    """The eight sentences a reader who stops here should have."""
    r = s.get("robustness", {})
    nc = r.get("nonfinite_cases") or {}
    cl = r.get("clamp") or {}
    auc = s.get("auc", {}).get("all", {})
    single = auc.get("single", {})
    paired = auc.get("paired", {})
    cal = s.get("calibration", {}).get("all", {})
    c42 = cal.get("priora All", {})
    cd2 = cal.get("digest2 raw/100", {})
    ops = (s.get("operating_points", {}) or {}).get("all", {})
    census = s.get("census", {})

    A("## Headline")
    A("")
    A(f"- **{s['n_tracklets']} real NEOCP discovery tracklets** rebuilt from "
      f"MPC astrometry and labelled by mature orbits: "
      f"**{s['n_neo']} NEO / {s['n_non_neo']} non-NEO** (q < 1.3 AU, D-2). "
      f"{census.get('built', 0)} of {census.get('n_sampled', 0)} drawn objects "
      "survived the pipeline.")
    if nc:
        A(f"- **v0 did not crash, hang, or fabricate — but it does refuse.** "
          f"{nc['n']} tracklets ({_fmt(nc['frac_of_sample'], '.1%')}), "
          f"**{_fmt(nc['frac_true_neo'], '.0%')} of them true NEOs**, return "
          "`p_neo = nan`: their entire admissible region implies H > 25.5, so "
          "both hard-truncated components are zero everywhere and the "
          "posterior is 0/0 (DESIGN §2.6, D-4). These are the close, fast, "
          "small objects the NEOCP exists for. Scoring with digest2's own "
          f"[24, inf) top H bin recovers {cl.get('recovered_by_clamp', 0)} of "
          f"them and changes nothing else (median |Δp| "
          f"{_fmt(cl.get('median_abs_dp_where_both_finite'), '.0e')}).")
    if c42 and cd2:
        A(f"- **Calibration survived the prior mismatch**, which was the "
          f"surprise: predicted/observed NEOs "
          f"{_fmt(c42.get('ratio_expected_observed'), '.2f')} "
          f"(ECE {_fmt(c42.get('ece'), '.1e')}) against digest2 `raw`/100 at "
          f"{_fmt(cd2.get('ratio_expected_observed'), '.2f')} "
          f"(ECE {_fmt(cd2.get('ece'), '.1e')}). The NEOCP cut is a function "
          "of the observables alone, so it is exactly the ε(A, V) that "
          "DESIGN §2.4 shows cancels in the posterior.")
    d_raw = paired.get("priora_All_vs_digest2_raw")
    if d_raw and single:
        A(f"- **Ranking**: AUC {_fmt(single.get('priora All'), '.4f')} "
          f"(priora All) vs {_fmt(single.get('digest2 raw (All)'), '.4f')} "
          f"(digest2 raw), ΔAUC {_ci(d_raw)}, P(Δ ≤ 0) = "
          f"{_fmt(d_raw.get('p_delta_le_0'), '.3f')} — the same sign and "
          "magnitude as WP7's synthetic +0.007, not resolved at this sample "
          "size, **and censored against digest2** (the sample is what digest2 "
          "already accepted).")
    d_unk = paired.get("priora_All_vs_priora_Unk")
    if d_unk:
        A(f"- **D-10, All vs Unk: keep `All`.** ΔAUC(All - Unk) = "
          f"{_ci(d_unk)}, P(Δ ≤ 0) = "
          f"{_fmt(d_unk.get('p_delta_le_0'), '.3f')}; digest2's own `raw` beats "
          "its `noid` on the same tracklets. The `Unk` tables subtract a "
          "known-object catalogue frozen at the model's 2011 vintage, which "
          "inflates P(NEO) for main-belt-like tracklets rather than deflating "
          "it.")
    if ops:
        d2pt = ops.get("digest2_neocp", {})
        m42 = ops.get("priora_matched", {})
        p50 = ops.get("priora_p50", {})
        A(f"- **Operationally**, inside this already-filtered sample: "
          f"digest2's own cut (`raw >= 65`) flags {d2pt.get('n_flagged')} "
          f"tracklets at {_fmt(d2pt.get('completeness'), '.1%')} completeness "
          f"and {_fmt(d2pt.get('contamination'), '.1%')} contamination; "
          f"priora matches that completeness at "
          f"{_fmt(m42.get('contamination'), '.1%')} contamination, and at its "
          f"natural p >= 0.5 gives {_fmt(p50.get('completeness'), '.1%')} "
          f"completeness at {_fmt(p50.get('contamination'), '.1%')}.")
    A(f"- **Two data-handling landmines** that no synthetic test could have "
      f"found: the MPC's ADES export uses **two-character band codes** "
      f"(`Pw`, `Ao`, `Lr`, `Vj`) that digest2's offset table does not know "
      f"({census.get('band_aliased', 0)} affected detections, ~0.7 mag each "
      "if missed), and **JPL SBDB returns display-rounded elements**, so "
      "boundary Amors arrive as exactly `q = 1.3` and get mislabelled. Both "
      "are handled and both are documented below.")
    A("- **One bug in a shared validation helper** "
      "(`digest2_comparison.auc_bootstrap`) that this sample's label balance "
      "exposed; reported, not patched — see the Ranking section.")
    A("")


def build_report(s: dict) -> str:
    """Render REPORT.md from ``summary.json`` (no re-computation)."""
    L: list[str] = []
    A = L.append
    A("# WP8 — real-data NEOCP pilot (DESIGN §4.3)")
    A("")
    A("*Produced by `validation/neocp_pilot.py`. This is a "
      "**prior-misspecification stress test**, not a correctness test: v0's "
      "prior is digest2's 2011 S3M-derived histogram [D-10], the real sky is "
      "not, and the astrometry is real. Read every calibration number below "
      "as a measurement of (prior mismatch + selection + method), in that "
      "expected order of size.*")
    A("")
    _headline(s, A)
    A("## The sample, and what it is conditioned on")
    A("")
    sel = s["selection"]
    A("| | |")
    A("|---|---|")
    A(f"| label source | MPC `iau/NEO/elsewhere.csv` — every NEOCP posting "
      f"{s['archive_timestamp_range'][0]} to "
      f"{s['archive_timestamp_range'][1]} ({sel['rows']} rows) |")
    A("| astrometry | MPC observations API (`data.minorplanetcenter.net/api/"
      "get-obs`, ADES dataframe), one object per request |")
    A("| orbits / truth | JPL SBDB per object; NEO iff q < 1.3 AU [D-2], "
      "cross-checked against MPC `nea_extended.json.gz` |")
    A(f"| eligible pool | {sel['eligible']} objects "
      f"(published + designated + non-comet + ground-based discovery station, "
      f"deduplicated); NEO fraction {_fmt(s['pool_neo_fraction'], '.3f')} |")
    A(f"| drawn | {s['census'].get('n_sampled', '?')} uniform random "
      f"(seed in `sample.json`), label-blind |")
    A(f"| built | **{s['n_tracklets']}** tracklets: "
      f"**{s['n_neo']} NEO / {s['n_non_neo']} non-NEO** "
      f"(observed NEO fraction {_fmt(s['neo_fraction'])}) |")
    A("")
    A("**Selection caveats. These are load-bearing, not boilerplate.**")
    A("")
    A("1. **The sample is conditioned on digest2 saying yes.** The MPC posts a "
      "tracklet to the NEOCP only when digest2's NEO score clears its policy "
      "threshold (score >= 65) and an operator agrees. Every object here has "
      "already passed a digest2 cut, so digest2's false negatives are absent "
      "*by construction*. Any head-to-head ROC below is **censored against "
      "digest2** and must not be read as \"which code triages better\": it "
      "measures only the discrimination each code retains *inside* the set "
      "digest2 accepted — i.e. how much of the residual non-NEO "
      "contamination each could still remove. Note the asymmetry with "
      "calibration: because the cut is a deterministic function of the "
      "*observables* (digest2's score is a function of the tracklet, not of "
      "the object's class), it is exactly the ε(A, V) that DESIGN §2.4 shows "
      "cancels in the posterior — so the calibration question survives this "
      "selection even though the ranking comparison does not. That argument "
      "is worked through, and confirmed against the data, in the calibration "
      "section.")
    A("2. **Only designated objects carry a label.** "
      f"{sel['not_published'] + sel['no_designation']} archive rows are "
      "undesignated (\"was not confirmed\", \"does not exist\", ITF) and are "
      "dropped: their truth is genuinely unknown, and they are plausibly "
      "enriched in *both* hard NEOs (fast, lost) and noise. The measured NEO "
      "fraction is therefore the fraction *among objects that got an orbit*, "
      "not among NEOCP postings.")
    A(f"3. **Comets excluded** ({sel['comet']} rows): the digest2 model has no "
      "cometary component, so a comet is neither the NEO component nor the "
      "\"other\" component the two densities partition (DESIGN §5).")
    A(f"4. **Space-based / roving discovery stations excluded** "
      f"({sel['obscode_not_ground_based']} rows: NEOWISE C51, Gaia 258, ...) — "
      "no MPC parallax constants, and v0's observer model has no spacecraft "
      "ephemeris.")
    A("5. **The base rate is real, not balanced.** The draw is label-blind on "
      "purpose: a 50/50 sample would have made the calibration question "
      "(\"does mean p match the observed NEO fraction?\") unanswerable.")
    A("")
    A("### Selection accounting")
    A("")
    A("| stage | rows |")
    A("|---|---:|")
    for k in ("rows", "not_published", "no_designation", "comet", "no_obscode",
              "obscode_not_ground_based", "duplicate_designation", "eligible"):
        if k in sel:
            A(f"| {k.replace('_', ' ')} | {sel[k]} |")
    A("")

    # -- build census
    c = s["census"]
    A("## Fit / parse failure census (real data breaks things)")
    A("")
    A("| outcome | n |")
    A("|---|---:|")
    for k, v in c.items():
        if k in ("n_neo", "n_non_neo"):
            continue
        A(f"| {k.replace('_', ' ')} | {v} |")
    A("")
    t = s["tracklets"]
    losses = {
        k: v
        for k, v in c.items()
        if k not in ("n_sampled", "built", "n_neo", "n_non_neo",
                     "window_fallback", "truncated_span", "precovery_exists",
                     "label_source_disagree", "band_aliased",
                     "band_unknown_to_digest2")
        and v
    }
    top = sorted(losses.items(), key=lambda kv: -kv[1])[:3]
    A(f"Of the {c['n_sampled']} drawn objects, **{c['built']}** yielded a "
      f"usable discovery tracklet "
      f"({100.0 * c['built'] / max(c['n_sampled'], 1):.1f}%)"
      + ("; the losses, largest first: "
         + ", ".join(f"`{k}` ({v})" for k, v in top) + ". "
         if top else " — nothing was lost at all. ")
      + "The failure modes the census watches for, and did **not** see at "
      "any scale here, are the ones that would have been most awkward: no "
      "parse failure, no time-format failure, no fit failure, no obscode we "
      "could not resolve, and no case where the MPC observations database "
      "failed to retain two or more published observations at the archive's "
      "`discovery_obscode` (the station whose submission triggered the "
      "posting). Whatever else this sample stresses, the ingest path is not "
      "the fragile part.")
    A("")
    A("**Tracklet geometry actually seen** (medians unless noted):")
    A("")
    A("| quantity | p10 | median | p90 |")
    A("|---|---:|---:|---:|")
    for key, label in (
        ("n_det", "detections per tracklet"),
        ("arc_hours", "arc length [h]"),
        ("fit_resid_rms_arcsec", "linear-fit residual RMS [arcsec]"),
        ("total_rate_deg_per_day", "total sky rate [deg/day]"),
        ("v_mag", "V (median over tracklet)"),
        ("q_au", "true q [AU]"),
        ("digest2_rms_arcsec", "digest2 great-circle RMS [arcsec]"),
    ):
        d = t.get(key) or {}
        if not d.get("n"):
            continue
        A(f"| {label} | {_fmt(d.get('p10'))} | {_fmt(d.get('median'))} | "
          f"{_fmt(d.get('p90'))} |")
    A("")
    A(f"Stations: {', '.join(f'{k} ({v})' for k, v in list(t['obscodes'].items())[:8])}. "
      f"Native bands before conversion: "
      f"{', '.join(f'{k} ({v})' for k, v in t['bands'].items())}.")
    A("")
    A("### Two things real data broke that synthetic data could not")
    A("")
    A("**1. Two-character band codes.** The MPC observations API's ADES "
      "dataframe reports the photometric band as a *two-character* code for "
      "every modern survey — `Pw` (Pan-STARRS w), `Ao`/`Ac` (ATLAS "
      "orange/cyan), `Lr`/`Lz` (Rubin), `Vj` (Johnson V) — while MPC "
      "80-column, ADES PSV and ADES XML all carry a single character, which is "
      "what digest2's offset table is keyed on. A naive lookup therefore sends "
      f"those codes to the -0.8 \"unknown band\" fallback: "
      f"**{c.get('band_aliased', 0)} of the sample's detections** would have "
      "taken a 0.6-1.0 mag photometric error, silently. The alias table in "
      "`neocp_pilot.MPC_BAND_ALIASES` was verified against the MPC itself by "
      "requesting `ADES_DF` and `OBS80` for the same objects and matching the "
      "band histograms (`Pw`->`w` 105/105 on 2017 UQ124; `Ac`->`c` 38/38 and "
      "`Ao`->`o` 32/32 on 2006 UW157). The order is not uniform — the survey "
      "codes are (system, band), the Johnson/Cousins codes are (band, system), "
      "so `Vj` is V and not J — which is why the table is explicit rather than "
      "a string slice. After aliasing, "
      f"{c.get('band_unknown_to_digest2', 0)} detections still carry a band "
      "digest2 does not know (mostly `UNK`) and keep its -0.8 fallback, as "
      "digest2 itself would.")
    A("")
    A(f"**2. Rounded truth.** JPL's SBDB returns orbital elements as "
      "display-rounded strings, so a boundary Amor comes back as exactly "
      "`q = \"1.3\"` and a naive `q < 1.3` test calls it a non-NEO. "
      f"{t['n_label_source_disagree']} objects in this sample sit on that "
      "boundary. The label therefore uses the MPC's `nea_extended` "
      "perihelion distance (7 digits, and its membership *is* the MPC's "
      "q < 1.3 definition) where the object appears there, with SBDB as the "
      "cross-check — and the disagreement between the two sources is counted "
      "rather than hidden.")
    A("")
    A(f"Provenance flags: {t['n_precovery_exists']} tracklets have an *earlier* "
      "observation at some other station (a later-linked precovery — excluded "
      "from the fit on purpose, since it was not available at triage time), "
      f"{t['n_window_fallback']} fell back outside the "
      "NEOCP-exit window, "
      f"{t['n_truncated_span']} were truncated to the first "
      f"{DEFAULT_MAX_SPAN_HOURS:g} h of a longer night, and "
      f"{t['n_label_source_disagree']} disagree between the two label sources "
      "(SBDB q vs MPC NEA list).")
    A("")

    # -- robustness
    r = s["robustness"]
    A("## Did anything in v0 break?")
    A("")
    A("| check | value |")
    A("|---|---:|")
    A(f"| non-finite `p_neo` (All / Unk) | {r['p_neo_nonfinite_all']} / "
      f"{r['p_neo_nonfinite_unk']} |")
    if "h_wall_all" in r:
        A(f"| H-support-wall tracklets (0/0 over a non-empty region) | "
          f"{r['h_wall_all']} / {r['h_wall_unk']} |")
        A(f"| ... of which D-19 triage-policy scored (p = 1.0, log_lr = nan) | "
          f"{r['policy_assigned_all']} |")
    A(f"| empty admissible region | {r['empty_admissible_region']} |")
    A(f"| nodes zeroed by the Jacobian guard, total | {r['n_masked_det_total']} |")
    A(f"| median measure fraction at H > {H_TRUNC} | "
      f"{_fmt(r['ar_frac_h_above_median'], '.4f')} |")
    A(f"| fraction with that > 1e-2 (\"broken\" premise) | "
      f"{_fmt(r['ar_frac_h_above_gt_1e2'], '.3f')} |")
    A(f"| fraction with that > 1e-1 | "
      f"{_fmt(r['ar_frac_h_above_gt_1e1'], '.3f')} |")
    A(f"| clean / broken subset sizes | {r['n_clean']} / {r['n_broken']} |")
    A(f"| `p_neo` exactly 1.0 / exactly 0.0 | "
      f"{r.get('p_neo_exactly_one', 0)} / {r.get('p_neo_exactly_zero', 0)} |")
    if "p_neo_exactly_one_policy_assigned" in r:
        A(f"| ... of the 1.0s: one-sided posteriors / D-19 policy assignments | "
          f"{r['p_neo_exactly_one_one_sided_posterior']} / "
          f"{r['p_neo_exactly_one_policy_assigned']} |")
    A(f"| median priora CPU per tracklet [s] | "
      f"{_fmt(r['priora_seconds_median'], '.2f')} |")
    if "digest2_unscored" in r:
        A(f"| digest2 tracklets it declined to score | {r['digest2_unscored']} |")
    A("")
    A("Nothing raised, nothing hung, nothing produced a wrong-looking number "
      "silently — but v0 does have one **H-support wall**, and real NEOCP "
      "data walks straight into it. Under D-4 alone it is a hard refusal "
      "(`p_neo = nan`); since D-19 the same tracklets are assigned a triage "
      "score of 1.0 with `log_lr = nan` and a distinct "
      "`ClassificationResult.status`, which is a policy decision and not a "
      "posterior. Either way the *model* has run out, and the count below is "
      "detected from the recorded (n_nodes > 0, Λ_NEO = Λ_other = 0) triple so "
      "that it means the same thing before and after that change.")
    A("")
    nc = r.get("nonfinite_cases")
    if nc:
        A("### The H-support wall (DESIGN §2.6, D-4) — measured")
        A("")
        A(f"**{nc['n']} of {s['n_tracklets']} tracklets "
          f"({_fmt(nc['frac_of_sample'], '.1%')}) hit the wall, and "
          f"{_fmt(nc['frac_true_neo'], '.0%')} of them are "
          "true NEOs.** They are not degenerate geometry: the admissible "
          "region is non-empty and the fit is clean. They are tracklets whose "
          "*entire* admissible region implies H > 25.5 — the minimum "
          f"measure fraction above the support among them is "
          f"{_fmt(nc['ar_frac_h_above_min'], '.4f')}. With both components "
          "hard-truncated at the same H (D-4), every node evaluates both "
          "densities to zero, Λ_NEO = Λ_other = 0, and the posterior is 0/0.")
        A("")
        A(f"Detected as `{nc['detector']}` — **not** as `~isfinite(p_neo)`, "
          "which no longer finds them: under the D-19 triage policy "
          f"{nc['n_policy_assigned']} of them are assigned `p_neo = 1.0` with "
          f"`log_lr = nan`, and only {nc['n_unscored_under_d4']} come back "
          "non-finite. Nor can the measure fraction above the support be used "
          "as the detector: it counts NaN-H nodes as \"not above\", so it "
          "cannot tell a fully truncated tracklet from one whose nodes were "
          "all masked. **This measurement — 11 of 11 true NEOs — is the "
          "empirical basis D-19 cites, so it has to be detectable "
          "independently of the policy it justifies.**")
        A("")
        A(f"Their signature is exactly the DESIGN §2.6 corner made physical: "
          f"median sky rate {_fmt(nc['median_rate_deg_per_day'], '.1f')} deg/day, "
          f"median V {_fmt(nc['median_v_mag'], '.2f')}, median true q "
          f"{_fmt(nc['median_q_au'], '.3f')} AU. Close, fast, small — i.e. the "
          "objects a NEOCP exists for. digest2 scores every one of them 100, "
          "because its top H bin is [24, inf) and never runs out of model. "
          "Affected objects: " + ", ".join(nc["designations"]) + ".")
        A("")
    cl = r.get("clamp")
    if cl:
        A("**The fix already exists in the codebase.** Re-scoring the same "
          "tracklets with `Digest2Histogram(h_top_bin_infinite=True)` — "
          "digest2's own [24, inf) top H bin, the WP7 clamped control — "
          f"recovers **{cl['recovered_by_clamp']} of the {nc['n'] if nc else 0}** "
          "(i.e. produces a real likelihood-based score where the D-4 pair had "
          f"none), leaves {cl['h_wall_clamped']} still at the wall and "
          f"{cl['p_neo_nonfinite_clamped']} non-finite, while changing the "
          "other tracklets by a median |Δp| of "
          f"{_fmt(cl['median_abs_dp_where_both_finite'], '.2e')}. The "
          "recommendation this pilot makes is therefore concrete: **for "
          "NEOCP-like operation on a digest2-derived model, use the clamped "
          "pair.** D-4's hard truncation is right in principle (no "
          "extrapolation beyond declared support) and right in practice on "
          "the synthetic gate, where the corner was a 2.5%-of-measure "
          "curiosity; on real discovery tracklets it is a refusal to score "
          "the highest-value 2% of candidates. The honest longer-term answer "
          "is a population model whose H support reaches the objects surveys "
          "actually find (DESIGN §5, v1's NEOMOD3 side), not a clamp.")
        A("")
    sc = s.get("selection_check")
    if sc:
        A("**Does the locally re-run digest2 reproduce the NEOCP cut?** Of the "
          f"{sc['n_with_digest2_score']} tracklets it scored, "
          f"{_fmt(sc['frac_raw_ge_65'], '.1%')} reach `raw >= 65` and "
          f"{_fmt(sc['frac_noid_ge_65'], '.1%')} reach `noid >= 65` — "
          f"{_fmt(sc['frac_raw_ge_65_neo'], '.1%')} of the true NEOs and "
          f"{_fmt(sc['frac_raw_ge_65_non_neo'], '.1%')} of the true non-NEOs. "
          "Every one of these objects *was* posted to the NEOCP, so a "
          "fraction well below 100% is not a digest2 failure: it measures how "
          "far our reconstructed discovery tracklet (and this model file / "
          "this digest2 build) is from whatever the MPC scored at the time, "
          "plus the operator discretion and the `noid`-vs-`raw` choice in the "
          "real posting policy. Treat it as the honest upper bound on how "
          "literally the \"already digest2-filtered\" caveat applies to *this* "
          "reconstruction.")
        A("")

    # -- distributions
    A("## Score distributions by true label")
    A("")
    A("| score | label | median | mean | p10 | p90 |")
    A("|---|---|---:|---:|---:|---:|")
    for name, d in s["score_distributions"].items():
        for lab in ("neo", "non_neo"):
            e = d[lab]
            A(f"| {name} | {lab} | {_fmt(e.get('median'), '.4g')} | "
              f"{_fmt(e.get('mean'), '.4g')} | {_fmt(e.get('p10'), '.4g')} | "
              f"{_fmt(e.get('p90'), '.4g')} |")
    A("")
    A("![score distributions](score_hist.png)")
    A("")

    # -- ROC
    A("## Ranking (ROC/AUC) — with the censoring caveat above")
    A("")
    first = next(iter(s["auc"].values()), {})
    cols = list(first.get("single", {}))
    A("| subset | n | NEOs | " + " | ".join(cols) + " |")
    A("|---|---:|---:|" + "---:|" * len(cols))
    for subset, e in s["auc"].items():
        A(f"| {subset} | {e['n']} | {e['n_neo']} | "
          + " | ".join(_fmt(e["single"].get(c), ".4f") for c in cols) + " |")
    A("")
    A("Paired, label-stratified bootstrap on the differences "
      "(`neocp_pilot.paired_auc_bootstrap`). Note that a *paired* comparison "
      "must drop any tracklet either score cannot rank, so the "
      "`All-clamped vs All` row is computed on the 488 tracklets where both "
      "are finite — on which the two are all but identical. The 0.0009 gap "
      "between their single-score AUCs in the table above is entirely the 11 "
      "H-wall NEOs that only the clamped pair scores at all.")
    A("")
    A("| subset | comparison | ΔAUC [95% CI] | P(Δ ≤ 0) |")
    A("|---|---|---|---:|")
    for subset, e in s["auc"].items():
        for name, b in e.get("paired", {}).items():
            A(f"| {subset} | {name.replace('_', ' ')} | {_ci(b, 'delta')} | "
              f"{_fmt(b.get('p_delta_le_0'), '.3f')} |")
    A("")
    A("> **Bug found in `validation/digest2_comparison.py:auc_bootstrap`, "
      "reported not patched** (that file is outside this task's ownership). "
      "Its resample builds `take = concatenate((integers(0, n_pos, n_pos), "
      "integers(0, n_neg, n_neg)))` and applies it to "
      "`concatenate((pos, neg))` **without offsetting the negative half by "
      "`n_pos`**, so the negative indices address the positive block. When "
      "`n_pos >= n_neg` — which is this sample, 305 NEOs against 194 non-NEOs, "
      "because the NEOCP is already digest2-filtered — every resample is 100% "
      "positives, every bootstrap AUC is NaN, the CI comes back `[nan, nan]` "
      "and `p_delta_le_0` silently reports `0.000` (because `NaN <= 0` is "
      "`False`). In WP6/WP7's regime (185 NEOs in 2x10^4) the sign is the "
      "other way and the effect is small but nonzero: the negative half is "
      "drawn from a block that is `n_pos/n_neg` ~ 0.9% positives, so those "
      "published intervals come from a slightly wrong resampling "
      "distribution. The numbers above use this module's own corrected "
      "implementation; `DIGEST2_COMPARISON.md`'s CIs should be regenerated "
      "once the shared helper is fixed.")
    A("")
    A("![roc](roc.png)")
    A("")
    if "scatter" in s.get("plots", {}):
        A("![scatter](scatter.png)")
        A("")

    # -- calibration
    A("## Calibration — the prior-mismatch measurement")
    A("")
    A("| subset | score | predicted NEOs | observed | pred/obs | Brier | ECE |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for subset, block in s["calibration"].items():
        for name, st in block.items():
            if not st:
                continue
            A(f"| {subset} | {name} | "
              f"{_fmt(st['expected_positives'], '.1f')} | "
              f"{_fmt(st['observed_positives'], '.1f')} | "
              f"{_fmt(st['ratio_expected_observed'], '.2f')} | "
              f"{_fmt(st['brier'], '.3e')} | {_fmt(st['ece'], '.3e')} |")
    A("")
    A("![calibration](calibration.png)")
    A("")
    _cal_all = s["calibration"].get("all", {}) or {}
    cal_42 = _cal_all.get("priora All", {})
    cal_d2 = _cal_all.get("digest2 raw/100", {})
    cal_clamp = _cal_all.get("priora All H-clamped", {}) or {}
    A("**How to read this — and the pilot's most interesting result.** The "
      "expectation going in was that calibration *had* to degrade here: the "
      "sample is a digest2-selected slice of the sky, and v0's prior is a "
      "2011 population model. It did not. priora's posterior predicts "
      f"{_fmt(cal_42.get('expected_positives'), '.1f')} NEOs against "
      f"{_fmt(cal_42.get('observed_positives'), '.1f')} observed "
      f"(ratio {_fmt(cal_42.get('ratio_expected_observed'), '.2f')}, "
      f"ECE {_fmt(cal_42.get('ece'), '.2e')}), while digest2's `raw`/100 "
      f"predicts {_fmt(cal_d2.get('expected_positives'), '.1f')} against the "
      f"same truth "
      f"(ratio {_fmt(cal_d2.get('ratio_expected_observed'), '.2f')}, "
      f"ECE {_fmt(cal_d2.get('ece'), '.2e')}).")
    A("")
    A("The reason the posterior survives the selection is worth stating "
      "precisely, because it is the DESIGN §2.4 argument doing real work for "
      "the first time. §2.4 requires the selection to be a function "
      "ε(A, V) of the **observables** — then it multiplies both Λ_NEO and "
      "Λ_other and cancels in the ratio. The NEOCP cut is "
      "`digest2_score(A, V) >= 65`: digest2 is a deterministic function of "
      "the tracklet, so the cut is *exactly* of that form. Conditioning on "
      "\"digest2 said yes\" therefore changes **which** (A, V) we see, not "
      "P(NEO | A, V) — and a calibrated posterior stays calibrated on the "
      "selected subsample. Ranking is a different matter: censoring removes "
      "digest2's own false negatives from the sample, and no cancellation "
      "argument repairs a ROC computed on a set chosen by one of the two "
      "contestants (caveat 1).")
    A("")
    A("What the selection does *not* excuse, and what this table therefore "
      "still measures, is prior mismatch — and its verdict is that in the "
      "regime this sample occupies (median V "
      f"{_fmt(s['tracklets']['v_mag'].get('median'), '.1f')}, median arc "
      f"{_fmt(s['tracklets']['arc_hours'].get('median'), '.2f')} h, true q "
      f"{_fmt(s['tracklets']['q_au'].get('min'), '.2f')} to "
      f"{_fmt(s['tracklets']['q_au'].get('max'), '.2f')} AU) "
      "the mismatch between digest2's 2011 S3M "
      "populations and the real 2019-2021 sky is **mild enough not to break "
      "the posterior**. Two honest asterisks: the label censoring (undesignated "
      "postings, caveat 2) is *not* a function of (A, V) and does not cancel; "
      "and the H-support wall above removes the 11 true NEOs it cannot score "
      "from the All row's denominator, which is why the H-clamped row's "
      f"observed count is {_fmt(cal_clamp.get('observed_positives'), '.0f')} "
      f"and the All row's is {_fmt(cal_42.get('observed_positives'), '.0f')}.")
    A("")

    # -- variants
    A("## All vs Unk: which variant belongs in NEOCP-like use? (D-10)")
    A("")
    A("NEOCP candidates are by construction *undiscovered* objects, which is "
      "exactly the population the model's `Unk` (total minus known) tables "
      "describe; digest2's own default score (`noid`) uses `Unk` for the same "
      "reason, while its `raw` score uses `All`. v0 defaults to `All`. Both "
      "were scored here on identical inputs.")
    A("")
    cal_all = s["calibration"].get("all", {})
    if cal_all:
        A("| variant | pred/obs | ECE | Brier | AUC (all) |")
        A("|---|---:|---:|---:|---:|")
        auc_all = s["auc"].get("all", {}).get("single", {})
        for name in ("priora All", "priora Unk", "priora All H-clamped"):
            st = cal_all.get(name)
            if not st:
                continue
            A(f"| {name} | {_fmt(st['ratio_expected_observed'], '.2f')} | "
              f"{_fmt(st['ece'], '.3e')} | {_fmt(st['brier'], '.3e')} | "
              f"{_fmt(auc_all.get(name), '.4f')} |")
        A("")
    auc_all = s["auc"].get("all", {}).get("single", {})
    d_all_unk = (
        s["auc"].get("all", {}).get("paired", {}).get("priora_All_vs_priora_Unk")
    )
    if d_all_unk:
        A(f"**The answer is All, and it is not close.** ΔAUC(All - Unk) = "
          f"{_ci(d_all_unk)}, bootstrap P(Δ ≤ 0) = "
          f"{_fmt(d_all_unk.get('p_delta_le_0'), '.3f')} — and digest2 shows "
          f"the same ordering on its own two scores "
          f"(`raw` {_fmt(auc_all.get('digest2 raw (All)'), '.4f')} vs `noid` "
          f"{_fmt(auc_all.get('digest2 noid (Unk)'), '.4f')}), so this is a "
          "property of the model file rather than of either method. That "
          "contradicts the a-priori argument, and the reason is instructive: "
          "the `Unk` tables are *total minus known*, where \"known\" is the "
          "discovered catalogue **frozen at the model's 2011 vintage**. "
          "Subtracting it removes a large share of the bright main belt and "
          "almost none of the NEOs, which inflates P(NEO) for exactly the "
          "main-belt-like tracklets that make up this sample's negatives. "
          "The premise — \"a NEOCP candidate is undiscovered, so use the "
          "undiscovered population\" — needs the *current* known catalogue to "
          "hold; with a 15-year-stale one it does the opposite of what it "
          "intends. **Recommendation for D-10: keep `All` as the v0 default, "
          "and treat `Unk` as usable only with a contemporaneous known-object "
          "subtraction.**")
        A("")
    A("![variant](variant_scatter.png)")
    A("")

    # -- operating points
    if "operating_points" in s:
        A("## Operating points at digest2's own cut")
        A("")
        A("*Within the already-digest2-selected sample, so \"completeness\" "
          "here is completeness relative to the NEOs that reached the NEOCP, "
          "not to the sky.*")
        A("")
        A("| subset | pair | operating point | flagged | completeness | contamination |")
        A("|---|---|---|---:|---:|---:|")
        for tag, block in (
            ("All/raw", s.get("operating_points", {})),
            ("Unk/noid", s.get("operating_points_unk", {})),
            ("All-clamped/raw", s.get("operating_points_clamped", {})),
        ):
            for subset, pts in block.items():
                for _key, pt in pts.items():
                    if not isinstance(pt, dict):
                        continue
                    A(f"| {subset} | {tag} | {pt['label']} | {pt['n_flagged']} | "
                      f"{_fmt(pt['completeness'], '.1%')} | "
                      f"{_fmt(pt['contamination'], '.1%')} |")
        A("")
        ops_all = (s.get("operating_points", {}) or {}).get("all", {})
        if ops_all:
            d2pt = ops_all.get("digest2_neocp", {})
            m42 = ops_all.get("priora_matched", {})
            p50 = ops_all.get("priora_p50", {})
            c_d2 = d2pt.get("contamination") or float("nan")
            c_50 = p50.get("contamination") or float("nan")
            gain = c_d2 / c_50 if c_50 else float("nan")
            comp_cost = (d2pt.get("completeness") or 0.0) - (
                p50.get("completeness") or 0.0
            )
            A("The WP7 synthetic result was a ~10x contamination reduction at "
              "matched completeness. Here it is much smaller — "
              f"{_fmt(d2pt.get('contamination'), '.1%')} -> "
              f"{_fmt(m42.get('contamination'), '.1%')} at matched "
              f"{_fmt(d2pt.get('completeness'), '.1%')} completeness — and "
              "that shrinkage is the censoring at work rather than a weaker "
              "method: the easy non-NEOs that made up most of WP7's "
              "false-positive budget were never posted to the NEOCP, so they "
              "are not in this sample to remove. The operationally useful "
              "number is the other row: priora at its own p >= 0.5 flags "
              f"{p50.get('n_flagged')} tracklets, "
              f"{_fmt(p50.get('completeness'), '.1%')} complete at "
              f"{_fmt(p50.get('contamination'), '.1%')} contamination — a "
              f"~{_fmt(gain, '.0f')}x contamination reduction for "
              f"{_fmt(comp_cost, '.1%')} "
              "of completeness, and unlike a digest2 score that threshold "
              "can be *chosen* against a stated contamination budget.")
            A("")

    # -- q bins
    A("## Behaviour by true perihelion distance")
    A("")
    A("| q bin | n | mean p (All) | mean p (Unk) | frac p≥0.5 | mean d2 raw | frac d2≥65 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for name, e in s["by_q_bin"].items():
        A(f"| {name} | {e['n']} | {_fmt(e['mean_p_all'], '.3g')} | "
          f"{_fmt(e['mean_p_unk'], '.3g')} | {_fmt(e['frac_p_all_ge_0.5'], '.2f')} | "
          f"{_fmt(e['mean_d2_raw'], '.1f')} | {_fmt(e['frac_d2_ge_65'], '.2f')} |")
    A("")

    # -- disagreement
    dis = s.get("disagreement") or {}
    if dis:
        A("## Disagreement census")
        A("")
        A(f"Percentile-rank correlation between priora (All) and digest2 "
          f"(raw): **{_fmt(dis['rank_correlation'], '.3f')}**; "
          f"{_fmt(dis['frac_rank_gap_gt_0.25'], '.1%')} of tracklets differ by "
          "more than a quarter of the sample in rank. That is far tighter "
          "than WP7's 0.558 on synthetic tracklets, for a mechanical reason: "
          "the NEOCP cut has already removed the population where the two "
          "codes diverge most (WP7's distant-object tail), so what is left is "
          "mostly the region where both agree.")
        A("")
        A("The residual tails still carry WP7's signature. The tracklets "
          "digest2 ranks furthest above priora are dominated by **slow, "
          "distant** objects — Hilda/Trojan/Centaur-like, q ~ 4.6-5.0 AU, "
          "rates ~0.1 deg/day — which digest2 saturates near 75-80 and "
          "priora puts at 1e-4, exactly the failure mode WP7 identified. In "
          "the other tail priora sits above digest2 on Mars-crossers and "
          "high-e inner-belt objects, where both codes are low in absolute "
          "terms and the rank gap costs little. Note also the tracklets where "
          "both saturate (p = 1.0 and raw = 100): they appear in these tables "
          "only because saturation ties are broken by rank, and carry no "
          "information.")
        A("")
        for key, title in (
            ("digest2_ranks_above", "digest2 ranks far above priora"),
            ("priora_ranks_above", "priora ranks far above digest2"),
        ):
            A(f"**{title}**")
            A("")
            A("| designation | true NEO | p_neo (42) | raw (d2) | Δrank | V | "
              "q | e | i | rate [°/d] | n | stn |")
            A("|---|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for row in dis[key]:
                A(f"| {row['designation']} | "
                  f"{'yes' if row['true_neo'] else 'no'} | "
                  f"{_fmt(row['p_neo_42'], '.3e')} | "
                  f"{_fmt(row['d2_raw'], '.1f')} | "
                  f"{_fmt(row['delta_rank'], '+.3f')} | "
                  f"{_fmt(row['V'], '.2f')} | {_fmt(row['q'], '.3f')} | "
                  f"{_fmt(row['e'], '.3f')} | {_fmt(row['i_deg'], '.1f')} | "
                  f"{_fmt(row['rate_deg_day'], '.2f')} | {row['n_det']} | "
                  f"{row['obscode']} |")
            A("")

    A("## Truncation diagnostic (DESIGN §2.6(b))")
    A("")
    A(f"Median admissible-region measure fraction with implied H > {H_TRUNC}: "
      f"**{_fmt(r['ar_frac_h_above_median'], '.4f')}**; "
      f"{_fmt(r['ar_frac_h_above_gt_1e2'], '.1%')} of tracklets exceed 1e-2 "
      f"and {_fmt(r['ar_frac_h_above_gt_1e1'], '.1%')} exceed 1e-1. The "
      f"WP7 \"clean\" subset (< 1e-3, where digest2's [24, inf) top H bin and "
      f"priora's hard truncation at {H_TRUNC} really do read the same "
      f"effective model) holds **{r['n_clean']}** of {s['n_tracklets']} "
      "tracklets here, against 22% of the synthetic gate catalogue. That is "
      "a finding rather than an inconvenience: real NEOCP discovery tracklets "
      "are faint and fast, so the implied-H range they sweep routinely runs "
      "past the model's support, and the DESIGN §4.2 \"same inputs\" premise "
      "is broken for nearly all of them. The WP7 clamped control "
      "(`DIGEST2_COMPARISON_CLAMPED.md`) is the mechanism that repairs the "
      "premise, and it is run here as a third priora pass: with digest2's "
      "top H bin on both sides no tracklet is scored against a different "
      "effective model, and the ranking barely moves "
      f"(ΔAUC(clamped - All) = "
      f"{_ci((s['auc'].get('all', {}).get('paired', {}) or {}).get('priora_All_clamped_vs_All', {}))}"
      ") — the clamp's value here is that it *scores at all* the 2% of "
      "candidates the truncated pair refuses, not that it re-ranks anything.")
    A("")
    A("![truncation](truncation.png)")
    A("")

    A("## Assumptions and known limitations")
    A("")
    A("- **Delta-function attributable** [D-6]: v0 ignores the fitted "
      "attributable's covariance. Real discovery tracklets here are much "
      "shorter than the 45-minute arcs of the ASTROMETRIC_NOISE study "
      "(median arc "
      f"{_fmt(s['tracklets']['arc_hours'].get('median'), '.2f')} h), and rate "
      "error scales as 1/dt, so the \"covariance marginalisation is not "
      "urgent\" conclusion of that study does **not** transfer to this regime. "
      "This is the single largest untested assumption in the pilot.")
    A("- **Unweighted linear fit** of alpha(t), delta(t): ADES "
      "per-observation uncertainties are reported inconsistently across this "
      "archive, so weighting a 3-point fit by sporadic sigmas would make the "
      "estimator depend on which observer happened to report them.")
    A("- **Photometry**: median of the band->V converted magnitudes over the "
      "tracklet, using digest2's own offset table verbatim "
      "(`digest2/observation.py:_BAND_CORRECTIONS`). No colour or taxonomy "
      "coupling [D-5]; the PSV handed to digest2 carries the already-converted "
      "V with band `V`, so digest2's conversion is a no-op and both codes "
      "consume the identical magnitude.")
    A("- **Observatory geolocation**: MPC parallax constants -> ITRS -> WGS84 "
      "geodetic, then passed to `classify(lon_deg=..., lat_deg=..., "
      "height_m=...)`. The unit tests check this reproduces every entry of "
      "`priora.attributable.MPC_OBSCODES` to sub-metre.")
    A("- **Times** are UTC as published; priora converts to TDB internally.")
    A("- **No light-time correction** anywhere, matching v0's convention.")
    A("- **The discovery tracklet is a reconstruction**, not a record of what "
      "digest2 saw at NEOCP time: the MPC back-fills the NEOCP `trksub` onto "
      "every observation once linked, so the tracklet is identified by "
      "(discovery station, earliest night inside the NEOCP-exit window). "
      "Astrometry may also have been re-reduced since.")
    A("- **The labels are today's orbits.** An object designated as a "
      "main-belter on a 3-night arc in 2019 has had 7 years of follow-up; "
      "its q is reliable, but it is not the q anybody knew at triage time.")
    A("")
    A("## Reproduce")
    A("")
    A("```")
    A("PRIORA_BACKEND=python python -m validation.neocp_pilot all "
      "--n 500 --workers 90")
    A("```")
    A("")
    A("`fetch` is the only networked stage and it caches every byte under "
      "`raw/`; `build`, `score` and `analyze` re-run offline.")
    A("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="validation.neocp_pilot",
        description="Real-data NEOCP pilot for priora v0 (DESIGN §4.3).",
    )
    ap.add_argument("stage", choices=["fetch", "build", "score", "analyze", "all"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=500, help="objects to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-rho", type=int, default=1200)
    ap.add_argument("--n-rhodot", type=int, default=50)
    ap.add_argument("--gap-hours", type=float, default=DEFAULT_GAP_HOURS)
    ap.add_argument("--max-span-hours", type=float, default=DEFAULT_MAX_SPAN_HOURS)
    ap.add_argument("--window-days", type=float, default=DEFAULT_WINDOW_DAYS)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--skip-digest2", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download cached raw")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    if args.stage in ("fetch", "all"):
        fetch(out_dir=args.out, n=args.n, seed=args.seed, force=args.force)
    if args.stage in ("build", "all"):
        build(
            out_dir=args.out,
            gap_hours=args.gap_hours,
            max_span_hours=args.max_span_hours,
            window_days=args.window_days,
        )
    if args.stage in ("score", "all"):
        score(
            out_dir=args.out,
            workers=args.workers,
            model=args.model,
            n_rho=args.n_rho,
            n_rhodot=args.n_rhodot,
            skip_digest2=args.skip_digest2,
        )
    if args.stage in ("analyze", "all"):
        analyze(out_dir=args.out, n_boot=args.n_boot)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
