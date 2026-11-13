"""Label unknown-alert tracklet detections by matching them to the current MPC catalogue (data/mpcorb_extended.json.gz,
downloaded 2026-11-02).

Stage 1 (candidates): two-body propagation of every catalogue orbit, vectorised, to 10-min time slices of each night;
  KD-tree match of every tracklet detection within R1 = 10 arcmin.
Stage 2 (exact time): two-body position of each candidate at the detection's own time, with light-time and Rubin's
  topocentric offset (DE440s Earth + astropy site). Tested on known detections: 0.2-1.3 arcsec, July and October.
  (An ASSIST N-body refinement was tried and dropped: far too slow here, and two-body is already ~1 arcsec.)
Output: match.parquet, one row per tracklet detection with the closest object, its separation (arcsec), q and H.
  Labels are applied downstream with a separation cut chosen from the residual distribution.
Run from $DS42_ALERTS:  python3 unk/match/match_mpcorb.py
"""
import os
import gzip, json, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u
from scipy.spatial import cKDTree

K2 = 0.01720209895 ** 2                            # GM_sun in AU^3/day^2
EPS = np.radians(84381.448 / 3600)                 # J2000 obliquity: ecliptic -> equatorial
CE, SE = np.cos(EPS), np.sin(EPS)
C_AUD = 173.1446326846693                          # speed of light, AU/day
R1 = np.radians(10 / 60)                           # stage-1 radius
CACHE = os.path.join(os.environ["DS42_EXTERNAL"], "ar_cache/")
X05 = EarthLocation.from_geodetic(-70.749417 * u.deg, -30.244639 * u.deg, 2663 * u.m)

# --- inputs -------------------------------------------------------------------------------------------------
det = pd.read_parquet("unk/score/trk_dets.parquet")[["detnum", "MJD", "RA", "Dec"]].drop_duplicates("detnum").reset_index(drop=True)
det["night"] = np.floor(det.MJD - 0.5).astype(int)
J = json.load(gzip.open("data/mpcorb_extended.json.gz"))
O = pd.DataFrame({k: [o.get(k) for o in J] for k in ("Principal_desig", "a", "e", "i", "Node", "Peri", "M", "Epoch", "H", "Perihelion_dist")})
O = O.dropna(subset=["a", "e", "i", "Node", "Peri", "M", "Epoch"]).reset_index(drop=True)
O = O[(O.e < 1) & (O.a > 0)].reset_index(drop=True)
del J
print(f"detections {len(det):,}; orbits {len(O):,}", flush=True)


def tdb_jd(mjd_tai):
    return Time(mjd_tai, format="mjd", scale="tai").tdb.jd


def observer_helio(jd_tdb):
    """Rubin heliocentric equatorial position (AU), from DE440s Earth/Sun + astropy GCRS site offset."""
    import spiceypy as sp
    sp.furnsh(CACHE + "de440s.bsp")
    et = (np.atleast_1d(jd_tdb) - 2451545.0) * 86400
    e = np.array([sp.spkgps(399, t, "J2000", 10)[0] for t in et]) / 1.495978707e8
    g = X05.get_gcrs(Time(np.atleast_1d(jd_tdb), format="jd", scale="tdb")).cartesian.xyz.to(u.au).value.T
    return e + g


def kepler_xyz(el, jd):
    """Two-body heliocentric equatorial positions for element arrays el (dict of arrays) at scalar jd (TDB~TT)."""
    a, e = el["a"], el["e"]
    i, W, w = np.radians(el["i"]), np.radians(el["Node"]), np.radians(el["Peri"])
    n = np.sqrt(K2 / a ** 3)
    M = np.radians(el["M"]) + n * (jd - el["Epoch"])
    E = M + e * np.sin(M)
    for _ in range(30):
        E -= (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    xp, yp = a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E)
    cW, sW, cw, sw, ci, si = np.cos(W), np.sin(W), np.cos(w), np.sin(w), np.cos(i), np.sin(i)
    x = (cW * cw - sW * sw * ci) * xp + (-cW * sw - sW * cw * ci) * yp
    y = (sW * cw + cW * sw * ci) * xp + (-sW * sw + cW * cw * ci) * yp
    z = (sw * si) * xp + (cw * si) * yp
    return np.c_[x, y * CE - z * SE, y * SE + z * CE]


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]


def kepler_xyz_t(el, jd):
    """As kepler_xyz but with one time per element row."""
    return kepler_xyz(el, jd)


# --- stage 1 --------------------------------------------------------------------------------------------------
el = {k: O[k].values.astype(float) for k in ("a", "e", "i", "Node", "Peri", "M", "Epoch")}
det["slice"] = np.floor(det.MJD * 144).astype(int)              # 10-min slices
cand = []
t0 = time.time()
for night, dn in det.groupby("night"):
    jm = tdb_jd(np.median(dn.MJD)); ob = observer_helio(jm)[0]
    r = kepler_xyz(el, jm); rho = np.linalg.norm(r - ob, axis=1)
    r = kepler_xyz(el, jm - rho / C_AUD)
    v = (r - ob); v /= np.linalg.norm(v, axis=1)[:, None]
    du = unit(dn.RA.values, dn.Dec.values)
    dist, _ = cKDTree(du).query(v, k=1, distance_upper_bound=2 * np.sin(np.radians(3.0) / 2))
    near = np.where(np.isfinite(dist))[0]                       # orbits within 3 deg of some detection tonight
    sub = {k: x[near] for k, x in el.items()}
    for s, ds in dn.groupby("slice"):
        js = tdb_jd(ds.MJD.median()); ob = observer_helio(js)[0]
        r = kepler_xyz(sub, js); rho = np.linalg.norm(r - ob, axis=1)
        r = kepler_xyz(sub, js - rho / C_AUD); vv = r - ob; vv /= np.linalg.norm(vv, axis=1)[:, None]
        tree = cKDTree(vv)
        for di, L in zip(ds.index.values, tree.query_ball_point(unit(ds.RA.values, ds.Dec.values), 2 * np.sin(R1 / 2))):
            cand.extend((di, int(near[j])) for j in L)
    print(f"night {night}: {len(dn):,} dets, {near.size:,} orbits within 3 deg, candidates so far {len(cand):,} ({time.time() - t0:.0f} s)", flush=True)
C = pd.DataFrame(cand, columns=["det", "orb"]).drop_duplicates()
C.to_parquet("unk/match/stage1_candidates.parquet")
print(f"stage 1: {len(C):,} candidate pairs, {C.det.nunique():,} detections, {C.orb.nunique():,} orbits", flush=True)


# --- stage 2 --------------------------------------------------------------------------------------------------
jd_all = tdb_jd(det.MJD.values); ob_all = observer_helio(jd_all)
d, o = C.det.values, C.orb.values
e2 = {k: x[o] for k, x in el.items()}
jd = jd_all[d]
r = kepler_xyz_t(e2, jd); rho = np.linalg.norm(r - ob_all[d], axis=1)
r = kepler_xyz_t(e2, jd - rho / C_AUD); vv = r - ob_all[d]; rho = np.linalg.norm(vv, axis=1); vv /= rho[:, None]
sep = np.degrees(np.arccos(np.clip((vv * unit(det.RA.values[d], det.Dec.values[d])).sum(1), -1, 1))) * 3600
R = pd.DataFrame({"det": d, "orb": o, "sep_arcsec": sep, "rho_au": rho})
R["desig"] = O.Principal_desig.values[R.orb]; R["q"] = O.Perihelion_dist.values[R.orb]; R["H"] = O.H.values[R.orb]
R.to_parquet("unk/match/stage2_all.parquet")
best = R.sort_values("sep_arcsec").drop_duplicates("det")
M = det.join(best.set_index("det"), how="left")
M.to_parquet("unk/match/match.parquet")
for c in (1, 2, 5, 10, 30, 60):
    print(f"detections matched within {c:>2} arcsec: {(M.sep_arcsec < c).sum():,}")
