"""Simple check on the 343 real CNEOS impactors of Chow et al. (2026) (orbits from dirac.us/qkw, chow2026/).
For each: barycentric ecliptic state at impact -> heliocentric equatorial; integrate backwards with Sun + Earth gravity
(Earth started from the ephemeris at impact); find the earliest time within 30 days before impact at which the object is
brighter than V = 24 from the geocentre (H_v from the file, G = 0.15); observe it there as a 45-min geocentric tracklet and
score with ds42 Unk (H to 30) and digest2 NoID. No survey pointing or visibility constraint except reporting elongation.
Run from the repo root: PYTHONPATH=src python validation/impactors/cneos_check.py"""
import sys, numpy as np, warnings
from types import SimpleNamespace
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
from scipy.integrate import solve_ivp
sys.path.insert(0, "."); sys.path.insert(0, "src")
from ds42.attributable import Attributable, observer_state
from ds42.classifier import classify_many
from ds42.population import Digest2Histogram
from validation.digest2_comparison import emit_ades_psv, run_digest2
warnings.simplefilter("ignore")
MU, MUE = 2.959122082855911e-4, 8.887692445125634e-10
EPS = np.radians(23.43928111)
ROT = np.array([[1, 0, 0], [0, np.cos(EPS), -np.sin(EPS)], [0, np.sin(EPS), np.cos(EPS)]])     # ecliptic -> equatorial
D = "validation/impactors/chow2026/"
orb = np.genfromtxt(D + "cneos_bcart_orbits.txt", names=True, dtype=None, encoding=None)
par = np.genfromtxt(D + "cneos_all_params.txt", names=True, dtype=None, encoding=None)
Hv = dict(zip(par["ObjID"], par["H_v"]))
def f(t, s):
    r, re = s[:3], s[6:9]; d = r - re
    return np.r_[s[3:6], -MU * r / np.linalg.norm(r) ** 3 - MUE * d / np.linalg.norm(d) ** 3, s[9:12], -MU * re / np.linalg.norm(re) ** 3]
def phi(a, G=0.15):
    t = np.tan(a / 2); return (1 - G) * np.exp(-3.33 * t ** 0.63) + G * np.exp(-1.87 * t ** 1.22)
rows, atts, Ro, Vo = [], [], [], []
orb = orb[np.isin(orb["ObjID"], par["ObjID"])]          # the 343 of the readme (orbit file has 2 extra)
for o in orb:
    tI = Time(o["epochMJD_TDB"], format="mjd", scale="tdb")
    sp, sv = get_body_barycentric_posvel("sun", tI)
    rs = ROT @ np.array([o["x"], o["y"], o["z"]]) - sp.xyz.to("au").value
    vs = ROT @ np.array([o["xdot"], o["ydot"], o["zdot"]]) - sv.xyz.to("au/d").value
    RE, VE = observer_state(tI)
    ts = -np.r_[np.arange(1, 24) / 24, np.arange(1, 30.01, 0.25)]            # 1 h .. 30 d before impact
    sol = solve_ivp(f, (0, ts[-1]), np.r_[rs, vs, RE, VE], t_eval=ts, rtol=1e-11, atol=1e-14, method="DOP853")
    best = None
    for k, t in enumerate(sol.t):                                             # earliest time with V < 24
        T = Time(tI.tdb.jd + t, format="jd", scale="tdb"); R0, _ = observer_state(T)
        r = sol.y[:3, k]; d = r - R0; rho, rr = np.linalg.norm(d), np.linalg.norm(r)
        al = np.arccos(np.clip(r @ d / (rr * rho), -1, 1)); V = Hv[o["ObjID"]] + 5 * np.log10(rho * rr) - 2.5 * np.log10(phi(al))
        if V < 24: best = (k, t, V, rho, rr, al)
    if best is None: continue
    k, t, V, rho, rr, al = best
    t0 = tI.tdb.jd + t; dt = 45 / 1440
    s1 = solve_ivp(f, (t, t + dt), sol.y[:, k], rtol=1e-11, atol=1e-14, method="DOP853").y[:, -1]
    T0, T1 = Time(t0, format="jd", scale="tdb"), Time(t0 + dt, format="jd", scale="tdb")
    R0, V0 = observer_state(T0); R1, _ = observer_state(T1)
    d0, d1 = sol.y[:3, k] - R0, s1[:3] - R1
    ra0, de0 = np.arctan2(d0[1], d0[0]), np.arcsin(d0[2] / np.linalg.norm(d0)); ra1, de1 = np.arctan2(d1[1], d1[0]), np.arcsin(d1[2] / np.linalg.norm(d1))
    dra = (ra1 - ra0 + np.pi) % (2 * np.pi) - np.pi
    elong = np.degrees(np.arccos(np.clip(-R0 @ d0 / (np.linalg.norm(R0) * rho), -1, 1)))
    rate = np.degrees(np.hypot(dra * np.cos(de0), de1 - de0)) / dt
    atts.append(Attributable(ra0 % (2 * np.pi), de0, dra / dt, (de1 - de0) / dt, T0, V)); Ro.append(R0); Vo.append(V0)
    rows.append((-t, rho, elong, rate, V, Hv[o["ObjID"]]))
R = np.array(rows); n = len(atts)
neo, oth = Digest2Histogram.neo_other_pair("data/digest2.model.csv", "Unk", h_extend_to=30.0)
p = np.array([x.p_neo for x in classify_many(atts, neo, oth, R_obs=np.array(Ro), V_obs=np.array(Vo), backend="rust", diagnostics=False)])
run = SimpleNamespace(n=n, index=np.arange(n), gen=dict(ra=np.array([a.ra for a in atts]), dec=np.array([a.dec for a in atts]),
      ra_rate=np.array([a.ra_rate for a in atts]), dec_rate=np.array([a.dec_rate for a in atts]),
      epoch_jd_tdb=np.array([a.epoch.tdb.jd for a in atts]), v_mag=np.array([a.v_mag for a in atts])))
emit_ades_psv(run, "validation/impactors/cneos_check.psv", dt_min=45.0); s = run_digest2("validation/impactors/cneos_check.psv", np.arange(n), workers=16, verbose=False)[0]["noid_neo"]
np.savez("validation/impactors/cneos_check.npz", p=p, d2=s, rows=R, cols="days_before rho elong rate V H")
days, rho, el, rate, V, H = R.T
print(f"{len(orb)} impactors; {n} brighter than V=24 within 30 d (others too small/faint)")
print(f"  time before impact: median {np.median(days)*24:.1f} h (10-90%: {np.percentile(days,10)*24:.1f}-{np.percentile(days,90)*24:.1f} h); distance median {np.median(rho):.4f} AU; sky rate median {np.median(rate):.2f} deg/d")
print(f"  undefined ds42: {np.isnan(p).sum()}")
for nm, m in (("all", np.ones(n, bool)), ("elongation >= 60", el >= 60)):
    print(f"  {nm:17s} n {m.sum():3d} | ds42 P>=0.5 {np.nanmean(p[m]>=0.5):.2f}  P>=0.1 {np.nanmean(p[m]>=0.1):.2f}  median {np.nanmedian(p[m]):.3f} | digest2 NoID>=65 {np.nanmean(s[m]>=65):.2f}")
for a, b in ((0, 60), (60, 90), (90, 120), (120, 150), (150, 181)):
    m = (el >= a) & (el < b)
    if m.any(): print(f"  elong {a:3d}-{b:3d}: n {m.sum():3d} ds42 P>=0.5 {np.nanmean(p[m]>=0.5):.2f}  digest2>=65 {np.nanmean(s[m]>=65):.2f}  median rate {np.median(rate[m]):.2f} deg/d")
