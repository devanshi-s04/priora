"""Imminent impactors: are Earth-impacting NEOs, observed ~2 weeks before impact, classified as NEOs by priora?

Simulation (Sun-only two-body; Earth's and the Moon's gravity ignored, which is a small effect at >0.05 AU):
  * impact epoch uniform in 2026; Earth's heliocentric state from priora.attributable.observer_state (geocentre);
  * relative speed at infinity v_inf uniform in [5, 30] km/s, direction isotropic: heliocentric velocity at impact
    = V_earth + v_inf, position = R_earth; heliocentric-unbound draws are discarded;
  * propagated back by DT_DAYS (default 14) and observed from the geocentre twice, 45 min apart (rates by difference,
    no light time), as in the self-consistency test;
  * H uniform in [18, 30]; V from the H,G (G = 0.15) phase function; kept if 14 <= V < 24;
  * solar elongation recorded (a real survey cannot look within ~60 deg of the Sun).
Scored with priora Unk, extended to H = 30, default grid. Output: impactors_<dt>d.npz and a printed summary.
Run from the repo root: PYTHONPATH=src python validation/impactors/impactor_test.py [dt_days] [n]"""
import sys, numpy as np, warnings
from astropy.time import Time
from scipy.integrate import solve_ivp
from priora.attributable import Attributable, observer_state
from priora.classifier import classify_many
from priora.population import Digest2Histogram
warnings.simplefilter("ignore")
MU = 2.959122082855911e-4; KMS = 86400 / 1.495978707e8          # AU^3/d^2; km/s -> AU/d
DT = float(sys.argv[1]) if len(sys.argv) > 1 else 14.0; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
rng = np.random.default_rng(20261109)
import os
OBS = os.environ.get("OBS")                                  # e.g. X05 for the Rubin site; default geocentre
SFX = f"_{OBS}" if OBS else ""
def f(t, s): r = s[:3]; return np.r_[s[3:], -MU * r / np.linalg.norm(r) ** 3]
def prop(s, dt): return solve_ivp(f, (0, dt), s, rtol=1e-12, atol=1e-14, method="DOP853").y[:, -1]
def phi_hg(a, G=0.15):
    t = np.tan(a / 2); return (1 - G) * np.exp(-3.33 * t ** 0.63) + G * np.exp(-1.87 * t ** 1.22)
rows, atts, Ro, Vo = [], [], [], []
while len(atts) < N:
    jd = 2461041.5 + rng.uniform(0, 365)
    RE, VE = observer_state(Time(jd, format="jd", scale="tdb"))
    vinf = rng.uniform(5, 30); u = rng.normal(size=3); u /= np.linalg.norm(u)
    s_imp = np.r_[RE, VE + vinf * KMS * u]
    if 0.5 * s_imp[3:] @ s_imp[3:] - MU / np.linalg.norm(RE) >= 0: continue
    t0 = Time(jd - DT, format="jd", scale="tdb"); t1 = Time(jd - DT + 45 / 1440, format="jd", scale="tdb")
    s0 = prop(s_imp, -DT); s1 = prop(s_imp, -DT + 45 / 1440)
    R0, V0 = observer_state(t0, obscode=OBS); R1, _ = observer_state(t1, obscode=OBS)
    d0, d1 = s0[:3] - R0, s1[:3] - R1
    ra0, de0 = np.arctan2(d0[1], d0[0]), np.arcsin(d0[2] / np.linalg.norm(d0))
    ra1, de1 = np.arctan2(d1[1], d1[0]), np.arcsin(d1[2] / np.linalg.norm(d1))
    dra = (ra1 - ra0 + np.pi) % (2 * np.pi) - np.pi
    rho, r = np.linalg.norm(d0), np.linalg.norm(s0[:3])
    alpha = np.arccos(np.clip(s0[:3] @ d0 / (r * rho), -1, 1))
    elong = np.degrees(np.arccos(np.clip(-R0 @ d0 / (np.linalg.norm(R0) * rho), -1, 1)))
    H = rng.uniform(18, 30); V = H + 5 * np.log10(rho * r) - 2.5 * np.log10(phi_hg(alpha))
    if not 14 <= V < 24: continue
    rate = np.degrees(np.hypot(dra * np.cos(de0), de1 - de0)) / (45 / 1440)
    a = 1 / (2 / r - s0[3:] @ s0[3:] / MU); hvec = np.cross(s0[:3], s0[3:]); e = np.sqrt(max(0, 1 - hvec @ hvec / (MU * a)))
    atts.append(Attributable(ra0 % (2 * np.pi), de0, dra / (45 / 1440), (de1 - de0) / (45 / 1440), t0, V)); Ro.append(R0); Vo.append(V0)
    rows.append((vinf, rho, r, np.degrees(alpha), elong, H, V, rate, a, e, a * (1 - e)))
neo, oth = Digest2Histogram.neo_other_pair("data/digest2.model.csv", "Unk", h_extend_to=30.0)
res = classify_many(atts, neo, oth, R_obs=np.array(Ro), V_obs=np.array(Vo), backend="rust", diagnostics=False)
p = np.array([x.p_neo for x in res]); R = np.array(rows)
np.savez(f"validation/impactors/impactors_{DT:g}d{SFX}.npz", p=p, rows=R, cols="vinf rho r phase elong H V rate a e q",
         ra=np.array([x.ra for x in atts]), dec=np.array([x.dec for x in atts]), ra_rate=np.array([x.ra_rate for x in atts]),
         dec_rate=np.array([x.dec_rate for x in atts]), epoch_jd_tdb=np.array([x.epoch.tdb.jd for x in atts]), v_mag=np.array([x.v_mag for x in atts]))
vinf, rho, r, ph, el, H, V, rate, a, e, q = R.T
obs = el >= 60
print(f"dt = {DT:g} d, n = {p.size}, undefined {np.isnan(p).sum()}; q<1.3 for all: {(q < 1.3).all()}")
for nm, m in (("all", np.ones_like(obs)), ("elongation >= 60 deg", obs)):
    pp = p[m]; print(f"  {nm:22s} n {m.sum():5d}  median P {np.nanmedian(pp):.3f}  P>=0.5 {np.mean(pp >= 0.5):.3f}  P>=0.1 {np.mean(pp >= 0.1):.3f}  P<0.01 {np.mean(pp < 0.01):.3f}")
print(f"  distance median {np.median(rho):.3f} AU, sky rate median {np.median(rate):.2f} deg/d, phase median {np.median(ph):.0f} deg")
low = p < 0.1
for nm, x in (("sky rate deg/d", rate), ("distance AU", rho), ("v_inf km/s", vinf), ("elongation", el), ("H", H), ("phase", ph)):
    print(f"  {nm:15s} P<0.1 median {np.median(x[low]) if low.any() else np.nan:.2f}   P>=0.5 median {np.median(x[p >= 0.5]):.2f}")
