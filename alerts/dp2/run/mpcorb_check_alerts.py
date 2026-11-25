"""ALERT-RERUN copy of mpcorb_check.py (targets from full_nd2). Deterministic known-object check, run locally (no DP2 data leaves epyc).
Orbits: MPC MPCORB.DAT (asteroids, epoch 2026-06-09) + CometEls.txt (comets), downloaded 2026-11-04.
1. Screen: 2-body positions of every asteroid at each night's mean time; keep objects within 5 deg of a target tracklet.
2. Propagate the kept objects + all comets with ASSIST (DE440 + 16 massive asteroids) to each night's mean time,
   then f-g series to each detection time with light-time; observer = Rubin (X05) topocentric. Astrometric RA/Dec.
3. Per tracklet: the object with the smallest median separation over its detections.
Targets: all 655 candidates + 300 random labelled tracklets (validation: they must recover their own designation).
-> score/mpcorb_matches.parquet"""
import gzip, os, sys
from pathlib import Path
import numpy as np, pandas as pd, spiceypy as sp, rebound, assist
from multiprocessing import Pool
from scipy.spatial import cKDTree
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u
A = os.path.join(os.environ["PRIORA_EXTERNAL"], "ar_cache/"); M = os.path.join(os.environ["PRIORA_ALERTS"], "dp2/mpcorb/"); S = os.path.join(os.environ["PRIORA_ALERTS"], "full_nd2/score/")
sp.furnsh(A + "naif0012.tls"); sp.furnsh(A + "de440s.bsp")
GM = 2.9591220828559115e-04; C = 173.1446326846693; EPS = np.radians(84381.448 / 3600)
R_ECL = np.array([[1, 0, 0], [0, np.cos(EPS), -np.sin(EPS)], [0, np.sin(EPS), np.cos(EPS)]])   # ecliptic -> equatorial
def unit(ra, de):
    ra, de = np.radians(ra), np.radians(de); return np.c_[np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)]
def body(name, jd_tdb, rel="SSB"):
    st = np.array([sp.spkezr(name, (j - 2451545.0) * 86400, "J2000", "NONE", rel)[0] for j in np.atleast_1d(jd_tdb)])
    return st[:, :3] / 1.495978707e8, st[:, 3:] / 1.495978707e8 * 86400
# ---------------- targets (alert rerun full_nd2: the 44 candidates + 2 partly-known P >= 0.5 + 200 labelled 3+ tracklets as validation)
AL = os.path.join(os.environ["PRIORA_ALERTS"], "full_nd2/score/")
u3 = pd.read_parquet(AL + "scores_joined.parquet"); u3 = u3[u3.n >= 3]
cand = pd.read_csv(AL + "candidates_annot.csv", index_col=0).index
part = u3[(u3.status == "partial") & (u3.priora_All >= 0.5)].index
val = u3[(u3.status == "labelled") & u3.desig.notna()].sample(200, random_state=1)
c = pd.DataFrame(index=cand.union(part))
T = pd.read_parquet(AL + "trk_dets.parquet"); T = T[T.trkSub.isin(c.index.union(val.index))].copy()
T["jd_tdb"] = Time(T.MJD.values, format="mjd", scale="tai").tdb.jd
loc = EarthLocation.of_site("Rubin Observatory") if "Rubin Observatory" in EarthLocation.get_site_names() else EarthLocation(lon=-70.7494 * u.deg, lat=-30.2444 * u.deg, height=2663 * u.m)
g = loc.get_gcrs(Time(T.MJD.values, format="mjd", scale="tai")).cartesian.xyz.to(u.au).value.T
T[["ox", "oy", "oz"]] = body("EARTH", T.jd_tdb.values)[0] + g                                    # observer, barycentric ICRF
nights = T.groupby(u3.night.reindex(T.trkSub).values).jd_tdb.mean()
T["night"] = u3.night.reindex(T.trkSub).values
print("targets", T.trkSub.nunique(), "detections", len(T), "nights", len(nights), flush=True)
# ---------------- MPCORB
rows = []
with gzip.open(M + "MPCORB.DAT.gz", "rt", errors="ignore") as f:
    go = False
    for L in f:
        if L.startswith("-----"): go = True; continue
        if not go or len(L) < 160 or not L[:7].strip(): continue
        try: rows.append((L[166:194].strip(), L[20:25], float(L[26:35]), float(L[37:46]), float(L[48:57]), float(L[59:68]), float(L[70:79]), float(L[92:103])))
        except ValueError: pass
O = pd.DataFrame(rows, columns=["name", "epoch", "M", "w", "W", "i", "e", "a"])
def unpack(p):
    y = {"I": 1800, "J": 1900, "K": 2000}[p[0]] + int(p[1:3]); m = int(p[3], 36); d = int(p[4], 36)
    return Time(f"{y}-{m:02d}-{d:02d}", scale="tt").tdb.jd
ep = {p: unpack(p) for p in O.epoch.unique()}; O["jd0"] = O.epoch.map(ep)
print("asteroids", len(O), "epochs", sorted(O.epoch.value_counts().head(3).to_dict().items()), flush=True)
def kepler_xyz(a, e, i, W, w, M):   # heliocentric equatorial position and velocity, vectorised, e < 1
    E = M.copy()
    for _ in range(30): E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    n = np.sqrt(GM / a ** 3); cE, sE = np.cos(E), np.sin(E); b = a * np.sqrt(1 - e * e)
    xp, yp = a * (cE - e), b * sE; r = a * (1 - e * cE); vxp, vyp = -a * n * sE / (1 - e * cE), b * n * cE / (1 - e * cE)
    cW, sW, cw, sw, ci, si = np.cos(W), np.sin(W), np.cos(w), np.sin(w), np.cos(i), np.sin(i)
    P = np.c_[cW * cw - sW * sw * ci, sW * cw + cW * sw * ci, sw * si]; Q = np.c_[-cW * sw - sW * cw * ci, -sW * sw + cW * cw * ci, cw * si]
    return (P * xp[:, None] + Q * yp[:, None]) @ R_ECL.T, (P * vxp[:, None] + Q * vyp[:, None]) @ R_ECL.T
rad = lambda x: np.radians(x.values)
ell = O.e < 1
# ---------------- 1. screen
keep = set()
KF = Path(S + "mpcorb_screen_keep.npy")
for nt, jd in ([] if KF.exists() else nights.items()):
    Mt = rad(O.M) + np.sqrt(GM / O.a.values ** 3) * (jd - O.jd0.values)
    r, _ = kepler_xyz(O.a.values, O.e.values, rad(O.i), rad(O.W), rad(O.w), np.where(ell, Mt, 0))
    eh = body("EARTH", jd, "SUN")[0][0]; d = r - eh; d /= np.linalg.norm(d, axis=1)[:, None]
    tg = T[T.night == nt].groupby("trkSub").first(); tu = unit(tg.RA.values, tg.Dec.values)
    near = cKDTree(tu).query_ball_point(d, 2 * np.sin(np.radians(5) / 2))
    keep |= {j for j, hit in enumerate(near) if hit and ell.iat[j]}
    print("night", nt, "kept so far", len(keep), flush=True)
if KF.exists(): keep = set(np.load(KF).tolist())
else: np.save(KF, np.array(sorted(keep)))
K = O.iloc[sorted(keep)].reset_index(drop=True)
# ---------------- comets
cm = []
for L in open(M + "CometEls.txt"):
    try: cm.append((L[102:158].strip(), int(L[14:18]), int(L[19:21]), float(L[22:29]), float(L[30:39]), float(L[41:49]), float(L[51:59]), float(L[61:69]), float(L[71:79]), L[81:89]))
    except ValueError: pass
CM = pd.DataFrame(cm, columns=["name", "Ty", "Tm", "Td", "q", "e", "w", "W", "i", "epoch"])
CM["Tp"] = [Time(f"{y}-{m:02d}-01", scale="tt").tdb.jd + d - 1 for y, m, d in zip(CM.Ty, CM.Tm, CM.Td)]
CM["jd0"] = [Time(f"{s[:4]}-{s[4:6]}-{s[6:8]}", scale="tt").tdb.jd if s.strip().isdigit() else tp for s, tp in zip(CM.epoch, CM.Tp)]   # no epoch: elements at perihelion
CM = CM[(CM.q < 15)]
print("kept asteroids", len(K), "comets", len(CM), flush=True)
# ---------------- 2. initial barycentric states at each object's epoch
def states_ast(D):
    r, v = kepler_xyz(D.a.values, D.e.values, rad(D.i), rad(D.W), rad(D.w), rad(D.M)); return r, v
def states_com(D):
    r, v = [], []
    for _, o in D.iterrows():
        s = rebound.Simulation(); s.G = GM; s.add(m=1)
        try: s.add(primary=s.particles[0], a=o.q / (1 - o.e) if abs(1 - o.e) > 1e-8 else o.q / 1e-8, e=o.e if abs(1 - o.e) > 1e-8 else 1 - 1e-8,
                   inc=np.radians(o.i), Omega=np.radians(o.W), omega=np.radians(o.w), T=o.Tp - o.jd0)
        except Exception: r.append([np.nan] * 3); v.append([np.nan] * 3); continue
        p = s.particles[1]; r.append(R_ECL @ np.array(p.xyz)); v.append(R_ECL @ np.array(p.vxyz))
    return np.array(r), np.array(v)
ra_, va_ = states_ast(K); rc_, vc_ = states_com(CM)
names = list(K.name) + list(CM.name); jd0 = np.r_[K.jd0.values, CM.jd0.values]; R0 = np.r_[ra_, rc_]; V0 = np.r_[va_, vc_]
ok = np.isfinite(R0).all(1); names = [n for n, k in zip(names, ok) if k]; jd0, R0, V0 = jd0[ok], R0[ok], V0[ok]
print("objects to propagate", len(names), flush=True)
NT = nights.values
SUN0 = {e: body("SUN", e) for e in np.unique(jd0)}   # computed here: SPICE must not be read from forked workers
import signal, time as _time
class Slow(Exception): pass
def _alarm(*a): raise Slow()
def integrate(idx, limit):
    """ASSIST states at the night times for objects idx; raises Slow after `limit` s."""
    eph = assist.Ephem(A + "linux_p1550p2650.440", A + "sb441-n16.bsp"); out = np.full((len(idx), len(NT), 6), np.nan)
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(limit)
    try:
        for e0 in np.unique(jd0[idx]):
            sel = [k for k in range(len(idx)) if jd0[idx[k]] == e0]
            sim = rebound.Simulation(); sim.t = e0 - eph.jd_ref; sr, sv = SUN0[e0]
            for k in sel:
                j = idx[k]; x = R0[j] + sr[0]; vv = V0[j] + sv[0]; sim.add(x=x[0], y=x[1], z=x[2], vx=vv[0], vy=vv[1], vz=vv[2])
            sim.ri_ias15.min_dt = 1e-3; ex = assist.Extras(sim, eph)   # min step 86 s: sungrazing comets otherwise crawl
            for ti in np.argsort(-NT):     # epoch is later than DP2: integrate backwards, in 10-day pieces so the timeout can fire
                tgt = NT[ti] - eph.jd_ref
                while abs(sim.t - tgt) > 10: sim.integrate(sim.t + np.sign(tgt - sim.t) * 10)
                sim.integrate(tgt)
                for q, k in enumerate(sel):
                    p = sim.particles[q]; out[k, ti] = [p.x, p.y, p.z, p.vx, p.vy, p.vz]
    finally: signal.alarm(0)
    return out
def run(idx):
    """Batch of ~200 together; if the batch is slow (one object on a close approach), each object alone with its own limit."""
    try: return idx, integrate(idx, 120), 0
    except Slow:
        out = np.full((len(idx), len(NT), 6), np.nan); bad = 0
        for k, j in enumerate(idx):
            try: out[k] = integrate(np.array([j]), 30)[0]
            except Slow: bad += 1
        return idx, out, bad
chunks = np.array_split(np.arange(len(names)), max(1, len(names) // 200))
ST = np.full((len(names), len(NT), 6), np.nan); nbad = 0; t0 = _time.time()
with Pool(90) as pool:
    for n, (idx, out, bad) in enumerate(pool.imap_unordered(run, chunks)):
        ST[idx] = out; nbad += bad
        if n % 200 == 0: print(f"chunks {n}/{len(chunks)}  {(_time.time() - t0) / 60:.1f} min  not propagated {nbad}", flush=True)
print("not propagated (timed out):", nbad, [names[k] for k in np.where(~np.isfinite(ST[:, 0, 0]))[0][:20]], flush=True)
np.save(S + "mpcorb_states.npy", ST)
print("propagated", ST.shape, flush=True)
# ---------------- 3. per detection, per object: astrometric topocentric RA/Dec via f-g series from the night state
night_idx = {nt: k for k, nt in enumerate(nights.index)}
res = []
for t, D in T.groupby("trkSub"):
    k = night_idx[D.night.iloc[0]]; X = ST[:, k, :]; sr, sv = body("SUN", NT[k])
    r0, v0 = X[:, :3] - sr[0], X[:, 3:] - sv[0]
    # preselect objects within 3 deg of the first detection at the night mean time
    rb = np.nan_to_num(X[:, :3]); d0 = rb - D[["ox", "oy", "oz"]].values[0]; d0 /= np.linalg.norm(d0, axis=1)[:, None]
    near = np.where(d0 @ unit([D.RA.iloc[0]], [D.Dec.iloc[0]])[0] > np.cos(np.radians(3)))[0]
    near = near[np.isfinite(X[near, 0])]
    if not len(near): res.append(dict(trkSub=t)); continue
    r0n, v0n = r0[near], v0[near]; rr = np.linalg.norm(r0n, axis=1); uu = GM / rr ** 3; p = (r0n * v0n).sum(1) / rr ** 2; qq = (v0n ** 2).sum(1) / rr ** 2 - uu
    seps = []
    for _, det in D.iterrows():
        dt = det.jd_tdb - NT[k]; sd = body("SUN", det.jd_tdb)[0][0]
        for _ in range(2):   # light time
            f = 1 - uu * dt ** 2 / 2 + uu * p * dt ** 3 / 2 + uu * (uu - 15 * p ** 2 + 3 * qq) * dt ** 4 / 24
            gg = dt - uu * dt ** 3 / 6 + uu * p * dt ** 4 / 4
            pos = f[:, None] * r0n + gg[:, None] * v0n + sd - np.array([det.ox, det.oy, det.oz])
            dt = det.jd_tdb - np.linalg.norm(pos, axis=1) / C - NT[k]
        pos /= np.linalg.norm(pos, axis=1)[:, None]
        seps.append(np.degrees(np.arccos(np.clip(pos @ unit([det.RA], [det.Dec])[0], -1, 1))) * 3600)
    seps = np.array(seps); med = np.median(seps, axis=0); b = int(np.argmin(med))
    res.append(dict(trkSub=t, mpc_object=names[near[b]], sep_med=float(med[b]), sep_max=float(seps[:, b].max()), sep_first=float(seps[0, b]), sep_last=float(seps[-1, b]),
                    n_within_120=int((med < 120).sum())))
R = pd.DataFrame(res).set_index("trkSub")
R["set"] = np.where(R.index.isin(val.index), "validation", np.where(R.index.isin(part), "partial", "candidate")); R["true_desig"] = val.desig.reindex(R.index)
R.to_parquet(S + "mpcorb_matches.parquet")
V = R[R.set == "validation"]
print("validation:", len(V), "sep_med quantiles 50/90/95/99", V.sep_med.quantile([.5, .9, .95, .99]).round(2).tolist(), "within 5/30 arcsec", int((V.sep_med < 5).sum()), int((V.sep_med < 30).sum()))
print(V.sort_values("sep_med").tail(8)[["true_desig", "mpc_object", "sep_med", "sep_max"]].round(1).to_string())
Cn = R[R.set == "candidate"]; print("candidates:", len(Cn), "matched <5\"", int((Cn.sep_med < 5).sum()), "<30\"", int((Cn.sep_med < 30).sum()), "<120\"", int((Cn.sep_med < 120).sum()))
