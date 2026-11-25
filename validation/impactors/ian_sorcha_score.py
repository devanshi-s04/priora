"""Score Ian Chow's Sorcha-simulated LSST detections of real CNEOS impactors (all_obs_precovery.csv; Chow et al. 2026)
with priora (All and Unk, operational CLI defaults incl. H to 30) and digest2 2.8.2 (raw, NoID).
Tracklets = same-night detections of one object (night = floor(MJD_TAI + 0.33), i.e. split at ~16h UT, local noon at
Rubin), >= 2 detections. Positions are Sorcha's noisy RA_deg/Dec_deg, magnitudes Sorcha's PSFMag in optFilter, site X05.
Run from the repo root."""
import sys, subprocess, numpy as np, pandas as pd
from astropy.time import Time
sys.path.insert(0, "."); sys.path.insert(0, "src")
D = "validation/impactors/"
d = pd.read_csv(D + "all_obs_precovery.csv")
d["night"] = np.floor(d.fieldMJD_TAI + 0.33).astype(int)
d = d.sort_values(["ObjID", "fieldMJD_TAI"])
n_det = d.groupby(["ObjID", "night"]).FieldID.transform("size")
d = d[n_det >= 2].copy()
keys = d.groupby(["ObjID", "night"]).ngroup(); d["trk"] = ["T%05d" % k for k in keys]
utc = Time(d.fieldMJD_TAI.values, format="mjd", scale="tai").utc; utc.precision = 3
with open(D + "ian_sorcha.psv", "w") as fh:
    fh.write("permID |provID |trkSub  |mode |stn  |obsTime                     |ra          |dec         |mag   |band |astCat\n")
    for (_, r), t in zip(d.iterrows(), utc.isot):
        fh.write(f"       |       |{r.trk:<8s}|CCD  |X05  |{t + 'Z':<27s} |{r.RA_deg:12.8f} |{r.Dec_deg:+12.8f} |{r.PSFMag:6.2f} |{r.optFilter:<5s}|Gaia3\n")
for V in ("All", "Unk"):
    subprocess.run([sys.executable, "-c", "import sys, priora.cli as c; sys.exit(c.main(sys.argv[1:]))", "score", D + "ian_sorcha.psv",
                    "--variant", V, "--model", "data/digest2.model.csv", "--format", "csv", "--output", D + f"ian_sorcha_priora_{V}.csv"],
                   check=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"} | dict(__import__("os").environ))
from digest2 import Digest2
from digest2.model import ensure_obscodes
with Digest2(model_path="data/digest2.model.csv", obscodes_path=ensure_obscodes(), repeatable=True) as d2:
    res = d2.classify_file(D + "ian_sorcha.psv", max_workers=8)
dg = pd.DataFrame([dict(trk=r.designation, raw=float(r.raw.NEO), noid=float(r.noid.NEO)) for r in res])
A = pd.read_csv(D + "ian_sorcha_priora_All.csv"); U = pd.read_csv(D + "ian_sorcha_priora_Unk.csv")
print(A.columns.tolist())
# ---- analysis -------------------------------------------------------------------------------------------------
def unit(ra, de): ra, de = np.radians(ra), np.radians(de); return np.c_[np.cos(de)*np.cos(ra), np.cos(de)*np.sin(ra), np.sin(de)]
osun = -d[["Obs_Sun_x_km", "Obs_Sun_y_km", "Obs_Sun_z_km"]].values; osun /= np.linalg.norm(osun, axis=1)[:, None]
d["elong"] = np.degrees(np.arccos(np.clip((unit(d.RA_deg, d.Dec_deg) * osun).sum(1), -1, 1)))
meta = d.groupby("trk").agg(obj=("ObjID", "first"), dbi=("days_before_impact_obs", "median"), elong=("elong", "median"),
                            rate=("movement_deg_day", "median"), H=("H_r", "first")).reset_index()
T = meta.merge(A[["object_id", "n_obs", "arc_h", "V", "p_neo"]].rename(columns={"object_id": "trk", "p_neo": "pA"}), on="trk") \
        .merge(U[["object_id", "p_neo"]].rename(columns={"object_id": "trk", "p_neo": "pU"}), on="trk").merge(dg, on="trk")
T["arc_min"] = T.arc_h * 60
T.to_csv(D + "ian_sorcha_scores.csv", index=False)
pd.set_option("display.width", 220)
print(T[["obj", "trk", "dbi", "n_obs", "arc_min", "V", "elong", "rate", "pU", "pA", "noid", "raw"]].round(3).to_string(index=False))
for nm, m in (("all tracklets", np.ones(len(T), bool)), ("arc >= 15 min", T.arc_min >= 15)):
    t = T[m]; print(f"{nm}: n {len(t)} ({t.obj.nunique()} objects) | priora Unk P>=0.5 {np.mean(t.pU>=0.5):.2f} P>=0.1 {np.mean(t.pU>=0.1):.2f} | priora All P>=0.5 {np.mean(t.pA>=0.5):.2f} | digest2 NoID>=65 {np.mean(t.noid>=65):.2f} raw>=65 {np.mean(t.raw>=65):.2f}")
best = T.groupby("obj").agg(pU=("pU", "max"), noid=("noid", "max"))
print(f"per object (best tracklet): priora Unk P>=0.5 {np.mean(best.pU>=0.5):.2f} ({(best.pU>=0.5).sum()}/{len(best)}) | digest2 NoID>=65 {np.mean(best.noid>=65):.2f} ({(best.noid>=65).sum()}/{len(best)})")
