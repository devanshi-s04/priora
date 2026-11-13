"""Pull DP2 DiaSources for dayObs 60858..60866 (2025-07-03..07-11), one async TAP query per night and kind.
unknown: ssObjectId = 0, reliability >= 0.9, psfFlux > 0, isDipole = 0; joined to DiaObject for nDiaSources (no cut on it here).
known:   ssObjectId > 0, same quality cuts; joined to SSSource for designation and ephemeris distances.
-> data/unknown_<night>.parquet, data/known_<night>.parquet (skips nights already saved). DP2 is data-rights only: local use."""
import pathlib, time, sys, requests, pyvo
s = requests.Session(); s.headers["Authorization"] = "Bearer " + (pathlib.Path.home() / ".rsp-tap-token").read_text().strip()
tap = pyvo.dal.TAPService("https://data.lsst.cloud/api/tap", session=s)
COLS = ("s.diaSourceId, s.visit, s.detector, s.diaObjectId, s.ssObjectId, s.midpointMjdTai, s.ra, s.dec, s.raErr, s.decErr, s.psfFlux, s.psfFluxErr, "
        "s.snr, s.band, s.reliability, s.extendedness, s.trailLength, s.templateFlux, s.templateFluxErr, s.scienceFlux, s.isNegative, s.glint_trail, "
        "s.pixelFlags_streak, s.pixelFlags_saturatedCenter")
Q = {"unknown": f"SELECT {COLS}, o.nDiaSources FROM dp2.DiaSource AS s JOIN dp2.DiaObject AS o ON s.diaObjectId = o.diaObjectId WHERE {{w}} AND s.ssObjectId = 0",
     "known": f"SELECT {COLS}, ss.designation, ss.topoRange, ss.helioRange, ss.ephVmag, ss.ephOffset FROM dp2.DiaSource AS s JOIN dp2.SSSource AS ss ON s.diaSourceId = ss.diaSourceId WHERE {{w}} AND s.ssObjectId > 0"}
for n in range(60858, 60867):
    w = f"s.midpointMjdTai >= {n + 0.5} AND s.midpointMjdTai < {n + 1.5} AND s.reliability >= 0.9 AND s.psfFlux > 0 AND s.isDipole = 0"
    for kind, q in Q.items():
        f = pathlib.Path(f"data/{kind}_{n}.parquet")
        if f.exists(): continue
        for att in range(3):
            try:
                t = time.time(); tb = tap.run_async(q.format(w=w), maxrec=5_000_000).to_table()
                tb.to_pandas().to_parquet(f); print(n, kind, len(tb), f"{time.time() - t:.0f} s", flush=True); break
            except Exception as e:
                print(n, kind, "attempt", att, "error", str(e)[:200], flush=True); time.sleep(60)
print("DONE")
