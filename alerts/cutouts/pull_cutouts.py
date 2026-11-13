"""Pull Science / Template / Difference cutouts (FITS + PNG) from Fink's LSST API for every detection in the hand-off CSV.
Retries with backoff while the service times out; skips files already saved. Tier A first. -> cutouts/<tier>/<tracklet>/"""
import time, csv, pathlib, requests
H = list(csv.DictReader(open("../handoff/lsst_neo_candidates_detections_2026-11-03.csv")))
order = {"A": 0, "B": 1, "set aside": 2}; H.sort(key=lambda r: (order[r["tier"]], r["tracklet_id"], int(r["det_index_in_tracklet"])))
URL = "https://api.lsst.fink-portal.org/api/v1/cutouts"
def get(sid, kind, fmt):
    for a in range(6):
        try:
            r = requests.post(URL, json={"diaSourceId": sid, "kind": kind, "output-format": fmt}, timeout=180)
            if r.status_code == 200 and len(r.content) > 500: return r.content
            print("  ", sid, kind, fmt, "HTTP", r.status_code, flush=True)
        except Exception as e:
            print("  ", sid, kind, fmt, "error", type(e).__name__, flush=True)
        time.sleep(min(600, 30 * 2 ** a))
    return None
ok = fail = 0
for r in H:
    d = pathlib.Path(r["tier"].replace(" ", "_")) / r["tracklet_id"]; d.mkdir(parents=True, exist_ok=True)
    for kind in ("Science", "Template", "Difference"):
        for fmt, ext in (("FITS", "fits"), ("PNG", "png")):
            f = d / f"det{r['det_index_in_tracklet']}_{r['diaSourceId']}_{kind}.{ext}"
            if f.exists(): continue
            b = get(r["diaSourceId"], kind, fmt)
            if b: f.write_bytes(b); ok += 1
            else: fail += 1
            time.sleep(1)
    print(r["tier"], r["tracklet_id"], r["det_index_in_tracklet"], "saved", ok, "failed", fail, flush=True)
print("DONE saved", ok, "failed", fail)
