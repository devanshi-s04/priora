"""Final verdict per DP2 candidate (655 unlabelled, ds42 All P >= 0.1, 3+ detections), using the local MPCORB check.
known: every detection within 10" of a known object, or a steady offset < 60" (first/last/median agree within 2": orbit error).
mixed: some detections on a known object (median < 5") but others > 10" off: a known asteroid paired with unrelated detections.
Then the earlier groups: strong (trail-confirmed, every test passed, field not crowded), strong but crowded, passed a test, untested,
rejected (pair motion, template at every detection, or trail < 0.7 x expected). -> score/candidate_verdict_final.parquet"""
import gzip, re, numpy as np, pandas as pd
S = "score/"
C = pd.read_parquet(S + "mpcorb_candidates_annot.parquet")
steady = (C.sep_max < 60) & ((C.sep_first - C.sep_last).abs() < 2) & ((C.sep_max - C.sep_med) < 2)
C["known"] = (C.sep_max < 10) | steady
C["mixed"] = ~C.known & (C.sep_med < 5) & (C.sep_max >= 10)
names = set(C.loc[C.known, "mpc_object"].astype(str))
cls = {}
with gzip.open("../mpcorb/MPCORB.DAT.gz", "rt", errors="ignore") as f:
    for L in f:
        if len(L) < 160: continue
        n = L[166:194].strip()
        if n in names:
            a, e = float(L[92:103]), float(L[70:79]); q = a * (1 - e)
            cls[n] = "NEO" if q < 1.3 else "Mars-crosser" if q < 1.666 else "main belt" if a < 5.5 else "outer"
cls.update({n: "comet" for n in names if n.startswith(("C/", "P/")) or re.match(r"^\d+P/", n)})
C["obj_class"] = np.where(C.known, C.mpc_object.astype(str).map(cls), None)
old = {"3": "strong", "4": "passed a test", "5": "untested", "6": "rejected (flag)"}
def final(r):
    if r.known: return "known NEO" if r.obj_class == "NEO" else f"known {r.obj_class}"
    if r.mixed: return "mixed (partly a known object)"
    k = r["cat"][0]
    if k == "3": return "strong" if r.neighbours <= 1 else "strong but crowded field"
    return old.get(k, "rejected (flag)") if k in old else ("passed a test" if k == "2" else "rejected (flag)")
C["verdict"] = [final(r) for _, r in C.iterrows()]
C.to_parquet(S + "candidate_verdict_final.parquet")
print(pd.crosstab(C.verdict, C.n >= 4, margins=True).to_string())
print(C[C.known].sort_values(["verdict", "sep_med"])[["verdict", "n", "mpc_object", "obj_class", "sep_med", "sep_max", "ds42_All"]].round(2).to_string())
