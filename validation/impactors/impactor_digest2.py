"""Score the impactor tracklets (impactors_<dt>d.npz from impactor_test.py) with digest2 (NoID and raw), using the same
two-detection, 45-minute, geocentric ADES emission as the simulated comparison. Run from the repo root."""
import sys, numpy as np
from types import SimpleNamespace
sys.path.insert(0, "."); sys.path.insert(0, "src")
from validation.digest2_comparison import emit_ades_psv, run_digest2
DT = sys.argv[1] if len(sys.argv) > 1 else "14"
z = np.load(f"validation/impactors/impactors_{DT}d.npz")
n = z["p"].size
run = SimpleNamespace(n=n, index=np.arange(n), gen={k: z[k] for k in ("ra", "dec", "ra_rate", "dec_rate", "epoch_jd_tdb", "v_mag")})
path = f"validation/impactors/impactors_{DT}d.psv"
emit_ades_psv(run, path, dt_min=45.0)
cols, _ = run_digest2(path, np.arange(n), workers=32)
np.savez(f"validation/impactors/impactors_{DT}d_digest2.npz", **cols)
p = z["p"]; s = cols["noid_neo"]; el = z["rows"][:, 4]; obs = el >= 60
for nm, m in (("all", np.ones(n, bool)), ("elongation >= 60", obs)):
    print(f"{nm:18s} n {m.sum():5d} | priora P>=0.5 {np.mean(p[m]>=0.5):.3f} P>=0.1 {np.mean(p[m]>=0.1):.3f} | digest2 NoID>=65 {np.nanmean(s[m]>=65):.3f} | either {np.mean((p[m]>=0.5)|(s[m]>=65)):.3f}")
