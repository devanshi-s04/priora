"""Section 4: closed-form Jacobian (Rust kernel, Eq. jacqei) against the Python reference's numerical 6x6 determinant
(transforms.ranging_jacobian_det), on 40 randomized observing geometries (seed 19, as tests/test_backend_parity.py),
nodes at mid admissible rho_dot for 400 rho values. Prints n comparisons, median and max relative difference.
Run from the repo root: PYTHONPATH=src python analysis/numbers/jacobian_closed_form_check.py"""
import numpy as np
from astropy.time import Time
import ds42.transforms as T
from ds42._core import jacobian_closed_form
from ds42.ranging import admissible_rhodot_interval
from ds42.attributable import Attributable
rng = np.random.default_rng(19); errs = []; nbad = 0; ngeo = 0
for _ in range(40):
    ra = rng.uniform(0, 2 * np.pi); dec = np.arcsin(rng.uniform(-1, 1)); rr = rng.normal(0, .01); dr = rng.normal(0, .01)
    R = rng.normal(0, 1, 3); R = R / np.linalg.norm(R) * rng.uniform(.9, 1.05); V = rng.normal(0, .008, 3)
    rho = np.geomspace(.02, 60, 400)
    lo, hi = admissible_rhodot_interval(Attributable(ra, dec, rr, dr, Time(2460000.5, format="jd", scale="tdb"), 20.0), R, V, rho)
    w = hi - lo; g = np.isfinite(w) & (w > 0)
    if not g.any(): continue
    ngeo += 1; rho = rho[g]; rd = lo[g] + .5 * w[g]
    r, v = T.topocentric_to_heliocentric(ra, dec, rr, dr, rho, rd, R, V); el = T.elements_from_state(T.state_equ_to_ecl(r), T.state_equ_to_ecl(v))
    for p in ("aei", "qei"):
        d = T.ranging_jacobian_det(ra, dec, rr, dr, rho, rd, R, V, parameterization=p)
        for k in range(len(rho)):
            if not np.all(np.isfinite(el[k])): continue
            if not np.isfinite(d[k]): nbad += 1; continue
            errs.append(abs(jacobian_closed_form(float(rho[k]), float(dec), *map(float, el[k, :3]), p) / d[k] - 1))
e = np.array(errs)
print(f"geometries {ngeo}, comparisons {e.size}, skipped by the conditioning guard {nbad}, median rel {np.median(e):.2e}, max rel {e.max():.2e}")
