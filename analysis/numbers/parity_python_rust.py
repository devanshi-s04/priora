"""Section 4: Rust kernel vs Python reference P(NEO) on all 20,000 tracklets of the simulated comparison (gate_v0unk_hext30,
Unk, H to 30, 1200x50 grid), plus the 96-thread batch timing. Writes parity_python_rust.json (max |dP| 1.2e-12, 0.24 ms/tracklet).
Run from the repo root: PYTHONPATH=src python analysis/numbers/parity_python_rust.py  (about 5 min on 48 processes)"""
import sys, os, time, json, warnings, numpy as np
from multiprocessing import Pool
os.environ.update(VAR="Unk",HEXT="30",GENRUN="gate_v0unk_hext30"); sys.argv=["x"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_kernel_unk_hext30 as B
from ds42.classifier import classify, classify_many
warnings.simplefilter("ignore")
neo, other = B.Digest2Histogram.neo_other_pair(B.MODEL,"Unk",h_extend_to=30.0)
N=int(os.environ.get("N","20000")); atts,R,V = B.load(N)
cfg=dict(n_rho=1200,n_rhodot=50,diagnostics=False)
def py(i): return classify(atts[i],neo,other,R[i],V[i],backend="python",**cfg).p_neo
if __name__=="__main__":
    out={"n":len(atts)}
    for w in (96,):
        t0=time.perf_counter(); rr=classify_many(atts,neo,other,R_obs=R,V_obs=V,backend="rust",n_threads=w,**cfg); dt=time.perf_counter()-t0
        out[f"rust_{w}t_ms"]=1e3*dt/len(atts); print(w,"threads",out[f"rust_{w}t_ms"],"ms/tracklet", 3.6e6/out[f"rust_{w}t_ms"],"per hour",flush=True)
    pr=np.array([r.p_neo for r in rr])
    with Pool(48) as p: pp=np.array(p.map(py, range(len(atts)), chunksize=20))
    ok=np.isfinite(pr)&np.isfinite(pp); d=np.abs(pr-pp)[ok]
    out.update(both_finite=int(ok.sum()), nan_mismatch=int((np.isfinite(pr)!=np.isfinite(pp)).sum()), max_abs_dp=float(d.max()), median=float(np.median(d)))
    print(out); json.dump(out,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"parity_python_rust.json"),"w"),indent=1)
