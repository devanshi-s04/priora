"""Section 4: profile the Python reference on 10 tracklets; prints the share of run time in ranging_jacobian_det (88%).
Run from the repo root: PYTHONPATH=src python analysis/numbers/profile_python_reference.py"""
import sys, os, cProfile, pstats, warnings, numpy as np
sys.argv=["x","--n","10"]; os.environ.update(VAR="Unk",HEXT="30",GENRUN="gate_v0unk_hext30")
sys.path.insert(0, os.path.dirname(__file__))
import bench_kernel_unk_hext30 as B
from ds42.classifier import classify
warnings.simplefilter("ignore")
neo, other = B.Digest2Histogram.neo_other_pair(B.MODEL,"Unk",h_extend_to=30.0)
atts,R,V = B.load(10)
classify(atts[0],neo,other,R[0],V[0],backend="python",n_rho=1200,n_rhodot=50)
pr=cProfile.Profile(); pr.enable()
for a,r,v in zip(atts,R,V): classify(a,neo,other,r,v,backend="python",n_rho=1200,n_rhodot=50,diagnostics=False)
pr.disable(); st=pstats.Stats(pr); tot=st.total_tt
for name in ("ranging_jacobian_det","jacobian_elements_wrt_state","jacobian_state_wrt_observables","det"):
    for k,v in st.stats.items():
        if k[2]==name: print(f"{name:34s} {v[3]:7.2f}s  {100*v[3]/tot:5.1f}%  ({k[0].split('/')[-1]})")
print("total", round(tot,2))
