"""heliolinc_omp writes 31 summary columns; link_purify's reader (read_clustersum_file) expects 32, with orbit_incl after
orbit_e. Insert orbit_incl = 0 (heliolinc does no orbit fit; link_purify fills the orbit columns). In place."""
import sys
p = sys.argv[1]; L = open(p).read().splitlines()
h = L[0].split(",")
if "orbit_incl" in h: sys.exit(0)
k = h.index("orbit_e") + 1
out = [",".join(h[:k] + ["orbit_incl"] + h[k:])] + [",".join((f := l.split(","))[:k] + ["0.000000"] + f[k:]) for l in L[1:] if l]
open(p, "w").write("\n".join(out) + "\n"); print(p, len(out) - 1, "rows fixed")
