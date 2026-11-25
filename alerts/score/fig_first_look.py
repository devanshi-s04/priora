"""First-look figure for the test page: priora reliability on pure tracklets (All, Unk) and ROC vs digest2."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.metrics import roc_curve
P = pd.read_parquet("scores_joined.parquet"); P = P[P.pure]; y = P.neo.values
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11, "xtick.direction": "in", "ytick.direction": "in"})
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
E = np.array([0, .01, .1, .3, .5, .7, .9, 1.0001])
a.plot([0, 1], [0, 1], ":", color="0.4")
for k, (V, c) in enumerate((("priora_All", "#2a78d6"), ("priora_Unk", "#eb6834"))):
    p = P[V].values; xs, fs, lo, hi = [], [], [], []
    for l, h in zip(E[:-1], E[1:]):
        s = (p >= l) & (p < h); n, o = s.sum(), y[s].sum()
        if n == 0: continue
        xs.append(p[s].mean()); fs.append(o / n); lo.append(beta.ppf(.16, o + .5, n - o + .5)); hi.append(beta.ppf(.84, o + .5, n - o + .5))
    xs, fs = np.array(xs), np.array(fs)
    a.errorbar(xs + (k - .5) * .01, fs, yerr=[np.clip(fs - lo, 0, None), np.clip(np.array(hi) - fs, 0, None)], fmt="o", color=c, mfc="white", capsize=0,
               label=f"{V.replace('_', ' ')}: predicted {p.sum():.0f}, observed {y.sum()}")
a.set(xlim=(0, 1), ylim=(-.02, 1.02), xlabel="mean predicted P(NEO) in bin", ylabel="observed NEO fraction",
      title=f"(a) Calibration, {len(P):,} pure tracklets"); a.legend(frameon=False, fontsize=9, loc="upper left")
for V, c, lab in (("priora_All", "#2a78d6", "priora All"), ("priora_Unk", "#eb6834", "priora Unk"), ("d2_raw", "0.15", "digest2 raw"), ("d2_noid", "0.55", "digest2 NoID")):
    f, t, _ = roc_curve(y, P[V]); b.plot(f, t, color=c, lw=1.6, label=lab)
b.set(xscale="log", xlim=(1e-5, 1), ylim=(0, 1.01), xlabel="false-positive rate (non-NEOs flagged)", ylabel="NEO completeness", title="(b) ROC, 81 NEO tracklets")
b.legend(frameon=False, fontsize=9, loc="lower right")
fig.savefig("../page/img/first_look.png", dpi=170)
