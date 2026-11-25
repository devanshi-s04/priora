#!/bin/bash
PY=${PY:-python}
start=$(date +%s)
cd $PRIORA_REPO && $PY - <<'PYEOF' > $PRIORA_ALERTS/score/digest2.log 2>&1
import numpy as np, pandas as pd, re
from digest2 import Digest2
from digest2.model import ensure_obscodes
from importlib.metadata import version; print("digest2", version("digest2"))
with Digest2(model_path="data/digest2.model.csv", obscodes_path=ensure_obscodes(), repeatable=True) as d2:
    res = d2.classify_file("$PRIORA_ALERTS/score/tracklets_v5_arc15.psv", max_workers=90)
pd.DataFrame({"trkSub": [str(r.designation).strip() for r in res], "raw_neo": [float(r.raw.NEO) for r in res],
              "noid_neo": [float(r.noid.NEO) for r in res], "rms": [float(r.rms) for r in res]}).to_csv(
    "$PRIORA_ALERTS/score/digest2.csv", index=False)
print("scored", len(res))
PYEOF
echo "digest2 exit $? in $(( $(date +%s) - start )) s" >> $PRIORA_ALERTS/score/run.log
