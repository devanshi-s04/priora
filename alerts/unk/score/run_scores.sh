#!/bin/bash
# priora (All, Unk; operational defaults incl. h_extend_to 30) then digest2 (raw + NoID) on the same PSV.
PY=${PY:-python}
cd $PRIORA_ALERTS/unk/score
for V in All Unk; do
  start=$(date +%s)
  $PY -c "import sys, priora.cli as c; sys.exit(c.main(sys.argv[1:]))" score tracklets_r90_arc15.psv --variant $V --model $PRIORA_REPO/data/digest2.model.csv --workers ${WORKERS:-8} --format csv --output priora_$V.csv 2> priora_$V.log
  echo "priora $V exit $? in $(( $(date +%s) - start )) s" >> run.log
done
start=$(date +%s)
cd $PRIORA_REPO && $PY - <<'PYEOF' > $PRIORA_ALERTS/unk/score/digest2.log 2>&1
import numpy as np, pandas as pd, re
from digest2 import Digest2
from digest2.model import ensure_obscodes
from importlib.metadata import version; print("digest2", version("digest2"))
with Digest2(model_path="data/digest2.model.csv", obscodes_path=ensure_obscodes(), repeatable=True) as d2:
    res = d2.classify_file("$PRIORA_ALERTS/unk/score/tracklets_r90_arc15.psv", max_workers=90)
pd.DataFrame({"trkSub": [str(r.designation).strip() for r in res], "raw_neo": [float(r.raw.NEO) for r in res],
              "noid_neo": [float(r.noid.NEO) for r in res], "rms": [float(r.rms) for r in res]}).to_csv(
    "$PRIORA_ALERTS/unk/score/digest2.csv", index=False)
print("scored", len(res))
PYEOF
echo "digest2 exit $? in $(( $(date +%s) - start )) s" >> $PRIORA_ALERTS/unk/score/run.log
