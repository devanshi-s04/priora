#!/bin/bash
# DP2 scoring queue: known all -> unknown 3+ -> unknown all. ds42 All, Unk and digest2 raw/NoID per set.
PY=${PY:-python}; cd $DS42_ALERTS/dp2/run; mkdir -p score
score () {  # $1 stem
  for V in All Unk; do s=$(date +%s); $PY -c "import sys, ds42.cli as c; sys.exit(c.main(sys.argv[1:]))" score $1.psv --variant $V --model $DS42_REPO/data/digest2.model.csv --workers ${WORKERS:-8} --format csv --output $1_ds42_$V.csv 2> $1_ds42_$V.log; echo "$1 ds42 $V exit $? $(( $(date +%s) - s )) s" >> score/run.log; done
  s=$(date +%s); (cd $DS42_REPO && $PY - "$1" <<'PYEOF'
import os
import sys, pandas as pd
from digest2 import Digest2
from digest2.model import ensure_obscodes
stem = os.path.join(os.environ["DS42_ALERTS"], "dp2/run/") + sys.argv[1]
with Digest2(model_path="data/digest2.model.csv", obscodes_path=ensure_obscodes(), repeatable=True) as d2:
    res = d2.classify_file(stem + ".psv", max_workers=int(os.environ.get("WORKERS", 8)))
pd.DataFrame({"trkSub": [str(r.designation).strip() for r in res], "raw_neo": [float(r.raw.NEO) for r in res], "noid_neo": [float(r.noid.NEO) for r in res]}).to_csv(stem + "_digest2.csv", index=False)
print("scored", len(res))
PYEOF
  ) > $1_digest2.log 2>&1; echo "$1 digest2 exit $? $(( $(date +%s) - s )) s" >> score/run.log
}
$PY build_psv.py known all score/known_all && score score/known_all
$PY build_psv.py unknown 3plus score/unknown_3plus && score score/unknown_3plus
$PY build_psv.py unknown all score/unknown_all && score score/unknown_all
echo ALL SCORED >> score/run.log
