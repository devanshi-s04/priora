#!/bin/bash
# Finish unknown_all: digest2 first (needed for the scatter), then priora Unk. Same commands as score_all.sh.
PY=${PY:-python}; cd $PRIORA_ALERTS/dp2/run; S=score/unknown_all
s=$(date +%s); (cd $PRIORA_REPO && $PY - "$S" <<'PYEOF'
import os
import sys, pandas as pd
from digest2 import Digest2
from digest2.model import ensure_obscodes
stem = os.path.join(os.environ["PRIORA_ALERTS"], "dp2/run/") + sys.argv[1]
with Digest2(model_path="data/digest2.model.csv", obscodes_path=ensure_obscodes(), repeatable=True) as d2:
    res = d2.classify_file(stem + ".psv", max_workers=int(os.environ.get("WORKERS", 8)))
pd.DataFrame({"trkSub": [str(r.designation).strip() for r in res], "raw_neo": [float(r.raw.NEO) for r in res], "noid_neo": [float(r.noid.NEO) for r in res]}).to_csv(stem + "_digest2.csv", index=False)
print("scored", len(res))
PYEOF
) > ${S}_digest2.log 2>&1; echo "$S digest2 exit $? $(( $(date +%s) - s )) s" >> score/run.log
s=$(date +%s); $PY -c "import sys, priora.cli as c; sys.exit(c.main(sys.argv[1:]))" score $S.psv --variant Unk --model $PRIORA_REPO/data/digest2.model.csv --workers ${WORKERS:-8} --format csv --output ${S}_priora_Unk.csv 2> ${S}_priora_Unk.log; echo "$S priora Unk exit $? $(( $(date +%s) - s )) s" >> score/run.log
echo ALL SCORED >> score/run.log
