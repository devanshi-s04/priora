#!/usr/bin/env bash
# Download third-party inputs into $DS42_EXTERNAL (default data/external):
#   opsim/baseline_v5.3.7_10yrs.db   LSST baseline cadence simulation (Sec. 7 arc lengths)
#   ar_cache/                        Sorcha/ASSIST ephemeris files (sorcha bootstrap; used by the known-object checks)
#   heliolinx/, heliolinx-aux/       HelioLinC at the commits used, make_tracklets built
#   ../../paper/aastex701.cls, aasjournalv7.bst   AASTeX 7.0.1 (CTAN)
set -euo pipefail
: "${DS42_EXTERNAL:?source env.sh or run through make}"
mkdir -p "$DS42_EXTERNAL/opsim" "$DS42_EXTERNAL/ar_cache"
cd "$DS42_EXTERNAL"
DB=opsim/baseline_v5.3.7_10yrs.db
[ -f "$DB" ] || curl -L --fail -o "$DB" https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs5.3/baseline/baseline_v5.3.7_10yrs.db
[ -f ar_cache/sb441-n16.bsp ] || sorcha bootstrap --cache ar_cache
if [ ! -d heliolinx ]; then
  git clone https://github.com/heliolinx/heliolinx.git && git -C heliolinx checkout e9798170327eb4f59a67d49182ec4a5c3c88ed6e
  (cd heliolinx/src && make make_tracklets heliolinc_omp link_purify)
fi
if [ ! -d heliolinx-aux ]; then
  git clone https://github.com/heliolinx/heliolinx-aux.git && git -C heliolinx-aux checkout 03e34116fd247aa6ba15aae64c02be2a783c5797
fi
cd "$DS42_REPO/paper"
if [ ! -f aastex701.cls ]; then
  curl -L --fail -o /tmp/aastex.zip https://mirrors.ctan.org/macros/latex/contrib/aastex.zip
  python - <<'PY'
import zipfile
z = zipfile.ZipFile("/tmp/aastex.zip")
for n in z.namelist():
    if n.endswith(("aastex701.cls", "aasjournalv7.bst")):
        open(n.split("/")[-1], "wb").write(z.read(n))
PY
fi
echo "external inputs ready in $DS42_EXTERNAL"
