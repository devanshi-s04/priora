# Source from the repository root before running scripts by hand:   source env.sh
# (the Makefile sets the same variables itself).
export DS42_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DS42_ALERTS="${DS42_ALERTS:-$DS42_REPO/alerts}"          # Rubin alert + DP2 working tree (scripts in git, data local)
export DS42_EXTERNAL="${DS42_EXTERNAL:-$DS42_REPO/data/external}" # downloaded third-party inputs (see data/README.md)
export PYTHONPATH="$DS42_REPO/src:$DS42_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PY="${PY:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
