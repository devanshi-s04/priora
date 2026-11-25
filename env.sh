# Source from the repository root before running scripts by hand:   source env.sh
# (the Makefile sets the same variables itself).
export PRIORA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PRIORA_ALERTS="${PRIORA_ALERTS:-$PRIORA_REPO/alerts}"          # Rubin alert + DP2 working tree (scripts in git, data local)
export PRIORA_EXTERNAL="${PRIORA_EXTERNAL:-$PRIORA_REPO/data/external}" # downloaded third-party inputs (see data/README.md)
export PYTHONPATH="$PRIORA_REPO/src:$PRIORA_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PY="${PY:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
