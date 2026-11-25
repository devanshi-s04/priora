#!/bin/bash
# Bright-only Unk sample for Mario's NEOCP-matching test (2026-11-06): V 14-21.2, same settings as gate_v0unk_hext30 otherwise.
PY=${PY:-python}; cd $PRIORA_REPO
OMP_NUM_THREADS=1 PYTHONPATH=. $PY -m validation.unk_noid.run_sc run --out validation/output/gate_v0unk_bright212_hext30 --n 3500 --seed 20261106 --variant Unk \
  --v-max 21.2 --v-min 14.0 --h-extend-to 30 --workers ${WORKERS:-8} --no-plots && \
OMP_NUM_THREADS=1 $PY -m validation.digest2_comparison run --run validation/output/gate_v0unk_bright212_hext30 --out validation/output/digest2_comparison_unk_bright212 --workers ${WORKERS:-8} && echo BRIGHT DONE
