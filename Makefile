# ds42 paper: build, data, analysis and paper targets.  `make help` lists them.
SHELL := /bin/bash
export DS42_REPO     := $(CURDIR)
export DS42_ALERTS   ?= $(CURDIR)/alerts
export DS42_EXTERNAL ?= $(CURDIR)/data/external
export PYTHONPATH    := $(CURDIR)/src:$(CURDIR)
export OMP_NUM_THREADS ?= 1
export PY ?= python
WORKERS ?= 8
# Zenodo record holding the stored simulation and NEOCP outputs (validation/output); set once published.
DATA_URL ?=

.PHONY: help install test model external data analysis numbers figures paper simulations alerts dp2 clean

help:
	@echo "make install      install ds42 (editable) and build the Rust kernel"
	@echo "make test         run the test suite"
	@echo "make model        download the digest2 population model from the MPC (data/digest2.model.csv)"
	@echo "make external     download third-party inputs: LSST cadence db, Sorcha/ASSIST ephemerides, HelioLinC, AASTeX"
	@echo "make data         download the stored outputs (validation/output) from Zenodo (DATA_URL)"
	@echo "make analysis     regenerate every figure and check every number in the paper (minutes)"
	@echo "make paper        build paper/main.pdf"
	@echo "make simulations  regenerate all simulated samples and comparisons from scratch (many hours; WORKERS=N)"
	@echo "make alerts       rerun the Rubin alert pipeline (needs the Fink transfers in alerts/; see alerts/fink/)"
	@echo "make dp2          rerun the DP2 pipeline (needs Rubin data rights; see alerts/dp2/README.md)"

install:
	$(PY) -m pip install -e .
	bash rust/build.sh

test:
	$(PY) -m pytest -q tests

model: data/digest2.model.csv
data/digest2.model.csv:
	$(PY) scripts/fetch_digest2_model.py

external:
	bash tools/fetch_external.sh

data:
	@test -n "$(DATA_URL)" || { echo "DATA_URL is not set: the Zenodo record is not published yet. Run 'make simulations' instead."; exit 1; }
	bash tools/fetch_data.sh "$(DATA_URL)"

# ---- the paper -------------------------------------------------------------------------------------------------------
numbers: model
	$(PY) analysis/numbers/check_sim_numbers.py
	@test -f $(DS42_EXTERNAL)/opsim/baseline_v5.3.7_10yrs.db && $(PY) analysis/numbers/cadence_arcs.py || echo "skip cadence_arcs (run make external)"

figures: model
	bash analysis/figures/run_all.sh

analysis: numbers figures

paper:
	bash tools/build_paper.sh

# ---- full regeneration -----------------------------------------------------------------------------------------------
simulations: model
	WORKERS=$(WORKERS) bash analysis/simulations.sh

alerts:
	bash alerts/run_alerts.sh

dp2:
	bash alerts/dp2/run_dp2.sh

clean:
	rm -f analysis/figures/*.png analysis/figures/*.pdf analysis/figures/*_provenance.json
