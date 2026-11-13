# ds42

**A Bayesian approach to prioritizing the follow-up of near-Earth objects.**
Code, analysis scripts and paper source for Singh, Jurić & Ivezić (AJ, submitted).

Given a single-night tracklet (two or more detections), ds42 gives the posterior probability that the object is
a near-Earth object (NEO). It integrates an absolute population model over the tracklet's admissible region.
Because it uses the population's absolute numbers, the score is a calibrated probability, not a ranking.
The classifier runs in Rust with a Python reference implementation, and takes the same ADES/MPC-80 input as digest2.

Everything in the paper can be regenerated from this repository:

```bash
conda env create -f environment.yml && conda activate ds42
make install        # ds42 (editable) + Rust kernel
make model          # digest2 population model from the MPC
make data           # stored simulation / NEOCP outputs (Zenodo), or: make simulations  (many hours)
make analysis       # every figure in paper/figures and a check of every number in the paper
make paper          # paper/main.pdf
```

`make analysis` regenerates each figure and checks each number quoted in the paper against its stored output.
A mismatch fails the check. `analysis/PROVENANCE.md` maps every figure, table and number in the paper to the
script and output that produce it.

## Layout

| Path | Contents |
|---|---|
| `src/ds42`, `rust/` | the classifier (Python package, Rust kernel with the closed-form Jacobian) |
| `tests/` | unit and regression tests (`make test`) |
| `docs/` | design notes, decisions, implementation and population-model notes; validation write-ups |
| `validation/` | simulation, digest2 comparison, astrometric noise, NEOCP and impactor pipelines (Secs. 5–8, 10) |
| `analysis/` | `figures/` (paper figures), `numbers/` (number checks), `simulations.sh` (full regeneration with the paper's seeds) |
| `alerts/` | Rubin alert test (Sec. 9): Fink transfers → HelioLinC tracklets → scores; `dp2/`: DP2 precovery test |
| `paper/` | `main.tex`, `bibliography.bib`, `figures/` |
| `data/` | downloaded inputs (not versioned; see `data/README.md`) |

## Data not included

* **Fink alert data.** Not redistributed. `alerts/fink/` gives each transfer's request, so the same data can be
  requested again. Then run `make alerts`.
* **Rubin DP2.** Available only to Rubin data-rights holders. Only the scripts are here (`alerts/dp2/README.md`).
* **Imminent-impactor simulation.** The Sorcha detections of imminent impactors in Sec. 10 were provided by Ian Chow,
  based on the population of Chow et al. (2026). They are available from him.

Without these inputs, `make analysis` still builds every simulation and NEOCP figure. It skips the alert and DP2
figures and says why.

## License

BSD 3-Clause; see `LICENSE`. Third-party tools are fetched by `make external` and keep their own licenses:
digest2 (MPC), HelioLinC, Sorcha / ASSIST and AASTeX.
