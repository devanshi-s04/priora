# D-19 — finite H extension (`h_extend_to`) and the 0/0 triage policy

*Two maintainer-approved features, implemented and validated together because
they are two answers to the same problem: the H-support wall of DESIGN §2.6 /
D-4, which `validation/NEOCP_PILOT.md` measured on real MPC discovery
astrometry as a refusal to score 11 of 499 tracklets — all 11 of which were
true NEOs.*

*Everything below is a statistic of a stored array. Regenerate with the three
drivers in this directory (`rescore_neocp.py`, `gate_delta.py`, `analyze.py`)
and the gate command in §4.*

---

## 0. What was built

**Feature 1 — `Digest2Histogram(h_extend_to: float | None = None)`.** When set
(canonical value `30.0`), every (q, e, i) cell gains one **synthetic 19th H
bin** `[25.5, h_extend_to)` whose object count is a flat continuation of the
top bin's per-unit-H density,

```
counts[..., 18] = counts[..., 17] / 1.5 * (h_extend_to - 25.5)
```

(1.5 mag being the width of the model's top bin `[24, 25.5)`), with the density
exactly zero at and above `h_extend_to`. This is still D-4 hard truncation —
just at a declared boundary further out — and, crucially, the extension carries
**finite** mass. Threaded through `neo_other_pair` / `from_classes`, added to
`backend._PAIRED_MODE_FLAGS` and to the model fingerprint, carried into the
Rust `CoreModel`, and exposed to the WP6 harness as `--h-extend-to`.

**Feature 2 — the 0/0 triage policy.** In `classifier._posterior`, the shared
post-processing both backends call: when a 0/0 outcome is **caused by the H
support running out** — a non-empty admissible region, at least one evaluable
node, and at least 95% of the region's *measure* at or above the pair's H
support end — return `p_neo = 1.0`, `log_lr = nan`, and the new status
`"fully_truncated_flagged"`, with a warning that says in terms that this is a
triage **policy assignment** and not a posterior. Every other route to 0/0
returns `nan`: `n_nodes == 0` as `"empty_region"`, and a zero with any other
mechanism as `"both_zero"`. The mechanism gate is the adversarial review's
MAJOR 2 fix and is derived in §2.

*Feature 1 passed adversarial review unchanged and is reported here as it was
accepted; §1 is unmodified apart from the secondary items of §1.7. Feature 2's
§2 is a rewrite.*

---

## 1. Construction and guarantees — the evidence

### 1.1 Why the extension and not the clamp

| | D-14 clamp (`h_top_bin_infinite`) | D-19 extension (`h_extend_to`) |
|---|---|---|
| density above 25.5 | top bin's constant, to H = inf | top bin's constant, to `h_extend_to`, then 0 |
| integral | divergent | finite |
| `sample()` | raises | works, no special case |
| gate-eligible | **no** (no generative counterpart) | **yes** |
| composes with `h_interpolation="pchip"` | no (ValueError) | **yes** |
| `total_count` | not the integral of the density | is the integral of the density |
| status | digest2-emulation only (WP7 same-inputs) | operational + modelling mode |

The two are **mutually exclusive**: constructing both raises `ValueError` at
all three entry points and in both kwarg orders
(`tests/test_population.py::test_h_extend_to_and_the_d14_clamp_are_mutually_exclusive`,
parameterized over `h_interpolation`). `h_extend_to` is validated finite and
`> 25.5`; `inf` is refused with a message naming the clamp as the thing being
asked for.

### 1.2 Guarantees, each with its test

| guarantee | how it is achieved | test |
|---|---|---|
| **Exact continuity at H = 25.5** (histogram mode) | the synthetic bin's density-table entry is a bit-for-bit **copy** of the top bin's, not a recomputation | `test_extension_density_is_exactly_continuous_at_25_5` — `np.array_equal` on the whole slab, and `density(nextafter(25.5,0)) == density(25.5)` over every cell centre |
| **Everything below 25.5 bit-identical to the unextended model** (histogram mode) | the extended bin-volume array is `concatenate((BIN_VOLUME, dq de di dH_ext))`, so the first 18 slabs are the *same numbers*; the first 18 count slabs are the input counts | `test_extension_leaves_the_model_below_the_wall_bit_identical[none]` — `array_equal` on 200 000 random probes |
| **...with one qualifier under pchip** | the extra cumulative knot turns the 25.5 knot from an *end* knot into an *interior* one, changing the cubic on `[24, 25.5)`. Mass in that bin is still exactly `counts[...,17]`; its shape is not | same test, `[pchip]` branch: asserts bit-identity for `H < 24` **and** asserts that `[24, 25.5)` genuinely differs; `test_extension_pchip_conserves_the_top_bin_mass_exactly` closes the mass half using the exact antiderivative |
| **Mass conservation incl. the synthetic bin, both modes** | histogram: by construction; pchip: PCHIP interpolates the *cumulative* count and is pinned at every knot | `test_extension_conserves_mass_including_the_synthetic_bin[none,pchip]` — midpoint-rule integral of `density()` x cell volume x (2 pi)^3 per H bin vs `counts`, rtol 2e-6, synthetic bin and total called out separately |
| **`support["H"] = (0, h_extend_to)`** | `h_edges[-1]` *is* `h_extend_to`, no branch needed | `test_extension_support_and_edges` |
| **Boundary semantics** | half-open `[lo, hi)` throughout | `test_extension_boundary_semantics` — `nextafter` on both sides of `h_extend_to`, `+inf`, `nan`, negative zero-minus; `bin_indices` asserted to agree with `density` about support |
| **`sample()` works and draws at the right rate** | the synthetic bin participates in the bin CDF like any other | `test_extended_component_is_samplable_at_the_expected_rate[none,pchip]` — draw rate above 25.5 matches `extension mass / total_count` to 5 sigma (All/NEO: expected 0.6261) |
| **Sampled histogram reproduces the extended model** | | `test_extended_sample_histogram_reproduces_the_extended_model` — 6e5 draws vs the extended counts, Poisson 5 sigma, synthetic bin included |
| **Rust parity, bit-identical, both modes** | Python builds the extended arrays before handoff; the kernel's H axis is data-driven | `test_extended_density_lookup_is_bit_exact[hist,pchip]` — `array_equal` on probes hitting every extended edge from both sides plus +/-inf and nan; `dims=[29,8,11,19]` asserted to have crossed over |
| **End-to-end parity** | | `test_extended_end_to_end_parity_on_the_gate_catalogue[hist,pchip]` — 120 gate_v0 tracklets: max abs dp **5.1e-16** (hist) / **4.4e-16** (pchip), max Lambda rel **4.8e-12** — the pre-existing D-15 level, no regression |
| **Pair-mismatch guard, every entry point** | `h_extend_to` added to `_PAIRED_MODE_FLAGS` | `test_extended_pair_mismatch_is_refused_at_every_entry_point` — `check_component_pair`, `core_model_for`, `classify` and `classify_many` on **both** backends, for on/off mismatch *and* 30-vs-40 reach mismatch |
| **Model-cache safety** | `h_extend_to` in the fingerprint | `test_two_reaches_do_not_share_a_cached_core_model` (see §1.4) |

### 1.3 `total_count` grows, and the growth is real mass

The NEO component keeps **55.8%** of its mass in the single top H bin (DESIGN
§2.6) against 1.4% for the "other" component, so at `h_extend_to = 30`
(width ratio 4.5/1.5 = 3):

| component | unextended | extended to 30 | factor |
|---|---:|---:|---:|
| All / NEO | 277 469 | **742 145** | **x2.6747** |
| All / other (SS - NEO) | 14 264 678 | **14 868 899** | **x1.0424** |
| mixture P(NEO) | 1.908% | **4.754%** | x2.49 |

This is stated in `total_count`'s docstring and in the harness module
docstring, because it is the one number a reader of a gate run must not be
surprised by. Unlike the clamp's, this mass is **normalizable**: `total_count`
is still exactly the integral of the density over the declared support, so the
mixture fraction built from it is the extended model's own mixture fraction —
which is precisely what makes the extended pair a legitimate §4.1 generator.

### 1.4 Two hazards found and closed during implementation

**(a) The fingerprint hazard — load-bearing, not belt-and-braces.** Because the
synthetic bin's density entry is a *copy* of the top bin's, two components
extended to `h_extend_to = 30` and `= 40` have **bit-identical density tables
of identical shape** and differ only in `h_edges[-1]` — which
`backend._fingerprint` does not hash. Without `h_extend_to` in the key the
second pair would silently reuse the first's cached `CoreModel` and score every
H in [30, 40) as out of support. `_fingerprint` now carries
`repr(h_extend_to)`, and `test_two_reaches_do_not_share_a_cached_core_model`
asserts the premise (tables identical) *and* the fix (fingerprints differ,
handles differ, and the 40-model is non-zero at H = 32 where the 30-model is
zero).

**(b) The pchip curve in the extension is flat — but as a derived fact, not a
promise.** The brief allowed for the reconstruction not being flat inside
`[25.5, X)`. Measurement says it *is* flat, to floating-point noise, and there
is a one-line proof: the extension interval's secant is
`[counts[17]*(w/1.5)]/w = counts[17]/1.5`, exactly the top bin's secant, so
PCHIP's harmonic mean at the 25.5 knot and its three-point end slope at the `X`
knot both return that common value, the two Hermite slopes equal the secant,
and the quadratic and cubic coefficients vanish (measured `b ~ 2e-13`,
`c = 0`). Consequence, visible in §3 and §5: **extended+pchip and
extended+histogram differ only on `[24, 25.5)`, not above it.** This is
documented as a derived observation rather than a guarantee, so that a future
extension law with a non-flat continuation does not silently break a promise
the module never made.

### 1.5 The Rust side needed no kernel change — and that is a measurement

`rust/src/model.rs` already takes `dims` and the four edge arrays as data;
`CoreModel::new`'s "finite and strictly increasing" edge check is what rejects a
nonsensical extension. So the extension crosses the boundary as nothing more
than `dims[3] == 19` with a 20-long `h_edges`. Rather than assert that, it is
tested: a new cargo test
(`model::tests::extended_h_axis_needs_no_kernel_special_case`) builds the axis
exactly as the glue hands it over and checks exact continuity at the old
boundary, flatness across the synthetic bin, and hard zero at/above the new
one; the Python parity tests check bit-identity against
`Digest2Histogram.density` on the real model in both H modes. Doc headers in
`model.rs` and `lib.rs` record why no change was needed.

### 1.6 The one documented imprecision

Exact continuity is bought by copying the density-table entry; the synthetic
bin's `count / (volume * (2 pi)^3)` then agrees with its stored density only to
**<= 1 ulp** (worst 3.5e-16 relative over the shipped model; bit-equal in 296
of the 546 non-empty NEO cells, which is why "to 1 ulp" was not accepted as
"by construction"). The residual is asserted rather than hidden
(`test_extension_count_and_density_agree_to_one_ulp`, bound 1e-15) and is
explained in the module docstring. Nothing else in the construction is
approximate.

### 1.7 Secondary items from the review

| item | change |
|---|---|
| MINOR 7 | `_extend_h_axis` now checks the extended counts are finite and names the cause; `h_extend_to` is capped at a declared `H_EXTEND_TO_MAX = 100` mag (~70x the model's own H range, and implied H only reaches ~34 at V = 24), so a mistyped exponent or a unit error is refused with the mass-inflation factor quoted rather than producing an `inf` density that surfaces much later as "table must be finite". |
| MINOR 8 | The fingerprint now hashes **all four bin-edge arrays in full** (`_edges_key`, ~70 doubles, so sampled-stride caveats do not apply). This closes the whole class rather than the 30-vs-40 instance: `h_extend_to` is a `Digest2Histogram` attribute, and a custom component — or a future mode — can move an edge without touching it. |
| MINOR 9 | `measure_fraction_H_above` and `posterior_mass_where` no longer imply 25.5; both point at `classifier.h_support_end(neo, other)` and the example uses it. The harness's `H_max^model` is now single-sourced from `h_support_end` (it read `other.support` alone before — identical in v0, but not the pair-level definition), the "support ends at H = 25.5" summary line and module docstring are mode-aware, the summary prints the boundary next to the §2.6(b) fractions, and `rust/README.md` documents the result fields including `n_h_at_or_above_support`, `status` and `policy_assigned`. |
| MINOR 10 | `tests/test_backend_parity.py::_compare` and `validation/backend_parity.py` now compare `status`, `policy_assigned` and a NaN-aware `log_lr` between backends, and both are verdict criteria rather than readouts. |
| MINOR 11 | `h_extend_to` must be a real number: a `str` (`"30"`, or a `"none"` pasted from `h_interpolation`) and a `bool` are refused by name instead of sailing through `float()` or raising a bare `ValueError` from it. |
| MINOR 12 | The exact-zero-Lambda assumption is documented where the trigger lives, with the measured margin (min non-zero density 3.2e-2 / 3.6e-5, ~300 orders above denormal) and the v1 condition that would invalidate it. See §2.4. |
| MINOR 14 | The pchip `[24, 25.5)` change is now quantified two ways in the module docstring: relative to the bin's **mean** density (the scale-free measure) median 5.4% / p90 15.9% / max 93.5% for NEO and 8.6% / 25.5% / 93.5% for "other"; relative to the *pointwise* unextended density 5.6% / 15.7% / **2222%**, whose tail is an artifact of the denominator approaching a zero of the unextended derivative. Mass is exactly preserved either way. **This does not reproduce the review's "max 250% relative"** — see §7.8. |
| NOTE 13 | `measure_fraction_H_above`'s docstring now carries an explicit warning that it **cannot** detect a policy row (it counts NaN-H nodes as "not above", so an all-masked tracklet reads 0.0), and names the stored triple as the detector. |

---

## 2. The policy, as implemented — **revised after adversarial review**

### 2.1 What the review found, and what changed

The first implementation triggered on the **symptom**: `n_nodes > 0` and both
Lambdas exactly zero. The review showed that condition is far too broad and
constructed two classes of false positive, both of which returned `p = 1.0`
*with a warning asserting support truncation that had not occurred*:

| class | geometry | n_nodes | implied H range | measure frac >= 25.5 | status (before) | status (after) |
|---|---|---:|---|---:|---|---|
| hole in the model, dense grid | V=6.0, 0.001 deg/day, default rho window | 29900 | [4.56, 15.99] | 0.000000 | `fully_truncated_flagged`, p = 1.0 | `both_zero`, p = nan |
| hole in the model, distant sliver | V=6.0, 0.005 deg/day, rho in [1, 100] AU | 50 | [4.50, 4.50] | 0.000000 | `fully_truncated_flagged`, p = 1.0 | `both_zero`, p = nan |
| implied H below zero | V=14.0, 0.01 deg/day, rho in [50, 100] AU | 15100 | [-6.02, -5.27] | 0.000000 | `fully_truncated_flagged`, p = 1.0 | `both_zero`, p = nan |

Neither is the DESIGN §2.6 wall. In the first, every implied H is comfortably
**inside** [0, 25.5) and the model simply holds no object that bright at those
geometries — nothing was truncated; the model said "no". In the second the
region is **below** the support (implied H < 0), which is also truncation but
at the other end, where the empirical argument inverts: absurdly large and
bright is the opposite of small and close. The class is not rare — a coarse
sweep over V in {6, 8, 10, 12} x 4 rates x 4 rho windows produced 24 hits.

### 2.2 The trigger, precisely stated

The policy now fires **iff all four of the following hold**:

1. `n_nodes > 0` — the admissible region is non-empty;
2. `Lambda_NEO + Lambda_other <= 0.0` — exactly, no tolerance (see §2.4);
3. `n_h_finite > 0` — at least one node has finite implied elements **and**
   finite implied H, i.e. the density was actually evaluated somewhere;
4. `w_h_at_or_above_support / w_h_finite >= H_TRUNCATION_MEASURE_MIN`
   (= **0.95**) — of the quadrature **measure** carried by those evaluable
   nodes, at least 95% lies at or above the pair's H support end,
   `classifier.h_support_end(neo, other)` = `max` over the pair of
   (`+inf` if that component clamps, else its last H edge; so 25.5 for D-4,
   `h_extend_to` under D-19, `+inf` under D-14).

Anything else returns `p_neo = nan` with status **`"both_zero"`** — which is
therefore no longer unreachable and has been un-retired — and a warning that
says what did *not* happen and lists the three candidate causes. Condition 3
failing (`n_h_finite == 0`, every node masked as unbound or non-finite) is a
**deliberate** non-trigger and documented as such: a region with no evaluable
point carries no evidence about where it sits relative to the support.

The mechanism evidence reaches the shared post-processing as three scalars
(`n_h_finite`, `w_h_finite`, `w_h_at_or_above_support`) that **both** backends
accumulate while walking the grid. In the Rust kernel they are new fields on
`Outcome` (`n_h_at_or_above_support`, `w_h_finite`,
`w_h_at_or_above_support`), emitted in the result dict **including in
`diagnostics=False` mode** — which is the mode every production batch runs in,
and the reason this had to be a counter rather than a pass over `node_H`.

### 2.3 Why 0.95 and not 1.0 — the tension, and the measurement

The review's literal prescription was "every finite-H node at/above the support
end", i.e. a fraction of exactly 1. Applied to the real data that **loses 5 of
the 11 tracklets the policy exists for**:

### The D-19 trigger's mechanism evidence on these 11

| designation | measure frac >= support end | node-count frac | min implied H | n evaluable nodes | fires at 0.95? | fires at 1.0? |
|---|---:|---:|---:|---:|:--:|:--:|
| 2019 TK | 0.986203 | 0.966292 | 25.400 | 8900 | yes | **NO** |
| 2019 UM2 | 1.000000 | 1.000000 | 25.827 | 14200 | yes | yes |
| 2019 VH | 1.000000 | 1.000000 | 25.809 | 10900 | yes | yes |
| 2020 GO | 1.000000 | 1.000000 | 26.374 | 10450 | yes | yes |
| 2020 TR6 | 1.000000 | 1.000000 | 26.296 | 13350 | yes | yes |
| 2020 UT | 1.000000 | 1.000000 | 25.892 | 14250 | yes | yes |
| 2021 BO2 | 0.990791 | 0.980469 | 25.416 | 12800 | yes | **NO** |
| 2021 JX5 | 0.996763 | 0.990291 | 25.471 | 10300 | yes | **NO** |
| 2021 KQ1 | 0.999418 | 0.996183 | 25.500 | 13100 | yes | **NO** |
| 2021 TG10 | 1.000000 | 1.000000 | 26.485 | 12450 | yes | yes |
| 2021 UY | 0.985628 | 0.966667 | 25.430 | 7500 | yes | **NO** |

Minimum measure fraction over the 11: **0.985628**; **5 of 11 are below 1.0**, which is why the criterion is 0.95 and not 1.0.


Five of the 11 have a thin sliver of nodes just *below* 25.5 — minimum implied
H 25.40–25.50, i.e. still inside the top bin [24, 25.5) — which happen to fall
in **empty** (q, e, i) cells. A strict `== 1` rule reads those slivers as "part
of the region is inside the support", drops the tracklet to nan, and destroys
5/11 of the empirical basis D-19 rests on. So the two stated requirements
("fire only on the mechanism" and "the 11 must still flag") are **inconsistent
under a node-count-equals-1 reading**, and the report says so rather than
quietly satisfying one.

The measure-weighted criterion resolves it with an enormous margin, because the
two populations are **totally separated**:

| population | measure frac at/above the support end |
|---|---|
| the 11 real NEOCP wall tracklets | **0.985628 … 1.000000** |
| both false-positive classes (3 geometries) | **exactly 0.000000** |

Any constant in (0, 0.9856] separates them perfectly; 0.95 sits inside that gap
with the whole interval to spare on the false-positive side and 3.5% on the
true side. It is a **declared constant**
(`classifier.H_TRUNCATION_MEASURE_MIN`) with the measurement written next to
it, and a unit test asserts it stays inside the gap.

Two further reasons the criterion is measure-weighted rather than a node count:

* **it is the pre-existing definition of this population.** NEOCP_PILOT.md
  already characterises these 11 by exactly this statistic and in exactly this
  way — *"the minimum measure fraction above the support among them is
  0.9856"*. A measure criterion below that number does not invent a category;
  it names the one the pilot already reported.
* **it is grid-independent.** The measure is what DESIGN §2.6(b) reports; a
  node-count fraction moves with node placement (the same 11 give
  0.9663 … 1.0000 by node count, against 0.9856 … 1.0000 by measure).

### 2.4 Why an exact-zero Lambda is a safe trigger *for this model*

Condition 2 tests `<= 0.0` with no tolerance. That is only sound because a
non-zero contribution from the digest2 histogram cannot underflow: the smallest
non-zero density the shipped pair can return is **3.2e-2 (NEO) / 3.6e-5
(other)** in the interface's units — some 300 orders of magnitude above the
double denormal floor — and the Jacobian and cell weights are
ordinary-magnitude numbers. So "exactly 0.0" really does mean "no node had any
density", never "the mass rounded away". A v1 component with genuinely tiny
per-cell counts (or one normalized to unit total) would break that reasoning
and the trigger would then need a relative floor. Recorded in
`_fully_truncated_in_H`'s docstring, because the assumption is invisible at the
call site.

### 2.5 Semantics, unchanged from the first implementation

* Implemented **once**, in `classifier._posterior` + `_warn_status` — the
  shared post-processing that `classify` (Python reference) and
  `backend._result_from_core` (Rust glue) both call. Verified rather than
  assumed: `test_both_backends_agree_exactly_on_the_policy` asserts equal
  `p_neo`, equal status, and **equal warning text**; the 11-tracklet regression
  test runs all four (backend x diagnostics) combinations and requires
  identical answers.
* `p_neo = 1.0`; **`log_lr = nan`, deliberately not `+inf`** — `log_lr` is the
  prior-free sufficient statistic (D-3) and there is no statistic here.
  Reporting `+inf` would assert an infinitely strong *likelihood* ratio, the
  exact false claim the policy must avoid.
* `ClassificationResult` gains `status: str = "ok"` and the derived property
  `policy_assigned`. Vocabulary: `"ok"`, `"fully_truncated_flagged"`,
  `"both_zero"`, `"empty_region"`, `"overflow"`, `"nan"`.
* The warning names it a "TRIAGE POLICY ASSIGNMENT, NOT a posterior", states
  that the likelihoods carry no information and that `log_lr` is `nan` for that
  reason, quotes the 95% measure condition, cites **D-19** and the 11-of-11
  measurement, names the status field, and says calibration analyses must
  separate these rows. The `"both_zero"` warning is now equally specific about
  what it is *not*.
* A likelihood-based `p = 1.0` and a policy `p = 1.0` are asserted
  distinguishable: same score, different `status`, `log_lr` `+inf` vs `nan`.
* `classify_many`'s docstring is rewritten around the changed signal (a policy
  1.0 is a **defined** score, so `isnan(p_neo)` no longer finds it;
  `isnan(log_lr)` finds both degenerate classes; `status == "ok"` is the filter
  for "likelihood-based").
* One existing test changed behaviour
  (`test_truncated_H_gives_zero_likelihoods_and_warns`), with the change and
  its reason written into its docstring.

### 2.6 Downstream accounting — every `isfinite` filter in the repository

A policy row is a **defined number with no likelihood content**, so every
statistic that means "likelihood-based" had to be taught to exclude it. Four
modules, all now under this task's ownership:

**`validation/self_consistency.py` (the WP6 gate harness).** Three *disjoint*
counts that partition the run — `n_likelihood_defined_p`,
`n_policy_assigned_p`, `n_undefined_p` — with an assert. Policy rows are
**EXCLUDED**, not diluted, from the reliability bins, the global-Z, the
odds-scale fit, the bootstrap thresholds, the ROC/AUC and the impossible-event
counters. The review measured why dilution is not an option: ~34 policy rows
(the real-data rate at N = 2e4) already make one of five
equal-expected-NEO-count bins vacuous — every member `p = 1` exactly — and at
the real-data 2.2% rate four of five go vacuous; separately, a policy row on a
true non-NEO contributes zero predicted against an observed failure and pushes
`max|z|` arbitrarily. `SCORES_DEFINED`'s criterion is now
`(undefined + policy) / scored`: counting policy rows as defined would let a
run pass the undefined-fraction criterion by assigning scores it cannot
justify, and at 2.2% that is exactly the FAIL/PASS margin on the default
tolerance. Policy false positives (policy row on a true non-NEO) get their own
reported count. The summary prints all three counts on distinct lines, always,
including the zero case (so a reader can tell "none happened" from "this
summary predates D-19"). **No chunk-schema change**: policy rows are recognised
from the recorded `(isfinite(p), n_nodes > 0, lambda_neo == lambda_other == 0)`
quadruple, so every cached chunk in the repository stays valid.

The `isfinite(p)` term in that quadruple is load-bearing and closes a
double-count on **pre-D-19** chunks: back then a wall tracklet was stored with
the same zero Lambdas and a *nan* `p_neo`, so the triple alone tagged it both
"undefined" and "policy-assigned" (the review verified the double-count on
`enriched_f01`'s one such row, and the summary sentence about them was
therefore false).

**`validation/neocp_pilot.py`.** Its H-wall detector was `~isfinite(p_all)`,
which after D-19 is **all-False** — silently erasing the 11/11 measurement the
policy cites as its justification. It is now the classifier's own trigger,
reconstructed from the stored columns: the Lambda triple **and**
`ar_frac_h_above >= H_TRUNCATION_MEASURE_MIN` (imported from the classifier,
not restated). That measure term is not decoration: without it the detector
would also catch the two non-wall 0/0 mechanisms. The detector gives the same
answer on both score-file vintages, and the `nonfinite_cases` /
`recovered_by_clamp` accounting is restored under it, with `recovered_by_clamp`
now meaning "the D-4 pair hit the wall and the clamped pair produced a real
likelihood score" (a policy assignment is not a recovery). The
`p_neo_exactly_one` comment claiming "Lambda_other underflowed" was false twice
over — the word was wrong (see §2.4) and the case now has two causes — and is
replaced by two counted sub-fields: one-sided posteriors vs policy
assignments. The ranking and calibration passes read likelihood-only views of
the score columns, which **reproduces the pilot's published numbers exactly**
(pred/obs 1.0055, ECE 2.2322e-02, AUC 0.9755768 on 488 tracklets — bit-for-bit)
because before D-19 those 11 were excluded automatically by `isfinite`.

**`validation/digest2_comparison.py`.** `load_run`'s keep-filter now drops
policy rows as well as non-finite ones, and reports both counts. This is
load-bearing for that module specifically: its premise is a *symmetric* drop —
a tracklet enters both codes' ROC or neither — and a policy row is `p = 1.0`
for priora with no likelihood behind it, on a class that is empirically ~100%
true NEO, so keeping it hands priora free true positives that digest2 has to
earn and biases ΔAUC in priora's favour by construction. Separately,
`ClampedScoreConfig` silently dropped `h_interpolation` and `h_extend_to` from
the run's own `score_config.json`, so the clamped pass would score a pchip or
D-19-extended run against a plain clamped histogram and call the difference
"boundary handling". Both keys are now carried and both **refuse** with the
reason, in `clamped_score_config` *and* in `_init_clamped_worker` — the clamp is
mutually exclusive with each of them, so those runs have no clamped control at
all. Verified: `gate_v0` still builds; `gate_v0_pchip` and `gate_v0_hext30`
raise.

**`validation/astrometric_noise.py`.** A new `mask_policy_rows` blanks policy
rows back to `nan` at every load point, which restores two things D-19 broke
silently: the **C3 break arm** (which counts tracklets that lose their score
under noise — the fully-truncated direction is one of the two ways that happens
and would otherwise have come back as a *score*, and C3's threshold is 1% of
tracklets, so losing a whole failure direction from it is not cosmetic), and
the calibration/ROC blocks (a 1.0 on a ~100%-NEO class flatters the noisy
column exactly where the study is trying to detect degradation). C3's written
criterion is updated to name both directions. The policy counts are reported
alongside. `h_interpolation`/`h_extend_to` are now expressible on its scoring
subcommands (`ScoreConfig` already carried them, and `sc.score` already
hard-failed on a mismatch, but there was no way to *ask* for either).
Re-verified: every published number is preserved (`c_hat` 0.9254739,
AUC 0.9822182 / 0.9812452, 0 undefined, 0 policy).

**`validation/backend_parity.py`.** Now compares `status` and a NaN-aware
`log_lr` between the backends, not just `p_neo` — both are verdict criteria,
not readouts, because both come out of the shared `_posterior` and a mismatch
means one backend fed it different mechanism evidence. The stored golden
reference predates the field, so its status is *reconstructed* from the
recorded triple and compared. It also threads the run's own
`h_interpolation`/`h_extend_to` into both the Rust and the Python pair, which
it previously ignored — a silent model mismatch that would have surfaced as a
spurious backend disagreement.

## 3. Validation 1 — the 11 previously unscorable NEOCP tracklets

Re-scored from the pilot's cached tracklets (read-only; `rescore_neocp.py`),
same grid (1200, 50), same rho window [0.01, 100] AU, Rust backend, `All`
variant. The pilot's 11 non-finite tracklets were re-identified from
`validation/output/neocp_pilot/scores.npz` and match NEOCP_PILOT.md's list
exactly.

**All 11 score under all four configurations, and all four give exactly 1.0** —
including under the *revised* (mechanism-gated) trigger, which is the D-19
acceptance criterion and is now pinned by
`tests/test_classifier.py::test_the_eleven_real_neocp_wall_tracklets_still_flag`
on both backends and in both diagnostics modes. §2.3 shows the mechanism
evidence tracklet by tracklet and why a strict `== 1` rule would have lost 5 of
them.

| designation | true NEO | q [AU] | V | rate [deg/d] | p (extended) | p (ext+pchip) | p (clamp) | p (policy-only) | status (policy-only) |
|---|:--:|---:|---:|---:|---:|---:|---:|---:|---|
| 2019 TK | yes | 0.852 | 19.54 | 10.27 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | `fully_truncated_flagged` |
| 2019 UM2 | yes | 1.018 | 21.24 | 4.15 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | `fully_truncated_flagged` |
| 2019 VH | yes | 0.977 | 20.79 | 6.82 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | `fully_truncated_flagged` |
| 2020 GO | yes | 0.938 | 20.74 | 9.26 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | `fully_truncated_flagged` |
| 2020 TR6 | yes | 0.807 | 21.31 | 4.49 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | `fully_truncated_flagged` |
| 2020 UT | yes | 1.043 | 21.35 | 3.95 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | `fully_truncated_flagged` |
| 2021 BO2 | yes | 0.932 | 20.73 | 5.64 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | `fully_truncated_flagged` |
| 2021 JX5 | yes | 1.033 | 19.53 | 8.58 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | `fully_truncated_flagged` |
| 2021 KQ1 | yes | 0.792 | 20.74 | 17.18 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | `fully_truncated_flagged` |
| 2021 TG10 | yes | 0.913 | 21.22 | 5.38 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | `fully_truncated_flagged` |
| 2021 UY | yes | 0.975 | 19.44 | 13.95 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | `fully_truncated_flagged` |

**The four routes to 1.0 are not the same thing, and the distinction is the
point of Feature 2.** Under the extension (and the clamp) the 1.0 is a
*genuine likelihood-based posterior*: `status == "ok"`, `log_lr = +inf`,
`Lambda_NEO > 0` and `Lambda_other == 0` — the "other" component really has no
support at these geometries, because no node combines q > 1.3 AU with an
in-support H. Under the unextended pair the 1.0 is the D-19 policy:
`Lambda_NEO = Lambda_other = 0`, `log_lr = nan`, `policy_assigned == True`.
Same number, different epistemic status, and the flag says which.

That the policy and the extension agree on all 11 is the strongest empirical
support the policy gets from this dataset: on the 11 tracklets where the model
was too small to answer, every bigger model answers 1.0.

### Lambda_NEO, and the residual admissible measure above H = 30

| designation | Lambda_NEO ext | Lambda_NEO clamp | rel. diff | Lambda_other (all four) | f(H>25.5) | **f(H>30)** | f(H>34) | max implied H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 TK | 0.252617 | 0.252617 | +0.00e+00 | 0 | 0.9862 | **0.0000** | 0.0000 | 28.389 |
| 2019 UM2 | 13.1478 | 13.2134 | -4.96e-03 | 0 | 1.0000 | **0.0676** | 0.0000 | 30.684 |
| 2019 VH | 1.0873 | 1.0873 | +0.00e+00 | 0 | 1.0000 | **0.0000** | 0.0000 | 29.473 |
| 2020 GO | 0.424606 | 0.424606 | +0.00e+00 | 0 | 1.0000 | **0.0000** | 0.0000 | 29.905 |
| 2020 TR6 | 88.5314 | 89.1138 | -6.54e-03 | 0 | 1.0000 | **0.1016** | 0.0000 | 30.856 |
| 2020 UT | 35.0469 | 35.2455 | -5.64e-03 | 0 | 1.0000 | **0.0769** | 0.0000 | 30.765 |
| 2021 BO2 | 11.4338 | 11.4338 | +0.00e+00 | 0 | 0.9908 | **0.0000** | 0.0000 | 29.755 |
| 2021 JX5 | 0.503748 | 0.503748 | +0.00e+00 | 0 | 0.9968 | **0.0000** | 0.0000 | 28.958 |
| 2021 KQ1 | 0.397062 | 0.397062 | +0.00e+00 | 0 | 0.9994 | **0.0000** | 0.0000 | 29.955 |
| 2021 TG10 | 4.90179 | 4.95506 | -1.07e-02 | 0 | 1.0000 | **0.0997** | 0.0000 | 30.725 |
| 2021 UY | 0.0713051 | 0.0713051 | +0.00e+00 | 0 | 0.9856 | **0.0000** | 0.0000 | 27.932 |

**The residual tail, stated plainly.** Extending to H = 30 does **not** cover
these tracklets' whole admissible region. Four of the 11 — 2019 UM2, 2020 TR6,
2020 UT, 2021 TG10 — still have **6.8% to 10.2%** of their region above H = 30
and therefore still hard-truncated; the other seven are fully inside the
extended support. The consequence is bounded and one-signed: Lambda_NEO comes
out **0.50% to 1.07% lower** than the clamp's (which never runs out of model),
and `p_neo` is unaffected because `Lambda_other = 0` regardless. Nothing above
H = 34 is reachable for any of them.

*Correction to an a-priori estimate in the task brief.* The brief expected
"implied H reaches ~34 at rho_min = 0.01". For these 11 the maximum implied H is
**27.9-30.9**, not 34, because they are bright (V = 19.4-21.4): implied H at
rho_min is approximately V + 10 - (phase term). H ~ 34 *is* reached — on the
*synthetic* gate_v0 catalogue, which runs to V = 24 (§5: max implied H =
**33.78**, p99 = 33.11). Both numbers are correct for their own sample; the 34
figure is a statement about the faint limit, not about the NEOCP wall cases.

*A choice worth flagging for the supervisor:* `h_extend_to = 30` does not fully
remove the wall even on the sample that motivated it. Extending to 34 would (at
the cost of a still larger NEO total, ~x6.7). 30 was the canonical value in the
brief and is what is validated here; the knob is a float and the machinery is
reach-agnostic (the 30-vs-40 fingerprint and pairing tests exist precisely so a
different reach is safe to adopt).

### All 499 pilot tracklets — the "changes nothing else" claim, measured

| pair | non-finite p | policy-assigned | median abs dp vs D-4 (488 likelihood-scored) | p90 | max | n(abs dp > 1e-3) |
|---|---:|---:|---:|---:|---:|---:|
| truncated (D-4 + policy) | 0 | **11** | — | — | — | — |
| extended | 0 | 0 | **2.262e-06** | 6.274e-04 | 9.079e-03 | 35 |
| extended + pchip | 0 | 0 | 1.252e-04 | 5.814e-03 | 2.880e-02 | 138 |
| clamped (D-14) | 0 | 0 | **2.262e-06** | 6.274e-04 | 9.083e-03 | 35 |

The extension's median abs dp against D-4 on the already-scorable tracklets is
**2.262e-06 — bit-for-bit the same statistic NEOCP_PILOT.md publishes for the
clamp (2.26e-06)**. Extended vs clamped over all 499: median exactly **0**,
p90 2.9e-08, max **8.6e-06**, none above 1e-3. On real data the extension
*is* the clamp, to five decimal places, without being non-normalizable.

`extended + pchip` is larger (median 1.25e-04) and that is the D-17 mode
change, not the extension: pchip reshapes the density over the whole H range.
Extended vs extended+pchip: median 1.27e-04, p90 5.5e-03, max 2.45e-02 — the
size D-17 records for the pchip mode change generally.

### Calibration on the pilot sample

| pair | predicted | observed | ratio | n | predicted excl. policy | observed excl. policy | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| truncated (D-4 + policy) | 306.6 | 305 | 1.005 | 499 | 295.6 | 294 | 1.005 |
| extended | 306.7 | 305 | 1.006 | 499 | 306.7 | 305 | 1.006 |
| extended + pchip | 306.5 | 305 | 1.005 | 499 | 306.5 | 305 | 1.005 |
| clamped | 306.7 | 305 | 1.006 | 499 | 306.7 | 305 | 1.006 |

All four sit at pred/obs ~ 1.005-1.006 on the full 499. Note the truncated
row's two columns: excluding the 11 policy rows reproduces the pilot's
published 295.6 / 294.0 exactly, which is the arithmetic check that the policy
accounting is doing what it says. NEOCP_PILOT.md's 1.01 was that
policy-excluded number; the whole-sample figure is now available because
nothing is refused.

---

## 4. Validation 2 — the extended-mode gate: **PASS**

The first §4.1 gate run possible in extended mode, and it is only possible
because the extended density is normalizable and samplable. Prior = generator =
extended model, so this tests **the extended model as a legitimate
population**, not the extension as a scoring hack.

```
OMP_NUM_THREADS=1 PRIORA_BACKEND=rust python -m validation.self_consistency run \
  --n 20000 --seed 20260901 --model data/digest2.model.csv --variant All \
  --h-extend-to 30 --n-rho 1200 --n-rhodot 50 --workers 90 --chunk 500 \
  --gate --boot 2000 --bins 5 --out validation/output/gate_v0_hext30
```

| | extended (this run) | pchip gate_v0_pchip (record) |
|---|---|---|
| N | 20000 | 20000 |
| mixture P(NEO) drawn | **0.04754** | 0.01908 |
| true NEOs | 170 | 179 |
| Z_tot | **+0.7119** (99th pct 2.552, p = 0.526) | -0.2349 |
| c_hat | **1.134 [0.803, 1.601]** | 0.958 [0.670, 1.370] |
| max abs z | 0.9727 vs 2.639 (p = 0.655) | — |
| AUC | 0.987668 | — |
| HPD KS D (nearest / exact) | 0.04798 / 0.05610 (limit 0.075) | — |
| estimator gap | 0.00812 | — |
| coverage / truth-density-positive | 1.000000 / 1.000000 | — |
| undefined p | 0 | 0 |
| **policy-assigned** | **0** | — |
| §2.6(b) measure above H_max^model (= **30** here) | median 0, mean 1.48e-03, frac>1e-3 0.343, frac>1e-2 0.0098, frac>1e-1 **0** | — |
| **VERDICT** | **OVERALL PASS** (all 7 verdicts) | PASS |

**The extended-mode gate was NOT re-run after the Feature 2 rework, and that
is a checked claim rather than a convenience.** The gate run of record has
`policy-assigned = 0`: no tracklet on the 20 000-tracklet synthetic catalogue
hit the D-19 branch in either mode (§5 confirms the same for the D-4, clamped
and extended+pchip re-scores of `gate_v0`). With a policy count of zero the
Feature 2 changes are *identically* no-ops on this run:

* the trigger narrowing can only ever turn a policy row into a `nan` — with no
  policy rows there is nothing to turn;
* the harness's exclusion of policy rows from the reliability bins, the
  global-Z, the odds fit and the ROC removes an empty set, so `Z_tot`, ĉ,
  `max|z|`, the bin table and the AUC are bit-identical inputs;
* `SCORES_DEFINED` now reads `(undefined + policy) / scored` = `(0 + 0) / 20000`
  = the same 0 it read before.

`tests/test_self_consistency.py::test_scores_defined_accounting_under_the_triage_policy`
asserts the zero-policy case explicitly, so "the exclusion changes nothing when
the count is 0" is a test rather than an argument.

And it is also a **direct measurement on this very run**: `analyze` was re-run
on the stored chunks after all Feature 2 changes and every statistic came back
bit-identical.

| statistic | before the rework | after | identical |
|---|---|---|:--:|
| `Z_tot` | 0.7118872857663574 | 0.7118872857663574 | yes |
| `max\|z\|` | 0.9727432685302909 | 0.9727432685302909 | yes |
| observed / expected NEO count | 170.0 / 166.13155538915635 | same | yes |
| `Z_tot` p-value | 0.5263349590678583 | same | yes |
| `max\|z\|` p-value | 0.6551724137931034 | same | yes |
| ĉ | 1.1336095693977044 | same | yes |
| AUC | 0.9876684168372342 | same | yes |
| the 5 reliability bins | — | — | identical row for row |
| all 7 verdicts + OVERALL | PASS | PASS | yes |

The regenerated `summary.txt` differs from the run of record in exactly two
ways, both additive: the three new score-accounting lines
(`likelihood-defined : 20000 (1)`, `policy-assigned : 0 (0)`,
`undefined P(NEO) : 0 (0)`, plus the line spelling out that `SCORES_DEFINED`
reads `(undefined + policy) / scored`), and the §2.6(b) header now naming the
boundary it measures against (`H_max^model = 30`) instead of leaving it
implicit. No number moved.

Notes a reviewer should have:

* The mixture fraction moved 1.908% -> 4.754%, as §1.3 predicts. The generator
  drew 6217 NEO candidates (vs 2432 in the histogram run) and accepted 170;
  the V-cut acceptance for the NEO class drops from 7.5% to 2.8%, because the
  extension's new objects are faint and mostly fail V < 24.
* **Zero policy assignments on the synthetic catalogue**, in extended mode *and*
  in the D-4 re-score of §5. The policy is a real-data phenomenon: the WP6
  generator's own observability cut (V < 24) never produces a tracklet whose
  entire region is above the wall. This is the honest reason the gate is not a
  test of Feature 2 — it is a test of Feature 1.
* The §2.6(b) diagnostic is now measured against the *extended* support, and
  the picture improves markedly: `frac > 1e-1` goes from 2.55% of tracklets
  (measured against 25.5, §5) to **0**.
* Threading discipline matches `--h-interpolation` exactly: one flag drives
  generator and scorer, `score()` hard-fails on a mismatch (including a
  30-vs-40 reach mismatch), the value is hashed into the chunk provenance
  stamp, and it is popped from the stamp payload at its `None` default so every
  cached chunk in the repository keeps its historical stamp.

---

## 5. Validation 3 — Delta-p vs the clamp on gate_v0, below the wall

`gate_delta.py` re-scores the **stored** gate_v0 catalogue (generated from the
*unextended* model, so this is a scoring comparison, not a gate) under all four
pairs, N = 20000, grid (1200, 50), Rust backend.

*Sanity check first:* the `truncated` column reproduces the gate_v0 run of
record (Python backend) to max abs dp = **7.7e-13**, and the
`measure_fraction_H_above(25.5)` column to 6.9e-18 — i.e. this driver is
scoring the same thing the record did, at the D-15 parity level.

| comparison | median abs dp | p90 | p99 | max | n(>1e-6) | n(>1e-3) | n(>1e-2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| extended - truncated | 3.256e-08 | 1.347e-06 | 5.224e-05 | 2.169e-02 | 2437 | 14 | 3 |
| clamped - truncated | 3.256e-08 | 1.349e-06 | 5.237e-05 | 2.169e-02 | 2438 | 14 | 3 |
| **extended - clamped** | **0.000e+00** | **3.587e-10** | 1.029e-08 | **3.000e-05** | 8 | **0** | **0** |
| extended_pchip - extended | 2.175e-05 | 2.229e-04 | 2.122e-03 | 6.408e-02 | 18031 | 440 | 38 |
| extended_pchip - truncated | 2.156e-05 | 2.246e-04 | 2.189e-03 | 6.408e-02 | 17942 | 444 | 36 |

**The extension and the clamp are the same answer on synthetic data too**:
median difference exactly zero, max 3.0e-05, nothing above 1e-3. And the
extension's difference from D-4 is one-signed: **16113 tracklets up, 0 down,
3887 exactly unchanged** — as it must be, since the extension only *adds*
non-negative density above 25.5.

### The prediction "dp ~ 0 where the measure above 25.5 is negligible", checked

| f(H>25.5) bin | n | median abs dp(ext-trunc) | max | median abs dp(ext-clamp) | max | median f(H>30) |
|---|---:|---:|---:|---:|---:|---:|
| **exactly 0** | 3837 | **0.000e+00** | **0.000e+00** | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (0, 1e-6] | 0 | — | — | — | — | — |
| (1e-6, 1e-4] | 46 | 8.169e-13 | 5.000e-11 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1e-4, 1e-3] | 457 | 1.091e-11 | 3.344e-08 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1e-3, 1e-2] | 2761 | 9.675e-10 | 1.592e-05 | 0.000e+00 | 2.751e-12 | 0.000e+00 |
| (1e-2, 1e-1] | 12390 | 1.407e-07 | 6.660e-03 | 1.622e-11 | 1.221e-05 | 1.323e-03 |
| > 1e-1 | 509 | 1.235e-05 | 2.169e-02 | 0.000e+00 | 3.000e-05 | 0.000e+00 |

All **3837** tracklets whose admissible region has *exactly zero* measure above
H = 25.5 have dp **exactly zero, bit-for-bit** — not "small", zero, because in
histogram mode the sub-25.5 model is bit-identical (§1.2). abs dp then rises
monotonically with the measure above the wall, spanning six orders of magnitude
across the bins. The prediction holds.

Two footnotes an adversarial reader will want:

* 50 further tracklets have dp exactly 0 *despite* non-zero measure above 25.5
  (up to f = 0.406). 49 of them are already saturated at `p = 1.0` under D-4
  (`Lambda_other = 0`), so extra NEO mass cannot move them; the remaining one
  sits at p = 2.3e-04 with f = 0.018 and evidently adds no density in either
  component over the cells its above-wall nodes occupy.
* The pchip column is *not* conditioned by the wall at all: 3768 of the 3837
  zero-measure tracklets move under `extended_pchip` (median 9.0e-05). That is
  D-17's mode change acting over the whole H range and is the correct
  behaviour; it is called out because it would otherwise look like the
  extension leaking below 25.5.

### Where they differ — the 12 largest shifts

| # | V | true NEO | f(H>25.5) | f(H>30) | p trunc | p ext | p clamp | dp(ext-trunc) |
|---|---:|:--:|---:|---:|---:|---:|---:|---:|
| 8117 | 23.77 | yes | 0.2458 | 0.0000 | 0.828427 | 0.850115 | 0.850115 | +2.169e-02 |
| 18909 | 23.13 | yes | 0.3071 | 0.0315 | 0.608180 | 0.624301 | 0.624307 | +1.612e-02 |
| 14294 | 23.56 | no | 0.3236 | 0.0338 | 0.149942 | 0.161643 | 0.161659 | +1.170e-02 |
| 13382 | 23.58 | no | 0.2201 | 0.0221 | 0.314913 | 0.324286 | 0.324291 | +9.373e-03 |
| 18374 | 23.42 | no | 0.0917 | 0.0114 | 0.039097 | 0.045757 | 0.045770 | +6.660e-03 |
| 10173 | 23.43 | no | 0.1677 | 0.0000 | 0.010746 | 0.016551 | 0.016551 | +5.805e-03 |
| 11106 | 20.11 | no | 0.0513 | 0.0000 | 0.003231 | 0.005916 | 0.005916 | +2.685e-03 |
| 4643 | 23.70 | no | 0.1354 | 0.0128 | 0.167618 | 0.169988 | 0.169988 | +2.370e-03 |
| 11521 | 23.19 | yes | 0.1941 | 0.0135 | 0.408123 | 0.410395 | 0.410396 | +2.272e-03 |
| 14412 | 23.85 | yes | 0.1244 | 0.0089 | 0.598019 | 0.600215 | 0.600215 | +2.196e-03 |
| 8418 | 22.80 | yes | 0.2128 | 0.0148 | 0.818433 | 0.820124 | 0.820124 | +1.691e-03 |
| 8455 | 22.33 | no | 0.0985 | 0.0000 | 0.001545 | 0.002979 | 0.002979 | +1.434e-03 |

The signature is unambiguous: **faint** (11 of 12 at V > 22.3, 9 of 12 at
V > 23), **a large fraction of the region above the wall** (f = 0.05-0.32), and
the shift is always **upward** and always **matched by the clamp to <~ 2e-05**.
These are the DESIGN §2.6 corner tracklets, and the extension moves them the
way the clamp does.

### Truncation diagnostics and ranking on gate_v0

* measure above **H = 25.5**: median 2.10e-02, mean 2.82e-02, frac > 1e-3 /
  1e-2 / 1e-1 = 0.783 / 0.645 / **0.0255**, max 0.406.
* measure above **H = 30** (the extension's own residual): median **0**, mean
  1.48e-03, frac > 1e-3 / 1e-2 / 1e-1 = 0.348 / 0.0096 / **0**, max 0.0655.
* max implied H over the catalogue **33.78**; p50/p90/p99 = 29.46 / 32.11 /
  33.11. So `h_extend_to = 30` removes most of the exposure but not all of it:
  ~1% of tracklets still have >1% of their region truncated, and none has more
  than 6.6%.
* ROC AUC, same tracklets, four models: truncated **0.981245**, extended
  **0.982246**, clamped **0.982246**, extended+pchip **0.982393**. The
  extension buys +0.0010 AUC over D-4 and is indistinguishable from the clamp
  (identical to six decimals).

---

## 6. Test and build summary

| suite | result |
|---|---|
| `pytest tests/` (default `auto` => Rust) | **661 passed, 1 skipped, 1 xfailed** (221 s) |
| `PRIORA_BACKEND=python pytest tests/` | **661 passed, 1 skipped, 1 xfailed** (243 s) |
| `cargo test --manifest-path rust/Cargo.toml` | **20 passed, 0 failed** (19 before D-19; +1 `extended_h_axis_needs_no_kernel_special_case`) |
| build | `rust/build.sh` (release), `priora._core 0.0.1` |

Tests added or reworked for the Feature 2 rework, on top of Feature 1's:

* `tests/test_classifier.py` — `_posterior`'s unit table rewritten for the
  three-scalar mechanism signature, including a **guard on the constant
  itself** (`0 < H_TRUNCATION_MEASURE_MIN <= 0.985628`, i.e. it must stay
  inside the measured separation gap); `h_support_end` (pair-level max, the
  clamp's `+inf`, the extended `30.0`, and the fail-safe `nan` for a component
  with no declared H support); the two false-positive classes as a
  parameterized regression over {3 geometries} x {2 backends} x {2 diagnostics
  modes}, asserting `nan`/`"both_zero"` **and** that the warning does not
  claim truncation; a companion test that pins the *geometry* of those cases
  (measure fraction above the support exactly 0.0) so the regression cannot go
  vacuous; the deliberate `n_h_finite == 0` non-trigger; and the 11 real NEOCP
  tracklets by designation, all four backend/diagnostics combinations.
* `tests/test_self_consistency.py` — the accounting test rewritten for the
  three disjoint counts and the changed `SCORES_DEFINED`; a new test that
  proves *exclusion* rather than dilution by comparing `analyze()` on a forged
  run against `analyze()` on the same run with those rows physically deleted
  (`global_z`, `max|z|`, expected/observed counts, AUC and the bin sizes must
  agree exactly); a pre-D-19 double-count regression (MINOR 6); and four
  downstream tests covering `digest2_comparison`'s keep-filter and its clamp
  refusal, `astrometric_noise.mask_policy_rows`, and the pilot's detector.
* `tests/test_backend_parity.py` — `_compare` now also asserts `status`,
  `policy_assigned` and a NaN-aware `log_lr`, so every one of the ~250 parity
  comparisons in that file carries the new contract.

`tests/test_digest2_comparison.py` is outside this task's ownership and needed
**no** change: making the Lambda columns *optional* in `load_run` (rather than
required, as the first cut had it) keeps every hand-built score table working
and is the better design anyway — see §7.9.

## 7. Disagreements, corrections and open choices

**7.1 The review's two stated requirements for Feature 2 are inconsistent as
literally written, and this is the report's main disagreement.** "Fire only when
every finite-H node lies at/above the support end (i.e. the fraction must be
1)" and "the 11 real wall tracklets must still flag under D-4" cannot both hold:
5 of the 11 sit at measure fraction 0.9856–0.9994 (node-count fraction
0.9663–0.9962), because a thin sliver of their nodes lands at implied H
25.40–25.50 — inside the top bin — in empty (q, e, i) cells. Measured, tracklet
by tracklet, in §2.3. The resolution taken is a **measure-weighted** criterion
at a declared 0.95, which satisfies both intents and is justified by a total
separation (the false positives are at *exactly* 0.0, so the admissible interval
for the constant is the whole of (0, 0.9856]). Setting the constant to 1.0
recovers the review's literal spec and costs 5 of the 11; the maintainer can
make that call by editing one named constant, and a unit test will flag it as
outside the measured gap.

**7.2 `h_extend_to = 30` does not fully repair the wall on the sample that
motivated it.** 4 of the 11 keep 6.8–10.2% of their region truncated; the effect
on those four is a 0.5–1.1% deficit in Λ_NEO versus the clamp and no effect on
`p_neo`. The knob is a float and the machinery is reach-agnostic (the 30-vs-40
fingerprint and pairing tests exist for exactly that), so this is a modelling
decision, not an implementation one, and it is not taken here.

**7.3 The brief's "implied H reaches ~34 at rho_min = 0.01" is right for the
faint limit, not for the NEOCP wall cases.** Those 11 are bright (V = 19.4–21.4)
and top out at implied H 27.9–30.9. H = 33.78 is reached on the V <= 24 synthetic
catalogue.

**7.4 The brief's "the pchip curve need not be flat inside the extension" is
satisfied but vacuous:** with the flat-continuation count the curve *is* flat,
provably (§1.4b). The allowance is kept as the guarantee and the flatness is
recorded as a derived observation, because the two claims are different and a
future extension law could separate them.

**7.5 The gate cannot test Feature 2, and did not need re-running.** Zero policy
assignments on 20 000 synthetic tracklets in every mode, because the WP6
observability cut (V < 24) never manufactures a fully-truncated tracklet.
Feature 2's evidence is the real-data 11/11 plus the unit tests; §4 gives the
checked argument for why the gate verdict is unaffected.

**7.6 `log_lr = nan` for policy rows remains a deliberate incompatibility**, and
the sweep through the four downstream modules (§2.6) is what makes it safe:
`isnan(p_neo)` no longer finds a degenerate tracklet, and `isfinite(log_lr)` now
rejects a defined score. All four modules now filter on the stored triple
instead. Anything outside this repository that consumes `scores.npz` needs the
same treatment.

**7.7 `ClassificationResult` gained a field.** `status` is keyword-defaulted
(`"ok"`), so positional construction and existing call sites are unaffected; it
does change the dataclass's `repr` and `==`.

**7.8 I cannot reproduce the review's "max 250% relative" for the pchip
`[24, 25.5)` change.** Measured two ways over every populated cell at 61 H
samples: relative to the bin's **mean** density (scale-free) median 5.4% / p90
15.9% / **max 93.5%**; relative to the *pointwise* unextended density 5.6% /
15.7% / **max 2222%**. Neither is 250%. The pointwise tail is an artifact of the
denominator approaching a zero of the unextended derivative, not of the
reconstruction. Both metrics and that explanation are now in the module
docstring; mass is exactly preserved either way. If the review's 250% came from
a third normalisation, it should be stated, because as written the claim is
neither of the two I can measure.

**7.9 The Lambda columns are *optional*, not required, in
`digest2_comparison.load_run`.** Requiring them (the first cut) broke 18 tests
in `test_digest2_comparison.py` — a file outside this task's ownership — because
that module is legitimately driven with hand-built score tables. They are now
loaded when present, and `RunData.policy_detectable` records whether the filter
could run at all. That distinction is load-bearing and is tested: on a table
without them the policy rows **do** leak through, because nothing in the table
distinguishes a policy 1.0 from a saturated posterior. "0 detected" is not "0
present", and the flag is which claim is being made.

**7.10 Not done, deliberately.** (a) The NEOCP pilot's three scoring passes are
still All / Unk / clamped: no extended pass was added, because that would change
its published outputs while the supervisor is writing D-19 and DESIGN around
them. Its `_init_worker` is unchanged in behaviour. (b) `validation/output/`
artifacts for the pilot and the noise study were **not** regenerated in place;
both were re-run into scratch copies to verify every published number is
preserved (pilot: pred/obs 1.0055, ECE 2.2322e-02, AUC 0.9755768, wall count 11
at 100% true NEO; noise: ĉ 0.9254739, AUC 0.9822182/0.9812452, 0 undefined,
0 policy). Promoting them is a supervisor decision. (c) `NEOCP_PILOT.md`'s
H-wall section still describes `p_neo = nan` for the 11; the code now assigns
them a flagged 1.0 and the module's regenerated REPORT.md says so. Doc
ownership is the supervisor's.

## 8. Artifacts

| path | what |
|---|---|
| `validation/output/gate_v0_hext30/` | the extended-mode gate run (summary.txt/json, plots, chunks) |
| `validation/output/h_extension/rescore_neocp.py` | NEOCP re-score driver (read-only on the pilot) |
| `validation/output/h_extension/neocp_rescore.json` | the 11 wall tracklets x 4 pairs, incl. the D-19 mechanism evidence per tracklet |
| `validation/output/h_extension/neocp_delta_all.json` | all 499 pilot tracklets x 4 pairs |
| `validation/output/h_extension/gate_delta.py` | gate_v0 scoring-mode comparison driver |
| `validation/output/h_extension/gate_delta.npz` | 20000 x 4 pairs + §2.6(b) columns + per-pair `status`/`policy_assigned` |
| `validation/output/h_extension/analyze.py` | regenerates every table above |
| `validation/output/h_extension/{gate_run,rescore,gate_delta}.log` | run logs |
