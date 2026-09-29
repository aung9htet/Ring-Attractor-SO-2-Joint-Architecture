# Equivalence report: vectorised model vs the legacy `SingleRingModel`

Date: 2026-09-29. Branch `blocks-refactor`, phase 2. NEST `HEAD@41892a5`, Python
3.8.10, image `aung9htet/ubuntu-20.04:tiago_ring_forward`, one thread.

The vectorised model (`tiago_ring_controller.nest.populations`,
`nest.single_ring`) is the same circuit as the frozen facade in
`legacy/single_ring.py`: two N=200 rings, two Fourier readouts (K=20), the
ridge-fitted warm/cold/left/right comparator, opponent gain populations and the
shifted ±1 feedback. Three engineering changes were made, each with its gate.

## 1. One `Connect` per projection

Weights are computed in NumPy (`math.ring.ring_weight_matrix` for the rings,
the artifact masks for the readouts, the fitted vectors for the comparator) and
handed to NEST as `all_to_all` matrices or `one_to_one` vectors. NEST wants
`(n_target, n_source)`; the transpose is done in one place, `connect_matrix`.

Gate (fake NEST, `test/test_vectorised_topology.py`): every Connect call of both
builders is expanded into `(pre, post, weight)` rows, node ids are relabelled to
`(population, index)`, and the row *sets* are compared. They are identical for
rings (N = 4, 8, 9, both variants), readouts, the comparator with its feature
projections, the gain populations with feedback (N = 10, 12, 16) and for the
whole N=200 model built from `config/`. The only additional synapses are the 400
generator → ring synapses (one per ring neuron, weight 4500, rate 0).

| | legacy facade | vectorised |
|---|---:|---:|
| `Connect` calls, full model | 98 630 | 31 |
| NEST connections after build | 98 630 | 99 030 (+400 generators) |
| build time, N=200, 1 thread (`scripts/benchmark_build.py`, min of 3) | 12.01 s | 0.041 s |

Exact spike parity is **not** reproduced, and was not expected: the connection
order changes floating-point summation of synaptic input, and the 400 extra
generator nodes change the per-thread RNG streams. The rate-profile gate below
is the equivalence criterion, and a new golden is pinned for the vectorised
model.

## 2. Build-time stimulus generators (plan B.4)

One `poisson_generator` per ring neuron, rate 0, `one_to_one` to the ring with
weight 4500 (`nest.populations.build_stimulus_generators`). A bump is a rate
change on a window (`set_bump`), cleared after `duration_ms` of loop time by the
cosim `GeneratorStimulusPort`. No `Simulate` is hidden inside injection
(`hidden_ms` is 0), and a bump is legal at any tick.

Finding: with this NEST build a `poisson_generator` reads its rate at
`Prepare` only. A rate set between `Run` calls has no effect (0 spikes in the
window over two ticks) until `Cleanup`/`Prepare` (29 spikes in the next tick);
`Simulate` per tick sees it because it prepares every call. `NestEngine.
recalibrate()` therefore issues `Cleanup`/`Prepare` after every rate change made
while prepared; the pair costs 0.2 ms against 5.8 ms for a 50 ms `Run`.

Gate (real NEST, `test_engine_with_generator_port_accepts_a_mid_trial_state_bump`):
trial start with goal 140 and state 60, six ticks, then a state bump at index
120 for 300 ms:

| | value |
|---|---:|
| r1 window spikes (115–125) during the six quiet ticks before | ≤ 3 |
| r1 window spikes during the six driven ticks | 209 |
| r1 centroid before → after | 60.0 → 104.4 |

The attractor followed the drive but had not reached 120 after 300 ms; how a
sustained proprioceptive stimulus should compete with the gain transport is the
phase 5 characterisation (`Encoder.mode`, rate, half width), not a phase 2
result.

Bump timing differs from the legacy protocol: the legacy facade injects the
goal for 50 ms and *then* the state for 50 ms (100 ms of hidden simulated time,
r2 settles before r1 is stimulated); the loop applies both bumps at tick 0 and
they run concurrently for the first 50 ms tick. The pinned golden below uses the
legacy sequential protocol so that the comparison is like for like.

## 3. Recorders

Measured at N=200 over 40 ticks of 50 ms (probe in the container):

| readout | cost per tick |
|---|---:|
| one `spike_recorder` per neuron, `NodeCollection.get("n_events")` | 0.54 ms (ring) / 1.63 ms for r1 + left + right |
| one `spike_recorder` per population, `events` since last tick + `bincount` | 0.10 ms |

Both are small against the 5.8 ms `Run`; the per-population recorder is faster
but its `events` read grows with trial length (plan B acceptance A2 asks for a
flat cost) and loses the per-neuron `n_events` deltas the raster and bump index
use. **Per-neuron recorders are kept**, created as one collection and connected
`one_to_one`.

## 4. Parameter specs

`blocks/params.py`: `ParamSpec` (type, default, bounds, choices, unit, doc) and
`ParamSchema` per primitive (Ring, FourierReadout, Homeostasis, Gain, Encoder,
Decoder, Joint, Goal). `resolve` rejects unknown names, coerces and bounds-checks
values and reports every problem of a block at once; `dumps`/`loads` round-trip
through JSON. Neuron parameter sets are checked against `GetDefaults(model)` at
build time when the backend provides it (`validate_neuron_parameters`); build
errors are `BuildError`s that name the population.

## 5. Threads and seeds

`build_single_ring_network(backend, seed=, local_num_threads=)` records both in
`describe()`. Determinism gate (`test_build_is_fast_and_deterministic`): two
builds with seed 13579 give identical per-neuron counts; seed 13580 differs.

## 6. Rate-profile gate and the new golden

Protocol (both models, seed 13579, one thread): r1 bump at 60 (half width 5,
200 Hz, 50 ms), then r2 bump at 140 (50 ms), then 300 ms. Counts are cumulative.

| quantity | legacy | vectorised | tolerance in test |
|---|---:|---:|---|
| r1 total spikes | 206 | 206 | ±15 % |
| r1 centroid | 59.94 | 60.12 | ±3 of 60, ±3 of each other |
| r2 total spikes | 215 | 216 | ±15 % |
| r2 centroid | 140.02 | 140.00 | ±3 |
| profile distance Σ\|Δcounts\| / total, r1 / r2 | | 0.049 / 0.051 | ≤ 0.25 |
| warm / cold decision spikes | 85 / 161 | 83 / 156 | ±35 % |
| left / right decision spikes | 19 / 122 | 18 / 120 | (reported) |
| left / right gain totals | 90 / 293 | 88 / 295 | same sign of right−left, ±35 % |

A "width at half height" metric was tried and dropped: it flips by ±6 neurons
when the peak count changes by one (r2: 11 vs 17 for visibly identical
plateaus), so the L1 profile distance is used instead.

The golden `test/golden/vectorised_single_ring_seed13579.json` pins the
vectorised per-neuron counts (r1, r2, left, right), the decision counts, the
totals and the centroids for this protocol; `test_pinned_counts_are_the_phase_two_golden`
checks them exactly. The legacy parity test `test_cosim_real_nest.py` (cosim
engine vs the legacy per-tick loop, exact) still passes with `nest_model="legacy"`.

## 7. What the runtime gained

`CosimConfig.nest_model = "legacy" | "vectorised"` (`run_cosim_trial.py --model`).
With `vectorised`, `make_nest_engine` builds the new model and installs the
`GeneratorStimulusPort`; `ProprioceptionTF(mode="continuous")` bumps are now
accepted every tick. Default stays `legacy` until phase 4 switches the runner
to graph files.
