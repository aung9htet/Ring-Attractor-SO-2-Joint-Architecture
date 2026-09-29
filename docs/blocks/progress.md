# Progress: block architecture (branch `blocks-refactor`)

Dated entries per phase of `plan.md`. Each entry records what changed, the gate
results with the actual command output, and what was not run.

## 2026-09-29 — Phase 1: legacy folder and test policy

Open questions of plan section 9 were taken as answered by the hand-over brief:
Q1 delete the freeze tests (D1 restated as not to re-open), Q2 accept new pinned
goldens after phase 2 (the brief says "keep that parity test passing until the
phase 2 re-baseline"), Q4 vanilla JS/SVG (D5 restated), Q5 no hardware joints in
this pass (plan non-goal "hardware use"; a per-block interlock is not designed).

### Changes

- `git mv` of every flat script, `builders/`, `builders_analysis/`, `helpers/`,
  `trainer/`, the recovered `__pycache__/` folders and the entrypoint guide
  (`src/README.md`) into `src/tiago_ring_controller/legacy/`. The scripts are
  byte-identical (rename-only diff). `legacy/config` is a symlink to `../src/config`
  so the historical `./config/...` defaults resolve when a script runs from
  `legacy/`; `legacy/README.md` carries the frozen notice and the run recipe
  (`PYTHONPATH=$PWD:$PWD/../src:$PYTHONPATH`, appended so NEST's own path survives).
  `src/Neurips_2026.pdf` stays where it is.
- Deleted the freeze tests and their manifests: `test_api_compatibility`,
  `test_environment_contracts`, `test_bytecode_provenance`,
  `test_packaging_contracts`; `api_manifest`, `import_graph_manifest`,
  `bytecode_manifest`, `orphan_bytecode_manifest`, `environment_manifest`,
  `checked_output_manifest`. `test_output_schemas` lost the PNG byte freeze, the
  checked-in `fourier_results.npz` check and the external-world hash; it keeps the
  configuration, calibration, local world and launch contracts. `test_ring_math`
  lost the two import-shim tests that ran `builders/ring_attractor.py` from `src/`
  without the package on the path (the shim now points at `legacy/`, where the
  package is not). `test_artifact_contracts` is scoped to `src/config/**` and
  `worlds/*.world`; the two golden manifests were pruned to those entries
  (34 → 23 hashed files, 32 → 22 schemas), nothing re-hashed.
- Kept tests repointed at `legacy/`: `test_control_ros_contracts`,
  `test_multi_ring_contracts`, `test_ring_math`, `test_evaluation_facade_plots`,
  `test_real_nest_smoke`, `test_cosim_real_nest`.
- Package: `config.legacy_root()`; `cosim/runner.make_nest_engine` imports
  `single_ring` from there; `scripts/measure_legacy_tick.py` likewise.
- Packaging: `setup.py` lists the package only (no `py_modules`); `CMakeLists.txt`
  installs `scripts/*.py` plus `legacy/record_experiments.py` (the wrist FT logger
  that `tiago_ring_controller.launch` starts); the devel-space `env-hooks/`
  PYTHONPATH hook for flat modules is gone. `package.xml` unchanged.
- Untracked generated results: `src/outputs/` (34 MB), `results/`, `outputs/`
  (incl. the nine `fourier_results.npz` copies) and the 23 recovered `.pyc` files
  inside the package. Files stay on disk; `.gitignore` rewritten (package
  bytecode ignored, legacy bytecode kept, legacy working-directory outputs ignored).
  `docker/check_recovery.py` will report these as differences, as documented.
- READMEs: package README describes the layout, the container test command and the
  legacy run recipe; cosim commands now run from the package root.

### Gates

Host (`/usr/bin/python3` 3.13, `-B`, `unittest discover`): before the change
178 tests, 66 failures + 1 error, all in the deleted freeze tests plus
`test_evaluation_facade_plots` (no `colorcet` on the host). After:

```
Ran 162 tests in 6.128s
FAILED (errors=1, skipped=2)      # test_evaluation_facade_plots: No module named 'colorcet' (host only, pre-existing)
```

Container (`aung9htet/ubuntu-20.04:tiago_ring_forward`, package bind-mounted,
`python3 -B -m unittest discover -s test -p "test_*.py"`), previously 178 tests
with 59 pre-existing inventory failures:

```
Ran 170 tests in 73.796s
OK
```

`scripts/run_cosim_trial.py --engines nest --seed 13579 --goal 0.6` from the
package root in the container (imports `SingleRingModel` from `legacy/`):

```
trial 1 timing: reset [nest 12.3 s, robot 0.0 s]; lead 0.06 s; 177 ticks in 1.3 s (7.5 ms/tick)
trial 1: goal=0.6000 start=0.0000 final=0.4628 |err|=0.1372 steps=177 stop=drive_settled tick_wall_ms(mean=7.54 max=17.28)
```

`cd legacy && PYTHONPATH=$PWD:$PWD/../src:$PYTHONPATH python3 -B single_ring.py`
in the container (100 s of simulated time, figures under `legacy/outputs/single_ring/`):

```
exit=0   real 16m29.729s  user 260m43.119s   (one 100 s run, 10 repeat runs, 10x10 goal sweep)
legacy/outputs/single_ring/: goal_distance_mean_std_across_runs.{csv,png}, raster.png,
  relative_goal_error_boxplot.png, relative_goal_error_per_run.csv, relative_goal_error_summary.csv,
  ring_activity_heatmap.png, ring_centroid_change_over_time.png
```

The first attempt failed with `No module named 'nest'` because `export
PYTHONPATH=$PWD:$PWD/../src` replaced the path that `nest_vars.sh` sets; the
READMEs append `:$PYTHONPATH` for that reason.

Not run: Gazebo (needs `./run_model_docker.bash` with the simulation launched; no
robot code changed in this phase) and the browser dashboard (no dashboard code
changed; its end-to-end test with the fakes is part of the suite above).

## 2026-09-29 — Phase 2: optimisation and robustness pass

Full report with the measured numbers: `equivalence.md`.

### Changes

- `math/ring.py`: `ring_weight_matrix` (the legacy per-synapse weights as a
  `(pre, post)` matrix, both variants).
- `nest/populations.py`: vectorised builders (`Population` = neuron collection +
  one recorder per neuron; ring, Fourier readout, decision circuit with feature
  projections, gain populations with cross-inhibition and shifted feedback,
  build-time Poisson generators with `set_bump` / `clear_stimulus`);
  `validate_neuron_parameters` against `GetDefaults`; `BuildError` names the
  population. The per-synapse builders the legacy scripts import are untouched.
- `nest/single_ring.py`: `load_single_ring_artifacts` (all `config/` inputs
  validated up front) and `build_single_ring_network` (seed and threads recorded,
  `describe()`, `set_bump` / `clear_bumps`, recorder lists in the facade's shape).
- `blocks/params.py`: `ParamSpec`, `ParamSchema`, schemas for the eight primitives,
  JSON round trip, all problems of a block reported at once.
- cosim: `GeneratorStimulusPort` (bumps at any tick, no hidden time, windows expire
  after `duration_ms` of loop time, `NestEngine.recalibrate()` = Cleanup/Prepare
  after a rate change because NEST reads generator rates at Prepare);
  `RingModelPorts.from_network`; `CosimConfig.nest_model` (`legacy` default,
  `vectorised`); `run_cosim_trial.py --model`; `scripts/benchmark_build.py`.
- Tests: `test_vectorised_topology.py` (fake-NEST connection-set equivalence per
  builder and for the whole N=200 model, 11 tests), `test_block_params.py` (5),
  generator-port tests in `test_cosim_engines.py` (3), `test_vectorised_real_nest.py`
  (rate-profile equivalence, determinism + build time, pinned golden, mid-trial
  bump; 4, container only). Golden: `test/golden/vectorised_single_ring_seed13579.json`.

### Gates

Host (`/usr/bin/python3` 3.13):

```
Ran 181 tests in 7.171s
FAILED (errors=1, skipped=3)      # test_evaluation_facade_plots: No module named 'colorcet' (host only, pre-existing)
```

Container (full suite; the pinned-golden, equivalence, determinism and mid-trial
tests ran, as did the legacy parity test `test_cosim_real_nest.py` unchanged):

```
Ran 193 tests in 87.411s
OK
EQUIVALENCE {'r1_l1': 0.049, 'r2_l1': 0.051, ...}
MIDTRIAL {'centroid_before': 60.0, 'centroid_after': 104.43, 'window_spikes': 209.0}
```

Build time N=200 (`scripts/benchmark_build.py`, container, 1 thread, min of 3):
legacy 12.01 s → vectorised 0.041 s (target was < 3 s). Per tick: `Run(50)` 5.8 ms,
readout of r1 + left + right 1.6 ms.

`run_cosim_trial.py --engines nest --model vectorised --seed 13579 --goal 0.6`:

```
trial 1 timing: reset [nest 0.1 s, robot 0.0 s]; lead 0.04 s; 203 ticks in 1.4 s (7.0 ms/tick)
trial 1: goal=0.6000 start=0.0000 final=0.5917 |err|=0.0083 steps=203 stop=drive_settled
```

Same with `--proprioception continuous` (a 200 Hz state bump every tick):

```
trial 1: goal=0.6000 start=0.0000 final=-0.0840 |err|=0.6840 steps=400 stop=max_steps
```

The continuous stimulus at the legacy rate pins the state bump to the measured
joint and the loop does not converge; this is exactly the phase 5 sweep
(`Encoder.mode`, rate, half width, dead band), not a defect of the port. Recorded
here so the phase 5 baseline is known.

Item 3 (recorders): measured both options, kept per-neuron recorders (flat cost,
per-neuron deltas); numbers in `equivalence.md`.

Not run: Gazebo (no robot code changed) and the browser dashboard (unchanged).

## 2026-09-29 — Phase 3: block API

### Changes

- `blocks/base.py`: `Block` (type name, `ParamSchema`, typed `Port`s, `build(ctx,
  inputs)`, `connect_late`, `counts_sources`, `describe_type`), `PortKind`
  SPIKES/SIGNAL, `PortRef` (`ring.stim`), `Connection`, `BuildContext` (backend,
  config dir, seed, threads, neuron-parameter loader, build log), `Composite`
  (`expand()` → sub-blocks, internal edges, port map). Spike inputs are
  *build-bound* (source built first: readout ← ring) or *late-bound* (wired
  after both exist, pattern owned by the target: `Ring.stim` ← encoder
  generators / gain feedback), which is how feedback cycles build.
- Primitives, each wrapping the phase-2 builders: `Ring`, `FourierReadout`
  (analytic masks by default, identical to the artifacts), `Homeostasis` (fitted
  artifact keyed by the readout's ring size), `Gain` (`connect_stim` implements
  the shifted ±1 feedback; the edge may override weight and margin), `Encoder`
  (generators created when connected to a ring; modes once / continuous /
  corrective with dead band; `drive`, `expire`, `drive_index`), `FeatureEncoder`
  (2K push-pull generators for a reference with no ring), `Decoder` (wraps
  `DriveControlCore`; `step` equals `MotorTF` sample for sample; sources
  gain_counts / centroid_velocity / decision_counts), `ProfileDecoder`
  (sawtooth / centroid / scalar ramp), `Joint`, `Goal` (constant or schedule),
  `Probe`, `SignedProduct` and `OutputRing` (wrap `nest/multi_ring.py`; one
  output ring per block, artifact key per block), `TaskGain` (port contract
  only; `build` raises). Composites `Transport` (Homeostasis + Gain) and
  `JointTriple` (T/B/A rings, two encoders, two transports). `blocks.REGISTRY`,
  `block_from_type`, `describe_types()` (the editor palette). Schema defaults now
  equal the config files (Homeostasis, Gain, readout, ring; tested).
- `graph/graph.py`: `Graph` (`add`, `connect`, composites expanded on `add`,
  `problems()`/`validate()` reporting everything at once: kinds, directions,
  required inputs, producer counts, unknown edge parameters, signal cycles,
  build-bound cycles; `build_order()`; `build(ctx)` → `BuiltGraph`).
- `cosim/graph_engine.py`: `GraphNestEngine` holds any graph: inputs
  `"<encoder>.angle"`, one output datapack per neural block (ring: per-neuron
  deltas, total, bump index, centroid; gain: left/right; comparator:
  warm/cold/left/right; conjunction/output rings: per-cell deltas), generator
  rate changes followed by Cleanup/Prepare, raster per ring, continue mode.
- Tests: `test_blocks_fake.py` (15: hand-built graph creates the vectorised
  network with the *same node creation sequence and the same Connect calls in
  the same order*; Transport composite equals the primitives; validation
  messages; block errors name the block; schema defaults vs config; registry
  descriptions; encoder modes and expiry; Decoder vs MotorTF; Goal/Joint/Probe/
  ProfileDecoder; SignedProduct + OutputRing on the fake with a temporary
  artifact; JointTriple expands to 12 blocks and builds; engine datapacks and
  recalibration), `test_graph_real_nest.py` (2, container).

### Gates

Phase 3 gate, container: the single-joint architecture built by hand from
blocks (no graph file) reproduces the phase-2 golden **exactly** (per-neuron
r1/r2/left/right counts and decision counts, seed 13579), because the block
build order yields the same NEST node ids as `build_single_ring_network`.
`GraphNestEngine` runs the graph with once-mode encoders and accepts a
continuous proprioceptive encoder mid-trial (window driven, recalibrations per
tick).

```
Ran 210 tests in 103.139s   (container, full suite; one failure fixed below)
Ran 2 tests in 0.286s  OK   (test_graph_real_nest.py after the fix)
```

The one failure was the test's own naive angle→index formula (62 ≠ 60); it now
inverts the encoder's collector mapping by search. Host (`/usr/bin/python3`
3.13): 196 tests, only the pre-existing `colorcet` import error.

Not run: Gazebo and the dashboard (unchanged).
