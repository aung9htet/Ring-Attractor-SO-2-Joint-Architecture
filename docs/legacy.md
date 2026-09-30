# Legacy scripts and the frozen baseline

Everything about the original research code in one place. The maintained
system is described in the top-level `README.md`.

## What "legacy" is

The flat scripts that produced the paper results (`single_ring.py`,
`ring_attractor.py`, `ring_component.py`, `homeostasis.py`, `gain_modulation.py`,
the three robot workflows `single_joint_data_collector.py`,
`analysis_single_joint.py`, `calibrate_single_joint.py`, the trainers, analyses,
figure scripts, the multi-ring stack `multi_ring_component.py` and
`compositional_fourier_decoder.py`, `tiago_controller.py`, the camera and torque
recorders) together with `builders/`, `builders_analysis/`, `helpers/` and
`trainer/` and their recovered Python 3.8 bytecode.

- On `main` (and the tag `legacy-loop-baseline`) they live at
  `src/tiago_ring_controller/src/` with golden tests that freeze them byte for
  byte (API manifest, import graph, file modes, bytecode, checked-in outputs).
- On `blocks-refactor` they live unchanged in
  `src/tiago_ring_controller/legacy/`: frozen, unmaintained, not installed by
  Catkin (except `record_experiments.py`, which the launch file starts). The
  freeze tests were deleted there; the scientific inputs under `src/config/`
  keep their hash and schema contracts (`test/test_artifact_contracts.py`).
- The maintained runtime imports one thing from the folder: the `SingleRingModel`
  facade, used by `scripts/run_cosim_trial.py --model legacy` and by the parity
  tests that prove the new model reproduces the old one.

## Running a legacy script

From the `legacy/` folder, with the folder and the package on the path;
`legacy/config` is a symlink to `../src/config`, so the historical
`./config/...` defaults resolve, and outputs land in `legacy/outputs/` (ignored
by git):

```bash
cd /tiago_public_ws/src/tiago_ring_controller/legacy      # inside the container
export PYTHONPATH=$PWD:$PWD/../src:$PYTHONPATH
python3 -B single_ring.py                                  # model only (about 16 min: one run, a 10-run repeat, a 10×10 goal sweep)
python3 -B multi_ring_component.py
python3 -B experiment_camera.py --headless --duration 10   # with the simulation running
```

The robot workflows (`single_joint_data_collector.py`, `analysis_single_joint.py`,
`calibrate_single_joint.py`, `demo_graphs.py`, `single_joint_pid_data_collector.py`)
move the arm with their own loops and their historical defaults (joint, trial
counts); `calibrate_single_joint.py` contains a manual joint-limit routine.
Treat them as documented in `src/tiago_ring_controller/docs/robot_safety.md`,
not as smoke tests. Training scripts overwrite checked-in artifact locations;
run them in a disposable copy. The original entrypoint guide is
`src/tiago_ring_controller/legacy/README.md`.

## What the legacy loop did and why it was replaced

`docs/cosim/assessment.md`: one hand-written, single-threaded tick loop copied
into three scripts, `rospy.sleep(0.05)` under simulated time, no shared clock
with Gazebo, robot state entering NEST once per trial, hidden simulated time
inside stimulus injection, readout cost growing with trial length. The
replacement is the co-simulation loop (`docs/cosim/plan-b-inhouse-loop.md`)
and, on top of it, the block graphs (`docs/blocks/`).

## How the new model relates to the old

- The block model builds the same synapses (proven on a recording fake NEST)
  through vectorised `Connect` calls and stimulates through build-time Poisson
  generators. Exact spike times differ from the legacy scripts (RNG streams,
  summation order); rate profiles agree within a few percent and a new golden is
  pinned. Report: `docs/blocks/equivalence.md`.
- The cosim runner on the legacy facade reproduces the legacy per-tick counts
  exactly (`test/test_cosim_real_nest.py`), and the graph runtime reproduces the
  cosim runner on the vectorised model tick for tick.
- The three legacy control profiles (calibration, analysis, collection:
  different spike scaling, joint-to-ring mapping and injection width) are kept
  as `tiago_ring_controller/control/profiles.py` and selected by the `Encoder`'s
  `mapping` parameter.
- Two behaviours of the original model surfaced during characterisation and are
  unchanged: the comparator produces no drive below about 18 ring indices of
  goal distance, and outcomes for some goals depend on the seed
  (`docs/blocks/feedback.md` §2).

## The frozen baseline and its checks

- `docs/recovery-manifest.json` and `docker/check_recovery.py` describe the
  files extracted from the original image (480 files with modes). On the host,
  `python3 docker/check_recovery.py` reports every difference from that
  extraction; after the move to `legacy/` and the untracking of generated
  results it reports those as differences by design, and never restores or
  changes anything.
- Generated results (`src/outputs/`, `results/`, `outputs/`, the
  `fourier_results.npz` copies, recorded videos and torque logs) are not
  tracked on `blocks-refactor`; `main` keeps every file.
- The original image already had three failing historical inventory tests; the
  baseline of the freeze tests on the recovered package was 178 tests with 59
  inventory failures identical on `main`.
- The model and artifact documentation of the original code:
  `src/tiago_ring_controller/docs/architecture.md` (ring implementations,
  control flow, profiles), `artifacts.md`, `paper_traceability.md`,
  `reproducibility.md`.
