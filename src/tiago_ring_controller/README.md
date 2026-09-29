# TIAGo spiking ring controller

This ROS 1 package contains the research code for a NEST-based ring-attractor
controller for revolute TIAGo joints.  The executable implementation includes
two distinct ring builders, analytic positional Fourier readouts, a
ridge-fitted homeostatic readout, opponent gain modulation, two-ring
orientation decoding, robot calibration and data collection, and figure
generation.

The code uses NEST populations and ordinary NumPy linear algebra.  It is
**NEF-inspired**, but it is not a Nengo/NEF implementation: there are no Nengo
ensembles, encoders, intercepts, gains, evaluation-point objects, or NEF
solvers.

## Layout and legacy policy (branch `blocks-refactor`)

- `src/tiago_ring_controller/` is the importable package (pure math, NEST
  builders, control, evaluation, ROS contracts, the `cosim/` loop).  It imports
  without NEST or ROS.
- `scripts/` holds the executables (`run_cosim_trial.py`, `cosim_gazebo_smoke.py`,
  `measure_legacy_tick.py`); these are what Catkin installs.
- `legacy/` holds every flat research script that produced the paper results,
  frozen and unmaintained (see `legacy/README.md`).  `main` and the tag
  `legacy-loop-baseline` keep them at their historical `src/` paths together with
  the byte-for-byte freeze tests, which this branch deletes.
- `src/config/` keeps the scientific inputs (neuron parameters, fitted weights,
  calibration); their hashes and schemas are still pinned by
  `test/test_artifact_contracts.py`.  Generated results (`src/outputs/`, `results/`,
  `outputs/`) are no longer tracked.

The block architecture that replaces the flat scripts is planned in the
repository's `docs/blocks/plan.md`; progress is logged in `docs/blocks/progress.md`.

Important observed behavior:

- The robot path uses `right_spike_count - left_spike_count`, followed by an
  exponential filter, integer delay, and separate positive/negative gains.
- The positional readout has `2K` positive/negative **sine** channels.  The
  checked-in N=200 model therefore has 40 readout neurons for K=20.
- `train_ring_model.py` generates analytic half-wave sine masks; it does not
  optimize those weights.
- `train_homeostasis.py` and the multi-ring decoder use ordinary ridge
  regression on measured NEST spike-count features.
- The legacy and `builders/` ring implementations are not interchangeable.
  They agree for the common N=100 even-size case but differ for other sizes.
- Loaded homeostasis biases and several historical configuration fields are
  currently unused.  They remain part of the artifact/configuration contract.

See `docs/architecture.md`, `docs/artifacts.md`, and
`docs/paper_traceability.md` for the detailed executable map.

## Environment

The characterized environment is Ubuntu 20.04, ROS Noetic, Python 3.8, and
PyNEST.  The Python workflows also use NumPy, Matplotlib, Pandas, SciPy,
Colorcet, OpenCV, and `cv_bridge` where relevant.

Build and test with Catkin:

```bash
cd /tiago_public_ws
catkin build tiago_ring_controller --no-deps
source devel/setup.bash
catkin run_tests tiago_ring_controller
catkin_test_results
```

The repository uses the standard-library `unittest` runner rather than pytest:

```bash
cd /tiago_public_ws/src/tiago_ring_controller
python3 -B -m unittest discover -s test -p 'test_*.py'
```

Without the launcher, the same suite runs in the image with the package
bind-mounted (the real-NEST tests need the pinned NEST, so this is the reference
run):

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e MPLBACKEND=Agg \
  -v "$PWD/src/tiago_ring_controller:/tiago_public_ws/src/tiago_ring_controller" \
  --entrypoint bash aung9htet/ubuntu-20.04:tiago_ring_forward -c 'source /opt/ros/noetic/setup.bash; \
  source /tiago_public_ws/devel/setup.bash; source /usr/local/nest/bin/nest_vars.sh; \
  export PYTHONPATH=/tiago_public_ws/src/tiago_ring_controller/src:$PYTHONPATH; \
  cd /tiago_public_ws/src/tiago_ring_controller && python3 -B -m unittest discover -s test -p "test_*.py"'
```

## Non-robot workflows (legacy scripts)

Run these from `legacy/` with the folder and the package on the path; the
configuration and output paths are working-directory-relative (`legacy/config`
links to `src/config`):

```bash
cd /tiago_public_ws/src/tiago_ring_controller/legacy
export PYTHONPATH=$PWD:$PWD/../src:$PYTHONPATH
python3 single_ring.py
python3 gain_modulation_analysis.py
python3 readout_mean_std_analysis.py
python3 multi_ring_component.py
```

Training scripts overwrite their default artifact destinations.  Run them only
in a disposable package copy when validating or regenerating artifacts:

```bash
python3 train_ring_model.py
python3 trainer/train_ring_model.py
python3 train_homeostasis.py
python3 compositional_fourier_decoder.py
```

## ROS and robot workflows

The launch interface is:

```bash
roslaunch tiago_ring_controller tiago_ring_controller.launch
```

Launch arguments are `world`, `gui`, `public_sim`, `arm`, `end_effector`, and
`ft_sensor`.  The included TIAGo launch stack currently resolves
`tiago_ring_controller.world` from `pal_gazebo_worlds`, not from this package.
That external world contains the `/recording_camera/image_raw` camera used by
the camera/demo scripts.

Robot-moving entrypoints (legacy scripts, same `legacy/` working directory and
`PYTHONPATH` as above) must be run only in simulation or after an explicit
hardware safety review:

```bash
python3 single_joint_data_collector.py
python3 single_joint_pid_data_collector.py
python3 analysis_single_joint.py
python3 calibrate_single_joint.py
python3 demo_graphs.py
```

`calibrate_single_joint.py` contains a manual joint-limit routine that can
command sustained motion.  It is not an automated regression test.  Additional
safety notes are in `docs/robot_safety.md`.

## Co-simulation loop (branch `cosim-loop`, experimental)

`src/tiago_ring_controller/cosim/` implements the NRP-style fixed-time-increment
loop described in the repository's `docs/cosim/plan-b-inhouse-loop.md`: engines
that own simulated time (`NestEngine`, `GazeboRosEngine`, and fakes), datapacks,
transceiver functions, and `FTILoop`. It wraps the unchanged `SingleRingModel`
and the existing `CommandState` / `build_receding_trajectory` contracts; the
legacy entrypoints above still run their own loops. Importing the package needs
neither NEST nor ROS.

```bash
cd /tiago_public_ws/src/tiago_ring_controller
python3 scripts/run_cosim_trial.py --engines fake --goal 0.6     # no simulators
python3 scripts/run_cosim_trial.py --engines nest --seed 13579   # real NEST, fake robot
python3 scripts/run_cosim_trial.py --engines full --goal 0.6     # NEST + Gazebo (simulation running)
python3 scripts/cosim_gazebo_smoke.py                            # lock-stepped Gazebo check
python3 scripts/run_cosim_trial.py --engines full --dashboard             # browser dashboard, http://localhost:8765/
python3 scripts/run_cosim_trial.py --engines full --goal 0.6 --monitor --monitor-hold   # Matplotlib window
```

The scripts resolve `src/config/` themselves and run from any working
directory. `--model legacy` (default) imports the frozen `SingleRingModel` facade
from `legacy/`; `--model vectorised` builds the same circuit through
`tiago_ring_controller.nest.single_ring` in about 40 ms instead of 12 s and uses
build-time Poisson generators, so proprioceptive bumps are accepted at every tick
(`--proprioception continuous`). The two are compared in the repository's
`docs/blocks/equivalence.md`; `scripts/benchmark_build.py` reproduces the numbers.

`--dashboard` serves a page (standard-library HTTP server, server-sent events)
that shows the ring live and starts or stops trials on request; it needs no
display inside the container.

`--monitor` opens a matplotlib window (state ring, rolling raster, gain counts,
joint angle versus goal) redrawn every tick; `--monitor-every N` redraws less
often, `--monitor-frames DIR` saves a PNG per redraw (works headless), and
`--monitor-hold` keeps the window open after the last trial. The monitor is a
loop observer: it reads the recorded datapacks and never changes the trial.
The window needs X11 inside the container, i.e. a launcher started *without*
`--headless`; the image's matplotlib defaults to Agg even with a display, so the
monitor forces QtAgg, Qt5Agg, TkAgg or GTK3Agg in turn (`--monitor-backend` picks
one explicitly). Without a display it says so and only `--monitor-frames` works.

Tests are `test/test_cosim_loop.py`, `test/test_cosim_engines.py` (fake NEST and a
fake ROS transport) and `test/test_cosim_real_nest.py` (parity with the legacy
per-tick loop; runs only with the pinned NEST).

## Artifacts and results

Checked-in files under `src/config/` are scientific inputs and compatibility
artifacts.  Do not delete or rewrite apparently unused artifacts without a
separate provenance decision.  Characterization tests record their filenames,
hashes, NPZ key order, shapes, and dtypes.

Runtime and analysis outputs are written beneath `legacy/outputs/`,
`legacy/collected_data/`, `legacy/plots/`, `legacy/results_plots/`, `results/`,
`src/outputs/` (co-simulation runs) and the package-adjacent
`experiment_results/`, depending on the entrypoint.  None of them is tracked on
this branch.

