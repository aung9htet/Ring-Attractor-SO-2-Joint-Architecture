# tiago_ring_controller (ROS 1 package)

The research code of the NEST ring-attractor controller for TIAGo's arm
joints. **How to run, calibrate, measure and record every version is in the
repository's top-level `README.md`**; this page only holds what is specific to
the package.

## Layout

| Path | Content |
| --- | --- |
| `src/tiago_ring_controller/` | The importable package: `blocks/` (parameterised blocks), `graph/` (graphs, templates, calibration), `cosim/` (the lock-stepped loop, engines, dashboard), `nest/` (vectorised NEST builders), `control/`, `evaluation/`, `training/`, `ros/`, `ui/`. Importing it needs neither NEST nor ROS |
| `scripts/` | The executables (what Catkin installs): `run_graph.py`, `validate_velocity.py`, `run_legacy_trials.py`, `results_report.py`, `record_video.py`, `calibrate_transport.py`, `fit_comparator.py`, `identify_actuator.py`, and smaller tools |
| `src/config/` | Scientific inputs: neuron parameters, fitted comparator and readout weights, calibrations. Pinned by hash and schema in `test/test_artifact_contracts.py` (`scripts/update_artifact_manifest.py` registers a new file) |
| `legacy/` | Every flat research script that produced the paper results, frozen (see `legacy/README.md` and the repository's `docs/legacy.md`) |
| `test/` | `unittest` suite: fakes for NEST and ROS, real-NEST gates that run only with the pinned NEST |
| `docs/` | The original model's documentation: `architecture.md`, `artifacts.md`, `paper_traceability.md`, `reproducibility.md`, `robot_safety.md` |
| `launch/`, `worlds/` | `tiago_ring_controller.launch` (arguments `world`, `gui`, `public_sim`, `arm`, `end_effector`, `ft_sensor`) |

## Things worth knowing about the code

- It uses NEST populations and ordinary NumPy linear algebra. It is
  NEF-inspired but not a Nengo/NEF implementation.
- The robot path is `right spikes - left spikes`, a dead zone, an exponential
  filter, an integer delay and separate positive and negative gains
  (`blocks/decoder.py`).
- The positional readout has `2K` positive and negative **sine** channels: 40
  readout neurons for K = 20. Its weights are analytic half-wave sine masks;
  the comparator ("homeostasis") weights are ridge-fitted on measured NEST
  spike counts.
- The legacy and `builders/` ring wirings agree for N = 100 and differ for
  other sizes; the block `Ring` has a `variant` parameter for it.
- Loaded homeostasis biases and several historical configuration fields are
  unused. They stay part of the artifact contract.
- The world file comes from `pal_gazebo_worlds` (overridden by the
  repository's `overrides/`), not from this package. It holds the table and
  the `/recording_camera/image_raw` camera.
- The robot's mobile base is not anchored. `cosim/gazebo_ros_engine.py` puts
  it back at its spawn pose before every trial and logs how far a trial pushed it.

## Build and test

The characterised environment is Ubuntu 20.04, ROS Noetic, Python 3.8 and
PyNEST `HEAD@41892a5` (the image pinned in the repository's `docker/image.env`).

```bash
cd /tiago_public_ws && catkin build tiago_ring_controller --no-deps && source devel/setup.bash
cd /tiago_public_ws/src/tiago_ring_controller
python3 -B -m unittest discover -s test -p 'test_*.py'
```

The suite uses the standard-library `unittest` runner, not pytest. From the
host: `docker/batch.bash nest -- 'python3 -B -m unittest discover -s test -p "test_*.py"'`.

## The legacy scripts

Run them from `legacy/` with that folder and the package on the path
(`legacy/config` links to `src/config`); configuration and output paths are
relative to the working directory:

```bash
cd /tiago_public_ws/src/tiago_ring_controller/legacy
export PYTHONPATH=$PWD:$PWD/../src:$PYTHONPATH
python3 single_ring.py                       # no robot
python3 single_joint_data_collector.py       # moves the simulated robot
```

Training scripts (`train_ring_model.py`, `train_homeostasis.py`,
`compositional_fourier_decoder.py`) overwrite their default artifact
destinations: run them only in a disposable copy. Robot-moving entry points
(`single_joint_data_collector.py`, `single_joint_pid_data_collector.py`,
`analysis_single_joint.py`, `calibrate_single_joint.py`, `demo_graphs.py`) are
for simulation, or hardware only after a safety review (`docs/robot_safety.md`).
To run the two collectors on chosen goals and score them like the other
systems, use `scripts/run_legacy_trials.py` (top-level README, section 3.4).

## Outputs

Runtime outputs go to the `--out` folder of the script that wrote them; the
legacy scripts write beneath `legacy/outputs/`, `legacy/collected_data/`,
`legacy/plots/`, `legacy/results_plots/` and `experiment_results/`. None of
these is tracked.
