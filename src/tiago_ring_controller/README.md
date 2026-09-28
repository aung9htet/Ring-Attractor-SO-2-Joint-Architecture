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

## Compatibility policy

The flat modules in `src/` are the research entrypoints and compatibility API.
Their class names, method signatures, ROS interfaces, configuration defaults,
artifact formats, and output paths are retained.  Reusable implementation is
being separated under `src/tiago_ring_controller/` behind those facades.

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
python3 -m unittest discover -s test -p 'test_*.py'
```

## Non-robot workflows

Run these from `src/` because legacy configuration and output paths are
working-directory-relative:

```bash
cd /tiago_public_ws/src/tiago_ring_controller/src
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

Robot-moving entrypoints must be run only in simulation or after an explicit
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

## Artifacts and results

Checked-in files under `src/config/` are scientific inputs and compatibility
artifacts.  Do not delete or rewrite apparently unused artifacts without a
separate provenance decision.  Characterization tests record their filenames,
hashes, NPZ key order, shapes, and dtypes.

Runtime and analysis outputs are written beneath `src/outputs/`,
`src/collected_data/`, `src/plots/`, `src/results_plots/`, `results/`, and the
package-adjacent `experiment_results/`, depending on the legacy entrypoint.

