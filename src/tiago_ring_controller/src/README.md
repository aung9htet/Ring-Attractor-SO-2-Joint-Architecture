# Source entrypoint guide

The files in this directory remain the public research scripts and import
compatibility layer.  Shared, side-effect-free implementation lives in the
`tiago_ring_controller` Python package beside them.

Run legacy scripts from this directory unless a script explicitly accepts an
absolute path; many defaults intentionally retain their historical
`./config/...` and `./outputs/...` resolution.

## Model-only entrypoints

- `ring_attractor.py`: legacy NEST ring and standalone raster.
- `builders_analysis/ring_attractor_analysis.py`: distinct builder-ring raster.
- `single_ring.py`: two rings, fitted comparator, opponent gain, and feedback.
- `gain_modulation_analysis.py`: feedforward gain sweeps; it does not add ring
  feedback.
- `readout_mean_std_analysis.py`: repeated warm/cold readout statistics.
- `multi_ring_component.py`: load and evaluate the two-ring orientation model.
- `multi_ring_sawtooth_scalar_decoder.py`: downstream scalar-ramp analysis.

## Weight generation and fitting

- `train_ring_model.py` and `trainer/train_ring_model.py` generate deterministic
  analytic positive/negative sine masks.  The two files use different ring
  stacks and must not be merged solely because their output formulas match.
- `train_homeostasis.py` fits a normalized sawtooth phase target using ordinary
  ridge regression on NEST count deltas.
- `compositional_fourier_decoder.py` calibrates the signed feature populations
  and ridge-fits three output-ring profiles.

These scripts write into checked-in artifact locations by default.  Use a
disposable copy for validation.

## Robot entrypoints

- `single_joint_data_collector.py`: neural-controller dataset.
- `single_joint_pid_data_collector.py`: PID comparison dataset.
- `analysis_single_joint.py`: neural-controller trial and NEST diagnostics.
- `calibrate_single_joint.py`: decoder gain/time-constant/delay calibration.
- `demo_graphs.py`: camera plus neural-activity snapshots.
- `experiment_camera.py`: camera viewer/headless topic check.
- `record_experiments.py`: wrist force/torque CSV logger launched by Catkin.

The calibration, analysis, and collection controllers deliberately retain
different preprocessing profiles:

| Workflow | Spike scale | Joint-to-ring map | Injection half-width |
|---|---:|---|---:|
| Calibration | `100/N` | 10% edge margin | 5 |
| Analysis | raw counts | full range | effectively 10 |
| Collection | raw counts | full range | 5 |

All three use the executed sign convention `right-left`.  Do not consolidate
these profiles without updating the characterization baseline as a deliberate
scientific change.

## Common commands

```bash
cd /tiago_public_ws/src/tiago_ring_controller/src

# Model-only
python3 single_ring.py
python3 gain_modulation_analysis.py
python3 readout_mean_std_analysis.py

# Offline visualization of already collected data
python3 single_joint_data_visualizer.py

# Camera smoke check after Gazebo is running
python3 experiment_camera.py --headless --duration 10
```

Robot-moving scripts are not routine regression commands.  Start the intended
TIAGo simulation/hardware stack, confirm topics/actions and joint state, and
follow `../docs/robot_safety.md` first.

## Configuration and outputs

- Fixed model parameters: `config/model_params/`
- Runtime joint calibration: `config/calibration/velocity_calibration.json`
- Fourier and multi-ring weights: `config/ring_decoding_weights/`
- Homeostasis weights: `config/homeostasis/`
- Neural/PID trial data: `collected_data/joint_<index>/`

The runtime calibration loaders use top-level decoder fields before legacy raw
or nested calibration-progress fields.  Unknown fields and historical records
must be retained on round trips.

