# kangaroo_ring_controller (library)

The Python package behind the scripts. Design and the reasons behind it: `docs/kangaroo/DESIGN.md`.

## One reach in code
```python
import rclpy
from kangaroo_ring_controller import save_trial
from kangaroo_ring_controller.arm import connect
from kangaroo_ring_controller.joint_setup import newest_setup
from kangaroo_ring_controller.ring import run_trial

setup = newest_setup('left', 1)              # JointSetup from the newest calibration.json of left j1
node, arms = connect('my_reach', ['left'])   # exits if the simulation is not running
arms['left'].reset(setup.start)              # the start pose the decoder was calibrated from
result = run_trial(arms['left'], setup, goal=0.5, seed=100)
rclpy.shutdown()
save_trial('ring_reach', setup, result)      # -> results/ring_reach/left_j1_seed100_<time>/trial.json
print(result['q_start'], result['q_final'], result['stop_reason'])
```
Requires ROS 2 sourced (`ring.py` imports NEST and rclpy) and runs inside the container.

## Modules
| Module | Main names | Needs sim | What to know |
|---|---|---|---|
| `joint_setup.py` | `JointSetup`, `load`, `save`, `newest_setup`, `uncalibrated` | no | Everything a trial runs under: side, joint, profile, decoder, ring range, start pose. Frozen; change with `dataclasses.replace` |
| `ring.py` | `RingDrive`, `run_trial` | yes | Port of TIAGo's collector loop. Building a ring takes 22-30 s; pass a prebuilt `RingDrive(setup, seed)` as `ring=` to skip it. `on_tick` returning True stops the reach |
| `settle.py` | `GuardedDriveCore` | no | TIAGo's decoder core; settle counted only after the drive first reaches the threshold |
| `arm.py` | `connect`, `ArmVelocityPublisher` (`hold`, `sleep`, `reset`, `move_to`) | yes | Velocity → receding-horizon trajectory. `reset` checks the pose was reached and retries |
| `calibration.py` | `calibrate`, `with_decoder`, `load_limits` | yes | TIAGo's online calibration (fit every 3 trials, β = 0.5) |
| `refit.py` | `refit`, `save_refit` | no | TIAGo's fit once over the last N trials of a calibration; the result is what the controller should load |
| `limits.py` | `measure_limits` | yes | Drives the joint to each stop from home |
| `evaluation.py` | `protocol`, `run`, `remaining`, `summary`, `verdict` | no | Multi-seed protocol and statistics; `run` takes the reach as a function, so it is tested without the robot |
| `plots.py` | `*_figure`, `relative_error`, `ratios` | no | Parsed JSON in, matplotlib `Figure` out; the caller saves |
| `gui.py`, `gui_page.py` | `Gui`, `make_server` | yes | HTTP server thread + main-thread ring loop; the page is plain HTML/canvas |
| `warm_spare.py` | `WarmSpare` | no | Holds at most one prebuilt ring, handed out only for the setup and seed it was built for |
| `ring_core.py` | `RING_PARAMS`, `RING_WEIGHTS` | no | Importing it puts the TIAGo ring core on `sys.path`; the only module that knows the TIAGo layout |
| `__init__.py` | `RESULTS`, `new_run_dir`, `save_trial` | no | The single `trial.json` writer; a new timestamped directory per run |

## Conventions
- Angles in rad, velocities in rad/s, ticks of 50 ms (`arm.DT`). Joint index 0-3 = `arm_<side>_1..4_joint`.
- Joint positions map linearly onto ring indices over the ring range; they are not wrapped.
- One NEST ring per process (NEST's kernel is global); rclpy and NEST stay in the main thread.
- Nothing writes into an existing results directory.
- Comments cite decision numbers (D…, F…, Q…) from the author's working log, not in the repo; `DESIGN.md` summarises them.
