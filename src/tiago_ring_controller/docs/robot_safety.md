# Robot and simulation safety

The automated suite is deliberately non-moving. Import tests use fake NEST/ROS
boundaries, artifact tests are read-only, and no test publishes a trajectory,
starts Gazebo, calls `play_motion`, discovers physical joint limits, or rewrites
calibration.

## Motion-capable interfaces

| Interface | Type | Effect |
|---|---|---|
| `/arm_controller/command` | `trajectory_msgs/JointTrajectory` | Commands all seven arm joints. |
| `/gravity_compensation/arm_N_joint/command` | `std_msgs/Float64` | Publishes torque values per joint. |
| `play_motion` | `play_motion_msgs/PlayMotionAction` | Runs named whole-body motions. |

The reset action uses `motion_name="tiago_experiment_start_1"` with planning
enabled. Fixed subscriptions are `/joint_states` and `/wrist_ft`. Camera discovery
prefers `/recording_camera/image_raw` and may select another advertised image topic.

Absolute topic names bypass a surrounding ROS namespace. Namespacing a launch group
does not isolate the arm command publishers.

## Preserved control behavior

- Joint order is `arm_1_joint` through `arm_7_joint`.
- Default command cadence is 50 ms with a four-point receding horizon.
- Internal command state advances by the first point, not the horizon endpoint.
- Stop publishes three zero-velocity points.
- Neural workflows can run up to 400 steps and require ten consecutive samples
  under their threshold before declaring settled.
- Calibration can atomically rewrite `velocity_calibration.json` after a trial or
  fit update.
- Manual limit discovery intentionally moves a joint until it appears stationary;
  it has no independent collision or torque guard in this package.
- High-level robot classes can create FT/joint subscribers and CSV files as
  constructor side effects.

These are compatibility facts, not safety endorsements.

## Validation gates

### Gate 1: offline

- Golden hashes and all pure characterization tests pass.
- Mocked ROS traces match topic names, message types, joint order, point timing, and
  one-step state advancement.
- No calibration or experiment file changed.

### Gate 2: isolated Gazebo

- Confirm `ROS_MASTER_URI` and topic publishers refer only to the intended simulator.
- Confirm `/joint_states` contains all seven named arm joints.
- Confirm the `play_motion` server is simulated.
- Record old and refactored trajectory messages for identical replay input and
  compare them before allowing execution.
- Verify which `tiago_ring_controller.world` is actually resolved. The package
  launch currently selects the external `pal_gazebo_worlds` version.

### Gate 3: hardware, explicit approval required

- Obtain operator approval for the named robot, joint, workflow, range, trial
  count, and output/calibration destination.
- Place the robot in a collision-free pose and keep people and equipment outside
  the arm workspace.
- Verify an external emergency-stop path and a qualified operator are present.
- Confirm joint-state freshness, configured limits, action server, controller, and
  FT stream before publishing.
- Start with a non-persistent, reduced-range command-trace comparison. Do not begin
  with calibration or manual limit discovery.
- Monitor commanded and measured position/velocity and stop on stale/missing state,
  timeout, unexpected controller ownership, or divergence.

Hardware approval is per run; previous approval does not authorize later calibration
or manual-limit execution.

## Default entrypoint impact

| Entrypoint | Baseline default |
|---|---|
| `analysis_single_joint.py` | joint index 6; 25 trials in five batches |
| `calibrate_single_joint.py` | indices 0, 2, 3, 5; 100 trials each |
| `single_joint_data_collector.py` | joint index 5; 50 trials |
| `single_joint_pid_data_collector.py` | all seven joints; 50 trials each |
| `demo_graphs.py` | joint index 3; current position plus 15 degrees |
| `record_experiments.py` | no motion, but writes FT CSV immediately |

Do not use these mains as generic smoke tests.

## Data and rollback

Before a simulation or approved hardware comparison, use a new output directory and
copy the calibration JSON outside the package. Never run trainer or calibration
validation against checked-in artifacts. If command traces differ, stop at the
offline/Gazebo gate, restore the compatibility facade for that subsystem, and keep
the golden files unchanged.
