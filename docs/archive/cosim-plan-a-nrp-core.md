# Plan A — migrate to the Neurorobotics Platform (`nrp-core`)

Status: **documented, not recommended** (see README). Revisit if the project moves to
ROS 2 or a collaborator already operates nrp-core.

## A.1 What nrp-core offers

- A fixed-time-increment simulation loop (`FTILoop`). Each engine has an
  `EngineTimestep`; the loop runs at the smallest one and, at every `t` that is a
  multiple of an engine's timestep, waits for that engine, fetches its datapacks,
  runs preprocessing + transceiver functions (TFs), sends datapacks back, then
  advances. Data produced at `t` is consumed at `t + dt` (one-step delay).
- Engines: `nest_json` / `nest_server` (NEST 3), `gazebo_grpc` / `gazebo_json`
  (Gazebo 11 Classic), `python_json` (arbitrary Python via an `EngineScript`
  subclass with `initialize / runLoop(timestep_ns) / shutdown`), plus PySim,
  OpenSim, TVB, EDLUT, DataTransfer (MQTT/ROS streaming).
- NEST datapacks are `nest.GetStatus` / `nest.SetStatus` on a registered NEST object
  (`RegisterDataPack(name, nodes)` / `CreateDataPack(...)`) in a brain script named by
  `NestInitFileName`.
- Gazebo datapacks come from NRP plugins added to the SDF: world plugin
  (`NRPGazeboGrpcWorldPlugin.so`), joint plugin with per-joint PID and
  `position` / `velocity` targets, link, model and camera plugins. Extra system
  plugins can be passed through `GazeboPlugins`.
- ROS access from experiments is via ROS nodes in the "computational graph"
  (Python), restricted to message types with Python bindings; no time sync, service
  or action support is documented.
- Shipped example: `examples/husky_braitenberg` (NEST + Gazebo).

Documentation quality is adequate for the loop, sync model, engine configuration,
TFs and the Python engine. It is thin on running an existing ROS robot stack under
NRP's Gazebo.

## A.2 Version facts (checked 2026-09-28)

| | Tagged releases 1.5.0 (2026-01-23), 1.5.1 (2026-04-27) | `development` branch (last push 2026-09-18) |
| --- | --- | --- |
| OS | Ubuntu 20.04 only | Ubuntu 22.04 only (20.04 chain dropped) |
| ROS | Noetic (`ros-noetic-ros-base`) | ROS 2 Humble |
| Gazebo | Gazebo 11 Classic | Gazebo Classic (jammy packages) |
| NEST | NEST 3 built in-tree | NEST 3.10 |
| Python | 3.8 | 3.10 |

Repository: `https://github.com/vvorobjov/nrp-core` (canonical; the EBRAINS GitLab is
a mirror). One active maintainer, 0 stars. Docs at
`https://neurorobotics.net/Documentation/latest/nrp-core/`. Only
`hbpneurorobotics/nrp-vanilla` and `hbpneurorobotics/nrp-nest-gazebo` images are
published; the others (including the docker-compose examples' `nrp-gazebo`) must be
built locally.

Consequence: to stay on TIAGo's Noetic stack you pin nrp-core **1.5.1** with no
upgrade path; to follow nrp-core you must move TIAGo to ROS 2 first.

## A.3 The structural problem

nrp-core's Gazebo engine runs `gzserver` inside its own engine process and steps it.
TIAGo's simulation is launched by `tiago_gazebo.launch` (gazebo_ros +
`gazebo_ros_control` + PAL's `arm_controller` + `play_motion` + the custom world and
motion overrides in `overrides/`). Two ways to reconcile them, both unattractive:

**A-1 "NRP owns Gazebo".** Load the TIAGo model through `GazeboSDFModels` and drive
the arm joints with `NRPGazeboGrpcJointControllerPlugin` (velocity targets + PID).
Costs: convert TIAGo's xacro/URDF (with PAL transmissions, gripper, FT sensor) to a
plain SDF; lose `ros_control`, `/arm_controller/command`, `play_motion` resets,
`/wrist_ft` via ros plugins, and any comparability with the calibration data
collected through the PAL controllers. Everything downstream (calibration JSON,
trajectory contract, safety doc) changes.

**A-2 "TIAGo owns Gazebo, NRP wraps it".** Keep `tiago_gazebo.launch` and write a
`python_json` engine whose `runLoop` pauses/steps Gazebo through ROS and reads
`/joint_states`. Then NRP contributes only the loop and the NEST engine; you pay the
full nrp-core install and its process model (gRPC/REST servers per engine, JSON
datapacks) for ~200 lines of scheduling logic, and you still have to write the
Gazebo stepping code yourself — which is exactly Plan B's core.

A third variant — pass `libgazebo_ros_api_plugin.so` and `gazebo_ros_control` through
`GazeboPlugins` so PAL's controllers run inside NRP's `gzserver` — is undocumented
and untested by upstream. It may work (both are ordinary system plugins) but it has
no support path and mixes two owners of the same simulation.

## A.4 Datapack cost

A NEST datapack read is `GetStatus` on the registered object. Registering the
spike recorders returns their full `events` arrays each step, i.e. the same growing
cost as today. Workarounds: register recorders with `record_to: "ascii"` or read
`n_events` via a small custom datapack in the brain script; either way the
readout restructuring in Plan B (R5) is still needed.

## A.5 Step list, if pursued

Environment (Ubuntu 20.04 VM or a derived Docker image; do **not** install into the
pinned TIAGo image in place — derive a new image and update `docker/image.env`):

1. `git clone -b 1.5.1 https://github.com/vvorobjov/nrp-core` and follow its
   README "Dependency Installation" for 20.04: Pistache PPA, Gazebo 11 dev packages,
   ~30 apt packages, pip packages (flask, gunicorn, paho-mqtt, grpcio-tools, ...),
   `ros-noetic-ros-base`, optional Paho MQTT C/C++. Set
   `NRP_INSTALL_DIR`, `NRP_DEPS_INSTALL_DIR`.
2. `cmake` with `-DBUILD_NEST_ENGINE_SERVER=ON`. nrp-core compiles its own NEST 3
   (not `HEAD@41892a5`); verify the model still reproduces the golden/smoke tests
   under that NEST before going further. `-DENABLE_MQTT=OFF` unless streaming is
   needed. Budget: hours of build, days of environment work.
3. Run `examples/husky_braitenberg` to validate the install.
4. Choose A-1 or A-2 (A.3). For A-2 write:
   - `experiments/tiago_ring/simulation_config.json` with a `nest_json` engine
     (`NestInitFileName: brain.py`, `EngineTimestep: 0.05`) and a `python_json`
     engine (`PythonFileName: tiago_engine.py`, `EngineTimestep: 0.05`);
   - `brain.py`: build `SingleRingModel` via the existing builders; replace
     `inject_stimulus` with pre-created generators (Plan B, B.4); register datapacks
     `state_bump_rates`, `goal_bump_rates` (write) and `ring_counts` (read);
   - `tiago_engine.py`: `EngineScript` that on `initialize` starts rospy, subscribes
     `/joint_states`, publishes `/arm_controller/command`; on `runLoop` steps Gazebo
     by `timestep_ns` (Plan B, B.3) and publishes the pending command; datapacks
     `joint_state` (read) and `arm_velocity_cmd` (write);
   - TFs `proprioception_tf.py`, `motor_tf.py` reusing `DriveControlCore` and the
     `LegacyControlProfile` mappings.
5. Port collection / analysis / calibration to launch NRP (`NRPCoreSim -c ...`) and
   consume the recorded datapacks instead of running their own loops.

## A.6 Go / no-go gates

- G1: nrp-core 1.5.1 builds in a derived image and `husky_braitenberg` runs. If not
  within ~2 working days, stop.
- G2: the model's real-NEST smoke test passes on nrp-core's NEST build.
- G3: for A-2, a `python_json` engine can pause/step TIAGo's Gazebo and the
  `/arm_controller` still tracks a trajectory under stepping. (This gate is identical
  to Plan B phase 1; if it passes, Plan B is already most of the way done, which is
  the argument for B.)

## A.7 Effort estimate

Environment 2-4 days; engines + TFs 3-5 days; porting the three workflows 3-5 days;
unknown debugging of ROS-under-NRP. Total 2-3 weeks with high variance, versus
~1-1.5 weeks for Plan B with low variance.
