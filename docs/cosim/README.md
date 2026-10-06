# Co-simulation layer

The lock-stepped NEST and Gazebo loop that every graph version runs on. The
assessment of the original hand-written loop, the two candidate designs for
its replacement and the progress record of the one that was built are archived:

| File | Purpose |
| --- | --- |
| [assessment](../archive/cosim-assessment.md) | What the current loop does, why it is defective, and what any replacement must fix. Cites file:line in the legacy code. |
| [plan A](../archive/cosim-plan-a-nrp-core.md) | Option A: migrate onto the HBP Neurorobotics Platform (`nrp-core`). Feasibility, risks, step list, go/no-go gates. |
| [plan B](../archive/cosim-plan-b-inhouse-loop.md) | Option B (**recommended**): implement an NRP-style fixed-time-increment loop inside this package. Design, phased tasks, acceptance tests. |

Implementation status is kept in the "Progress" section of plan B. The loop,
fakes, real-NEST engine, Gazebo/ROS engine and dashboard live under
`src/tiago_ring_controller/src/tiago_ring_controller/cosim/`, with drivers in
`src/tiago_ring_controller/scripts/` and tests in
`src/tiago_ring_controller/test/test_cosim_*.py`. On `blocks-refactor` the loop
became the runtime of the block graphs (`docs/blocks/`), which replaced the
flat scripts' own loops; plan B's phase 7 is marked superseded there.

## Decision summary

- **Recommendation: Option B.** The defective part of the code is ~100 lines of tick
  logic replicated three times. NRP's core concept (engines, datapacks, transceiver
  functions, one-step-delay fixed-time-increment loop) is small enough to implement
  in ~500 lines of Python, and doing so keeps TIAGo's `ros_control` / `play_motion`
  stack intact. Option A forces a choice between an unsupported nrp-core + ROS 1 +
  PAL-controllers combination and giving up PAL's controllers entirely.
- Option A is documented in full so it can be revisited if a collaborator already
  runs nrp-core, or if the project moves to ROS 2 (where nrp-core is now developed).
- **Branch, not fork.** See below.

## Branch vs. fork

Work on a branch of this repository (`cosim-loop`), not a fork:

- The goal is to *replace* the loop inside the same research package, keeping the
  NEST model, the calibration/analysis/collection workflows and the artifact
  contracts. That is a feature branch by definition; a fork is for a different owner
  or a project that will diverge permanently.
- `main` stays the frozen, reproducible baseline behind the existing results
  (`docs/recovery-manifest.json`, golden tests). Tagging it makes that explicit:
  `git tag legacy-loop-baseline main`.
- A single branch holding both plan and implementation keeps the history of *why*
  next to *what*. Merge back with a PR when the acceptance tests in plan B pass;
  until then nothing on `main` changes.
- If the refactor later needs to break compatibility contracts that `main`'s tests
  freeze (see `src/tiago_ring_controller/docs/architecture.md`, "Compatibility
  rule"), do it on this branch behind the new loop and retire the legacy facades
  in the PR, rather than forking.

## Environment

Linux amd64. The Docker workflow in the top-level `README.md` is the runtime:
`./run_model_docker.bash --software`. The image supplies Ubuntu 20.04, ROS Noetic,
Gazebo 11, Python 3.8 and NEST `HEAD@41892a5`. Nothing in plan B needs a new image;
the optional Gazebo step plugin (plan B, phase 5) builds with the existing
`catkin build` of `tiago_ring_controller`.

## Gazebo checks

Run on 2026-10-02 against the headless simulation (`gui:=false`, software
rendering, pinned image); method and full output in `docs/archive/velocity/calibration.md`.

| Check | Result |
| --- | --- |
| `/use_sim_time` | `true` |
| `/gazebo/pause_physics`, `unpause_physics`, `get_physics_properties` | present; `time_step` 0.001 s, `max_update_rate` 1000 |
| `/clock`, `/joint_states` | 1000 and 50 messages per simulated second, real-time factor 1.00 |
| Clock-wait stepper | overshoots one physics iteration (1 ms) per step; with the absolute schedule a tick is 0.0500 s on average (0.049-0.051), 51 ms wall |
| Velocity steps through `GazeboRosEngine` (`scripts/identify_actuator.py`) | displacement gain 0.985-0.991, lag 8-14 ms on joints 1, 4, 5 |
| Graph trials (`scripts/validate_velocity.py --engines full`) | see `docs/archive/velocity/calibration.md` |

Two facts found there shape the engine: the arm has no velocity interface
(`arm_controller` is a `position_controllers/JointTrajectoryController`), and the
velocity field of `/joint_states` is not usable for the wrist (about -1.45 rad/s
for a resting `arm_6_joint`), so stillness and measured velocity come from
position differences.

Still open: `scripts/measure_legacy_tick.py` (the legacy loop's real tick
period), `scripts/cosim_gazebo_smoke.py --joints 5 6`, the dashboard and editor
against Gazebo, and a run with the Gazebo GUI on a GPU.
