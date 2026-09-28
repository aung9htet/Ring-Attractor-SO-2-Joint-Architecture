# Co-simulation refactor (branch `cosim-loop`)

This folder holds the plan for replacing the hand-written NEST → Gazebo control loop
with a proper co-simulation layer. It is written so that a fresh session (human or
Claude) on a Linux machine can take over without re-deriving anything.

| File | Purpose |
| --- | --- |
| [assessment.md](assessment.md) | What the current loop does, why it is defective, and what any replacement must fix. Cites file:line in the legacy code. |
| [plan-a-nrp-core.md](plan-a-nrp-core.md) | Option A: migrate onto the HBP Neurorobotics Platform (`nrp-core`). Feasibility, risks, step list, go/no-go gates. |
| [plan-b-inhouse-loop.md](plan-b-inhouse-loop.md) | Option B (**recommended**): implement an NRP-style fixed-time-increment loop inside this package. Design, phased tasks, acceptance tests. |

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

## Environment for this work

Linux amd64 only (Ubuntu VM or native). The existing Docker workflow in the top-level
`README.md` is the runtime: `./run_model_docker.bash --software`. The image supplies
Ubuntu 20.04, ROS Noetic, Gazebo 11, Python 3.8 and NEST `HEAD@41892a5`. Nothing in
plan B needs a new image; the optional Gazebo step plugin (plan B, phase 2) builds
with the existing `catkin build` of `tiago_ring_controller`.

## Hand-over checklist for the Linux session

Do these first; each answers a question the plans depend on. Record answers in
`assessment.md` under "Verified on Linux".

1. Start the simulation: `roslaunch tiago_ring_controller tiago_ring_controller.launch gui:=false`
   inside the container. Then check:
   - `rosparam get /use_sim_time` (expected `true`; gazebo_ros sets it).
   - `rosservice list | grep gazebo` — confirm `/gazebo/pause_physics`,
     `/gazebo/unpause_physics`, `/gazebo/get_physics_properties` exist.
   - `rostopic hz /clock` and `rostopic hz /joint_states` (expected ~1000 Hz and
     ~50-100 Hz).
   - `rosservice call /gazebo/pause_physics` then `rostopic echo -n 2 /clock` —
     confirm the clock stops.
2. Run the legacy unit suite and the real-NEST smoke test as in the top-level README,
   to confirm the baseline is green before touching anything.
3. Measure the legacy loop's real tick period once (plan B, phase 0) so there is a
   before/after number.
4. Then follow `plan-b-inhouse-loop.md` phase by phase.
