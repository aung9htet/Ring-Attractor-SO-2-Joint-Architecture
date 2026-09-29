# Migration validation

Validation was performed on this Linux amd64 host on 2026-09-28 and 2026-09-29.
The recovered scientific source and reference manifests were not edited.

## Recovery and runtime

- All 480 recovered files match their extraction SHA-256 and permission modes.
- Clean Catkin build passes: controller and Catkin prebuild succeed; 182 upstream
  packages are skipped. Subsequent startup performs an incremental controller build.
- ROS package lookup resolves to `/tiago_public_ws/src/tiago_ring_controller`.
- Python imports use the mounted source, with Python 3.8 and NEST `HEAD@41892a5`.
- Runtime commands run as UID/GID 1000:1000. Host edits are visible in the container;
  container-created files remain host-owned and persist after container replacement.
- Headless execution, invocation from another host directory, argument errors,
  missing-display errors, and command exit-code propagation were checked.
- The environment uses the original source path for Catkin-generated Python
  relays and test commands. An initial symlink-based source layout introduced an
  extra path-contract failure; the final layout removes that failure without
  modifying the research code or tests.

## Unit and ROS tests

The final Catkin run reports **141 unit tests, zero errors, three failures, zero
skips**, matching the original image baseline. Real-NEST smoke cases are included
and ran successfully. The ROS interface rostest passes (one assertion test;
Catkin reports both its wrapper and result, giving a combined total of 143).

The three existing failures are:

| Test | Existing discrepancy |
| --- | --- |
| `ArtifactImmutabilityTests.test_manifest_is_versioned_and_complete` | Newer output JSON files are absent from the historical manifest |
| `OrphanBytecodeProvenanceTests.test_complete_pre_refactor_cache_tree_is_present_and_byte_identical` | The image contains additional bytecode beyond the historical inventory |
| `CheckedFigureOutputTests.test_all_checked_png_bytes_and_dimensions_match_the_baseline` | The image contains additional plots beyond the historical inventory |

Independent checks also found that four historical cache files differ from the
old bytecode hashes: `gain_modulation`, `homeostasis`, `ring_attractor`, and
`ring_component`. The original 34 scientific artifact hashes, 21 figure hashes,
and both source-less bytecode hashes still match their historical records.
The inventory test stops before checking those four cache hashes, so this
additional discrepancy is recorded explicitly.

These known failures remain visible: `unittest` and `catkin_test_results` return
nonzero. No baselines were regenerated, tests weakened, or scientific inputs
replaced to make the suite green.

## Gazebo and graphics

Full TIAGo simulation was tested with the desktop forwarded through XWayland,
both with NVIDIA enabled and with `--software`. The software renderer identifies
itself as Mesa llvmpipe. Both runs verified:

- 1280×720 messages on `/recording_camera/image_raw`;
- messages on `/wrist_ft` and `/joint_states`;
- the custom `tiago_experiment_start_1` configuration;
- successful startup motion action result and final joints within 0.12 of their
  configured targets (radians for arm joints, metres for the torso);
- a nonempty torque CSV written into the host-mounted controller package.

Shutdown produced occasional Gazebo segmentation-fault output and a torque
recorder callback reporting a closed log file, after the functional checks had
passed. These shutdown diagnostics are not fixed by this environment migration;
the recovered controller is unchanged. Logs are retained under
`.runtime/validation/` on this machine.

Headless **model and test execution** was verified using `--headless --software`.
Full camera-enabled Gazebo with no X server was not validated. `gui:=false`
disables the Gazebo client window, but Gazebo Classic camera rendering may still
require an X server. For a simulation without a client window on this desktop,
retain display forwarding and run:

```bash
./run_model_docker.bash --name tiago-ring -- \
  roslaunch tiago_ring_controller tiago_ring_controller.launch gui:=false
```

## Reproduce checks

```bash
python3 docker/check_recovery.py
./run_model_docker.bash --headless --software -- bash -c \
  'catkin run_tests --workspace /model_repo/.runtime/catkin_ws tiago_ring_controller --no-status; catkin_test_results /model_repo/.runtime/catkin_ws/build/tiago_ring_controller/test_results'
```

The expected outcome is a successful build, a passing ROS interface test, and
exactly the three historical unit failures listed above.
