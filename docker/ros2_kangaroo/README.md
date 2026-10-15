# ROS 2 Kangaroo image

**What it is:** a ROS 2 Humble container with MuJoCo and PAL's Kangaroo simulation, installed from PAL's
official apt releases. It renders on the host NVIDIA GPU and shows its window on your desktop (X11, or Wayland
through XWayland).

**Purpose:** to provide the simulator for migrating the ring-attractor velocity controller from TIAGo
(ROS 1 / Gazebo, see `../../README.md`) to Kangaroo. The existing TIAGo Docker setup is separate and unchanged.

## Requirements

- Linux amd64 with Docker Engine, NVIDIA driver and NVIDIA Container Toolkit.
- A desktop session with `xauth` installed. On Ubuntu with Wayland, XWayland provides `DISPLAY` by default.

## Run

From the repository root (your current directory is mounted at `/work`):

```bash
docker/ros2_kangaroo/run.bash            # builds the image on first use, then opens a shell
```

Inside the container:

```bash
ros2 launch /opt/kangaroo_sim.launch.py  # PAL's 4-DoF Kangaroo in an empty MuJoCo world + arm controllers
```

To move individual arm joints by velocity, see `../../src/kangaroo_ring_controller/README.md`.

A second terminal into the same container:

```bash
docker exec -it ros2-kangaroo bash
ros2 topic echo /joint_states            # arm_{left,right}_{1..4}_joint
ros2 control list_hardware_interfaces    # each arm joint: position / velocity / effort
```

Only one container runs at a time (its name is fixed). To rebuild after changing the Dockerfile, run
`docker build -t ros2-kangaroo:humble docker/ros2_kangaroo`.

## Check that it works

```bash
docker/ros2_kangaroo/run.bash docker/ros2_kangaroo/smoke_test.bash
```

It launches the simulation and checks five things: the 8 arm joints appear on `/joint_states`, each arm joint
accepts position, velocity and effort commands, the joint state broadcaster is active, `/clock` advances, and
MuJoCo renders on the NVIDIA GPU. It prints PASS/FAIL for each and exits with the number of failures.

## Files

| File | Role |
|---|---|
| `Dockerfile` | Pinned `osrf/ros:humble-desktop` base plus PAL's apt packages |
| `kangaroo_sim.launch.py` | PAL's launch, unchanged, plus the Humble remap (below), plus PAL's left/right arm trajectory controllers. Defaults: 4-DoF, MoveIt off |
| `run.bash` | Starts the container with GPU, display, and your current directory at `/work` |
| `smoke_test.bash` | The acceptance check above |

## Versions (verified in the image, 2026-10-13)

The base image is `osrf/ros@sha256:fb07245b…` (built 2026-09-10). PAL packages are not pinned in apt, because apt
mirrors drop old versions. A rebuild may therefore install newer ones; re-run the smoke test after rebuilding.

| Package | Version |
|---|---|
| ros-humble-kangaroo-mujoco | 2.7.0 |
| ros-humble-kangaroo-description / bringup / controller-configuration | 2.15.1 |
| ros-humble-pal-sea-arm-description | 2.8.4 |
| ros-humble-mujoco-ros2-control | 0.1.2 |
| MuJoCo (via ros-humble-mujoco-vendor 0.1.1) | 3.12.0 |
| NEST (built from source at commit `41892a50f95d…`, the repo's pinned `HEAD@41892a5`; reports `HEAD@41892a50f`) | for the ring attractor in `src/kangaroo_ring_controller/` |

## Known upstream issues (PAL's public Humble release)

Kangaroo is released only for Humble; no other ROS 2 distribution has it. Its public release needs two
additions to run, and both are already handled here:

1. **Missing dependency.** PAL's launch looks up `kangaroo_moveit_config` even with `moveit:=False`, but does
   not declare it. The Dockerfile installs it.
2. **Missing Humble remap.** On Humble, `ros2_control_node` listens on `~/robot_description`, and PAL's launch
   does not remap it to `/robot_description`. Without the remap the controller manager waits forever.
   `kangaroo_sim.launch.py` adds the same remap that upstream's demos use on Humble
   (`mujoco_ros2_control_demos/launch/01_basic_robot.launch.py`).

The 5-DoF RH8D model is not used. It needs `rh8d_description`, which is not publicly released, and the public
source (`pal-robotics/pal_rh8d`) does not match `pal_sea_arm` 2.8.4.

Harmless warning: `Could not enable FIFO RT scheduling policy`. The container has no real-time priority; the
simulation runs on simulation time (`/clock`).
