# TIAGo ring-attractor workspace

The recovered controller, weights, calibration and existing experiment results live
in this checkout. Docker supplies Ubuntu 20.04, ROS Noetic, Python 3.8, NEST
`HEAD@41892a5`, Gazebo and the upstream TIAGo packages.

## Co-simulation refactor (branch `cosim-loop`)

The NEST → Gazebo control loop is being replaced by a lock-stepped, NRP-style
co-simulation layer. The assessment of the current loop, the two candidate plans
(migrate to `nrp-core`, or an in-house loop — recommended) and the hand-over
checklist for a Linux session are in [`docs/cosim/`](docs/cosim/README.md). `main`
remains the frozen baseline behind the existing results.

### Running the co-simulation loop

The loop lives in `src/tiago_ring_controller/src/tiago_ring_controller/cosim/` with
drivers in `src/tiago_ring_controller/scripts/`. The NEST model is unchanged and the
legacy scripts still run their own loops.

Tests without any simulator (host Python with NumPy and Matplotlib is enough):

```bash
cd src/tiago_ring_controller
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s test -p "test_cosim*.py"
```

Inside the container (see "Start" below), from the package `src` directory:

```bash
python3 -B -m unittest discover -s ../test -p "test_cosim*.py"     # adds the real-NEST parity test
python3 -B ../scripts/run_cosim_trial.py --engines fake --goal 0.6   # fake NEST + fake robot
python3 -B ../scripts/run_cosim_trial.py --engines nest --seed 13579 --goal 0.6 --out /tmp/cosim_nest
```

With Gazebo (launch the simulation in the ready shell, then use a second terminal as
described under "TIAGo simulation"):

```bash
python3 -B ../scripts/cosim_gazebo_smoke.py --joint 5 --out /tmp/smoke.json      # stepping check, no NEST
python3 -B ../scripts/run_cosim_trial.py --engines full --goal 0.6 --out /tmp/cosim_full
python3 -B ../scripts/run_cosim_trial.py --engines full --goal 0.6 --monitor --monitor-hold   # live ring view
```

The browser dashboard is the easiest way to watch and drive trials, and it needs no
X11 inside the container:

```bash
python3 -B ../scripts/run_cosim_trial.py --engines full --dashboard --out /tmp/cosim_full
```

Then open <http://localhost:8765/> on the host (the container uses host networking).
The page shows the state ring, a rolling raster, the gain counts and the joint angle
against the goal, updated every tick, and has *Start trial* / *Stop* buttons with a
goal and an optional step budget; finished trials are listed with their timings.
Trials run only when started from the page; Ctrl-C in the terminal ends the session.
*Before trial* chooses between the legacy behaviour (rebuild the NEST network and
home the robot, about 20 s) and continuing from the network and arm state the previous
trial left, with only the new goal (and state) bumps injected; *Reset now* rebuilds
and homes without running a trial. The same choice is `--reset-mode` on the command
line. Continuing is not characterised yet: the old goal bump is still in the target
ring when the new one arrives.
`--dashboard-port` changes the port and `--dashboard-host 0.0.0.0` exposes it beyond
the machine.

`--monitor` is the Matplotlib alternative (a window with the same panels; needs a
launcher started without `--headless`). `--monitor-frames DIR` saves a PNG per redraw
instead, `--monitor-every N` redraws less often, `--monitor-hold` keeps the window
open after the last trial.
Outputs use the collector's layout (`trials_summary.csv`, `trials/*.npz`) plus one
`trial_XXXX_cosim.json` record per trial. Always run with `-B`: a golden test fails if
new `.pyc` files appear under the package. Progress and design notes are in
[`docs/cosim/plan-b-inhouse-loop.md`](docs/cosim/plan-b-inhouse-loop.md).

## Start

Use Linux amd64 with Docker Engine accessible to your user. Desktop operation needs
X11/XWayland and host `xauth`; NVIDIA acceleration also needs the host driver and
NVIDIA Container Toolkit. The approximately 20 GB image is pulled only if missing.
Its immutable digest is recorded in `docker/image.env`.

```bash
cd /home/aung/SHU/Ring-Attractor-SO-2-Joint-Architecture
./run_model_docker.bash --name tiago-ring
```

This builds only your controller incrementally, then opens a shell with ROS and
NEST configured. The shell starts in the controller's `src` directory, where its
relative configuration paths work. Run, for example:

```bash
python3 single_ring.py
python3 multi_ring_component.py
```

Use `--software` for software rendering. For terminal-only work:

```bash
./run_model_docker.bash --headless --software -- python3 -c 'import nest; print(nest.__version__)'
```

The launcher works from any host directory. Options go before `--`; commands and
their arguments go after it. Exit the shell to remove the container. Mounted files
and results persist on your PC, owned by your host user.

## TIAGo simulation

Inside the ready shell:

```bash
roslaunch tiago_ring_controller tiago_ring_controller.launch
```

This launches the existing camera-equipped world, TIAGo with its custom
`tiago_experiment_start_1` starting pose, and the wrist torque recorder. The
neural controller is not automatically launched by that launch file.

In another **host terminal**, enter the same container:

```bash
docker exec --user "$(id -u):$(id -g)" -it \
  --workdir /tiago_public_ws/src/tiago_ring_controller/src \
  tiago-ring bash --rcfile /model_repo/docker/bashrc -i
```

For example, inspect camera frames without opening a second viewer:

```bash
python3 experiment_camera.py --headless --duration 10
```

The launcher uses host networking. `ROS_MASTER_URI` defaults to
`http://localhost:11311`; host `ROS_MASTER_URI`, `ROS_IP`, and `ROS_HOSTNAME` are
forwarded when set. Use a different master port for independent simulations.

## Where your work lives

| Host location | Purpose |
| --- | --- |
| `src/tiago_ring_controller/` | Complete original package, including scientific inputs and saved results |
| `overrides/` | Camera world, custom motion template/YAML, and modified TIAGo startup script |
| `recovered/experiment_results/` | Recovered workspace-level recordings |
| `.runtime/catkin_ws/` | Writable Catkin overlay and build logs, ignored by Git |
| `.runtime/home/` | Container user home, ROS/Gazebo logs and caches, ignored by Git |

The checkout is mounted at `/model_repo`. The controller is also mounted at its
original `/tiago_public_ws/src/tiago_ring_controller` location. The override mapping
in `docker/overrides.list` mounts your custom files over their original upstream
locations. Edit overrides on the host and restart Docker to pick up all changes.

Model outputs retain their original locations beneath the package, including
`src/outputs/`, `src/collected_data/`, `src/plots/`, `src/results_plots/`, `results/`
and `experiment_results/`. Existing results are preserved in the Git baseline;
new result files appear in Git status so you can choose which to save.

Code and configuration edits need no image rebuild or `docker commit`. New Python
processes read the mounted files. Training scripts can overwrite their default
weight destinations: use a disposable copy when testing training. See the
recovered package's README and `docs/` for the original research workflows.

## Build and test

Every launch performs an incremental build in a Catkin overlay extending the
image's prebuilt TIAGo workspace. To rebuild within a running container:

```bash
catkin build --workspace /model_repo/.runtime/catkin_ws tiago_ring_controller --no-deps -j2 -p1
source /model_repo/docker/environment.bash
```

For a clean build, exit all project containers and remove only
`.runtime/catkin_ws` on the host, then rerun the launcher. This is generated build
state; source, results and the container home are separate.

Run the recovered unit suite:

```bash
./run_model_docker.bash --headless --software -- bash -c \
  'cd /tiago_public_ws/src/tiago_ring_controller && python3 -B -m unittest discover -s test -p "test_*.py"'
```

Or use Catkin inside the ready shell:

```bash
catkin run_tests --workspace /model_repo/.runtime/catkin_ws tiago_ring_controller
catkin_test_results /model_repo/.runtime/catkin_ws/build/tiago_ring_controller/test_results
```

The original image already has three failing historical inventory tests. These
remain visible; see `docs/validation.md` for exact results and migration checks.
Historical bytecode is preserved and automatic bytecode writes are disabled.

To check the original extraction on the host:

```bash
python3 docker/check_recovery.py
```

This verifies all 480 original files and modes. After intentional research edits,
it reports those differences; it never restores files or changes manifests.

## Environment maintenance

The base is the original `aung9htet/ubuntu-20.04:tiago_ring_forward_2` image pinned
by digest. All installed dependencies remain in that image. Changing system
dependencies requires a deliberately updated environment image and image pin;
clear generated Catkin state when changing that environment. Normal startup never
installs dependencies, refreshes a moving image tag, or extracts source files.

If Docker is inaccessible, check the daemon and Docker group membership. For
missing display/cookie errors, start from a desktop terminal with `DISPLAY` and
`XAUTHORITY`, or use `--headless`. If NVIDIA startup fails, try `--software`.
`--headless` disables display forwarding; Gazebo's own `gui:=false` additionally
disables its GUI. Camera rendering may still require a virtual X display; see the
validation notes for tested headless simulation commands.
