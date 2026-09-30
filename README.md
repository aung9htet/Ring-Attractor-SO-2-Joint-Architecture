# TIAGo ring-attractor workspace

A spiking ring-attractor controller (NEST) that drives joints of a simulated
TIAGo arm (Gazebo, ROS Noetic), run as a lock-stepped co-simulation and
assembled from parameterised blocks that you can edit in the browser.

## 1. The system in one page

```text
 graph file (*.graph.json)  <──>  Python Graph API  <──>  browser editor (/editor)
                                        │ compile
                                        v
      blocks: Ring, FourierReadout, Homeostasis, Gain, Encoder, Decoder, Joint, Goal, ...
                                        │ build (NEST)          │ generated transceivers
                                        v                       v
                NEST engine  <──  fixed-time-increment loop (50 ms ticks)  ──>  robot engine
                                                                               (fake, or Gazebo via ROS)
```

- **Model.** Two ring attractors (state and target, N = 200 neurons each) are read
  out by Fourier feature populations; a fitted comparator decides left or right;
  opponent gain populations move the state bump and their spike counts, filtered
  and scaled, become the joint velocity. The model is a *graph of blocks* with
  declared parameters (`docs/blocks/blocks.md`); three reference architectures
  exist as templates: `two_ring_single_joint` (the model above),
  `multi_joint_two_ring` (one circuit per joint), `three_ring_single_joint`
  (target / belief / actual rings), plus `two_joint_forward_kinematics`.
- **Runtime.** A co-simulation loop (`tiago_ring_controller.cosim`) advances the
  NEST engine and the robot engine by exactly one tick at a time, with a
  one-tick delay between them. The robot is either a fake first-order model
  (no simulator needed) or Gazebo through `ros_control`.
- **Interfaces.** `scripts/run_graph.py` runs a graph file or template from the
  command line; `--dashboard` serves a trial page and the graph editor at
  <http://localhost:8765/>.
- **Where things are.** Package `src/tiago_ring_controller/`: importable code in
  `src/tiago_ring_controller/` (`blocks/`, `graph/`, `cosim/`, `nest/`, `ui/`),
  drivers in `scripts/`, tests in `test/`, scientific inputs in `src/config/`,
  the frozen original scripts in `legacy/`. Documentation index: section 6.

## 2. Main workflow: Gazebo, dashboard, editor

NEST and Gazebo live in the Docker image; nothing is installed on the host for
this workflow. Requirements: Linux amd64, Docker Engine usable by your user;
a desktop session with X11/XWayland and `xauth` for the Gazebo GUI (or
`--headless`).

**Step 1 — start the container** (pulls the pinned image the first time, about
20 GB, then builds the controller package incrementally):

```bash
./run_model_docker.bash --name tiago-ring            # with display
./run_model_docker.bash --name tiago-ring --software  # software rendering
```

You get a shell inside the container with ROS and NEST configured.

**Step 2 — launch the simulation** in that shell:

```bash
roslaunch tiago_ring_controller tiago_ring_controller.launch          # gui:=false for no window
```

This brings up the camera world, TIAGo in its starting pose and the wrist
force-torque recorder. Wait for the arm to settle.

**Step 3 — open a second terminal in the same container** (host terminal):

```bash
docker exec --user "$(id -u):$(id -g)" -it \
  --workdir /tiago_public_ws/src/tiago_ring_controller \
  tiago-ring bash --rcfile /model_repo/docker/bashrc -i
```

**Step 4 — serve the dashboard and editor with the model** (still inside the
container, from the package root):

```bash
python3 -B scripts/run_graph.py \
  src/tiago_ring_controller/graph/examples/two_ring_single_joint.graph.json \
  --engines full --dashboard --out /tmp/session_$(date +%H%M)
```

`--engines full` means real NEST plus Gazebo. The container uses host
networking, so open on the host:

- <http://localhost:8765/> — the trial page: state ring, raster, gain counts,
  joint angle against the goal; *Start trial* with a goal in radians and an
  optional step budget; *Stop*; *Reset now*; *Before trial* chooses between
  rebuilding the network and homing the arm (legacy behaviour) or continuing.
- <http://localhost:8765/editor> — the graph editor: block palette on the left,
  canvas in the middle, parameters on the right. *Load example* opens a
  template, *Load file* / *Save* read and write graph files (the server writes
  the canonical form), *Validate* lists every problem, *Run in session* compiles
  the drawn graph and swaps it into the running session between trials; while
  a trial runs the state ring, gain, decoder and joint blocks show live values.

Trials run only when started from the page; Ctrl-C in the terminal ends the
session. Outputs go to `--out` in the collector layout (`trials_summary.csv`,
`trials/*.npz`, one `trial_XXXX_cosim.json` per trial).

**Step 5 — run a saved experiment without the browser:**

```bash
python3 -B scripts/run_graph.py my_experiment.graph.json --engines full --goal 0.6 --goal -0.5 --out /tmp/exp
```

Safety: a graph run commands every `Joint` block of the graph, all in one
trajectory per tick. Check the `Joint` blocks of a file before `--engines
full`. The drivers never talk to hardware; see
`src/tiago_ring_controller/docs/robot_safety.md`.

Always pass `-B` to Python inside the container so no bytecode is written into
the mounted package.

## 3. Parallel workflows

### 3.1 Programmatic runs, with or without the robot

The same loop runs from Python. `engines` selects the simulators: `fake` (fake
NEST backend and fake robot: no simulator, runs on the host), `nest` (real NEST,
fake robot: inside the container), `full` (NEST and Gazebo).

```python
from tiago_ring_controller.graph import Graph, run_graph, two_ring_single_joint

graph = two_ring_single_joint(joint=5, goal_rad=0.6)      # or Graph.load("my.graph.json")
graph.blocks["enc_state"].params["mode"] = "corrective"   # edit any declared parameter
graph.blocks["enc_state"].params["dead_band"] = 5
results = run_graph(graph, engines="nest", goals=[0.6, -0.5], out_dir="/tmp/exp")
record = results[0]["record"]           # TrialRecord: every tick's datapacks, meta["graph"]
scalars = results[0]["legacy"]["scalars"]   # collector-style summary (q_final, error, steps)
graph.save("/tmp/exp/graph.graph.json")
```

To watch a programmatic run in the browser, attach the dashboard observer the
way `scripts/run_graph.py --dashboard` does, or simply run the script with
`--dashboard` and use the editor to change the graph. The model-only NEST layer
is usable on its own too: `tiago_ring_controller.nest.single_ring.build_single_ring_network(nest, seed=...)`
builds the two-ring model in about 40 ms.

Command-line equivalents:

```bash
python3 -B scripts/run_graph.py --template multi_joint_two_ring --engines nest --goal 0.6 --out /tmp/mj
python3 -B scripts/run_graph.py my.graph.json --validate
python3 -B scripts/sweep_feedback.py --engines nest --out /tmp/sweep.json --markdown /tmp/sweep.md
python3 -B scripts/run_cosim_trial.py --engines nest --model vectorised --goal 0.6   # loop without graphs
```

### 3.2 Local installation (host, without Docker)

Nothing needs installing for the fake engines and the unit tests: a Python 3
with NumPy and Matplotlib is enough (one test imports the legacy plotting
modules and also needs `colorcet`; without it that single module errors and the
rest of the suite runs). Real NEST runs are
only supported in the container: the pinned NEST is `HEAD@41892a5` and the
real-NEST tests skip on any other version.

```bash
cd src/tiago_ring_controller
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s test -p "test_*.py"     # fakes only, ~10 s
python3 -B scripts/run_graph.py --template two_ring_single_joint --engines fake --dashboard
```

The editor works on the host with the fake engines (no spikes, but every
editing, validation and save path). The headless-Chrome editor tests run
wherever `google-chrome` or `chromium` is installed.

The full suite with real NEST, without the launcher (bind-mounts the package
into the image):

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e MPLBACKEND=Agg \
  -v "$PWD/src/tiago_ring_controller:/tiago_public_ws/src/tiago_ring_controller" \
  --entrypoint bash aung9htet/ubuntu-20.04:tiago_ring_forward -c 'source /opt/ros/noetic/setup.bash; \
  source /tiago_public_ws/devel/setup.bash; source /usr/local/nest/bin/nest_vars.sh; \
  export PYTHONPATH=/tiago_public_ws/src/tiago_ring_controller/src:$PYTHONPATH; \
  cd /tiago_public_ws/src/tiago_ring_controller && python3 -B -m unittest discover -s test -p "test_*.py"'
```

### 3.3 Implementing or modifying a model or architecture

- **Change parameters or wiring only:** open a template in the editor, edit,
  *Save* to a new `*.graph.json`, run it (section 2 step 5); or do the same in
  Python (3.1). Every parameter of every block is declared with type, default,
  bounds and unit (`docs/blocks/blocks.md`); the file format is
  `docs/blocks/graph-schema.md`.
- **New architecture from existing blocks:** write a template function in
  `src/tiago_ring_controller/graph/templates.py` (it may only call `Graph.add`
  and `Graph.connect`), register it in `TEMPLATES`, regenerate the example
  files with `write_examples()`, and add a gate test (a parity test against a
  reference if one exists, otherwise a fake-engine run).
- **New block type:** follow `docs/blocks/adding-a-block.md` (declare a
  `ParamSchema`, write the class with typed ports, build through a vectorised
  builder in `nest/populations.py`, register it, test it on the fake NEST and
  with real NEST). The palette, the block reference and validation follow from
  the registry.
- **Changing the science of an existing block** (weights, neuron parameters,
  feedback rules) needs an equivalence gate and a re-baselined golden; the
  precedent is `docs/blocks/equivalence.md`.

## 4. Repository layout

| Path | Purpose |
| --- | --- |
| `src/tiago_ring_controller/` | The ROS package: `src/tiago_ring_controller/` (importable code), `scripts/`, `test/`, `src/config/` (scientific inputs), `legacy/` (frozen original scripts), `launch/`, `worlds/` |
| `docs/blocks/` | Block architecture: map, reference, file format, measurements, progress log |
| `docs/cosim/` | The co-simulation loop: assessment of the original loop, designs, progress |
| `docs/legacy.md` | Everything about the original scripts and the frozen baseline |
| `docker/`, `run_model_docker.bash`, `overrides/` | Container launcher, environment, pinned image, files mounted over the upstream TIAGo stack |
| `.runtime/` (ignored) | Catkin overlay, container home, logs |

Container details: the checkout is mounted at `/model_repo` and the package at
`/tiago_public_ws/src/tiago_ring_controller`; every launch does an incremental
`catkin build` in `.runtime/catkin_ws`; `ROS_MASTER_URI` defaults to
`http://localhost:11311`; `--headless` disables display forwarding, `--software`
disables NVIDIA acceleration; commands after `--` run instead of the shell
(`./run_model_docker.bash --headless --software -- python3 -c 'import nest'`).
For a clean build remove `.runtime/catkin_ws` with no container running. The
image is pinned by digest in `docker/image.env`; changing system dependencies
means a new image and pin.

## 5. Status and open items

Implemented and tested (container suite 235 tests, host 218): phases 1–7 of
`docs/blocks/plan.md`, with each phase's gate output in `docs/blocks/progress.md`.
Open: every gate that needs Gazebo has been run by hand at most once, never by a
test (`docs/cosim/README.md`, "Gazebo checks still open"); the feedback sweep
(`docs/blocks/feedback.md`) was done on the fake robot and must be repeated in
Gazebo; the comparator produces no drive below about 18 ring indices of goal
distance; `TaskGain` is a stub.

## 6. Documentation index

- `docs/blocks/architecture.md` — the map of layers, templates and open items.
- `docs/blocks/blocks.md` — block reference (generated from the code).
- `docs/blocks/graph-schema.md` — the graph file format and its rules.
- `docs/blocks/adding-a-block.md` — how to add a block.
- `docs/blocks/equivalence.md` — vectorised model vs the original, measured.
- `docs/blocks/feedback.md` — encoder modes, goal distance, multi-joint.
- `docs/blocks/plan.md`, `docs/blocks/progress.md` — the plan and the dated record.
- `docs/cosim/README.md`, `assessment.md`, `plan-b-inhouse-loop.md`, `plan-a-nrp-core.md` — the loop.
- `src/tiago_ring_controller/README.md` — package-level commands and tests.
- `src/tiago_ring_controller/docs/robot_safety.md` — motion-capable interfaces and gates.
- `src/tiago_ring_controller/docs/architecture.md`, `artifacts.md`, `paper_traceability.md`, `reproducibility.md` — the original model and its artifacts.
- `docs/legacy.md` — the original scripts, the frozen baseline, recovery checks.

## 7. Contributing

- `main` is the frozen baseline behind the published results (tag
  `legacy-loop-baseline`); nothing is committed to it directly.
- New work goes on a feature branch cut from the current integration branch
  (`git checkout -b my-feature blocks-refactor`, or from `main` once the PR has
  merged). Keep a branch to one topic; put the design or plan in `docs/` on the
  same branch so the *why* travels with the *what*.
- Before opening a PR: the container suite green (section 3.2), the host suite
  green apart from known environment gaps, new behaviour covered by a test on the
  fakes and, when NEST is involved, a real-NEST gate with a documented tolerance;
  any run against Gazebo recorded in the relevant doc with its actual output.
- Open a pull request against `main` (or the integration branch while one is
  open) with the summary of what changed, the test output, and what was not run
  and why. Review, then merge with the branch history kept; delete the branch
  after the merge. Scientific changes (weights, neuron parameters, feedback rules)
  also need a re-baselined golden and an entry in `docs/blocks/equivalence.md`.
