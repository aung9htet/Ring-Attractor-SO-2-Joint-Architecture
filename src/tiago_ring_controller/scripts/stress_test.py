#!/usr/bin/env python3
"""Stress test: a random joint, the whole arm in a random pose, a random goal (docs/stress-test.md).

    python3 -B scripts/stress_test.py plan --trials 300 --seed 1 --screen-moveit --out /out/gazebo_plan.json
    python3 -B scripts/stress_test.py run --plan /out/gazebo_plan.json --system F --engines full --out /out/gazebo_v4
    python3 -B scripts/stress_test.py run --plan /out/gazebo_plan.json --system PID --engines full --out /out/gazebo_pid
    python3 -B scripts/stress_test.py report --data /out --out /out/report

``plan`` draws the trials from one seed: a joint 0-6, a start and a goal
anywhere in its range (limits included, no minimum distance), an angle for each
of the other six joints, and a network seed.  With ``--screen-moveit`` (needs
the simulation) a draw is thrown away and drawn again when MoveIt reports a
self-collision in the random pose, on the straight path to it from the reset
pose, or on the path of the driven joint from start to goal.

``run`` runs a plan with one system: ``F`` (v4), ``B`` (v3 on the bounded
ring), ``G``, ``T``, ``PID`` (the PID baseline: the frozen PID collector of the
draft, Gazebo only), or the upper references of ``scripts/run_baselines.py``
(``P``: the tuned controller; ``N``: the robot's own controller, Gazebo only).  After
the reset motion the arm is moved to the trial's pose with one smooth
trajectory, then the controller drives the one joint.  Results are written
after every trial and finished trials are skipped, so a run can be resumed, or
split over several simulations with ``--shard I/N`` (the same ``--out``).
``--engines nest`` uses the simulated joints, which are independent of each
other: there the pose of the other joints changes nothing.

``report`` reads every result under ``--data`` (folders named
``<nest|gazebo>_<label>``) and writes ``tables.md`` and the figures: goal error
per joint and per move distance (bars), joint by move distance and start by
goal (heat maps).

The Gazebo runs are meant for the world without the table:
``docker/batch.bash up NAME --world tiago_ring_stress``.
"""

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
LEGACY = os.path.join(os.path.dirname(HERE), "legacy")
for path in (SRC, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from tiago_ring_controller.control.mapping import RingMapper  # noqa: E402
from tiago_ring_controller.evaluation.motion import SETTLE_TOLERANCE  # noqa: E402
from tiago_ring_controller.evaluation.serialization import json_safe  # noqa: E402
from tiago_ring_controller.graph.templates import BOUNDED, joint_limits  # noqa: E402

JOINTS = 7
SYSTEMS = ("F", "B", "G", "T", "PID", "P", "N")
#: move distance as a fraction of the joint range: bin edges and names
DISTANCE_EDGES = (0.0, 0.05, 0.15, 0.30, 0.50, 1.0)
DISTANCE_NAMES = ("under 5%", "5-15%", "15-30%", "30-50%", "over 50%")
#: result label (folder name after ``nest_`` / ``gazebo_``) -> key and name used in the report
LABELS = {"pid": ("PID", "PID baseline"), "p_controller": ("P", "upper reference: tuned controller"),
          "native": ("N", "upper reference: robot's own controller"),
          "v3": ("B3", "v3 (two rings)"), "v4": ("F3", "v4 (three rings)")}


# -- plan --------------------------------------------------------------------
def radians_per_index(joint):
    lo, hi = joint_limits(joint)
    return RingMapper("bounded", lo, hi, 200, 5, BOUNDED["arc_fraction"]).radians_per_index


def draw_trial(rng, limits):
    """One random trial: joint, start, goal, the other joints' angles, a network seed."""

    joint = int(rng.integers(0, JOINTS))
    fractions = rng.random(JOINTS)                      # where every joint starts, as a fraction of its range
    goal_fraction = float(rng.random())
    pose = [lo + float(f) * (hi - lo) for f, (lo, hi) in zip(fractions, limits)]
    lo, hi = limits[joint]
    return {"joint": joint, "start_fraction": float(fractions[joint]), "goal_fraction": goal_fraction,
            "start_rad": pose[joint], "goal_rad": lo + goal_fraction * (hi - lo), "pose": pose,
            "seed": int(rng.integers(1, 2 ** 31 - 1))}


def make_plan(trials, seed, accept=None, limits=None, max_draws=None):
    """``trials`` accepted draws.  ``accept(trial)`` False rejects a draw (MoveIt screening)."""

    limits = limits or [joint_limits(j) for j in range(JOINTS)]
    rng = np.random.default_rng(seed)
    accepted, rejected = [], 0
    max_draws = max_draws or 200 * trials
    while len(accepted) < trials:
        if len(accepted) + rejected >= max_draws:
            raise RuntimeError("only %d of %d trials accepted in %d draws" % (len(accepted), trials, max_draws))
        trial = draw_trial(rng, limits)
        if accept is not None and not accept(trial):
            rejected += 1
            continue
        trial["id"] = len(accepted)
        accepted.append(trial)
    return {"seed": int(seed), "screened": accept is not None, "rejected": rejected,
            "limits": [list(pair) for pair in limits], "trials": accepted}


def moveit_accept(screen, reset_pose):
    """The three checks of one trial: the pose, the way to it, the driven joint's path."""

    def accept(trial):
        pose = list(trial["pose"])
        goal_pose = list(pose)
        goal_pose[trial["joint"]] = trial["goal_rad"]
        return screen.pose_path_is_valid(reset_pose, pose) and screen.pose_path_is_valid(pose, goal_pose)

    return accept


def plan_command(args):
    accept = None
    if args.screen_moveit:
        from tiago_ring_controller.cosim.gazebo_ros_engine import RospyTransport
        from validate_velocity import START_POSE, MoveItScreen

        transport = RospyTransport(node_name="stress_plan")
        transport.initialize()
        accept = moveit_accept(MoveItScreen(transport), START_POSE)
    plan = make_plan(args.trials, args.seed, accept)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=1)
    per_joint = [sum(t["joint"] == j for t in plan["trials"]) for j in range(JOINTS)]
    print("plan: %d trials (per joint %s), %d draws rejected%s -> %s"
          % (len(plan["trials"]), per_joint, plan["rejected"], " by MoveIt" if plan["screened"] else " (not screened)", args.out))
    return 0


# -- run ---------------------------------------------------------------------
def results_path(out, shard):
    return os.path.join(out, "results.json" if shard is None else "results_shard%d_of_%d.json" % shard)


def load_results(out):
    """Every trial row found under ``out`` (all shards), by trial id."""

    rows, failed = {}, []
    for path in sorted(glob.glob(os.path.join(out, "results*.json"))):
        with open(path, encoding="utf-8") as stream:
            document = json.load(stream)
        for row in document["trials"]:
            rows[int(row["trial"])] = row
        failed.extend(document.get("failed", []))
    return rows, failed


class GraphRunner:
    """v3 / v4 through ``validate_velocity.run_trial``."""

    def __init__(self, system, engines, time_scale=None):
        from validate_velocity import run_trial

        self.system, self.engines, self.time_scale, self.run_trial = system, engines, time_scale, run_trial
        self.transport = None
        if engines == "full":
            from tiago_ring_controller.cosim.gazebo_ros_engine import RospyTransport

            self.transport = RospyTransport(node_name="stress_test")
            self.transport.initialize()

    def run(self, trial):
        joint = trial["joint"]
        others = {j: angle for j, angle in enumerate(trial["pose"]) if j != joint}
        row = self.run_trial(joint, self.system, trial["start_rad"], trial["goal_rad"], self.engines, self.transport,
                             trial["seed"], self.time_scale, other_joints=others)
        return row

    def close(self):
        if self.transport is not None:
            try:
                self.transport.unpause()
            finally:
                self.transport.shutdown()


class PidRunner:
    """The frozen PID collector: its reset motion, then the arm moved to the trial's pose, then its own ``run_trial``."""

    def __init__(self):
        import importlib

        import rospy

        from run_legacy_trials import CONTROLLERS, BaseKeeper, _NoLogger, legacy_metrics
        from tiago_ring_controller.cosim.gazebo_ros_engine import GazeboRosEngine, RospyTransport

        if LEGACY not in sys.path:
            sys.path.insert(0, LEGACY)
        module_name, class_name, _ = CONTROLLERS["pid"]
        self.module = importlib.import_module(module_name)
        self.module.TiagoSubscriber = _NoLogger
        self.class_name, self.metrics, self.rospy = class_name, legacy_metrics, rospy
        rospy.init_node("stress_test_pid", anonymous=True)
        self.base = BaseKeeper()
        transport = RospyTransport(init_node=False)
        transport.initialize()
        self.mover = GazeboRosEngine(transport, base_pose=None)     # only its move_to(): one smooth trajectory
        self.collectors = {}
        os.chdir(SRC)                                               # the legacy code reads ./config/

    def collector(self, joint):
        if joint not in self.collectors:
            out = os.path.join("/tmp", "stress_pid")

            class Collector(getattr(self.module, self.class_name)):
                def _output_dir(self):
                    os.makedirs(out, exist_ok=True)
                    return out

            self.collectors[joint] = Collector(joint, use_manual_limits=False)
        return self.collectors[joint]

    def run(self, trial):
        joint = trial["joint"]
        collector = self.collector(joint)
        started = time.monotonic()
        collector.publisher.reset_pose()
        found = self.base.restore()
        self.mover.move_to(dict(enumerate(trial["pose"])))
        collector.publisher.prime_command_state(force=True)
        base_start = self.base.read()
        result = collector.run_trial(float(trial["goal_rad"]))
        base_end = self.base.read()
        row = self.metrics(result, collector.joint_min, collector.joint_max, dt_s=getattr(collector, "time_step", 50.0) / 1000.0)
        row.update(system="PID", joint=joint, wall_s=time.monotonic() - started, rho=radians_per_index(joint), clamp_ticks=0,
                   base_shift_m=self.base.distance(base_start, base_end), base_found_m=self.base.distance(found, self.base.pose))
        return row

    def close(self):
        pass


class BaselineRunner:
    """The upper references of ``scripts/run_baselines.py``: ``P`` (the tuned controller), ``N`` (the robot's own controller)."""

    def __init__(self, system, engines):
        import run_baselines

        self.module, self.controller = run_baselines, {"P": "p", "N": "native"}[system]
        self.transport = None
        if engines == "full":
            from tiago_ring_controller.cosim.gazebo_ros_engine import RospyTransport

            self.transport = RospyTransport(node_name="stress_test")
            self.transport.initialize()
        self.engine = run_baselines.make_engine(engines, self.transport)

    def run(self, trial):
        return self.module.run_trial(self.engine, self.controller, trial["joint"], trial["goal_rad"], dict(enumerate(trial["pose"])))

    def close(self):
        if self.transport is not None:
            try:
                self.transport.unpause()
            finally:
                self.transport.shutdown()


def run_command(args):
    from tiago_ring_controller.cosim.engine import EngineError

    with open(args.plan, encoding="utf-8") as stream:
        plan = json.load(stream)
    shard = None
    if args.shard:
        index, _, count = args.shard.partition("/")
        shard = (int(index), int(count))
        if not 1 <= shard[0] <= shard[1]:
            raise SystemExit("--shard I/N needs 1 <= I <= N")
    if args.system == "PID" and args.engines != "full":
        raise SystemExit("the PID collector only runs against Gazebo (--engines full)")
    if args.system == "N" and args.engines != "full":
        raise SystemExit("the robot's own controller only exists in Gazebo (--engines full)")
    os.makedirs(args.out, exist_ok=True)
    done, _ = load_results(args.out)
    trials = [t for t in plan["trials"][:args.limit] if shard is None or t["id"] % shard[1] == shard[0] - 1]
    path = results_path(args.out, shard)
    mine = [row for row in done.values() if shard is None or row["trial"] % shard[1] == shard[0] - 1]
    failed = []
    if args.system == "PID":
        runner = PidRunner()
    elif args.system in ("P", "N"):
        runner = BaselineRunner(args.system, "fake" if args.engines == "nest" else args.engines)
    else:
        runner = GraphRunner(args.system, args.engines, args.time_scale)
    started = time.monotonic()

    def write():
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(json_safe({"system": args.system, "engines": args.engines, "plan": os.path.basename(args.plan),
                                 "plan_seed": plan["seed"], "trials": sorted(mine, key=lambda r: r["trial"]), "failed": failed,
                                 "measured": time.strftime("%Y-%m-%d %H:%M:%S")}), stream)

    try:
        for trial in trials:
            if trial["id"] in done:
                continue
            try:
                row = runner.run(trial)
            except EngineError as exc:
                failed.append({"trial": trial["id"], "joint": trial["joint"], "error": str(exc)})
                print("trial %d: FAILED, %s" % (trial["id"], exc), flush=True)
                continue
            if not args.keep_trace:
                row.pop("trace", None)
            row.update(trial=trial["id"], start_fraction=trial["start_fraction"], goal_fraction=trial["goal_fraction"],
                       planned_start_rad=trial["start_rad"], pose=trial["pose"])
            mine.append(row)
            write()
            start = row.get("start_rad", row.get("q_start"))
            final = row.get("final_rad", row.get("q_final"))
            print("trial %3d %s arm_%d: start %+.3f (planned %+.3f) goal %+.3f final %+.3f error %+.3f%s"
                  % (trial["id"], args.system, trial["joint"] + 1, start, trial["start_rad"], trial["goal_rad"], final, row["goal_error_rad"],
                     ", BUMP LOST" if row.get("lost_bump") else ""), flush=True)
    finally:
        write()
        runner.close()
    errors = np.abs([r["goal_error_rad"] for r in mine]) if mine else np.array([np.nan])
    print("%s on %s: %d trials (%d failed) in %.0f s, goal error median %.3f p90 %.3f max %.3f rad -> %s"
          % (args.system, args.engines, len(mine), len(failed), time.monotonic() - started, np.median(errors),
             np.percentile(errors, 90), errors.max(), path))
    return 0


# -- report ------------------------------------------------------------------
def load_set(data, prefix):
    """``{key: [records]}`` of the result folders ``<prefix>_<label>`` under ``data``."""

    import results_report

    systems, failed = {}, {}
    for label, (key, _) in LABELS.items():
        rows, fails = load_results(os.path.join(data, "%s_%s" % (prefix, label)))
        if not rows:
            continue
        records = []
        for raw in rows.values():
            record = results_report.normalise(raw, key)
            record.update(pair_key=int(raw["trial"]), trial=int(raw["trial"]), start_fraction=float(raw["start_fraction"]), goal_fraction=float(raw["goal_fraction"]),
                          distance=abs(float(raw["goal_fraction"]) - float(raw["start_fraction"])),
                          start_miss=abs(record["start"] - float(raw["planned_start_rad"])))
            records.append(record)
        systems[key], failed[key] = sorted(records, key=lambda r: r["trial"]), len(fails)
    return systems, failed


def distance_bin(distance):
    return int(min(np.searchsorted(DISTANCE_EDGES, distance, side="right") - 1, len(DISTANCE_NAMES) - 1))


def median_error(rows, index=False):
    values = [abs(r["error"]) / (r["rho"] if index else 1.0) for r in rows]
    return float(np.median(values)) if values else float("nan")


def joint_distance_grid(rows):
    """Median goal error (rad) and trial count per joint (rows) and move-distance bin (columns)."""

    grid = np.full((JOINTS, len(DISTANCE_NAMES)), np.nan)
    count = np.zeros_like(grid, dtype=int)
    for joint in range(JOINTS):
        for column in range(len(DISTANCE_NAMES)):
            cell = [r for r in rows if r["joint"] == joint and distance_bin(r["distance"]) == column]
            count[joint, column] = len(cell)
            if cell:
                grid[joint, column] = median_error(cell)
    return grid, count


def start_goal_grid(rows, bins):
    """Median goal error in ring indices per start (rows) and goal (columns) position, joints pooled."""

    grid = np.full((bins, bins), np.nan)
    count = np.zeros((bins, bins), dtype=int)
    cell_of = lambda fraction: int(min(fraction * bins, bins - 1))  # noqa: E731
    cells = {}
    for r in rows:
        cells.setdefault((cell_of(r["start_fraction"]), cell_of(r["goal_fraction"])), []).append(r)
    for (row, column), cell in cells.items():
        grid[row, column], count[row, column] = median_error(cell, index=True), len(cell)
    return grid, count


def system_table(systems, failed):
    import results_report as rr

    lines = ["| system | trials | goal error, median / p90 / max (rad) | in ring indices | within %.2f rad | settled | settling time, median (s) "
             "| overshoot, p90 / max (rad) | bumps lost | wrong way | no drive | clamped | base moved over 1 cm | start off by over 0.02 rad | reset failed |"
             % SETTLE_TOLERANCE, "|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, (key, name) in LABELS.items():
        rows = systems.get(key)
        if not rows:
            continue
        over = [r["overshoot"] for r in rows if np.isfinite(r["overshoot"])]
        lines.append("| %s | %d | %s | %s | %d%% | %d%% | %s | %.3f / %.3f | %d | %d | %d | %d | %d | %d | %d |" % (
            name, len(rows), rr.fmt3([r["error"] for r in rows]), rr.fmt3([r["error"] / r["rho"] for r in rows], 2),
            round(100 * np.mean([abs(r["error"]) < SETTLE_TOLERANCE for r in rows])), round(100 * np.mean([r["reached"] for r in rows])),
            rr.median([r["settle"] for r in rows]), np.percentile(over, 90), max(over),
            sum(r["lost"] for r in rows), sum(r["wrong"] for r in rows), sum(r["no_drive"] for r in rows), sum(r["clamp"] for r in rows),
            sum(r["base_shift"] > 0.01 for r in rows), sum(r["start_miss"] > 0.02 for r in rows), failed.get(key, 0)))
    return lines


def grouped_table(systems, groups, group_of, header):
    present = [(key, name) for key, name in LABELS.values() if key in systems]
    lines = ["| %s | trials | %s |" % (header, " | ".join(name for _, name in present)), "|---|---:|" + "---|" * len(present)]
    for index, group in enumerate(groups):
        cells, n = [], 0
        for key, _ in present:
            rows = [r for r in systems[key] if group_of(r) == index]
            n = max(n, len(rows))
            pair = tuple(np.percentile([abs(r["error"]) for r in rows], [50, 90])) if rows else None
            cells.append("-" if pair is None else ("%.4f / %.4f" if pair[0] < 0.0005 else "%.3f / %.3f") % pair)
        lines.append("| %s | %d | %s |" % (group, n, " | ".join(cells)))
    return lines


def worst_table(rows, plan, count=10):
    poses = {t["id"]: t for t in plan["trials"]} if plan else {}
    lines = ["| trial | joint | start (rad) | goal (rad) | final (rad) | error (rad) | error (indices) | move (% of range) | steps | settled | other joints (rad) |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in sorted(rows, key=lambda r: -abs(r["error"]))[:count]:
        pose = poses.get(r["trial"], {}).get("pose")
        others = " ".join("%+.2f" % a for j, a in enumerate(pose) if j != r["joint"]) if pose else "-"
        lines.append("| %d | arm_%d | %+.3f | %+.3f | %+.3f | %+.3f | %.1f | %.0f | %d | %s | %s |" % (
            r["trial"], r["joint"] + 1, r["start"], r["goal"], r["final"], r["error"], abs(r["error"]) / r["rho"], 100 * r["distance"],
            r["steps"], "yes" if r["reached"] else "no", others))
    return lines


def glossary():
    """How every number and percentage in the tables and figures is calculated."""

    edges = ["%d%%" % round(100 * e) for e in DISTANCE_EDGES]
    return [
        "## How the numbers are calculated", "",
        "**Angles as a percentage of the joint range.** Every joint has a lower and an upper limit (`velocity_calibration.json`). "
        "A position is written as a share of that range: `share = (angle - lower limit) / (upper limit - lower limit)`. "
        "0% is the lower limit, 100% the upper limit, 50% the middle. Example: arm_6 goes from -1.417 to +1.417 rad, so +0.709 rad is "
        "(0.709 + 1.417) / 2.834 = 75%. Shares make joints with different ranges comparable (arm_3 spans 4.8 rad, arm_6 2.8 rad).", "",
        "**Start and goal.** Each trial draws the start share and the goal share of the driven joint independently and uniformly "
        "between 0% and 100%. The planned start is where the arm is put before the trial; the tables use the angle actually measured "
        "at the first tick.", "",
        "**Move distance.** `|goal share - start share|`, the part of the joint's range the move has to cover. A move from 20% to 65% "
        "is a 45% move, whichever joint it is. Trials are grouped into five bins: " + ", ".join(
            "%s (%s to %s)" % (name, lo, hi) for name, lo, hi in zip(DISTANCE_NAMES, edges, edges[1:])) +
        ". A value on a bin edge goes into the higher bin. Because start and goal are drawn independently, short moves are rarer than "
        "long ones, so the bins hold different numbers of trials (given in the tables).", "",
        "**Goal error (rad).** `|goal angle - final joint angle|`, the final angle being measured after the trial has ended and the "
        "joint has stopped. Always positive.", "",
        "**Goal error in ring indices.** The goal error divided by the joint's radians per ring index (`rho`): how many neurons of the "
        "200-neuron ring the joint is away from the goal. On the bounded ring the joint range lies on 140 indices, so "
        "`rho = range / 140`: 0.020 rad for arm_6, 0.034 rad for arm_3. An error of 0.5 index is 0.010 rad on arm_6 and 0.017 rad on "
        "arm_3. PID has no ring; its error is divided by the same `rho` only so that it can be shown on the same scale.", "",
        "**Median, p90, max.** Of the trials in a group, sorted by goal error: the median is the middle one (half the trials are "
        "better), p90 the value that 90% of the trials stay under (one trial in ten is worse), max the single worst trial.", "",
        "**Within %.2f rad (%%).** Trials whose goal error is under %.2f rad, divided by all trials of that system, times 100."
        % (SETTLE_TOLERANCE, SETTLE_TOLERANCE), "",
        "**Settled (%%).** Trials in which the error stayed under %.2f rad for 5 consecutive samples (0.25 s) at some point during the "
        "trial, divided by all trials, times 100. A trial can count as settled and still end outside %.2f rad if the joint drifts out "
        "again afterwards (this happens with PID); 'within %.2f rad' is about the end of the trial."
        % (SETTLE_TOLERANCE, SETTLE_TOLERANCE, SETTLE_TOLERANCE), "",
        "**Settling time.** The first time from which the error stays under %.2f rad for 5 samples; the median over the settled trials."
        % SETTLE_TOLERANCE, "",
        "**Overshoot.** The furthest the joint went past the goal in its direction of travel; 0 if it never passed it.", "",
        "**Counts.** *Bumps lost*: the predictive (v4) ring went silent. *Wrong way*: the joint moved over 0.03 rad away from the goal. "
        "*No drive*: the joint moved less than 0.03 rad in total (this includes trials whose goal was that close). *Clamped*: the "
        "command hit the joint's velocity limit or a joint limit in at least one tick. *Base moved*: the robot's base shifted over "
        "1 cm during the trial. *Start off*: the arm was more than 0.02 rad from the planned start. *Reset failed*: the reset motion "
        "could not be run and the trial was skipped.", ""]


def figure_notes(prefix, bins, names, trials):
    """The four figures of one simulator, each with what its axes, bars, cells and colours mean."""

    panels = ", ".join(names)
    step = 100.0 / bins
    return [
        "### Figures", "",
        "#### Bar plot: goal error per joint", "",
        "![goal error per joint](figures/%s_error_by_joint.png)" % prefix, "",
        "- **Horizontal axis:** the driven joint, `arm_1` to `arm_7`. Each trial drives exactly one joint, so every trial is in one group.",
        "- **Vertical axis:** goal error in rad.",
        "- **Bar:** the median goal error of that system's trials on that joint. **Whisker** (the line with a cap above the bar): "
        "the p90 of the same trials.",
        "- **Colours:** one per system (%s), the same in every figure." % panels,
        "- No percentage is involved; the number of trials per joint is in the per-joint table above.", "",
        "#### Bar plot: goal error by move distance", "",
        "![goal error by move distance](figures/%s_error_by_distance.png)" % prefix, "",
        "- **Horizontal axis:** the move distance bin. The percentage is the move as a share of the joint's range, "
        "`|goal share - start share| x 100`: 'under 5%' holds the moves that cover less than a twentieth of the range, 'over 50%' "
        "the moves across more than half of it. All joints are pooled.",
        "- **Vertical axis:** goal error in rad.",
        "- **Bar and whisker:** median and p90 of the trials in that bin, per system.", "",
        "#### Heat map: joint by move distance", "",
        "![joint by move distance](figures/%s_heat_joint_distance.png)" % prefix, "",
        "- **One panel per system** (%s), all with the same colour scale so the panels can be compared directly." % panels,
        "- **Rows (vertical axis):** the driven joint. **Columns (horizontal axis):** the move distance bin, as in the bar plot above.",
        "- **Cell:** the median goal error, in rad, of the trials of that joint whose move fell into that bin. The number is printed "
        "in the cell; the colour shows the same value (light = small error, dark = large; the bar on the right gives the scale, "
        "starting at 0). A blank cell had no trial.",
        "- With seven joints and five bins the %d trials are spread over 35 cells: about %d trials per cell on average, fewer in "
        "the 'under 5%%' column. The fewer trials a cell has, the rougher its median." % (trials, round(trials / 35.0)), "",
        "#### Heat map: start by goal", "",
        "![start by goal](figures/%s_heat_start_goal.png)" % prefix, "",
        "- **One panel per system**, same colour scale.",
        "- **Rows (vertical axis):** where the driven joint started, as a share of its range, in %d bands of %.3g%% "
        "(bottom row = starts near the lower limit, top row = near the upper limit)." % (bins, step),
        "- **Columns (horizontal axis):** where the goal was, as a share of the range, in the same bands "
        "(left = goals near the lower limit, right = near the upper limit). The axis labels are rounded down to whole percent.",
        "- **Cell:** the median goal error of the trials with that start band and that goal band, all joints pooled, "
        "in **ring indices** (not rad), so that joints with different ranges can share a cell. About %d trials per cell."
        % round(trials / float(bins * bins)),
        "- **Reading the position of a cell:** the diagonal from bottom left to top right holds the short moves (start and goal in "
        "the same band); the top-left and bottom-right corners hold the longest moves (from one end of the range to the other). "
        "Cells above the diagonal are moves downward (goal below start), cells below it moves upward.", ""]


def sequential_map():
    from matplotlib.colors import LinearSegmentedColormap

    # one hue, light to dark (the reference sequential ramp, steps 100-700)
    return LinearSegmentedColormap.from_list("blue_ramp", ["#eef5fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def heat_panels(plt, grids, counts, names, row_labels, column_labels, title, xlabel, ylabel, unit, fmt, origin_lower=False, width=4.3):
    """One annotated heat map per system on a shared colour scale; empty cells stay blank."""

    import results_report as rr

    top = float(np.nanmax([np.nanmax(g) for g in grids if np.isfinite(g).any()]))
    figure, axes = plt.subplots(1, len(grids), figsize=(max(width * len(grids) + 1.0, 6.0), 0.5 * len(row_labels) + 2.4), squeeze=False)
    cmap = sequential_map()
    cmap.set_bad(rr.SURFACE)
    image = None
    for axis, grid, count, name in zip(axes[0], grids, counts, names):
        image = axis.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0.0, vmax=top, aspect="auto", origin="lower" if origin_lower else "upper")
        for (row, column), value in np.ndenumerate(grid):
            if np.isfinite(value):
                axis.text(column, row, fmt % value, ha="center", va="center", fontsize=8,
                          color="#ffffff" if value > 0.55 * top else rr.INK)
        axis.set_xticks(range(len(column_labels)))
        axis.set_xticklabels(column_labels, fontsize=8)
        axis.set_yticks(range(len(row_labels)))
        axis.set_yticklabels(row_labels if axis is axes[0][0] else [], fontsize=8.5)
        axis.set_xlabel(xlabel, color=rr.INK, fontsize=9)
        axis.set_title("%s (%d trials)" % (name, int(count.sum())), loc="left", fontsize=10, color=rr.INK)
        axis.tick_params(colors=rr.MUTED, length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        # a thin surface gap between cells
        axis.set_xticks(np.arange(-0.5, len(column_labels)), minor=True)
        axis.set_yticks(np.arange(-0.5, len(row_labels)), minor=True)
        axis.grid(which="minor", color=rr.SURFACE, linewidth=2)
        axis.tick_params(which="minor", length=0)
    axes[0][0].set_ylabel(ylabel, color=rr.INK, fontsize=9)
    figure.suptitle(title, x=0.01, y=0.99, ha="left", va="top", fontsize=10.5, color=rr.INK, linespacing=1.4)
    figure.tight_layout(rect=(0, 0, 0.93, 0.93))
    bar = figure.colorbar(image, cax=figure.add_axes([0.945, 0.2, 0.012, 0.6]))
    bar.set_label(unit, color=rr.INK, fontsize=8.5)
    bar.ax.tick_params(colors=rr.MUTED, labelsize=8)
    bar.outline.set_visible(False)
    figure.patch.set_facecolor(rr.SURFACE)
    return figure


def distance_bars(plt, systems, title):
    import results_report as rr

    figure, axis = plt.subplots(figsize=(9.2, 4.2))
    present = [k for k in rr.FIGURE_SYSTEMS if k in systems]
    width = 0.8 / max(len(present), 1)
    for n, key in enumerate(present):
        for column in range(len(DISTANCE_NAMES)):
            rows = [r for r in systems[key] if distance_bin(r["distance"]) == column]
            if not rows:
                continue
            x = column + (n - (len(present) - 1) / 2) * width
            median, p90 = np.percentile([abs(r["error"]) for r in rows], [50, 90])
            axis.bar(x, median, width - 0.03, color=rr.COLORS[key], label=rr.FIGURE_NAMES[key] if column == 0 else None)
            axis.plot([x, x], [median, p90], color=rr.INK, linewidth=1)
            axis.plot([x - 0.06, x + 0.06], [p90, p90], color=rr.INK, linewidth=1)
    axis.set_xticks(range(len(DISTANCE_NAMES)))
    axis.set_xticklabels(DISTANCE_NAMES)
    axis.set_xlabel("move distance, as a share of the joint range", color=rr.INK)
    axis.set_ylabel("goal error (rad): bar = median, whisker = p90", color=rr.INK)
    axis.set_title(title, loc="left", fontsize=10.5, color=rr.INK)
    axis.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    rr.style(axis)
    return figure


def save(figure, out, name):
    import results_report as rr

    figure.patch.set_facecolor(rr.SURFACE)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(os.path.join(out, "figures", "%s.%s" % (name, extension)), dpi=160, facecolor=rr.SURFACE)


def report_command(args):
    import results_report as rr

    out = ["# Stress test tables", "",
           "Written by `scripts/stress_test.py report`; the reading is in [`../../stress-test.md`](../../stress-test.md). "
           "Every trial: a random joint, a random start and goal anywhere in its range, the other six joints at random angles, "
           "a random network seed. Goal error is |goal - final joint angle|; settled: under %.2f rad for 5 samples." % SETTLE_TOLERANCE, ""]
    out += glossary()
    figures = 0
    plt = None
    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    for prefix, heading, bins in (("gazebo", "Gazebo (world without the table)", 5), ("nest", "NEST with the simulated joints", 8)):
        systems, failed = load_set(args.data, prefix)
        if not systems:
            out += ["## %s" % heading, "", "No results under `%s_*`." % prefix, ""]
            continue
        plan_path = os.path.join(args.data, "%s_plan.json" % prefix)
        plan = None
        if os.path.exists(plan_path):
            with open(plan_path, encoding="utf-8") as stream:
                plan = json.load(stream)
        out += ["## %s" % heading, ""]
        if plan:
            out += ["Plan: %d trials, seed %d, %s." % (len(plan["trials"]), plan["seed"],
                    "%d draws rejected by MoveIt" % plan["rejected"] if plan["screened"] else "not screened (no collisions in this simulator)"), ""]
        out += system_table(systems, failed)
        if "F3" in systems and len(systems) > 1:
            out += ["", "### v4 against each other system on the same trials (paired)", "",
                    "Per trial, v4's goal error minus the other system's. 'Closer in': the share of trials in which that system ended "
                    "nearer the goal. Sign test p under 0.05 is read as a real difference.", ""]
            out += rr.paired_table(systems, {key: name for key, name in LABELS.values()})
        out += ["", "### Goal error per joint, median / p90 (rad)", ""]
        out += grouped_table(systems, ["arm_%d" % (j + 1) for j in range(JOINTS)], lambda r: r["joint"], "joint")
        out += ["", "### Goal error by move distance, median / p90 (rad)", ""]
        out += grouped_table(systems, DISTANCE_NAMES, lambda r: distance_bin(r["distance"]), "move, share of the range")
        if "F3" in systems:
            out += ["", "### The ten largest v4 errors", ""] + worst_table(systems["F3"], plan)
        out.append("")
        if plt is not None:
            present = [(key, name) for key, name in LABELS.values() if key in systems and key in rr.FIGURE_SYSTEMS]
            median = lambda rows: float(np.median([abs(r["error"]) for r in rows]))  # noqa: E731
            p90 = lambda rows: float(np.percentile([abs(r["error"]) for r in rows], 90))  # noqa: E731
            figure = rr.grouped_bars(plt, systems, median, p90, "goal error (rad): bar = median, whisker = p90",
                                     "Stress test, %s: goal error per joint" % heading)
            figure.tight_layout()
            save(figure, args.out, "%s_error_by_joint" % prefix)
            figure = distance_bars(plt, systems, "Stress test, %s: goal error by move distance" % heading)
            figure.tight_layout()
            save(figure, args.out, "%s_error_by_distance" % prefix)
            grids = [joint_distance_grid(systems[key]) for key, _ in present]
            save(heat_panels(plt, [g for g, _ in grids], [c for _, c in grids], [name for _, name in present],
                             ["arm_%d" % (j + 1) for j in range(JOINTS)], DISTANCE_NAMES,
                             "Median goal error per joint and move distance\n%s" % heading,
                             "move distance, share of the joint range", "", "median goal error (rad)", "%.3f"), args.out, "%s_heat_joint_distance" % prefix)
            grids = [start_goal_grid(systems[key], bins) for key, _ in present]
            ticks = ["%d-%d%%" % (100 * k // bins, 100 * (k + 1) // bins) for k in range(bins)]
            save(heat_panels(plt, [g for g, _ in grids], [c for _, c in grids], [name for _, name in present], ticks, ticks,
                             "Median goal error by start and goal position, all joints\n%s" % heading,
                             "goal, share of the joint range", "start, share of the joint range", "median goal error (ring indices)", "%.1f",
                             origin_lower=True, width=0.55 * bins + 1.2), args.out, "%s_heat_start_goal" % prefix)
            figures += 4
            plt.close("all")
            out += figure_notes(prefix, bins, [name for _, name in present], len(next(iter(systems.values()))))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tables.md"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(out) + "\n")
    print("written: %s (%d lines), %d figures" % (os.path.join(args.out, "tables.md"), len(out), figures))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--screen-moveit", action="store_true", help="reject draws MoveIt reports in collision (needs the simulation)")
    p.add_argument("--out", required=True, help="the plan file to write")
    p = sub.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--system", choices=SYSTEMS, required=True)
    p.add_argument("--engines", choices=("fake", "nest", "full"), default="nest",
                   help="full: Gazebo; nest: real NEST, simulated joints; fake: no simulator (checks the plumbing only)")
    p.add_argument("--time-scale", type=int, default=None)
    p.add_argument("--shard", default=None, help="I/N: run every N-th trial, starting with the I-th")
    p.add_argument("--limit", type=int, default=None, help="only the first trials of the plan")
    p.add_argument("--keep-trace", action="store_true", help="store the joint trace of every trial")
    p.add_argument("--out", required=True, help="result folder; name it <nest|gazebo>_<v4|v3|pid|p_controller|native> for the report")
    p = sub.add_parser("report")
    p.add_argument("--data", required=True, help="folder with nest_*/ and gazebo_*/ result folders and the plan files")
    p.add_argument("--out", required=True)
    p.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    return {"plan": plan_command, "run": run_command, "report": report_command}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
