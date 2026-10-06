#!/usr/bin/env python3
"""Validation: does the joint go where the ring says? (docs/archive/velocity/calibration.md stage E, docs/archive/velocity/bounded-ring.md)

Runs the two-ring graph for the same goals and start poses with up to three systems

* ``G``: the legacy ring (joint range on indices 5-195) with the legacy decoder gains,
* ``T``: the legacy ring with the transport calibration,
* ``B``: the bounded ring (joint range on an arc, wall, faded drive, bounded comparator)
  with its own transport calibration,

and compares per trial

* the joint displacement with the bump displacement times rho (tracking, the calibration target),
* the joint displacement with the integral of the commanded velocity (the robot),
* the ring's joint estimate (``RingState``: circular population vector -> angle) with the
  measured joint, aligned by the NEST lead, during the trial and after it settles,
* the final joint angle with the goal (goal error), wrong-direction and no-drive trials,
  limit clamps, and for ``B`` bumps outside the arc.

    python3 -B scripts/validate_velocity.py --engines full --joints 5 4 --trials 20 --out /tmp/validation
    python3 -B scripts/validate_velocity.py --engines nest --joints 0 1 2 3 4 5 6 --systems T B --starts 0.25 0.5 0.75 --seeds 13579 7 99

Goals and starts are fractions of the joint range (``velocity_calibration.json``
limits), so every system gets the same radians.  Without ``--starts`` each joint
starts where the reset motion leaves it.  ``--engines full`` needs the simulation
(``roslaunch tiago_ring_controller tiago_ring_controller.launch gui:=false``); a
start pose is reached with one smooth trajectory after the reset motion, and
``--screen-moveit`` skips goals whose straight single-joint path MoveIt reports
in collision (``/check_state_validity``).  ``nest`` uses the fake robot.

Pass (systems with the transport calibration): tracking error median <= 6% and
p90 <= 15%, direction bias within +-3%, joint/command 0.97-1.03, no clamp events.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

from tiago_ring_controller.evaluation.motion import NO_DRIVE_RAD, motion_metrics  # noqa: E402
from tiago_ring_controller.cosim.disturbance import Disturbance  # noqa: E402
from tiago_ring_controller.cosim.engine import EngineError  # noqa: E402
from tiago_ring_controller.graph import compile_graph, run_graph_trial, three_ring_single_joint, two_ring_single_joint  # noqa: E402
from tiago_ring_controller.graph.calibration import transport_motif, transport_trial_from_record  # noqa: E402
from tiago_ring_controller.graph.templates import joint_limits  # noqa: E402
from tiago_ring_controller.training.transport import TransportOptions, classify_trial, summarize_errors, trial_displacement  # noqa: E402

#: arm joints in the pose of the reset motion ``tiago_experiment_start_1``
START_POSE = (0.21, -0.2, -2.2, 1.15, -1.57, 0.2, 0.0)
CRITERIA = {"median": 0.06, "p90": 0.15, "bias": 0.03, "command_low": 0.97, "command_high": 1.03}
SYSTEMS = {
    "G": dict(calibration="legacy", bounds="legacy"),
    "T": dict(calibration="transport", bounds="legacy"),
    "B": dict(calibration="transport", bounds="bounded"),
    "F": dict(bounds="bounded", rings=3),
}
SYSTEM_NAMES = {"G": "legacy ring, legacy gains", "T": "legacy ring, transport calibration", "B": "bounded ring",
                "F": "three rings with body feedback (v4)"}
ARM_JOINTS = ["arm_%d_joint" % (i + 1) for i in range(7)]


def goal_plan(start, count, seed, low=0.05, high=0.95, step=0.05, min_distance=0.15):
    """``count`` ``(goal fraction, seed)`` pairs: goals on both sides of the start fraction, at least
    ``min_distance`` of the range away, spread evenly; when a side has fewer goals than its share
    they repeat with new seeds."""

    grid = [round(x, 4) for x in np.arange(low, high + 1e-9, step)]
    sides = [side for side in ([g for g in grid if start - g >= min_distance - 1e-9],
                               [g for g in grid if g - start >= min_distance - 1e-9]) if side]
    plan = []
    for k, side in enumerate(sides):
        share = count // len(sides) + (1 if k < count % len(sides) else 0)
        unique = sorted({side[int(round(x))] for x in np.linspace(0, len(side) - 1, min(share, len(side)))})
        plan.extend((unique[n % len(unique)], seed + n // len(unique)) for n in range(share))
    return plan


class MoveItScreen:
    """Straight single-joint paths checked by MoveIt's ``/check_state_validity``."""

    def __init__(self, transport, points=10):
        import rospy
        from moveit_msgs.srv import GetStateValidity
        from sensor_msgs.msg import JointState

        rospy.wait_for_service("/check_state_validity", timeout=30.0)
        self.service = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
        self.rospy, self.JointState, self.points, self.transport = rospy, JointState, points, transport

    def path_is_valid(self, joint, start, goal):
        from moveit_msgs.msg import RobotState

        self.transport.unpause()                     # /joint_states only flows while the simulation runs
        current = self.rospy.wait_for_message("/joint_states", self.JointState, timeout=5.0)
        names, positions = list(current.name), list(current.position)
        for value in np.linspace(start, goal, self.points):
            state = RobotState()
            state.joint_state.name = names
            # the other arm joints as the reset motion leaves them, not where the previous trial ended
            state.joint_state.position = [float(value) if n == ARM_JOINTS[joint] else START_POSE[ARM_JOINTS.index(n)] if n in ARM_JOINTS else p
                                          for n, p in zip(names, positions)]
            if not self.service(robot_state=state, group_name="arm").valid:
                return False
        return True

    def poses_are_valid(self, poses):
        """Whole arm poses (seven angles each, ``arm_1`` .. ``arm_7``): True when MoveIt accepts every one."""

        from moveit_msgs.msg import RobotState

        self.transport.unpause()
        current = self.rospy.wait_for_message("/joint_states", self.JointState, timeout=5.0)
        names, positions = list(current.name), list(current.position)
        for pose in poses:
            state = RobotState()
            state.joint_state.name = names
            state.joint_state.position = [float(pose[ARM_JOINTS.index(n)]) if n in ARM_JOINTS else p for n, p in zip(names, positions)]
            if not self.service(robot_state=state, group_name="arm").valid:
                return False
        return True

    def pose_path_is_valid(self, start_pose, end_pose):
        """The straight joint-space path between two whole arm poses."""

        start, end = np.asarray(start_pose, dtype=float), np.asarray(end_pose, dtype=float)
        return self.poses_are_valid([start + f * (end - start) for f in np.linspace(0.0, 1.0, self.points)])


def state_error(record, joint, lead, state_id="state"):
    """The ring's joint estimate against the measured joint.

    ``RingState`` at main tick k reads the ring counts of NEST step k + lead, which
    the joint reaches ``lead`` ticks later: the estimate of tick k is compared with
    the joint measured at tick k + lead.  ``final``: the mean of the last five
    estimates against the final joint angle.
    """

    main = record.main_ticks
    estimates = [t.outputs[state_id].data["angle"] if t.outputs.get(state_id) is not None else float("nan") for t in main]
    in_arc = [bool(t.outputs[state_id].data["in_arc"]) for t in main if t.outputs.get(state_id) is not None]
    positions = [float(t.inputs["joint_state"]["positions"][joint]) for t in main]
    pairs = [(estimates[k], positions[k + lead]) for k in range(len(main) - lead) if np.isfinite(estimates[k])]
    errors = np.array([e - q for e, q in pairs])
    tail = [e for e in estimates[-5:] if np.isfinite(e)]
    final = float(record.final_state["joint_state"]["positions"][joint])
    return {
        "state_error_abs_median": float(np.median(np.abs(errors))) if len(errors) else float("nan"),
        "state_error_abs_p95": float(np.percentile(np.abs(errors), 95)) if len(errors) else float("nan"),
        "state_error_final": float(np.mean(tail) - final) if tail else float("nan"),
        "out_of_arc_ticks": int(sum(1 for flag in in_arc if not flag)),
    }


def speed_metrics(record, joint, goal_rad, rho, state_id="state"):
    """Motion metrics of the joint trace (``evaluation.motion``), the joint's lag behind the bump, and the trace.

    The lag is the largest gap during the trial between the bump (as the ring's joint
    estimate, ring indices) and the joint, in ring indices.  ``trace`` holds the joint
    angle, the commanded velocity and the ring's estimate per tick, for the plots of
    ``scripts/results_report.py``.
    """

    dt = record.dt_ms / 1000.0
    q = np.array([float(t.inputs["joint_state"]["positions"][joint]) for t in record.main_ticks]
                 + [float(record.final_state["joint_state"]["positions"][joint])])
    estimates = np.array([t.outputs[state_id].data["angle"] if t.outputs.get(state_id) is not None else np.nan
                          for t in record.main_ticks])
    gap = np.abs(estimates - q[:-1])
    metrics = motion_metrics(q, goal_rad, dt)
    return {
        "time_to_90_s": metrics["time_to_90_s"], "peak_speed": metrics["peak_speed"], "overshoot_rad": metrics["overshoot_rad"],
        "settling_time_s": metrics["settling_time_s"], "reached_tolerance": metrics["reached_tolerance"],
        "max_lag_index": float(np.nanmax(gap) / rho) if np.isfinite(gap).any() else float("nan"),
        # how still the joint is at the end: range of the angle over the last 2 s of the trial
        "hold_range_rad": float(np.ptp(q[-int(round(2.0 / dt)) - 1:-1])) if len(q) > 2 else float("nan"),
        "trace": {"dt_s": dt, "q": [round(float(v), 5) for v in q],
                  "v_cmd": [round(float(t.outputs["arm_velocity_cmd"]["velocities"][0]), 5) for t in record.main_ticks],
                  "ring": [None if not np.isfinite(v) else round(float(v), 5) for v in estimates]},
    }


def ring_gaps(record, goal_index, tail=20):
    """System F: where the three bumps end, in ring indices, and whether the predictive bump died.

    ``body_gap_index``: predictive bump minus state bump; ``goal_gap_index``:
    predictive bump minus goal bump; ``goal_bump_error_index``: goal bump minus
    the goal's index (drift of the goal ring); each the mean of the last ``tail`` ticks.
    """

    def trace(ring, key):
        return np.array([t.inputs[ring].data[key] if t.inputs.get(ring) is not None else np.nan for t in record.main_ticks], dtype=float)

    p, g, s = (trace("jt__" + name, "centroid_circular") for name in ("P", "G", "S"))
    total = trace("jt__P", "total")
    mean = lambda values: float(np.nanmean(values[-tail:])) if np.isfinite(values[-tail:]).any() else float("nan")  # noqa: E731
    return {
        "body_gap_index": mean(p - s), "goal_gap_index": mean(p - g),
        "goal_bump_error_index": mean(g - goal_index) if goal_index is not None else float("nan"),
        "body_gap_max_index": float(np.nanmax(np.abs(p - s))) if np.isfinite(p - s).any() else float("nan"),
        "lost_bump": bool(np.any(total[5:] == 0)),
    }


def run_trial(joint, system, start_rad, goal_rad, engines, transport, seed, time_scale=None, bounded=None,
              feedback=None, motor_gain_scale=1.0, push_rad=0.0, hold_s=0.0, disturb_at_s=1.0, observers=(), other_joints=None):
    """One trial.  ``motor_gain_scale`` multiplies the decoder gains (a wrong motor calibration);
    ``push_rad`` / ``hold_s`` disturb the robot at ``disturb_at_s`` (``cosim.disturbance``);
    ``observers`` are loop observers (``scripts/record_video.py`` records through one);
    ``other_joints`` maps further joint indices to the angles they are moved to before the
    trial (``scripts/stress_test.py``: the whole arm in a random pose)."""

    options = dict(SYSTEMS[system])
    three_rings = options.pop("rings", 2) == 3
    if options["bounds"] == "bounded" and bounded:
        options["bounded"] = bounded
    if three_rings:
        graph = three_ring_single_joint(joint=joint, seed=seed, time_scale=time_scale, bounded=bounded or None, feedback=feedback or None)
    else:
        graph = two_ring_single_joint(joint=joint, seed=seed, state_readout=True, time_scale=time_scale, **options)
    state_id = "jt__P_state" if three_rings else "state"
    motif = transport_motif(graph)
    if motor_gain_scale != 1.0:
        for key in ("gain_positive", "gain_negative"):
            motif["decoder"].params[key] *= float(motor_gain_scale)
    encoder = motif["state_encoder"]
    size = int(motif["state_ring"].params["population_size"])
    compiled = compile_graph(graph, engines=engines, transport=transport if engines == "full" else None, observers=list(observers))
    if engines != "full":
        compiled.robot_engine.home[:] = list(START_POSE)
    start_pose = dict(other_joints or {})
    if start_rad is not None:
        start_pose[joint] = start_rad
    if start_pose:
        compiled.robot_engine.set_start_pose(start_pose)
    compiled.loop.initialize()
    disturbance = Disturbance(compiled.robot_engine, joint, push_rad, hold_s, disturb_at_s)
    started = time.monotonic()
    try:
        record = run_graph_trial(compiled, goal_rad)
    finally:
        disturbance.remove()
        if engines == "full":
            compiled.nest_engine.shutdown()          # the Gazebo engine (shared transport) stays up
        else:
            compiled.loop.shutdown()
    wall = time.monotonic() - started               # the whole trial: reset motion, lead and main ticks
    timing = record.meta.get("timing", {})
    reset_wall = float(sum(timing.get("reset_wall_s", {}).values()))
    tick_walls = [t.wall_s for t in record.main_ticks]
    options = TransportOptions()
    trial = transport_trial_from_record(record, motif["state_ring"].id, motif["gain"].id, group=seed)
    main = record.main_ticks
    start = float(main[0].inputs["joint_state"]["positions"][joint])
    final = float(record.final_state["joint_state"]["positions"][joint])
    commanded = float(sum(t.outputs["arm_velocity_cmd"]["velocities"][0] for t in main) * record.dt_ms / 1000.0)
    clamps = sum(1 for t in main if t.outputs["arm_velocity_cmd"].get("velocity_clamped") or t.inputs["joint_state"].get("clamped_joints"))
    centroid = trial.centroid[np.isfinite(trial.centroid)]
    goal_index = encoder.ring_index(goal_rad) if hasattr(encoder, "ring_index") else None
    moved = final - start
    row = {
        "joint": joint, "system": system, "seed": seed, "goal_rad": goal_rad, "goal_index": goal_index,
        "start_index": record.meta.get("initial_ring_index"), "start_rad": start, "final_rad": final,
        "status": classify_trial(trial, options), "bump_displacement": trial_displacement(trial, options),
        "joint_displacement": moved, "commanded_displacement": commanded, "rho": encoder.radians_per_index(size),
        "endpoint_error": float(np.mean(centroid[-options.window:]) - record.meta.get("goal_ring_index", np.nan)) if len(centroid) else float("nan"),
        "goal_error_rad": goal_rad - final, "n_steps": record.n_steps, "stop_reason": record.stop_reason,
        "settle_s": record.n_steps * record.dt_ms / 1000.0, "clamp_ticks": clamps, "wall_s": wall, "reset_wall_s": reset_wall,
        "tick_wall_ms": 1000.0 * float(np.mean(tick_walls)) if tick_walls else float("nan"),
        "no_drive": bool(abs(moved) < NO_DRIVE_RAD),
        "wrong_direction": bool(abs(moved) >= NO_DRIVE_RAD and np.sign(moved) != np.sign(goal_rad - start)),
    }
    row.update(state_error(record, joint, int(compiled.graph.simulation.nest_lead_steps), state_id))
    row.update(speed_metrics(record, joint, goal_rad, row["rho"], state_id), time_scale=int(compiled.graph.simulation.time_scale))
    row.update(motor_gain_scale=float(motor_gain_scale), push_rad=float(push_rad), hold_s=float(hold_s))
    shift = compiled.robot_engine.base_shift() if hasattr(compiled.robot_engine, "base_shift") else {}
    # the mobile base is not anchored: a shift during the trial means the arm pushed on something
    row.update(base_shift_m=shift.get("trial_m", float("nan")), base_found_m=shift.get("found_m", float("nan")))
    if three_rings:
        row.update(ring_gaps(record, goal_index))
    # The bump's travel from the injected index to where it settles: unlike ``bump_displacement`` (which
    # starts after the first ticks, the calibration's convention) it does not depend on the time scale.
    travel = (row["goal_index"] + row["endpoint_error"] - row["start_index"]
              if row["goal_index"] is not None and row["start_index"] is not None else float("nan"))
    row["bump_travel_index"] = float(travel)
    row["travel_tracking_rel"] = (float((moved / row["rho"] - travel) / abs(travel))
                                  if np.isfinite(travel) and abs(travel) > 5 else float("nan"))
    return row


def summarize(rows):
    """Tracking, robot, ring-state and goal figures of one joint with one system."""

    usable = [r for r in rows if r["status"] == "ok"]
    summary = {
        "trials": len(rows), "usable": len(usable),
        "statuses": {s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})},
        "clamp_ticks": int(sum(r["clamp_ticks"] for r in rows)),
        "no_drive": int(sum(r["no_drive"] for r in rows)), "wrong_direction": int(sum(r["wrong_direction"] for r in rows)),
        "out_of_arc_ticks": int(sum(r["out_of_arc_ticks"] for r in rows)),
        "goal_error_rad": _abs_stats([r["goal_error_rad"] for r in rows]),
        "state_error_rad": _abs_stats([r["state_error_abs_median"] for r in rows]),
        "state_error_p95_rad": float(np.nanmedian([r["state_error_abs_p95"] for r in rows])) if rows else float("nan"),
        "state_error_final_rad": _abs_stats([r["state_error_final"] for r in rows]),
        "settle_s_mean": float(np.mean([r["settle_s"] for r in rows])),
        "wall_s_mean": float(np.mean([r["wall_s"] for r in rows])),
        "reset_wall_s_mean": float(np.mean([r["reset_wall_s"] for r in rows])),
        "tick_wall_ms_mean": float(np.mean([r["tick_wall_ms"] for r in rows])),
        "real_time_factor": float(sum(r["settle_s"] for r in rows) / max(sum(r["wall_s"] for r in rows), 1e-9)),
        "repeat_spread_rad": repeat_spread(rows),
        "travel_tracking": _abs_stats([r["travel_tracking_rel"] for r in rows]),
        "time_to_90_s": _abs_stats([r["time_to_90_s"] for r in rows]),
        "settling_time_s": _abs_stats([r["settling_time_s"] for r in rows]),
        "reached_tolerance": int(sum(r["reached_tolerance"] for r in rows)),
        "peak_speed": _abs_stats([r["peak_speed"] for r in rows]),
        "overshoot_rad": _abs_stats([r["overshoot_rad"] for r in rows]),
        "hold_range_rad": _abs_stats([r["hold_range_rad"] for r in rows]),
        "base_shift_m": _abs_stats([r["base_shift_m"] for r in rows]),
        "max_steps_trials": int(sum(r["stop_reason"] == "max_steps" for r in rows)),
    }
    if rows and "lost_bump" in rows[0]:
        summary.update({
            "lost_bumps": int(sum(r["lost_bump"] for r in rows)),
            "body_gap_index": _abs_stats([r["body_gap_index"] for r in rows]),
            "goal_gap_index": _abs_stats([r["goal_gap_index"] for r in rows]),
            "goal_bump_error_index": _abs_stats([r["goal_bump_error_index"] for r in rows]),
        })
    if not usable:
        return summary
    moved = np.array([r["bump_displacement"] for r in usable])
    joint_in_indices = np.array([r["joint_displacement"] / r["rho"] for r in usable])
    tracking = summarize_errors(joint_in_indices - moved, moved)
    command = np.array([r["joint_displacement"] / r["commanded_displacement"] for r in usable if abs(r["commanded_displacement"]) > 1e-6])
    summary.update({
        "tracking": tracking.as_dict(),
        "joint_over_command": {"min": float(command.min()), "max": float(command.max())} if len(command) else None,
        "endpoint_error_index": {"median_abs": float(np.median(np.abs([r["endpoint_error"] for r in usable]))),
                                 "median": float(np.median([r["endpoint_error"] for r in usable]))},
    })
    biases = [b for b in (tracking.bias_left, tracking.bias_right) if np.isfinite(b)]
    summary["passes"] = bool(
        tracking.median_abs_rel <= CRITERIA["median"] and tracking.p90_abs_rel <= CRITERIA["p90"]
        and all(abs(b) <= CRITERIA["bias"] for b in biases) and summary["clamp_ticks"] == 0
        and (not len(command) or (command.min() >= CRITERIA["command_low"] and command.max() <= CRITERIA["command_high"]))
    )
    return summary


def parse_settings(items):
    """``key=value`` items as a dict; the value is JSON when it parses, ``a.b=value`` sets ``{"a": {"b": value}}``."""

    settings = {}
    for item in items or ():
        key, _, value = item.partition("=")
        try:
            value = json.loads(value)
        except ValueError:
            pass
        target = settings
        *parents, leaf = key.split(".")
        for parent in parents:
            target = target.setdefault(parent, {})
        target[leaf] = value
    return settings


def repeat_spread(rows):
    """Largest range of the final angle over runs of the same start, goal and seed (nan without repeats)."""

    groups = {}
    for r in rows:
        groups.setdefault((round(r["start_rad"], 2), round(r["goal_rad"], 6), r["seed"]), []).append(r["final_rad"])
    spreads = [max(v) - min(v) for v in groups.values() if len(v) > 1]
    return float(max(spreads)) if spreads else float("nan")


def _abs_stats(values):
    values = np.abs(np.asarray([v for v in values if np.isfinite(v)], dtype=float))
    if not len(values):
        return {"median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {"median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "max": float(values.max())}


def write_results(args, summaries, rows, skipped, failed):
    if not args.out:
        return None
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "validation.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump({"engines": args.engines, "criteria": CRITERIA, "time_scale": args.time_scale, "settings": parse_settings(args.set),
                   "feedback": parse_settings(args.feedback), "motor_gain_scale": args.motor_gain_scale, "push_rad": args.push,
                   "hold_s": args.hold, "systems": {k: SYSTEM_NAMES[k] for k in args.systems},
                   "summaries": summaries, "trials": rows, "skipped": skipped, "failed": failed,
                   "measured": time.strftime("%Y-%m-%d %H:%M:%S")}, stream, indent=1)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", choices=("nest", "full"), default="nest")
    parser.add_argument("--joints", type=int, nargs="+", default=[5, 4])
    parser.add_argument("--trials", type=int, default=20, help="goals per joint, start and seed")
    parser.add_argument("--systems", nargs="+", default=["G", "T", "B"], choices=sorted(SYSTEMS))
    parser.add_argument("--starts", type=float, nargs="+", default=None,
                        help="start poses as fractions of the joint range (default: where the reset motion leaves the joint)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13579])
    parser.add_argument("--repeats", type=int, default=1, help="runs of every planned trial (repeatability: spread of the final angle)")
    parser.add_argument("--time-scale", type=int, default=None,
                        help="NEST steps per robot tick (docs/archive/velocity/speed.md); default: the template's (1 legacy, 3 bounded)")
    parser.add_argument("--set", action="append", default=[], help="systems B, F: bounded-ring setting key=value (graph.templates.BOUNDED)")
    parser.add_argument("--feedback", action="append", default=[], help="system F: v4 setting key=value (graph.templates.FEEDBACK)")
    parser.add_argument("--motor-gain-scale", type=float, default=1.0, help="multiplies the decoder gains (a wrong motor calibration)")
    parser.add_argument("--push", type=float, default=0.0, help="rad the joint is displaced mid-trial without the network knowing")
    parser.add_argument("--hold", type=float, default=0.0, help="seconds the joint is held mid-trial (commands dropped)")
    parser.add_argument("--disturb-at", type=float, default=1.0, help="time of --push / --hold in the trial, s")
    parser.add_argument("--screen-moveit", action="store_true", help="--engines full: skip goals MoveIt reports in collision")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    transport, screen = None, None
    if args.engines == "full":
        from tiago_ring_controller.cosim.gazebo_ros_engine import RospyTransport

        transport = RospyTransport(node_name="validate_velocity")
        transport.initialize()
        if args.screen_moveit:
            screen = MoveItScreen(transport)
    rows, summaries, skipped, failed = [], {}, [], []
    started = time.monotonic()
    try:
        for joint in args.joints:
            lo, hi = joint_limits(joint)
            starts = [None] if args.starts is None else [lo + f * (hi - lo) for f in args.starts]
            for start_rad in starts:
                start_fraction = ((START_POSE[joint] if start_rad is None else start_rad) - lo) / (hi - lo)
                start_fraction = float(np.clip(start_fraction, 0.0, 1.0))
                if screen is not None and start_rad is not None and not screen.path_is_valid(joint, START_POSE[joint], start_rad):
                    # the move from the reset pose to this start pose would collide: leave the start out
                    skipped.append({"joint": joint, "start_rad": start_rad, "goal_fraction": None})
                    print("joint %d: start %+.3f skipped by MoveIt" % (joint, start_rad))
                    continue
                for seed in args.seeds:
                    plan = goal_plan(start_fraction, args.trials, seed)
                    if screen is not None:
                        origin = START_POSE[joint] if start_rad is None else start_rad
                        valid = [(g, s) for g, s in plan if screen.path_is_valid(joint, origin, lo + g * (hi - lo))]
                        skipped.extend({"joint": joint, "start_rad": origin, "goal_fraction": g} for g, s in plan if (g, s) not in valid)
                        plan = valid
                    print("joint %d [%.3f, %.3f]: start %s, seed %d, goal fractions %s"
                          % (joint, lo, hi, "reset pose" if start_rad is None else "%+.3f" % start_rad, seed, sorted({g for g, _ in plan})))
                    for system in args.systems:
                        for goal_fraction, trial_seed in [p for p in plan for _ in range(args.repeats)]:
                            try:
                                row = run_trial(joint, system, start_rad, lo + goal_fraction * (hi - lo), args.engines, transport, trial_seed,
                                                args.time_scale, parse_settings(args.set), parse_settings(args.feedback),
                                                args.motor_gain_scale, args.push, args.hold, args.disturb_at)
                            except EngineError as exc:
                                # the robot could not be brought back to its start (the reset motion failed after its retries)
                                failed.append({"joint": joint, "system": system, "goal_fraction": goal_fraction, "error": str(exc)})
                                print("  %s goal fraction %.2f: FAILED, %s" % (system, goal_fraction, exc))
                                continue
                            row["start_fraction"], row["goal_fraction"] = start_fraction, goal_fraction
                            rows.append(row)
                            ratio = row["joint_displacement"] / (row["rho"] * row["bump_displacement"]) if abs(row["bump_displacement"]) > 1 else float("nan")
                            print("  %s goal %+.3f: start %+.3f final %+.3f (error %+.3f), joint/bump %.3f, ring - joint %+.3f, %s%s, %d steps, %.0f s"
                                  % (system, row["goal_rad"], row["start_rad"], row["final_rad"], row["goal_error_rad"], ratio,
                                     row["state_error_final"], row["status"],
                                     ", WRONG WAY" if row["wrong_direction"] else ", no drive" if row["no_drive"] else "",
                                     row["n_steps"], row["wall_s"]))
            for system in args.systems:
                joint_rows = [r for r in rows if r["joint"] == joint and r["system"] == system]
                if joint_rows:
                    summaries["joint %d, %s" % (joint, system)] = summarize(joint_rows)
            write_results(args, summaries, rows, skipped, failed)        # after every joint: a long run keeps what it has
    finally:
        if transport is not None:
            try:
                transport.unpause()
            finally:
                transport.shutdown()

    print()
    print("| joint, system | usable | tracking median | p90 | bias left | bias right | joint/command | clamps | goal error median / p90 / max (rad) "
          "| ring - joint median / final (rad) | no drive | wrong way | settle (s) | wall (s) | to 90% median (s) | peak speed max (rad/s) "
          "| overshoot median / max (rad) | joint vs bump travel median / p90 | pass |")
    print("|---|---:|---:|---:|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for name, s in summaries.items():
        g, st, sf = s["goal_error_rad"], s["state_error_rad"], s["state_error_final_rad"]
        common = "%d | %.3f / %.3f / %.3f | %.3f / %.3f | %d | %d | %.1f | %.1f | %.2f | %.2f | %.3f / %.3f | %.1f%% / %.1f%%" % (
            s["clamp_ticks"], g["median"], g["p90"], g["max"], st["median"], sf["median"], s["no_drive"], s["wrong_direction"],
            s["settle_s_mean"], s["wall_s_mean"], s["time_to_90_s"]["median"], s["peak_speed"]["max"],
            s["overshoot_rad"]["median"], s["overshoot_rad"]["max"],
            100 * s["travel_tracking"]["median"], 100 * s["travel_tracking"]["p90"])
        if "tracking" not in s:
            print("| %s | %d/%d | | | | | | %s | |" % (name, s["usable"], s["trials"], common))
            continue
        t, c = s["tracking"], s["joint_over_command"]
        print("| %s | %d/%d | %.1f%% | %.1f%% | %+.1f%% | %+.1f%% | %s | %s | %s |" % (
            name, s["usable"], s["trials"], 100 * t["median_abs_rel"], 100 * t["p90_abs_rel"], 100 * t["bias_left"], 100 * t["bias_right"],
            "%.3f-%.3f" % (c["min"], c["max"]) if c else "", common, "yes" if s["passes"] else "no"))
    if skipped:
        print("skipped by MoveIt: %d goals" % len(skipped))
    print("done in %.0f s" % (time.monotonic() - started))
    if failed:
        print("failed trials (reset motion): %d" % len(failed))
    path = write_results(args, summaries, rows, skipped, failed)
    if path:
        print("written:", path)
    lost = {name: s["lost_bumps"] for name, s in summaries.items() if "lost_bumps" in s}
    for name, s in summaries.items():
        if "lost_bumps" in s:
            print("%s: bumps lost %d, predicted - state %.2f / %.2f, predicted - goal bump %.2f / %.2f, goal bump - goal %.2f / %.2f (indices, median / p90)"
                  % (name, s["lost_bumps"], s["body_gap_index"]["median"], s["body_gap_index"]["p90"], s["goal_gap_index"]["median"],
                     s["goal_gap_index"]["p90"], s["goal_bump_error_index"]["median"], s["goal_bump_error_index"]["p90"]))
    # the pass rule (joint follows the bump) is the transport calibration's; system F has feedback instead
    transport_ok = [s.get("passes", False) for name, s in summaries.items() if name.endswith((", T", ", B"))] or [not any(lost.values())]
    return 0 if transport_ok and all(transport_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
