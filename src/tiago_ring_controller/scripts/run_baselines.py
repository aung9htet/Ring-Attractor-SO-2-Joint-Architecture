#!/usr/bin/env python3
"""Upper references for the comparison (docs/results.md): a tuned controller and the robot's own controller.

    python3 -B scripts/run_baselines.py run --controller p      --engines full --joints 0 1 2 3 4 5 6 --trials 8 --screen-moveit --out /out/p
    python3 -B scripts/run_baselines.py run --controller native --engines full --joints 0 1 2 3 4 5 6 --trials 8 --screen-moveit --out /out/native
    python3 -B scripts/run_baselines.py tune --engines full --out /out/tuning

The baseline of the comparison is the PID of the draft
(``scripts/run_legacy_trials.py --controller pid``).  These two show how far
conventional position control can go on the same joints:

``p`` (system ``P``): ``control.pid.AntiWindupPID`` with the gains a grid chose
(``control.pid.DEFAULT_GAINS``: proportional only, 60% of the joint's velocity
limit), in the same lock-stepped 50 ms loop and through the same velocity
command as the ring controllers.
``native`` (system ``N``): the goal is sent to the robot's joint trajectory
controller as one trajectory point, reached after ``distance / (0.5 x velocity
limit)`` seconds (at least 1 s); nothing of ours is in the loop.  Gazebo only.

Both use the engine of the graph systems (reset motion, base put back, start
pose, 50 ms ticks) and are scored by ``evaluation.motion`` from the joint
trace, so their rows drop into ``scripts/results_report.py`` and
``scripts/stress_test.py`` like those of ``scripts/validate_velocity.py``:
the same goal plan (``--trials``, ``--starts``, ``--repeats``,
``--screen-moveit``) and for ``p`` the same disturbances
(``--motor-gain-scale``, ``--push``, ``--hold``).  A trial ends when the
command has been under 0.01 rad/s (``p``) or the joint still (``native``) for
2 s, or after 20 s.  ``--engines fake`` uses the simulated joints (``p`` only).

``tune`` runs the grid (``kp``, ``ki``, ``kd`` and the output limit as a fraction
of the joint's velocity limit) on arm_6 and arm_5 from three start poses and
picks, among the settings that settle every trial with a p90 overshoot under
0.02 rad, the one with the lowest p90 goal error; ties go to the shorter
settling time.
"""

import argparse
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for path in (SRC, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from tiago_ring_controller.control.mapping import RingMapper  # noqa: E402
from tiago_ring_controller.control.pid import DEFAULT_GAINS, AntiWindupPID  # noqa: E402
from tiago_ring_controller.cosim.datapack import DataPack  # noqa: E402
from tiago_ring_controller.cosim.disturbance import Disturbance  # noqa: E402
from tiago_ring_controller.cosim.engine import EngineError  # noqa: E402
from tiago_ring_controller.evaluation.motion import NO_DRIVE_RAD, motion_metrics  # noqa: E402
from tiago_ring_controller.evaluation.serialization import json_safe  # noqa: E402
from tiago_ring_controller.graph.templates import ARM_MAX_VELOCITY, BOUNDED, joint_limits  # noqa: E402

DT_MS = 50.0
MAX_STEPS = 400
#: ticks without command (PID) or motion (native) that end a trial: 2 s, as ``n_settle`` of v4
N_STILL = 40
QUIET_COMMAND = 0.01          # rad/s
STILL_STEP = 1.0e-3           # rad per tick
#: ``p``: the tuned controller (in effect proportional; system ``P``); ``native``: the robot's own controller (``N``)
SYSTEMS = {"p": "P", "native": "N"}
SYSTEM_NAMES = {"P": "P controller, tuned", "N": "the robot's joint trajectory controller"}
DEFAULTS = {"p": DEFAULT_GAINS}
TUNE_GRID = {"kp": (2.0, 4.0, 6.0, 8.0, 10.0), "ki": (0.0, 0.5, 2.0), "kd": (0.0, 0.02), "speed_fraction": (1.0, 0.8, 0.6)}
TUNE_OVERSHOOT = 0.02


def radians_per_index(joint):
    lo, hi = joint_limits(joint)
    return RingMapper("bounded", lo, hi, 200, 5, BOUNDED["arc_fraction"]).radians_per_index


def make_engine(engines, transport=None):
    if engines == "full":
        from tiago_ring_controller.cosim.gazebo_ros_engine import GazeboRosEngine

        engine = GazeboRosEngine(transport)
    else:
        from tiago_ring_controller.cosim.fakes import FakeRobotEngine
        from validate_velocity import START_POSE

        engine = FakeRobotEngine(home=list(START_POSE))
    engine.initialize()
    return engine


def run_trial(engine, controller, joint, goal_rad, start_pose=None, gains=None, motor_gain_scale=1.0,
              push_rad=0.0, hold_s=0.0, disturb_at_s=1.0, max_steps=MAX_STEPS, on_tick=None):
    """One trial of a baseline on an initialised robot engine; the row has the fields of ``validate_velocity``.

    ``on_tick(t_s, angle, command)`` is called after every 50 ms tick (``scripts/record_video.py``)."""

    lo, hi = joint_limits(joint)
    limit = ARM_MAX_VELOCITY[joint]
    dt_s = DT_MS / 1000.0
    engine.set_start_pose(dict(start_pose or {}))
    started = time.monotonic()
    engine.reset("rebuild")
    reset_wall = time.monotonic() - started
    disturbance = Disturbance(engine, joint, push_rad, hold_s, disturb_at_s)
    feedback = controller in DEFAULTS
    gains = dict(DEFAULTS.get(controller, {}), **(gains or {}))
    pid_limit = float(gains.get("speed_fraction", 1.0)) * limit
    if feedback:
        pid = AntiWindupPID(gains["kp"], gains["ki"], gains["kd"], pid_limit)
    q = [float(engine.get_datapacks()["joint_state"]["positions"][joint])]
    commands, clamp_ticks, quiet, stop_reason, duration = [], 0, 0, "max_steps", 0.0
    tick_walls = []
    try:
        if controller == "native":
            from tiago_ring_controller.control.trajectory import TrajectoryPointData

            snapshot = list(engine.get_datapacks()["joint_state"]["positions"])
            target = list(snapshot)
            target[joint] = float(goal_rad)
            duration = max(1.0, abs(goal_rad - snapshot[joint]) / (0.5 * limit))
            point = TrajectoryPointData(positions=tuple(target), velocities=(0.0,) * len(target), time_from_start_s=duration)
            engine.transport.publish_trajectory(engine.joint_names, [point], engine.transport.sim_time_s())
        for step in range(max_steps):
            tick = time.monotonic()
            if feedback:
                error = goal_rad - q[-1]
                output, _ = pid.step(error, q[-1], dt_s)
                command = float(np.clip(output * motor_gain_scale, -limit, limit))
                engine.set_datapacks({"arm_velocity_cmd": DataPack("arm_velocity_cmd", engine.t_ms, {
                    "joint_index": joint, "velocities": [command], "dt_s": dt_s,
                    "commands": [{"joint_index": joint, "velocities": [command], "limits": [lo, hi]}]})})
                commands.append(command)
            engine.advance(DT_MS)
            state = engine.get_datapacks()["joint_state"]
            q.append(float(state["positions"][joint]))
            # clipped by the joint itself (its velocity or position limit), not by the PID's own output limit
            clamp_ticks += int(bool(state.get("clamped_joints")) or (feedback and abs(commands[-1]) >= limit - 1e-9))
            tick_walls.append(time.monotonic() - tick)
            if on_tick is not None:
                on_tick((step + 1) * dt_s, q[-1], commands[-1] if commands else float("nan"))
            if feedback:
                still = abs(commands[-1]) < QUIET_COMMAND
            else:
                still = (step + 1) * dt_s >= duration and abs(q[-1] - q[-2]) < STILL_STEP
            quiet = quiet + 1 if still else 0
            if quiet >= N_STILL:
                stop_reason = "settled"
                break
        engine.finish_trial()
    finally:
        disturbance.remove()
    final = float(engine.get_datapacks()["joint_state"]["positions"][joint])
    trace = q + [final]
    metrics = motion_metrics(trace, float(goal_rad), dt_s)
    moved = final - q[0]
    shift = engine.base_shift() if hasattr(engine, "base_shift") else {}
    row = {
        "joint": joint, "system": SYSTEMS[controller], "seed": 0, "goal_rad": float(goal_rad), "start_rad": q[0], "final_rad": final,
        "status": "ok", "n_steps": len(q) - 1, "stop_reason": stop_reason, "settle_s": (len(q) - 1) * dt_s,
        "clamp_ticks": clamp_ticks, "wall_s": time.monotonic() - started, "reset_wall_s": reset_wall,
        "tick_wall_ms": 1000.0 * float(np.mean(tick_walls)) if tick_walls else float("nan"),
        "rho": radians_per_index(joint), "time_scale": 1, "motor_gain_scale": float(motor_gain_scale), "push_rad": float(push_rad),
        "hold_s": float(hold_s), "base_shift_m": shift.get("trial_m", float("nan")), "base_found_m": shift.get("found_m", float("nan")),
        "no_drive": bool(abs(moved) < NO_DRIVE_RAD), "gains": gains if feedback else None,
        "trajectory_duration_s": duration if controller == "native" else None,
        "trace": {"dt_s": dt_s, "q": [round(v, 5) for v in trace], "v_cmd": [round(v, 5) for v in commands]},
    }
    row.update(metrics)
    return row


def summary_line(label, rows):
    errors = np.abs([r["goal_error_rad"] for r in rows])
    over = [r["overshoot_rad"] for r in rows]
    settle = [r["settling_time_s"] for r in rows if r["reached_tolerance"]]
    return ("%s: %d trials | goal error median %.4f p90 %.4f max %.4f rad | settled %d%% in %.2f s | overshoot p90 %.4f max %.4f | to 90%% %.2f s"
            % (label, len(rows), np.median(errors), np.percentile(errors, 90), errors.max(), round(100 * np.mean([r["reached_tolerance"] for r in rows])),
               np.median(settle) if settle else float("nan"), np.percentile(over, 90), max(over), np.nanmedian([r["time_to_90_s"] for r in rows])))


def planned_trials(args, screen):
    """``(joint, start_rad or None, goal_rad, goal_fraction, start_fraction)`` in the order of ``validate_velocity``."""

    from validate_velocity import START_POSE, goal_plan

    trials, skipped = [], []
    for joint in args.joints:
        lo, hi = joint_limits(joint)
        starts = [None] if args.starts is None else [lo + f * (hi - lo) for f in args.starts]
        for start_rad in starts:
            origin = START_POSE[joint] if start_rad is None else start_rad
            start_fraction = float(np.clip((origin - lo) / (hi - lo), 0.0, 1.0))
            if screen is not None and start_rad is not None and not screen.path_is_valid(joint, START_POSE[joint], start_rad):
                skipped.append({"joint": joint, "start_rad": start_rad, "goal_fraction": None})
                continue
            for seed in args.seeds:
                for fraction, _ in goal_plan(start_fraction, args.trials, seed):
                    goal = lo + fraction * (hi - lo)
                    if screen is not None and not screen.path_is_valid(joint, origin, goal):
                        skipped.append({"joint": joint, "start_rad": origin, "goal_fraction": fraction})
                        continue
                    trials.extend([(joint, start_rad, goal, fraction, start_fraction)] * args.repeats)
    return trials, skipped


def open_transport(args, need_screen):
    if args.engines != "full":
        return None, None
    from tiago_ring_controller.cosim.gazebo_ros_engine import RospyTransport
    from validate_velocity import MoveItScreen

    transport = RospyTransport(node_name="run_baselines")
    transport.initialize()
    return transport, (MoveItScreen(transport) if need_screen else None)


def write(out, document):
    if not out:
        return None
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "validation.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(json_safe(document), stream)
    return path


def run_command(args):
    if args.controller == "native" and args.engines != "full":
        raise SystemExit("the robot's own controller only exists in Gazebo (--engines full)")
    transport, screen = open_transport(args, args.screen_moveit)
    engine = make_engine(args.engines, transport)
    gains = {key: value for key, value in (("kp", args.kp), ("ki", args.ki), ("kd", args.kd), ("speed_fraction", args.speed_fraction))
             if value is not None}
    system = SYSTEMS[args.controller]
    rows, failed = [], []
    trials, skipped = planned_trials(args, screen)
    document = lambda: {"engines": args.engines, "systems": {system: SYSTEM_NAMES[system]}, "trials": rows, "skipped": skipped,  # noqa: E731
                        "failed": failed, "gains": dict(DEFAULTS[args.controller], **gains) if args.controller in DEFAULTS else None,
                        "motor_gain_scale": args.motor_gain_scale, "push_rad": args.push, "hold_s": args.hold,
                        "measured": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        for joint, start_rad, goal, fraction, start_fraction in trials:
            try:
                row = run_trial(engine, args.controller, joint, goal, None if start_rad is None else {joint: start_rad}, gains,
                                args.motor_gain_scale, args.push, args.hold, args.disturb_at)
            except EngineError as exc:
                failed.append({"joint": joint, "system": system, "goal_fraction": fraction, "error": str(exc)})
                print("  %s arm_%d goal fraction %.2f: FAILED, %s" % (system, joint + 1, fraction, exc), flush=True)
                continue
            row.update(start_fraction=start_fraction, goal_fraction=fraction)
            rows.append(row)
            write(args.out, document())
            print("  %s arm_%d goal %+.3f: start %+.3f final %+.3f (error %+.4f), overshoot %.3f, %d steps %s"
                  % (system, joint + 1, goal, row["start_rad"], row["final_rad"], row["goal_error_rad"], row["overshoot_rad"],
                     row["n_steps"], row["stop_reason"]), flush=True)
    finally:
        if transport is not None:
            try:
                transport.unpause()
            finally:
                transport.shutdown()
    if rows:
        print(summary_line(SYSTEM_NAMES[system], rows))
    path = write(args.out, document())
    if path:
        print("written:", path)
    return 0


def tune_command(args):
    """The gain grid on arm_6 and arm_5 from three start poses (goals that are in no test set)."""

    transport, _ = open_transport(args, False)
    engine = make_engine(args.engines, transport)
    trials = []
    for joint in args.joints:
        lo, hi = joint_limits(joint)
        for start in (0.2, 0.5, 0.8):
            for goal in (0.07, 0.46, 0.88):
                if abs(goal - start) >= 0.1:
                    trials.append((joint, lo + start * (hi - lo), lo + goal * (hi - lo)))
    results = []
    try:
        grid = itertools.product(args.kp_values or TUNE_GRID["kp"], args.ki_values or TUNE_GRID["ki"], TUNE_GRID["kd"],
                                 args.speed_fractions or TUNE_GRID["speed_fraction"])
        for kp, ki, kd, fraction in grid:
            rows = [run_trial(engine, "p", joint, goal, {joint: start}, {"kp": kp, "ki": ki, "kd": kd, "speed_fraction": fraction})
                    for joint, start, goal in trials]
            errors = np.abs([r["goal_error_rad"] for r in rows])
            over = [r["overshoot_rad"] for r in rows]
            settle = [r["settling_time_s"] for r in rows if r["reached_tolerance"]]
            results.append({"kp": kp, "ki": ki, "kd": kd, "speed_fraction": fraction, "trials": len(rows), "error_median": float(np.median(errors)),
                            "error_p90": float(np.percentile(errors, 90)), "error_max": float(errors.max()),
                            "overshoot_p90": float(np.percentile(over, 90)), "overshoot_max": float(max(over)),
                            "settled": float(np.mean([r["reached_tolerance"] for r in rows])),
                            "settling_time_median": float(np.median(settle)) if settle else float("nan"),
                            "max_steps": int(sum(r["stop_reason"] == "max_steps" for r in rows))})
            print("kp %4.1f ki %3.1f kd %4.2f speed %.1f: error median %.4f p90 %.4f max %.4f | overshoot p90 %.4f max %.4f | settled %3d%% in %.2f s | at the step limit %d"
                  % (kp, ki, kd, fraction, results[-1]["error_median"], results[-1]["error_p90"], results[-1]["error_max"], results[-1]["overshoot_p90"],
                     results[-1]["overshoot_max"], round(100 * results[-1]["settled"]), results[-1]["settling_time_median"], results[-1]["max_steps"]), flush=True)
    finally:
        if transport is not None:
            try:
                transport.unpause()
            finally:
                transport.shutdown()
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        name = "pid_grid%s%s.json" % ("" if not args.kp_values else "_kp" + "_".join("%g" % k for k in args.kp_values),
                                      "" if not args.speed_fractions else "_v" + "_".join("%g" % k for k in args.speed_fractions))
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as stream:
            json.dump(json_safe({"engines": args.engines, "joints": args.joints, "trials_per_setting": len(trials), "grid": results}), stream, indent=1)
        choose(args.out)
    return 0


def choose(directory):
    """Apply the tuning rule to every ``pid_grid*.json`` in ``directory`` and write ``pid_tuning.json``."""

    import glob

    results, meta = [], {}
    for path in sorted(glob.glob(os.path.join(directory, "pid_grid*.json"))):
        with open(path, encoding="utf-8") as stream:
            document = json.load(stream)
        results.extend(dict({"speed_fraction": 1.0}, **row) for row in document["grid"])
        meta = {key: document[key] for key in ("engines", "joints", "trials_per_setting")}
    results.sort(key=lambda r: (-r["speed_fraction"], r["kp"], r["ki"], r["kd"]))
    allowed = [r for r in results if r["overshoot_p90"] < TUNE_OVERSHOOT and r["settled"] >= 1.0]
    best = min(allowed or results, key=lambda r: (round(r["error_p90"], 3), r["settling_time_median"]))
    rule = ("among the settings that settle every trial with p90 overshoot under %.2f rad: the lowest p90 goal error "
            "(to 0.001 rad), ties to the shorter settling time" % TUNE_OVERSHOOT)
    print("chosen of %d settings (%d allowed): kp %.1f ki %.1f kd %.2f speed fraction %.1f | %s"
          % (len(results), len(allowed), best["kp"], best["ki"], best["kd"], best["speed_fraction"], rule))
    with open(os.path.join(directory, "pid_tuning.json"), "w", encoding="utf-8") as stream:
        json.dump(dict(meta, rule=rule, grid=results, chosen=best), stream, indent=1)
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--controller", choices=sorted(SYSTEMS), required=True)
    p.add_argument("--engines", choices=("fake", "full"), default="full")
    p.add_argument("--joints", type=int, nargs="+", default=[5, 4])
    p.add_argument("--trials", type=int, default=8, help="goals per joint, start and seed (the plan of validate_velocity.py)")
    p.add_argument("--starts", type=float, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[13579], help="seeds of the goal plan (the baselines have no random part)")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--screen-moveit", action="store_true")
    p.add_argument("--kp", type=float, default=None)
    p.add_argument("--ki", type=float, default=None)
    p.add_argument("--kd", type=float, default=None)
    p.add_argument("--speed-fraction", type=float, default=None, help="output limit as a fraction of the joint's velocity limit")
    p.add_argument("--motor-gain-scale", type=float, default=1.0, help="PID: the joint moves this factor times the command")
    p.add_argument("--push", type=float, default=0.0)
    p.add_argument("--hold", type=float, default=0.0)
    p.add_argument("--disturb-at", type=float, default=1.0)
    p.add_argument("--out", default=None)
    p = sub.add_parser("tune")
    p.add_argument("--engines", choices=("fake", "full"), default="full")
    p.add_argument("--joints", type=int, nargs="+", default=[5, 4])
    p.add_argument("--kp-values", type=float, nargs="+", default=None, help="only these kp of the grid (to split the grid over simulations)")
    p.add_argument("--ki-values", type=float, nargs="+", default=None)
    p.add_argument("--speed-fractions", type=float, nargs="+", default=None, help="only these output limits, as fractions of the joint's velocity limit")
    p.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    return {"run": run_command, "tune": tune_command}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
