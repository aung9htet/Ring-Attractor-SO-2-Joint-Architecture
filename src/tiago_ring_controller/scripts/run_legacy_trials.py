#!/usr/bin/env python3
"""Run the frozen legacy controllers on chosen goals, for comparison (docs/results.md).

    python3 -B scripts/run_legacy_trials.py --joint 5 --goals 0.8 -0.6 --repeats 1 --out /tmp/legacy
    python3 -B scripts/run_legacy_trials.py --joint 5 --plan 8 --out /tmp/legacy    # validate_velocity's goals
    python3 -B scripts/run_legacy_trials.py --controller pid --joint 5 --plan 8 --out /tmp/pid

``--controller ring`` drives ``legacy/single_joint_data_collector.py``
(``SingleRingModel`` built per synapse, a ``rospy.sleep`` loop that is not
lock-stepped, hand gains from ``velocity_calibration.json``); ``--controller
pid`` drives ``legacy/single_joint_pid_data_collector.py`` (the PID baseline of
the paper).  Neither file is changed: a subclass only redirects the output
directory, and the force-torque CSV logger they would start is replaced by a
stub.  Each trial follows the collector's ``run()``: the reset motion,
``prime_command_state``, its own ``run_trial(goal)``, then (ring only) a fresh
network for the next trial.  ``--plan N`` takes the goals
``scripts/validate_velocity.py`` gives the graph systems from the reset pose.
``--motor-gain-scale``, ``--push`` and ``--hold`` apply the robustness tests of
``scripts/validate_velocity.py`` from outside the collector (``LegacyDisturbance``).
Needs the simulation running and unpaused.  Writes one JSON per trial (scalars
and timeseries) plus ``<system>_trials_j<joint>.json`` with the metrics of
``tiago_ring_controller.evaluation.motion`` and the joint trace of every trial.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
LEGACY = os.path.join(os.path.dirname(HERE), "legacy")
for path in (SRC, LEGACY, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from tiago_ring_controller.control.profiles import COLLECTOR_PROFILE  # noqa: E402
from tiago_ring_controller.evaluation.motion import motion_metrics  # noqa: E402
from tiago_ring_controller.evaluation.serialization import json_safe  # noqa: E402

#: controller -> (legacy module, collector class, system name in the result files)
CONTROLLERS = {
    "ring": ("single_joint_data_collector", "SingleJointDataCollector", "legacy"),
    "pid": ("single_joint_pid_data_collector", "SingleJointPIDDataCollector", "pid"),
}


class _NoLogger:
    """Stands in for ``TiagoSubscriber``: no CSV file, no subscriptions."""

    def __init__(self, *args, **kwargs):
        pass

    def shutdown(self):
        pass


class BaseKeeper:
    """Puts the robot's mobile base back where it was spawned and tells how far it was pushed.

    The base is not anchored: an arm that pushes on the table shoves the whole
    robot, and the next trial would start in a different world.  The graph
    systems do the same in ``GazeboRosEngine`` (``base_pose``).
    """

    def __init__(self, model="tiago", pose=(0.0, 0.0, 0.0)):
        self.model, self.pose = model, tuple(pose)
        try:
            from tiago_ring_controller.cosim.gazebo_ros_engine import RospyTransport

            self.transport = RospyTransport(init_node=False)   # only its Gazebo model services are used
        except ImportError:                                    # no Gazebo messages: nothing to keep
            self.transport = None

    def read(self):
        return None if self.transport is None else self.transport.model_pose(self.model)

    def restore(self):
        import rospy

        found = self.read()
        if self.transport is not None and self.transport.set_model_pose(self.model, *self.pose):
            rospy.sleep(1.0)
        return found

    @staticmethod
    def distance(a, b):
        return float("nan") if a is None or b is None else float(np.hypot(a[0] - b[0], a[1] - b[1]))


class LegacyDisturbance:
    """The robustness tests of ``scripts/validate_velocity.py`` for a frozen collector.

    The collector is not changed: its publisher's ``publish_receding_trajectory``
    is wrapped for the duration of one trial.  ``motor_gain_scale`` multiplies
    the velocities it sends (the joint moves that factor times what the
    controller asked for); ``push_rad`` shifts the commanded position once,
    ``at_s`` into the trial; ``hold_s`` drops the commands for that long from
    ``at_s`` (the joint stays where it is).  Time is counted in commands of
    ``dt_s``, as the collector sends one per tick.
    """

    def __init__(self, publisher, joint, dt_s, motor_gain_scale=1.0, push_rad=0.0, hold_s=0.0, at_s=1.0):
        self.publisher, self.joint, self.dt_s = publisher, int(joint), float(dt_s)
        self.scale, self.push_rad, self.hold_s, self.at_s = float(motor_gain_scale), float(push_rad), float(hold_s), float(at_s)
        self.calls, self.pushed, self.dropped = 0, False, 0
        self._original = None

    @property
    def active(self):
        return self.scale != 1.0 or bool(self.push_rad) or bool(self.hold_s)

    def __enter__(self):
        if self.active:
            self._original = self.publisher.publish_receding_trajectory
            self.publisher.publish_receding_trajectory = self._publish
        return self

    def __exit__(self, *exc):
        if self._original is not None:
            self.publisher.publish_receding_trajectory = self._original
            self._original = None
        return False

    def _publish(self, joint_idx, velocities, dt, *args, **kwargs):
        t_s = self.calls * self.dt_s + 1e-9
        self.calls += 1
        if self.push_rad and not self.pushed and t_s >= self.at_s:
            self.publisher.commanded_positions[self.joint] += self.push_rad
            self.pushed = True
        if self.hold_s and self.at_s <= t_s < self.at_s + self.hold_s:
            self.dropped += 1
            return None
        return self._original(joint_idx, [float(v) * self.scale for v in velocities], dt, *args, **kwargs)


def legacy_metrics(result, joint_min, joint_max, population_size=200, half_width=5, dt_s=0.05):
    """The comparison metrics from one ``run_trial`` result of a legacy collector.

    Motion metrics come from the recorded joint trace (the angle before every
    tick, then the settled final angle).  For the ring collector the ring's
    joint estimate is the recorded ``r1_centroid`` (its linear centroid, valid
    away from the seam) through the inverse of the collector mapping; row k
    pairs the consumed NEST sample with the joint measured after that tick.
    The PID collector has no ring: those fields are NaN.
    """

    scalars, series = result["scalars"], result["timeseries"]
    position = np.asarray(series["joint_position"], dtype=float)
    trace = np.append(position, float(scalars["q_final"]))
    metrics = motion_metrics(trace, float(scalars["q_goal"]), dt_s)
    metrics.update({
        "q_start": float(scalars["q_start"]), "q_goal": float(scalars["q_goal"]), "q_final": float(scalars["q_final"]),
        "joint_moved": float(scalars["q_final"] - scalars["q_start"]),
        "n_steps": int(scalars["n_steps"]), "stop_reason": scalars["stop_reason"],
        "bump_moved_rad": float("nan"), "tracking_rel": float("nan"),
        "state_error_abs_median": float("nan"), "state_error_abs_p95": float("nan"),
        "trace": {"dt_s": dt_s, "q": [round(float(v), 5) for v in trace],
                  "v_cmd": [round(float(v), 5) for v in np.asarray(series.get("decoded_velocity", []), dtype=float)]},
    })
    if "r1_centroid" in series:
        centroid = np.asarray(series["r1_centroid"], dtype=float)
        estimate = np.array([COLLECTOR_PROFILE.ring_index_to_joint(c, joint_min, joint_max, population_size, half_width)
                             if np.isfinite(c) else np.nan for c in centroid])
        good = np.isfinite(estimate)
        rho = COLLECTOR_PROFILE.radians_per_index(joint_min, joint_max, population_size, half_width)
        if good.any():
            bump_moved = float((np.nanmean(centroid[-4:]) - np.nanmean(centroid[:4])) * rho)
            metrics["bump_moved_rad"] = bump_moved
            if np.isfinite(bump_moved) and abs(bump_moved) > 1e-3:
                metrics["tracking_rel"] = float((metrics["joint_moved"] - bump_moved) / abs(bump_moved))
            metrics["state_error_abs_median"] = float(np.median(np.abs(estimate[good] - position[good])))
            metrics["state_error_abs_p95"] = float(np.percentile(np.abs(estimate[good] - position[good]), 95))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", choices=sorted(CONTROLLERS), default="ring")
    parser.add_argument("--joint", type=int, required=True)
    goals = parser.add_mutually_exclusive_group(required=True)
    goals.add_argument("--goals", type=float, nargs="+", help="goal angles (rad)")
    goals.add_argument("--plan", type=int, help="this many goals from validate_velocity's plan (reset-pose start)")
    parser.add_argument("--seed", type=int, default=13579, help="--plan: seed of the goal plan")
    parser.add_argument("--repeats", type=int, default=1, help="runs of every goal (repeatability)")
    parser.add_argument("--screen-moveit", action="store_true",
                        help="skip goals whose straight path MoveIt reports in collision (as validate_velocity does)")
    parser.add_argument("--motor-gain-scale", type=float, default=1.0, help="the joint moves this factor times the commanded velocity")
    parser.add_argument("--push", type=float, default=0.0, help="rad the joint is displaced mid-trial without the controller being told")
    parser.add_argument("--hold", type=float, default=0.0, help="seconds the joint is held mid-trial (commands dropped)")
    parser.add_argument("--disturb-at", type=float, default=1.0, help="time of --push / --hold in the trial, s")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import importlib

    import rospy

    module_name, class_name, system = CONTROLLERS[args.controller]
    collector_module = importlib.import_module(module_name)

    if args.plan:
        from validate_velocity import START_POSE, goal_plan
        from tiago_ring_controller.graph.templates import joint_limits

        lo, hi = joint_limits(args.joint)
        start = float(np.clip((START_POSE[args.joint] - lo) / (hi - lo), 0.0, 1.0))
        args.goals = [lo + fraction * (hi - lo) for fraction, _ in goal_plan(start, args.plan, args.seed)]
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    collector_module.TiagoSubscriber = _NoLogger

    class Collector(getattr(collector_module, class_name)):
        def _output_dir(self):
            path = os.path.join(out, "joint_%d" % self.target_joint)
            os.makedirs(path, exist_ok=True)
            return path

    rospy.init_node("legacy_trials", anonymous=True)
    skipped = []
    if args.screen_moveit:
        from validate_velocity import START_POSE, MoveItScreen

        class _Running:                               # the legacy loop never pauses the simulation
            def unpause(self):
                pass

        screen = MoveItScreen(_Running())
        valid = [g for g in args.goals if screen.path_is_valid(args.joint, START_POSE[args.joint], g)]
        skipped = [g for g in args.goals if g not in valid]
        args.goals = valid
        if skipped:
            print("skipped by MoveIt: %s" % ", ".join("%+.3f" % g for g in skipped))
    os.chdir(SRC)                                     # the legacy model reads ./config/ (as measure_legacy_tick.py does)
    build_started = time.monotonic()
    collector = Collector(args.joint, use_manual_limits=False)
    first_build_s = time.monotonic() - build_started
    rebuild = getattr(collector, "reset_ring_model", None)          # the PID collector has no network to rebuild
    base = BaseKeeper()
    rows = []
    for repeat in range(args.repeats):
        for goal in args.goals:
            started = time.monotonic()
            collector.publisher.reset_pose()
            found = base.restore()
            collector.publisher.prime_command_state(force=True)
            reset_s = time.monotonic() - started
            base_start = base.read()
            ran = time.monotonic()
            dt_s = getattr(collector, "time_step", 50.0) / 1000.0
            with LegacyDisturbance(collector.publisher, args.joint, dt_s, args.motor_gain_scale, args.push, args.hold, args.disturb_at):
                result = collector.run_trial(float(goal))
            run_s = time.monotonic() - ran
            base_end = base.read()
            built = time.monotonic()
            if rebuild is not None:
                rebuild()
            build_s = time.monotonic() - built
            metrics = legacy_metrics(result, collector.joint_min, collector.joint_max, dt_s=getattr(collector, "time_step", 50.0) / 1000.0)
            metrics.update(base_shift_m=base.distance(base_start, base_end), base_found_m=base.distance(found, base.pose))
            metrics.update(motor_gain_scale=args.motor_gain_scale, push_rad=args.push, hold_s=args.hold)
            metrics.update(system=system, joint=args.joint, goal=float(goal), repeat=repeat,
                           wall_s=time.monotonic() - started, reset_wall_s=reset_s, build_wall_s=build_s,
                           run_wall_s=run_s, tick_wall_ms=1000.0 * run_s / max(metrics["n_steps"], 1))
            rows.append(metrics)
            with open(os.path.join(out, "%s_j%d_goal%+.3f_r%d.json" % (system, args.joint, goal, repeat)), "w", encoding="utf-8") as stream:
                json.dump(json_safe({"metrics": metrics, "scalars": result["scalars"], "timeseries": result["timeseries"]}), stream)
            print("%s joint %d goal %+.3f repeat %d: start %+.3f final %+.3f error %+.3f, to 90%% %.1f s, settled %s, %d steps %s, "
                  "wall %.0f s (reset %.0f, build %.0f, %.0f ms/tick)"
                  % (system, args.joint, goal, repeat, metrics["q_start"], metrics["q_final"], metrics["goal_error_rad"],
                     metrics["time_to_90_s"], "%.1f s" % metrics["settling_time_s"] if metrics["reached_tolerance"] else "no",
                     metrics["n_steps"], metrics["stop_reason"], metrics["wall_s"], reset_s, build_s, metrics["tick_wall_ms"]))
    with open(os.path.join(out, "%s_trials_j%d.json" % (system, args.joint)), "w", encoding="utf-8") as stream:
        json.dump(json_safe({"controller": args.controller, "system": system, "joint": args.joint,
                             "first_build_s": first_build_s, "skipped_goals": skipped, "trials": rows}), stream, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
