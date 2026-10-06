#!/usr/bin/env python3
"""Actuator check: how the arm follows streamed velocity commands (run inside the container).

Requires ``roslaunch tiago_ring_controller tiago_ring_controller.launch gui:=false``.
It does not use NEST.  For each joint it plays a velocity-step profile through
the real command path (``GazeboRosEngine``: receding horizon on
``/arm_controller/command``, lock-stepped physics) and reports, from positions
and message stamps only:

* displacement gain: joint displacement / commanded displacement, per segment and overall;
* lag: steady-state position error divided by the commanded velocity;
* static offset: position error while the command is zero;
* simulated time per tick and the stepper's overshoot.

    python3 -B scripts/identify_actuator.py --joints 5 4 --out /tmp/actuator.json
    python3 -B scripts/identify_actuator.py --joints 5 --artifact /tmp/cal/transport_calibration.json

Pass: |gain - 1| <= 0.03 and lag under one tick.  ``--artifact`` writes the
result into the ``actuator`` section of a transport calibration file; a joint
out of tolerance then gets its decoder gain divided by the measured gain.
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

from tiago_ring_controller.cosim import DataPack  # noqa: E402
from tiago_ring_controller.cosim.gazebo_ros_engine import GazeboRosEngine, RospyTransport  # noqa: E402
from tiago_ring_controller.cosim.limits import joint_limits_from_transport  # noqa: E402
from tiago_ring_controller.graph.calibration import ACTUATOR_TOLERANCE  # noqa: E402

#: (velocity in rad/s, ticks); zero segments separate the steps
PROFILE = ((0.0, 10), (0.1, 40), (0.0, 10), (-0.1, 40), (0.0, 10), (0.4, 20), (0.0, 10), (-0.4, 20), (0.0, 10), (0.02, 40), (0.0, 10))
#: the speeds of time_scale 2-4 (docs/archive/velocity/speed.md); each step moves at most 0.45 rad and comes back
FAST_PROFILE = ((0.0, 10), (0.8, 10), (0.0, 10), (-0.8, 10), (0.0, 10), (1.5, 6), (0.0, 10), (-1.5, 6), (0.0, 10))
PROFILES = {"standard": PROFILE, "fast": FAST_PROFILE}
HORIZON = 4


def run_profile(engine, transport, joint, profile, dt_ms, limits):
    """Play the profile on one joint; returns per-tick rows and the positions before and after."""

    engine.reset()
    dt_s = dt_ms / 1000.0
    velocities = np.concatenate([np.full(ticks, value, dtype=float) for value, ticks in profile] + [np.zeros(HORIZON)])
    start = transport.latest_joint_state()[0][joint]
    rows = []
    for k in range(len(velocities) - HORIZON):
        horizon = [float(v) for v in velocities[k:k + HORIZON]]
        command = {"joint_index": joint, "velocities": horizon}
        if limits is not None:
            command["limits"] = list(limits)
        engine.set_datapacks({"arm_velocity_cmd": DataPack("arm_velocity_cmd", engine.t_ms, {
            "joint_index": joint, "velocities": horizon, "dt_s": dt_s, "commands": [command]})})
        engine.advance(dt_ms)
        state = engine.get_datapacks()["joint_state"]
        step = engine.last_step
        rows.append((velocities[k], state["positions"][joint], state["commanded_positions"][joint], state["velocities_fd"][joint],
                     state["stamp_ms"] / 1000.0, step.sim_time_before_s, step.sim_time_after_s, step.overshoot_s, step.wall_s,
                     float(bool(state["clamped_joints"]))))
    finished = time.monotonic()
    engine.finish_trial()
    return start, transport.latest_joint_state()[0][joint], np.array(rows), time.monotonic() - finished


def analyse(joint, profile, start, end, rows, dt_ms, finish_wall_s):
    dt_s = dt_ms / 1000.0
    command, position, commanded, _, stamp, before, after, overshoot, wall, clamped = rows.T
    error = position - commanded
    rest = np.concatenate([np.arange(k, k + n) for k, (value, n) in zip(np.cumsum([0] + [n for _, n in profile[:-1]]), profile) if value == 0.0])
    static_offset = float(np.mean(error[rest[5:]]))
    segments = []
    k = 0
    for value, ticks in profile:
        index = slice(k, k + ticks)
        k += ticks
        if value == 0.0:
            continue
        previous = position[index.start - 1] if index.start > 0 else start
        moved = float(position[index][-1] - previous)
        steady = slice(index.start + ticks // 2, index.stop)
        measured = float(np.polyfit(stamp[steady], position[steady], 1)[0])       # slope against message stamps
        segments.append({
            "velocity": value, "ticks": ticks, "moved_rad": moved, "commanded_rad": value * ticks * dt_s,
            "gain": moved / (value * ticks * dt_s), "measured_velocity": measured,
            "lag_ms": float(-1000.0 * (np.mean(error[steady]) - static_offset) / value),
        })
    commanded_path = float(np.sum(np.abs(command)) * dt_s)
    travelled = float(sum(abs(s["moved_rad"]) for s in segments))
    gain = travelled / commanded_path
    lag = float(np.median([s["lag_ms"] for s in segments if abs(s["velocity"]) >= 0.1]))
    return {
        "joint": joint, "displacement_gain": gain, "lag_ms": lag, "static_offset_rad": static_offset,
        "max_position_error_rad": float(np.max(np.abs(error))), "net_error_rad": float((end - start) - np.sum(command) * dt_s),
        "tick_sim_s": {"mean": float(np.mean(after - before)), "min": float(np.min(after - before)), "max": float(np.max(after - before))},
        "overshoot_ms": {"mean": float(1000 * overshoot.mean()), "max": float(1000 * overshoot.max())},
        "tick_wall_ms": float(1000 * wall.mean()), "finish_wall_s": finish_wall_s, "clamped_ticks": int(clamped.sum()),
        "within_tolerance": bool(abs(gain - 1.0) <= ACTUATOR_TOLERANCE and abs(lag) < dt_ms),
        "segments": segments,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joints", type=int, nargs="+", default=[5])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard", help="velocity steps: up to 0.4 rad/s, or 0.8 and 1.5 rad/s")
    parser.add_argument("--dt-ms", type=float, default=50.0)
    parser.add_argument("--stepper", choices=("clock_wait", "plugin"), default="clock_wait")
    parser.add_argument("--out", default=None, help="write the measurements as JSON")
    parser.add_argument("--artifact", default=None, help="transport calibration file whose actuator section is updated")
    args = parser.parse_args(argv)

    transport = RospyTransport(node_name="identify_actuator")
    engine = GazeboRosEngine(transport, stepper_name=args.stepper)
    engine.initialize()
    report = {"dt_ms": args.dt_ms, "stepper": args.stepper, "profile": args.profile, "measured": time.strftime("%Y-%m-%d %H:%M:%S"), "joints": {}}
    try:
        for joint in args.joints:
            limits = joint_limits_from_transport(transport, joint)
            reset_started = time.monotonic()
            profile = PROFILES[args.profile]
            start, end, rows, finish_wall_s = run_profile(engine, transport, joint, profile, args.dt_ms, limits)
            result = analyse(joint, profile, start, end, rows, args.dt_ms, finish_wall_s)
            result["urdf_limits"] = limits
            result["trial_wall_s"] = time.monotonic() - reset_started
            report["joints"][str(joint)] = result
            print("joint %d: displacement gain %.4f, lag %.0f ms, static offset %+.4f rad, max |q - q_cmd| %.4f rad, %s"
                  % (joint, result["displacement_gain"], result["lag_ms"], result["static_offset_rad"],
                     result["max_position_error_rad"], "PASS" if result["within_tolerance"] else "OUT OF TOLERANCE"))
            print("   sim time per tick %.4f s (min %.4f, max %.4f), overshoot %.2f ms, %.0f ms wall per tick, finish %.1f s"
                  % (result["tick_sim_s"]["mean"], result["tick_sim_s"]["min"], result["tick_sim_s"]["max"],
                     result["overshoot_ms"]["mean"], result["tick_wall_ms"], finish_wall_s))
            for segment in result["segments"]:
                print("   v = %+.2f rad/s x %2d ticks: moved %+.4f of %+.4f rad (gain %.3f), measured %+.4f rad/s, lag %.0f ms"
                      % (segment["velocity"], segment["ticks"], segment["moved_rad"], segment["commanded_rad"], segment["gain"],
                         segment["measured_velocity"], segment["lag_ms"]))
    finally:
        engine.shutdown()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        print("written:", args.out)
    if args.artifact:
        with open(args.artifact, encoding="utf-8") as stream:
            artifact = json.load(stream)
        section = artifact.setdefault("actuator", {})
        for joint, result in report["joints"].items():
            section[joint] = {key: result[key] for key in ("displacement_gain", "lag_ms", "static_offset_rad", "within_tolerance")}
            section[joint]["measured"] = report["measured"]
        with open(args.artifact, "w", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2)
            stream.write("\n")
        print("actuator section updated:", args.artifact)
    return 0 if all(r["within_tolerance"] for r in report["joints"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
