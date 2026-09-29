#!/usr/bin/env python3
"""Phase 3 smoke test for lock-stepped Gazebo (run inside the container).

Requires ``roslaunch tiago_ring_controller tiago_ring_controller.launch gui:=false``
to be running.  It does not use NEST.

1. Pause physics; step 50 iterations twenty times; assert ``/clock`` advanced
   1.000 s within the measured overshoot (ClockWaitStepper) or exactly
   (PluginStepper).
2. Publish a constant-velocity four-point horizon for one joint through the
   unchanged ``CommandState``/``build_receding_trajectory`` contract while
   stepping, and report how far the joint moved versus the commanded angle
   (validates ``ros_control`` under stepping — Plan A gate G3 as well).

Nothing here calls ``rospy.sleep``; the simulation is left unpaused at exit.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

from tiago_ring_controller.cosim import DataPack  # noqa: E402
from tiago_ring_controller.cosim.gazebo_ros_engine import GazeboRosEngine, RospyTransport  # noqa: E402
from tiago_ring_controller.cosim.limits import joint_limits_from_transport  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joint", type=int, default=5)
    parser.add_argument("--joints", type=int, nargs="*", default=None,
                        help="phase 5: command these joints together (one trajectory per tick, e.g. --joints 5 6)")
    parser.add_argument("--velocity", type=float, default=0.2, help="rad/s for the motion check")
    parser.add_argument("--ticks", type=int, default=20)
    parser.add_argument("--stepper", choices=("clock_wait", "plugin"), default="clock_wait")
    parser.add_argument("--dt-ms", type=float, default=50.0)
    parser.add_argument("--out", default=None, help="write the measurements as JSON")
    parser.add_argument("--no-reset", action="store_true", help="skip play_motion; start from the current pose")
    args = parser.parse_args(argv)

    transport = RospyTransport(node_name="cosim_gazebo_smoke")
    engine = GazeboRosEngine(transport, stepper_name=args.stepper)
    engine.initialize()
    report = {"stepper": args.stepper, "dt_ms": args.dt_ms, "joint": args.joint}
    try:
        limits = joint_limits_from_transport(transport, args.joint)
        report["urdf_limits"] = limits
        print("URDF limits for joint %d: %s" % (args.joint, limits))

        if args.no_reset:
            transport.pause()
            engine.paused = True
            positions, velocities, _ = transport.latest_joint_state()
            engine.command_state.reset()
            engine.command_state.prime(positions, velocities, force=True)
        else:
            engine.reset()

        # 1. Clock stepping.
        t0 = transport.sim_time_s()
        for _ in range(args.ticks):
            engine.advance(args.dt_ms)
        t1 = transport.sim_time_s()
        overshoot = np.array([s.overshoot_s for s in engine.step_log]) * 1000.0
        wall = np.array([s.wall_s for s in engine.step_log]) * 1000.0
        expected = args.ticks * args.dt_ms / 1000.0
        report["clock"] = {
            "advanced_s": t1 - t0, "expected_s": expected,
            "overshoot_ms": {"mean": float(overshoot.mean()), "p95": float(np.percentile(overshoot, 95)),
                             "max": float(overshoot.max())},
            "step_wall_ms": {"mean": float(wall.mean()), "p95": float(np.percentile(wall, 95))},
        }
        print("clock advanced %.6f s (expected %.3f s); overshoot ms mean=%.3f p95=%.3f max=%.3f; "
              "wall ms mean=%.1f p95=%.1f"
              % (t1 - t0, expected, overshoot.mean(), np.percentile(overshoot, 95), overshoot.max(),
                 wall.mean(), np.percentile(wall, 95)))
        slack = max(overshoot.max() / 1000.0 * args.ticks, 1e-9)
        assert abs((t1 - t0) - expected) <= slack + 1e-6, "clock did not advance the expected amount"

        # 2. Constant-velocity horizon while stepping (several joints in one trajectory with --joints).
        del engine.step_log[:]
        joints = args.joints or [args.joint]
        # alternate the sign so two joints are visibly independent
        velocities = {joint: args.velocity * (1 if k % 2 == 0 else -1) for k, joint in enumerate(joints)}
        starts = {joint: transport.latest_joint_state()[0][joint] for joint in joints}
        for _ in range(args.ticks):
            engine.set_datapacks({
                "arm_velocity_cmd": DataPack("arm_velocity_cmd", engine.t_ms, {
                    "joint_index": joints[0], "velocities": [velocities[joints[0]]] * 4,
                    "dt_s": args.dt_ms / 1000.0,
                    "commands": [{"joint_index": joint, "velocities": [velocities[joint]] * 4} for joint in joints],
                })
            })
            engine.advance(args.dt_ms)
        report["motion"] = {}
        for joint in joints:
            moved = transport.latest_joint_state()[0][joint] - starts[joint]
            commanded = velocities[joint] * args.ticks * args.dt_ms / 1000.0
            report["motion"][str(joint)] = {"start": starts[joint], "moved_rad": moved, "commanded_rad": commanded,
                                            "tracking_ratio": moved / commanded if commanded else None}
            print("joint %d moved %.4f rad, commanded %.4f rad (ratio %.3f)"
                  % (joint, moved, commanded, moved / commanded if commanded else float("nan")))
        engine.finish_trial()
    finally:
        engine.shutdown()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        print("written:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
