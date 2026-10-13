"""Measure the reachable range of one Kangaroo arm joint: port of TiagoCalibration.find_joint_limits_manually
(src/tiago_ring_controller/src/calibrate_single_joint.py:1064-1086). Decision: docs/kangaroo/DECISIONS.md D20.

TIAGo drives the joint at -v for T seconds and reads the measured position as joint_min, then at +v for T as
joint_max (v = 1 rad/s, T = 5 s). Here the arm is reset to PAL's home pose before each sweep and at the end, so
the range is measured in the configuration calibration starts from, and a 5 rad sweep need not cross the whole
5.24 rad range of joints 0 and 2. The reading is the measured joint position [rad], so an obstruction (F9) bounds it.

Run from src/tiago_ring_controller/src, simulation running:
    python3 /work/kangaroo/joint_limits.py <left|right> <joint 0-3>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.parameter import Parameter

from arm_velocity import DT, HORIZON, ArmVelocityPublisher, sleep_sim, wait_ready
from calibrate import reset_arm

VELOCITY, SECONDS = 1.0, 5.0  # rad/s, s: calibrate_single_joint.py:1067-1068


def measure_limits(node, arm, side, joint, velocity=VELOCITY, seconds=SECONDS):
    """Return (joint_min, joint_max) in rad, measured after sweeping each way from home."""
    ends = []
    for v in (-velocity, velocity):
        reset_arm(node, arm, side)
        for _ in range(round(seconds / DT)):
            arm.publish_receding_trajectory(joint, [v] * HORIZON)
            sleep_sim(node, DT)
        ends.append(arm.current_positions[joint])
    reset_arm(node, arm, side)
    return tuple(ends)


def main():
    parser = argparse.ArgumentParser(description='Measure the reachable range of one Kangaroo arm joint.')
    parser.add_argument('side', choices=['left', 'right'])
    parser.add_argument('joint', type=int, choices=range(4), help='joint index 0-3')
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node('joint_limits', parameter_overrides=[Parameter('use_sim_time', value=True)])
    arm = ArmVelocityPublisher(node, args.side)
    if not wait_ready(node, [arm]):
        sys.exit('simulation or arm controller not running')
    lo, hi = measure_limits(node, arm, args.side, args.joint)
    rclpy.shutdown()

    out_dir = Path(__file__).resolve().parent / 'results/joint_limits' / \
        f'{args.side}_j{args.joint}_{time.strftime("%Y%m%d_%H%M%S")}'  # new dir per run (rule 15)
    out_dir.mkdir(parents=True)
    path = out_dir / 'limits.json'
    path.write_text(json.dumps(dict(side=args.side, joint=args.joint, joint_min=lo, joint_max=hi,
                                    velocity=VELOCITY, seconds=SECONDS), indent=1))
    print(f'{arm.joint_names[args.joint]}: [{lo:+.4f}, {hi:+.4f}] rad -> {path}')


if __name__ == '__main__':
    main()
