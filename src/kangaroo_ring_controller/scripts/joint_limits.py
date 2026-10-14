"""Measure the reachable range of one Kangaroo arm joint; saves results/joint_limits/.../limits.json (D20).

    python3 src/kangaroo_ring_controller/scripts/joint_limits.py <left|right> <joint 0-3>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
import rclpy  # noqa: E402
from kangaroo_ring_controller import new_run_dir  # noqa: E402
from kangaroo_ring_controller.arm import add_arm_args, connect  # noqa: E402
from kangaroo_ring_controller.limits import SECONDS, VELOCITY, measure_limits  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Measure the reachable range of one Kangaroo arm joint.')
    add_arm_args(parser)
    args = parser.parse_args()

    node, arms = connect('joint_limits', [args.side])
    arm = arms[args.side]
    lo, hi = measure_limits(arm, args.joint)
    rclpy.shutdown()

    path = new_run_dir('joint_limits', f'{args.side}_j{args.joint}') / 'limits.json'
    path.write_text(json.dumps(dict(side=args.side, joint=args.joint, joint_min=lo, joint_max=hi,
                                    velocity=VELOCITY, seconds=SECONDS), indent=1))
    print(f'{arm.joint_names[args.joint]}: [{lo:+.4f}, {hi:+.4f}] rad -> {path}')


if __name__ == '__main__':
    main()
