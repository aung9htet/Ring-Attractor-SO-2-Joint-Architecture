"""Move one Kangaroo arm joint at a constant velocity and report how far it went (DECISIONS.md D1-D7).

    python3 src/kangaroo_ring_controller/scripts/arm_velocity.py <left|right> <joint 0-3> <velocity rad/s> <seconds>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
import rclpy  # noqa: E402
from kangaroo_ring_controller.arm import add_arm_args, connect  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Move one Kangaroo arm joint at a constant velocity.')
    add_arm_args(parser)
    parser.add_argument('velocity', type=float, help='rad/s')
    parser.add_argument('seconds', type=float)
    args = parser.parse_args()

    node, arms = connect('arm_velocity', [args.side])
    arm = arms[args.side]
    start = arm.current_positions[args.joint]
    arm.hold(args.joint, args.velocity, args.seconds)
    arm.stop_joint(args.joint)
    arm.sleep(1.0)
    end = arm.current_positions[args.joint]
    print(f'{arm.joint_names[args.joint]}: {start:+.3f} -> {end:+.3f} rad '
          f'(moved {end - start:+.3f}, expected {args.velocity * args.seconds:+.3f})')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
