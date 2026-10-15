"""One ring-attractor reach of one Kangaroo arm joint, saved to results/ring_reach/ (DECISIONS.md D11-D25).

    python3 src/kangaroo_ring_controller/scripts/ring_reach.py <left|right> <joint 0-3> <goal rad> [--seed 12345]
        [--calibration results/calibration/.../calibration.json] [--from-home] [--limits .../limits.json]
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
import rclpy  # noqa: E402
from kangaroo_ring_controller import save_trial  # noqa: E402
from kangaroo_ring_controller.arm import add_arm_args, connect  # noqa: E402
from kangaroo_ring_controller.calibration import load_limits  # noqa: E402
from kangaroo_ring_controller.joint_setup import load, uncalibrated  # noqa: E402
from kangaroo_ring_controller.ring import run_trial  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Ring-attractor reach of one Kangaroo arm joint.')
    add_arm_args(parser)
    parser.add_argument('goal', type=float, help='goal position, rad')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--calibration', help='calibration.json from calibrate.py; runs under its profile (Q9)')
    parser.add_argument('--from-home', action='store_true', help='reset to the start pose first (D18, D25)')
    parser.add_argument('--limits', help='limits.json from joint_limits.py (D20), uncalibrated reaches only: a '
                        "calibration fixes its own ring range (D35); default: URDF range")
    args = parser.parse_args()
    if args.calibration and args.limits:
        parser.error('--limits is for uncalibrated reaches; the calibration fixes the ring range (D35)')
    setup = load(args.calibration) if args.calibration else uncalibrated(args.side, args.joint)
    if (setup.side, setup.joint) != (args.side, args.joint):
        parser.error(f'{args.calibration} is for {setup.side} joint {setup.joint}')
    if args.limits:
        setup = replace(setup, limits=load_limits(args.limits, args.side, args.joint))

    node, arms = connect('ring_reach', [args.side])
    arm = arms[args.side]
    if args.from_home:
        arm.reset(setup.start)
    result = run_trial(arm, setup, args.goal, args.seed)
    rclpy.shutdown()

    path, _ = save_trial('ring_reach', setup, result, from_home=args.from_home, limits_file=args.limits)
    print(f'{arm.joint_names[args.joint]}: {result["q_start"]:+.3f} -> {result["q_final"]:+.3f} rad '
          f'(goal {args.goal:+.3f}, {result["stop_reason"]}, {len(result["signed_spike"])} ticks) -> {path.parent}')


if __name__ == '__main__':
    main()
