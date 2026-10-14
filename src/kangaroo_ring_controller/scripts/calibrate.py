"""Calibrate the ring decoder for one Kangaroo arm joint; saves results/calibration/.../calibration.json (D16-D25).

    python3 src/kangaroo_ring_controller/scripts/calibrate.py <left|right> <joint 0-3> [--trials 100] [--seed 2026]
        [--limits .../limits.json] [--balanced] [--start-joint <rad>]
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
import rclpy  # noqa: E402
from kangaroo_ring_controller import new_run_dir  # noqa: E402
from kangaroo_ring_controller.arm import add_arm_args, connect  # noqa: E402
from kangaroo_ring_controller.calibration import (BETA, FIT_EVERY, N_TRIALS, calibrate, load_limits,  # noqa: E402
                                                  with_decoder)
from kangaroo_ring_controller.joint_setup import save, uncalibrated  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Calibrate the ring decoder for one Kangaroo arm joint.')
    add_arm_args(parser)
    parser.add_argument('--trials', type=int, default=N_TRIALS)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--limits', help='limits.json from joint_limits.py (D20); default: URDF range')
    parser.add_argument('--balanced', action='store_true', help='alternate goals above/below the start (D23)')
    parser.add_argument('--start-joint', type=float, help='start the calibrated joint here (rad) instead of home (D25)')
    args = parser.parse_args()
    setup = uncalibrated(args.side, args.joint)  # URDF range, PAL home
    if args.limits:
        setup = replace(setup, limits=load_limits(args.limits, args.side, args.joint))
    if args.start_joint is not None:
        setup = replace(setup, start=setup.start[:args.joint] + (args.start_joint,) + setup.start[args.joint + 1:])

    node, arms = connect(f'calibrate_{args.side}', [args.side])
    arm = arms[args.side]
    spec, trials = calibrate(arm, setup, args.seed, args.trials, balanced=args.balanced)
    arm.reset()
    rclpy.shutdown()

    path = new_run_dir('calibration', f'{args.side}_j{args.joint}_seed{args.seed}') / 'calibration.json'
    save(with_decoder(setup, spec), path, seed=args.seed, n_trials=args.trials, fit_every=FIT_EVERY, beta=BETA,
         limits_file=args.limits, balanced=args.balanced, trials=trials)
    print(f'{spec} -> {path}')


if __name__ == '__main__':
    main()
