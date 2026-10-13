"""Decoder calibration for one Kangaroo arm joint: port of TiagoCalibration.calibration as its main() runs it
(src/tiago_ring_controller/src/calibrate_single_joint.py:1091-1386). Decisions: docs/kangaroo/DECISIONS.md D16-D19.

Each trial: reset the arm to PAL's home pose, draw a goal uniformly over the joint range, and run the ring loop
(ring_reach.run_trial) under CALIBRATION_PROFILE (Q9) with the current decoder. Every FIT_EVERY trials, the existing
fit (training/calibration.py) grid-searches tau/delay and solves, by regularised least squares,
    dq_desired = k_pos·S_pos + k_neg·S_neg,   S_pos/neg = sum(max(+/-d_delayed, 0))·dt   [rad = (rad/s per drive)·drive·s]
and blends p <- (1-beta)·p + beta·p_fit (delay taken as fitted).
Seeds: trial i uses NEST seed `seed + i`; goals come from numpy default_rng(seed) (TIAGo's run was unseeded).

Run from src/tiago_ring_controller/src, simulation running:
    python3 /work/kangaroo/calibrate.py <left|right> <joint 0-3> [--trials 100] [--seed 2026]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.parameter import Parameter

from arm_velocity import DT, ArmVelocityPublisher, sleep_sim, wait_ready
from ring_reach import CALIBRATION_PROFILE, LIMITS, run_trial
from tiago_ring_controller.contracts import DecoderSpec
from tiago_ring_controller.control.controller import DecoderParameters
from tiago_ring_controller.training.calibration import fit_decoder_from_trials, update_decoder_parameters

HOME = {'left': [0.239, 1.321, 1.57, 0.8], 'right': [-0.239, 1.321, -1.57, 0.8]}  # PAL 4-DoF home (KB 4c)
N_TRIALS, FIT_EVERY, BETA = 100, 3, 0.5  # calibrate_single_joint.py main(): max_iterations, fit_every, beta


def reset_arm(node, arm, side, seconds=2.0, attempts=3, pose=None):
    """Return the arm to PAL's home pose, as reset_pose did on TIAGo (tiago_controller.py:240-259).

    Checks the controller reference reached home and resends if not: in a fresh process right after a trial that
    ended on max_steps, a single reset was ignored (2 of 20 evaluation resets; reproduced 1 of 3; cause unknown).
    With retries: 4 of 4 succeeded (D18). `pose` (rad, 4 joints) replaces home (D25).
    """
    pose = pose or HOME[side]
    for _ in range(attempts):
        arm.move_to(pose, seconds)
        sleep_sim(node, seconds + 1.0)
        if max(abs(r - h) for r, h in zip(arm.reference_positions, pose)) < 1e-3:
            return
    raise RuntimeError(f'{side} arm did not reach {pose} after {attempts} attempts')


def calibrate(node, arm, side, joint, seed, n_trials=N_TRIALS, log=print, limits=None, balanced=False, start=None):
    """Calibrate `joint` over `limits` (rad; default URDF, D20); returns the final DecoderSpec and per-trial summaries.

    balanced (D23, owner Q11 (d)): even trials draw the goal from (start, max), odd trials from (min, start), so every
    3-trial fit sees drive of both signs; False keeps TIAGo's uniform draw over the whole range.
    start (D25, owner Q13 (c)): pose (rad, 4 joints) each trial starts from; default PAL home.
    """
    limits = tuple(limits or LIMITS[joint])
    start = start or HOME[side]
    rng = np.random.default_rng(seed)
    spec, batch, trials = DecoderSpec(), [], []
    for i in range(n_trials):
        reset_arm(node, arm, side, pose=start)
        lo, hi = limits
        if balanced:
            lo, hi = (start[joint], hi) if i % 2 == 0 else (lo, start[joint])
        goal = float(rng.uniform(lo, hi))
        decoder = DecoderParameters(spec.gain_positive, spec.gain_negative, spec.tau, spec.delay_steps)
        t = run_trial(node, arm, joint, goal, seed + i, CALIBRATION_PROFILE, decoder, limits)
        trial = dict(spike_history=t['signed_drive'], dt=DT, dq_desired=goal - t['q_start'],
                     dq_actual=t['q_final'] - t['q_start'], q_start=t['q_start'], q_goal=goal,
                     q_final=t['q_final'], stop_reason=t['stop_reason'], seed=seed + i, decoder_used=spec.__dict__)
        batch.append(trial)
        trials.append(trial)
        log(f'[{i + 1}/{n_trials}] goal {goal:+.3f} dq_desired {trial["dq_desired"]:+.3f} '
            f'dq_actual {trial["dq_actual"]:+.3f} {t["stop_reason"]}')
        if len(batch) >= FIT_EVERY or i == n_trials - 1:
            if len(batch) >= 2:
                fit = fit_decoder_from_trials(batch)
                spec = update_decoder_parameters(spec, fit, BETA)
                trial['fit'] = None if fit is None else {k: v for k, v in fit.as_legacy_dict().items()
                                                         if not isinstance(v, np.ndarray)}
                log(f'  fit {trial["fit"]} -> {spec}')
            batch = []
    return spec, trials


def main():
    parser = argparse.ArgumentParser(description='Calibrate the ring decoder for one Kangaroo arm joint.')
    parser.add_argument('side', choices=['left', 'right'])
    parser.add_argument('joint', type=int, choices=range(len(LIMITS)), help='joint index 0-3')
    parser.add_argument('--trials', type=int, default=N_TRIALS)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--limits', help='limits.json from joint_limits.py (D20); default: URDF range')
    parser.add_argument('--balanced', action='store_true', help='alternate goals above/below the start (D23)')
    parser.add_argument('--start-joint', type=float, help='start the calibrated joint here (rad) instead of home (D25)')
    args = parser.parse_args()
    limits = LIMITS[args.joint]
    if args.limits:
        lim = json.loads(Path(args.limits).read_text())
        assert (lim['side'], lim['joint']) == (args.side, args.joint), lim
        limits = (lim['joint_min'], lim['joint_max'])
    start = list(HOME[args.side])
    if args.start_joint is not None:
        start[args.joint] = args.start_joint

    rclpy.init()
    node = rclpy.create_node(f'calibrate_{args.side}', parameter_overrides=[Parameter('use_sim_time', value=True)])
    arm = ArmVelocityPublisher(node, args.side)
    if not wait_ready(node, [arm]):
        sys.exit('simulation or arm controller not running')
    spec, trials = calibrate(node, arm, args.side, args.joint, args.seed, args.trials,
                             limits=limits, balanced=args.balanced, start=start)
    reset_arm(node, arm, args.side)
    rclpy.shutdown()

    out_dir = Path(__file__).resolve().parent / 'results/calibration' / \
        f'{args.side}_j{args.joint}_seed{args.seed}_{time.strftime("%Y%m%d_%H%M%S")}'  # new dir per run (rule 15)
    out_dir.mkdir(parents=True)
    path = out_dir / 'calibration.json'
    path.write_text(json.dumps(dict(
        side=args.side, joint=args.joint, profile=CALIBRATION_PROFILE.name, seed=args.seed, n_trials=args.trials,
        fit_every=FIT_EVERY, beta=BETA, joint_limits=limits, limits_file=args.limits, balanced=args.balanced,
        start_pose=start, decoder_gain_positive=spec.gain_positive,
        decoder_gain_negative=spec.gain_negative, decoder_tau=spec.tau, decoder_delay_steps=spec.delay_steps,
        trials=trials), indent=1))
    print(f'{spec} -> {path}')


if __name__ == '__main__':
    main()
