"""Decoder calibration for one Kangaroo arm joint, and loading its results: port of TiagoCalibration.calibration as
its main() runs it (src/tiago_ring_controller/src/calibrate_single_joint.py:1091-1386). Decisions: DECISIONS.md D16-D25.

Each trial: reset the arm to the start pose, draw a goal over the joint range, and run the ring loop (ring.run_trial)
under CALIBRATION_PROFILE (Q9) with the current decoder. Every FIT_EVERY trials, the existing fit
(training/calibration.py) grid-searches tau/delay and solves, by regularised least squares,
    dq_desired = k_pos·S_pos + k_neg·S_neg,   S_pos/neg = sum(max(+/-d_delayed, 0))·dt   [rad = (rad/s per drive)·drive·s]
and blends p <- (1-beta)·p + beta·p_fit (delay taken as fitted).
Seeds: trial i uses NEST seed `seed + i`; goals come from numpy default_rng(seed) (TIAGo's run was unseeded).
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .arm import DT
from .ring import run_trial
from tiago_ring_controller.contracts import DecoderSpec
from tiago_ring_controller.control.controller import DecoderParameters
from tiago_ring_controller.control.profiles import CALIBRATION_PROFILE
from tiago_ring_controller.training.calibration import fit_decoder_from_trials, update_decoder_parameters

N_TRIALS, FIT_EVERY, BETA = 100, 3, 0.5  # calibrate_single_joint.py main(): max_iterations, fit_every, beta


def with_decoder(setup, spec):
    """`setup` under CALIBRATION_PROFILE (D17) with the decoder `spec` describes."""
    return replace(setup, profile=CALIBRATION_PROFILE,
                   decoder=DecoderParameters(spec.gain_positive, spec.gain_negative, spec.tau, spec.delay_steps))


def calibrate(arm, setup, seed, n_trials=N_TRIALS, log=print, balanced=False):
    """Calibrate `setup.joint` over its ring range (D20), each trial from its start pose (D25), under
    CALIBRATION_PROFILE whatever its profile; its decoder is ignored. Returns the final DecoderSpec and per-trial
    summaries; with_decoder(setup, spec) is the calibrated setup.

    balanced (D23): even trials draw the goal from (start, max), odd trials from (min, start), so every 3-trial fit
    sees drive of both signs; False keeps TIAGo's uniform draw over the whole range.
    """
    joint, limits, start = setup.joint, setup.limits, setup.start
    rng = np.random.default_rng(seed)
    spec, batch, trials = DecoderSpec(), [], []
    for i in range(n_trials):
        arm.reset(start)
        lo, hi = limits
        if balanced:
            lo, hi = (start[joint], hi) if i % 2 == 0 else (lo, start[joint])
        goal = float(rng.uniform(lo, hi))
        t = run_trial(arm, with_decoder(setup, spec), goal, seed + i)
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


def load_limits(path, side, joint):
    """(joint_min, joint_max) rad from a limits.json written by scripts/joint_limits.py for this side and joint."""
    lim = json.loads(Path(path).read_text())
    assert (lim['side'], lim['joint']) == (side, joint), lim
    return lim['joint_min'], lim['joint_max']

