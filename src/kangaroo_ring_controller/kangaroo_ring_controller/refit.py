"""Batch refit of a calibration's decoder (DECISIONS.md F15, D39).

TIAGo's calibration fits every 3 trials and blends p <- (1-beta)·p + beta·p_fit (calibration.py); the final gains then
follow the last few 3-trial fits. Here TIAGo's own fit (training/calibration.py fit_decoder_from_trials: tau/delay grid,
regularised least squares dq_desired = k_pos·S_pos + k_neg·S_neg [rad]) runs once over the last `last` trials of a
recorded calibration, without blending. The ring never sees the robot, so the recorded drive is valid for any decoder.
"""
import json
from dataclasses import replace
from pathlib import Path

from . import RESULTS, new_run_dir
from .joint_setup import load, save
from tiago_ring_controller.control.controller import DecoderParameters
from tiago_ring_controller.training.calibration import fit_decoder_from_trials


def _fit(path, last):
    """Raises ValueError naming `path` if it has no trials (a refit) or the fit finds fewer than 2 usable trials."""
    if last < 2:
        raise ValueError(f'last = {last}: a fit needs at least 2 trials')
    record = json.loads(Path(path).read_text())
    if not record.get('trials'):
        raise ValueError(f'{path}: no trials to refit (a refit? refit its source)')
    fit = fit_decoder_from_trials(record['trials'][-last:])
    if fit is None:
        raise ValueError(f'{path}: fewer than 2 of the last {last} trials have drive; nothing to fit')
    return record, fit, replace(load(path), decoder=DecoderParameters(fit.k_pos, fit.k_neg, fit.tau, fit.delay_steps))


def refit(path, last=50):
    """The setup recorded in calibration.json `path`, with the decoder fitted once over its last `last` trials."""
    return _fit(path, last)[2]


def save_refit(path, last=50, root=RESULTS):
    """Save refit(path, last) as a new <root>/calibration/<side>_j<joint>_seed<seed>_<time>/calibration.json, which
    then is the joint's newest calibration; it names its source (`refit_of`) instead of copying the trials."""
    record, fit, setup = _fit(path, last)
    name = f'{setup.side}_j{setup.joint}_seed{record["seed"]}_refit{Path(path).parent.name[-15:]}'  # source's time
    out = new_run_dir('calibration', name, root) / 'calibration.json'
    save(setup, out, seed=record['seed'], refit_of=str(Path(path).resolve()), refit_trials=last, refit_desired_mae=fit.desired_mae,
         refit_actual_mae=fit.actual_mae, limits_file=record.get('limits_file'), balanced=record.get('balanced'))
    return out
