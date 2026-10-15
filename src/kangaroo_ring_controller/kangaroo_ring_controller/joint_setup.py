"""Joint setup: everything a trial of one joint runs under, and its calibration.json (CONTEXT.md; DECISIONS.md D35).

Fields: side, joint, profile (CALIBRATION_PROFILE for calibrated gains, D17), decoder (DecoderParameters; None = the
collector's uncalibrated default), limits = ring range (joint_min, joint_max) rad (D20), start = start pose, 4 joint
angles rad (D25), calibration = the file it came from, or None. Frozen: equal setups build interchangeable rings.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from . import RESULTS, ring_core  # noqa: F401  (ring_core: TIAGo package on sys.path)
from tiago_ring_controller.control.controller import DecoderParameters
from tiago_ring_controller.control.profiles import CALIBRATION_PROFILE, COLLECTOR_PROFILE

LIMITS = [(-2.618, 2.618), (-1.134, 2.443), (-2.618, 2.618), (-1.134, 2.443)]  # PAL 4-DoF URDF, rad, joints 0-3
HOME = {'left': [0.239, 1.321, 1.57, 0.8], 'right': [-0.239, 1.321, -1.57, 0.8]}  # PAL 4-DoF home, rad (KB 4c)


@dataclass(frozen=True)
class JointSetup:
    side: str
    joint: int
    profile: object
    decoder: object
    limits: tuple
    start: tuple
    calibration: object = None


def save(setup, path, **run):
    """Write `setup` as calibration.json, plus the calibration run's own fields (seed, n_trials, trials, ...)."""
    d = setup.decoder
    Path(path).write_text(json.dumps(dict(
        side=setup.side, joint=setup.joint, profile=setup.profile.name, joint_limits=list(setup.limits),
        start_pose=list(setup.start), decoder_gain_positive=d.gain_positive, decoder_gain_negative=d.gain_negative,
        decoder_tau=d.tau_s, decoder_delay_steps=d.delay_steps, **run), indent=1))


def load(path):
    """The setup recorded in a calibration.json written by scripts/calibrate.py."""
    cal = json.loads(Path(path).read_text())
    assert cal['profile'] == CALIBRATION_PROFILE.name, path
    return JointSetup(cal['side'], cal['joint'], CALIBRATION_PROFILE,
                      DecoderParameters(cal['decoder_gain_positive'], cal['decoder_gain_negative'],
                                        cal['decoder_tau'], cal['decoder_delay_steps']),
                      tuple(cal['joint_limits']), tuple(cal.get('start_pose') or HOME[cal['side']]), str(path))


def uncalibrated(side, joint):
    """The collector's default decoder over the URDF range, starting from PAL home (D11)."""
    return JointSetup(side, joint, COLLECTOR_PROFILE, None, LIMITS[joint], tuple(HOME[side]))


def newest_setup(side, joint, root=RESULTS / 'calibration'):
    """Newest calibration for side/joint (by its directory's trailing YYYYmmdd_HHMMSS), else uncalibrated (D27)."""
    runs = sorted(Path(root).glob(f'{side}_j{joint}_*/calibration.json'), key=lambda p: p.parent.name[-15:])
    return load(runs[-1]) if runs else uncalibrated(side, joint)
