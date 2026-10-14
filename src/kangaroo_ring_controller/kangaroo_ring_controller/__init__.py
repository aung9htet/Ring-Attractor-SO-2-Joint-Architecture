"""Ring-attractor joint control of PAL's Kangaroo arms (ROS 2 Humble, MuJoCo).

Layout and why: docs/kangaroo/adr/0001-kangaroo-package-layout.md. Decisions: docs/kangaroo/DECISIONS.md.
"""
import json
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE_ROOT / 'results'


def new_run_dir(kind, name, root=RESULTS):
    """Create and return <root>/<kind>/<name>_<YYYYmmdd_HHMMSS>/: a new directory per run (CLAUDE.md rule 15)."""
    out = Path(root) / kind / f'{name}_{time.strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True)
    return out


def save_trial(kind, setup, result, root=RESULTS, **extra):
    """Write <root>/<kind>/<side>_j<joint>_seed<seed>_<time>/trial.json: run_trial's `result` plus the setup's side,
    joint and calibration file and `extra`; returns (path, trial) (D38)."""
    trial = dict(result, side=setup.side, joint=setup.joint, calibration=setup.calibration, **extra)
    path = new_run_dir(kind, f'{setup.side}_j{setup.joint}_seed{result["seed"]}', root) / 'trial.json'
    path.write_text(json.dumps(trial, indent=1))
    return path, trial
