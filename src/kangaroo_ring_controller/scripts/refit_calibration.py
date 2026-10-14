"""Refit recorded calibrations' decoders in one batch; each refit is saved as a new, newest calibration (D39, F15).

    python3 src/kangaroo_ring_controller/scripts/refit_calibration.py <calibration.json> ... [--last 50]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller.joint_setup import load  # noqa: E402
from kangaroo_ring_controller.refit import save_refit  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Batch-refit recorded calibrations (no robot needed).')
    parser.add_argument('calibrations', nargs='+', help='calibration.json files from calibrate.py')
    parser.add_argument('--last', type=int, default=50, help='fit over the last N trials (default 50)')
    args = parser.parse_args()
    for source in args.calibrations:
        path = save_refit(source, args.last)
        print(f'{source}: {load(source).decoder} -> {load(path).decoder} -> {path}')


if __name__ == '__main__':
    main()
