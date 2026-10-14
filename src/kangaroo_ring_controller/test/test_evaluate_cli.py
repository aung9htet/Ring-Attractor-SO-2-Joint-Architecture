"""Behaviour tests for scripts/evaluate.py without the robot (--dry-run) (DECISIONS.md D42):
    python3 -m unittest discover -s src/kangaroo_ring_controller/test -v -p test_evaluate_cli.py
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CLI = Path(__file__).resolve().parents[1] / 'scripts/evaluate.py'


def run(*args):
    return subprocess.run([sys.executable, str(CLI), *map(str, args)], capture_output=True, text=True)


class EvaluateDryRun(unittest.TestCase):
    def test_a_dry_run_lists_forty_reaches_per_joint(self):
        r = run('left', '--dry-run')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.splitlines()[-1], '160 reaches')

    def test_resume_refuses_a_run_with_another_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            protocol = [json.loads(line) for line in run('left', '--dry-run').stdout.splitlines()[:-1]]
            protocol[5]['goal'] += 0.01  # e.g. a newer calibration moved a start pose
            (Path(d) / 'manifest.json').write_text(json.dumps(dict(protocol=protocol, failed=[])))
            r = run('left', '--resume', d, '--dry-run')
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('reach 5 differs', r.stderr)
