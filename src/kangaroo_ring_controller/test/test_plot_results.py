"""Behaviour tests for scripts/plot_results.py (DECISIONS.md D40). Simulation-free; runs the CLI as a subprocess:
    python3 -m unittest discover -s src/kangaroo_ring_controller/test -v -p test_plot_results.py
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CLI = Path(__file__).resolve().parents[1] / 'scripts/plot_results.py'
TRIAL = dict(q_start=0.0, q_goal=1.0, q_final=0.8, dt=0.05, joint_position=[0.2, 0.5, 0.8], seed=1,
             signed_drive=[10.0, 10.0, 10.0], decoded_velocity=[4.0, 4.0, 4.0], side='left', joint=3)
H = [0, 5, 10, 20, 20, 10, 5, 0, 0, 0, 0, 0]
CAL = dict(side='left', joint=3, seed=2026, decoder_gain_positive=4e-3, decoder_gain_negative=-3e-3, decoder_tau=0.3,
           decoder_delay_steps=0,
           trials=[dict(spike_history=[s * x for x in H], dt=0.05, dq_desired=0.1 * s, dq_actual=0.09 * s,
                        decoder_used=dict(gain_positive=4e-3, gain_negative=-3e-3, tau=0.3, delay_steps=0))
                   for s in (1, -1, 2, -2)])


def write(root, rel, record):
    path = Path(root) / rel
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record))
    return path


def run(*args):
    return subprocess.run([sys.executable, str(CLI), *map(str, args)], capture_output=True, text=True)


class PlotResults(unittest.TestCase):
    def test_writes_every_figure_and_a_manifest_into_a_new_run_dir(self):
        with tempfile.TemporaryDirectory() as root:
            trial = write(root, 'in/reach_a/trial.json', TRIAL).parent
            cal = write(root, 'in/cal_a/calibration.json', CAL)
            refit = write(root, 'in/cal_b/calibration.json', dict(CAL, trials=None, refit_of=str(cal)))
            r = run('--trials', trial, '--calibrations', cal.parent, '--refit', refit,
                    '--group', f'A={Path(root)}/in/reach_*', '--root', root)
            self.assertEqual(r.returncode, 0, r.stderr)
            out, = (Path(root) / 'figures').iterdir()
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ['calibration_progress_cal_a.png', 'decoder_fit_cal_a.png', 'evaluation_error.pdf',
                              'evaluation_error.png', 'manifest.json', 'plant_vs_decoder.pdf', 'plant_vs_decoder.png',
                              'trial_reach_a.png'])
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['groups'], {'A': [str(trial / 'trial.json')]})
            self.assertEqual(manifest['refits'], [str(refit)])

    def test_bad_inputs_stop_with_the_offending_path(self):
        with tempfile.TemporaryDirectory() as root:
            cal = write(root, 'in/cal_a/calibration.json', CAL)
            refit = write(root, 'in/cal_b/calibration.json', dict(CAL, trials=None, refit_of='/elsewhere.json'))
            cases = [(['--group', f'A={root}/nothing_*'], f'{root}/nothing_*'),       # empty glob
                     (['--calibrations', refit], 'no trials'),                        # a refit as a calibration
                     (['--calibrations', cal, '--refit', refit], '/elsewhere.json')]  # refit source not given
            for args, message in cases:
                r = run(*args, '--root', root)
                self.assertNotEqual(r.returncode, 0, args)
                self.assertIn(message, r.stderr, args)

    def test_a_refit_made_from_the_repo_root_is_matched_from_any_directory(self):
        # Review finding: the D39 refits name their source relative to the repo root (where they were made).
        results = CLI.parents[1] / 'results/calibration'
        source = results / 'left_j3_seed2026_20261014_133218/calibration.json'
        refit = results / 'left_j3_seed2026_20261014_160729/calibration.json'
        with tempfile.TemporaryDirectory() as root:
            r = subprocess.run([sys.executable, str(CLI), '--calibrations', str(source), '--refit', str(refit),
                                '--root', root], capture_output=True, text=True, cwd=root)
            self.assertEqual(r.returncode, 0, r.stderr)
            r = run('--calibrations', source, '--refit', source, '--root', root)  # not a refit
            self.assertNotEqual(r.returncode, 0)
            self.assertIn(f'{source}: not a refit', r.stderr)


    def test_an_evaluation_gets_its_figures_and_a_summary_with_the_verdict(self):
        with tempfile.TemporaryDirectory() as root:
            for i, (set_, goal) in enumerate((('fixed', 1.0), ('fixed', -1.0), ('sampled', 0.5))):
                write(root, f'eval/trials/r{i}/trial.json', dict(TRIAL, set=set_, q_goal=goal, q_final=0.8 * goal))
            r = run('--evaluation', Path(root) / 'eval', '--root', root)
            self.assertEqual(r.returncode, 0, r.stderr)
            out, = (Path(root) / 'figures').iterdir()
            self.assertTrue({'fixed_relative_error.png', 'sampled_absolute_error.pdf', 'summary.json'}
                            <= {p.name for p in out.iterdir()})
            s = json.loads((out / 'summary.json').read_text())
            self.assertAlmostEqual(s['summary']['fixed']['left j3 up']['median'], 0.2)
            self.assertFalse(s['verdict']['passed'])  # left j3 alone: incomplete

if __name__ == '__main__':
    unittest.main()
