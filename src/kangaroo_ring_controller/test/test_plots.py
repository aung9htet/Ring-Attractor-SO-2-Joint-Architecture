"""Behaviour tests for kangaroo_ring_controller/plots.py (DECISIONS.md D40). NEST-, ROS- and simulation-free:
    python3 -m unittest discover -s src/kangaroo_ring_controller/test -v -p test_plots.py
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller import plots  # noqa: E402

DT = 0.05
H = [0, 5, 10, 20, 20, 10, 5, 0, 0, 0, 0, 0]  # one drive shape, one sign throughout
# From 0 toward 1.0, ends at 0.8; commanded = (4+4+4)·0.05 = 0.6 rad.
TRIAL = dict(q_start=0.0, q_goal=1.0, q_final=0.8, dt=DT, joint_position=[0.2, 0.5, 0.8],
             signed_drive=[10.0, 10.0, 10.0], decoded_velocity=[4.0, 4.0, 4.0], side='left', joint=3)


class Helpers(unittest.TestCase):
    def test_relative_error_is_the_miss_over_the_wanted_distance(self):
        self.assertAlmostEqual(plots.relative_error(TRIAL), 0.2)

    def test_ratios_split_plant_from_decoder(self):
        moved_over_commanded, commanded_over_wanted = plots.ratios(TRIAL)
        self.assertAlmostEqual(moved_over_commanded, 0.8 / 0.6)
        self.assertAlmostEqual(commanded_over_wanted, 0.6)

    def test_degenerate_reaches_give_nan_not_an_error(self):
        self.assertTrue(math.isnan(plots.relative_error(dict(TRIAL, q_goal=0.0))))
        self.assertTrue(math.isnan(plots.ratios(dict(TRIAL, decoded_velocity=[0.0, 0.0, 0.0]))[0]))

    def test_predicted_dq_is_linear_and_uses_each_signs_gain(self):
        # Oracle: S(s·h) = s·S(h) for a one-signed h, and S_neg(-h) = S_pos(h); so with k_pos = 1, k_neg = -3,
        # doubling h doubles the prediction and -h predicts -3 times what h does.
        p = lambda spikes: plots.predicted_dq(spikes, DT, 1.0, -3.0, 0.3, 2)  # noqa: E731
        self.assertGreater(p(H), 0)
        self.assertAlmostEqual(p([2 * x for x in H]), 2 * p(H))
        self.assertAlmostEqual(p([-x for x in H]), -3 * p(H))


CAL = dict(side='left', joint=3, decoder_gain_positive=4e-3, decoder_gain_negative=-3e-3, decoder_tau=0.3,
           decoder_delay_steps=0,
           trials=[dict(spike_history=[s * x for x in H], dt=DT, dq_desired=0.1 * s, dq_actual=0.09 * s,
                        decoder_used=dict(gain_positive=1e-3 * s, gain_negative=-1e-3, tau=0.3, delay_steps=0))
                   for s in (1, -1, 2, -2)])


def titles(fig):
    return [ax.get_title() for ax in fig.axes]


class PerRunFigures(unittest.TestCase):
    def test_trial_figure_has_position_drive_velocity_and_error_panels(self):
        self.assertEqual(titles(plots.trial_figure(TRIAL, 'left j3')),
                         ['left j3: joint position [rad] (green: goal)', 'signed drive', 'decoded velocity [rad/s]',
                          'position error q - goal [rad]'])

    def test_calibration_progress_has_error_gains_and_tau_panels(self):
        self.assertEqual(titles(plots.calibration_progress_figure(CAL, 'left j3')),
                         ['left j3: |dq_actual - dq_desired| per trial [rad]', 'decoder gains used [rad/s per drive]',
                          'tau used [s]'])

    def test_decoder_fit_compares_prediction_with_desired_and_actual(self):
        refit = dict(CAL, decoder_gain_negative=-1e-3, trials=None)
        fig = plots.decoder_fit_figure(CAL, 'left j3', refit)
        self.assertEqual(titles(fig), ['left j3: predicted vs desired dq [rad]', 'predicted vs actual dq [rad]'])
        self.assertEqual(len(fig.axes[0].collections), 2)  # final gains and the refit

    def test_decoder_fit_needs_the_calibration_trials(self):
        with self.assertRaisesRegex(ValueError, 'no trials'):
            plots.decoder_fit_figure(dict(CAL, trials=None), 'refit')


DOWN = dict(TRIAL, q_goal=-1.0, q_final=-0.5, decoded_velocity=[-4.0, -4.0, -4.0])  # relative error 0.5


class SummaryFigures(unittest.TestCase):
    def test_evaluation_error_has_one_column_per_joint_direction_and_a_median_per_group(self):
        fig = plots.evaluation_error_figure({'A': [TRIAL, DOWN], 'B': [TRIAL]})
        ax, = fig.axes
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ['left j3 down', 'left j3 up'])
        medians = {line.get_label(): line.get_ydata()[0] for line in ax.lines if line.get_label() in ('A', 'B')}
        self.assertAlmostEqual(medians['A'], (0.2 + 0.5) / 2)  # median of 0.2 and 0.5
        self.assertAlmostEqual(medians['B'], 0.2)

    def test_a_reach_with_no_distance_to_go_does_not_spoil_the_median(self):
        # Review finding: statistics.median over a list with NaN depends on its order.
        fig = plots.evaluation_error_figure({'A': [TRIAL, dict(TRIAL, q_goal=0.0), DOWN]})
        median, = [line.get_ydata()[0] for line in fig.axes[0].lines if line.get_label() == 'A']
        self.assertAlmostEqual(median, (0.2 + 0.5) / 2)

    def test_plant_vs_decoder_plots_moved_vs_commanded_and_commanded_vs_wanted(self):
        fig = plots.plant_vs_decoder_figure({'A': [TRIAL]})
        self.assertEqual(titles(fig), ['plant: moved vs commanded [rad]', 'decoder: commanded vs wanted [rad]'])
        (commanded, moved), = fig.axes[0].collections[0].get_offsets()
        self.assertAlmostEqual(commanded, 0.6)
        self.assertAlmostEqual(moved, 0.8)



class EvaluationFigures(unittest.TestCase):
    def test_fixed_boxes_are_one_per_joint_direction(self):
        fixed = [dict(TRIAL, set='fixed'), dict(DOWN, set='fixed'), dict(TRIAL, set='sampled')]
        ax, = plots.fixed_boxplot_figure(fixed).axes
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ['left j3 down', 'left j3 up'])
        self.assertEqual(ax.get_title(), 'fixed goals: relative error over seeds')

    def test_sampled_boxes_are_one_per_joint(self):
        ax, = plots.sampled_error_figure([dict(TRIAL, set='sampled'), dict(DOWN, set='sampled')]).axes
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ['left j3'])
        self.assertEqual(ax.get_title(), 'sampled goals: absolute final error [rad]')

if __name__ == '__main__':
    unittest.main()
