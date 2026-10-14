"""Behaviour tests for kangaroo_ring_controller/evaluation.py (DECISIONS.md D42). NEST-, ROS- and simulation-free:
    python3 -m unittest discover -s src/kangaroo_ring_controller/test -v -p test_evaluation.py
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller import evaluation as ev  # noqa: E402
from kangaroo_ring_controller.joint_setup import JointSetup  # noqa: E402

# left j1 starting at 0.8 rad inside the ring range (-1.0, 2.0): fixed goals 0.8 + 0.5·1.2 = 1.4, 0.8 - 0.5·1.8 = -0.1
SETUP = JointSetup('left', 1, None, None, (-1.0, 2.0), (0.239, 0.8, 1.57, 0.8))


class Protocol(unittest.TestCase):
    def test_each_joint_gets_ten_seeds_per_fixed_goal_and_twenty_sampled_goals(self):
        reaches = ev.protocol([SETUP])
        fixed = [r for r in reaches if r['set'] == 'fixed']
        sampled = [r for r in reaches if r['set'] == 'sampled']
        self.assertEqual((len(fixed), len(sampled)), (20, 20))
        self.assertEqual({(r['goal_index'], round(r['goal'], 9)) for r in fixed}, {(0, 1.4), (1, -0.1)})
        self.assertEqual(sorted(r['seed'] for r in fixed if r['goal_index'] == 0), list(range(100, 110)))
        self.assertEqual([r['seed'] for r in sampled], list(range(300, 320)))
        for r in sampled:
            self.assertTrue(-1.0 <= r['goal'] <= 2.0 and abs(r['goal'] - 0.8) >= 0.1, r)
        self.assertEqual(ev.protocol([SETUP]), reaches)  # seeded: the same goals every time

    def test_a_range_too_narrow_to_leave_the_start_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'too narrow'):
            ev.protocol([JointSetup('left', 1, None, None, (0.75, 0.85), (0.239, 0.8, 1.57, 0.8))])

    def test_resume_skips_exactly_the_saved_reaches(self):
        reaches = ev.protocol([SETUP])
        done = [dict(reaches[0], q_final=1.0), dict(reaches[25], q_final=0.0)]  # saved trials carry more keys
        left = ev.remaining(reaches, done)
        self.assertEqual(len(left), 38)
        self.assertNotIn(reaches[0], left)
        self.assertNotIn(reaches[25], left)


def trial(error, side='left', joint=1, up=True, set_='fixed', right_way=True):
    """A reach from 0 toward ±1 rad that ends `error` short of the goal (relative error = error); commanded = moved;
    right_way=False ends `error` on the wrong side of the start."""
    sign = 1 if up else -1
    q_final = sign * (1 - error) if right_way else -sign * error
    return dict(set=set_, side=side, joint=joint, q_start=0.0, q_goal=float(sign), q_final=q_final, dt=0.1,
                decoded_velocity=[q_final / 0.1])


class Statistics(unittest.TestCase):
    def test_fixed_groups_give_the_median_and_iqr_over_seeds(self):
        no_distance = dict(trial(0.0), q_goal=0.0)  # goal = start: no direction, relative error and ratios NaN
        s = ev.summary([trial(e) for e in (0.1, 0.2, 0.3, 0.4)] + [no_distance])
        g = s['fixed']['left j1 up']
        self.assertEqual(g['n'], 4)
        self.assertAlmostEqual(g['median'], 0.25)
        self.assertAlmostEqual(g['iqr'], 0.325 - 0.175)  # inclusive quartiles of 0.1-0.4
        self.assertAlmostEqual(g['right_way'], 1.0)
        self.assertAlmostEqual(g['moved_over_commanded'], 1.0)
        self.assertTrue(math.isnan(s['fixed']['left j1 down']['median']))  # no ZeroDivisionError, NaN left as is

    def test_sampled_groups_give_the_absolute_error_per_joint(self):
        s = ev.summary([trial(e, set_='sampled') for e in (0.1, 0.3)])
        self.assertAlmostEqual(s['sampled']['left j1']['abs_error'][1], 0.2)


def run_with(fixed_median, sampled_median, sampled_right_way=1.0, groups=12):
    """Trials for the six judged joints: 3 fixed reaches per joint and direction with relative error
    `fixed_median` (a list gives one value per group, in sorted-joint order, up before down); `groups` = 6 keeps only
    "up"; 10 sampled reaches per joint with relative error `sampled_median`, a share 1 - sampled_right_way wrong way."""
    trials = []
    for i, (side, joint) in enumerate(sorted(ev.JUDGED)):
        for up in (True, False)[:groups // 6]:
            m = fixed_median[i * 2 + (not up)] if isinstance(fixed_median, list) else fixed_median
            trials += [trial(m, side, joint, up) for _ in range(3)]
        n_wrong = round(10 * (1 - sampled_right_way))
        trials += [trial(sampled_median, side, joint, set_='sampled', right_way=k >= n_wrong) for k in range(10)]
    return trials


class Verdict(unittest.TestCase):
    def test_both_bars_hold(self):
        self.assertTrue(ev.verdict(run_with(0.4, 0.3))['passed'])

    def test_three_bad_directions_fail_the_fixed_bar(self):
        v = ev.verdict(run_with([0.6, 0.6, 0.6] + [0.4] * 9, 0.3))
        self.assertEqual(v['fixed']['below'], 9)
        self.assertFalse(v['passed'])

    def test_two_bad_directions_still_pass(self):
        self.assertTrue(ev.verdict(run_with([0.6, 0.6] + [0.4] * 10, 0.3))['passed'])

    def test_the_sampled_bar_needs_the_median_and_the_right_way_fraction(self):
        self.assertFalse(ev.verdict(run_with(0.4, 0.36))['passed'])
        self.assertFalse(ev.verdict(run_with(0.4, 0.3, sampled_right_way=0.8))['passed'])

    def test_an_incomplete_run_does_not_pass(self):
        v = ev.verdict(run_with(0.4, 0.3, groups=6))  # only the "up" direction of each joint
        self.assertFalse(v['passed'])
        self.assertIn('incomplete', v['fixed']['note'])


class Run(unittest.TestCase):
    def test_a_failing_reach_is_recorded_and_the_rest_still_run(self):
        reaches = ev.protocol([SETUP])[:3]
        ran = []

        def do(r):
            if r is reaches[1]:
                raise RuntimeError('left arm did not reach the start')
            ran.append(r)

        failed = ev.run(reaches, do)
        self.assertEqual(ran, [reaches[0], reaches[2]])
        self.assertEqual(failed, [dict(reaches[1], error="RuntimeError('left arm did not reach the start')")])


if __name__ == '__main__':
    unittest.main()
