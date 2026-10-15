"""Behaviour tests for kangaroo_ring_controller/gui.py and ring.run_trial(on_tick=...)
(docs/superpowers/specs/2026-10-13-kangaroo-ring-gui-design.md). Simulation running:
    python3 -m unittest discover -s src/kangaroo_ring_controller/test -v -p test_gui.py
"""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller import RESULTS  # noqa: E402
from kangaroo_ring_controller.arm import SIDES, connect  # noqa: E402
from kangaroo_ring_controller.joint_setup import newest_setup, uncalibrated  # noqa: E402
from kangaroo_ring_controller.gui import Gui, make_server, process  # noqa: E402
from kangaroo_ring_controller.ring import LIMITS, run_trial  # noqa: E402

HOME_LEFT_J0 = 0.239  # PAL 4-DoF home, left j0 (KB 4c)
ENV = {}


def setUpModule():
    ENV['node'], ENV['arms'] = connect('test_ring_gui')


def tearDownModule():
    rclpy.shutdown()


class OnTick(unittest.TestCase):
    def test_on_tick_sees_every_tick_and_can_stop_the_trial(self):
        arm = ENV['arms']['left']
        arm.reset()
        ticks = []

        def on_tick(tick):
            ticks.append(tick)
            return len(ticks) == 5

        r = run_trial(arm, uncalibrated('left', 0), HOME_LEFT_J0 + 1.0, seed=1, on_tick=on_tick)
        self.assertEqual(r['stop_reason'], 'stopped')
        self.assertEqual([t['step'] for t in ticks], [0, 1, 2, 3, 4])
        self.assertEqual([t['joint_position'] for t in ticks], r['joint_position'])
        self.assertTrue(all(t['goal_index'] == r['goal_ring_index'] for t in ticks))
        self.assertTrue(all(0 <= t['state_index'] < t['neurons'] for t in ticks))
        json.dumps(ticks)  # the page receives ticks as JSON


def write_cal(root, name, start_pose=None, **drop):
    """Minimal calibration.json as calibrate.py writes it; keys in `drop` are left out (malformed file)."""
    d = Path(root) / name
    d.mkdir(parents=True)
    cal = dict(side=name.split('_')[0], joint=int(name.split('_')[1][1]), profile='calibration',
               decoder_gain_positive=0.004, decoder_gain_negative=-0.004, decoder_tau=0.3, decoder_delay_steps=0,
               joint_limits=[-1.0, 1.0], start_pose=start_pose)
    for k in drop:
        cal.pop(k)
    (d / 'calibration.json').write_text(json.dumps(cal))
    return str(d / 'calibration.json')


def call(port, path, body=None):
    """GET when body is None, else POST (dict/list as JSON, bytes raw); returns (status, decoded JSON)."""
    data = body if body is None or isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class GuiServer(unittest.TestCase):
    def serve(self, cal_root):
        out = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        gui = Gui(Path(cal_root), Path(out.name))
        server = make_server(gui, 0)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return gui, server.server_address[1]

    def test_bad_requests_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_cal(root, 'left_j3_seed1_20261013_100000', decoder_tau=None)  # malformed: no decoder_tau
            gui, port = self.serve(root)
            for body in ({'side': 'left', 'joint': 0, 'goal': 9.0}, {'side': 'left', 'joint': 0, 'goal': 'nan'},
                         {'side': 'left', 'joint': 0, 'goal': 'abc'}, {'side': 'middle', 'joint': 0, 'goal': 0.0},
                         {'side': 'left', 'joint': 7, 'goal': 0.0}, {'side': 'left', 'joint': 3, 'goal': 0.0},
                         b'not json', [1, 2]):
                code, msg = call(port, '/api/start', body)
                self.assertEqual(code, 400, (body, msg))
            self.assertEqual(call(port, '/api/start', {'side': 'left', 'joint': 0, 'goal': 0.5})[0], 202)
            self.assertEqual(call(port, '/api/start', {'side': 'left', 'joint': 0, 'goal': 0.6})[0], 202)
            self.assertEqual(gui.pending['goal'], 0.6)  # the latest goal replaces the queued one

    def test_start_moves_the_joint_toward_the_goal(self):
        ENV['arms']['left'].reset()
        gui, port = self.serve(RESULTS / 'calibration')
        body = {'side': 'left', 'joint': 0, 'goal': HOME_LEFT_J0 + 0.5, 'seed': 3}
        self.assertEqual(call(port, '/api/start', body)[0], 202)
        trial = process(ENV['arms'], gui, timeout_s=5.0)
        self.assertIsNotNone(trial)
        self.assertGreater(trial['q_final'] - trial['q_start'], 0.05)
        state = call(port, '/state')[1]
        self.assertEqual(state['status'], 'done')
        self.assertIn('left_j0', state['calibration'])
        self.assertTrue((Path(state['path']) / 'trial.json').is_file())

    def test_stop_and_closed_browser(self):
        gui, port = self.serve(RESULTS / 'calibration')
        stream = urllib.request.urlopen(f'http://127.0.0.1:{port}/events', timeout=5)
        self.assertEqual(stream.readline(), b'event: status\n')
        stream.close()  # browser tab closed
        self.assertEqual(call(port, '/api/start', {'side': 'left', 'joint': 0, 'goal': HOME_LEFT_J0 + 0.5})[0], 202)
        self.assertEqual(call(port, '/api/stop', {})[0], 200)
        trial = process(ENV['arms'], gui, timeout_s=5.0)
        self.assertEqual(trial['stop_reason'], 'stopped')
        self.assertEqual(len(trial['joint_position']), 1)
        self.assertEqual(call(port, '/state')[1]['status'], 'done')
        for _ in range(50):  # the server thread notices the closed stream on its next write
            if not gui.subscribers:
                break
            time.sleep(0.1)
        self.assertEqual(gui.subscribers, [])

    def test_save_failure_reports_error_and_keeps_serving(self):
        gui, port = self.serve(RESULTS / 'calibration')
        blocker = tempfile.NamedTemporaryFile()  # a file where the results dir should be: saving must fail
        self.addCleanup(blocker.close)
        gui.out_root = Path(blocker.name)
        self.assertEqual(call(port, '/api/start', {'side': 'left', 'joint': 0, 'goal': HOME_LEFT_J0 + 0.5})[0], 202)
        call(port, '/api/stop', {})
        self.assertIsNone(process(ENV['arms'], gui, timeout_s=5.0))
        self.assertEqual(call(port, '/state')[1]['status'], 'error')
        self.assertEqual(call(port, '/api/start', {'side': 'left', 'joint': 0, 'goal': HOME_LEFT_J0})[0], 202)

    def test_queued_is_broadcast_before_the_trial_can_be_taken(self):
        with tempfile.TemporaryDirectory() as root:
            gui, seen = Gui(Path(root), Path(root)), []

            class Probe:  # a browser's event queue: records the status and whether a command was already takeable
                def put(self, msg):
                    seen.append((json.loads(msg.split('data: ')[1])['status'], gui.pending is not None))

            gui.subscribers.append(Probe())
            self.assertEqual(gui.request_start({'side': 'left', 'joint': 0, 'goal': 0.5})[0], 202)
            self.assertEqual(seen, [('queued', False)])

    def test_joints_lists_all_eight_with_range_and_position(self):
        gui, port = self.serve(RESULTS / 'calibration')
        process(ENV['arms'], gui, timeout_s=0.5)  # an idle pass shares the joint positions
        joints = call(port, '/api/joints')[1]
        self.assertEqual([(j['side'], j['joint']) for j in joints], [(s, i) for s in SIDES for i in range(4)])
        for j in joints:
            self.assertLess(j['limits'][0], j['limits'][1])
            self.assertIsInstance(j['position'], float)

    def test_joints_survives_a_calibration_for_another_profile(self):
        with tempfile.TemporaryDirectory() as root:
            bad = write_cal(root, 'left_j2_seed1_20261013_100000')
            cal = json.loads(Path(bad).read_text())
            Path(bad).write_text(json.dumps(dict(cal, profile='collector')))  # not the calibration profile
            gui, port = self.serve(root)
            code, joints = call(port, '/api/joints')
            self.assertEqual(code, 200)
            j = next(j for j in joints if (j['side'], j['joint']) == ('left', 2))
            self.assertEqual(j['limits'], list(LIMITS[2]))  # falls back to the uncalibrated ring range
            self.assertIn('error', j)

    def test_reset_returns_the_arm_to_the_start_pose(self):
        arm = ENV['arms']['left']
        arm.reset([0.8, 1.321, 1.57, 0.8])
        gui, port = self.serve(RESULTS / 'calibration')
        self.assertEqual(call(port, '/api/reset', {'side': 'left', 'joint': 0})[0], 202)
        self.assertIsNone(process(ENV['arms'], gui, timeout_s=5.0))
        for q, s in zip(arm.current_positions, newest_setup('left', 0).start):
            self.assertAlmostEqual(q, s, delta=0.02)

    def test_a_new_goal_starts_from_where_the_joint_is(self):
        arm = ENV['arms']['left']
        arm.reset([0.8, 1.321, 1.57, 0.8])
        gui, port = self.serve(RESULTS / 'calibration')
        self.assertEqual(call(port, '/api/start', {'side': 'left', 'joint': 0, 'goal': 0.4})[0], 202)
        trial = process(ENV['arms'], gui, timeout_s=5.0)
        self.assertAlmostEqual(trial['q_start'], 0.8, delta=0.02)  # not the start pose (home, 0.239)

    def test_a_newer_goal_replaces_the_running_trial(self):
        ENV['arms']['left'].reset()
        gui, port = self.serve(RESULTS / 'calibration')
        newer = {'side': 'left', 'joint': 0, 'goal': HOME_LEFT_J0 - 0.3}

        class Retarget:  # the slider moved again as soon as the first reach shows a tick
            fired = False

            def put(self, msg):
                if msg.startswith('event: tick') and not self.fired:
                    self.fired = True
                    threading.Thread(target=gui.request_start, args=(newer,)).start()

        gui.subscribers.append(Retarget())
        call(port, '/api/start', {'side': 'left', 'joint': 0, 'goal': HOME_LEFT_J0 + 0.8})
        self.assertEqual(process(ENV['arms'], gui, timeout_s=5.0)['stop_reason'], 'stopped')
        self.assertEqual(process(ENV['arms'], gui, timeout_s=5.0)['q_goal'], newer['goal'])

    def test_a_prepared_ring_starts_the_reach_at_once(self):
        first_tick = []
        gui, port = self.serve(RESULTS / 'calibration')
        self.assertEqual(call(port, '/api/select', {'side': 'left', 'joint': 0, 'seed': 5})[0], 200)
        self.assertIsNone(process(ENV['arms'], gui, timeout_s=60.0))  # idle: builds the spare ring
        self.assertEqual(call(port, '/state')[1]['status'], 'ready')

        class FirstTick:
            def put(self, msg):
                if msg.startswith('event: tick') and not first_tick:
                    first_tick.append(time.monotonic())

        gui.subscribers.append(FirstTick())
        t0 = time.monotonic()
        call(port, '/api/start', {'side': 'left', 'joint': 0, 'goal': HOME_LEFT_J0 + 0.3, 'seed': 5})
        process(ENV['arms'], gui, timeout_s=5.0)
        self.assertLess(first_tick[0] - t0, 5.0)  # building a fresh ring takes 22-30 s


if __name__ == '__main__':
    unittest.main()
