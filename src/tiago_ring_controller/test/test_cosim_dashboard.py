"""Browser dashboard: HTTP endpoints, event stream and on-demand trials."""

import json
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiago_ring_controller.config import load_joint_calibration  # noqa: E402
from tiago_ring_controller.control.profiles import COLLECTOR_PROFILE  # noqa: E402
from tiago_ring_controller.cosim import CosimConfig, LoopObserver  # noqa: E402
from tiago_ring_controller.cosim.dashboard import (  # noqa: E402
    DashboardObserver,
    DashboardServer,
    DashboardState,
    run_dashboard_session,
)
from tiago_ring_controller.cosim.runner import build_loop, make_fake_engines  # noqa: E402


CALIBRATION_PATH = SRC / "config/calibration/velocity_calibration.json"


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _post(url, payload=None):
    data = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


class SlowTicks(LoopObserver):
    """Fakes run a trial in milliseconds; give HTTP requests time to land."""

    def on_tick(self, record):
        time.sleep(0.005)


def _wait_for(predicate, timeout_s=30.0, poll_s=0.05):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(poll_s)
    raise AssertionError("condition not met within %.1f s" % timeout_s)


class DashboardSessionTests(unittest.TestCase):
    def setUp(self):
        calibration = load_joint_calibration(str(CALIBRATION_PATH), 5)
        self.config = CosimConfig.from_profile(COLLECTOR_PROFILE, 5, calibration, max_steps=400)
        self.state = DashboardState()
        # Constant right-gain counts: the drive never settles, so trials run
        # to their step budget or to a user stop.
        engines = make_fake_engines(self.config, script=[(0, 40)] * 2000)
        observer = DashboardObserver(self.state, self.config.joint_index, 100)
        self.loop = build_loop(self.config, engines, observers=[observer, SlowTicks()])
        self.server = DashboardServer(self.state, "127.0.0.1", 0).start()
        self.results = []
        self.trials = []
        self.thread = threading.Thread(
            target=lambda: self.results.extend(
                run_dashboard_session(self.loop, self.config, self.state,
                                      on_trial=lambda i, r, l: self.trials.append((i, r, l)), poll_s=0.02)
            ),
            daemon=True,
        )
        self.thread.start()
        self.url = self.server.url.rstrip("/")

    def tearDown(self):
        _post(self.url + "/api/quit")
        self.thread.join(timeout=10)
        self.server.stop()

    def test_page_state_start_stop_and_events(self):
        with urllib.request.urlopen(self.url + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
        self.assertIn("<html", html)
        self.assertIn("EventSource('/events')", html)

        _wait_for(lambda: _get(self.url + "/state")["status"]["status"] == "idle")
        code, body = _post(self.url + "/api/start")
        self.assertEqual(code, 400)
        self.assertIn("error", body)

        # A full trial with the fakes settles on its own.
        code, body = _post(self.url + "/api/start", {"goal": 0.5, "max_steps": 60})
        self.assertEqual(code, 200)
        status = _wait_for(lambda: (lambda s: s if s["status"] == "idle" and len(s["trials"]) == 1 else None)(
            _get(self.url + "/state")["status"]))
        summary = status["trials"][0]
        self.assertEqual(summary["trial"], 1)
        self.assertEqual(summary["goal"], 0.5)
        self.assertEqual(summary["stop_reason"], "max_steps")
        self.assertEqual(summary["n_steps"], 60)
        self.assertIn("reset_wall_s", summary["timing"])
        snapshot = _get(self.url + "/state")["snapshot"]
        self.assertEqual(snapshot["type"], "tick")
        self.assertEqual(snapshot["trial"], 1)
        self.assertIsNotNone(snapshot["goal_index"])
        self.assertEqual(len(self.trials), 1)

        # Next goal via the API is used when start carries no goal.
        _post(self.url + "/api/goal", {"goal": -0.3})
        _wait_for(lambda: _get(self.url + "/state")["status"]["next_goal"] == -0.3)

        # A long trial can be stopped from the page.
        subscriber = self.state.subscribe()
        code, body = _post(self.url + "/api/start", {"max_steps": 400})
        self.assertEqual(code, 200)
        self.assertEqual(body["queued"]["goal"], -0.3)
        _wait_for(lambda: _get(self.url + "/state")["status"]["status"] == "running")
        _wait_for(lambda: (_get(self.url + "/state")["snapshot"] or {}).get("trial") == 2)
        _post(self.url + "/api/stop")
        status = _wait_for(lambda: (lambda s: s if len(s["trials"]) == 2 else None)(_get(self.url + "/state")["status"]))
        self.assertEqual(status["trials"][1]["stop_reason"], "user_stop")
        self.assertEqual(self.trials[1][1].stop_reason, "user_stop")

        # The pub/sub stream carried ticks and the final status.
        messages = []
        while not subscriber.empty():
            messages.append(json.loads(subscriber.get_nowait()))
        self.state.unsubscribe(subscriber)
        types = [m["type"] for m in messages]
        self.assertIn("trial_start", types)
        self.assertIn("tick", types)
        self.assertEqual(types[-1], "status")
        ticks = [m for m in messages if m["type"] == "tick" and m["new_sample"]]
        self.assertTrue(ticks)
        self.assertEqual(len(ticks[0]["r1"]), 100)

    def test_reset_now_and_continue_between_trials(self):
        _wait_for(lambda: _get(self.url + "/state")["status"]["status"] == "idle")
        nest, robot = self.loop.engines
        code, body = _post(self.url + "/api/reset")
        self.assertEqual(code, 200)
        _wait_for(lambda: _get(self.url + "/state")["status"].get("resets") == 1)
        self.assertEqual((nest.rebuild_count, robot.rebuild_count), (1, 1))
        self.assertIn("reset done", _get(self.url + "/state")["status"]["message"])

        code, body = _post(self.url + "/api/start", {"goal": 0.4, "max_steps": 6, "reset_mode": "continue"})
        self.assertEqual(code, 200)
        status = _wait_for(lambda: (lambda s: s if len(s["trials"]) == 1 else None)(_get(self.url + "/state")["status"]))
        self.assertEqual(status["trials"][0]["reset_mode"], "continue")
        self.assertEqual((nest.rebuild_count, robot.rebuild_count), (1, 1))
        end = self.trials[0][1].final_state["joint_state"]["positions"][5]

        _post(self.url + "/api/start", {"goal": -0.2, "max_steps": 6, "reset_mode": "continue"})
        _wait_for(lambda: len(_get(self.url + "/state")["status"]["trials"]) == 2)
        self.assertEqual(self.trials[1][1].main_ticks[0].inputs["joint_state"]["positions"][5], end)
        _post(self.url + "/api/start", {"goal": 0.1, "max_steps": 6, "reset_mode": "rebuild"})
        _wait_for(lambda: len(_get(self.url + "/state")["status"]["trials"]) == 3)
        self.assertEqual((nest.rebuild_count, robot.rebuild_count), (2, 2))
        code, body = _post(self.url + "/api/start", {"goal": 0.1, "reset_mode": "sometimes"})
        self.assertEqual(code, 400)

    def test_event_stream_serves_status_and_ticks(self):
        _wait_for(lambda: _get(self.url + "/state")["status"]["status"] == "idle")
        # Run one trial first: a late subscriber gets the whole trial replayed.
        _post(self.url + "/api/start", {"goal": 0.3, "max_steps": 8})
        _wait_for(lambda: len(_get(self.url + "/state")["status"]["trials"]) == 1)
        request = urllib.request.Request(self.url + "/events")
        with urllib.request.urlopen(request, timeout=10) as response:
            first = response.readline().decode("utf-8")
            self.assertTrue(first.startswith("data: "))
            self.assertEqual(json.loads(first[6:])["type"], "status")
            replayed = []
            while True:
                line = response.readline().decode("utf-8")
                if line.startswith("data: "):
                    replayed.append(json.loads(line[6:])["type"])
                    if replayed[-1] == "trial_end":
                        break
            self.assertEqual(replayed[0], "trial_start")
            self.assertEqual(replayed.count("tick"), 4 + 8)
            _post(self.url + "/api/start", {"goal": 0.2, "max_steps": 5})
            seen = []
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and "tick" not in seen:
                line = response.readline().decode("utf-8")
                if line.startswith("data: "):
                    seen.append(json.loads(line[6:])["type"])
            self.assertIn("tick", seen)


if __name__ == "__main__":
    unittest.main()
