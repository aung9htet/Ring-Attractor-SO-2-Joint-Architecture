"""Graph editor (phase 6): server endpoints, canonical save, headless-Chrome load.

The browser part runs where ``google-chrome`` or ``chromium`` is installed (the
host); it is skipped in the container image, which has no browser.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiago_ring_controller.cosim.dashboard import DashboardServer, DashboardState, run_dashboard_session  # noqa: E402
from tiago_ring_controller.graph import EXAMPLES_DIR, compile_graph, dumps, two_ring_single_joint  # noqa: E402
from tiago_ring_controller.ui import STATIC_FILES, read_static  # noqa: E402


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def chrome_binary():
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return None


class EditorServerTests(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()
        self.server = DashboardServer(self.state, "127.0.0.1", 0).start()
        self.url = self.server.url.rstrip("/")

    def tearDown(self):
        self.server.stop()

    def test_static_files_and_palette(self):
        for name, content_type in STATIC_FILES.items():
            with self.subTest(file=name):
                with urllib.request.urlopen(self.url + "/ui/" + name, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers["Content-Type"], content_type)
                    self.assertEqual(response.read(), read_static(name))
        with urllib.request.urlopen(self.url + "/editor", timeout=5) as response:
            page = response.read().decode("utf-8")
        self.assertIn('<svg id="canvas"', page)
        self.assertIn("/ui/editor.js", page)
        with urllib.request.urlopen(self.url + "/", timeout=5) as response:
            self.assertIn('href="/editor"', response.read().decode("utf-8"))
        status, blocks = get(self.url + "/api/blocks")
        self.assertEqual(status, 200)
        self.assertIn("Ring", blocks["types"])
        ring = blocks["types"]["Ring"]
        self.assertEqual([p["name"] for p in ring["ports"]], ["stim", "spikes", "counts"])
        self.assertEqual(ring["params"][0]["name"], "population_size")
        self.assertTrue(ring["neural"])
        status, examples = get(self.url + "/api/graph/examples")
        self.assertIn("two_ring_single_joint", examples["examples"])
        status, example = get(self.url + "/api/graph/examples/two_ring_single_joint")
        self.assertEqual(example["graph"]["name"], "two ring single joint 5")
        self.assertEqual(example["problems"], [])
        self.assertEqual(get(self.url + "/api/graph/examples/nope")[0], 404)
        self.assertEqual(get(self.url + "/ui/nope.js")[0], 404)
        self.assertEqual(get(self.url + "/ui/../setup.py")[0], 404)

    def test_validate_reports_problems_and_returns_the_canonical_document(self):
        document = two_ring_single_joint().to_dict()
        status, result = post(self.url + "/api/graph/validate", {"graph": document})
        self.assertEqual((status, result["valid"], result["problems"]), (200, True, []))
        self.assertEqual(result["text"], dumps(two_ring_single_joint()))
        broken = json.loads(json.dumps(document))
        broken["edges"] = [e for e in broken["edges"] if e["to"] != "gain.left_in"]
        broken["blocks"][0]["params"]["population_size"] = 1
        status, result = post(self.url + "/api/graph/validate", {"graph": broken})
        self.assertFalse(result["valid"])
        self.assertTrue(any("population_size: 1 is below the minimum 3" in p for p in result["problems"]))
        status, result = post(self.url + "/api/graph/validate", {"graph": {"schema": "ring-blocks/1", "blocks": [{"id": "g", "type": "Gain"}], "edges": []}})
        self.assertFalse(result["valid"])
        self.assertIn("gain.ring: required input is not connected".replace("gain", "g"), result["problems"])
        self.assertEqual(post(self.url + "/api/graph/validate", {"nope": 1})[0], 400)

    def test_a_graph_drawn_in_the_editor_saves_byte_identically_to_the_python_graph(self):
        # The editor posts what it holds: blocks with the palette defaults plus the user's edits and
        # ui positions, edges as drawn.  Build such a document from the type descriptions only.
        status, blocks = get(self.url + "/api/blocks")
        types = blocks["types"]
        reference = two_ring_single_joint()
        drawn = {"schema": "ring-blocks/1", "name": reference.name, "simulation": dict(reference.simulation.to_dict()),
                 "blocks": [], "edges": [], "robot": dict(reference.robot)}
        for index, block in enumerate(reference.declared.values()):
            params = {spec["name"]: spec["default"] for spec in types[block.type_name]["params"]}
            params.update({k: v for k, v in block.params.items() if v != params.get(k)})     # the user's edits
            drawn["blocks"].append({"id": block.id, "type": block.type_name, "params": params, "ui": {"x": 40 + 30 * index, "y": 60}})
        for edge in reference.declared_edges:
            entry = {"from": edge.declared_source.key, "to": edge.declared_target.key}
            if edge.params:
                entry["params"] = dict(edge.params)
            drawn["edges"].append(entry)
        for index, block in enumerate(reference.declared.values()):
            block.ui = {"x": 40 + 30 * index, "y": 60}
        expected = dumps(reference)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "drawn.graph.json")
            status, result = post(self.url + "/api/graph/save", {"graph": drawn, "path": path})
            self.assertEqual((status, result["saved"], result["problems"]), (200, path, []))
            self.assertEqual(Path(path).read_text(encoding="utf-8"), expected)
            status, loaded = get(self.url + "/api/graph/file?path=" + urllib.request.quote(path))
            self.assertEqual(loaded["graph"], json.loads(expected))
            self.assertEqual(self.state.graph_dict()["path"], path)
        self.assertEqual(post(self.url + "/api/graph/save", {"graph": drawn})[0], 400)
        self.assertEqual(get(self.url + "/api/graph/file?path=/nope.graph.json")[0], 404)

    def test_run_queues_a_graph_command_that_the_session_swaps(self):
        graph = two_ring_single_joint()
        compiled = compile_graph(graph, engines="fake")
        compiled.loop.initialize()
        self.state.set_graph(graph.to_dict(), None)
        swapped = []

        def on_graph(document):
            new = compile_graph(two_ring_single_joint(joint=3), engines="fake")
            new.loop.initialize()
            swapped.append(document["name"])
            return new.loop, new.config

        thread = threading.Thread(target=run_dashboard_session, args=(compiled.loop, compiled.config, self.state), kwargs={"on_graph": on_graph, "poll_s": 0.05}, daemon=True)
        thread.start()
        try:
            document = two_ring_single_joint(joint=3).to_dict()
            status, result = post(self.url + "/api/graph", {"graph": document, "request_id": "r1"})
            self.assertTrue(result["queued"])
            for _ in range(100):
                if "r1" in self.state.graph_replies:
                    break
                time.sleep(0.05)
            self.assertEqual(self.state.graph_replies.get("r1"), {"ok": True})
            self.assertEqual(swapped, ["two ring single joint 3"])
            self.assertEqual(self.state.status["graph"], "two ring single joint 3")
            self.assertEqual(self.state.status["config"]["joint_index"], 3)
            self.assertEqual(get(self.url + "/api/graph")[1]["graph"]["name"], "two ring single joint 3")
        finally:
            post(self.url + "/api/quit", {})
            thread.join(timeout=5)


@unittest.skipIf(chrome_binary() is None, "no headless Chrome available")
class EditorBrowserTests(unittest.TestCase):
    """Opens the editor with the example graph loaded and checks the rendered DOM and a screenshot."""

    def setUp(self):
        self.state = DashboardState()
        self.server = DashboardServer(self.state, "127.0.0.1", 0).start()
        self.url = self.server.url.rstrip("/")
        graph = two_ring_single_joint()
        self.state.set_graph(graph.to_dict(), str(Path(EXAMPLES_DIR) / "two_ring_single_joint.graph.json"))

    def tearDown(self):
        self.server.stop()

    def test_editor_renders_the_single_joint_architecture(self):
        directory = tempfile.mkdtemp()
        try:
            screenshot = os.path.join(directory, "editor.png")
            command = [
                chrome_binary(), "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                "--window-size=1400,900", "--virtual-time-budget=5000", "--enable-logging=stderr", "--v=0",
                "--user-data-dir=" + os.path.join(directory, "profile"),
                "--screenshot=" + screenshot, "--dump-dom", self.url + "/editor?static=1",
            ]
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            dom = completed.stdout
            self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
            for block_id in ("r1", "r2", "f1", "f2", "cmp", "gain", "enc_state", "enc_goal", "dec", "j5", "goal"):
                with self.subTest(block=block_id):
                    self.assertIn('data-id="%s"' % block_id, dom)
            self.assertIn('data-editor-ready="1"', dom)
            self.assertIn("session graph loaded", dom)
            self.assertGreaterEqual(dom.count('class="edge spikes'), 9)
            self.assertGreaterEqual(dom.count('class="edge signal'), 5)
            self.assertNotIn("Uncaught", completed.stderr)
            self.assertNotIn('class="status error"', dom)
            self.assertTrue(os.path.isfile(screenshot))
            self.assertGreater(os.path.getsize(screenshot), 10000)
        finally:
            shutil.rmtree(directory, ignore_errors=True)   # Chrome may still be writing its profile

    def test_drawing_the_example_from_an_empty_canvas_gives_the_same_file(self):
        """Phase 6 gate, in the browser: palette add, rename, parameters, connect, then canonical text."""

        directory = tempfile.mkdtemp()
        try:
            command = [
                chrome_binary(), "--headless=new", "--disable-gpu", "--no-sandbox", "--window-size=1400,900",
                "--virtual-time-budget=8000", "--user-data-dir=" + os.path.join(directory, "profile"),
                "--dump-dom", self.url + "/editor?static=1&selftest=two_ring_single_joint",
            ]
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            dom = completed.stdout
            self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
            self.assertIn('data-selftest="ok"', dom, dom[dom.find("data-selftest"):dom.find("data-selftest") + 200])
            self.assertIn('data-selftest-blocks="11"', dom)
            self.assertIn('data-selftest-edges="15"', dom)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
