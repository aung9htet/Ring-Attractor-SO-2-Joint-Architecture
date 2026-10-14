"""Browser GUI for ring-attractor reaches on Kangaroo: pick arm and joint, drag the goal slider; the ring drives the
joint from where it is, live. Simplified port of the TIAGo dashboard (origin/velocity-calibration: cosim/dashboard.py).
Design: docs/kangaroo/design/specs/2026-10-13-kangaroo-ring-gui-design.md; decisions: DECISIONS.md D27, D30.

Threads: rclpy and NEST run only in the main thread (process); HTTP threads only queue commands and read snapshots.
One ring at a time (NEST's kernel is global, D12). Building a ring takes 22-30 s, so while idle the main thread
pre-builds one for the selected joint (the warm spare): a goal sent then starts at once.
"""
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rclpy

from . import RESULTS, save_trial
from .arm import N_JOINTS, SIDES
from .joint_setup import newest_setup, uncalibrated
from .gui_page import PAGE
from .ring import RingDrive, run_trial
from .warm_spare import WarmSpare

BAD_REQUEST = (KeyError, TypeError, ValueError, AttributeError, AssertionError)  # becomes HTTP 400


def side_and_joint(body):
    side, joint = body['side'], int(body['joint'])
    if side not in SIDES or joint not in range(N_JOINTS):
        raise ValueError('side must be left or right, joint 0-3')
    return side, joint


class Gui:
    """Bridge between HTTP threads (queue commands, read snapshots) and the main thread (runs them)."""

    def __init__(self, calibration_root=RESULTS / 'calibration', out_root=RESULTS):
        self.calibration_root, self.out_root = Path(calibration_root), Path(out_root)
        self.lock = threading.RLock()
        self.pending = None   # the next command; a newer one replaces it (the latest goal wins)
        self.subscribers = []  # ponytail: unbounded per-browser queues; bound them if a tab stalls for long
        self.stop_requested = False
        self.status = dict(status='idle')
        self.selected = None  # (side, joint, seed) whose ring is pre-built while idle
        self.positions = {}   # side -> 4 joint positions [rad], copied by the main thread
        self.spare = WarmSpare(self.build_ring)  # main thread only

    def build_ring(self, selected):
        """The ring for `selected` (side, joint, seed) under the joint's newest setup (22-30 s)."""
        side, joint, seed = selected
        return RingDrive(newest_setup(side, joint, self.calibration_root), seed)

    def publish(self, kind, data):
        msg = f'event: {kind}\ndata: {json.dumps(data)}\n\n'
        with self.lock:
            for q in self.subscribers:
                q.put(msg)

    def set_status(self, **status):
        with self.lock:
            self.status = status
            self.publish('status', status)

    def submit(self, command, **status):
        """Make `command` the next one, replacing any pending one; announced before it can be taken."""
        with self.lock:
            self.stop_requested = False
            self.set_status(status='queued', **status)
            self.pending = command

    def take(self):
        """The pending command (or None), cleared."""
        with self.lock:
            command, self.pending = self.pending, None
            return command

    def request_start(self, body):
        """Queue a reach toward body['goal'] (rad, inside the joint's ring range); raises a BAD_REQUEST error."""
        side, joint = side_and_joint(body)
        goal, seed = float(body['goal']), int(body.get('seed', 12345))
        lo, hi = newest_setup(side, joint, self.calibration_root).limits
        if not lo <= goal <= hi:
            raise ValueError(f'goal {goal} outside the ring range [{lo}, {hi}]')
        self.submit(dict(kind='reach', side=side, joint=joint, goal=goal, seed=seed),
                    side=side, joint=joint, goal=goal, seed=seed)
        return 202, 'queued'

    def request_reset(self, body):
        side, joint = side_and_joint(body)
        self.submit(dict(kind='reset', side=side, joint=joint), side=side, joint=joint, reset=True)
        return 202, 'queued'

    def select(self, body):
        selected = (*side_and_joint(body), int(body.get('seed', 12345)))
        with self.lock:
            self.selected = selected
        return 200, 'selected'

    def joints(self):
        """Every joint's ring range, decoder file, start pose and last known position; a joint whose newest
        calibration cannot be loaded shows the uncalibrated range and an `error`, instead of failing the page."""
        with self.lock:
            positions = dict(self.positions)
        out = []
        for side in SIDES:
            for joint in range(N_JOINTS):
                try:
                    s, error = newest_setup(side, joint, self.calibration_root), {}
                except BAD_REQUEST as e:
                    s, error = uncalibrated(side, joint), dict(error=repr(e))
                out.append(dict(side=side, joint=joint, limits=list(s.limits), calibration=s.calibration,
                                start=list(s.start), position=positions.get(side, [None] * N_JOINTS)[joint], **error))
        return out


def prepare_spare(gui, selected):
    """Build the selected joint's ring now (22-30 s), so its next reach starts at once."""
    side, joint, _ = selected
    gui.set_status(status='building', side=side, joint=joint)
    try:
        gui.spare.prepare(selected)
        gui.set_status(status='ready', side=side, joint=joint)
    except Exception as e:  # keep serving; the spare does not retry this selection until it changes
        gui.set_status(status='error', message=repr(e), side=side, joint=joint)


def process(arms, gui, timeout_s=None):
    """Run the next command: a reach from where the joint is, or a reset to the start pose.

    While idle: share joint positions and pre-build the selected joint's ring. Returns a reach's trial dict; None after
    a reset, an error, or `timeout_s` seconds idle.
    """
    t_end = None if timeout_s is None else time.monotonic() + timeout_s
    while (cmd := gui.take()) is None:
        if t_end is not None and time.monotonic() > t_end:
            return None
        with gui.lock:
            gui.positions = {side: list(arm.current_positions) for side, arm in arms.items()}
            selected = gui.selected
        if selected and gui.spare.due(selected):
            prepare_spare(gui, selected)
        rclpy.spin_once(next(iter(arms.values())).node, timeout_sec=0.1)  # one node serves every arm
    side, joint, arm = cmd['side'], cmd['joint'], arms[cmd['side']]

    def on_tick(tick):
        gui.publish('tick', tick)
        return gui.stop_requested or gui.pending is not None  # Stop, or a newer goal replaces this reach

    try:
        setup = newest_setup(side, joint, gui.calibration_root)
        if cmd['kind'] == 'reset':
            arm.reset(setup.start)
            gui.set_status(status='reset', side=side, joint=joint, start=list(setup.start))
            return None
        goal, seed = cmd['goal'], cmd['seed']
        ring = gui.spare.take(setup, seed)
        running = dict(side=side, joint=joint, joint_name=arm.joint_names[joint], goal=goal, seed=seed,
                       calibration=setup.calibration, limits=list(setup.limits), building=ring is None)
        gui.set_status(status='running', **running)
        result = run_trial(arm, setup, goal, seed, on_tick=on_tick, ring=ring)
        path, trial = save_trial('ring_gui', setup, result, gui.out_root)
    except Exception as e:  # keep serving; the page shows the error
        gui.set_status(status='error', message=repr(e), side=side, joint=joint)
        return None
    gui.set_status(status='done', **running, q_start=result['q_start'], q_final=result['q_final'],
                   stop_reason=result['stop_reason'], ticks=len(result['joint_position']), path=str(path.parent))
    return trial


def make_server(gui, port, host='127.0.0.1'):
    """Serve the page and API in a daemon thread. 127.0.0.1 only: the container uses host networking."""
    posts = {'/api/start': gui.request_start, '/api/reset': gui.request_reset, '/api/select': gui.select}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype='application/json'):
            data = (json.dumps(body) if ctype == 'application/json' else body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/':
                self.send(200, PAGE, 'text/html; charset=utf-8')
            elif self.path == '/state':
                with gui.lock:
                    status = dict(gui.status)
                self.send(200, status)
            elif self.path == '/api/joints':
                self.send(200, gui.joints())
            elif self.path == '/events':
                self.events()
            else:
                self.send(404, 'not found')

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            except ValueError:
                return self.send(400, 'bad JSON')
            if self.path == '/api/stop':
                gui.stop_requested = True
                self.send(200, 'stop requested')
            elif self.path in posts:
                try:
                    self.send(*posts[self.path](body))
                except BAD_REQUEST as e:
                    self.send(400, f'bad request: {e!r}')
            else:
                self.send(404, 'not found')

        def events(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            q = queue.Queue()
            with gui.lock:
                gui.subscribers.append(q)
                q.put(f'event: status\ndata: {json.dumps(gui.status)}\n\n')
            try:
                while True:
                    self.wfile.write(q.get().encode())
                    self.wfile.flush()
            except OSError:
                pass  # browser tab closed
            finally:
                with gui.lock:
                    gui.subscribers.remove(q)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
