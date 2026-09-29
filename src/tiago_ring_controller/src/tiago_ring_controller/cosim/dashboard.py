"""Browser dashboard: watch the ring and start/stop trials, no X11 needed.

A :class:`DashboardServer` (standard-library ``http.server`` in a daemon
thread) serves one static page and a few JSON endpoints:

* ``GET /``            the page (:mod:`dashboard_page`)
* ``GET /state``       latest status and tick snapshot
* ``GET /events``      server-sent events: one ``tick`` message per loop tick
                       and a ``status`` message on every state change
* ``POST /api/start``  ``{"goal": <rad>}`` queue a trial (goal optional if a
                       next goal is set)
* ``POST /api/stop``   end the running trial after the current tick
* ``POST /api/goal``   ``{"goal": <rad>}`` set the goal for the next trial
* ``POST /api/quit``   leave the session loop

The loop itself is untouched: :class:`DashboardObserver` publishes what each
tick recorded, and :meth:`DashboardState.stop_condition` is an ordinary stop
condition that fires when the page asked for a stop.  Trials run in the main
thread through :func:`run_dashboard_session`; the server threads only read
snapshots and enqueue commands.  Inside the Docker container (host
networking) the page is reachable from the host browser at
``http://localhost:<port>/``.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

from .config import CosimConfig
from .loop import FTILoop, LoopObserver, TickRecord, TrialRecord
from .dashboard_page import PAGE_HTML


class DashboardState:
    """Thread-safe bridge between the loop (main thread) and HTTP threads."""

    def __init__(self, max_queue: int = 200, history: int = 600) -> None:
        self._lock = threading.Lock()
        self._subscribers: List["queue.Queue[str]"] = []
        self._max_queue = max_queue
        self.snapshot: Optional[Dict[str, Any]] = None
        #: messages of the current trial, replayed to late subscribers so a
        #: page opened mid-trial or after it still shows the whole trial
        self.history: "deque[str]" = deque(maxlen=history)
        self.status: Dict[str, Any] = {
            "status": "idle",
            "message": "",
            "trial": 0,
            "next_goal": None,
            "trials": [],
            "config": {},
        }
        self.commands: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop_requested = False

    # -- pub/sub ----------------------------------------------------------
    def subscribe(self) -> "queue.Queue[str]":
        subscriber: "queue.Queue[str]" = queue.Queue(maxsize=self._max_queue)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: "queue.Queue[str]") -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def publish(self, message: Dict[str, Any]) -> None:
        text = json.dumps(message)
        with self._lock:
            kind = message.get("type")
            if kind == "trial_start":
                self.history.clear()
            if kind == "tick":
                self.snapshot = message
            if kind in ("trial_start", "tick", "trial_end"):
                self.history.append(text)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(text)
            except queue.Full:
                # A slow browser tab must never slow the loop down.
                pass

    def set_status(self, **changes: Any) -> Dict[str, Any]:
        with self._lock:
            self.status.update(changes)
            message = dict(self.status)
        message["type"] = "status"
        self.publish(message)
        return message

    def state_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {"status": dict(self.status), "snapshot": self.snapshot}

    def replay(self) -> List[str]:
        with self._lock:
            return list(self.history)

    # -- control ------------------------------------------------------------
    def request_stop(self) -> None:
        self._stop_requested = True

    def clear_stop(self) -> None:
        self._stop_requested = False

    def stop_condition(self, record: TickRecord) -> Optional[str]:
        return "user_stop" if self._stop_requested else None


class DashboardObserver(LoopObserver):
    """Publishes one compact snapshot per tick."""

    def __init__(self, state: DashboardState, joint_index: int, population_size: int) -> None:
        self.state = state
        self.joint_index = int(joint_index)
        self.population_size = int(population_size)
        self.trial = 0
        self._reset()

    def _reset(self) -> None:
        self.goal_index: Optional[int] = None
        self.goal_rad: Optional[float] = None
        self.init_index: Optional[int] = None
        self.last_counts = None

    def on_reset(self, loop: FTILoop) -> None:
        self._reset()
        self.trial += 1
        self.state.publish({"type": "trial_start", "trial": self.trial, "population_size": self.population_size})
        # Engines have finished rebuilding / homing when this is called.
        self.state.set_status(status="running", trial=self.trial, message="trial %d running" % self.trial)

    def on_tick(self, record: TickRecord) -> None:
        goal = record.outputs.get("goal_bump")
        if goal is not None:
            self.goal_index = int(goal["center_index"])
            self.goal_rad = goal.get("goal_rad")
        state = record.outputs.get("state_bump")
        if state is not None and self.init_index is None:
            self.init_index = int(state["center_index"])

        counts = record.inputs.get("ring_counts")
        new_sample = counts is not None and counts is not self.last_counts
        if new_sample:
            self.last_counts = counts
        cmd = record.outputs.get("arm_velocity_cmd")
        joint = record.inputs.get("joint_state")
        consumed = cmd["consumed"] if cmd is not None else None
        position = None if joint is None else float(joint["positions"][self.joint_index])

        message: Dict[str, Any] = {
            "type": "tick",
            "trial": self.trial,
            "phase": record.phase,
            "tick": record.tick,
            "t_s": record.t_ms / 1000.0,
            "goal_index": self.goal_index,
            "goal_rad": self.goal_rad,
            "init_index": self.init_index,
            "new_sample": new_sample,
            "nest_step": None if counts is None else counts["nest_step"],
            "left": None if counts is None else counts["left"],
            "right": None if counts is None else counts["right"],
            "r1": None if not new_sample else counts["r1_delta"],
            "r1_bump": None if counts is None else counts.get("r1_bump_index"),
            "centroid": None if counts is None else _finite(counts.get("r1_centroid")),
            "drive": None if consumed is None else consumed["filtered_drive"],
            "decoded_velocity": None if consumed is None else consumed["decoded_velocity"],
            "signed_spike": None if consumed is None else consumed["signed_spike"],
            "joint_position": position,
            "joint_velocity": None if joint is None else float(joint["velocities"][self.joint_index]),
            "error": None if position is None or self.goal_rad is None else self.goal_rad - position,
            "settled_count": None if cmd is None else cmd["consecutive_settled"],
            "sim_time_ms": None if joint is None else joint.get("sim_time_ms"),
        }
        self.state.publish(message)

    def on_trial_end(self, record: TrialRecord) -> None:
        self.state.publish({"type": "trial_end", "trial": self.trial, "stop_reason": record.stop_reason})


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if value == value else None


class _Handler(BaseHTTPRequestHandler):
    server: "DashboardHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return None

    def _send_json(self, payload: Any, code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        state = self.server.state
        if self.path in ("/", "/index.html"):
            body = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state":
            self._send_json(state.state_dict())
        elif self.path == "/events":
            self._stream_events()
        else:
            self._send_json({"error": "not found"}, 404)

    def _stream_events(self) -> None:
        state = self.server.state
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        subscriber = state.subscribe()
        try:
            current = state.state_dict()
            self._write_event(json.dumps(dict(current["status"], type="status")))
            for item in state.replay():
                self._write_event(item)
            while not self.server.shutting_down:
                try:
                    item = subscriber.get(timeout=1.0)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self._write_event(item)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            state.unsubscribe(subscriber)

    def _write_event(self, text: str) -> None:
        self.wfile.write(("data: " + text + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        state = self.server.state
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            self._send_json({"error": "invalid JSON"}, 400)
            return
        if not isinstance(payload, dict):
            payload = {}

        if self.path == "/api/start":
            goal = payload.get("goal", state.status.get("next_goal"))
            if goal is None:
                self._send_json({"error": "no goal given and no next goal set"}, 400)
                return
            command = {"type": "start", "goal": float(goal)}
            if payload.get("max_steps") is not None:
                command["max_steps"] = int(payload["max_steps"])
            state.commands.put(command)
            self._send_json({"queued": command})
        elif self.path == "/api/stop":
            state.request_stop()
            self._send_json({"stop_requested": True})
        elif self.path == "/api/goal":
            if payload.get("goal") is None:
                self._send_json({"error": "goal missing"}, 400)
                return
            state.commands.put({"type": "goal", "goal": float(payload["goal"])})
            self._send_json({"next_goal": float(payload["goal"])})
        elif self.path == "/api/quit":
            state.request_stop()
            state.commands.put({"type": "quit"})
            self._send_json({"quit": True})
        else:
            self._send_json({"error": "not found"}, 404)


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, state: DashboardState) -> None:
        super().__init__(address, _Handler)
        self.state = state
        self.shutting_down = False


class DashboardServer:
    def __init__(self, state: DashboardState, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.state = state
        self.httpd = DashboardHTTPServer((host, port), state)
        self.host, self.port = self.httpd.server_address[:2]
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="cosim-dashboard", daemon=True)

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("127.0.0.1", "0.0.0.0") else self.host
        return "http://%s:%d/" % (host, self.port)

    def start(self) -> "DashboardServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutting_down = True
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=5.0)


def run_dashboard_session(
    loop: FTILoop,
    config: CosimConfig,
    state: DashboardState,
    on_trial: Optional[Callable[[int, TrialRecord, Dict[str, Any]], None]] = None,
    poll_s: float = 0.2,
) -> List[Dict[str, Any]]:
    """Serve trials on demand until ``quit``.  Runs in the calling thread."""

    from .loop import AnyOf, MaxSteps, SettledFlag
    from .runner import legacy_collector_record, run_trial, trial_raster

    results: List[Dict[str, Any]] = []
    index = 0
    state.set_status(status="idle", config=config.to_dict(), message="ready")
    while True:
        try:
            command = state.commands.get(timeout=poll_s)
        except queue.Empty:
            continue
        kind = command.get("type")
        if kind == "quit":
            state.set_status(status="quit", message="session closed")
            break
        if kind == "goal":
            state.set_status(next_goal=command["goal"], message="next goal %.4f rad" % command["goal"])
            continue
        if kind != "start":
            continue

        index += 1
        goal = float(command["goal"])
        state.clear_stop()
        state.set_status(status="resetting", trial=index, next_goal=goal,
                         message="trial %d: building NEST network and homing the robot" % index)
        # Rebuilt per trial: user stop first, then the legacy settle/budget order.
        loop.stop_condition = AnyOf(
            state.stop_condition, SettledFlag(), MaxSteps(command.get("max_steps") or config.max_steps)
        )
        started = time.monotonic()
        try:
            record = run_trial(loop, goal, config, meta={"iteration_idx": index})
        except Exception as exc:  # keep the page alive; the operator decides what to do
            state.set_status(status="error", message="trial %d failed: %s" % (index, exc))
            continue
        legacy = legacy_collector_record(record, config, trial_raster(loop))
        scalars = legacy["scalars"]
        summary = {
            "trial": index,
            "goal": goal,
            "q_start": scalars["q_start"],
            "q_final": scalars["q_final"],
            "abs_error": scalars["abs_position_error_rad"],
            "n_steps": scalars["n_steps"],
            "stop_reason": record.stop_reason,
            "wall_s": time.monotonic() - started,
            "timing": record.meta.get("timing", {}),
        }
        trials = list(state.status.get("trials", [])) + [summary]
        state.set_status(status="idle", trials=trials,
                         message="trial %d finished: %s, |error| %.4f rad" % (index, record.stop_reason, summary["abs_error"]))
        results.append({"record": record, "legacy": legacy, "summary": summary})
        if on_trial is not None:
            on_trial(index, record, legacy)
    return results


__all__ = ["DashboardObserver", "DashboardServer", "DashboardState", "run_dashboard_session"]
