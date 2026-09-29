"""Live view of the state ring while the robot moves.

:class:`RingMonitor` is a :class:`LoopObserver`: it reads the datapacks the
loop already records (``ring_counts``, ``joint_state``, ``arm_velocity_cmd``,
``goal_bump``, ``state_bump``) and redraws four panels after every tick:

* the state ring r1 as a polar activity plot with the goal index (red), the
  initial state index (grey) and the current centroid (green);
* a rolling raster of r1 counts per tick;
* left / right gain counts and the signed drive;
* measured joint angle against the goal, plus the decoded velocity.

With ``show=True`` it opens an interactive matplotlib window (Tk or Qt in the
container, through the launcher's X11 forwarding).  With ``show=False`` it
renders off-screen with Agg, which is what ``frame_dir`` (PNG per render) and
the tests use.  Rendering happens after the engines advanced and never feeds
back into the loop, so the trial record is identical with or without it; it
only costs wall time (about 50 to 100 ms per frame, see ``render_every``).
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Deque, List, Optional

import numpy as np

from .loop import FTILoop, LoopObserver, TickRecord, TrialRecord


_NON_INTERACTIVE = ("agg", "pdf", "ps", "svg", "template", "cairo")
_INTERACTIVE_CANDIDATES = ("QtAgg", "Qt5Agg", "TkAgg", "GTK3Agg")


def select_interactive_backend(preferred: Optional[str] = None) -> Optional[str]:
    """Pick a window-capable matplotlib backend, or return ``None``.

    ``MPLBACKEND=Agg`` (set by the launcher's ``--headless`` mode) and an
    empty ``DISPLAY`` both resolve to Agg; here a backend is forced when a
    display exists so the monitor can still open a window.
    """

    import matplotlib

    if not os.environ.get("DISPLAY"):
        return None
    candidates = [preferred] if preferred else list(_INTERACTIVE_CANDIDATES)
    current = matplotlib.get_backend().lower()
    if not any(current.startswith(name) for name in _NON_INTERACTIVE) and not preferred:
        return matplotlib.get_backend()
    for name in candidates:
        try:
            matplotlib.use(name, force=True)
            import matplotlib.pyplot  # noqa: F401 - binds the backend

            return name
        except Exception:  # pragma: no cover - depends on installed toolkits
            continue
    return None


class RingMonitor(LoopObserver):
    def __init__(
        self,
        population_size: int,
        joint_index: int,
        dt_ms: float = 50.0,
        history: int = 120,
        render_every: int = 1,
        show: bool = True,
        frame_dir: Optional[str] = None,
        title: str = "Ring attractor co-simulation",
        figsize=(12.0, 7.5),
        backend: Optional[str] = None,
    ) -> None:
        self.population_size = int(population_size)
        self.joint_index = int(joint_index)
        self.dt_ms = float(dt_ms)
        self.history = int(history)
        self.render_every = max(1, int(render_every))
        self.frame_dir = frame_dir
        self.title = title
        self.frames_written = 0
        self.renders = 0
        self.trial_index = 0
        self._plt = None

        self.backend = None
        if show:
            self.backend = select_interactive_backend(backend)
            if self.backend is None:
                if os.environ.get("DISPLAY"):
                    print("[RingMonitor] no interactive matplotlib backend could be loaded "
                          "(tried %s); rendering off-screen only." % ", ".join(_INTERACTIVE_CANDIDATES))
                else:
                    print("[RingMonitor] DISPLAY is not set inside the container, so no window "
                          "can open. Start the launcher without --headless (it forwards X11), "
                          "or use --monitor-frames DIR to save PNGs instead.")
                show = False
            else:
                import matplotlib.pyplot as plt

                self._plt = plt
                print("[RingMonitor] live window on backend %s" % self.backend)
        self.show = show

        if self.show:
            self._plt.ion()
            self.fig = self._plt.figure(figsize=figsize)
            self._plt.show(block=False)
        else:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            self.fig = Figure(figsize=figsize)
            FigureCanvasAgg(self.fig)
        if frame_dir:
            os.makedirs(frame_dir, exist_ok=True)

        self._build_axes()
        self._reset_buffers()

    # -- figure -------------------------------------------------------------
    def _build_axes(self) -> None:
        N = self.population_size
        fig = self.fig
        fig.suptitle(self.title)
        self.ax_ring = fig.add_subplot(2, 2, 1, projection="polar")
        self.ax_raster = fig.add_subplot(2, 2, 2)
        self.ax_gain = fig.add_subplot(2, 2, 3)
        self.ax_joint = fig.add_subplot(2, 2, 4)

        self.angles = 2.0 * np.pi * np.arange(N) / N
        self.bars = self.ax_ring.bar(
            self.angles, np.zeros(N), width=2.0 * np.pi / N, color="#1f77b4", alpha=0.85
        )
        (self.goal_line,) = self.ax_ring.plot([0, 0], [0, 1], color="red", lw=2, label="goal")
        (self.init_line,) = self.ax_ring.plot([0, 0], [0, 1], color="grey", lw=1.5, ls="--", label="start")
        (self.centroid_line,) = self.ax_ring.plot([0, 0], [0, 1], color="green", lw=2, label="centroid")
        self.ax_ring.set_theta_zero_location("N")
        self.ax_ring.set_theta_direction(-1)
        self.ax_ring.set_yticklabels([])
        self.ax_ring.set_title("state ring r1 (spikes per tick)")
        self.ax_ring.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.15), fontsize=8)

        self.raster_img = self.ax_raster.imshow(
            np.zeros((N, self.history)), aspect="auto", origin="lower",
            cmap="viridis", interpolation="nearest", vmin=0, vmax=1,
        )
        (self.raster_goal,) = self.ax_raster.plot([], [], color="red", lw=1)
        self.ax_raster.set_title("r1 activity (last %d ticks)" % self.history)
        self.ax_raster.set_xlabel("NEST step")
        self.ax_raster.set_ylabel("neuron index")

        (self.left_line,) = self.ax_gain.plot([], [], color="#1f77b4", label="left")
        (self.right_line,) = self.ax_gain.plot([], [], color="#ff7f0e", label="right")
        (self.drive_line,) = self.ax_gain.plot([], [], color="black", lw=1, ls="--", label="filtered drive")
        self.ax_gain.axhline(0.0, color="grey", lw=0.5)
        self.ax_gain.set_title("gain populations")
        self.ax_gain.set_xlabel("NEST step")
        self.ax_gain.legend(loc="upper right", fontsize=8)

        (self.pos_line,) = self.ax_joint.plot([], [], color="#2ca02c", label="joint angle")
        self.goal_hline = self.ax_joint.axhline(0.0, color="red", lw=1, ls="--", label="goal")
        self.ax_vel = self.ax_joint.twinx()
        (self.vel_line,) = self.ax_vel.plot([], [], color="#9467bd", lw=1, label="decoded velocity")
        self.ax_joint.set_title("joint %d" % self.joint_index)
        self.ax_joint.set_xlabel("time [s]")
        self.ax_joint.set_ylabel("rad")
        self.ax_vel.set_ylabel("rad/s")
        self.ax_joint.legend(loc="upper left", fontsize=8)
        self.ax_vel.legend(loc="lower right", fontsize=8)
        self.status = self.fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=9)
        self.fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    def _reset_buffers(self) -> None:
        self.raster: Deque[np.ndarray] = deque(maxlen=self.history)
        self.steps: List[int] = []
        self.left: List[int] = []
        self.right: List[int] = []
        self.drive: List[float] = []
        self.time_s: List[float] = []
        self.position: List[float] = []
        self.velocity: List[float] = []
        self.goal_index: Optional[int] = None
        self.goal_rad: Optional[float] = None
        self.init_index: Optional[int] = None
        self.centroid: Optional[float] = None
        self.last_counts: Optional[Any] = None
        self.tick_count = 0
        self.stop_reason: Optional[str] = None

    # -- observer -----------------------------------------------------------
    def on_reset(self, loop: FTILoop) -> None:
        self._reset_buffers()
        self.trial_index += 1

    def on_tick(self, record: TickRecord) -> None:
        goal = record.outputs.get("goal_bump")
        if goal is not None:
            self.goal_index = int(goal["center_index"])
            self.goal_rad = goal.get("goal_rad")
        state = record.outputs.get("state_bump")
        if state is not None and self.init_index is None:
            self.init_index = int(state["center_index"])

        counts = record.inputs.get("ring_counts")
        if counts is not None and counts is not self.last_counts:
            self.last_counts = counts
            self.raster.append(np.asarray(counts["r1_delta"], dtype=float))
            self.steps.append(int(counts["nest_step"]))
            self.left.append(int(counts["left"]))
            self.right.append(int(counts["right"]))
            centroid = counts.get("r1_centroid")
            self.centroid = None if centroid is None or np.isnan(centroid) else float(centroid)
        cmd = record.outputs.get("arm_velocity_cmd")
        if counts is not None and counts is self.last_counts and cmd is not None and len(self.drive) < len(self.steps):
            self.drive.append(float(cmd["consumed"]["filtered_drive"]))
        elif counts is not None and len(self.drive) < len(self.steps):
            self.drive.append(self.drive[-1] if self.drive else 0.0)

        joint = record.inputs.get("joint_state")
        if joint is not None and record.phase == "main":
            self.time_s.append(record.t_ms / 1000.0)
            self.position.append(float(joint["positions"][self.joint_index]))
            self.velocity.append(
                float(cmd["velocities"][0]) if cmd is not None and cmd["velocities"] else 0.0
            )

        self.tick_count += 1
        if self.tick_count % self.render_every == 0:
            self.render()

    def on_trial_end(self, record: TrialRecord) -> None:
        self.stop_reason = record.stop_reason
        self.render()

    # -- drawing ------------------------------------------------------------
    def _angle(self, index: float) -> float:
        return 2.0 * np.pi * float(index) / self.population_size

    def render(self) -> None:
        N = self.population_size
        current = self.raster[-1] if self.raster else np.zeros(N)
        top = max(1.0, float(current.max()))
        for bar, height in zip(self.bars, current):
            bar.set_height(float(height))
        self.ax_ring.set_ylim(0, top * 1.05)
        for line, index in ((self.goal_line, self.goal_index), (self.init_line, self.init_index),
                            (self.centroid_line, self.centroid)):
            if index is None:
                line.set_data([], [])
            else:
                line.set_data([self._angle(index)] * 2, [0, top * 1.05])

        if self.raster:
            grid = np.stack(list(self.raster), axis=1)
            x0 = self.steps[-len(self.raster)]
            self.raster_img.set_data(grid)
            self.raster_img.set_extent((x0 - 0.5, x0 - 0.5 + grid.shape[1], -0.5, N - 0.5))
            self.raster_img.set_clim(0, max(1.0, float(grid.max())))
            if self.goal_index is not None:
                self.raster_goal.set_data([x0 - 0.5, x0 - 0.5 + grid.shape[1]], [self.goal_index] * 2)
            self.ax_raster.set_xlim(x0 - 0.5, x0 - 0.5 + max(grid.shape[1], self.history))
            self.ax_raster.set_ylim(-0.5, N - 0.5)

        if self.steps:
            n = min(len(self.steps), len(self.drive))
            self.left_line.set_data(self.steps, self.left)
            self.right_line.set_data(self.steps, self.right)
            self.drive_line.set_data(self.steps[:n], self.drive[:n])
            lo = min(min(self.drive[:n] or [0.0]), 0.0)
            hi = max(max(self.left), max(self.right), max(self.drive[:n] or [0.0]), 1.0)
            self.ax_gain.set_xlim(self.steps[0], max(self.steps[-1], self.steps[0] + 1))
            self.ax_gain.set_ylim(lo * 1.1 - 0.5, hi * 1.1 + 0.5)

        if self.time_s:
            self.pos_line.set_data(self.time_s, self.position)
            self.vel_line.set_data(self.time_s, self.velocity)
            values = list(self.position) + ([self.goal_rad] if self.goal_rad is not None else [])
            lo, hi = min(values), max(values)
            pad = max(0.05, 0.1 * (hi - lo))
            self.ax_joint.set_xlim(0.0, max(self.time_s[-1], self.dt_ms / 1000.0))
            self.ax_joint.set_ylim(lo - pad, hi + pad)
            vmax = max(max(abs(v) for v in self.velocity), 1e-3)
            self.ax_vel.set_ylim(-vmax * 1.2, vmax * 1.2)
            if self.goal_rad is not None:
                self.goal_hline.set_ydata([self.goal_rad, self.goal_rad])

        text = "trial %d   NEST step %s   t = %.2f s" % (
            self.trial_index, self.steps[-1] if self.steps else "-", self.time_s[-1] if self.time_s else 0.0)
        if self.left:
            text += "   L=%d R=%d" % (self.left[-1], self.right[-1])
        if self.position and self.goal_rad is not None:
            text += "   error = %.4f rad" % (self.goal_rad - self.position[-1])
        if self.stop_reason:
            text += "   stop: %s" % self.stop_reason
        self.status.set_text(text)

        self.renders += 1
        if self.show:
            # A synchronous draw plus a short pause services the GUI event
            # loop from inside the tick loop; draw_idle alone can defer every
            # repaint until the loop returns.
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self._plt.pause(0.001)
        else:
            self.fig.canvas.draw()
        if self.frame_dir:
            path = os.path.join(self.frame_dir, "trial_%03d_frame_%05d.png" % (self.trial_index, self.renders))
            self.fig.savefig(path, dpi=80)
            self.frames_written += 1

    def frame(self) -> np.ndarray:
        """Return the current figure as an RGBA array (used by tests)."""

        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())

    def hold(self) -> None:
        """Keep the window open until it is closed (interactive mode only)."""

        if self.show and self._plt is not None:
            self._plt.ioff()
            self._plt.show()

    def close(self) -> None:
        if self.show and self._plt is not None:
            self._plt.close(self.fig)


__all__ = ["RingMonitor", "select_interactive_backend"]
