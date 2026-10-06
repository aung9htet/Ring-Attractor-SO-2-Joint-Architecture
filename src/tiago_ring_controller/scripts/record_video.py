#!/usr/bin/env python3
"""Record one trial from the simulation's camera, or put recordings side by side.

    python3 -B scripts/record_video.py trial --system F --joint 5 --goal-fraction 0.9 --out /out/v4_arm6.mp4
    python3 -B scripts/record_video.py trial --system B --joint 5 --goal-fraction 0.9 --motor-gain-scale 0.7 --out /out/v3_arm6_gain07.mp4
    python3 -B scripts/record_video.py trial --system PID --joint 5 --goal-fraction 0.9 --out /out/pid_arm6.mp4
    python3 -B scripts/record_video.py grid --columns 2 --out /out/side_by_side.mp4 /out/v3_arm6.mp4 /out/v4_arm6.mp4

``trial`` needs the simulation with the recording camera
(``/recording_camera/image_raw``), which only renders with a GPU:
``docker/batch.bash up NAME --gpu --out DIR``.  Systems: ``G``, ``T``, ``B``,
``F`` (the graph systems of ``scripts/validate_velocity.py``), ``P`` and ``N``
(the baselines of ``scripts/run_baselines.py``): one frame per 50 ms tick, so
these videos play in simulated time; ``L`` and ``PID`` (the frozen collectors
of ``scripts/run_legacy_trials.py``): frames at 20 per second of wall time.  The goal is an angle (``--goal``) or a fraction of the joint
range (``--goal-fraction``).  Every frame carries the joint, the goal, the
error and, for the ring systems, what the rings read and the gain spikes.
A JSON with the trial's numbers is written next to the video.

``grid`` tiles videos (``--columns`` per row); shorter ones hold their last frame.
"""

import argparse
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
LEGACY = os.path.join(os.path.dirname(HERE), "legacy")
for path in (SRC, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

SIZE = (1280, 720)
FPS = 20.0
LABELS = {
    "L": "Original scripts (two rings, own loop)",
    "PID": "PID baseline",
    "P": "Tuned controller (upper reference)",
    "N": "The robot's own controller (upper reference)",
    "G": "v3: two rings, legacy ring, hand-set gains",
    "T": "v3: two rings, legacy ring, transport calibration",
    "B": "v3: two rings, bounded ring",
    "F": "v4: three rings, body feedback",
}


class Camera:
    """The latest frame of ``/recording_camera/image_raw`` (BGR)."""

    def __init__(self):
        import rospy
        from sensor_msgs.msg import Image

        self.frame, self.lock = None, threading.Lock()
        rospy.Subscriber("/recording_camera/image_raw", Image, self._on_image, queue_size=1, buff_size=2 ** 24)
        rospy.wait_for_message("/recording_camera/image_raw", Image, timeout=60.0)

    def _on_image(self, msg):
        import cv2

        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        with self.lock:
            self.frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


class Writer:
    def __init__(self, path):
        import cv2

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.cv2 = cv2
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
        self.last, self.frames = None, 0

    def write(self, frame, lines):
        cv2 = self.cv2
        if frame.shape[1::-1] != SIZE:
            frame = cv2.resize(frame, SIZE)
        cv2.rectangle(frame, (0, 0), (SIZE[0], 14 + 34 * len(lines)), (20, 20, 20), -1)
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (20, 38 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.85 if i else 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        self.writer.write(frame)
        self.last, self.frames = frame, self.frames + 1

    def close(self, hold_s=1.5):
        if self.last is not None:
            for _ in range(int(hold_s * FPS)):
                self.writer.write(self.last)
        self.writer.release()


def header(label, joint, limits, goal, t_s):
    return [label, "arm_%d_joint  limits [%+.2f, %+.2f]  goal %+.3f rad   t = %.1f s" % (joint + 1, limits[0], limits[1], goal, t_s)]


def record_graph_trial(args, camera, goal, limits, label, transport):
    from tiago_ring_controller.cosim.loop import LoopObserver
    from validate_velocity import parse_settings, run_trial

    writer = Writer(args.out)
    joint = args.joint
    state_ids = ("jt__P_state", "jt__S_state") if args.system == "F" else ("state",)

    class Recorder(LoopObserver):
        def on_tick(self, record):
            if record.phase != "main":
                return
            frame = camera.read()
            if frame is None:
                return
            q = record.inputs["joint_state"]["positions"][joint]
            reads = [record.outputs[i].data["angle"] if record.outputs.get(i) is not None else float("nan") for i in state_ids]
            command = record.outputs["arm_velocity_cmd"]
            consumed = command.get("consumed", {})
            rings = ("prediction %+.3f   state ring %+.3f" % tuple(reads)) if args.system == "F" else ("ring reads %+.3f" % reads[0])
            writer.write(frame, header(label, joint, limits, goal, record.t_ms / 1000.0) + [
                "joint %+.3f rad   %s   joint to goal %+.3f rad" % (q, rings, goal - q),
                "gain spikes per tick  L %3d  R %3d   commanded %+.3f rad/s" % (consumed.get("left", 0), consumed.get("right", 0), command["velocities"][0])])

    try:
        row = run_trial(joint, args.system, None, goal, "full", transport, args.seed, args.time_scale, parse_settings(args.set),
                        parse_settings(args.feedback), args.motor_gain_scale, args.push, args.hold, args.disturb_at, observers=[Recorder()])
    finally:
        writer.close()
        transport.unpause()
        transport.shutdown()
    row.pop("trace", None)
    return row, writer.frames


def record_legacy_trial(args, camera, goal, limits, label):
    import importlib

    import rospy
    from sensor_msgs.msg import JointState

    if LEGACY not in sys.path:
        sys.path.insert(0, LEGACY)
    from run_legacy_trials import CONTROLLERS, BaseKeeper, LegacyDisturbance, _NoLogger, legacy_metrics

    module_name, class_name, system = CONTROLLERS["ring" if args.system == "L" else "pid"]
    module = importlib.import_module(module_name)
    module.TiagoSubscriber = _NoLogger
    out = os.path.dirname(os.path.abspath(args.out))

    class Collector(getattr(module, class_name)):
        def _output_dir(self):
            return out

    joint, name = args.joint, "arm_%d_joint" % (args.joint + 1)
    angle = {"q": float("nan")}

    def on_joints(msg):
        if name in msg.name:
            angle["q"] = float(msg.position[msg.name.index(name)])

    rospy.Subscriber("/joint_states", JointState, on_joints, queue_size=1)
    os.chdir(SRC)                                     # the legacy model reads ./config/
    collector = Collector(joint, use_manual_limits=False)
    collector.publisher.reset_pose()
    BaseKeeper().restore()
    collector.publisher.prime_command_state(force=True)
    writer, running, started = Writer(args.out), threading.Event(), time.monotonic()

    def capture():
        while not running.is_set():
            frame = camera.read()
            if frame is not None:
                q = angle["q"]
                writer.write(frame, header(label, joint, limits, goal, time.monotonic() - started) + [
                    "joint %+.3f rad   joint to goal %+.3f rad" % (q, goal - q)])
            time.sleep(1.0 / FPS)

    thread = threading.Thread(target=capture, daemon=True)
    thread.start()
    try:
        dt_s = getattr(collector, "time_step", 50.0) / 1000.0
        with LegacyDisturbance(collector.publisher, joint, dt_s, args.motor_gain_scale, args.push, args.hold, args.disturb_at):
            result = collector.run_trial(float(goal))
    finally:
        running.set()
        thread.join()
        writer.close()
    row = legacy_metrics(result, collector.joint_min, collector.joint_max, dt_s=getattr(collector, "time_step", 50.0) / 1000.0)
    row.pop("trace", None)
    row.update(system=system, joint=joint)
    return row, writer.frames


def record_baseline_trial(args, camera, goal, limits, label, transport):
    """The baselines of ``scripts/run_baselines.py``: one frame per 50 ms tick, as for the graph systems."""

    import run_baselines

    joint = args.joint
    engine = run_baselines.make_engine("full", transport)
    writer = Writer(args.out)

    def on_tick(t_s, q, command):
        frame = camera.read()
        if frame is None:
            return
        lines = ["joint %+.3f rad   joint to goal %+.3f rad" % (q, goal - q)]
        if args.system != "N":
            lines.append("commanded %+.3f rad/s" % command)
        writer.write(frame, header(label, joint, limits, goal, t_s) + lines)

    try:
        row = run_baselines.run_trial(engine, {"P": "p", "N": "native"}[args.system], joint, goal, None, None,
                                      args.motor_gain_scale, args.push, args.hold, args.disturb_at, on_tick=on_tick)
    finally:
        writer.close()
        transport.unpause()
        transport.shutdown()
    row.pop("trace", None)
    return row, writer.frames


def trial(args):
    import rospy

    from tiago_ring_controller.evaluation.serialization import json_safe
    from tiago_ring_controller.graph.templates import joint_limits

    limits = joint_limits(args.joint)
    if (args.goal is None) == (args.goal_fraction is None):
        raise SystemExit("give --goal or --goal-fraction")
    goal = args.goal if args.goal is not None else limits[0] + args.goal_fraction * (limits[1] - limits[0])
    label = args.label or LABELS[args.system]
    for text, value in (("motor gain x%g" % args.motor_gain_scale, args.motor_gain_scale != 1.0),
                        ("pushed %+.2f rad at %g s" % (args.push, args.disturb_at), args.push),
                        ("held %g s from %g s" % (args.hold, args.disturb_at), args.hold)):
        if value and not args.label:
            label += ", " + text
    if args.system in ("L", "PID"):
        rospy.init_node("record_video", anonymous=True)
        camera = Camera()
        row, frames = record_legacy_trial(args, camera, goal, limits, label)
    else:
        from tiago_ring_controller.cosim.gazebo_ros_engine import RospyTransport

        transport = RospyTransport(node_name="record_video")  # the node must exist before the camera subscribes
        transport.initialize()
        transport.unpause()
        camera = Camera()
        if args.system in ("P", "N"):
            row, frames = record_baseline_trial(args, camera, goal, limits, label, transport)
        else:
            row, frames = record_graph_trial(args, camera, goal, limits, label, transport)
    row.update(video=os.path.basename(args.out), frames=frames, label=label)
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as stream:
        json.dump(json_safe(row), stream, indent=1)
    final = row.get("final_rad", row.get("q_final"))
    print("%s arm_%d goal %+.3f: final %+.3f (error %+.3f), %d frames -> %s"
          % (args.system, args.joint + 1, goal, final, row["goal_error_rad"], frames, args.out))
    return 0


def grid(args):
    import cv2

    captures = [cv2.VideoCapture(path) for path in args.videos]
    counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in captures]
    if not all(counts):
        raise SystemExit("cannot read: %s" % [p for p, n in zip(args.videos, counts) if not n])
    columns = max(1, min(args.columns, len(captures)))
    rows = -(-len(captures) // columns)
    tile = (SIZE[0] // columns, int(SIZE[1] / columns))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (tile[0] * columns, tile[1] * rows))
    last = [np.zeros((tile[1], tile[0], 3), dtype=np.uint8) for _ in captures]
    for _ in range(max(counts)):
        for index, capture in enumerate(captures):
            ok, frame = capture.read()
            if ok:
                last[index] = cv2.resize(frame, tile)
        canvas = np.zeros((tile[1] * rows, tile[0] * columns, 3), dtype=np.uint8)
        for index, frame in enumerate(last):
            r, c = divmod(index, columns)
            canvas[r * tile[1]:(r + 1) * tile[1], c * tile[0]:(c + 1) * tile[0]] = frame
        writer.write(canvas)
    writer.release()
    print("%d videos, %d frames -> %s" % (len(captures), max(counts), args.out))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("trial")
    p.add_argument("--system", choices=sorted(LABELS), required=True)
    p.add_argument("--joint", type=int, required=True)
    p.add_argument("--goal", type=float, default=None, help="goal angle, rad")
    p.add_argument("--goal-fraction", type=float, default=None, help="goal as a fraction of the joint range")
    p.add_argument("--seed", type=int, default=13579)
    p.add_argument("--time-scale", type=int, default=None)
    p.add_argument("--set", action="append", default=[], help="bounded-ring setting key=value")
    p.add_argument("--feedback", action="append", default=[], help="v4 setting key=value")
    p.add_argument("--motor-gain-scale", type=float, default=1.0)
    p.add_argument("--push", type=float, default=0.0)
    p.add_argument("--hold", type=float, default=0.0)
    p.add_argument("--disturb-at", type=float, default=1.0)
    p.add_argument("--label", default=None, help="first overlay line (default: the system's name and the disturbance)")
    p.add_argument("--out", required=True, help="the .mp4 to write")
    p = sub.add_parser("grid")
    p.add_argument("videos", nargs="+")
    p.add_argument("--columns", type=int, default=2)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"trial": trial, "grid": grid}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
