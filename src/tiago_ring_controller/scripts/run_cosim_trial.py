#!/usr/bin/env python3
"""Run lock-stepped trials through the co-simulation loop (plan B).

Examples (inside the container, from the package's ``src`` directory or with
it on ``PYTHONPATH``):

    # No simulators: fake NEST + fake robot, prints a summary.
    python3 ../scripts/run_cosim_trial.py --engines fake --goal 0.6

    # Real NEST + fake robot (phase 2 check), records to a directory.
    python3 ../scripts/run_cosim_trial.py --engines nest --seed 13579 --out /tmp/cosim_nest

    # Real NEST + Gazebo through ROS (phase 3), requires the simulation running.
    python3 ../scripts/run_cosim_trial.py --engines full --goal 0.6 --out /tmp/cosim_full

The legacy scripts are untouched; this driver writes the collector's artifact
layout (``trials_summary.csv``, ``trials/*_timeseries.npz``, ``*_raster.npz``)
plus one ``*_cosim.json`` TrialRecord per trial.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tiago_ring_controller.config import load_joint_calibration, module_config_path  # noqa: E402
from tiago_ring_controller.control.profiles import get_legacy_control_profile  # noqa: E402
from tiago_ring_controller.cosim import CosimConfig  # noqa: E402
from tiago_ring_controller.cosim.runner import (  # noqa: E402
    build_loop,
    make_fake_engines,
    make_gazebo_engine,
    make_nest_engine,
    run_session,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", choices=("fake", "nest", "full"), default="fake")
    parser.add_argument("--profile", choices=("collector", "analysis", "calibration"), default="collector")
    parser.add_argument("--joint", type=int, default=5)
    parser.add_argument("--goal", type=float, action="append", help="goal angle in rad (repeatable)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lead", type=int, default=None, help="NEST lead steps (default: profile lookahead)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--step-mode", choices=("run", "simulate"), default="run")
    parser.add_argument("--model", choices=("legacy", "vectorised"), default="legacy",
                        help="NEST model: the frozen legacy facade or the vectorised build with "
                             "build-time stimulus generators (bumps at any tick)")
    parser.add_argument("--stepper", choices=("clock_wait", "plugin"), default="clock_wait")
    parser.add_argument("--proprioception", choices=("once", "continuous", "off"), default="once")
    parser.add_argument("--reset-mode", choices=("rebuild", "continue"), default="rebuild",
                        help="between trials: rebuild the network and home the robot (legacy) or continue")
    parser.add_argument("--out", default=None, help="output directory (collector layout)")
    parser.add_argument("--config", default=None, help="load a CosimConfig JSON instead of the profile")
    parser.add_argument("--monitor", action="store_true", help="open a live matplotlib view of the ring")
    parser.add_argument("--monitor-every", type=int, default=1, help="redraw every N ticks (default 1)")
    parser.add_argument("--monitor-frames", default=None, help="also save one PNG per redraw into this directory")
    parser.add_argument("--monitor-history", type=int, default=120, help="ticks shown in the raster panel")
    parser.add_argument("--monitor-hold", action="store_true", help="keep the window open after the last trial")
    parser.add_argument("--monitor-backend", default=None,
                        help="matplotlib backend for the window (default: QtAgg, Qt5Agg, TkAgg, GTK3Agg in turn)")
    parser.add_argument("--dashboard", action="store_true",
                        help="serve a browser dashboard and run trials on request instead of --goal")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--dashboard-host", default="127.0.0.1", help="use 0.0.0.0 to reach it from another machine")
    return parser.parse_args(argv)


def build_config(args):
    if args.config:
        config = CosimConfig.load(args.config)
    else:
        calibration_path = module_config_path(
            os.path.join(SRC, "placeholder.py"), "calibration", "velocity_calibration.json"
        )
        calibration = load_joint_calibration(calibration_path, args.joint)
        config = CosimConfig.from_profile(get_legacy_control_profile(args.profile), args.joint, calibration)
        if config.joint_min is None:
            # Fakes do not care about the mapping range; real runs must have limits.
            config = CosimConfig.from_dict(dict(config.to_dict(), joint_min=-1.0, joint_max=1.0))
    overrides = {"nest_step_mode": args.step_mode, "stepper": args.stepper,
                 "proprioception_mode": args.proprioception, "reset_mode": args.reset_mode,
                 "nest_model": args.model}
    if args.seed is not None:
        overrides["rng_seed"] = args.seed
    if args.lead is not None:
        overrides["nest_lead_steps"] = args.lead
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    return CosimConfig.from_dict(dict(config.to_dict(), **overrides))


def build_engines(args, config):
    if args.engines == "fake":
        return make_fake_engines(config)
    nest_engine = make_nest_engine(config)
    if args.engines == "nest":
        from tiago_ring_controller.cosim.fakes import FakeRobotEngine

        return [nest_engine, FakeRobotEngine("robot")]
    return [nest_engine, make_gazebo_engine(config)]


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)
    goals = args.goal or [0.5]
    engines = build_engines(args, config)
    monitor = None
    if args.monitor or args.monitor_frames:
        from tiago_ring_controller.cosim.runner import population_size_for
        from tiago_ring_controller.cosim.visualization import RingMonitor

        monitor = RingMonitor(
            population_size_for(engines, config), config.joint_index, dt_ms=config.dt_ms,
            history=args.monitor_history, render_every=args.monitor_every,
            show=args.monitor, frame_dir=args.monitor_frames, backend=args.monitor_backend,
        )
    observers = [monitor] if monitor else []
    dashboard = None
    if args.dashboard:
        from tiago_ring_controller.cosim.dashboard import DashboardObserver, DashboardServer, DashboardState
        from tiago_ring_controller.cosim.runner import population_size_for

        dashboard = DashboardState()
        observers.append(DashboardObserver(dashboard, config.joint_index, population_size_for(engines, config)))
    loop = build_loop(config, engines, observers=observers or None)
    print("config:", config.to_json(indent=None))

    def report_timing(index, record, legacy):
        timing = record.meta.get("timing", {})
        resets = ", ".join("%s %.1f s" % (name, secs) for name, secs in timing.get("reset_wall_s", {}).items())
        walls = [tick.wall_s for tick in record.main_ticks]
        print("trial %d timing: reset [%s]; lead %.2f s; %d ticks in %.1f s (%.1f ms/tick)"
              % (index, resets, timing.get("lead_wall_s", 0.0), len(walls), sum(walls),
                 1000.0 * sum(walls) / max(len(walls), 1)))

    server = None
    writer = None
    try:
        started = time.monotonic()
        loop.initialize()
        print("engines initialised in %.1f s (ROS node, joint states, NEST import)" % (time.monotonic() - started))
        if dashboard is not None:
            from tiago_ring_controller.cosim.dashboard import run_dashboard_session
            from tiago_ring_controller.cosim.runner import TrialWriter, trial_row

            server = DashboardServer(dashboard, args.dashboard_host, args.dashboard_port).start()
            print("dashboard: %s  (start trials from the page; Ctrl-C or the quit endpoint ends the session)" % server.url)
            if args.goal:
                dashboard.set_status(next_goal=goals[0])
            if args.out:
                writer = TrialWriter(args.out, config)

            def on_dashboard_trial(index, record, legacy):
                report_timing(index, record, legacy)
                if writer is not None:
                    writer.write(index, record, legacy, trial_row(config, index, legacy))

            try:
                results = run_dashboard_session(loop, config, dashboard, on_trial=on_dashboard_trial)
            except KeyboardInterrupt:
                print("interrupted; stopping")
                results = []
        else:
            results = run_session(loop, config, goals, out_dir=args.out, on_trial=report_timing)
    finally:
        if writer is not None:
            writer.close()
        if server is not None:
            server.stop()
        loop.shutdown()
    for index, result in enumerate(results, start=1):
        scalars = result["legacy"]["scalars"]
        record = result["record"]
        line = (
            "trial %d: goal=%.4f start=%.4f final=%.4f |err|=%.4f steps=%d stop=%s"
            % (index, scalars["q_goal"], scalars["q_start"], scalars["q_final"],
               scalars["abs_position_error_rad"], scalars["n_steps"], scalars["stop_reason"])
        )
        if "gazebo_overshoot_ms" in record.meta:
            overshoot = record.meta["gazebo_overshoot_ms"]
            line += " overshoot_ms(mean=%.3f p95=%.3f max=%.3f)" % (
                overshoot["mean"], overshoot["p95"], overshoot["max"])
        if record.ticks:
            walls = [tick.wall_s for tick in record.main_ticks]
            line += " tick_wall_ms(mean=%.2f max=%.2f)" % (
                1000.0 * sum(walls) / len(walls), 1000.0 * max(walls))
        print(line)
    if args.out:
        print("written:", args.out)
    if monitor is not None:
        if args.monitor_frames:
            print("frames:", monitor.frames_written, "in", args.monitor_frames)
        if args.monitor_hold:
            print("close the monitor window to exit")
            monitor.hold()
        monitor.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
