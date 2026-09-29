#!/usr/bin/env python3
"""Run a graph file through the co-simulation loop.

    python3 scripts/run_graph.py GRAPH.graph.json --engines fake|nest|full [--goal RAD ...] [--out DIR]
    python3 scripts/run_graph.py GRAPH.graph.json --validate
    python3 scripts/run_graph.py GRAPH.graph.json --engines full --dashboard   # http://localhost:8765/
    python3 scripts/run_graph.py --template two_ring_single_joint --engines nest

``--engines fake`` needs no simulator (fake NEST backend, fake robot), ``nest``
needs PyNEST (fake robot), ``full`` uses the graph's ``robot.engine`` (Gazebo).
Outputs go to ``--out`` in the collector layout when the graph has the
single-joint motif, otherwise as one JSON record per trial.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tiago_ring_controller.graph import TEMPLATES, Graph, GraphError, compile_graph, run_graph  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", nargs="?", help="graph file (.graph.json)")
    parser.add_argument("--template", choices=sorted(TEMPLATES), help="use a built-in template instead of a file")
    parser.add_argument("--engines", choices=("fake", "nest", "full"), default="fake")
    parser.add_argument("--goal", type=float, action="append", help="goal angle in rad (repeatable)")
    parser.add_argument("--seed", type=int, default=None, help="override simulation.rng_seed")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--validate", action="store_true", help="validate and describe the graph, do not run")
    parser.add_argument("--save", default=None, help="write the (resolved) graph file here and exit")
    parser.add_argument("--dashboard", action="store_true", help="serve the browser dashboard and run trials on request")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    return parser.parse_args(argv)


def load(args):
    if args.template:
        graph = TEMPLATES[args.template]()
    elif args.graph:
        graph = Graph.load(args.graph)
    else:
        raise SystemExit("give a graph file or --template")
    if args.seed is not None:
        graph.simulation.rng_seed = args.seed
    if args.max_steps is not None:
        graph.simulation.max_steps = args.max_steps
    return graph


def report_timing(index, record, legacy):
    timing = record.meta.get("timing", {})
    resets = ", ".join("%s %.1f s" % (name, secs) for name, secs in timing.get("reset_wall_s", {}).items())
    walls = [tick.wall_s for tick in record.main_ticks]
    print("trial %d timing: reset [%s]; lead %.2f s; %d ticks in %.1f s (%.1f ms/tick)"
          % (index, resets, timing.get("lead_wall_s", 0.0), len(walls), sum(walls), 1000.0 * sum(walls) / max(len(walls), 1)))


def main(argv=None):
    args = parse_args(argv)
    try:
        graph = load(args)
        problems = graph.problems()
    except GraphError as exc:
        print(exc)
        return 2
    if problems:
        print("graph is invalid:")
        for problem in problems:
            print("  -", problem)
        return 2
    if args.save:
        graph.save(args.save)
        print("written:", args.save)
        return 0
    if args.validate:
        order, _ = graph.build_order()
        print("graph %r: %d blocks (%d neural), %d edges; build order: %s"
              % (graph.name, len(graph.blocks), len(graph.neural_blocks()), len(graph.edges), " ".join(order)))
        return 0
    goals = args.goal or [None]

    if args.dashboard:
        from tiago_ring_controller.cosim.dashboard import DashboardObserver, DashboardServer, DashboardState, run_dashboard_session
        from tiago_ring_controller.cosim.runner import TrialWriter, trial_row

        state = DashboardState()
        compiled = compile_graph(graph, engines=args.engines)
        observer = DashboardObserver(state, compiled.config.joint_index, compiled.nest_engine.population_size or 0)
        compiled.loop.add_observer(observer)
        server = None
        writer = None
        try:
            compiled.loop.initialize()
            server = DashboardServer(state, args.dashboard_host, args.dashboard_port).start()
            state.set_status(graph=graph.name, message="graph %r loaded" % graph.name)
            print("dashboard: %s  graph %r (start trials from the page; Ctrl-C ends the session)" % (server.url, graph.name))
            if args.goal:
                state.set_status(next_goal=args.goal[0])
            if args.out and compiled.primary.complete:
                writer = TrialWriter(args.out, compiled.config)

            def on_trial(index, record, legacy):
                report_timing(index, record, legacy)
                if writer is not None:
                    writer.write(index, record, legacy, trial_row(compiled.config, index, legacy))

            try:
                run_dashboard_session(compiled.loop, compiled.config, state, on_trial=on_trial)
            except KeyboardInterrupt:
                print("interrupted; stopping")
        finally:
            if writer is not None:
                writer.close()
            if server is not None:
                server.stop()
            compiled.loop.shutdown()
        return 0

    goal_values = [g if g is not None else (graph.blocks[b].value() if (b := next((i for i, blk in graph.blocks.items() if blk.type_name == "Goal"), None)) else 0.0) for g in goals]
    started = time.monotonic()
    results = run_graph(graph, engines=args.engines, goals=goal_values, out_dir=args.out, on_trial=report_timing)
    for index, result in enumerate(results, start=1):
        record = result["record"]
        if "legacy" in result:
            s = result["legacy"]["scalars"]
            print("trial %d: goal=%.4f start=%.4f final=%.4f |err|=%.4f steps=%d stop=%s"
                  % (index, s["q_goal"], s["q_start"], s["q_final"], s["abs_position_error_rad"], s["n_steps"], s["stop_reason"]))
        else:
            last = record.main_ticks[-1] if record.main_ticks else None
            outputs = {} if last is None else {k: v.data for k, v in last.outputs.items() if k not in ("arm_velocity_cmd",)}
            print("trial %d: %d ticks, stop=%s, last outputs: %s" % (index, record.n_steps, record.stop_reason,
                  {k: {f: (round(x, 4) if isinstance(x, float) else x) for f, x in d.items() if not isinstance(x, list)} for k, d in outputs.items()}))
    print("done in %.1f s" % (time.monotonic() - started))
    if args.out:
        print("written:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
