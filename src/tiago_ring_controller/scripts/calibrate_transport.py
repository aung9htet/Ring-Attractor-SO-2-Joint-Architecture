#!/usr/bin/env python3
"""Transport calibration: fit the decoder gain from NEST trials (docs/archive/velocity/calibration.md).

    python3 -B scripts/calibrate_transport.py --template two_ring_single_joint --engines nest --out /tmp/transport_cal
    python3 -B scripts/calibrate_transport.py --bounds bounded --starts 45 70 95 120 145 --goals 45 70 95 120 145 155 --out /tmp/cal_bounded
    python3 -B scripts/calibrate_transport.py my.graph.json --out /tmp/cal
    python3 -B scripts/calibrate_transport.py --trials-from /tmp/transport_cal --out /tmp/refit

Runs the graph on real NEST with the fake robot over a grid of start and goal
ring indices (the gain spike train does not depend on the robot), fits the bump
displacement per gain spike with a linear and a dead-zone model, cross-validates
both by seed and writes to ``--out``:

    transport_calibration.json   the candidate artifact
    trials.npz, trials.csv       per-step traces and one row per trial
    report.md                    verdicts, fit, speed-rate curve, validity map, per-joint gains
    speed_rate.png, prediction.png

Nothing is written into the package unless ``--install`` is given; that copies
the artifact to ``src/config/calibration/`` (then update the two manifests in
``test/golden/``).
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

from tiago_ring_controller.graph import TEMPLATES, Graph  # noqa: E402
from tiago_ring_controller.graph.calibration import (  # noqa: E402
    TRANSPORT_SCHEMA,
    index_pairs,
    network_fingerprint,
    run_transport_trials,
    transport_calibration_path,
    transport_motif,
)
from tiago_ring_controller.graph.templates import BOUNDED, BOUNDS, decoder_params  # noqa: E402
from tiago_ring_controller.training.transport import (  # noqa: E402
    STATUSES,
    TransportOptions,
    fit_transport,
    prediction_errors,
    speed_rate_curve,
    trials_from_arrays,
    trials_to_arrays,
)

SERIES = ("#2a78d6", "#eb6834")       # the two models, in a fixed order


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", nargs="?", help="graph file (.graph.json) with the single-joint motif")
    parser.add_argument("--template", choices=sorted(TEMPLATES), default=None)
    parser.add_argument("--joint", type=int, default=5, help="joint of the template (the network is the same for every joint)")
    parser.add_argument("--set", action="append", default=[],
                        help="--bounds bounded: setting key=value (graph.templates.BOUNDED), e.g. gain_to_ring_weight=-1.2")
    parser.add_argument("--bounds", choices=BOUNDS, default="legacy",
                        help="template ring: legacy, or the bounded ring (its artifact installs as %s)" % BOUNDED["transport"])
    parser.add_argument("--engines", choices=("nest", "fake"), default="nest")
    parser.add_argument("--starts", type=int, nargs="+", default=[35, 60, 85, 110, 135], help="start ring indices")
    parser.add_argument("--goals", type=int, nargs="+", default=[35, 60, 85, 110, 135, 150], help="goal ring indices")
    parser.add_argument("--min-distance", type=int, default=22, help="smallest start-goal distance in ring indices")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13579, 7, 99, 2024])
    parser.add_argument("--tau", type=float, default=0.15, help="decoder filter time constant written to the artifact (s)")
    parser.add_argument("--delay-steps", type=int, default=0)
    parser.add_argument("--trials-from", default=None, help="refit from the trials.npz of an earlier --out instead of running NEST")
    parser.add_argument("--out", required=True)
    parser.add_argument("--install", action="store_true", help="copy the artifact into src/config/calibration/")
    return parser.parse_args(argv)


def parse_settings(items):
    """``key=value`` strings -> dict (values parsed as JSON where they parse)."""

    settings = {}
    for item in items or ():
        key, _, value = item.partition("=")
        try:
            settings[key] = json.loads(value)
        except ValueError:
            settings[key] = value
    return settings


def graph_factory(args):
    if args.graph:
        def factory(goal_rad, seed):
            graph = Graph.load(args.graph)
            if seed is not None:
                graph.simulation.rng_seed = seed
            return graph
        return factory, os.path.basename(args.graph)
    name = args.template or "two_ring_single_joint"

    def factory(goal_rad, seed):
        kwargs = {} if seed is None else {"seed": seed}
        if args.bounds == "bounded":
            # the fake robot's decoder gains do not change the bump transport; legacy gains need no artifact
            # calibrated at one NEST step per tick: the artifact is per 50 ms of network time
            kwargs.update(bounds="bounded", calibration="legacy", bounded=parse_settings(args.set), time_scale=1)
        return TEMPLATES[name](joint=args.joint, goal_rad=goal_rad, **kwargs)
    return factory, name if args.bounds == "legacy" else "%s (bounded)" % name


def validity_map(rows, statuses, errors_by_index, starts, goals):
    """Per start/goal cell: trials that moved, median endpoint error, median tracking error (indices)."""

    cells = {}
    for index, (row, status) in enumerate(zip(rows, statuses)):
        cell = cells.setdefault((row["start_index"], row["goal_index"]), {"n": 0, "ok": 0, "endpoint": [], "tracking": []})
        cell["n"] += 1
        if status == "ok":
            cell["ok"] += 1
            cell["endpoint"].append(row["endpoint_error"])
            cell["tracking"].append(errors_by_index[index])
    lines = ["| start \\ goal | " + " | ".join(str(g) for g in goals) + " |", "|---|" + "---|" * len(goals)]
    for start in starts:
        entries = []
        for goal in goals:
            cell = cells.get((start, goal))
            if cell is None:
                entries.append("")
            elif not cell["ok"]:
                entries.append("0/%d" % cell["n"])
            else:
                entries.append("%d/%d, %+.0f, %+.1f" % (cell["ok"], cell["n"], np.median(cell["endpoint"]), np.median(cell["tracking"])))
        lines.append("| %d | %s |" % (start, " | ".join(entries)))
    return lines


def plots(out_dir, trials, fit, options):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [t for t, status in zip(trials, fit.statuses) if status == "ok"]
    ink, muted = "#0b0b0b", "#52514e"

    def style(axis):
        axis.grid(True, color="#e4e3df", linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(muted)
        axis.tick_params(colors=muted)

    # Bump speed against gain rate: the curve the dead-zone model describes.
    curve = speed_rate_curve(usable, options)
    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    style(axis)
    rates = np.array([row["rate_mean"] for row in curve])
    axis.errorbar(rates, [row["speed_mean"] for row in curve], yerr=[row["speed_sd"] for row in curve],
                  fmt="o", color=ink, markersize=5, linewidth=1, capsize=2, label="NEST, 20-tick windows (mean, sd)")
    grid = np.linspace(0.0, max(rates.max(), 1.0) * 1.05, 200) if len(rates) else np.linspace(0, 120, 200)
    for color, name in zip(SERIES, ("linear", "dead_zone")):
        gain, theta = fit.candidates[name]
        axis.plot(grid, gain * np.maximum(grid - theta, 0.0), color=color, linewidth=2,
                  label="%s: a = %.5f, theta = %g" % (name.replace("_", " "), gain, theta))
    axis.set_xlabel("gain rate |right - left| (spikes per tick)", color=ink)
    axis.set_ylabel("bump speed (ring indices per tick)", color=ink)
    axis.set_title("Bump speed against gain rate", color=ink, loc="left")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, "speed_rate.png"), dpi=150)
    plt.close(figure)

    # Predicted against true bump displacement, one panel per model (same axes).
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), sharex=True, sharey=True)
    for axis, color, name in zip(axes, SERIES, ("linear", "dead_zone")):
        style(axis)
        gain, theta = fit.candidates[name]
        errors, moved = prediction_errors(usable, gain, theta, options)
        bound = float(np.max(np.abs(np.concatenate([moved, moved + errors])))) * 1.05
        axis.plot([-bound, bound], [-bound, bound], color=muted, linewidth=1)
        axis.plot(moved, moved + errors, "o", color=color, markersize=5, markeredgecolor="white", markeredgewidth=0.6)
        axis.set_title("%s (held-out p90 error %.0f%%)" % (name.replace("_", " "), 100 * fit.cv[name].p90_abs_rel), color=ink, loc="left")
        axis.set_xlabel("bump displacement (ring indices)", color=ink)
    axes[0].set_ylabel("predicted from gain spikes (ring indices)", color=ink)
    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, "prediction.png"), dpi=150)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    options = TransportOptions()
    os.makedirs(args.out, exist_ok=True)
    factory, graph_name = graph_factory(args)
    started = time.monotonic()

    if args.trials_from:
        with np.load(os.path.join(args.trials_from, "trials.npz")) as arrays:
            trials = trials_from_arrays(arrays)
        with open(os.path.join(args.trials_from, "trials.csv"), newline="") as stream:
            rows = [{key: (float(value) if key in ("displacement", "endpoint_error", "joint_displacement") else
                           int(value) if key in ("start_index", "goal_index", "n_steps") else value)
                     for key, value in row.items()} for row in csv.DictReader(stream)]
    else:
        pairs = index_pairs(args.starts, args.goals, args.min_distance)
        print("%d trials: %d start/goal pairs x %d seeds, engines=%s" % (len(pairs) * len(args.seeds), len(pairs), len(args.seeds), args.engines))

        def report(row):
            print("  seed %-6s %3d -> %3d : moved %+7.1f, endpoint %+6.1f, %3d steps, %s"
                  % (row["seed"], row["start_index"], row["goal_index"], row["displacement"], row["endpoint_error"],
                     row["n_steps"], row["status"]))

        trials, rows = run_transport_trials(factory, pairs, args.seeds, engines=args.engines, options=options, on_trial=report)

    np.savez_compressed(os.path.join(args.out, "trials.npz"), **trials_to_arrays(trials))
    try:
        fit = fit_transport(trials, options)
    except ValueError as exc:
        print("no fit: %s" % exc)
        return 1
    for row, status in zip(rows, fit.statuses):
        row["status"] = status
    with open(os.path.join(args.out, "trials.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    graph = factory(0.0, args.seeds[0])
    motif = transport_motif(graph)
    try:
        import nest
        nest_version = getattr(nest, "__version__", None)
    except ImportError:
        nest_version = None
    artifact = {
        "schema": TRANSPORT_SCHEMA,
        "model": fit.model,
        "index_per_spike": fit.index_per_spike,
        "dead_zone": fit.dead_zone,
        "decoder_tau_s": args.tau,
        "decoder_delay_steps": args.delay_steps,
        "dt_ms": float(graph.simulation.dt_ms),
        "network_fingerprint": network_fingerprint(graph),
        "fit": fit.as_dict(),
        "dataset": {
            "graph": graph_name, "starts_index": list(args.starts), "goals_index": list(args.goals),
            "min_distance": args.min_distance, "seeds": [str(t) for t in sorted({str(t.group) for t in trials})],
            "n_trials": len(trials), "ring_size": int(trials[0].ring_size),
            "valid_index_range": [options.edge_margin, trials[0].ring_size - 1 - options.edge_margin],
            "skip_steps": options.skip_steps, "min_displacement": options.min_displacement,
        },
        "actuator": {},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nest_version": nest_version,
    }
    artifact_path = os.path.join(args.out, "transport_calibration.json")
    with open(artifact_path, "w", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2)
        stream.write("\n")

    gain, theta = fit.index_per_spike, fit.dead_zone
    mapping = "bounded" if args.bounds == "bounded" else "collector"
    usable_index = [i for i, status in enumerate(fit.statuses) if status == "ok"]
    usable = [trials[i] for i in usable_index]
    errors, _ = prediction_errors(usable, gain, theta, options)
    errors_by_index = dict(zip(usable_index, errors))
    lines = [
        "# Transport calibration report", "",
        "Graph `%s`, %d trials, %s. Created %s." % (graph_name, len(trials), "NEST %s" % nest_version if nest_version else "no NEST", artifact["created"]),
        "", "## Trials", "",
        "| verdict | trials |", "|---|---:|",
    ] + ["| %s | %d |" % (status, fit.counts[status]) for status in STATUSES] + [
        "", "Only `ok` trials are fitted. `no_motion`: the bump moved less than %g indices. `edge`: it came within %g indices of a ring end. "
        "`stuck`: the gain populations fired and the bump did not follow." % (options.min_displacement, options.edge_margin),
        "", "## Fit", "",
        "| model | a (indices per spike) | theta (spikes per tick) | held-out median | held-out p90 | bias left | bias right |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("linear", "dead_zone"):
        a, th = fit.candidates[name]
        cv = fit.cv[name]
        lines.append("| %s%s | %.6f | %g | %.1f%% | %.1f%% | %+.1f%% | %+.1f%% |" % (
            name.replace("_", " "), " (chosen)" if name == fit.model else "", a, th, 100 * cv.median_abs_rel,
            100 * cv.p90_abs_rel, 100 * cv.bias_left, 100 * cv.bias_right))
    lines += [
        "", "Errors are of the predicted bump displacement, relative to the true one, on trials of the seed left out. "
        "Bias is positive when the prediction travels further than the bump.",
        "", "Dead zone chosen per fold: " + ", ".join("%s: theta %g, a %.6f" % fold for fold in fit.folds) + ".",
        "", "## Bump speed against gain rate", "",
        "| rate (spikes per tick) | windows | speed (indices per tick) | sd |", "|---|---:|---:|---:|",
    ]
    for row in speed_rate_curve(usable, options):
        lines.append("| %g-%g | %d | %.3f | %.3f |" % (row["rate_low"], row["rate_high"], row["n"], row["speed_mean"], row["speed_sd"]))
    starts = sorted({row["start_index"] for row in rows})
    goals = sorted({row["goal_index"] for row in rows})
    lines += ["", "## Validity map", "",
              "Per cell: usable trials / trials, median endpoint error (final bump minus goal index), "
              "median tracking error of the chosen model (predicted minus true displacement, indices).", ""]
    lines += validity_map(rows, fit.statuses, errors_by_index, starts, goals)
    lines += ["", "## Decoder gains per joint", "",
              "`k = a * rho / dt` with the %s mapping and the limits of `velocity_calibration.json`." % mapping, "",
              "| joint | rho (rad per index) | gain (rad/s per spike/tick) | dead zone | legacy gain |", "|---|---:|---:|---:|---:|"]
    joint_block = motif.get("joint")
    for joint in range(7):
        new = decoder_params(joint, mapping=mapping, calibration="transport", transport_path=artifact_path)
        old = decoder_params(joint, calibration="legacy")
        rho = new["gain_positive"] * artifact["dt_ms"] / 1000.0 / gain
        lines.append("| %d%s | %.5f | %.6f | %g | %.6f |" % (
            joint, " (calibrated on)" if joint_block is not None and joint_block.index == joint else "",
            rho, new["gain_positive"], new["dead_zone"], old["gain_positive"]))
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    try:
        plots(args.out, trials, fit, options)
    except ImportError:
        print("matplotlib is not installed: no plots")

    print("verdicts: %s" % fit.counts)
    for name in ("linear", "dead_zone"):
        a, th = fit.candidates[name]
        cv = fit.cv[name]
        print("%-9s a = %.6f, theta = %g: held-out median %.1f%%, p90 %.1f%%%s"
              % (name.replace("_", " "), a, th, 100 * cv.median_abs_rel, 100 * cv.p90_abs_rel, "  <- chosen" if name == fit.model else ""))
    print("done in %.0f s; written: %s" % (time.monotonic() - started, args.out))
    if args.install:
        target = transport_calibration_path()
        if args.bounds == "bounded":
            target = os.path.join(os.path.dirname(target), BOUNDED["transport"])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(artifact_path, target)
        print("installed: %s (update test/golden/artifact_manifest.json and artifact_schema_manifest.json)" % target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
