#!/usr/bin/env python3
"""Fit and check the comparator (Homeostasis) weights of a bounded ring (docs/archive/velocity/bounded-ring.md).

    python3 -B scripts/fit_comparator.py collect --out /tmp/cmp                      # readout library, ~30 s
    python3 -B scripts/fit_comparator.py fit --library /tmp/cmp --rise-fraction 0.8 --arc-fraction 0.7 --name N_200_bounded_a070 --out /tmp/cmp
    python3 -B scripts/fit_comparator.py map --bounds legacy --out /tmp/cmp/map_legacy
    python3 -B scripts/fit_comparator.py map --bounds bounded --set comparator=N_200_bounded_a070 --out /tmp/cmp/map_bounded

``collect`` places a bump at every other ring index on a plain ring with the
transport frozen and counts the Fourier readout's spikes.  ``fit`` ridge-fits
the signed sine features to a target: the continuous bounded ramp (rising
across ``rise-fraction`` of the ring, centred opposite the seam) or the legacy
sawtooth for reference.  It reports the held-out error and how monotone the
fitted ramp is over the arc, and writes
``<name>_homeostasis_weights.npy`` / ``_metadata.json`` (``--install`` copies
them into ``src/config/homeostasis/``).  ``map`` runs the comparator with the
transport frozen for a grid of state and goal indices and reports wrong-sign
decisions and dead zones (heatmap and JSON).

    python3 -B scripts/fit_comparator.py map --fine --bounds bounded --out /tmp/cmp/fine           # the ramp comparator alone
    python3 -B scripts/fit_comparator.py map --fine --sharp --bounds bounded --out /tmp/cmp/sharp  # plus the goal ring's projection (v4)
    python3 -B scripts/fit_comparator.py drift --bounds bounded --out /tmp/cmp/drift

``map --fine`` steps the goal one index at a time around five states and
reports the blind band: the goal offsets around 0 that give no drive toward
the goal.  ``--sharp`` adds the reference projection of architecture v4
(``Gain.reference``, docs/architecture.md).  ``drift`` leaves a bump alone for
``--seconds`` and reports how far it wanders, per position and seed.
"""

import argparse
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

from tiago_ring_controller.config import source_config_path  # noqa: E402
from tiago_ring_controller.cosim.loop import LoopObserver  # noqa: E402
from tiago_ring_controller.graph import compile_graph, run_graph_trial, two_ring_single_joint  # noqa: E402
from tiago_ring_controller.graph.templates import FEEDBACK, bounded_settings  # noqa: E402
from tiago_ring_controller.training.comparator import (  # noqa: E402
    bounded_ramp,
    decision_summary,
    fit_comparator,
    monotonicity,
    ramp,
    sawtooth_ramp,
)

RING_SIZE = 200
SETTLE_TICKS = 6


class ReadoutSnapshot(LoopObserver):
    """Readout spike counts at two main ticks (cumulative recorder events)."""

    def __init__(self, graph, first, last):
        self.graph, self.first, self.last = graph, first, last
        self.counts = {}

    def on_tick(self, record):
        if record.phase == "main" and record.tick in (self.first, self.last):
            recorders = self.graph.blocks["f1"].population.recorders
            self.counts[record.tick] = np.asarray(recorders.get("n_events"), dtype=float)


def frozen_trial(graph, state_index, goal_index, ticks, engines="nest", observers=()):
    """Run ``ticks`` main ticks with the bump transport switched off."""

    graph.simulation.time_scale = 1                  # counts per 50 ms of network time
    graph.blocks["gain"].params["gain_to_ring_weight"] = 0.0
    graph.blocks["dec"].params["n_settle"] = 10 ** 6
    graph.simulation.max_steps = ticks
    graph.simulation.nest_lead_steps = 0
    encoder = graph.blocks["enc_state"]
    joint = next(b for b in graph.blocks.values() if b.type_name == "Joint")
    compiled = compile_graph(graph, engines=engines, observers=list(observers))
    compiled.robot_engine.home[joint.index] = encoder.index_to_angle(state_index, RING_SIZE)
    compiled.loop.initialize()
    try:
        return run_graph_trial(compiled, encoder.index_to_angle(goal_index, RING_SIZE))
    finally:
        compiled.loop.shutdown()


def collect(args):
    positions = list(range(1, RING_SIZE, 2))
    rows = []
    started = time.monotonic()
    ticks = SETTLE_TICKS + args.window_ticks + 1
    for position in positions:
        graph = two_ring_single_joint(joint=5, calibration="legacy")
        snapshot = ReadoutSnapshot(graph, SETTLE_TICKS, ticks - 1)
        frozen_trial(graph, position, RING_SIZE // 2, ticks, args.engines, [snapshot])
        rows.append(snapshot.counts[ticks - 1] - snapshot.counts[SETTLE_TICKS])
    window_ms = args.window_ticks * graph.simulation.dt_ms
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "readout_library.npz"), positions=np.array(positions),
                        counts=np.array(rows), window_ms=np.array(window_ms))
    print("readout library: %d positions, window %.0f ms, %.0f s -> %s"
          % (len(positions), window_ms, time.monotonic() - started, os.path.join(args.out, "readout_library.npz")))
    return 0


def fit(args):
    with np.load(os.path.join(args.library, "readout_library.npz")) as library:
        positions, counts, window_ms = library["positions"], library["counts"], float(library["window_ms"])
    legacy = np.load(source_config_path("homeostasis", "N_%d_homeostasis_weights.npy" % RING_SIZE))
    k = counts.shape[1] // 2
    legacy_signed = legacy[0:2 * k:2, 0]
    targets = {"sawtooth": sawtooth_ramp, "bounded": lambda theta: bounded_ramp(theta, args.rise_fraction)}
    report = {"rise_fraction": args.rise_fraction, "window_ms": window_ms, "positions": int(len(positions)), "fits": {}}
    chosen = None
    for name, target in targets.items():
        result = fit_comparator(counts, positions, RING_SIZE, target, name, window_ms, args.ridge_lambda)
        values = ramp(result, counts, window_ms)
        entry = {"train_r2": result.train_r2, "test_rms": result.test_rms, "test_max_error": result.test_max_error,
                 "weight_norm_vs_legacy": float(np.linalg.norm(result.signed_weights) / np.linalg.norm(legacy_signed)),
                 "monotonicity": {}}
        for fraction in (0.95, args.arc_fraction, args.rise_fraction):
            low, high = RING_SIZE / 2 * (1 - fraction), RING_SIZE / 2 * (1 + fraction)
            entry["monotonicity"]["%.2f" % fraction] = monotonicity(values, positions, low, high)
        report["fits"][name] = entry
        print("%-9s train R2 %.4f, held-out rms %.4f, max %.3f, |w| / legacy |w| %.2f" % (
            name, result.train_r2, result.test_rms, result.test_max_error, entry["weight_norm_vs_legacy"]))
        for key, m in entry["monotonicity"].items():
            print("     arc %s: %d of %d steps go the wrong way, smallest %+.4f, mean %+.4f" % (
                key, m["wrong_way"], m["steps"], m["smallest_step"], m["mean_step"]))
        if name == "bounded":
            chosen = result
    os.makedirs(args.out, exist_ok=True)
    weights_path = os.path.join(args.out, args.name + "_homeostasis_weights.npy")
    metadata_path = os.path.join(args.out, args.name + "_homeostasis_metadata.json")
    np.save(weights_path, chosen.artifact())
    feature_names = []
    for harmonic in range(1, k + 1):
        feature_names += ["sin_pos_k%d" % harmonic, "sin_neg_k%d" % harmonic]
    metadata = {
        "population_size": RING_SIZE, "feature_names": feature_names, "target": "bounded_ramp",
        "rise_fraction": args.rise_fraction, "arc_fraction": args.arc_fraction, "num_positions": int(len(positions)),
        "window_ms": window_ms, "ridge_lambda": args.ridge_lambda, "train_r2": chosen.train_r2,
        "test_rms": chosen.test_rms, "bias": chosen.bias, "fit": report["fits"]["bounded"],
        "note": "fitted by scripts/fit_comparator.py; docs/archive/velocity/bounded-ring.md",
    }
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    with open(os.path.join(args.out, args.name + "_fit_report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("written:", weights_path, metadata_path)
    if args.install:
        for path in (weights_path, metadata_path):
            target = source_config_path("homeostasis", os.path.basename(path))
            shutil.copyfile(path, target)
            print("installed:", target)
    return 0


def parse_settings(items):
    settings = {}
    for item in items or ():
        key, _, value = item.partition("=")
        try:
            settings[key] = json.loads(value)
        except ValueError:
            settings[key] = value
    return settings


def decision_map(args):
    settings = parse_settings(args.set)
    if args.bounds == "bounded":
        low, high = bounded_settings(RING_SIZE, settings)["limit_arc"]
    else:
        low, high = 5, 194
    grid = [int(round(v)) for v in np.linspace(low, high, args.grid)]
    rows = []
    started = time.monotonic()
    for state in grid:
        for goal in grid:
            graph = two_ring_single_joint(joint=5, calibration="legacy", bounds=args.bounds,
                                          bounded=settings if args.bounds == "bounded" else None)
            record = frozen_trial(graph, state, goal, SETTLE_TICKS + 18, args.engines)
            ticks = [t for t in record.main_ticks[SETTLE_TICKS:] if t.inputs.get("gain") is not None]
            drive = float(np.mean([t.inputs["gain"]["right_counts"] - t.inputs["gain"]["left_counts"] for t in ticks]))
            cmp = [t.inputs["cmp"]["counts"] for t in ticks]
            rows.append({"state": state, "goal": goal, "drive": drive,
                         "warm": float(np.mean([c["warm"] for c in cmp])), "cold": float(np.mean([c["cold"] for c in cmp]))})
    summary = decision_summary(rows, low, high, args.min_distance)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "decision_map.json"), "w", encoding="utf-8") as stream:
        json.dump({"bounds": args.bounds, "settings": settings, "arc": [low, high], "grid": grid, "rows": rows,
                   "summary": {k: v for k, v in summary.items() if k != "dead_zone_by_state"},
                   "dead_zone_by_state": {str(k): v for k, v in summary["dead_zone_by_state"].items()}}, stream, indent=1)
    print("decision map %s %s: %d pairs in %.0f s | wrong sign %d, silent beyond %g indices %d, dead zone median %.1f max %.1f"
          % (args.bounds, settings or "", summary["pairs"], time.monotonic() - started, summary["wrong_sign"],
             args.min_distance, summary["silent_beyond_min_distance"], summary["dead_zone_median"], summary["dead_zone_max"]))
    try:
        heatmap(os.path.join(args.out, "decision_map.png"), grid, rows, "%s ring" % args.bounds)
    except ImportError:
        pass
    return 0


def blind_band(offsets, drives, floor=1.0):
    """``(width, centre)`` of the goal offsets around 0 without drive toward the goal.

    An offset is blind when the drive is under ``floor`` spikes per tick or
    points away from the goal; the band is the contiguous run of blind offsets
    that contains the smallest offsets.
    """

    blind = {int(o): (abs(d) < floor or np.sign(d) != np.sign(o)) for o, d in zip(offsets, drives) if o != 0}
    low = 0
    while blind.get(low - 1, False):
        low -= 1
    high = 0
    while blind.get(high + 1, False):
        high += 1
    return high - low, 0.5 * (high + low)


def fine_map(args):
    settings = parse_settings(args.set)
    low, high = bounded_settings(RING_SIZE, settings)["limit_arc"] if args.bounds == "bounded" else (5, 194)
    states = [int(round(v)) for v in np.linspace(low + args.reach + 5, high - args.reach - 5, 5)]
    offsets = list(range(-args.reach, args.reach + 1))
    rows, started = [], time.monotonic()
    for state in states:
        drives = []
        for offset in offsets:
            graph = two_ring_single_joint(joint=5, calibration="legacy", bounds=args.bounds,
                                          bounded=settings if args.bounds == "bounded" else None)
            if args.sharp:
                graph.blocks["gain"].params["cross_inhibition_weight"] = 0.0
                graph.connect(graph.blocks["r2"].spikes, graph.blocks["gain"].reference, **FEEDBACK["goal_reference"])
            record = frozen_trial(graph, state, state + offset, SETTLE_TICKS + 18, args.engines)
            ticks = [t for t in record.main_ticks[SETTLE_TICKS:] if t.inputs.get("gain") is not None]
            drives.append(float(np.mean([t.inputs["gain"]["right_counts"] - t.inputs["gain"]["left_counts"] for t in ticks])))
        width, centre = blind_band(offsets, drives)
        rows.append({"state": state, "offsets": offsets, "drive": drives, "blind_width": width, "blind_centre": centre})
        print("state %3d: blind band %2d indices wide, centred %+.1f | drive at -5 %+.1f, -2 %+.1f, -1 %+.1f, +1 %+.1f, +2 %+.1f, +5 %+.1f"
              % ((state, width, centre) + tuple(drives[offsets.index(o)] for o in (-5, -2, -1, 1, 2, 5))))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "fine_map.json"), "w", encoding="utf-8") as stream:
        json.dump({"bounds": args.bounds, "settings": settings, "sharp": bool(args.sharp), "rows": rows}, stream, indent=1)
    print("fine map %s%s: blind band median %.0f, max %d indices, %.0f s"
          % (args.bounds, ", with the reference projection" if args.sharp else "", np.median([r["blind_width"] for r in rows]),
             max(r["blind_width"] for r in rows), time.monotonic() - started))
    return 0


def drift(args):
    settings = parse_settings(args.set)
    low, high = bounded_settings(RING_SIZE, settings)["limit_arc"] if args.bounds == "bounded" else (5, 194)
    positions = [int(round(v)) for v in np.linspace(low + 15, high - 15, args.positions)]
    ticks = int(round(args.seconds / 0.05))
    rows = []
    for seed in args.seeds:
        for position in positions:
            graph = two_ring_single_joint(joint=5, calibration="legacy", bounds=args.bounds, seed=seed,
                                          bounded=settings if args.bounds == "bounded" else None)
            record = frozen_trial(graph, position, position, ticks, args.engines)
            centroid = np.array([t.inputs["r1"].data["centroid_circular"] if t.inputs.get("r1") is not None else np.nan
                                 for t in record.main_ticks], dtype=float)
            good = centroid[np.isfinite(centroid)]
            rows.append({"seed": seed, "position": position, "drift": float(np.mean(good[-10:]) - np.mean(good[SETTLE_TICKS:SETTLE_TICKS + 10]))
                         if len(good) > SETTLE_TICKS + 20 else float("nan")})
        values = [r["drift"] for r in rows if r["seed"] == seed]
        print("seed %d: drift over %.0f s at %s = %s indices" % (seed, args.seconds, positions, ["%+.1f" % v for v in values]))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "drift.json"), "w", encoding="utf-8") as stream:
        json.dump({"bounds": args.bounds, "settings": settings, "seconds": args.seconds, "rows": rows}, stream, indent=1)
    values = np.abs([r["drift"] for r in rows])
    print("drift %s: median %.1f, max %.1f indices in %.0f s" % (args.bounds, np.nanmedian(values), np.nanmax(values), args.seconds))
    return 0


def heatmap(path, grid, rows, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.array([[next(r["drive"] for r in rows if r["state"] == s and r["goal"] == g) for g in grid] for s in grid])
    bound = max(float(np.max(np.abs(values))), 1.0)
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(values, origin="lower", cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto",
                        extent=[grid[0], grid[-1], grid[0], grid[-1]])
    axis.plot([grid[0], grid[-1]], [grid[0], grid[-1]], color="#52514e", linewidth=1)
    axis.set_xlabel("goal ring index")
    axis.set_ylabel("state ring index")
    axis.set_title("Comparator drive, %s (red: move up, blue: move down)" % title, loc="left", fontsize=10)
    figure.colorbar(image, ax=axis, label="right - left gain spikes per tick")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--engines", choices=("nest", "fake"), default="nest")
    p.add_argument("--window-ticks", type=int, default=24)
    p.add_argument("--out", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--library", required=True)
    p.add_argument("--rise-fraction", type=float, default=0.9, help="fraction of the ring the target rises across")
    p.add_argument("--arc-fraction", type=float, default=0.70, help="joint arc, for the monotonicity report")
    p.add_argument("--ridge-lambda", type=float, default=1e-6)
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--install", action="store_true")
    p = sub.add_parser("map")
    p.add_argument("--bounds", choices=("legacy", "bounded"), default="legacy")
    p.add_argument("--set", action="append", help="bounded-ring setting key=value (see graph.templates.BOUNDED)")
    p.add_argument("--grid", type=int, default=16, help="state and goal indices per axis")
    p.add_argument("--min-distance", type=float, default=10.0)
    p.add_argument("--engines", choices=("nest", "fake"), default="nest")
    p.add_argument("--fine", action="store_true", help="one-index goal steps around five states: the blind band")
    p.add_argument("--reach", type=int, default=12, help="--fine: largest goal offset, indices")
    p.add_argument("--sharp", action="store_true", help="--fine: add the goal ring's projection onto the gain pair (v4)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("drift")
    p.add_argument("--bounds", choices=("legacy", "bounded"), default="bounded")
    p.add_argument("--set", action="append", help="bounded-ring setting key=value (see graph.templates.BOUNDED)")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--positions", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[13579, 7, 99])
    p.add_argument("--engines", choices=("nest", "fake"), default="nest")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "map" and args.fine:
        return fine_map(args)
    return {"collect": collect, "fit": fit, "map": decision_map, "drift": drift}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
