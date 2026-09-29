#!/usr/bin/env python3
"""Phase 5 feedback sweep (plan 5c): Encoder modes once / continuous / corrective.

Runs the two-ring single-joint graph with the fake robot (or Gazebo with
``--engines full``) and real NEST, varying the state encoder's mode, rate ``r``,
half width ``h`` and dead band, and reports per configuration:

* final |error| and settle steps (mean over goals),
* tracking error: mean |r1 centroid − ring index of the measured joint| over the
  main ticks (how well the belief follows the joint),
* bump-to-joint lag: the tick lag maximising the cross-correlation between the
  r1 centroid and the mapped joint index (positive = bump leads the joint).

    python3 scripts/sweep_feedback.py --out docs/feedback_sweep.json [--engines nest|full] [--goals 0.6 -0.5]
"""

import argparse
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

from tiago_ring_controller.graph import run_graph, two_ring_single_joint  # noqa: E402


def configurations(quick=False):
    yield {"mode": "once", "rate_hz": 200.0, "half_width": 5, "dead_band": 0}
    rates = (50.0, 200.0) if quick else (25.0, 50.0, 100.0, 200.0, 400.0)
    widths = (5,) if quick else (2, 5, 10)
    for rate, width in itertools.product(rates, widths):
        yield {"mode": "continuous", "rate_hz": rate, "half_width": width, "dead_band": 0}
    for band in ((2,) if quick else (0, 2, 5, 10)):
        yield {"mode": "corrective", "rate_hz": 200.0, "half_width": 5, "dead_band": band}


def analyse(record, encoder):
    ticks = record.main_ticks
    centroids, indices = [], []
    for tick in ticks:
        counts = tick.outputs.get("ring_counts")
        joint = tick.inputs.get("joint_state")
        if counts is None or joint is None:
            continue
        centroid = counts.get("r1_centroid")
        if centroid is None or not np.isfinite(centroid):
            continue
        centroids.append(float(centroid))
        indices.append(float(encoder.ring_index(float(joint["positions"][encoder_joint(encoder)]))))
    centroids, indices = np.asarray(centroids), np.asarray(indices)
    if len(centroids) < 4:
        return {"tracking_error": float("nan"), "lag_ticks": None, "samples": int(len(centroids))}
    tracking = float(np.mean(np.abs(centroids - indices)))
    c = centroids - centroids.mean()
    j = indices - indices.mean()
    best_lag, best = 0, -np.inf
    for lag in range(-20, 21):
        if lag >= 0:
            a, b = c[lag:], j[: len(j) - lag]
        else:
            a, b = c[: len(c) + lag], j[-lag:]
        if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
            continue
        value = float(np.corrcoef(a, b)[0, 1])
        if value > best:
            best, best_lag = value, lag
    return {"tracking_error": tracking, "lag_ticks": best_lag, "lag_correlation": None if best == -np.inf else best,
            "samples": int(len(centroids))}


def encoder_joint(encoder):
    return encoder._joint_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", choices=("nest", "full"), default="nest")
    parser.add_argument("--joint", type=int, default=5)
    parser.add_argument("--goals", type=float, nargs="+", default=[0.6, -0.5])
    parser.add_argument("--seed", type=int, default=13579)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown", default=None, help="also write a markdown table")
    args = parser.parse_args(argv)

    rows = []
    started = time.monotonic()
    for config in configurations(args.quick):
        graph = two_ring_single_joint(joint=args.joint, seed=args.seed)
        encoder = graph.blocks["enc_state"]
        for key, value in config.items():
            encoder.params[key] = value
        encoder._joint_index = args.joint
        trial_rows = []
        for goal in args.goals:
            result = run_graph(graph, engines=args.engines, goals=[goal])[0]
            record, scalars = result["record"], result["legacy"]["scalars"]
            metrics = analyse(record, encoder)
            trial_rows.append({
                "goal": goal, "abs_error": scalars["abs_position_error_rad"], "n_steps": scalars["n_steps"],
                "stop_reason": scalars["stop_reason"], **metrics,
            })
        row = dict(config)
        row["trials"] = trial_rows
        row["abs_error_mean"] = float(np.mean([t["abs_error"] for t in trial_rows]))
        row["n_steps_mean"] = float(np.mean([t["n_steps"] for t in trial_rows]))
        row["tracking_error_mean"] = float(np.nanmean([t["tracking_error"] for t in trial_rows]))
        lags = [t["lag_ticks"] for t in trial_rows if t["lag_ticks"] is not None]
        row["lag_ticks_mean"] = None if not lags else float(np.mean(lags))
        rows.append(row)
        print("%-10s r=%6.1f h=%2d band=%2d | err %.4f steps %5.1f tracking %.2f lag %s"
              % (config["mode"], config["rate_hz"], config["half_width"], config["dead_band"],
                 row["abs_error_mean"], row["n_steps_mean"], row["tracking_error_mean"], row["lag_ticks_mean"]))
    report = {"engines": args.engines, "joint": args.joint, "goals": args.goals, "seed": args.seed,
              "wall_s": time.monotonic() - started, "rows": rows}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=1)
        print("written:", args.out)
    if args.markdown:
        lines = ["| mode | r (Hz) | h | dead band | mean abs error (rad) | mean steps | tracking error (idx) | lag (ticks) |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append("| %s | %g | %d | %d | %.4f | %.1f | %.2f | %s |" % (
                row["mode"], row["rate_hz"], row["half_width"], row["dead_band"], row["abs_error_mean"],
                row["n_steps_mean"], row["tracking_error_mean"], "" if row["lag_ticks_mean"] is None else "%.1f" % row["lag_ticks_mean"]))
        with open(args.markdown, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        print("written:", args.markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
