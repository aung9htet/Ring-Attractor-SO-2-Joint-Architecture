#!/usr/bin/env python3
"""Speed against accuracy of validation runs (docs/archive/velocity/speed.md).

    D=../../docs/archive/velocity/media/speed/data
    python3 -B scripts/speed_report.py --base $D/lever_k1.json k1=$D/lever_k1.json k3=$D/lever_k3.json fb30=$D/lever_fb30.json
    python3 -B scripts/speed_report.py --base $D/gazebo_k1.json 1x=$D/gazebo_k1.json 3x=$D/gazebo_k3.json --overall

Each ``name=path`` is a ``validation.json`` of ``scripts/validate_velocity.py``.  The speed-up is
the median over trials of (time to 90% of the move in ``--base``) / (the same trial's time
here); trials are paired by joint, start, goal and seed.  "Joint vs bump travel" compares the
joint's displacement with the bump's travel from the injected index to where it settles,
which does not depend on the time scale.
"""

import argparse
import json
import sys

import numpy as np

HEADER = ("| setting | joint | trials | to 90% median (s) | speed-up | peak speed max (rad/s) | goal error median / p90 / max (rad) "
          "| joint vs bump travel, median / p90 | ring - joint at end, median / p90 (rad) | overshoot p90 / max (rad) "
          "| largest joint lag, median (idx) | no drive | wrong way | clamp ticks |")


def key(row):
    return (row["joint"], round(row["start_fraction"], 3), row["goal_fraction"], row["seed"])


def line(name, label, rows, base):
    speedups = []
    for row in rows:
        ref = base.get(key(row))
        if ref and np.isfinite(row["time_to_90_s"]) and np.isfinite(ref["time_to_90_s"]) and row["time_to_90_s"] > 0:
            speedups.append(ref["time_to_90_s"] / row["time_to_90_s"])
    error = np.abs([r["goal_error_rad"] for r in rows])
    travel = []
    for r in rows:
        bump = r["goal_index"] + r["endpoint_error"] - r["start_index"]
        if np.isfinite(bump) and abs(bump) > 5:
            travel.append(abs(r["joint_displacement"] / r["rho"] - bump) / abs(bump))
    ring = np.abs([r["state_error_final"] for r in rows if np.isfinite(r["state_error_final"])])
    overshoot = np.array([r["overshoot_rad"] for r in rows])
    t90 = [r["time_to_90_s"] for r in rows if np.isfinite(r["time_to_90_s"])]
    lag = [r["max_lag_index"] for r in rows if np.isfinite(r["max_lag_index"])]
    return "| %s | %s | %d | %.2f | %.2fx | %.2f | %.3f / %.3f / %.3f | %.1f%% / %.1f%% | %.3f / %.3f | %.3f / %.3f | %.1f | %d | %d | %d |" % (
        name, label, len(rows), np.median(t90), np.median(speedups) if speedups else float("nan"), max(r["peak_speed"] for r in rows),
        np.median(error), np.percentile(error, 90), error.max(), 100 * np.median(travel), 100 * np.percentile(travel, 90),
        np.median(ring), np.percentile(ring, 90), np.percentile(overshoot, 90), overshoot.max(), np.median(lag),
        sum(r["no_drive"] for r in rows), sum(r["wrong_direction"] for r in rows), sum(r["clamp_ticks"] for r in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="name=validation.json")
    parser.add_argument("--base", required=True, help="the validation.json speed-ups are measured against")
    parser.add_argument("--overall", action="store_true", help="add a row over all joints per run")
    args = parser.parse_args(argv)
    base = {key(r): r for r in json.load(open(args.base))["trials"]}
    print(HEADER)
    print("|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|")
    for item in args.runs:
        name, _, path = item.partition("=")
        trials = json.load(open(path))["trials"]
        if args.overall:
            print(line(name, "all", trials, base))
        for joint in sorted({r["joint"] for r in trials}):
            print(line(name, "arm_%d" % (joint + 1), [r for r in trials if r["joint"] == joint], base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
