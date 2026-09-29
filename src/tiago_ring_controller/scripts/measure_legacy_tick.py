#!/usr/bin/env python3
"""Phase 0: measure the legacy loop's real tick period (before number).

Reproduces one collector-style tick — ``rospy.sleep(0.05)``, the ~600
recorder reads and ``nest.Simulate(50)`` — for ``--ticks`` iterations with
the unchanged ``SingleRingModel``, and prints the wall-clock and simulated
(``/clock``) period statistics and the NEST-vs-sim-time drift.  Run inside
the container with the simulation launched; no trajectory is published, so
the robot does not move.

Save the output under ``docs/cosim/baseline/``.
"""

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13579)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    import nest
    import numpy as np
    import rospy

    from single_ring import SingleRingModel

    rospy.init_node("measure_legacy_tick", anonymous=True)
    use_sim_time = rospy.get_param("/use_sim_time", False)
    print("/use_sim_time:", use_sim_time)

    os.chdir(SRC)
    model = SingleRingModel(seed=args.seed, local_num_threads=1)
    model.r2._inject_bump(140, 5)
    model.r1._inject_bump(60, 5)

    def total(recorders):
        return sum(len(s) for s in model._collect_spikes(recorders))

    def r1_counts():
        return np.array(
            [len(s) for s in model._collect_spikes(model.r1.ring_attractor.ring_spike_recorders)]
        )

    wall_periods, sim_periods, simulate_wall = [], [], []
    nest_ms = 0.0
    last_wall = time.monotonic()
    last_sim = rospy.Time.now().to_sec()
    sim_start = last_sim
    for _ in range(args.ticks):
        rospy.sleep(0.05)
        total(model.gain_modulation.left_gain_spike_recorders)
        total(model.gain_modulation.right_gain_spike_recorders)
        r1_counts()
        started = time.monotonic()
        nest.Simulate(50.0)
        simulate_wall.append(time.monotonic() - started)
        nest_ms += 50.0
        total(model.gain_modulation.left_gain_spike_recorders)
        total(model.gain_modulation.right_gain_spike_recorders)
        r1_counts()
        now_wall = time.monotonic()
        now_sim = rospy.Time.now().to_sec()
        wall_periods.append(now_wall - last_wall)
        sim_periods.append(now_sim - last_sim)
        last_wall, last_sim = now_wall, now_sim

    def stats(values):
        values = sorted(values)
        return {"mean_ms": 1000.0 * statistics.mean(values),
                "p95_ms": 1000.0 * values[int(0.95 * (len(values) - 1))],
                "max_ms": 1000.0 * values[-1]}

    report = {
        "ticks": args.ticks,
        "use_sim_time": bool(use_sim_time),
        "wall_period": stats(wall_periods),
        "sim_period": stats(sim_periods),
        "simulate_wall": stats(simulate_wall),
        "nest_time_ms": nest_ms,
        "sim_time_elapsed_ms": 1000.0 * (last_sim - sim_start),
        "drift_ms": 1000.0 * (last_sim - sim_start) - nest_ms,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
