#!/usr/bin/env python3
"""Measure model build time and per-tick readout: legacy facade vs vectorised.

Runs only with real NEST (in the container).  Prints one JSON document with
the numbers that ``docs/blocks/equivalence.md`` cites.

    python3 scripts/benchmark_build.py [--repeats 3] [--ticks 40] [--threads 1]
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
LEGACY = os.path.join(os.path.dirname(HERE), "legacy")
for path in (SRC, LEGACY):
    if path not in sys.path:
        sys.path.insert(0, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13579)
    parser.add_argument("--skip-legacy", action="store_true")
    args = parser.parse_args(argv)

    import numpy as np
    import nest

    from tiago_ring_controller.config import source_config_path
    from tiago_ring_controller.cosim.nest_engine import SpikeCountReader
    from tiago_ring_controller.nest.single_ring import build_single_ring_network

    ring_params = source_config_path("model_params", "ring_params.json")
    weights_dir = source_config_path("ring_decoding_weights")
    report = {"nest_version": nest.__version__, "threads": args.threads, "repeats": args.repeats}

    if not args.skip_legacy:
        from single_ring import SingleRingModel  # legacy facade, frozen

        times = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            model = SingleRingModel(
                ring_params_file=ring_params, weights_dir=weights_dir,
                seed=args.seed, local_num_threads=args.threads,
            )
            times.append(time.perf_counter() - started)
        report["legacy_build_s"] = {"min": round(min(times), 3), "mean": round(sum(times) / len(times), 3)}
        report["legacy_connect_calls"] = int(nest.GetKernelStatus("num_connections"))
        started = time.perf_counter()
        model.r2._inject_bump(140, 5)
        model.r1._inject_bump(60, 5)
        report["legacy_inject_two_bumps_s"] = round(time.perf_counter() - started, 3)

    times = []
    for _ in range(args.repeats):
        network = build_single_ring_network(
            nest, seed=args.seed, local_num_threads=args.threads,
            ring_params_file=ring_params, weights_dir=weights_dir,
        )
        times.append(network.build_seconds)
    report["vectorised_build_s"] = {"min": round(min(times), 4), "mean": round(sum(times) / len(times), 4)}
    report["vectorised_connections"] = int(nest.GetKernelStatus("num_connections"))
    started = time.perf_counter()
    network.set_bump("r2", 140, 5)
    network.set_bump("r1", 60, 5)
    report["vectorised_set_two_bumps_s"] = round(time.perf_counter() - started, 4)

    readers = {
        "r1": SpikeCountReader(nest, network.r1_recorders),
        "left": SpikeCountReader(nest, network.left_recorders),
        "right": SpikeCountReader(nest, network.right_recorders),
    }
    nest.Prepare()
    run_s = read_s = 0.0
    for tick in range(args.ticks):
        started = time.perf_counter()
        nest.Run(50.0)
        run_s += time.perf_counter() - started
        if tick == 0:
            network.clear_bumps()
        started = time.perf_counter()
        counts = {key: reader.read() for key, reader in readers.items()}
        read_s += time.perf_counter() - started
    nest.Cleanup()
    report["per_tick_ms"] = {
        "run_50ms": round(1000 * run_s / args.ticks, 3),
        "readout_three_populations": round(1000 * read_s / args.ticks, 3),
        "readout_mode": readers["r1"].mode,
    }
    report["r1_total_spikes_after_%d_ticks" % args.ticks] = float(np.sum(counts["r1"]))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
