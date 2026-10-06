#!/usr/bin/env python3
"""The three-ring controller (v4) in spikes: who fires, how often, and how one move looks spike by spike.

    docker/batch.bash nest --out /tmp/ring-so2 -- 'python3 -B scripts/spike_census.py --out /out/spikes'

One move of one joint on NEST with the simulated joint (no Gazebo).  Every neuron of the
controller has its own spike recorder, so all spikes of the move are kept; a few neurons
also get a multimeter (membrane potential and synaptic currents).  Written to ``--out``:

* ``census.json`` and ``census.md``: for every population, how many neurons fire, at what
  rates, how many spikes a NEST step carries and how regular the spike trains are, while the
  joint moves and at rest at the goal;
* ``spikes_neurons.png``: a ring neuron inside and outside the bump, a gain neuron and the two
  decision neurons: potential, input and spikes;
* ``spikes_move.png``: the whole move as spikes: the three rings, the two gain pairs, the
  decision neurons, and the joint.

For part III of TECHNICAL.md; the copies the page shows are in docs/spikes/.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tiago_ring_controller.cosim.loop import LoopObserver  # noqa: E402
from tiago_ring_controller.graph import TEMPLATES, compile_graph  # noqa: E402
from tiago_ring_controller.graph.run import run_graph_trial  # noqa: E402
from tiago_ring_controller.nest.kernel import recorder_events  # noqa: E402

INK, GREY, LIGHT = "#1c2330", "#8a94a3", "#d9dee6"
BLUE, GREEN, ORANGE, PURPLE, RED = "#2a78d6", "#1baf7a", "#eb6834", "#7559c7", "#d9443a"
STEP_MS = 50.0


class Watch(LoopObserver):
    """Attaches multimeters when the network is built and keeps what every tick reports."""

    def __init__(self, backend, graph, watched):
        self.backend, self.graph, self.watched = backend, graph, watched
        self.meters = {}
        self.ticks = []
        self.spikes = {}
        self.trial_start_ms = 0.0
        self.engine = None

    def on_reset(self, loop):
        self.engine = next(engine for engine in loop.engines if getattr(engine, "built", None) is not None)
        self.trial_start_ms = float(self.engine.trial_start_ms)
        for label, (block_id, side, index) in self.watched.items():
            block = self.graph.blocks[block_id]
            population = getattr(block.populations, side) if side in ("left", "right") else block.population
            neuron = population.node(side) if side in ("warm", "cold") else population.neuron(index)
            meter = self.backend.Create("multimeter", params={"interval": 0.5, "record_from": ["V_m", "I_syn_ex", "I_syn_in"]})
            self.backend.Connect(meter, neuron)
            self.meters[label] = meter

    def on_tick(self, record):
        if record.phase != "main":
            return
        get = lambda pack, key: None if pack is None else pack.get(key)  # noqa: E731
        inputs, outputs = record.inputs, record.outputs
        joint = next((pack for name, pack in outputs.items() if name.startswith("j") and pack.get("angle") is not None and name != "goal"), None)
        self.ticks.append({
            "t_ms": float(record.t_ms), "angle": get(joint, "angle"), "goal": get(outputs.get("goal"), "angle"),
            "nest_ms": get(inputs.get("jt__P"), "t_nest_ms"),
            "G": get(inputs.get("jt__G"), "centroid_circular"), "P": get(inputs.get("jt__P"), "centroid_circular"),
            "S": get(inputs.get("jt__S"), "centroid_circular"),
            "goal_left": get(inputs.get("jt__ct_goal__gain"), "left_counts"), "goal_right": get(inputs.get("jt__ct_goal__gain"), "right_counts"),
            "body_left": get(inputs.get("jt__ct_body__gain"), "left_counts"), "body_right": get(inputs.get("jt__ct_body__gain"), "right_counts"),
            "goal_decision": get(inputs.get("jt__ct_goal__cmp"), "counts"), "body_decision": get(inputs.get("jt__ct_body__cmp"), "counts"),
            "velocity": (get(outputs.get("dec"), "velocity") or [None])[0],
        })

    def on_trial_end(self, trial):
        def gather(population):
            index, times = [], []
            for i, recorder in enumerate(population.recorder_list()):
                t = np.asarray(recorder_events(self.backend, recorder)["times"], dtype=float)
                times.append(t)
                index.append(np.full(t.size, i, dtype=int))
            return np.concatenate(index), np.concatenate(times) - self.trial_start_ms
        for block in self.graph.neural_blocks():
            if getattr(block, "population", None) is not None:
                self.spikes[block.id] = gather(block.population)
            elif getattr(block, "populations", None) is not None:
                self.spikes[block.id + ".left"] = gather(block.populations.left)
                self.spikes[block.id + ".right"] = gather(block.populations.right)
        self.traces = {}
        for label, meter in self.meters.items():
            events = self.backend.GetStatus(meter, "events")[0]
            self.traces[label] = {key: np.asarray(events[key], dtype=float) for key in ("times", "V_m", "I_syn_ex", "I_syn_in")}
            self.traces[label]["times"] = self.traces[label]["times"] - self.trial_start_ms


def train_stats(index, times, n, start_ms, end_ms):
    keep = (times >= start_ms) & (times < end_ms)
    index, times = index[keep], times[keep]
    window_s = (end_ms - start_ms) / 1000.0
    counts = np.bincount(index, minlength=n)[:n].astype(float)
    rates = counts / window_s
    active = rates >= 1.0
    cvs = []
    for i in np.flatnonzero(counts >= 6):
        gaps = np.diff(np.sort(times[index == i]))
        cvs.append(float(np.std(gaps) / np.mean(gaps)))
    return {"neurons": int(n), "fire": int(active.sum()), "rate_of_firing_hz": float(rates[active].mean()) if active.any() else 0.0,
            "rate_max_hz": float(rates.max()), "spikes_per_step": float(counts.sum() / (window_s * 1000.0 / STEP_MS)),
            "isi_cv": float(np.median(cvs)) if cvs else None, "rates_hz": rates.tolist()}


POPULATIONS = (
    ("jt__G", "goal ring G", 200), ("jt__P", "predictive ring P", 200), ("jt__S", "state ring S", 200),
    ("jt__fG", "Fourier readout of G", 40), ("jt__fP", "Fourier readout of P", 40), ("jt__fS", "Fourier readout of S", 40),
    ("jt__ct_goal__cmp", "goal comparator: warm, cold, left, right", 4), ("jt__ct_body__cmp", "body comparator: warm, cold, left, right", 4),
    ("jt__ct_goal__gain.right", "goal pair, right", 200), ("jt__ct_goal__gain.left", "goal pair, left", 200),
    ("jt__ct_body__gain.right", "body pair, right", 200), ("jt__ct_body__gain.left", "body pair, left", 200),
)


def census(watch, moving, resting):
    out = {"windows_ms": {"moving": list(moving), "at rest": list(resting)}, "populations": {}}
    for key, label, n in POPULATIONS:
        index, times = watch.spikes[key]
        out["populations"][key] = {"label": label, "moving": train_stats(index, times, n, *moving), "at rest": train_stats(index, times, n, *resting)}
    return out


def census_table(result, meta):
    lines = ["# The v4 controller in spikes: census", "", "Written by `scripts/spike_census.py`. Do not edit by hand.", "",
             "One move of %s from %.2f to %.2f rad on NEST with the simulated joint, %d robot ticks, %d NEST steps of 50 ms per tick. "
             "\"Moving\": network time %.1f to %.1f s; \"at rest\": %.1f to %.1f s. \"Fire\": at least one spike per second. "
             "\"Regularity\": standard deviation over mean of the gaps between a neuron's spikes (0: a clock; 1: random), median over the "
             "neurons with six spikes or more. Rates are per second of network time." % (
                 meta["joint"], meta["start"], meta["goal"], meta["ticks"], meta["time_scale"],
                 result["windows_ms"]["moving"][0] / 1000.0, result["windows_ms"]["moving"][1] / 1000.0,
                 result["windows_ms"]["at rest"][0] / 1000.0, result["windows_ms"]["at rest"][1] / 1000.0), ""]
    for window in ("moving", "at rest"):
        lines += ["**%s**" % ("While the joint moves" if window == "moving" else "At rest at the goal"), "",
                  "| Population | Neurons | Fire | Rate of those that fire | Highest rate | Spikes per NEST step | Regularity |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for key, row in result["populations"].items():
            s = row[window]
            lines.append("| %s | %d | %d | %.0f Hz | %.0f Hz | %.1f | %s |" % (
                row["label"], s["neurons"], s["fire"], s["rate_of_firing_hz"], s["rate_max_hz"], s["spikes_per_step"],
                "–" if s["isi_cv"] is None else "%.2f" % s["isi_cv"]))
        lines.append("")
    lines += ["The four neurons of each comparator, rate in Hz:", "", "| Comparator | Window | warm | cold | left | right |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for key, name in (("jt__ct_goal__cmp", "goal (P against G)"), ("jt__ct_body__cmp", "body (P against S)")):
        for window in ("moving", "at rest"):
            rates = result["populations"][key][window]["rates_hz"]
            lines.append("| %s | %s | %.0f | %.0f | %.0f | %.0f |" % (name, window, rates[0], rates[1], rates[2], rates[3]))
    return lines


def figure_neurons(plt, watch, path, window):
    lo, hi = window
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 9.2), sharex=True, gridspec_kw={"hspace": 0.3})

    def trace(label, key="V_m"):
        data = watch.traces[label]
        keep = (data["times"] >= lo) & (data["times"] <= hi)
        return (data["times"][keep] - lo), data[key][keep]

    def spikes_of(block, index):
        idx, times = watch.spikes[block]
        chosen = times[(idx == index) & (times >= lo) & (times <= hi)]
        return chosen - lo

    t, v = trace("ring inside")
    axes[0].plot(t, v, color=INK, lw=1.0, label="a neuron inside the bump of the predictive ring")
    t, v = trace("ring outside")
    axes[0].plot(t, v, color=GREY, lw=1.0, label="a neuron next to the bump")
    axes[0].axhline(0.8, color=RED, lw=0.9, ls="--")
    axes[0].text(2, 0.815, "threshold 0.8 mV", fontsize=8, color=RED)
    axes[0].set_ylabel("membrane\npotential (mV)")
    axes[0].set_ylim(-0.1, 0.95)
    axes[0].set_title("1. Ring neurons: the constant current holds every neuron just under its threshold; a little more makes it fire, a little less keeps it silent")
    axes[0].legend(fontsize=8, loc="lower right", ncol=2)

    for label, color, name in (("ring inside", INK, "inside the bump"), ("ring outside", GREY, "next to the bump")):
        t, ex = trace(label, "I_syn_ex")
        _, inh = trace(label, "I_syn_in")
        axes[1].plot(t, ex + inh, color=color, lw=1.0, label=name)
    axes[1].axhline(0.0, color=LIGHT, lw=1)
    axes[1].set_ylabel("current from\nspikes (pA)")
    axes[1].set_title("2. What decides: the current that arriving spikes add to the constant 450,000 pA")
    axes[1].legend(fontsize=8, loc="lower right", ncol=2)

    t, v = trace("gain right")
    axes[2].plot(t, v, color=ORANGE, lw=1.0, label="right neuron under the bump (the decision is \"right\")")
    t, v = trace("gain left")
    axes[2].plot(t, v, color=PURPLE, lw=1.0, label="left neuron under the bump")
    axes[2].axhline(1.35, color=RED, lw=0.9, ls="--")
    axes[2].text(2, 1.38, "threshold 1.35 mV", fontsize=8, color=RED)
    axes[2].set_ylabel("membrane\npotential (mV)")
    axes[2].set_title("3. Gain neurons of the goal pair: the ring's spikes alone do not reach the threshold; with the decision neuron's spikes they do")
    axes[2].legend(fontsize=8, loc="lower right", ncol=2)

    for row, (label, color) in enumerate((("warm", ORANGE), ("cold", BLUE))):
        t, v = trace(label)
        axes[3].plot(t, v + 1.2 * row, color=color, lw=0.8)
    for row, (side, color) in enumerate((("right", GREEN), ("left", PURPLE))):
        times = spikes_of("jt__ct_goal__cmp", 3 if side == "right" else 2)
        axes[3].scatter(times, np.full(times.size, 2.6 + 0.35 * row), s=5, color=color, linewidths=0)
    axes[3].set_yticks([0.4, 1.6, 2.6, 2.95])
    axes[3].set_yticklabels(["warm (P)", "cold (G)", "right: spikes", "left: spikes"])
    axes[3].set_ylim(-0.2, 3.25)
    axes[3].set_xlabel("network time (ms)")
    axes[3].set_title("4. Goal comparator: the ramp neurons warm and cold (potentials) and the spikes of the two decision neurons; the goal lies up the ring, so right wins")
    axes[3].set_xlim(0, hi - lo)
    fig.savefig(path)
    plt.close(fig)


def figure_move(plt, watch, path, end_ms):
    fig, axes = plt.subplots(5, 1, figsize=(10.5, 11.5), sharex=True, gridspec_kw={"height_ratios": [1.5, 1.0, 1.0, 0.9, 0.9], "hspace": 0.3})
    for key, color, label in (("jt__G", PURPLE, "goal ring G"), ("jt__S", BLUE, "state ring S"), ("jt__P", INK, "predictive ring P")):
        index, times = watch.spikes[key]
        keep = times <= end_ms
        axes[0].scatter(times[keep] / 1000.0, index[keep], s=1.2, color=color, linewidths=0, label=label, alpha=0.8)
    axes[0].set_ylabel("ring neuron")
    low = min(t["P"] for t in watch.ticks if t["P"] is not None) - 25
    high = max(t["G"] for t in watch.ticks if t["G"] is not None) + 25
    axes[0].set_ylim(low, high)
    axes[0].set_title("1. The three rings: P starts on S, the goal pair pushes it to G, and S follows as the joint moves")
    axes[0].legend(fontsize=8, loc="lower right", markerscale=6, ncol=3)
    for row, (pair, title) in enumerate((("jt__ct_goal__gain", "2. Goal pair (its spikes move P and are the motor signal): right in orange, left in purple"),
                                        ("jt__ct_body__gain", "3. Body pair (its spikes pull P back to S and command nothing)"))):
        for side, color in (("right", ORANGE), ("left", PURPLE)):
            index, times = watch.spikes["%s.%s" % (pair, side)]
            keep = times <= end_ms
            axes[1 + row].scatter(times[keep] / 1000.0, index[keep], s=1.2, color=color, linewidths=0)
        axes[1 + row].set_ylim(low, high)
        axes[1 + row].set_ylabel("gain neuron")
        axes[1 + row].set_title(title)
    step_s = np.array([t["nest_ms"] for t in watch.ticks]) / 1000.0
    for key, color, style, label in (("goal_decision", GREEN, "-", "goal comparator"), ("body_decision", GREY, "-", "body comparator")):
        right = np.array([t[key]["right"] for t in watch.ticks], dtype=float)
        left = np.array([t[key]["left"] for t in watch.ticks], dtype=float)
        axes[3].plot(step_s, right - left, color=color, lw=1.4, ls=style, label=label + ": right minus left")
    axes[3].axhline(0.0, color=LIGHT, lw=1)
    axes[3].set_ylabel("decision spikes\nper robot tick")
    axes[3].set_title("4. The decision neurons")
    axes[3].legend(fontsize=8, loc="upper right")
    axes[4].plot(step_s, [t["goal"] for t in watch.ticks], color=PURPLE, lw=1.2, ls="--", label="goal")
    axes[4].plot(step_s, [t["angle"] for t in watch.ticks], color=INK, lw=1.8, label="joint angle (rad)")
    axes[4].plot(step_s, [t["velocity"] for t in watch.ticks], color=ORANGE, lw=1.2, label="commanded velocity (rad/s)")
    axes[4].set_ylabel("joint")
    axes[4].set_xlabel("network time (s); the robot's clock runs 3 times slower")
    axes[4].set_title("5. The joint: the decoder counts the goal pair's spikes of each tick and turns them into a velocity")
    axes[4].legend(fontsize=8, loc="center right")
    axes[4].set_xlim(0, end_ms / 1000.0)
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--joint", type=int, default=5, help="arm joint index 0-6 (default 5: arm_6)")
    parser.add_argument("--goal", type=float, default=0.6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import nest
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
                         "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY, "text.color": INK,
                         "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 9.5,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    os.makedirs(args.out, exist_ok=True)
    graph = TEMPLATES["three_ring_single_joint"](joint=args.joint, goal_rad=args.goal)
    watched = {
        "ring inside": ("jt__P", None, 108), "ring outside": ("jt__P", None, 92),
        "gain right": ("jt__ct_goal__gain", "right", 108), "gain left": ("jt__ct_goal__gain", "left", 108),
        "warm": ("jt__ct_goal__cmp", "warm", 0), "cold": ("jt__ct_goal__cmp", "cold", 1),
    }
    watch = Watch(nest, graph, watched)
    compiled = compile_graph(graph, engines="nest", backend=nest, observers=[watch])
    compiled.loop.initialize()
    record = run_graph_trial(compiled, float(args.goal))
    time_scale = int(graph.simulation.time_scale)
    ticks = watch.ticks
    angles = np.array([t["angle"] for t in ticks], dtype=float)
    nest_ms = np.array([t["nest_ms"] for t in ticks], dtype=float)
    arrived = int(np.argmax(np.abs(angles - args.goal) < 0.05)) if np.any(np.abs(angles - args.goal) < 0.05) else len(ticks) - 1
    moving = (300.0, float(nest_ms[max(arrived, 4)]))
    end_ms = float(nest_ms[-1])
    resting = (end_ms - 3000.0, end_ms)
    meta = {"joint": "arm_%d" % (args.joint + 1), "start": float(angles[0]), "goal": float(args.goal), "final": float(angles[-1]),
            "ticks": len(ticks), "time_scale": time_scale, "stop": getattr(record, "stop_reason", None),
            "arrived_tick": arrived, "network_time_s": end_ms / 1000.0,
            "parameters": {block.id: {k: v for k, v in block.params.items() if isinstance(v, (int, float, str, bool))}
                           for block in graph.neural_blocks()}}
    result = census(watch, moving, resting)
    result["meta"] = meta
    result["ticks"] = ticks
    with open(os.path.join(args.out, "census.json"), "w", encoding="utf-8") as stream:
        json.dump(result, stream, separators=(",", ":"), default=float)
    with open(os.path.join(args.out, "census.md"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(census_table(result, meta)) + "\n")
    figure_neurons(plt, watch, os.path.join(args.out, "spikes_neurons.png"), (700.0, 1000.0))
    figure_move(plt, watch, os.path.join(args.out, "spikes_move.png"), end_ms)
    compiled.loop.shutdown()
    print("%s: %.2f -> %.2f rad (goal %.2f), %d ticks, stop %s; written: %s/{census.json,census.md,spikes_neurons.png,spikes_move.png}" % (
        meta["joint"], meta["start"], meta["final"], meta["goal"], meta["ticks"], meta["stop"], args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
