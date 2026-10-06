#!/usr/bin/env python3
"""Tables and figures of the comparison of every system (docs/results.md).

    python3 -B scripts/results_report.py --data ../../docs/results/data --out ../../docs/results

Reads what the measurement scripts wrote (the layout is listed in
``docs/results/data/README.md``):

* ``compare/``: every system on the same goals from the reset pose, in Gazebo.
  ``legacy_trials_j<n>.json`` and ``pid_trials_j<n>.json`` come from
  ``scripts/run_legacy_trials.py``; optionally ``p_controller.json`` and
  ``native.json`` (upper references) from ``scripts/run_baselines.py``;
  ``b3.json``, ``f3.json`` (and the archived ``gt.json``, ``b1.json``,
  ``f1.json``) are ``validation.json`` files of ``scripts/validate_velocity.py``; a directory ``b3/`` with its
  ``validation.json`` (what ``--out`` writes) is read as well as ``b3.json``.
* ``repeat/``: the same files for three runs of two goals.
* ``robust/``: v3, v4 and the PID baseline (``<condition>_pid/``) with a wrong motor gain, a push and a held joint.
* ``starts/``: v3 and v4 from three start poses (any number of validation files).
* ``nest/``: v3 and v4 on NEST with the fake robot.
* ``measure/``: the fine decision maps and the drift of ``scripts/fit_comparator.py``.

Writes ``tables.md``, ``figures/*.png`` and ``paper/*.{png,pdf}`` under
``--out``.  Every system is scored by ``tiago_ring_controller.evaluation.motion``
from the joint trace, so the numbers do not depend on who ran the trial.
Missing inputs are skipped and named.
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

from tiago_ring_controller.evaluation.motion import SETTLE_SAMPLES, SETTLE_TOLERANCE  # noqa: E402

#: order and names of the systems; ``short`` is used in tables and figures
SYSTEMS = [
    ("L", "original scripts", "The frozen original ring controller (`legacy/single_joint_data_collector.py`)"),
    ("PID", "PID baseline", "The PID of the draft (`legacy/single_joint_pid_data_collector.py`): three-term PID with fixed gains "
     "kp 2.0, ki 0.1, kd 0.05 for every joint, integral limit 2.0, output limit 0.5 rad/s"),
    ("P", "upper reference: tuned controller", "A controller tuned by grid search for these joints (it came out proportional: kp 8, "
     "60% of the joint's velocity limit), in the same 50 ms loop as the ring controllers (`scripts/run_baselines.py`)"),
    ("N", "upper reference: robot's own controller", "TIAGo's joint trajectory controller given the goal as one trajectory point "
     "(`scripts/run_baselines.py`)"),
    ("G", "v3, legacy ring, hand gains", "Two rings, joint range on indices 5-195, gains of `velocity_calibration.json`"),
    ("T", "v3, legacy ring, transport", "Two rings, legacy ring, decoder gains from the transport calibration"),
    ("B1", "v3, bounded ring, 1x", "Two rings on the bounded ring, one NEST step per robot tick"),
    ("B3", "v3, bounded ring, 3x", "Two rings on the bounded ring, three NEST steps per robot tick (the v3 default)"),
    ("F1", "v4, 1x", "Three rings with body feedback, one NEST step per robot tick"),
    ("F3", "v4, 3x", "Three rings with body feedback, three NEST steps per robot tick (the v4 default)"),
]
NAMES = {key: short for key, short, _ in SYSTEMS}
#: controllers tuned for these joints: in the tables as the ceiling, not in the figures
UPPER_REFERENCES = ("P", "N")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
#: fixed colour per system in every figure (reference categorical slots 1-5, in the order the systems were added)
COLORS = {"B3": "#2a78d6", "F3": "#eb6834", "PID": "#1baf7a", "P": "#eda100", "N": "#e87ba4"}
#: the systems drawn in the figures: the PID baseline, v3, v4; the upper references are in the tables
FIGURE_SYSTEMS = ("PID", "B3", "F3")
FIGURE_NAMES = {"PID": "PID baseline", "P": "tuned controller", "N": "robot's own controller",
                "B3": "v3 (two rings)", "F3": "v4 (three rings)"}
CONDITIONS = [("nominal", "nominal"), ("mg07", "motor gain x0.7"), ("mg13", "motor gain x1.3"),
              ("push", "push 0.2 rad"), ("hold", "held 2 s")]


# -- loading ---------------------------------------------------------------
def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def hold_range(trace, seconds=0.5):
    """Range of the joint angle over the last ``seconds`` of the trial (how still it is when the trial ends).

    Half a second, because a v3 trial ends half a second after its drive stops."""

    q = np.asarray(trace.get("q", []), dtype=float)
    samples = int(round(seconds / float(trace.get("dt_s", 0.05))))
    return float(np.ptp(q[-samples - 1:-1])) if q.size > 2 else float("nan")


def normalise(row, system):
    """One trial as the common record, whichever script wrote it."""

    trace = row.get("trace", {})
    legacy = "q_goal" in row
    return {
        "system": system, "joint": int(row["joint"]),
        "goal": float(row["q_goal"] if legacy else row["goal_rad"]),
        "start": float(row["q_start"] if legacy else row["start_rad"]),
        "final": float(row["q_final"] if legacy else row["final_rad"]),
        "error": float(row["goal_error_rad"]),
        "t90": float(row.get("time_to_90_s", np.nan)), "settle": float(row.get("settling_time_s", np.nan)),
        "reached": bool(row.get("reached_tolerance", False)), "overshoot": float(row.get("overshoot_rad", np.nan)),
        "peak": float(row.get("peak_speed", np.nan)),
        "no_drive": bool(row.get("no_drive", False)), "wrong": bool(row.get("wrong_direction", False)),
        # rows of the P controller written before its own output limit stopped being counted as a clamp
        "clamp": bool(row.get("clamp_ticks", 0)) and row.get("system") != "P", "lost": bool(row.get("lost_bump", False)),
        "ring_error": float(row.get("state_error_abs_median", np.nan)),
        "wall": float(row.get("wall_s", np.nan)), "tick_ms": float(row.get("tick_wall_ms", np.nan)),
        "steps": int(row.get("n_steps", 0)), "hold": hold_range(trace) if trace else float("nan"),
        "rho": float(row.get("rho", np.nan)), "repeat_key": (int(row["joint"]), round(float(row["q_goal"] if legacy else row["goal_rad"]), 3)),
        "start_fraction": row.get("start_fraction"), "trace": trace,
        "base_shift": float(row.get("base_shift_m", np.nan)),
        "body_gap": float(row.get("body_gap_index", np.nan)), "goal_gap": float(row.get("goal_gap_index", np.nan)),
        "goal_bump": float(row.get("goal_bump_error_index", np.nan)),
    }


def load_validation(path, rename=None):
    """Trials of a ``validation.json`` as ``{system: [records]}``; ``rename`` maps its system letters.

    ``path`` is ``NAME.json``; the directory form ``NAME/validation.json`` (what
    ``validate_velocity.py --out NAME`` writes) is read when the file is absent.
    """

    if not os.path.exists(path):
        path = os.path.join(os.path.splitext(path)[0], "validation.json")
        if not os.path.exists(path):
            return {}, 0
    document = read_json(path)
    result = {}
    for row in document["trials"]:
        system = (rename or {}).get(row["system"], row["system"])
        result.setdefault(system, []).append(normalise(row, system))
    return result, len(document.get("failed", []))


def load_directory(directory):
    """Every validation file under ``directory`` merged into ``{system: [records]}``."""

    merged = {}
    paths = sorted(glob.glob(os.path.join(directory, "*.json")) + glob.glob(os.path.join(directory, "*", "validation.json")))
    for path in paths:
        loaded, _ = load_validation(path)
        for system, rows in loaded.items():
            merged.setdefault(system, []).extend(rows)
    return merged


def load_legacy(directory, prefix, system):
    rows = []
    paths = glob.glob(os.path.join(directory, "%s_trials_j*.json" % prefix)) + glob.glob(os.path.join(directory, "*", "%s_trials_j*.json" % prefix))
    for path in sorted(paths):
        rows.extend(normalise(row, system) for row in read_json(path)["trials"])
    return rows


def load_set(directory):
    """``{system: [records]}`` of one directory in the ``compare/`` layout, plus failed-trial counts."""

    systems, failed, missing = {}, {}, []
    for system, prefix in (("L", "legacy"), ("PID", "pid")):
        rows = load_legacy(directory, prefix, system)
        if rows:
            systems[system] = rows
        else:
            missing.append("%s_trials_j*.json" % prefix)
    for name, rename, optional in (("p_controller.json", None, True), ("native.json", None, True), ("gt.json", None, True),
                                   ("b1.json", {"B": "B1"}, True), ("b3.json", {"B": "B3"}, False),
                                   ("f1.json", {"F": "F1"}, True), ("f3.json", {"F": "F3"}, False)):
        loaded, n_failed = load_validation(os.path.join(directory, name), rename)
        if not loaded and not optional:                # the intermediate variants live in docs/archive
            missing.append(name)
        for system, rows in loaded.items():
            systems[system] = rows
            failed[system] = n_failed
    return systems, failed, missing


# -- statistics --------------------------------------------------------------
def stats(values):
    values = np.abs(np.asarray([v for v in values if np.isfinite(v)], dtype=float))
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, 90)), float(values.max())


def fmt3(values, digits=3):
    """median / p90 / max; one more decimal when the median would otherwise print as zero."""

    triple = stats(values)
    if np.isfinite(triple[0]) and triple[0] < 0.5 * 10.0 ** -digits:
        digits += 1
    return " / ".join(("%." + str(digits) + "f") % v if np.isfinite(v) else "-" for v in triple)


def median(values, digits=2):
    values = [v for v in values if np.isfinite(v)]
    return ("%." + str(digits) + "f") % np.median(values) if values else "-"


def repeat_spread(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["repeat_key"], []).append(r["final"])
    spreads = [max(v) - min(v) for v in groups.values() if len(v) > 1]
    return (float(np.median(spreads)), float(max(spreads)), len(spreads)) if spreads else (float("nan"), float("nan"), 0)


def common_goals(systems, keys):
    """Goals (joint, rounded angle) that every system in ``keys`` ran."""

    sets = [{r["repeat_key"] for r in systems[k]} for k in keys if k in systems]
    return set.intersection(*sets) if sets else set()


def median_interval(values, level=0.95, draws=4000, seed=0):
    """Bootstrap interval of the median of |values| (resampling trials with replacement)."""

    values = np.abs(np.asarray([v for v in values if np.isfinite(v)], dtype=float))
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.median(values[rng.integers(0, len(values), size=(draws, len(values)))], axis=1)
    return tuple(float(v) for v in np.percentile(medians, [50 * (1 - level), 50 * (1 + level)]))


def sign_test(wins, losses):
    """Two-sided sign test: the probability of a split at least this uneven if neither system were better."""

    from fractions import Fraction
    from math import comb

    n = wins + losses
    if n == 0:
        return float("nan")
    tail = Fraction(sum(comb(n, k) for k in range(0, min(wins, losses) + 1)), 2 ** n)      # exact: n can be in the hundreds
    return float(min(Fraction(1), 2 * tail))


def paired_rows(systems, reference="F3"):
    """Per other system: the trials it shares with ``reference`` (same joint and goal), as error pairs."""

    base = {}
    for r in systems.get(reference, []):
        base.setdefault(r.get("pair_key", r["repeat_key"]), []).append(abs(r["error"]))
    result = {}
    for key, rows in systems.items():
        if key == reference:
            continue
        other = {}
        for r in rows:
            other.setdefault(r.get("pair_key", r["repeat_key"]), []).append(abs(r["error"]))
        shared = sorted(set(base) & set(other))
        result[key] = [(float(np.mean(base[k])), float(np.mean(other[k]))) for k in shared]
    return result


def paired_table(systems, names, reference="F3", reference_name="v4"):
    """On the same trials: is ``reference`` closer to the goal than each other system?"""

    lines = ["| %s against | shared trials | median of (%s error - other error) (rad) | %s closer in | other closer in | sign test p | reading |"
             % (reference_name, reference_name, reference_name), "|---|---:|---:|---:|---:|---:|---|"]
    for key, pairs in paired_rows(systems, reference).items():
        if not pairs:
            continue
        diff = np.array([a - b for a, b in pairs])
        wins, losses = int(np.sum(diff < 0)), int(np.sum(diff > 0))
        p = sign_test(wins, losses)
        reading = "no difference shown" if not p < 0.05 else ("%s is closer" % reference_name if wins > losses else "%s is closer" % names.get(key, key))
        lines.append("| %s | %d | %+.3f | %d%% | %d%% | %s | %s |" % (
            names.get(key, key), len(pairs), float(np.median(diff)), round(100.0 * wins / len(pairs)), round(100.0 * losses / len(pairs)),
            "under 0.001" if p < 0.001 else "%.3f" % p, reading))
    return lines


# -- tables ------------------------------------------------------------------
def overall_table(systems, failed, goals=None):
    lines = ["| system | trials | goal error, median / p90 / max (rad) | 95%% interval of the median (rad) | within %.2f rad at the end | settled | settling time, median (s) "
             "| time to 90%%, median (s) | overshoot, p90 / max (rad) | still when the trial ends, median range over the last 0.5 s (rad) "
             "| no drive | wrong way | clamped | base pushed over 1 cm | reset failed | wall time per trial (s) | per tick (ms) |" % SETTLE_TOLERANCE,
             "|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, short, _ in SYSTEMS:
        rows = [r for r in systems.get(key, []) if goals is None or r["repeat_key"] in goals]
        if not rows:
            continue
        over = [r["overshoot"] for r in rows if np.isfinite(r["overshoot"])]
        lines.append("| %s (%s) | %d | %s | %.4f - %.4f | %d%% | %d%% | %s | %s | %.3f / %.3f | %s | %d | %d | %d | %d | %s | %s | %s |" % (
            (key, short, len(rows), fmt3([r["error"] for r in rows])) + median_interval([r["error"] for r in rows]) + (
            round(100 * np.mean([abs(r["error"]) < SETTLE_TOLERANCE for r in rows])),
            round(100 * np.mean([r["reached"] for r in rows])), median([r["settle"] for r in rows]),
            median([r["t90"] for r in rows]), np.percentile(over, 90), max(over), median([r["hold"] for r in rows], 3),
            sum(r["no_drive"] for r in rows), sum(r["wrong"] for r in rows), sum(r["clamp"] for r in rows),
            sum(r["base_shift"] > 0.01 for r in rows), failed.get(key, 0) if key in failed else "-", median([r["wall"] for r in rows], 0), median([r["tick_ms"] for r in rows], 0))))
    return lines


def per_joint_table(systems, value, header):
    present = [key for key, _, _ in SYSTEMS if key in systems]
    lines = ["| joint | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    for joint in range(7):
        cells = [value([r for r in systems[key] if r["joint"] == joint]) for key in present]
        lines.append("| arm_%d | %s |" % (joint + 1, " | ".join(cells)))
    return [header, ""] + lines


def error_cell(rows):
    if not rows:
        return "-"
    m, p90, _ = stats([r["error"] for r in rows])
    return ("%.4f / %.4f" if m < 0.0005 else "%.3f / %.3f") % (m, p90)


def settle_cell(rows):
    if not rows:
        return "-"
    settled = [r["settle"] for r in rows if r["reached"] and np.isfinite(r["settle"])]
    return "%s (%d/%d)" % ("%.1f" % np.median(settled) if settled else "-", len(settled), len(rows))


def pair_table(sets, labels, keys=("PID", "B", "F", "P"), names=("PID baseline", "v3", "v4", "upper reference: tuned controller")):
    """The PID baseline, v3, v4 and the tuned reference controller over several runs (``sets[i]`` is ``{system: rows}``)."""

    lines = ["| condition | system | trials | goal error, median / p90 / max (rad) | settled | overshoot, p90 / max (rad) "
             "| time to 90%, median (s) | bumps lost | wrong way |", "|---|---|---:|---|---:|---|---:|---:|---:|"]
    for label, systems in zip(labels, sets):
        for key, name in zip(keys, names):
            rows = systems.get(key, [])
            if not rows:
                continue
            over = [r["overshoot"] for r in rows if np.isfinite(r["overshoot"])]
            lines.append("| %s | %s | %d | %s | %d%% | %.3f / %.3f | %s | %d | %d |" % (
                label, name, len(rows), fmt3([r["error"] for r in rows]), round(100 * np.mean([r["reached"] for r in rows])),
                np.percentile(over, 90), max(over), median([r["t90"] for r in rows]), sum(r["lost"] for r in rows), sum(r["wrong"] for r in rows)))
    return lines


def ring_table(rows_by_label):
    lines = ["| run | trials | predicted - state, median / p90 (indices) | predicted - goal bump (indices) | goal bump - goal (indices) | bumps lost |",
             "|---|---:|---|---|---|---:|"]
    for label, rows in rows_by_label:
        if not rows:
            continue
        cell = lambda key: "%.2f / %.2f" % stats([r[key] for r in rows])[:2]  # noqa: E731
        lines.append("| %s | %d | %s | %s | %s | %d |" % (label, len(rows), cell("body_gap"), cell("goal_gap"), cell("goal_bump"), sum(r["lost"] for r in rows)))
    return lines


def criteria_table(compare, robust, nest):
    """The acceptance criteria written before the runs, against what was measured (Gazebo unless said)."""

    v3, v4 = compare.get("B3", []), compare.get("F3", [])
    if not v4:
        return ["v4 was not measured."]
    in_index = lambda rows: [r["error"] / r["rho"] for r in rows if np.isfinite(r["rho"])]  # noqa: E731
    m4, p4, _ = stats([r["error"] for r in v4])
    i4 = stats(in_index(v4))
    lines = ["| criterion | target | v3 | v4 | met |", "|---|---|---|---|---|"]
    bad = lambda rows: sum(r["lost"] or r["wrong"] or r["clamp"] for r in rows)  # noqa: E731
    lines.append("| bumps lost, wrong-way or clamped trials | 0 | %d | %d | %s |" % (bad(v3), bad(v4), "yes" if bad(v4) == 0 else "no"))
    lines.append("| goal error, all joints, median / p90 | <= 0.045 / 0.12 rad (2 / 4.5 indices) | %s rad | %.3f / %.3f rad (%.1f / %.1f indices) | %s |" % (
        " / ".join("%.3f" % v for v in stats([r["error"] for r in v3])[:2]) if v3 else "-", m4, p4, i4[0], i4[1],
        "yes" if m4 <= 0.045 and p4 <= 0.12 else "no"))
    nominal = stats([r["error"] for r in robust.get("nominal", {}).get("F", [])])[1] if robust.get("nominal") else p4
    worst, cells3, cells4 = 0.0, [], []
    for key, label in CONDITIONS[1:3]:
        for system, cells in (("B", cells3), ("F", cells4)):
            rows = robust.get(key, {}).get(system, [])
            if rows:
                p90 = stats([r["error"] for r in rows])[1]
                cells.append("%s: %.3f" % (label.split()[-1], p90))
                if system == "F":
                    worst = max(worst, p90)
    if cells4:
        lines.append("| goal error p90 with the motor gain x0.7 / x1.3 | within 1.5x of the nominal p90 (%.3f rad) | %s | %s | %s |" % (
            nominal, ", ".join(cells3), ", ".join(cells4), "yes" if worst <= 1.5 * nominal else "no"))
    gap = stats([r["goal_gap"] for r in v4])[0]
    bump = stats([r["goal_bump"] for r in v4])[0]
    lines.append("| predicted bump - goal bump at rest, median | <= 1 index | no predictive ring | %.2f | %s |" % (gap, "yes" if gap <= 1 else "no"))
    lines.append("| goal bump - goal index at rest, median | <= 1 index | not recorded | %.2f | %s |" % (bump, "yes" if bump <= 1 else "no"))
    t3, t4 = [r["t90"] for r in v3 if np.isfinite(r["t90"])], [r["t90"] for r in v4 if np.isfinite(r["t90"])]
    if t3 and t4:
        lines.append("| time to 90%% of the move, median | <= 1.5x of v3 at the same time scale | %.2f s | %.2f s | %s |" % (
            np.median(t3), np.median(t4), "yes" if np.median(t4) <= 1.5 * np.median(t3) else "no"))
    push = robust.get("push", {})
    if push.get("F"):
        p3 = stats([r["error"] for r in push.get("B", [])])[1] if push.get("B") else float("nan")
        pf = stats([r["error"] for r in push["F"]])[1]
        lines.append("| goal error p90 after a 0.2 rad push | back within the nominal error (%.3f rad p90) | %.3f | %.3f | %s |" % (
            nominal, p3, pf, "yes" if pf <= 1.5 * nominal else "no"))
    return lines


def paper_numbers(compare, failed, robust):
    """One page with the numbers a text about v3 and v4 quotes (PID, v3 and v4 at 3x, Gazebo)."""

    subset = {key: rows for key, rows in compare.items() if key in FIGURE_SYSTEMS}
    lines = ["# Numbers for the paper", "",
             "Gazebo, joints arm_1 to arm_7, 8 goals per joint from the reset pose (fewer where MoveIt reports a collision). "
             "PID: the frozen baseline. v3: two rings on the bounded ring. v4: three rings with body feedback. Both ring systems at "
             "3 NEST steps per 50 ms robot tick. Settled: the error stays under %.2f rad for %d samples. Written by "
             "`scripts/results_report.py`; figures with the same names are next to this file." % (SETTLE_TOLERANCE, SETTLE_SAMPLES), "",
             "## All joints", ""] + overall_table(subset, failed)
    lines += [""] + per_joint_table(subset, error_cell, "## Goal error per joint, median / p90 (rad) (figure `error_by_joint`)")
    lines += [""] + per_joint_table(subset, settle_cell, "## Settling time per joint, median in s (trials settled / trials) (figure `settling_time`)")
    if robust:
        lines += ["", "## Disturbances on arm_2, arm_5, arm_6 (figure `robustness`)", ""]
        lines += pair_table([robust.get(c, {}) for c, _ in CONDITIONS], [label for _, label in CONDITIONS])
    v3, v4 = compare.get("B3", []), compare.get("F3", [])
    if v3 and v4:
        e3, e4 = stats([r["error"] for r in v3]), stats([r["error"] for r in v4])
        lines += ["", "## Ratios", "",
                  "- Goal error, v3 over v4: median %.1fx, p90 %.1fx, largest %.1fx." % tuple(a / b for a, b in zip(e3, e4)),
                  "- Trials settled: v3 %d of %d, v4 %d of %d." % (sum(r["reached"] for r in v3), len(v3), sum(r["reached"] for r in v4), len(v4))]
    return lines


# -- figures -----------------------------------------------------------------
def style(axis):
    axis.set_facecolor(SURFACE)
    axis.grid(True, axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=8.5)


def save(figure, out, name, paper=False):
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    directory = os.path.join(out, "paper" if paper else "figures")
    os.makedirs(directory, exist_ok=True)
    figure.savefig(os.path.join(directory, name + ".png"), dpi=160)
    if paper:
        figure.savefig(os.path.join(directory, name + ".pdf"))
    return os.path.join(os.path.basename(directory), name + ".png")


def grouped_bars(plt, systems, value, whisker, ylabel, title, note=None):
    figure, axis = plt.subplots(figsize=(9.2, 4.2))
    present = [k for k in FIGURE_SYSTEMS if k in systems]
    width = 0.8 / max(len(present), 1)
    for n, key in enumerate(present):
        for joint in range(7):
            rows = [r for r in systems[key] if r["joint"] == joint]
            if not rows:
                continue
            x = joint + (n - (len(present) - 1) / 2) * width
            height = value(rows)
            if not np.isfinite(height):
                continue
            axis.bar(x, height, width - 0.03, color=COLORS[key], label=FIGURE_NAMES[key] if joint == 0 else None)
            top = whisker(rows) if whisker else None
            if top is not None and np.isfinite(top):
                axis.plot([x, x], [height, top], color=INK, linewidth=1)
                axis.plot([x - 0.06, x + 0.06], [top, top], color=INK, linewidth=1)
            if note:
                axis.text(x, 0, note(rows), ha="center", va="bottom", fontsize=6.5, color=SURFACE)
    axis.set_xticks(range(7))
    axis.set_xticklabels(["arm_%d" % (j + 1) for j in range(7)])
    axis.set_ylabel(ylabel, color=INK)
    axis.set_title(title, loc="left", fontsize=10.5, color=INK)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), frameon=False, fontsize=9, loc="upper left", ncol=3)
    style(axis)
    return figure


def system_dots(plt, systems):
    figure, axis = plt.subplots(figsize=(8.4, 3.9))
    present = [(k, short) for k, short, _ in SYSTEMS if k in systems and k not in UPPER_REFERENCES]
    for n, (key, short) in enumerate(present):
        m, p90, top = stats([r["error"] for r in systems[key]])
        y = len(present) - 1 - n
        axis.plot([m, top], [y, y], color=GRID, linewidth=5, solid_capstyle="round")
        axis.plot([m, p90], [y, y], color="#86b6ef", linewidth=5, solid_capstyle="round")
        axis.plot([m], [y], "o", color="#2a78d6", markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.5)
        axis.text(top + 0.035, y, "%.3f / %.3f / %.3f" % (m, p90, top), va="center", fontsize=8.5, color=MUTED)
    axis.axvline(SETTLE_TOLERANCE, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axis.set_yticks(range(len(present)))
    axis.set_yticklabels(["%s  %s" % (k, short) for k, short in reversed(present)], color=INK, fontsize=9)
    axis.set_xlabel("goal error (rad): dot = median, dark bar to p90, light bar to the largest; dashed: settling tolerance %.2f rad"
                    % SETTLE_TOLERANCE, color=INK, fontsize=9)
    axis.set_title("Goal error of every system, Gazebo, all seven joints", loc="left", fontsize=10.5, color=INK)
    axis.set_xlim(0, None)
    style(axis)
    axis.grid(True, axis="x", color=GRID, linewidth=0.8)
    axis.grid(False, axis="y")
    return figure


#: label offsets (points) of the speed-accuracy figure: PID and v4 at 3x sit next to each other
LABEL_OFFSETS = {"F3": (9, -13), "PID": (9, 9), "G": (-150, -4), "L": (-108, -4), "T": (-150, -4)}


def speed_accuracy(plt, systems):
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    for key, short, _ in SYSTEMS:
        rows = systems.get(key)
        if not rows or key in UPPER_REFERENCES:
            continue
        t90 = np.median([r["t90"] for r in rows if np.isfinite(r["t90"])])
        p90 = stats([r["error"] for r in rows])[1]
        axis.plot([t90], [p90], "o", color="#2a78d6", markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.5)
        axis.annotate("%s  %s" % (key, short), (t90, p90), textcoords="offset points", xytext=LABEL_OFFSETS.get(key, (9, 4)),
                      fontsize=8.5, color=INK)
    axis.set_xlabel("time to 90% of the move, median (s)", color=INK)
    axis.set_ylabel("goal error, p90 (rad)", color=INK)
    axis.set_title("Speed against accuracy (lower left is better)", loc="left", fontsize=10.5, color=INK)
    axis.set_xlim(0, None)
    axis.set_ylim(0, None)
    style(axis)
    axis.grid(True, axis="x", color=GRID, linewidth=0.8)
    return figure


def traces(plt, systems, joints=(0, 3, 5)):
    """Error and joint velocity over time of the longest move every plotted system made, per joint."""

    present = [k for k in FIGURE_SYSTEMS if k in systems]
    figure, axes = plt.subplots(2, len(joints), figsize=(11.5, 5.6), sharex="col")
    for column, joint in enumerate(joints):
        shared = common_goals({k: [r for r in systems[k] if r["joint"] == joint] for k in present}, present)
        if not shared:
            continue
        start = next(r["start"] for r in systems[present[0]] if r["joint"] == joint)
        goal_key = max(shared, key=lambda item: abs(item[1] - start))
        for key in present:
            row = next(r for r in systems[key] if r["repeat_key"] == goal_key)
            q = np.asarray(row["trace"]["q"], dtype=float)
            dt = float(row["trace"]["dt_s"])
            t = np.arange(len(q)) * dt
            axes[0, column].plot(t, row["goal"] - q, color=COLORS[key], linewidth=2, label=FIGURE_NAMES[key])
            axes[1, column].plot(t[1:], np.diff(q) / dt, color=COLORS[key], linewidth=2)
        axes[0, column].axhspan(-SETTLE_TOLERANCE, SETTLE_TOLERANCE, color=GRID, alpha=0.7, linewidth=0)
        axes[0, column].set_title("arm_%d, goal %+.2f rad" % (joint + 1, goal_key[1]), loc="left", fontsize=10, color=INK)
        axes[1, column].set_xlabel("time (s)", color=INK)
        axes[1, column].set_xlim(0, 12)
        for axis in axes[:, column]:
            style(axis)
    axes[0, 0].set_ylabel("goal - joint (rad)", color=INK)
    axes[1, 0].set_ylabel("joint velocity (rad/s)", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="best")
    figure.suptitle("Error and joint velocity over time (grey band: settling tolerance)", x=0.01, ha="left", fontsize=11, color=INK)
    return figure


def robustness_figure(plt, robust):
    figure, axis = plt.subplots(figsize=(8.2, 4.0))
    series = [(system, key) for system, key in (("PID", "PID"), ("B", "B3"), ("F", "F3")) if any(robust.get(c, {}).get(system) for c, _ in CONDITIONS)]
    width = 0.8 / len(series)
    for n, (system, key) in enumerate(series):
        for x, (condition, _) in enumerate(CONDITIONS):
            rows = robust.get(condition, {}).get(system, [])
            if not rows:
                continue
            m, p90, top = stats([r["error"] for r in rows])
            position = x + (n - (len(series) - 1) / 2.0) * width
            axis.bar(position, p90, width - 0.03, color=COLORS[key], label=FIGURE_NAMES[key] if x == 0 else None)
            axis.text(position, p90, ("%.2f" if p90 >= 0.1 else "%.3f") % p90, ha="center", va="bottom", fontsize=7.5, color=MUTED)
    axis.set_xticks(range(len(CONDITIONS)))
    axis.set_xticklabels([label for _, label in CONDITIONS])
    axis.set_ylabel("goal error, p90 (rad)", color=INK)
    axis.set_title("What a disturbance does to the goal error (Gazebo, arm_2, arm_5, arm_6)", loc="left", fontsize=10.5, color=INK)
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    style(axis)
    return figure


def blind_band_figure(plt, data):
    plain, sharp = (read_json(os.path.join(data, "measure", name)) if os.path.exists(os.path.join(data, "measure", name)) else None
                    for name in ("fine_map.json", "fine_map_sharp.json"))
    if plain is None or sharp is None:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)
    for axis, document, title, color in ((axes[0], plain, "Ramp comparator alone (v3)", COLORS["B3"]),
                                         (axes[1], sharp, "With the goal ring's projection (v4)", COLORS["F3"])):
        for row in document["rows"]:
            axis.plot(row["offsets"], row["drive"], color=color, linewidth=1.6, alpha=0.75)
        axis.axhline(0, color=MUTED, linewidth=1)
        axis.axvline(0, color=MUTED, linewidth=1)
        widths = [row["blind_width"] for row in document["rows"]]
        axis.set_title("%s: no drive over %d-%d indices" % (title, min(widths), max(widths)), loc="left", fontsize=10, color=INK)
        axis.set_xlabel("goal bump - state bump (ring indices)", color=INK)
        style(axis)
    axes[0].set_ylabel("right - left gain spikes per tick", color=INK)
    figure.suptitle("Drive of the gain pair against the goal offset, five states on the arc (one line each)", x=0.01, ha="left", fontsize=11, color=INK)
    return figure


# -- main --------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="docs/results/data")
    parser.add_argument("--out", required=True, help="docs/results")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    compare, failed, missing = load_set(os.path.join(args.data, "compare"))
    repeat, _, _ = load_set(os.path.join(args.data, "repeat"))
    robust = {}
    for condition, _ in CONDITIONS:                    # <condition>.json: v3 and v4; <condition>_p_controller.json: the tuned reference
        loaded, _ = load_validation(os.path.join(args.data, "robust", condition + ".json"))
        loaded.update(load_validation(os.path.join(args.data, "robust", condition + "_p_controller.json"))[0])
        baseline = load_legacy(os.path.join(args.data, "robust", condition + "_pid"), "pid", "PID")     # the PID baseline (frozen script)
        if baseline:
            loaded["PID"] = baseline
        if loaded:
            robust[condition] = loaded
    if robust and not robust.get("nominal"):                                      # no nominal run of the same goals: the matching trials of the comparison
        joints = {r["joint"] for rows in next(iter(robust.values())).values() for r in rows}
        goals = {r["repeat_key"] for rows in next(iter(robust.values())).values() for r in rows}
        robust["nominal"] = {system: [r for r in compare.get(key, []) if r["joint"] in joints and r["repeat_key"] in goals]
                             for system, key in (("B", "B3"), ("F", "F3"))}
    starts = load_directory(os.path.join(args.data, "starts"))
    nest = {}
    for path in sorted(glob.glob(os.path.join(args.data, "nest", "*.json"))):
        nest[os.path.splitext(os.path.basename(path))[0]], _ = load_validation(path)
    nest_all = {system: [r for name, loaded in nest.items() if name.startswith("j") for r in loaded.get(system, [])] for system in ("B", "F")}

    out = ["# Result tables", "",
           "Written by `scripts/results_report.py` from `data/`; the reading of these numbers is in [`../results.md`](../results.md).",
           "Goal error is |goal - final joint angle|. Settled: the error stays under %.2f rad for %d samples (the paper's definition); "
           "the settling time is the first time it does." % (SETTLE_TOLERANCE, SETTLE_SAMPLES), ""]
    if missing:
        out += ["Missing inputs: " + ", ".join("`%s`" % name for name in missing), ""]
    out += ["## Systems", "", "| key | system | what it is |", "|---|---|---|"]
    out += ["| %s | %s | %s |" % item for item in SYSTEMS]
    out += ["", "## 1. Every system, Gazebo, joints arm_1 to arm_7, goals from the reset pose", ""] + overall_table(compare, failed)
    every = common_goals(compare, [k for k, _, _ in SYSTEMS if k in compare])
    if every and any(r["repeat_key"] not in every for rows in compare.values() for r in rows):
        out += ["", "### The same table on the %d goals that every system ran" % len(every), ""] + overall_table(compare, failed, every)
    elif every:
        out += ["", "Every system ran the same %d goals." % len(every)]
    if "F3" in compare:
        out += ["", "## 1a. v4 against each other system on the same goals (paired)", "",
                "For every goal both systems ran, the difference of their goal errors. 'Closer in' is the share of goals in which that "
                "system ended nearer the goal. The sign test asks how likely a split this uneven would be if neither were better; "
                "under 0.05 is read as a real difference.", ""] + paired_table(compare, NAMES)
    out += [""] + per_joint_table(compare, error_cell, "## 2. Goal error per joint, median / p90 (rad)")
    out += [""] + per_joint_table(compare, settle_cell, "## 3. Settling time per joint, median in s (trials settled / trials)")
    if repeat:
        out += ["", "## 4. Repeatability: range of the final angle over three runs of the same goal (rad)", "",
                "| system | goals | median range | largest range |", "|---|---:|---:|---:|"]
        for key, short, _ in SYSTEMS:
            if key in repeat:
                m, top, n = repeat_spread(repeat[key])
                out.append("| %s (%s) | %d | %.4f | %.4f |" % (key, short, n, m, top))
    if robust:
        out += ["", "## 5. Disturbances, Gazebo: the PID baseline, v3 and v4 (both at 3x)", ""]
        out += pair_table([robust.get(c, {}) for c, _ in CONDITIONS], [label for _, label in CONDITIONS])
    if starts:
        out += ["", "## 6. Three start poses (0.25, 0.5, 0.75 of the range), Gazebo, v3 against v4", ""]
        out += pair_table([{k: [r for r in rows if r["joint"] == j] for k, rows in starts.items()} for j in range(7)] + [starts],
                          ["arm_%d" % (j + 1) for j in range(7)] + ["all joints"])
    if any(nest_all.values()):
        out += ["", "## 7. NEST with the fake robot, v3 against v4", ""]
        labels = ["all joints, 3 starts x 3 seeds x 8 goals"] + ["%s (arm_2, arm_5, arm_6)" % label for c, label in CONDITIONS[1:] if "rob_" + c in nest]
        sets = [nest_all] + [nest["rob_" + c] for c, _ in CONDITIONS[1:] if "rob_" + c in nest]
        if "limits" in nest:
            labels.append("goals exactly at the joint limits (arm_3, arm_6)")
            sets.append(nest["limits"])
        for k in ("k1", "k2"):
            if k in nest:
                labels.append("time scale %s (arm_5, arm_6)" % k[1])
                sets.append(nest[k])
        out += pair_table(sets, labels)
    ring_rows = [("Gazebo, comparison set", compare.get("F3", [])), ("Gazebo, three start poses", starts.get("F", [])), ("NEST, all joints", nest_all.get("F", []))]
    ring_rows += [("Gazebo, %s" % label, robust.get(c, {}).get("F", [])) for c, label in CONDITIONS[1:]]
    if any(rows for _, rows in ring_rows):
        out += ["", "## 8. v4: where the three bumps end", ""] + ring_table(ring_rows)
    out += ["", "## 9. Acceptance criteria of v4 (written before the runs)", ""] + criteria_table(compare, robust, nest)

    figures = []
    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        p90 = lambda rows: stats([r["error"] for r in rows])[1]  # noqa: E731
        med = lambda rows: stats([r["error"] for r in rows])[0]  # noqa: E731
        settled = lambda rows: float(np.median([r["settle"] for r in rows if r["reached"]])) if any(r["reached"] for r in rows) else float("nan")  # noqa: E731
        share = lambda rows: "%d/%d" % (sum(r["reached"] for r in rows), len(rows))  # noqa: E731
        if compare:
            figures.append(save(system_dots(plt, compare), args.out, "error_by_system"))
            figures.append(save(speed_accuracy(plt, compare), args.out, "speed_accuracy"))
            for paper in (False, True):
                figures.append(save(grouped_bars(plt, compare, med, p90, "goal error (rad): bar = median, whisker = p90",
                                                 "Goal error per joint, Gazebo"), args.out, "error_by_joint", paper))
                figures.append(save(grouped_bars(plt, compare, settled, None, "settling time, median of the settled trials (s)",
                                                 "Settling time per joint, Gazebo (in the bar: trials settled / trials)", share),
                                    args.out, "settling_time", paper))
                figures.append(save(traces(plt, compare), args.out, "traces", paper))
        if robust:
            for paper in (False, True):
                figures.append(save(robustness_figure(plt, robust), args.out, "robustness", paper))
        blind = blind_band_figure(plt, args.data)
        if blind is not None:
            for paper in (False, True):
                figures.append(save(blind, args.out, "blind_band", paper))
        plt.close("all")
    os.makedirs(os.path.join(args.out, "paper"), exist_ok=True)
    with open(os.path.join(args.out, "tables.md"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(out) + "\n")
    with open(os.path.join(args.out, "paper", "numbers.md"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(paper_numbers(compare, failed, robust)) + "\n")
    print("written: %s (%d lines), %d figure files" % (os.path.join(args.out, "tables.md"), len(out), len(figures)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
