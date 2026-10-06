#!/usr/bin/env python3
"""Tables and figures of docs/archive/velocity/bounded-ring.md from the validation / legacy JSON files.

    D=../../docs/archive/velocity/media/bounded/data
    python3 -B scripts/bounded_ring_report.py --nest $D/validation_nest.json --gazebo $D/validation_gazebo.json \
        --legacy $D --ramp $D/ramp_sweep.json --maps $D/decision_map_legacy.json $D/decision_map_bounded.json --out /tmp/report

``--legacy`` is a directory of ``legacy_trials_j*.json`` (``scripts/run_legacy_trials.py``).
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette, fixed order, one slot per system (never cycled).
COLORS = {"B": "#2a78d6", "T": "#eb6834", "G": "#1baf7a", "L": "#eda100"}
NAMES = {"L": "L legacy scripts", "G": "G legacy ring, legacy gains", "T": "T legacy ring, transport cal.", "B": "B bounded ring"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(axis):
    axis.grid(True, axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED)


def stats(values):
    v = np.abs(np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float))
    if not len(v):
        return (float("nan"),) * 3
    return float(np.median(v)), float(np.percentile(v, 90)), float(v.max())


def tracking(rows):
    usable = [r for r in rows if r["status"] == "ok" and abs(r["bump_displacement"]) > 1]
    if not usable:
        return (float("nan"),) * 4
    rel = np.array([(r["joint_displacement"] / r["rho"] - r["bump_displacement"]) / abs(r["bump_displacement"]) for r in usable])
    left = [x for x, r in zip(rel, usable) if r["bump_displacement"] < 0]
    right = [x for x, r in zip(rel, usable) if r["bump_displacement"] > 0]
    return (float(np.median(np.abs(rel))), float(np.percentile(np.abs(rel), 90)),
            float(np.mean(left)) if left else float("nan"), float(np.mean(right)) if right else float("nan"))


def system_table(trials, systems, joints, title):
    lines = ["### %s" % title, "",
             "| joint | system | trials | goal error median / p90 / max (rad) | tracking median / p90 | bias left / right "
             "| ring - joint, during / at end (median, rad) | no drive | wrong way | clamps | settle (sim s) | wall per trial (s) |",
             "|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|"]
    for joint in joints:
        for system in systems:
            rows = [r for r in trials if r["joint"] == joint and r["system"] == system]
            if not rows:
                continue
            g = stats([r["goal_error_rad"] for r in rows])
            t = tracking(rows)
            during = np.nanmedian([r["state_error_abs_median"] for r in rows])
            end = np.nanmedian(np.abs([r["state_error_final"] for r in rows]))
            lines.append("| %d | %s | %d | %.3f / %.3f / %.3f | %.1f%% / %.1f%% | %+.1f%% / %+.1f%% | %.3f / %.3f | %d | %d | %d | %.1f | %.1f |" % (
                joint, system, len(rows), g[0], g[1], g[2], 100 * t[0], 100 * t[1], 100 * t[2], 100 * t[3], during, end,
                sum(r["no_drive"] for r in rows), sum(r["wrong_direction"] for r in rows), sum(r["clamp_ticks"] > 0 for r in rows),
                np.mean([r["settle_s"] for r in rows]), np.mean([r["wall_s"] for r in rows])))
    return lines


def overall_table(trials, systems, title):
    lines = ["### %s" % title, "",
             "| system | trials | goal error median / p90 / max (rad) | tracking median / p90 | ring - joint at end (median / p95, rad) "
             "| no drive | wrong way | clamps |", "|---|---:|---|---|---|---:|---:|---:|"]
    for system in systems:
        rows = [r for r in trials if r["system"] == system]
        if not rows:
            continue
        g = stats([r["goal_error_rad"] for r in rows])
        t = tracking(rows)
        end = np.abs([r["state_error_final"] for r in rows if np.isfinite(r["state_error_final"])])
        lines.append("| %s | %d | %.3f / %.3f / %.3f | %.1f%% / %.1f%% | %.3f / %.3f | %d | %d | %d |" % (
            NAMES[system], len(rows), g[0], g[1], g[2], 100 * t[0], 100 * t[1], np.median(end), np.percentile(end, 95),
            sum(r["no_drive"] for r in rows), sum(r["wrong_direction"] for r in rows), sum(r["clamp_ticks"] > 0 for r in rows)))
    return lines


def goal_error_figure(trials, systems, joints, path, title):
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    style(axis)
    width = 0.8 / len(systems)
    for k, system in enumerate(systems):
        xs, med, p90 = [], [], []
        for j, joint in enumerate(joints):
            rows = [r for r in trials if r["joint"] == joint and r["system"] == system]
            if not rows:
                continue
            s = stats([r["goal_error_rad"] for r in rows])
            xs.append(j + (k - (len(systems) - 1) / 2) * width)
            med.append(s[0])
            p90.append(s[1])
        axis.vlines(xs, med, p90, color=COLORS[system], linewidth=2)
        axis.plot(xs, med, "o", color=COLORS[system], markersize=8, markeredgecolor="white", markeredgewidth=1.2, label=NAMES[system])
    axis.set_xticks(range(len(joints)))
    axis.set_xticklabels(["arm_%d" % (j + 1) for j in joints])
    axis.set_ylabel("|goal - final joint| (rad)", color=INK)
    axis.set_title("%s: median (dot) to p90 (line)" % title, color=INK, loc="left", fontsize=11)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def ramp_figure(ramp, path):
    positions = ramp["positions"]
    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    style(axis)
    axis.axvspan(30, 170, color="#f1f0ec", zorder=0)
    axis.text(100, 1.5, "joint arc (indices 30-170)", ha="center", color=MUTED, fontsize=9)
    for key, color, label in (("0.0", "#eb6834", "no ramp current (floor below index ~45)"), ("800000.0", "#2a78d6", "ramp current 8e5 pA (chosen)")):
        values = ramp["ramps"][key]["warm"]
        axis.plot(positions, values, "-o", color=color, linewidth=2, markersize=5, label=label)
    axis.set_xlabel("state bump ring index", color=INK)
    axis.set_ylabel("warm neuron spikes per tick", color=INK)
    axis.set_title("Comparator ramp across the joint arc", color=INK, loc="left", fontsize=11)
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def decision_maps_figure(paths, titles, path, bound=200.0):
    figure, axes = plt.subplots(1, len(paths), figsize=(5.2 * len(paths), 4.6))
    for axis, map_path, title in zip(axes, paths, titles):
        data = json.load(open(map_path))
        grid = data["grid"]
        cells = {(r["state"], r["goal"]): r["drive"] for r in data["rows"]}
        values = np.array([[cells[(s, g)] for g in grid] for s in grid])
        image = axis.imshow(np.clip(values, -bound, bound), origin="lower", cmap="RdBu_r", vmin=-bound, vmax=bound,
                            aspect="auto", extent=[grid[0], grid[-1], grid[0], grid[-1]])
        axis.plot([grid[0], grid[-1]], [grid[0], grid[-1]], color=MUTED, linewidth=1)
        summary = data["summary"]
        axis.set_title("%s\nwrong sign %d, largest dead zone %.0f indices" % (title, summary["wrong_sign"], summary["dead_zone_max"]),
                       color=INK, loc="left", fontsize=10)
        axis.set_xlabel("goal ring index", color=INK)
        axis.set_ylabel("state ring index", color=INK)
    figure.subplots_adjust(wspace=0.3)
    figure.colorbar(image, ax=list(axes), label="right - left gain spikes per tick (clipped at +-%d)" % bound)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def legacy_rows(directory):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "legacy_trials_j*.json"))):
        rows.extend(json.load(open(path))["trials"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nest")
    parser.add_argument("--gazebo")
    parser.add_argument("--legacy")
    parser.add_argument("--ramp")
    parser.add_argument("--maps", nargs=2, help="decision_map.json of the legacy and the bounded ring")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lines = []
    for name, path in (("NEST, fake robot", args.nest), ("Gazebo", args.gazebo)):
        if not path or not os.path.exists(path):
            continue
        data = json.load(open(path))
        trials = data["trials"]
        joints = sorted({r["joint"] for r in trials})
        systems = [s for s in "GTB" if any(r["system"] == s for r in trials)]
        lines += overall_table(trials, systems, "%s, all joints" % name) + [""]
        lines += system_table(trials, systems, joints, "%s, per joint" % name) + [""]
        if data.get("skipped"):
            lines += ["MoveIt skipped %d goals: %s" % (len(data["skipped"]), data["skipped"]), ""]
        tag = "nest" if "NEST" in name else "gazebo"
        goal_error_figure(trials, systems, joints, os.path.join(args.out, "goal_error_%s.png" % tag), name)
    if args.legacy and os.path.isdir(args.legacy) and args.gazebo and os.path.exists(args.gazebo):
        legacy = legacy_rows(args.legacy)
        graph = json.load(open(args.gazebo))["trials"]
        joints = sorted({r["joint"] for r in legacy})
        lines += ["### Legacy scripts (L) against the graph systems, Gazebo, same joints and goals", "",
                  "| joint | system | trials | goal error median / p90 / max (rad) | ring - joint during trial (median, rad) | no drive | wrong way "
                  "| wall per trial (s) | wall per tick (ms) | sim s per trial | final-angle spread over repeats (rad) |",
                  "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
        for joint in joints:
            goals = sorted({round(r["goal"], 4) for r in legacy if r["joint"] == joint})
            for system in "LGTB":
                if system == "L":
                    rows = [r for r in legacy if r["joint"] == joint]
                    spreads = {}
                    for r in rows:
                        spreads.setdefault(round(r["goal"], 4), []).append(r["q_final"])
                    spread = max([max(v) - min(v) for v in spreads.values() if len(v) > 1], default=float("nan"))
                    wall = np.mean([r["wall_s"] for r in rows])
                    tick = np.mean([r["tick_wall_ms"] for r in rows])
                    sim = np.mean([r["n_steps"] * 0.05 for r in rows])
                    during = np.nanmedian([r["state_error_abs_median"] for r in rows])
                else:
                    rows = [r for r in graph if r["joint"] == joint and r["system"] == system and round(r["goal_rad"], 4) in goals]
                    if not rows:
                        continue
                    spread = float("nan")
                    wall = np.mean([r["wall_s"] for r in rows])
                    tick = np.mean([r["tick_wall_ms"] for r in rows])
                    sim = np.mean([r["settle_s"] for r in rows])
                    during = np.nanmedian([r["state_error_abs_median"] for r in rows])
                g = stats([r["goal_error_rad"] for r in rows])
                lines.append("| %d | %s | %d | %.3f / %.3f / %.3f | %.3f | %d | %d | %.1f | %.0f | %.1f | %s |" % (
                    joint, system, len(rows), g[0], g[1], g[2], during, sum(r["no_drive"] for r in rows),
                    sum(r["wrong_direction"] for r in rows), wall, tick, sim, "%.3f" % spread if np.isfinite(spread) else ""))
        lines.append("")
    if args.maps:
        decision_maps_figure(args.maps, ["Legacy ring: joint range on indices 5-195", "Bounded ring: joint range on indices 30-170"],
                             os.path.join(args.out, "decision_maps.png"))
    if args.ramp and os.path.exists(args.ramp):
        ramp_figure(json.load(open(args.ramp)), os.path.join(args.out, "comparator_ramp.png"))
    with open(os.path.join(args.out, "tables.md"), "w") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
