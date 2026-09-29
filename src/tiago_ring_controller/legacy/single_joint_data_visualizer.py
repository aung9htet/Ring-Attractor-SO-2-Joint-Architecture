#!/usr/bin/env python3
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tiago_ring_controller.evaluation.metrics import overshoot, settling_time
from tiago_ring_controller.evaluation.plots import (
    PID_COLOR,
    RING_COLOR,
    SETTLE_STEPS,
    SETTLE_TOLERANCE,
    final_absolute_error,
    gain_imbalance,
    goal_from_timeseries,
    paired_by_trial_index,
    representative_trial_index,
)

BASE    = os.path.join(os.path.dirname(__file__), "collected_data")
RESULTS = os.path.join(os.path.dirname(__file__), "figures")

SETTLE_TOL = SETTLE_TOLERANCE


def _apply_style(ax, fs, fw):
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fs)
        item.set_fontweight(fw)
    ax.tick_params(labelsize=fs)


def _savefig(fig, path, fs, fw):
    for ax in fig.axes:
        _apply_style(ax, fs, fw)
        leg = ax.get_legend()
        if leg:
            for t in leg.get_texts():
                t.set_fontsize(fs)
                t.set_fontweight(fw)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ts_path(joint, trial, controller):
    if controller == "ring":
        return os.path.join(BASE, f"joint_{joint}", "trials",
                            f"trial_{trial:04d}_timeseries.npz")
    return os.path.join(BASE, f"joint_{joint}", "pid", "trials",
                        f"trial_{trial:04d}_timeseries.npz")


def _n_trials(joint, controller):
    if controller == "ring":
        d = os.path.join(BASE, f"joint_{joint}", "trials")
    else:
        d = os.path.join(BASE, f"joint_{joint}", "pid", "trials")
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith("_timeseries.npz")])


def _load_ts(joint, trial, controller):
    p = _ts_path(joint, trial, controller)
    return np.load(p) if os.path.exists(p) else None


def _q_goal_from_ts(d):
    return goal_from_timeseries(d, preserve_input_type=True)


def _final_errors_all_trials(joint, controller):
    n = _n_trials(joint, controller)
    errs = []
    for i in range(1, n + 1):
        d = _load_ts(joint, i, controller)
        if d is not None:
            errs.append(final_absolute_error(d))
    return np.array(errs)


def _settling_times(joint, controller, tol=SETTLE_TOL, n_consec=SETTLE_STEPS):
    n = _n_trials(joint, controller)
    times = []
    for i in range(1, n + 1):
        d = _load_ts(joint, i, controller)
        if d is None:
            continue
        times.append(
            settling_time(
                d["position_error"],
                d["time"],
                tolerance=tol,
                consecutive_steps=n_consec,
            )
        )
    return np.array(times)


def _overshoots(joint, controller):
    n = _n_trials(joint, controller)
    ov = []
    for i in range(1, n + 1):
        d = _load_ts(joint, i, controller)
        if d is None:
            continue
        q = d["joint_position"]
        q_g = float(_q_goal_from_ts(d)[-1])
        ov.append(overshoot(q, q_g, preserve_input_type=True))
    return np.array(ov)


def _pick_representative_trial(joint, controller):
    n = _n_trials(joint, controller)
    if n == 0:
        return None, None
    errs = []
    for i in range(1, n + 1):
        d = _load_ts(joint, i, controller)
        e = final_absolute_error(d) if d is not None else np.inf
        errs.append(e)
    errs = np.array(errs)
    idx = representative_trial_index(errs) + 1
    return idx, _load_ts(joint, idx, controller)


def _print_table(joint):
    r_n = _n_trials(joint, "ring")
    p_n = _n_trials(joint, "pid")

    rows = [
        ("fig_01_example_joint_trajectory",
         "POSSIBLE" if r_n and p_n else "PARTIAL",
         "controller, time, q_measured, q_goal",
         "joint_position, position_error→q_goal",
         "q_goal = joint_position + position_error"),
        ("fig_02_error_over_time",
         "POSSIBLE" if r_n and p_n else "PARTIAL",
         "controller, time, q_measured, q_goal",
         "position_error (pre-computed), time",
         ""),
        ("fig_03_velocity_command_over_time",
         "POSSIBLE" if r_n and p_n else "PARTIAL",
         "controller, time, qdot_cmd",
         "decoded_velocity→qdot_cmd, time",
         ""),
        ("fig_04_final_error_boxplot",
         "POSSIBLE" if r_n and p_n else "PARTIAL",
         "controller, final_error",
         "abs(position_error[-1])",
         "computed from last timeseries step"),
        ("fig_05_paired_final_error_scatter",
         "PARTIAL – matched by trial index only" if r_n and p_n else "IMPOSSIBLE",
         "controller, trial_id, final_error",
         "abs(position_error[-1])",
         "q_goal differs per controller; no true pairing"),
        ("appendix_success_rate",
         "POSSIBLE" if r_n and p_n else "PARTIAL",
         "controller, final_error",
         "abs(position_error[-1])",
         f"tol=0.05 and 0.1 rad"),
        ("appendix_settling_time",
         "POSSIBLE" if r_n and p_n else "PARTIAL",
         "controller, time, position_error",
         "position_error, time",
         f"first {SETTLE_STEPS} steps < {SETTLE_TOL} rad"),
        ("appendix_overshoot",
         "POSSIBLE" if r_n and p_n else "PARTIAL",
         "controller, time, q_measured, q_goal",
         "joint_position, q_goal from position_error",
         "max excursion past goal"),
        ("appendix_control_effort",
         "POSSIBLE" if r_n and p_n else "PARTIAL",
         "controller, time, qdot_cmd",
         "decoded_velocity, time",
         "integral |qdot_cmd| dt"),
        ("appendix_gain_imbalance_vs_velocity",
         "POSSIBLE (Ring only)" if r_n else "IMPOSSIBLE",
         "controller, qdot_cmd, left/right_gain_spikes",
         "left_gain_spikes, right_gain_spikes, decoded_velocity",
         "Ring only; PID has no gain neurons"),
    ]

    w = [44, 44, 44, 44, 50]
    hdr = "  ".join(f"{h:{w[i]}}" for i, h in enumerate(
        ["plot_name", "status", "required_columns", "available_columns_used", "notes"]))
    print(f"\nJoint {joint} — column availability")
    print("=" * sum(w))
    print(hdr)
    print("-" * sum(w))
    for r in rows:
        print("  ".join(f"{r[i]:{w[i]}}" for i in range(len(w))))
    print()


def _draw_box(ax, x, data, color, width=0.35):
    m, s = np.mean(data), np.std(data)
    ax.bar(x, 2 * s, bottom=m - s, color=color, width=width, alpha=0.85)
    ax.hlines(m, x - width / 2, x + width / 2, colors="white", linewidths=2.5)


def plot_fig01(joint, out_dir, fs, fw):
    _, dr = _pick_representative_trial(joint, "ring")
    _, dp = _pick_representative_trial(joint, "pid")
    if dr is None or dp is None:
        print(f"  fig_01: skip (missing data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, d, label, color in zip(axes,
                                   [dr, dp],
                                   ["Ring", "PID"],
                                   [RING_COLOR, PID_COLOR]):
        t    = d["time"]
        q    = d["joint_position"]
        goal = float(_q_goal_from_ts(d)[-1])
        ax.plot(t, q, color=color, lw=2, label=label)
        ax.axhline(goal, color="k", ls="--", lw=1.5, label="q_goal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Joint position (rad)")
        ax.set_title(f"{label} – Joint {joint}")
        ax.legend()

    _savefig(fig, os.path.join(out_dir, "fig_01_example_joint_trajectory.png"), fs, fw)
    print("  fig_01 saved")


def plot_fig02(joint, out_dir, fs, fw):
    _, dr = _pick_representative_trial(joint, "ring")
    _, dp = _pick_representative_trial(joint, "pid")
    if dr is None or dp is None:
        print("  fig_02: skip (missing data)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for d, label, color in [(dr, "Ring", RING_COLOR), (dp, "PID", PID_COLOR)]:
        ax.plot(d["time"], np.abs(d["position_error"]), color=color, lw=2, label=label)

    ax.axhline(SETTLE_TOL, color="grey", ls=":", lw=1.2, label=f"tol={SETTLE_TOL} rad")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absolute error (rad)")
    ax.set_title(f"Error over time – Joint {joint}")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "fig_02_error_over_time.png"), fs, fw)
    print("  fig_02 saved")


def plot_fig03(joint, out_dir, fs, fw):
    _, dr = _pick_representative_trial(joint, "ring")
    _, dp = _pick_representative_trial(joint, "pid")
    if dr is None or dp is None:
        print("  fig_03: skip (missing data)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for d, label, color in [(dr, "Ring", RING_COLOR), (dp, "PID", PID_COLOR)]:
        ax.plot(d["time"], d["decoded_velocity"], color=color, lw=2, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity command (rad/s)")
    ax.set_title(f"Velocity command over time – Joint {joint}")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "fig_03_velocity_command_over_time.png"), fs, fw)
    print("  fig_03 saved")


def plot_fig04(joint, out_dir, fs, fw):
    r_err = _final_errors_all_trials(joint, "ring")
    p_err = _final_errors_all_trials(joint, "pid")
    if not len(r_err) and not len(p_err):
        print("  fig_04: skip")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    if len(r_err):
        _draw_box(ax, 1, r_err, RING_COLOR)
    if len(p_err):
        _draw_box(ax, 2, p_err, PID_COLOR)

    all_err = np.concatenate([r_err, p_err])
    top = max(all_err.max() * 1.15, 0.05)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Ring", "PID"])
    ax.set_ylim(0, top)
    ax.axhline(np.pi, color="grey", ls=":", lw=1.2, label="π (max circular error)")
    ax.set_ylabel("Final absolute error (rad)")
    ax.set_title(f"Final error – Joint {joint}")
    ax.legend(fontsize=fs - 4)
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, os.path.join(out_dir, "fig_04_final_error_boxplot.png"), fs, fw)
    print("  fig_04 saved")


def plot_fig05(joint, out_dir, fs, fw):
    r_err = _final_errors_all_trials(joint, "ring")
    p_err = _final_errors_all_trials(joint, "pid")
    r_err, p_err = paired_by_trial_index(
        r_err,
        p_err,
        preserve_input_type=True,
    )
    if len(r_err) == 0:
        print("  fig_05: skip")
        return

    lim = max(r_err.max(), p_err.max()) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(p_err, r_err, color=RING_COLOR, alpha=0.6, s=30, zorder=3)
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("PID final error (rad)")
    ax.set_ylabel("Ring final error (rad)")
    ax.set_title(f"Paired final error – Joint {joint}\n(matched by trial index)")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "fig_05_paired_final_error_scatter.png"), fs, fw)
    print("  fig_05 saved")



def plot_app07(joint, out_dir, fs, fw):
    r_t = _settling_times(joint, "ring")
    p_t = _settling_times(joint, "pid")
    if not len(r_t) and not len(p_t):
        print("  app_07: skip")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    if len(r_t):
        _draw_box(ax, 1, r_t, RING_COLOR)
    if len(p_t):
        _draw_box(ax, 2, p_t, PID_COLOR)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Ring", "PID"])
    ax.set_ylabel("Settling time (s)")
    ax.set_title(f"Settling time (tol={SETTLE_TOL} rad) – Joint {joint}")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, os.path.join(out_dir, "appendix_settling_time.png"), fs, fw)
    print("  app_07 saved")


def plot_app08(joint, out_dir, fs, fw):
    r_ov = _overshoots(joint, "ring")
    p_ov = _overshoots(joint, "pid")
    if not len(r_ov) and not len(p_ov):
        print("  app_08: skip")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    if len(r_ov):
        _draw_box(ax, 1, r_ov, RING_COLOR)
    if len(p_ov):
        _draw_box(ax, 2, p_ov, PID_COLOR)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Ring", "PID"])
    ax.set_ylabel("Overshoot (rad)")
    ax.set_title(f"Overshoot – Joint {joint}")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, os.path.join(out_dir, "appendix_overshoot.png"), fs, fw)
    print("  app_08 saved")


def plot_app10(joint, out_dir, fs, fw):
    n = _n_trials(joint, "ring")
    if n == 0:
        print("  app_10: skip (no ring data)")
        return

    imbalance_all, vcmd_all = [], []
    for i in range(1, n + 1):
        d = _load_ts(joint, i, "ring")
        if d is None or "left_gain_spikes" not in d:
            continue
        imbalance_all.append(gain_imbalance(d, preserve_input_type=True))
        vcmd_all.append(d["decoded_velocity"])

    if not imbalance_all:
        print("  app_10: skip (missing gain columns)")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        np.concatenate(imbalance_all),
        np.concatenate(vcmd_all),
        alpha=0.15, s=8, color=RING_COLOR, rasterized=True,
    )
    ax.set_xlabel("Gain imbalance (right − left spikes)")
    ax.set_ylabel("Velocity command (rad/s)")
    ax.set_title(f"Gain imbalance vs velocity cmd – Joint {joint}")
    ax.grid(True, alpha=0.3)
    _savefig(fig, os.path.join(out_dir, "appendix_gain_imbalance_vs_velocity.png"), fs, fw)
    print("  app_10 saved")


def plot_comparison(joint, out_dir, fs, fw):
    from matplotlib.patches import Patch

    pi_ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    pi_ticklabels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    def _load_batches(d):
        bd = os.path.join(d, "batches")
        if not os.path.isdir(bd):
            return None, None
        files = sorted(f for f in os.listdir(bd) if f.endswith("_abs_errors.npy"))
        if not files:
            return None, None
        return (
            [np.load(os.path.join(bd, f)) for f in files],
            [f.split("_abs")[0] for f in files],
        )

    r_data, r_labels = _load_batches(os.path.join(BASE, f"joint_{joint}"))
    p_data, p_labels = _load_batches(os.path.join(BASE, f"joint_{joint}", "pid"))

    error_ylim_top = 2 * np.pi

    for data, labels, title, color, fname in [
        (r_data, r_labels, f"Ring – absolute error | Joint {joint}", RING_COLOR, "ring_error.png"),
        (p_data, p_labels, f"PID – absolute error | Joint {joint}",  PID_COLOR,  "pid_error.png"),
    ]:
        if data is None:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(1, len(data) + 1)
        for i, d in enumerate(data):
            _draw_box(ax, x[i], d, color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, error_ylim_top)
        ax.set_yticks(pi_ticks)
        ax.set_yticklabels(pi_ticklabels)
        ax.axhline(np.pi, color="grey", ls=":", lw=1.2)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Absolute error (rad)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        _savefig(fig, os.path.join(out_dir, fname), fs, fw)
        print(f"  {fname} saved")

    if r_data is None or p_data is None:
        return

    n = min(len(r_data), len(p_data))
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, n + 1)
    for i in range(n):
        _draw_box(ax, x[i] - 0.2, r_data[i], RING_COLOR, width=0.35)
        _draw_box(ax, x[i] + 0.2, p_data[i], PID_COLOR,  width=0.35)

    ax.legend(handles=[Patch(color=RING_COLOR, label="Ring"),
                        Patch(color=PID_COLOR,  label="PID")])
    ax.set_xticks(x)
    ax.set_xticklabels((r_labels or p_labels)[:n])
    ax.set_ylim(0, error_ylim_top)
    ax.set_yticks(pi_ticks)
    ax.set_yticklabels(pi_ticklabels)
    ax.axhline(np.pi, color="grey", ls=":", lw=1.2)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Absolute error (rad)")
    ax.set_title(f"Ring vs PID – Joint {joint}")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, os.path.join(out_dir, "comparison_error.png"), fs, fw)
    print("  comparison_error.png saved")


def main(font_size=18, font_weight="bold"):
    matplotlib.rcParams["font.weight"]       = font_weight
    matplotlib.rcParams["axes.titleweight"]  = font_weight
    matplotlib.rcParams["axes.labelweight"]  = font_weight

    for joint in range(7):
        out_dir = os.path.join(RESULTS, f"joint_{joint}")
        os.makedirs(out_dir, exist_ok=True)

        _print_table(joint)

        print(f"Joint {joint} → {out_dir}")
        plot_fig01(joint, out_dir, font_size, font_weight)
        plot_fig02(joint, out_dir, font_size, font_weight)
        plot_fig03(joint, out_dir, font_size, font_weight)
        plot_fig04(joint, out_dir, font_size, font_weight)
        plot_fig05(joint, out_dir, font_size, font_weight)
        plot_app07(joint, out_dir, font_size, font_weight)
        plot_app08(joint, out_dir, font_size, font_weight)
        plot_app10(joint, out_dir, font_size, font_weight)
        plot_comparison(joint, out_dir, font_size, font_weight)
        print()


if __name__ == "__main__":
    main()
