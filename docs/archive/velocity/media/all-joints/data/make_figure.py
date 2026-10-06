"""Figure of docs/velocity/all-joints.md: python3 make_figure.py with docs/velocity/media mounted at /out."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {0.25: "#2a78d6", 0.5: "#eb6834", 0.75: "#1baf7a"}        # reference categorical slots 1-3, fixed per start pose
t = json.load(open("/out/all-joints/data/gazebo_all_joints_3x.json"))["trials"]
figure, axes = plt.subplots(2, 4, figsize=(13, 6.4))
for j, axis in enumerate(axes.flat):
    if j == 7:
        axis.axis("off")
        for k, (s, color) in enumerate(COLORS.items()):
            axis.plot([0.08], [0.8 - 0.14 * k], "o", color=color, markersize=8, markeredgecolor="white")
            axis.text(0.16, 0.8 - 0.14 * k, "start at %.2f of the range" % s, va="center", color=INK, fontsize=10)
        axis.plot([0.03, 0.13], [0.36, 0.36], color=MUTED, linewidth=1)
        axis.text(0.16, 0.36, "final = goal", va="center", color=INK, fontsize=10)
        axis.set_xlim(0, 1); axis.set_ylim(0, 1)
        continue
    rows = [r for r in t if r["joint"] == j]
    lo = min(min(r["goal_rad"], r["final_rad"], r["start_rad"]) for r in rows); hi = max(max(r["goal_rad"], r["final_rad"], r["start_rad"]) for r in rows)
    axis.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1)
    for s, color in COLORS.items():
        rr = [r for r in rows if abs(r["start_fraction"] - s) < 0.02]
        axis.plot([r["goal_rad"] for r in rr], [r["final_rad"] for r in rr], "o", color=color, markersize=7, markeredgecolor="white", markeredgewidth=1)
    err = np.abs([r["goal_error_rad"] for r in rows])
    axis.set_title("arm_%d   error median %.3f, max %.3f" % (j + 1, np.median(err), err.max()), loc="left", fontsize=9.5, color=INK)
    axis.grid(True, color=GRID, linewidth=0.8); axis.set_axisbelow(True)
    for side in ("top", "right"): axis.spines[side].set_visible(False)
    for side in ("left", "bottom"): axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=8)
    if j >= 4: axis.set_xlabel("goal (rad)", color=INK)
    if j % 4 == 0: axis.set_ylabel("final joint angle (rad)", color=INK)
figure.suptitle("Gazebo, bounded ring at 3x: where each joint ended against its goal (167 trials; goal error in rad)", x=0.01, ha="left", fontsize=12, color=INK)
figure.tight_layout()
figure.savefig("/out/all-joints/final_vs_goal.png", dpi=150)
