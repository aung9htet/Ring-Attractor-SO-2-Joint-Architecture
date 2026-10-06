#!/usr/bin/env python3
"""Draw the wiring of architecture v3 (two rings) and v4 (three rings) as it is built.

    python3 -B scripts/draw_architecture.py --out ../../docs/results/paper

Writes ``architecture_v3`` and ``architecture_v4`` as PNG and PDF.  The
diagrams follow ``graph.templates.two_ring_single_joint`` (bounded) and
``three_ring_single_joint``; docs/architecture.md explains every part.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, SURFACE, FILL = "#0b0b0b", "#52514e", "#fcfcfb", "#f0efec"
OLD, NEW = "#2a78d6", "#eb6834"          # parts v3 already has / parts v4 adds


def box(axis, x, y, w, h, text, color=MUTED, size=8.5):
    axis.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                  facecolor=FILL, edgecolor=color, linewidth=1.6))
    axis.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.25)


def ring(axis, x, y, text, color, radius=6.0):
    axis.add_patch(Circle((x, y), radius, facecolor=SURFACE, edgecolor=color, linewidth=5))
    axis.text(x, y, text, ha="center", va="center", fontsize=8.5, color=INK, linespacing=1.25)


def arrow(axis, start, end, text=None, at=None, color=MUTED, bend=0.0, size=7.5):
    """An arrow; ``text`` is drawn at ``at`` (default: above the middle) on the surface colour."""

    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.5, color=color,
                                   connectionstyle="arc3,rad=%g" % bend, shrinkA=0, shrinkB=0))
    if text:
        x, y = at if at is not None else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 1.8)
        axis.text(x, y, text, ha="center", va="center", fontsize=size, color=MUTED, linespacing=1.2,
                  bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))


def canvas(title, width, height):
    figure, axis = plt.subplots(figsize=(11.0, 11.0 * height / width))
    figure.patch.set_facecolor(SURFACE)
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.text(1, height - 3.5, title, fontsize=10.5, color=INK, va="center", linespacing=1.3)
    return figure, axis


def draw_v3():
    figure, axis = canvas("Architecture v3: two rings.\nThe measured joint angle enters the state ring once; after that the trial is open loop.", 130, 62)
    box(axis, 9, 42, 13, 6, "goal angle")
    ring(axis, 30, 42, "goal ring\nR2", OLD)
    ring(axis, 30, 18, "state ring\nR1", OLD)
    box(axis, 58, 30, 21, 10, "ramp comparator\n(Fourier readouts,\nwarm / cold, left / right)", OLD, 8)
    box(axis, 84, 30, 15, 10, "gain pair\nleft / right\n(gated by R1)", OLD)
    box(axis, 106, 30, 14, 9, "decoder\n(dead zone,\ngain a rho / dt)", OLD, 8)
    box(axis, 120, 8, 13, 7, "robot joint\n(Gazebo)")
    arrow(axis, (15.5, 42), (24, 42), "once", at=(19.7, 44))
    arrow(axis, (36.2, 40.5), (48.5, 33), "features", at=(45, 39.5))
    arrow(axis, (36.2, 19.5), (48.5, 27), "features", at=(40.5, 25.5))
    arrow(axis, (68.5, 30), (76.5, 30), "left / right", at=(72.6, 37.3))
    arrow(axis, (91.5, 30), (99, 30), "spikes", at=(95.2, 37.3))
    arrow(axis, (82, 25), (36.4, 17), "shifts the bump one index per spike", at=(62, 17.2), color=OLD, bend=-0.1)
    arrow(axis, (109, 25.5), (118, 11.5), "joint velocity", at=(120.5, 19.5))
    arrow(axis, (113.5, 7), (33.5, 12.8), "measured angle: encoder, once at the start", at=(72, 4.2), bend=-0.07)
    return figure


def draw_v4():
    figure, axis = canvas("Architecture v4: three rings.\nThe goal pair moves the prediction and the joint; the body pair pulls the prediction back to the measured joint.", 130, 88)
    box(axis, 9, 68, 13, 6, "goal angle")
    ring(axis, 30, 68, "goal ring\nG", OLD)
    ring(axis, 30, 40, "predictive\nring P", OLD)
    ring(axis, 30, 12, "state ring\nS", NEW)
    axis.text(11, 40, "starts at the\nmeasured angle", ha="center", va="center", fontsize=7.5, color=MUTED)
    box(axis, 58, 60, 21, 10, "goal comparison\nramp comparator (far)\n+ projection of G (near)", OLD, 8)
    box(axis, 84, 60, 15, 10, "goal gain pair\nleft / right\n(gated by P)", OLD)
    box(axis, 58, 20, 21, 10, "body comparison\nramp comparator (far)\n+ projection of S (near)", NEW, 8)
    box(axis, 84, 20, 15, 10, "body gain pair\nleft / right\n(gated by P)", NEW)
    box(axis, 106, 60, 14, 9, "decoder\n(loop gain\nkappa rho / dt)", OLD, 8)
    box(axis, 120, 34, 13, 7, "robot joint\n(Gazebo)")
    arrow(axis, (15.5, 68), (24, 68), "held", at=(19.7, 70))
    arrow(axis, (36.2, 66.5), (48.5, 62), "features, spikes", at=(44.5, 67.5))
    arrow(axis, (34.5, 44), (48.5, 57), "features", at=(37.5, 52))
    arrow(axis, (34.5, 36), (48.5, 23), "features", at=(37.5, 28))
    arrow(axis, (36.2, 13.5), (48.5, 18), "features, spikes", at=(44.5, 12.5), color=NEW)
    arrow(axis, (68.5, 60), (76.5, 60), "left / right", at=(72.6, 67.3))
    arrow(axis, (68.5, 20), (76.5, 20), "left / right", at=(72.6, 27.3), color=NEW)
    arrow(axis, (91.5, 60), (99, 60), "spikes", at=(95.2, 67.3))
    arrow(axis, (82, 55), (36.4, 42), "moves the bump toward the goal\n(efference copy of the motor command)", at=(66, 47.5), color=OLD, bend=-0.08)
    arrow(axis, (82, 25), (36.4, 38), "pulls the bump back to the body\n(feedback weight five times larger)", at=(66, 32.5), color=NEW, bend=0.08)
    arrow(axis, (109, 55.5), (119, 37.5), "joint velocity", at=(121.5, 47))
    arrow(axis, (118, 30.5), (36.3, 9.5), "measured angle: encoder,\nwhenever the bump is one index off", at=(98, 9), color=NEW, bend=-0.22)
    axis.plot([96, 100], [77, 77], color=OLD, linewidth=4)
    axis.text(101.5, 77, "as in v3", fontsize=8.5, color=INK, va="center")
    axis.plot([112, 116], [77, 77], color=NEW, linewidth=4)
    axis.text(117.5, 77, "new in v4", fontsize=8.5, color=INK, va="center")
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for name, figure in (("architecture_v3", draw_v3()), ("architecture_v4", draw_v4())):
        for extension in ("png", "pdf"):
            figure.savefig(os.path.join(args.out, "%s.%s" % (name, extension)), dpi=170, bbox_inches="tight", facecolor=SURFACE)
        plt.close(figure)
        print("written:", os.path.join(args.out, name + ".png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
