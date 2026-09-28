#!/usr/bin/env python3
"""
demo_graphs.py

Runs ONE trial of the ring-attractor decoder toward a goal joint angle and,
after every nest.Simulate call within that trial, saves:

  1) A composite PNG containing:
       LEFT   - robot camera snapshot
       RIGHT  - circular ring-attractor activity
       BOTTOM - unwrapped ring activity strip

  2) A raw, unmodified camera snapshot for that step

  3) A raw spike-count vector (.npy) for that step

At the end of the trial, the script builds a VPR-style summary figure:
  - a stack of sampled snapshots across the trajectory
  - a large first snapshot
  - a large last snapshot
  - arrows indicating progression

Outputs are written to:
    <src>/outputs/demo_graphs/joint_<idx>/frames/step_XXXX.png
    <src>/outputs/demo_graphs/joint_<idx>/snapshots/step_XXXX.png
    <src>/outputs/demo_graphs/joint_<idx>/spikes/step_XXXX.npy
    <src>/outputs/demo_graphs/joint_<idx>/snapshot_stack_summary.png
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

import rospy
from sensor_msgs.msg import Image as RosImage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_single_joint import TiagoDecoderAnalysis
from experiment_camera import CameraStream
from tiago_ring_controller.evaluation.plots import (
    circular_centroid_from_counts,
    evenly_spaced_indices,
    smooth_circular_counts,
    theta_to_ring_index,
)


BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs",
    "demo_graphs",
)


def _output_dir_for_joint(joint_idx: int) -> str:
    path = os.path.join(BASE_OUTPUT_DIR, f"joint_{joint_idx}")
    os.makedirs(path, exist_ok=True)
    return path


class DemoGraphRecorder(TiagoDecoderAnalysis):
    """
    Thin subclass of TiagoDecoderAnalysis.

    The only override is run_ring_simulation(): after calling the parent method
    we immediately capture:
      - the composite frame
      - the raw snapshot
      - the raw spike vector

    At the end, we build a VPR-style stacked snapshot summary instead of the
    old temporal heatmap.
    """

    _NEURIPS_RC = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "text.usetex": False,
    }

    def __init__(
        self,
        target_joint,
        visual_vmax=None,
        save_raw_spikes=True,
        make_snapshot_summary=True,
        summary_num_snapshots=10,
        **kwargs,
    ):
        """
        Parameters
        ----------
        target_joint : int
            Tiago arm joint index.
        visual_vmax : float or None
            Fixed maximum spike count for color normalization.
            Use a fixed value if you want every frame to be directly comparable.
            If None, the scale grows adaptively during the run.
        save_raw_spikes : bool
            Save per-step spike-count arrays as .npy files.
        make_snapshot_summary : bool
            Save the final VPR-style stacked snapshot summary.
        summary_num_snapshots : int
            Number of snapshots to sample across the trial for the final stack.
        kwargs :
            Passed to TiagoDecoderAnalysis.
        """
        super().__init__(target_joint, **kwargs)

        self._output_dir = _output_dir_for_joint(target_joint)
        self._frame_dir = os.path.join(self._output_dir, "frames")
        self._spike_dir = os.path.join(self._output_dir, "spikes")
        self._snapshot_dir = os.path.join(self._output_dir, "snapshots")

        os.makedirs(self._frame_dir, exist_ok=True)
        os.makedirs(self._spike_dir, exist_ok=True)
        os.makedirs(self._snapshot_dir, exist_ok=True)

        self._step_idx = 0
        self._visual_vmax = visual_vmax
        self._adaptive_vmax = 1.0

        self._save_raw_spikes = save_raw_spikes
        self._make_snapshot_summary = make_snapshot_summary
        self._summary_num_snapshots = max(2, int(summary_num_snapshots))

        self._snapshot_paths = []
        self._snapshot_step_indices = []

        self._camera = CameraStream(headless=True)
        resolved_topic = self._camera._resolve_topic()
        rospy.loginfo("[DemoGraphs] Subscribing to camera topic: %s", resolved_topic)

        self._cam_sub = rospy.Subscriber(
            resolved_topic,
            RosImage,
            self._camera._image_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

    # ------------------------------------------------------------------
    # Override: capture after every nest.Simulate call
    # ------------------------------------------------------------------

    def run_ring_simulation(self):
        result = super().run_ring_simulation()
        self._capture_snapshot()
        return result

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def _capture_snapshot(self):
        spike_counts = (
            np.asarray(self.r1_delta_spike_counts, dtype=float).copy()
            if self.r1_delta_spike_counts is not None
            else None
        )

        camera_frame_rgb = None
        if self._camera.latest_frame is not None:
            # CameraStream appears to store BGR, so convert to RGB.
            camera_frame_rgb = self._camera.latest_frame[:, :, ::-1].copy()

        # Save raw, unmodified camera snapshot
        snapshot_path = self._save_unmodified_snapshot(camera_frame_rgb, self._step_idx)
        self._snapshot_paths.append(snapshot_path)
        self._snapshot_step_indices.append(self._step_idx)

        if spike_counts is not None and spike_counts.size > 0:
            display_counts = self._smooth_circular(spike_counts)
            current_max = float(np.max(display_counts))
            self._adaptive_vmax = max(self._adaptive_vmax, current_max)

            if self._save_raw_spikes:
                np.save(
                    os.path.join(self._spike_dir, f"step_{self._step_idx:04d}.npy"),
                    spike_counts,
                )

        # Save composite figure
        fig = self._make_composite_figure(camera_frame_rgb, spike_counts, self._step_idx)
        fname = os.path.join(self._frame_dir, f"step_{self._step_idx:04d}.png")
        fig.savefig(fname, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        rospy.loginfo("[DemoGraphs] Saved composite frame %s", fname)
        if snapshot_path is not None:
            rospy.loginfo("[DemoGraphs] Saved raw snapshot %s", snapshot_path)

        self._step_idx += 1

    def _save_unmodified_snapshot(self, camera_frame_rgb, step_idx):
        """
        Save the raw camera image for this step, without any annotations.
        """
        if camera_frame_rgb is None:
            return None

        fname = os.path.join(self._snapshot_dir, f"step_{step_idx:04d}.png")
        plt.imsave(fname, camera_frame_rgb)
        return fname

    def finish(self):
        """
        Call this after run_trial() to save the stacked snapshot summary.
        """
        if self._make_snapshot_summary:
            self._save_snapshot_stack_summary()

    # ------------------------------------------------------------------
    # Figure construction for per-step composite
    # ------------------------------------------------------------------

    def _make_composite_figure(self, camera_frame_rgb, spike_counts, step_idx):
        """
        Returns a publication-style figure with:
          - camera frame
          - ring annulus
          - unwrapped activity strip
        """
        with matplotlib.rc_context(self._NEURIPS_RC):
            fig = plt.figure(figsize=(12.0, 6.4), facecolor="white")

            gs = fig.add_gridspec(
                nrows=2,
                ncols=2,
                height_ratios=[4.8, 1.0],
                width_ratios=[1.35, 1.0],
                hspace=0.18,
                wspace=0.12,
            )

            ax_cam = fig.add_subplot(gs[0, 0])
            self._draw_camera_panel(ax_cam, camera_frame_rgb, step_idx)

            ax_ring = fig.add_subplot(gs[0, 1], projection="polar")
            self._draw_ring_annulus(ax_ring, spike_counts)

            ax_strip = fig.add_subplot(gs[1, :])
            self._draw_ring_strip(ax_strip, spike_counts)

        return fig

    def _draw_camera_panel(self, ax, camera_frame_rgb, step_idx):
        ax.set_xticks([])
        ax.set_yticks([])

        if camera_frame_rgb is not None:
            ax.imshow(camera_frame_rgb, aspect="auto")
        else:
            ax.set_facecolor("#f2f2f2")
            ax.text(
                0.5,
                0.5,
                "No camera frame",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="#666666",
            )

        ax.set_title("Robot camera frame", fontsize=14, fontweight="bold", pad=6)

        ax.text(
            0.02,
            0.98,
            f"step {step_idx:04d}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#111111",
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="#cccccc",
                alpha=0.92,
            ),
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#bbbbbb")

    def _draw_ring_annulus(self, ax, spike_counts):
        ax.set_facecolor("white")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)

        if spike_counts is None or len(spike_counts) == 0:
            ax.set_axis_off()
            ax.text(0, 0, "No spike data", ha="center", va="center", fontsize=12)
            return

        raw_counts = np.asarray(spike_counts, dtype=float)
        counts = self._smooth_circular(raw_counts)
        N = len(counts)

        cmap = self._spike_cmap()
        vmax = self._get_vmax(counts)
        norm = Normalize(vmin=0.0, vmax=vmax)

        r_inner = 0.76
        r_outer = 0.98
        ring_width = r_outer - r_inner

        theta_edges = np.linspace(0.0, 2.0 * np.pi, N + 1)
        dtheta = theta_edges[1] - theta_edges[0]

        # Draw one bar per neuron so the ring does not look like one solid block.
        for i in range(N):
            theta = theta_edges[i]
            color = cmap(norm(counts[i]))
            ax.bar(
                theta,
                ring_width,
                width=dtheta * 0.96,
                bottom=r_inner,
                align="edge",
                color=color,
                edgecolor="white",
                linewidth=0.25,
                zorder=2,
            )

        theta_fill = np.linspace(0.0, 2.0 * np.pi, 500)

        # Inner hole and borders.
        ax.fill_between(theta_fill, 0.0, r_inner, color="white", zorder=3)
        ax.plot(
            theta_fill,
            np.full_like(theta_fill, r_inner),
            color="#999999",
            linewidth=0.8,
            zorder=4,
        )
        ax.plot(
            theta_fill,
            np.full_like(theta_fill, r_outer),
            color="#555555",
            linewidth=0.9,
            zorder=4,
        )

        # Peak marker and circular centroid marker.
        peak_idx = int(np.argmax(counts))
        peak_theta = 2.0 * np.pi * peak_idx / N

        centroid_theta = self._circular_centroid(counts)
        centroid_idx = self._theta_to_index(centroid_theta, N)

        ax.plot(
            [peak_theta, peak_theta],
            [r_inner - 0.025, r_outer + 0.025],
            color="#111111",
            linewidth=1.2,
            zorder=5,
        )
        ax.scatter(
            [peak_theta],
            [r_outer + 0.025],
            s=24,
            color="#111111",
            zorder=6,
            label="peak",
        )

        ax.plot(
            [centroid_theta, centroid_theta],
            [r_inner - 0.035, r_outer + 0.035],
            color="#2ca25f",
            linewidth=1.4,
            linestyle="--",
            zorder=5,
        )

        ax.text(
            0.5,
            0.5,
            f"peak={peak_idx}\ncentroid={centroid_idx:.1f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#333333",
        )

        ax.set_ylim(0.0, 1.10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)

        ax.set_title("Ring-attractor activity", fontsize=14, fontweight="bold", pad=8)

        # Neuron index labels.
        cardinal_idx = [0, N // 4, N // 2, 3 * N // 4]
        for idx in cardinal_idx:
            theta_c = 2.0 * np.pi * idx / N
            ax.text(
                theta_c,
                1.055,
                f"n={idx}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#222222",
                zorder=7,
            )

        # Slim colorbar.
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(
            sm,
            ax=ax,
            pad=0.08,
            fraction=0.046,
            shrink=0.82,
        )
        cbar.set_label("spikes / step", fontsize=10)
        cbar.ax.tick_params(labelsize=9, length=2)
        cbar.outline.set_visible(False)

    def _draw_ring_strip(self, ax, spike_counts):
        ax.set_facecolor("white")

        if spike_counts is None or len(spike_counts) == 0:
            ax.axis("off")
            return

        raw_counts = np.asarray(spike_counts, dtype=float)
        counts = self._smooth_circular(raw_counts)
        N = len(counts)

        cmap = self._spike_cmap()
        vmax = self._get_vmax(counts)

        strip = counts[np.newaxis, :]

        ax.imshow(
            strip,
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
            extent=[0, N, 0, 1],
        )

        peak_idx = int(np.argmax(counts))
        centroid_theta = self._circular_centroid(counts)
        centroid_idx = self._theta_to_index(centroid_theta, N)

        ax.axvline(
            peak_idx + 0.5,
            color="black",
            linewidth=1.0,
            alpha=0.85,
            label="peak",
        )
        ax.axvline(
            centroid_idx,
            color="#2ca25f",
            linewidth=1.2,
            linestyle="--",
            alpha=0.95,
            label="centroid",
        )

        ax.set_yticks([])
        ax.set_xlim(0, N)
        ax.set_xlabel("Ring neuron index")
        ax.set_title("Unwrapped ring activity", fontsize=12, fontweight="bold", pad=4)
        ax.set_xticks(np.linspace(0, N, 5))

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_edgecolor("#bbbbbb")

    # ------------------------------------------------------------------
    # VPR-style stacked snapshot summary
    # ------------------------------------------------------------------

    def _save_snapshot_stack_summary(self):
        valid_entries = [
            (step_idx, path)
            for step_idx, path in zip(self._snapshot_step_indices, self._snapshot_paths)
            if path is not None and os.path.exists(path)
        ]

        if len(valid_entries) < 2:
            rospy.logwarn("[DemoGraphs] Not enough snapshots for stack summary.")
            return

        selected_local_indices = self._select_evenly_spaced_indices(
            num_items=len(valid_entries),
            num_select=min(self._summary_num_snapshots, len(valid_entries)),
        )
        sampled_entries = [valid_entries[i] for i in selected_local_indices]

        first_step, first_path = valid_entries[0]
        last_step, last_path = valid_entries[-1]

        first_img = plt.imread(first_path)
        last_img = plt.imread(last_path)

        with matplotlib.rc_context(self._NEURIPS_RC):
            fig = plt.figure(figsize=(13.2, 6.2), facecolor="white")

            # Background axis for arrows / lines / figure-wide text
            ax_bg = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=0)
            ax_bg.set_xlim(0.0, 1.0)
            ax_bg.set_ylim(0.0, 1.0)
            ax_bg.axis("off")

            fig.text(
                0.5,
                0.965,
                "Snapshot trajectory summary",
                ha="center",
                va="top",
                fontsize=18,
                fontweight="bold",
            )
            fig.text(
                0.5,
                0.93,
                f"{len(sampled_entries)} sampled snapshots across {len(valid_entries)} recorded steps",
                ha="center",
                va="top",
                fontsize=11,
                color="#444444",
            )

            # Large panels: first and last
            left_pos = [0.03, 0.72, 0.22, 0.23]
            right_pos = [0.75, 0.54, 0.22, 0.23]

            self._add_image_panel(
                fig=fig,
                image=first_img,
                rect=left_pos,
                title="First snapshot",
                step_idx=first_step,
                border_color="#355C7D",
                border_width=2.0,
                alpha=1.0,
                zorder=3,
                corner_label="start",
            )

            self._add_image_panel(
                fig=fig,
                image=last_img,
                rect=right_pos,
                title="Last snapshot",
                step_idx=last_step,
                border_color="#355C7D",
                border_width=2.0,
                alpha=1.0,
                zorder=3,
                corner_label="end",
            )

            # Central stacked snapshots
            n_stack = len(sampled_entries)
            small_w = 0.12
            small_h = 0.14
            x_start = 0.34
            x_end = 0.58
            y_start = 0.30
            y_end = 0.82

            stack_centers = []

            for k, (step_idx, path) in enumerate(sampled_entries):
                img = plt.imread(path)
                t = 0.0 if n_stack == 1 else float(k) / float(n_stack - 1)

                cx = x_start + t * (x_end - x_start)
                cy = y_start + t * (y_end - y_start)
                rect = [cx - small_w / 2.0, cy - small_h / 2.0, small_w, small_h]

                alpha = 0.35 + 0.60 * t
                stack_centers.append((cx, cy))

                self._add_image_panel(
                    fig=fig,
                    image=img,
                    rect=rect,
                    title=None,
                    step_idx=step_idx,
                    border_color="#4C6A92",
                    border_width=1.6,
                    alpha=alpha,
                    zorder=4 + k,
                    corner_label=f"{k + 1}",
                    small=True,
                )

            # Thin connecting polyline through the sampled stack
            if len(stack_centers) >= 2:
                xs = [p[0] for p in stack_centers]
                ys = [p[1] for p in stack_centers]
                ax_bg.plot(xs, ys, color="black", linewidth=0.7, alpha=0.7, zorder=1)

            # Arrow from first large image to the first sampled stack image
            if len(stack_centers) > 0:
                arrow1 = FancyArrowPatch(
                    (left_pos[0] + left_pos[2] + 0.005, left_pos[1] + 0.09),
                    (stack_centers[0][0] - 0.07, stack_centers[0][1] + 0.02),
                    arrowstyle="simple",
                    mutation_scale=18,
                    linewidth=0.0,
                    color="black",
                    alpha=0.95,
                )
                ax_bg.add_patch(arrow1)

                # Arrow from last sampled stack image to the last large image
                arrow2 = FancyArrowPatch(
                    (stack_centers[-1][0] + 0.06, stack_centers[-1][1] - 0.01),
                    (right_pos[0] - 0.005, right_pos[1] + 0.08),
                    arrowstyle="simple",
                    mutation_scale=18,
                    linewidth=0.0,
                    color="black",
                    alpha=0.95,
                )
                ax_bg.add_patch(arrow2)

            # Optional guide lines similar to VPR figure
            if len(stack_centers) > 0:
                ax_bg.plot(
                    [left_pos[0] + left_pos[2], stack_centers[0][0] - 0.02],
                    [left_pos[1] + 0.04, stack_centers[0][1] - 0.05],
                    color="black",
                    linewidth=0.6,
                    alpha=0.7,
                )
                ax_bg.plot(
                    [stack_centers[-1][0] + 0.02, right_pos[0]],
                    [stack_centers[-1][1] - 0.02, right_pos[1] + 0.01],
                    color="black",
                    linewidth=0.6,
                    alpha=0.7,
                )

            fname = os.path.join(self._output_dir, "snapshot_stack_summary.png")
            fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

        rospy.loginfo("[DemoGraphs] Saved snapshot stack summary %s", fname)

    def _add_image_panel(
        self,
        fig,
        image,
        rect,
        title=None,
        step_idx=None,
        border_color="#355C7D",
        border_width=1.5,
        alpha=1.0,
        zorder=1,
        corner_label=None,
        small=False,
    ):
        """
        Helper to place an image in a manually positioned axis.
        """
        ax = fig.add_axes(rect)
        ax.set_zorder(zorder)
        ax.imshow(image, aspect="auto", alpha=alpha)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(border_width)
            spine.set_edgecolor(border_color)

        if title is not None:
            ax.set_title(title, fontsize=(11 if small else 12), fontweight="bold", pad=4)

        if step_idx is not None:
            ax.text(
                0.02,
                0.98,
                f"step {step_idx:04d}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=(7 if small else 9),
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    fc="black",
                    ec="none",
                    alpha=0.65,
                ),
            )

        if corner_label is not None:
            ax.text(
                0.02,
                0.06,
                str(corner_label),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=(8 if small else 10),
                fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    fc="#1f2d3d",
                    ec="none",
                    alpha=0.78,
                ),
            )

        return ax

    def _select_evenly_spaced_indices(self, num_items, num_select):
        """
        Select `num_select` indices distributed as evenly as possible from
        [0, ..., num_items - 1], always including the first and last.
        """
        return evenly_spaced_indices(num_items, num_select)

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------

    def _spike_cmap(self):
        """
        Light-to-dark sequential colormap for nonnegative spike counts.
        Low activity is near white so inactive neurons do not become a solid block.
        """
        return LinearSegmentedColormap.from_list(
            "spike_rate_clean",
            [
                "#f7fbff",
                "#deebf7",
                "#9ecae1",
                "#3182bd",
                "#54278f",
                "#b10026",
            ],
            N=256,
        )

    def _get_vmax(self, counts):
        if self._visual_vmax is not None:
            return max(float(self._visual_vmax), 1.0)

        # Adaptive fallback. For perfect frame-to-frame comparability, pass
        # visual_vmax explicitly in the constructor.
        self._adaptive_vmax = max(self._adaptive_vmax, float(np.max(counts)), 1.0)
        return self._adaptive_vmax

    def _smooth_circular(self, counts):
        """
        Small circular smoothing used only for visualization.
        The raw spike counts are still saved unchanged.
        """
        return smooth_circular_counts(counts)

    def _circular_centroid(self, counts):
        """
        Circular centroid of nonnegative activity.
        Returns theta in [0, 2*pi).
        """
        return circular_centroid_from_counts(counts)

    def _theta_to_index(self, theta, N):
        return theta_to_ring_index(theta, N)


def main():
    rospy.init_node("demo_graphs", anonymous=True)

    joint_idx = 3

    rospy.loginfo("[DemoGraphs] Initialising recorder for joint %d …", joint_idx)

    recorder = DemoGraphRecorder(
        target_joint=joint_idx,
        use_manual_limits=False,

        # Set this to a fixed value for frame-to-frame comparable colors.
        # Good starting values are often 10, 25, 50, or 100 depending on
        # your spike-count scale.
        visual_vmax=None,

        save_raw_spikes=True,

        # New VPR-style summary options
        make_snapshot_summary=True,
        summary_num_snapshots=10,
    )

    current_q = recorder.get_joint_position()[joint_idx]
    goal_q = current_q + np.deg2rad(15.0)

    rospy.loginfo(
        "[DemoGraphs] q_start=%.4f rad  goal=%.4f rad",
        current_q,
        goal_q,
    )

    recorder.run_trial(goal_q)
    recorder.finish()

    rospy.loginfo(
        "[DemoGraphs] Done. %d frames -> %s",
        recorder._step_idx,
        recorder._output_dir,
    )


if __name__ == "__main__":
    main()
