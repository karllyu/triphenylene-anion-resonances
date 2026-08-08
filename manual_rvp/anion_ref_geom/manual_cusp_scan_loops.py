import numpy as np
from scipy.constants import pi

from RVP_plot_og import (
    plot_overlay_theta_on_alpha_trajectories_with_labels,
    plot_single_alpha_and_theta,
)


# -----------------------------
# Manual cusp scan configuration
# -----------------------------
POINTS_FILE = "A2_res1.csv"

# Stage toggles
RUN_OVERLAY_ALPHA_SWEEP = True
RUN_SINGLE_THETA_SWEEP = True
RUN_SINGLE_ALPHA_SWEEP = True


# 1) Coarse sweep of anchor_alpha using overlay plots
OVERLAY_ALPHA_VALUES = np.linspace(0.70, 1.30, 13)
OVERLAY_THETA_START = 0.0
OVERLAY_THETA_END = pi
OVERLAY_N_THETAS = 100
OVERLAY_ALPHA_GRID_SIZE = 300
OVERLAY_ANCHOR_GRID_SIZE = 721
OVERLAY_LABEL_COUNT = 0
OVERLAY_SNAP_TO_NEAREST_ALPHA = False


# 2) Hold alpha fixed, sweep theta with single-plot
THETA_SWEEP_ALPHA = 1.00
THETA_SWEEP_VALUES = np.linspace(0.15, 0.35, 31)
THETA_SWEEP_ALPHA_TRAJ_START = 0.80
THETA_SWEEP_ALPHA_TRAJ_END = 1.20
THETA_SWEEP_THETA_TRAJ_START = 0.0
THETA_SWEEP_THETA_TRAJ_END = pi
THETA_SWEEP_THETA_GRID_SIZE = 3601
THETA_SWEEP_ALPHA_GRID_SIZE = 2000
THETA_SWEEP_SNAP_TO_NEAREST_ALPHA = False


# 3) Hold theta fixed, sweep alpha with single-plot
ALPHA_SWEEP_THETA = 0.25
ALPHA_SWEEP_VALUES = np.linspace(0.90, 1.10, 31)
ALPHA_SWEEP_ALPHA_TRAJ_START = 0.80
ALPHA_SWEEP_ALPHA_TRAJ_END = 1.20
ALPHA_SWEEP_THETA_TRAJ_START = 0.0
ALPHA_SWEEP_THETA_TRAJ_END = pi
ALPHA_SWEEP_THETA_GRID_SIZE = 3601
ALPHA_SWEEP_ALPHA_GRID_SIZE = 2000
ALPHA_SWEEP_SNAP_TO_NEAREST_ALPHA = False


if __name__ == "__main__":
    # Stage 1: overlay family for each anchor alpha
    if RUN_OVERLAY_ALPHA_SWEEP:
        for anchor_alpha in OVERLAY_ALPHA_VALUES:
            print(f"[Overlay sweep] anchor_alpha = {anchor_alpha:.6f}")
            plot_overlay_theta_on_alpha_trajectories_with_labels(
                POINTS_FILE,
                anchor_alpha=float(anchor_alpha),
                theta_start=OVERLAY_THETA_START,
                theta_end=OVERLAY_THETA_END,
                n_thetas=OVERLAY_N_THETAS,
                alpha_grid_size=OVERLAY_ALPHA_GRID_SIZE,
                anchor_grid_size=OVERLAY_ANCHOR_GRID_SIZE,
                label_count=OVERLAY_LABEL_COUNT,
                snap_to_nearest_alpha=OVERLAY_SNAP_TO_NEAREST_ALPHA,
            )

    # Stage 2: theta refinement at fixed alpha
    if RUN_SINGLE_THETA_SWEEP:
        for theta_value in THETA_SWEEP_VALUES:
            print(
                f"[Theta sweep] alpha = {THETA_SWEEP_ALPHA:.6f}, "
                f"theta = {theta_value:.6f}"
            )
            plot_single_alpha_and_theta(
                POINTS_FILE,
                alpha_value=float(THETA_SWEEP_ALPHA),
                theta_value=float(theta_value),
                alpha_traj_start=THETA_SWEEP_ALPHA_TRAJ_START,
                alpha_traj_end=THETA_SWEEP_ALPHA_TRAJ_END,
                theta_traj_start=THETA_SWEEP_THETA_TRAJ_START,
                theta_traj_end=THETA_SWEEP_THETA_TRAJ_END,
                theta_grid_size=THETA_SWEEP_THETA_GRID_SIZE,
                alpha_grid_size=THETA_SWEEP_ALPHA_GRID_SIZE,
                snap_to_nearest_alpha=THETA_SWEEP_SNAP_TO_NEAREST_ALPHA,
            )

    # Stage 3: alpha refinement at fixed theta
    if RUN_SINGLE_ALPHA_SWEEP:
        for alpha_value in ALPHA_SWEEP_VALUES:
            print(
                f"[Alpha sweep] alpha = {alpha_value:.6f}, "
                f"theta = {ALPHA_SWEEP_THETA:.6f}"
            )
            plot_single_alpha_and_theta(
                POINTS_FILE,
                alpha_value=float(alpha_value),
                theta_value=float(ALPHA_SWEEP_THETA),
                alpha_traj_start=ALPHA_SWEEP_ALPHA_TRAJ_START,
                alpha_traj_end=ALPHA_SWEEP_ALPHA_TRAJ_END,
                theta_traj_start=ALPHA_SWEEP_THETA_TRAJ_START,
                theta_traj_end=ALPHA_SWEEP_THETA_TRAJ_END,
                theta_grid_size=ALPHA_SWEEP_THETA_GRID_SIZE,
                alpha_grid_size=ALPHA_SWEEP_ALPHA_GRID_SIZE,
                snap_to_nearest_alpha=ALPHA_SWEEP_SNAP_TO_NEAREST_ALPHA,
            )
