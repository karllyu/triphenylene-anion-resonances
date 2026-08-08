import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
from RVP import read_csv, schlessinger, continued_fn

from RVP_plot import plot_alpha_trajectories;
from RVP_plot import plot_overlay_theta_on_alpha_trajectories;
from RVP_plot import plot_overlay_theta_on_alpha_trajectories_with_labels;
from RVP_plot import plot_single_alpha_and_theta;
from RVP_plot import find_cusp_candidate_pair;

##### PLOT OVERLAY THETA AND ALPHA #####

# plot_overlay_theta_on_alpha_trajectories_with_labels(
#     "A2_res1_alt.csv",
#     anchor_alpha=0.55,
#     theta_start=0,
#     theta_end=pi,
#     n_thetas=100,
#     alpha_grid_size=2000,
#     anchor_grid_size=3601,
#     label_count=0,
#     snap_to_nearest_alpha=False
# )

for alpha in np.linspace(0.4, 0.8, 15):
    plot_overlay_theta_on_alpha_trajectories_with_labels(
        "A2_res1_alt.csv",
        anchor_alpha=alpha,
        theta_start=0,
        theta_end=pi,
        n_thetas=100,
        alpha_grid_size=2000,
        anchor_grid_size=3601,
        label_count=0,
        snap_to_nearest_alpha=False
    )

##### PLOT SINGLE ALPHA AND THETA WITHIN A FOR LOOP #####

# for theta in np.linspace(0, 0.01, 10):
#     plot_single_alpha_and_theta(
#         "A2_res1_alt.csv",
#         alpha_value=0.710714,
#         theta_value=theta,
#         alpha_traj_start=0,
#         alpha_traj_end=1,
#         theta_traj_start=-1.5,
#         theta_traj_end=1.5,
#         theta_grid_size=36001,
#         alpha_grid_size=20000,
#         snap_to_nearest_alpha=False
#     )

# for alpha in np.linspace(1.243, 1.245, 10):
#     plot_single_alpha_and_theta(
#         "A2_res2.csv",
#         alpha_value=alpha,
#         theta_value=0.0497,
#         alpha_traj_start=0.5,
#         alpha_traj_end=1.5,
#         theta_traj_start=0,
#         theta_traj_end=pi/8,
#         theta_grid_size=3601,
#         alpha_grid_size=2000,
#         snap_to_nearest_alpha=False
#     )

##### PLOT SINGLE ALPHA AND THETA #####

# plot_single_alpha_and_theta(
#     "A2_res2.csv",
#     alpha_value=1.244085,
#     theta_value=0.04970,
#     alpha_traj_start=0.5,
#     alpha_traj_end=1.5,
#     theta_traj_start=0,
#     theta_traj_end=pi/8,
#     theta_grid_size=3601,
#     alpha_grid_size=2000,
#     snap_to_nearest_alpha=False
# )
