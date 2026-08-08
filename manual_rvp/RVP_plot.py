import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
from scipy.spatial import cKDTree
from RVP import read_csv, schlessinger, continued_fn

def plot_alpha_trajectories(points, theta_start=0.0, theta_end=pi/2, n_thetas=91, alpha_grid_size=200):
    # 1) Load original data (keep these as the base for continuation)
    alphas, energies = read_csv(points)
    alphas = np.asarray(alphas, dtype=float)
    energies = np.asarray(energies, dtype=complex)

    # Ensure α are sorted together with their energies
    order = np.argsort(alphas)
    alphas = alphas[order]
    energies = energies[order]

    # 2) Build Schlessinger coeffs once from the measured data
    coeffs = schlessinger(alphas, energies)

    # 3) Choose the θ’s to trace (inclusive endpoints)
    thetas = np.linspace(theta_start, theta_end, n_thetas)

    # 4) A continuous α line to march along (you can also use the original sample alphas)
    a_min, a_max = float(alphas[0]), float(alphas[-1])
    alpha_line = np.linspace(a_min, a_max, alpha_grid_size)

    # 5) For each fixed θ, vary α and plot the trajectory of f(η(α))
    for theta in thetas:
        etas = alpha_line * np.exp(1j * theta)
        cont_vals = np.asarray([continued_fn(alphas, energies, coeffs, eta) for eta in etas])

        plt.figure(figsize=(10, 5))
        plt.plot(cont_vals.real, cont_vals.imag, marker='o', ms=2, lw=1)

        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.title(f"Alpha trajectory at θ = {theta:.3g} rad")
        plt.grid(True)


        plt.show()

def plot_overlay_theta_on_alpha_trajectories(
    points,
    anchor_alpha=0,               # the α whose θ-trajectory you want to overlay
    theta_start=0.0,
    theta_end=pi/2,
    n_thetas=91,                  # how many θ values for the α-trajectories
    alpha_grid_size=200,          # how many α samples along each α-trajectory curve
    anchor_grid_size=361,         # how many θ samples for the anchor θ-trajectory
    snap_to_nearest_alpha=True,   # optionally snap anchor_alpha to the nearest input α
    show_start_markers=True       # mark the start of each curve
):


    alphas, energies = read_csv(points)
    alphas = np.asarray(alphas, dtype=float)
    energies = np.asarray(energies, dtype=complex)

    order = np.argsort(alphas)
    alphas = alphas[order]
    energies = energies[order]

    coeffs = schlessinger(alphas, energies)

    thetas = np.linspace(theta_start, theta_end, n_thetas)
    a_min, a_max = float(alphas[0]), float(alphas[-1])
    alpha_line = np.linspace(a_min, a_max, alpha_grid_size)

    plt.figure(figsize=(10, 5))
    for theta in thetas:
        etas = alpha_line * np.exp(1j * theta)
        cont_vals = np.asarray([continued_fn(alphas, energies, coeffs, eta) for eta in etas])
        plt.plot(cont_vals.real, cont_vals.imag, lw=1, alpha=0.8)
        if show_start_markers:
            plt.plot(cont_vals.real[0], cont_vals.imag[0], marker='o', ms=3)

    if snap_to_nearest_alpha:
        # snap to the nearest provided α (helps with exact interpolation at endpoints)
        idx = int(np.argmin(np.abs(alphas - anchor_alpha)))
        anchor_alpha = float(alphas[idx])

    thetas_anchor = np.linspace(theta_start, theta_end, anchor_grid_size)
    etas_anchor = anchor_alpha * np.exp(1j * thetas_anchor)
    cont_anchor = np.asarray([continued_fn(alphas, energies, coeffs, eta) for eta in etas_anchor])

    plt.plot(
        cont_anchor.real, cont_anchor.imag,
        lw=2.5, label=f"θ-trajectory at α={anchor_alpha:g}"
    )
    if show_start_markers:
        plt.plot(cont_anchor.real[0], cont_anchor.imag[0], marker='o', ms=5)

    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title(f"α-trajectories for θ ∈ [{theta_start:.3g}, {theta_end:.3g}] + overlay at α={anchor_alpha:g}")
    plt.grid(True)
    plt.legend(loc="best")
    # plt.axis("equal")  # uncomment if you want equal aspect in Re/Im

    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

def plot_overlay_theta_on_alpha_trajectories_with_labels(
    points,
    anchor_alpha=0,
    theta_start=0.0,
    theta_end=pi/2,
    n_thetas=91,
    alpha_grid_size=200,
    anchor_grid_size=361,
    snap_to_nearest_alpha=True,
    label_count=8,              # place ~label_count text labels along the family of curves
    label_fmt=r"$\theta={:.2f}$ rad"
):
    """
    Draw α-trajectories for a sweep of θ (colored by θ) AND overlay the θ-trajectory at anchor_alpha.
    Adds readable labels for a subset of α-trajectories to avoid legend clutter.
    """
    # 1) Load and sort
    alphas, energies = read_csv(points)
    alphas = np.asarray(alphas, float)
    energies = np.asarray(energies, complex)
    order = np.argsort(alphas)
    alphas = alphas[order]
    energies = energies[order]

    # 2) Build coefficients once
    coeffs = schlessinger(alphas, energies)

    # 3) Grids
    thetas = np.linspace(theta_start, theta_end, n_thetas)
    a_min, a_max = float(alphas[0]), float(alphas[-1])
    alpha_line = np.linspace(a_min, a_max, alpha_grid_size)

    # 4) Plot α-trajectories colored by θ
    cmap = plt.get_cmap("copper")
    fig, ax = plt.subplots(figsize=(10, 5))

    # choose which θ’s to label to avoid clutter
    label_indices = np.linspace(0, n_thetas - 1, min(label_count, n_thetas)).round().astype(int)
    for i, theta in enumerate(thetas):
        etas = alpha_line * np.exp(1j * theta)
        cont_vals = np.asarray([continued_fn(alphas, energies, coeffs, eta) for eta in etas])

        c = cmap((theta - theta_start) / max(1e-12, (theta_end - theta_start)))
        ax.plot(cont_vals.real, cont_vals.imag, lw=1.2, color=c)

        # place a tiny label near the starting point for a subset of curves
        if i in set(label_indices):
            xr, xi = cont_vals.real[0], cont_vals.imag[0]
            ax.text(xr, xi, label_fmt.format(theta), fontsize=8, color=c, va="bottom", ha="left")

    # 5) Overlay the θ-trajectory at anchor_alpha
    if snap_to_nearest_alpha:
        idx = int(np.argmin(np.abs(alphas - anchor_alpha)))
        anchor_alpha = float(alphas[idx])

    thetas_anchor = np.linspace(theta_start, theta_end, anchor_grid_size)
    etas_anchor = anchor_alpha * np.exp(1j * thetas_anchor)
    cont_anchor = np.asarray([continued_fn(alphas, energies, coeffs, eta) for eta in etas_anchor])

    ax.plot(cont_anchor.real, cont_anchor.imag, lw=2.5, color="black",
            label=fr"θ-trajectory at α={anchor_alpha:g}")

    # colorbar keyed to θ
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=float(theta_start), vmax=float(theta_end)))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label(r"$\theta$ (rad)")

    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(fr"α-trajectories for $\theta \in [{theta_start:.3g},{theta_end:.3g}]$ "
                 fr"+ overlay at α={anchor_alpha:g}")
    ax.grid(True)
    ax.legend(loc="best")
    # ax.set_aspect("equal", adjustable="datalim")  # uncomment if you prefer equal scaling
    plt.show()


def plot_single_alpha_and_theta(
    points,
    alpha_value,                 # the α for the θ-trajectory
    theta_value,                 # the θ for the α-trajectory
    alpha_traj_start,
    alpha_traj_end,
    theta_traj_start=0.0,
    theta_traj_end=pi/2,
    theta_grid_size=361,
    alpha_grid_size=200,
    snap_to_nearest_alpha=True
):
    """
    On one figure, plot:
      • the α-trajectory at a single fixed θ = theta_value
      • the θ-trajectory at a single fixed α = alpha_value
    """
    # 1) Load & sort
    alphas, energies = read_csv(points)
    alphas = np.asarray(alphas, float)
    energies = np.asarray(energies, complex)
    order = np.argsort(alphas)
    alphas = alphas[order]
    energies = energies[order]

    # 2) Build coefficients
    coeffs = schlessinger(alphas, energies)

    # 3) Optionally snap α to the nearest sample
    if snap_to_nearest_alpha:
        idx = int(np.argmin(np.abs(alphas - alpha_value)))
        alpha_value = float(alphas[idx])

    # 4) α-trajectory at fixed θ
    a_min, a_max = float(alphas[0]), float(alphas[-1])
    alpha_line = np.linspace(alpha_traj_start, alpha_traj_end, alpha_grid_size)
    etas_alpha_traj = alpha_line * np.exp(1j * theta_value)
    cont_alpha_traj = np.asarray([continued_fn(alphas, energies, coeffs, eta) for eta in etas_alpha_traj])

    # 5) θ-trajectory at fixed α
    thetas = np.linspace(theta_traj_start, theta_traj_end, theta_grid_size)
    etas_theta_traj = alpha_value * np.exp(1j * thetas)
    cont_theta_traj = np.asarray([continued_fn(alphas, energies, coeffs, eta) for eta in etas_theta_traj])

    # 6) Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cont_alpha_traj.real, cont_alpha_traj.imag, lw=2,
            label=fr"α-trajectory at θ={theta_value:.3g} rad")
    ax.plot(cont_theta_traj.real, cont_theta_traj.imag, lw=2,
            label=fr"θ-trajectory at α={alpha_value:g}")

    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(fr"RVP: α={alpha_value:g}, θ={theta_value:.3g} rad")
    ax.grid(True)
    ax.legend(loc="best")
    # ax.set_aspect("equal", adjustable="datalim")
    plt.show()


def find_cusp_candidate_pair(
    points,
    alpha_min=None,
    alpha_max=None,
    theta_start=0.0,
    theta_end=pi/2,
    n_alpha_candidates=61,
    n_theta_candidates=121,
    alpha_curve_samples=500,
    theta_curve_samples=500,
    tangent_weight=0.25,
    turn_weight=0.50,
    min_turn_angle=0.35,
    turn_span=3,
    max_contact_distance_frac=0.02,
    snap_alpha_candidates_to_data=False,
    show_overlay=True,
    show_single=True
):
    """
    Coarse-search (alpha, theta) pairs whose alpha- and theta-trajectories nearly meet
    and each makes a sharp local turn near the contact point.
    Returns a dict describing the best candidate and optionally plots it.
    """
    def local_turn_angle(xy, idx, span):
        n = int(xy.shape[0])
        i0 = max(0, int(idx) - int(span))
        i1 = min(n - 1, int(idx) + int(span))
        if i1 <= i0:
            return 0.0
        v_prev = xy[int(idx)] - xy[i0]
        v_next = xy[i1] - xy[int(idx)]
        n_prev = float(np.linalg.norm(v_prev))
        n_next = float(np.linalg.norm(v_next))
        if n_prev < 1e-14 or n_next < 1e-14:
            return 0.0
        c = float(np.dot(v_prev, v_next) / (n_prev * n_next))
        c = max(-1.0, min(1.0, c))
        # 0 means straight-through, pi means strongest turn/cusp-like reversal.
        return float(np.arccos(c))

    alphas, energies = read_csv(points)
    alphas = np.asarray(alphas, float)
    energies = np.asarray(energies, complex)
    order = np.argsort(alphas)
    alphas = alphas[order]
    energies = energies[order]
    coeffs = schlessinger(alphas, energies)

    data_alpha_min = float(alphas[0])
    data_alpha_max = float(alphas[-1])
    if alpha_min is None:
        alpha_min = data_alpha_min
    if alpha_max is None:
        alpha_max = data_alpha_max

    alpha_candidates = np.linspace(float(alpha_min), float(alpha_max), int(n_alpha_candidates))
    if snap_alpha_candidates_to_data:
        idx = np.abs(alphas[None, :] - alpha_candidates[:, None]).argmin(axis=1)
        alpha_candidates = alphas[idx]
    theta_candidates = np.linspace(float(theta_start), float(theta_end), int(n_theta_candidates))

    alpha_line = np.linspace(float(alpha_min), float(alpha_max), int(alpha_curve_samples))
    theta_line = np.linspace(float(theta_start), float(theta_end), int(theta_curve_samples))

    # Precompute alpha-trajectories for each fixed theta.
    alpha_curves = []
    for theta in theta_candidates:
        etas = alpha_line * np.exp(1j * theta)
        vals = np.asarray([continued_fn(alphas, energies, coeffs, eta) for eta in etas], complex)
        xy = np.column_stack((vals.real, vals.imag))
        tang = np.gradient(xy, axis=0)
        alpha_curves.append((theta, xy, tang))

    # Precompute theta-trajectories for each fixed alpha.
    theta_curves = []
    for alpha in alpha_candidates:
        etas = alpha * np.exp(1j * theta_line)
        vals = np.asarray([continued_fn(alphas, energies, coeffs, eta) for eta in etas], complex)
        xy = np.column_stack((vals.real, vals.imag))
        tang = np.gradient(xy, axis=0)
        tree = cKDTree(xy)
        theta_curves.append((alpha, xy, tang, tree))

    all_xy = np.vstack([c[1] for c in alpha_curves] + [c[1] for c in theta_curves])
    extent = max(
        np.ptp(all_xy[:, 0]),
        np.ptp(all_xy[:, 1]),
        1e-12
    )

    best = None
    for theta, alpha_xy, alpha_tang in alpha_curves:
        for alpha, theta_xy, theta_tang, theta_tree in theta_curves:
            dists, idx_theta = theta_tree.query(alpha_xy, k=1)
            i_alpha = int(np.argmin(dists))
            j_theta = int(idx_theta[i_alpha])
            min_dist = float(dists[i_alpha])
            if (max_contact_distance_frac is not None) and (min_dist > float(max_contact_distance_frac) * extent):
                continue

            v1 = alpha_tang[i_alpha]
            v2 = theta_tang[j_theta]
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-14 or n2 < 1e-14:
                angle = float(pi / 2)
            else:
                cosang = float(np.dot(v1, v2) / (n1 * n2))
                cosang = max(-1.0, min(1.0, abs(cosang)))
                angle = float(np.arccos(cosang))

            alpha_turn = local_turn_angle(alpha_xy, i_alpha, turn_span)
            theta_turn = local_turn_angle(theta_xy, j_theta, turn_span)
            # Require both curves to show at least a minimal sharp turn.
            if alpha_turn < float(min_turn_angle) or theta_turn < float(min_turn_angle):
                continue
            # Lower is better: small contact distance, small tangent mismatch, large turning on both curves.
            turn_reward = 0.5 * ((alpha_turn / pi) + (theta_turn / pi))
            score = (
                (min_dist / extent)
                + float(tangent_weight) * (angle / pi)
                - float(turn_weight) * turn_reward
            )
            if (best is None) or (score < best["score"]):
                contact = 0.5 * (alpha_xy[i_alpha] + theta_xy[j_theta])
                best = {
                    "alpha": float(alpha),
                    "theta": float(theta),
                    "score": float(score),
                    "distance": float(min_dist),
                    "angle_rad": float(angle),
                    "alpha_turn_rad": float(alpha_turn),
                    "theta_turn_rad": float(theta_turn),
                    "contact_re": float(contact[0]),
                    "contact_im": float(contact[1]),
                }

    if best is None:
        raise RuntimeError(
            "No cusp candidate found with current sharp-turn constraints. "
            "Try lowering min_turn_angle, increasing max_contact_distance_frac, "
            "or increasing candidate/sample counts."
        )

    if show_overlay:
        plot_overlay_theta_on_alpha_trajectories_with_labels(
            points=points,
            anchor_alpha=best["alpha"],
            theta_start=theta_start,
            theta_end=theta_end,
            n_thetas=n_theta_candidates,
            alpha_grid_size=max(200, alpha_curve_samples // 2),
            anchor_grid_size=max(361, theta_curve_samples // 2),
            snap_to_nearest_alpha=False,
            label_count=min(10, n_theta_candidates)
        )

    if show_single:
        plot_single_alpha_and_theta(
            points=points,
            alpha_value=best["alpha"],
            theta_value=best["theta"],
            alpha_traj_start=float(alpha_min),
            alpha_traj_end=float(alpha_max),
            theta_traj_start=float(theta_start),
            theta_traj_end=float(theta_end),
            theta_grid_size=max(361, theta_curve_samples),
            alpha_grid_size=max(200, alpha_curve_samples),
            snap_to_nearest_alpha=False
        )

    print(
        "Best cusp candidate: "
        f"alpha={best['alpha']:.8g}, theta={best['theta']:.8g} rad, "
        f"distance={best['distance']:.4e}, tangent_mismatch={best['angle_rad']:.4e} rad, "
        f"alpha_turn={best['alpha_turn_rad']:.4e} rad, theta_turn={best['theta_turn_rad']:.4e} rad, "
        f"score={best['score']:.4e}"
    )
    return best


# plot_overlay_theta_on_alpha_trajectories_with_labels(
#     "22.csv",
#     anchor_alpha=0.435,
#     theta_start=0,
#     theta_end=pi/8,
#     n_thetas=100,     # fewer curves -> labels are easier to read
#     alpha_grid_size=200,
#     anchor_grid_size=361,
#     label_count=8
# )

# for theta in np.linspace(-pi, pi, 90):
#     plot_single_alpha_and_theta(
#         "diiodomethane_9.csv",
#         alpha_value=0.0972,
#         theta_value=theta,
#         alpha_traj_start=0.01,
#         alpha_traj_end=20,
#         theta_traj_start=pi/20,
#         theta_traj_end=pi/6,
#         theta_grid_size=3601,
#         alpha_grid_size=2000,
#         snap_to_nearest_alpha=False
#     )

# for alpha in np.linspace(0.01, 20, 50):
#     plot_single_alpha_and_theta(
#         "diiodomethane_9.csv",
#         alpha_value=alpha,
#         theta_value=1.09,
#         alpha_traj_start=0.01,
#         alpha_traj_end=20,
#         theta_traj_start=0,
#         theta_traj_end=pi,
#         theta_grid_size=3601,
#         alpha_grid_size=2000,
#         snap_to_nearest_alpha=False
#     )

# plot_single_alpha_and_theta(
#     "22.csv",
#     alpha_value=0.43525,
#     theta_value=0.06625,
#     alpha_traj_start=0.3,
#     alpha_traj_end=0.5,
#     theta_traj_start=pi/256,
#     theta_traj_end=pi/8,
#     theta_grid_size=3601,
#     alpha_grid_size=2000,
#     snap_to_nearest_alpha=False
# )


# plot_single_alpha_and_theta(
#     "diiodomethane_9.csv",
#     alpha_value=0.418,
#     theta_value=1.09,
#     alpha_traj_start=0.25,
#     alpha_traj_end=0.8,
#     theta_traj_start=pi/4.5,
#     theta_traj_end=pi/2,
#     theta_grid_size=3601,
#     alpha_grid_size=2000,
#     snap_to_nearest_alpha=False
# )

# for theta in np.linspace(0.2, 0.3, 90):
#     plot_single_alpha_and_theta(
#         "di_6.csv",
#         alpha_value=0.0972,
#         theta_value=theta,
#         alpha_traj_start=0.01,
#         alpha_traj_end=20,
#         theta_traj_start=pi/20,
#         theta_traj_end=pi/6,
#         theta_grid_size=3601,
#         alpha_grid_size=2000,
#         snap_to_nearest_alpha=False
# )

# plot_single_alpha_and_theta(
#     "A2_trial1.csv",
#     alpha_value=0.5,
#     theta_value=0.22455,
#     alpha_traj_start=0.5,
#     alpha_traj_end=0.515,
#     theta_traj_start=pi/15,
#     theta_traj_end=pi/13,
#     theta_grid_size=3601,
#     alpha_grid_size=2000,
#     snap_to_nearest_alpha=False
# )
