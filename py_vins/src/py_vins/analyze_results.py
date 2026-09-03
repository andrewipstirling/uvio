from typing import List, Tuple
import matplotlib.pyplot as plt
import navlie as nav
import numpy as np
from matplotlib.figure import Figure
from py_vins.utils import (
    OVState,
    copy_covariance,
    load_tum_covar_trajectory,
    load_tum_trajectory,
)
from navlie.utils.alignment import associate_and_align_trajectories
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def compute_ate_and_nees(
    results: nav.GaussianResultList,
) -> Tuple[float, float, float, float, nav.GaussianResultList, nav.GaussianResultList]:
    """Computes rot/pos ATE and rot/pos average NEES for a single trial."""
    # 1. ATE Computation
    rot_err = np.rad2deg(results.error[:, 0:3])
    pos_err = results.error[:, 3:]

    rot_ate = float(np.sqrt(np.mean(np.square(np.linalg.norm(rot_err, axis=1)))))
    pos_ate = float(np.sqrt(np.mean(np.square(np.linalg.norm(pos_err, axis=1)))))

    # 2. Decompose state to compute Attitude and Position NEES
    attitude_est_list: List[nav.StateWithCovariance] = []
    attitude_true_list: List[OVState] = []
    position_est_list: List[nav.StateWithCovariance] = []
    position_true_list: List[OVState] = []

    for i, state in enumerate(results.state):
        assert isinstance(state, OVState)
        covariance = results.covariance[i]
        true_state: OVState = results.state_true[i]
        assert isinstance(true_state, OVState)

        att_state_covar = nav.StateWithCovariance(
            state=state.get_attitude_as_state(),
            covariance=covariance[0:3, 0:3],
        )
        pos_state_covar = nav.StateWithCovariance(
            state=state.get_position_as_state(),
            covariance=covariance[3:, 3:],
        )

        attitude_est_list.append(att_state_covar)
        attitude_true_list.append(true_state.get_attitude_as_state())
        position_est_list.append(pos_state_covar)
        position_true_list.append(true_state.get_position_as_state())

    attitude_results = nav.GaussianResultList.from_estimates(
        attitude_est_list, attitude_true_list
    )
    position_results = nav.GaussianResultList.from_estimates(
        position_est_list, position_true_list
    )

    # Calculate average normalized estimation error squared (aNEES)
    rot_anees = float(np.mean(attitude_results.nees) / attitude_results.dof[0])
    pos_anees = float(np.mean(position_results.nees) / position_results.dof[0])

    return rot_ate, pos_ate, rot_anees, pos_anees, attitude_results, position_results


def plot_estimators(
    est_dict: dict,
    path_gt: str,
    direction: str = "left",
    max_diff: float = 0.035,
    offset: float = 0.0,
) -> Tuple[Figure, List[plt.Axes]]:
    """est_dict: dict of {label: trajectory_path}

    path_gt: ground-truth trajectory path
    """
    # Load GT once
    gt_traj = load_tum_trajectory(path_gt, direction=direction)

    traj_fig, traj_ax = nav.plot_poses(gt_traj, step=None, kwargs_line={"color": "tab:blue"})
    traj_fig.tight_layout()
    traj_ax.set_xlabel(r"$x \; [\mathrm{m}]$", labelpad=8)
    traj_ax.set_ylabel(r"$y \; [\mathrm{m}]$", labelpad=8)
    traj_ax.set_zlabel(r"$z \; [\mathrm{m}]$", labelpad=8)
    traj_ax.set_zbound(0, 2.0)
    traj_ax.zaxis.set_major_locator(MultipleLocator(0.5))
    traj_ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    fig: Figure = None
    axs: List[plt.Axes] = None
    colors = plt.cm.tab10.colors  # automatic color cycle

    latex_table_str = {}

    for idx, (label, path_est) in enumerate(est_dict.items()):
        print(f"\nPerforming trajectory alignment for {label}: ...")

        # Load estimate + covariance
        est_traj = load_tum_trajectory(path_est, direction=direction)
        est_traj_covar = load_tum_covar_trajectory(
            path_est, direction=direction
        )

        # Align trajectories
        gt_aligned, est_aligned, _ = associate_and_align_trajectories(
            traj_ref_list=gt_traj,
            traj_est_list=est_traj,
            verbose=True,
            max_diff=max_diff,
            offset=offset,
        )

        gt_aligned_ov = []
        for x in gt_aligned:
            ov_state = OVState(
                state_list=[
                    nav.lib.SO3State(x.attitude, stamp=x.stamp),
                    nav.lib.VectorState(x.position, stamp=x.stamp),
                ],
                stamp=x.stamp,
            )
            gt_aligned_ov.append(ov_state)

        # Copy covariance to aligned trajectory
        est_traj_covar = copy_covariance(est_traj_covar, est_aligned)

        # Create result object
        results = nav.GaussianResultList.from_estimates(
            est_traj_covar, gt_aligned_ov
        )

        # Compute single-trial metrics
        rot_ate, pos_ate, rot_anees, pos_anees, att_res, pos_res = (
            compute_ate_and_nees(results)
        )

        print("+-------------------+-----------------+------------+-----------+")
        print("|     rot ATE (deg) |     pos ATE (m) |  rot aNEES | pos aNEES |")
        print("+-------------------+-----------------+------------+-----------+")
        print(
            f"& {rot_ate:^17.4f} & "
            f"{pos_ate:^15.4f} & "
            f"{rot_anees:^10.4f} & "
            f"{pos_anees:^9.4f} \\\\"
        )
        print("+-------------------+-----------------+------------+-----------+")

        latex_table_str[label] = (
            f"& {rot_ate:.4f} & {pos_ate:.4f} & {rot_anees:.4f} & {pos_anees:.4f} \\\\"
        )

        # Plot individual NEES for attitude & position per estimator
        nees_fig, nees_axs = plt.subplots(2, 1, figsize=(8, 5))
        nav.plot_nees(att_res, ax=nees_axs[0])
        nav.plot_nees(pos_res, ax=nees_axs[1])
        nees_axs[0].set_title(f"{label} - Attitude NEES")
        nees_axs[0].set_ylim(0, 10)
        nees_axs[1].set_title(f"{label} - Position NEES")
        nees_fig.tight_layout()

        # Plot errors across state dimensions
        fig, axs = nav.plot_error(
            results,
            label=label,
            color=colors[idx % len(colors)],
            axs=axs,
        )
        fig: Figure = fig
        axs: List[plt.Axes] = axs

        print(" ----------------------------------------------- ")

    print("\n=== Consolidated LaTeX Summary ===")
    for label, table_str in latex_table_str.items():
        print(f"{label:<30} {table_str}")

    # Finalize plot
    ax = axs.ravel()[0]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left")

    axis_labels = [
        r"$\delta \xi_{\phi}$",
        r"$\delta \xi_{x}$",
        r"$\delta \xi_{\theta}$",
        r"$\delta \xi_{y}$",
        r"$\delta \xi_{\psi}$",
        r"$\delta \xi_{z}$",
    ]

    for ax, label_str in zip(axs.ravel(), axis_labels):
        ax.set_ylabel(label_str)

    return fig, axs


def main():
    dataset = "miluv"
    run = "default_1_circular3D_0"  # "1b"
    save_figs = False
    save_name = "uvio_2tag_slam_vs_localisation_bias"

    path_gt = f"/workspace/datasets/{dataset}/{run}/results_uvio/ifo001_ground_truth.txt"

    # Define all parameters
    plt.style.use('seaborn-v0_8-whitegrid')
    params = {
        'text.usetex': False,           # Use matplotlib's internal mathtext (no LaTeX install needed)
        'mathtext.fontset': 'cm',       # Computer Modern - mimics LaTeX's default math font
        'font.family': 'serif',         # Set font family
        'font.size': 10,                # Set font size
        'lines.linewidth': 2,           # Set default line width
        'axes.grid': True,              # Enable grid by default
        'axes.labelsize': 12,           # Controls x/y/z label size
        'grid.linestyle': '--'          # Set grid line style
    }

    # Update rcParams with all settings at once
    plt.rcParams.update(params)

    estimators = {
        "Standard VIO": f"/workspace/datasets/{dataset}/{run}/results/standard_vio.txt",
        # "Frame-Aligned UVIO": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_frame_aligned_local_dstwr_marg_ransac.txt",
        # "Frame-Aligned UVIO Global": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_frame_aligned_local_dstwr_marg_ransac_global.txt",
        # "Schmidt AlignUVIO SLAM": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_frame_aligned_schmidt_slam_2.txt",
        # "Schmidt Marg AlignUVIO SLAM Local": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_frame_aligned_local_dstwr_margslam_schmidt.txt",
        # "Schmidt Marg AlignUVIO SLAM Local": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_frame_aligned_schmidt_slam.txt",
        # "Schmidt Marg AlignUVIO SLAM Global": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_frame_aligned_local_dstwr_margslam_schmidt_global.txt",
        # "UVIO": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_naive_slam_covar_inflate.txt",
        "AlignUVIO": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_frame_aligned_local_dstwr_schmidt_5_global.txt",
    }
    fig, axs = plot_estimators(
        est_dict=estimators, path_gt=path_gt, direction="right", offset=0.0
    )
    for ax in axs.ravel():
        lines = ax.get_lines()
        if not lines:
            continue

        t_min = min(line.get_xdata()[0] for line in lines)
        y_min, y_max = float("inf"), float("-inf")

        # Shift lines and collect y-bounds
        for line in lines:
            line.set_xdata(line.get_xdata() - t_min)
            y_data = line.get_ydata()
            if len(y_data) > 0:
                y_min = min(y_min, np.nanmin(y_data))
                y_max = max(y_max, np.nanmax(y_data))

        # Shift shaded uncertainty and collect y-bounds
        for poly in ax.collections:
            paths = poly.get_paths()
            for path in paths:
                path.vertices[:, 0] -= t_min
                # Include polygon vertices in y-bounds calculation
                y_vertices = path.vertices[:, 1]
                if len(y_vertices) > 0:
                    y_min = min(y_min, np.nanmin(y_vertices))
                    y_max = max(y_max, np.nanmax(y_vertices))

        # Add a small margin (e.g., 5%) so the line doesn't touch the edge
        margin = (y_max - y_min) * 0.05
        if margin == 0 or not np.isfinite(margin):
            margin = 0.1  # avoid singular limits

        # Update the view
        ax.relim()
        ax.autoscale_view()
        ax.set_xlabel("Time (s)")

        # Set y-limits based on computed bounds
        if np.isfinite(y_min) and np.isfinite(y_max):
            ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()
    if save_figs:
        if save_name is None:
            raise ValueError(
                "To save figure, 'save_name' argument must be provided. "
            )

        fig.savefig(
            f"/root/datasets/{dataset}/{run}/results_uvio/figs/{save_name}.pdf"
        )
    plt.show()


if __name__ == "__main__":
    main()
