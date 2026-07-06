import navlie as nav
import matplotlib.pyplot as plt
import numpy as np

from py_vins.utils import (
    load_tum_covar_trajectory,
    load_tum_trajectory,
    copy_covariance,
    OVState
)
from navlie.utils.alignment import associate_and_align_trajectories

# Typing stuff
from typing import List, Tuple
from matplotlib.figure import Figure

def plot_estimators(
    est_dict,
    path_gt,
    direction="left",
    max_diff=0.035,
    offset = 0.0
) -> Tuple[Figure, List[plt.Axes]]:
    """
    est_dict: dict of {label: trajectory_path}
    path_gt: ground-truth trajectory path
    """

    # Load GT once
    gt_traj = load_tum_trajectory(path_gt, direction=direction)
    fig: Figure = None
    axs: List[plt.Axes] = None
    colors = plt.cm.tab10.colors  # automatic color cycle

    for idx, (label, path_est) in enumerate(est_dict.items()):
        print(f"Performing trajectory alignment for {label}: ...")

        # Load estimate + covariance
        est_traj = load_tum_trajectory(
            path_est, direction=direction
        )
        est_traj_covar = load_tum_covar_trajectory(
            path_est, direction=direction
        )

        # Align trajectories
        gt_aligned, est_aligned, _ = associate_and_align_trajectories(
            traj_ref_list=gt_traj,
            traj_est_list=est_traj,
            verbose=True,
            max_diff=max_diff,
            offset=offset
        )

        gt_aligned_ov = []
        for x in gt_aligned:
            ov_state = OVState(state_list=[nav.lib.SO3State(x.attitude, stamp=x.stamp), nav.lib.VectorState(x.position, stamp=x.stamp)], stamp=x.stamp)
            gt_aligned_ov.append(ov_state)

        # Copy covariance to aligned trajectory
        est_traj_covar = copy_covariance(est_traj_covar, est_aligned)

        # Create result object
        results = nav.GaussianResultList.from_estimates(
            est_traj_covar, gt_aligned_ov
        )

        # Plot
        fig, axs= nav.plot_error(
            results,
            label=label,
            color=colors[idx % len(colors)],
            axs=axs,
        )
        fig: Figure = fig
        axs: List[plt.Axes] = axs

        print(" ----------------------------------------------- ")

    # Finalize plot
    ax = axs.ravel()[0]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left")
    # for ax in axs.ravel():
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels, loc="lower right")
    

    axis_labels = [
        r"$\delta \xi_{\phi}$",
        r"$\delta \xi_{x}$",
        r"$\delta \xi_{\theta}$",
        r"$\delta \xi_{y}$",
        r"$\delta \xi_{\psi}$",
        r"$\delta \xi_{z}$",
    ]

    for ax, label in zip(axs.ravel(), axis_labels):
        ax.set_ylabel(label)

    return fig, axs


def main():
    dataset = "miluv"
    run = "default_1_random3_0" #"1b"
    save_figs = False
    save_name = "uvio_2tag_slam_vs_localisation_bias"

    path_gt = f"/workspace/datasets/{dataset}/{run}/results_uvio/ifo001_ground_truth.txt"

    estimators = {
        # "Standard VIO": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_twotag_localisation_twr.txt",

        "Standard VIO": f"/workspace/datasets/{dataset}/{run}/results/standard_vio.txt",

        # "Frame-Aligned (PDOP) UVIO": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_local_frame_align_rangecost_yawcov_4dofjac_fix_grid_split_urls.txt",

        # "Frame-Aligned (PDOP) Global": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_local_frame_align_rangecost_yawcov_4dofjac_fix_grid_split_urls_global.txt",

        "Frame-Aligned UVIO Considered Map": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_frame_aligned_local_dstwr_schmidt.txt",

        "Frame-Aligned UVIO": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_frame_aligned_local.txt",

        # "Frame-Aligned UVIO Gobal": f"/workspace/datasets/{dataset}/{run}/results_uvio/uvio_frame_aligned_local_schmidt_global.txt",

        # Add more here:
        # "my_new_method": "/path/to/file.txt",
    }
    fig, axs = plot_estimators(
        est_dict=estimators,
        path_gt=path_gt,
        direction="right",
        offset=0.0
    )
    for ax in axs.ravel():
        lines = ax.get_lines()
        if not lines:
            continue
        
        t_min = min(line.get_xdata()[0] for line in lines)
        # print("Minimum time: ", t_min)
        y_min, y_max = float('inf'), float('-inf')
        
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
            raise ValueError("To save figure, \'save_name\' argument must be provided. ")
        
        fig.savefig(f"/root/datasets/{dataset}/{run}/results_uvio/figs/{save_name}.pdf")
    plt.show()


if __name__ == "__main__":
    main()

