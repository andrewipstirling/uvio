import navlie as nav
import matplotlib.pyplot as plt
import numpy as np

from py_vins.utils import (
    load_tum_covar_trajectory,
    load_tum_trajectory,
    copy_covariance,
)
from navlie.utils.alignment import associate_and_align_trajectories

# Typing stuff
from typing import List, Tuple
from matplotlib.figure import Figure

def plot_estimators(
    est_dict,
    path_gt,
    use_comp=True,
    direction="left",
    max_diff=0.035,
) -> Tuple[Figure, List[plt.Axes]]:
    """
    est_dict: dict of {label: trajectory_path}
    path_gt: ground-truth trajectory path
    """

    # Load GT once
    gt_traj = load_tum_trajectory(path_gt, composite=use_comp, direction=direction)
    fig: Figure = None
    axs: List[plt.Axes] = None
    colors = plt.cm.tab10.colors  # automatic color cycle

    for idx, (label, path_est) in enumerate(est_dict.items()):
        print(f"Performing trajectory alignment for {label}: ...")

        # Load estimate + covariance
        est_traj = load_tum_trajectory(
            path_est, composite=use_comp, direction=direction
        )
        est_traj_covar = load_tum_covar_trajectory(
            path_est, composite=use_comp, direction=direction
        )

        # Align trajectories
        gt_aligned, est_aligned, _ = associate_and_align_trajectories(
            traj_ref_list=gt_traj,
            traj_est_list=est_traj,
            verbose=True,
            max_diff=max_diff,
        )

        # Copy covariance to aligned trajectory
        est_traj_covar = copy_covariance(est_traj_covar, est_aligned)

        # Create result object
        results = nav.GaussianResultList.from_estimates(
            est_traj_covar, gt_aligned
        )

        # Normalize time
        results.stamp = results.stamp - results.stamp[0]

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
    run = "1b"
    save_figs = True
    save_name = "uvio_2tag_slam_vs_localisation"

    path_gt = f"/root/datasets/{dataset}/{run}/results/ifo001_ground_truth.txt"

    estimators = {
        # "ov": f"/root/datasets/{dataset}/{run}/results/ov_twotag_localisation.txt",
        "uvio localisation": f"/root/datasets/{dataset}/{run}/results_uvio/uvio_twotag_localisation.txt",
        "uvio slam": f"/root/datasets/{dataset}/{run}/results_uvio/uvio_twotag_slam_gt_init.txt",
        # Add more here:
        # "my_new_method": "/path/to/file.txt",
    }
    fig, axs = plot_estimators(
        est_dict=estimators,
        path_gt=path_gt,
        use_comp=True,
        direction="left",
    )

    fig.tight_layout()
    if save_figs:
        if save_name is None:
            raise ValueError("To save figure, \'save_name\' argument must be provided. ")
        
        fig.savefig(f"/root/datasets/{dataset}/{run}/results_uvio/figs/{save_name}.pdf")
    plt.show()


if __name__ == "__main__":
    main()

