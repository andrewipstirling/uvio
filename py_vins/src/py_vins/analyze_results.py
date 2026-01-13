import navlie as nav
import matplotlib.pyplot as plt
import numpy as np

from py_vins.utils import (
    load_tum_covar_trajectory,
    load_tum_trajectory,
    copy_covariance,
)
from navlie.utils.alignment import associate_and_align_trajectories


def plot_estimators(
    est_dict,
    path_gt,
    use_comp=True,
    direction="left",
    max_diff=0.035,
):
    """
    est_dict: dict of {label: trajectory_path}
    path_gt: ground-truth trajectory path
    """

    # Load GT once
    gt_traj = load_tum_trajectory(path_gt, composite=use_comp, direction=direction)

    fig, axs = None, None
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
        fig, axs = nav.plot_error(
            results,
            label=label,
            color=colors[idx % len(colors)],
            axs=axs,
        )

        print(" ----------------------------------------------- ")

    # Finalize plot
    for ax in axs.ravel():
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.show()


def main():
    dataset = "miluv"
    run = "1b"

    path_gt = f"/root/datasets/{dataset}/{run}/results/ifo001_ground_truth.txt"

    estimators = {
        "ov": f"/root/datasets/{dataset}/{run}/results/ov_twotag_localisation.txt",
        "uvio-2tag": f"/root/datasets/{dataset}/{run}/results_uvio/uvio_twotag_localisation.txt",
        "uvio-1tag": f"/root/datasets/{dataset}/{run}/results_uvio/uvio_onetag_localisation.txt",
        # Add more here:
        # "my_new_method": "/path/to/file.txt",
    }

    plot_estimators(
        est_dict=estimators,
        path_gt=path_gt,
        use_comp=True,
        direction="left",
    )


if __name__ == "__main__":
    main()

