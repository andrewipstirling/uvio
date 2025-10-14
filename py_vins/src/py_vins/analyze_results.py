import navlie as nav
import matplotlib
import matplotlib.pyplot as plt

from py_vins.utils import load_config, load_tum_covar_trajectory, load_tum_trajectory, copy_covariance
from evo.tools import file_interface
from navlie.utils.alignment import associate_and_align_trajectories

def main():
    dataset = "miluv"
    path_est = "/root/datasets/miluv/1a/results/traj_estimate.txt"
    path_est_serial = "/root/datasets/miluv/1a/results/traj_estimate_serial.txt"

    # dataset = "euroc"
    # path_est = "/root/datasets/euroc_mav/results/traj_estimate.txt"
    # path_est_serial = "/root/datasets/euroc_mav/results/traj_estimate_serial.txt"

    config = load_config(dataset)
    path_gt = config["path_gt"]

    est_traj_covar = load_tum_covar_trajectory(path_est)
    est_traj_serial_covar = load_tum_covar_trajectory(path_est_serial)

    est_traj = load_tum_trajectory(path_est)
    gt_traj = load_tum_trajectory(path_gt)

    est_traj_serial = load_tum_trajectory(path_est_serial)
    gt_traj_serial = load_tum_trajectory(path_gt)
    
    

    gt_traj_sub, est_traj, T_info = associate_and_align_trajectories(traj_ref_list=gt_traj, traj_est_list=est_traj, verbose=False)

    gt_traj_serial, est_traj_serial, T_info_serial = associate_and_align_trajectories(traj_ref_list=gt_traj_serial, traj_est_list=est_traj_serial)

    est_traj_covar = copy_covariance(est_traj_covar, est_traj)
    est_traj_serial_covar = copy_covariance(est_traj_serial_covar, est_traj_serial)

    print("Length of subscribed trajectory: ", len(est_traj_covar))
    print("Length of serial trajectory: ",len(est_traj_serial_covar))
    
    results_serial = nav.GaussianResultList.from_estimates(est_traj_serial_covar, gt_traj_serial)
    results_sub = nav.GaussianResultList.from_estimates(est_traj_covar, gt_traj_sub)

    fig, axs = nav.plot_error(results_sub, label="sub", color="orange")
    fig, axs = nav.plot_error(results_serial, label="serial", color="blue", axs=axs)
    # Add legends properly
    for ax in axs.ravel():
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.show()

if __name__== "__main__":
    main()
