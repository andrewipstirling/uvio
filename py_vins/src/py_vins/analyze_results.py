import navlie as nav
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from py_vins.utils import load_tum_covar_trajectory, load_tum_trajectory, copy_covariance
from evo.tools import file_interface
from navlie.utils.alignment import associate_and_align_trajectories

def main():
    dataset = "miluv"
    run = "1b"
    uvio_filename = "traj_uvio_slam_noinit_all.txt"
    
    path_est_ov_sub = f"/root/datasets/{dataset}/{run}/results/traj_estimate.txt"
    # path_est_ov_serial = f"/root/datasets/{dataset}/{run}/results/traj_estimate_serial.txt"
    path_est_uvio = f"/root/datasets/{dataset}/{run}/results_uvio/{uvio_filename}"
    path_gt = f"/root/datasets/{dataset}/{run}/results/ifo001_ground_truth.txt"

    # dataset = "euroc"
    # path_est = "/root/datasets/euroc_mav/results/traj_estimate.txt"
    # path_est_serial = "/root/datasets/euroc_mav/results/traj_estimate_serial.txt"
    # path_gt = config["path_gt"]
    use_comp = True
    direction = "left"

    est_traj_covar_uvio = load_tum_covar_trajectory(path_est_uvio, composite=use_comp, direction = direction)
    est_traj_covar_ov = load_tum_covar_trajectory(path_est_ov_sub, composite=use_comp, direction = direction)

    est_traj_uvio = load_tum_trajectory(path_est_uvio, composite=use_comp, direction=direction)
    gt_traj = load_tum_trajectory(path_gt, composite=use_comp, direction = direction)

    est_traj_ov = load_tum_trajectory(path_est_ov_sub, composite=use_comp, direction=direction)
    gt_traj_serial = load_tum_trajectory(path_gt, composite=use_comp, direction = direction)
    
    print("Performing UVIO trajectory alignment: ...")
    gt_traj_sub, est_traj_uvio, T_info = associate_and_align_trajectories(traj_ref_list=gt_traj, traj_est_list=est_traj_uvio, verbose=True, max_diff=0.035)
    print(" ----------------------------------------------- ")
    print("Performing OpenVINS trajectory alignment: ...")
    gt_traj_serial, est_traj_ov, T_info_serial = associate_and_align_trajectories(traj_ref_list=gt_traj_serial, traj_est_list=est_traj_ov, verbose=True, max_diff = 0.035)
    print(" ----------------------------------------------- ")
    est_traj_covar_uvio = copy_covariance(est_traj_covar_uvio, est_traj_uvio)
    est_traj_covar_ov = copy_covariance(est_traj_covar_ov, est_traj_ov)

    
    results_serial = nav.GaussianResultList.from_estimates(est_traj_covar_ov, gt_traj_serial)
    results_serial.stamp = results_serial.stamp - (np.ones_like(results_serial.stamp) * results_serial.stamp[0])
    
    results_serial_uvio = nav.GaussianResultList.from_estimates(est_traj_covar_uvio, gt_traj_sub)
    results_serial_uvio.stamp = results_serial_uvio.stamp - (np.ones_like(results_serial_uvio.stamp) * results_serial_uvio.stamp[0])
    

    fig, axs = nav.plot_error(results_serial_uvio, label="uvio", color="orange")
    fig, axs = nav.plot_error(results_serial, label="ov", color="blue", axs=axs)
    # Add legends properly
    for ax in axs.ravel():
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.show()

if __name__== "__main__":
    main()
