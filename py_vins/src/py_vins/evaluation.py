import navlie as nav
import matplotlib.pyplot as plt
import numpy as np

from navlie.lib.states import SE3State

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

def load_aligned_results_for_trial(
    path_est,
    path_gt,
    direction="left",
    max_diff=0.035,
):
    gt_traj = load_tum_trajectory(path_gt, direction=direction)

    est_traj = load_tum_trajectory(path_est, direction=direction)
    est_traj_covar = load_tum_covar_trajectory(
        path_est, direction=direction
    )

    gt_aligned, est_aligned, _ = associate_and_align_trajectories(
        traj_ref_list=gt_traj,
        traj_est_list=est_traj,
        verbose=False,
        max_diff=max_diff,
    )
    gt_aligned_ov = []
    for x in gt_aligned:
        ov_state = OVState(state_list=[nav.lib.SO3State(x.attitude, stamp=x.stamp), nav.lib.VectorState(x.position, stamp=x.stamp)], stamp=x.stamp)
        gt_aligned_ov.append(ov_state)
    
    est_traj_covar = copy_covariance(est_traj_covar, est_aligned)


    results = nav.GaussianResultList.from_estimates(
        est_traj_covar, gt_aligned_ov
    )

    results.stamp = results.stamp - results.stamp[0]
    return results

def _ate(
        results: nav.GaussianResultList
)-> Tuple[np.ndarray, np.ndarray]:
    rot_err = np.rad2deg(results.error[:, 0:3])
    pos_err = results.error[:, 3:]

    rot_ate = np.sqrt(np.mean(np.square(np.linalg.norm(rot_err, axis=1))))
    pos_ate = np.sqrt(np.mean(np.square(np.linalg.norm(pos_err, axis=1))))

    return rot_ate, pos_ate

def get_results_list(trial_paths, path_gt, **kwargs):
    results_list = [
        load_aligned_results_for_trial(p, path_gt, **kwargs)
        for p in trial_paths
    ]
    return results_list
    

def compute_ate_trials(
    results_list: List[nav.GaussianResultList]
):
    ate_rot_list = []
    ate_pos_list = []

    for r in results_list:
        ate_rot, ate_pos = _ate(r)
        ate_rot_list.append(ate_rot)
        ate_pos_list.append(ate_pos)
        
    ate_total = np.vstack([np.array(ate_rot_list), np.array(ate_pos_list)])

    return ate_total

def crop_trial(trial, t_start, t_end, ref_stamps=None):
    """
    Crop a trial to a common time interval and (optionally) reindex it
    onto a reference timestamp grid.

    This function is designed for Monte Carlo evaluation where all trials
    must have:
      - overlapping time support, AND
      - identical sample counts and timestamps

    Parameters
    ----------
    trial : navlie.GaussianResultList or navlie.TrialResult
        A single trial containing stamps, NEES, errors, etc.
    t_start : float
        Start time (seconds). Samples before this time are discarded.
    t_end : float
        End time (seconds). Samples after this time are discarded.
    ref_stamps : np.ndarray, optional
        Reference timestamps to reindex onto. If provided, the trial will
        be reindexed so that its samples correspond to these timestamps.
        This ensures identical array lengths across trials.

    Returns
    -------
    cropped_trial : same type as `trial`
        Cropped (and optionally reindexed) trial.
    """

    # Time-based cropping (ensure overlapping time interval)
    mask = (trial.stamp >= t_start) & (trial.stamp <= t_end)

    # Convert boolean mask → explicit integer indices
    # navlie trials only support integer indexing, not boolean masks
    idx = np.where(mask)[0]

    cropped_trial = trial[idx.tolist()]

    # Reindex to reference timestamps 
    # (required for MonteCarloResult initialization)
    if ref_stamps is not None:
        # For each reference timestamp, find the closest index in this trial
        # This guarantees identical sample counts across trials
        reidx = np.searchsorted(cropped_trial.stamp, ref_stamps)

        # Clamp indices to valid bounds
        reidx = np.clip(reidx, 0, len(cropped_trial.stamp) - 1)

        cropped_trial = cropped_trial[reidx.tolist()]

    return cropped_trial

def get_attitude_position_results(
        results_list: List[nav.GaussianResultList]
    ) -> Tuple[nav.MonteCarloResult, nav.MonteCarloResult]:
    """
    Returns a MonteCarloResult for the CompositeState SO(3) x R(3)
    """
    attitude_mc_list = []
    position_mc_list = []
    
    for result in results_list:
        # Outer Loop: Individual Trials
        attitude_est_list: List[nav.StateWithCovariance] = []
        attitude_true_list: List[OVState] = []
        position_est_list: List[nav.StateWithCovariance]= []
        position_true_list: List[OVState] = []
        for i, state in enumerate(result.state):
            assert isinstance(state, OVState)
            covariance = result.covariance[i]
            true_state: OVState = result.state_true[i]
            assert isinstance(true_state, OVState)
            att_state_covar = nav.StateWithCovariance(state=state.get_attitude_as_state(), covariance=covariance[0:3, 0:3])
            pos_state_covar = nav.StateWithCovariance(state=state.get_position_as_state(), covariance=covariance[3:, 3:])
            attitude_est_list.append(att_state_covar)
            attitude_true_list.append(true_state.get_attitude_as_state())
            position_est_list.append(pos_state_covar)
            position_true_list.append(true_state.get_position_as_state())
            
        attitude_gauss_list = nav.GaussianResultList.from_estimates(attitude_est_list, attitude_true_list)
        position_gauss_list = nav.GaussianResultList.from_estimates(position_est_list, position_true_list)
        attitude_mc_list.append(attitude_gauss_list)
        position_mc_list.append(position_gauss_list)

    t_start = max(t.stamp[0] for t in attitude_mc_list)
    t_end   = min(t.stamp[-1] for t in attitude_mc_list)
    ref_trial = min(attitude_mc_list, key=lambda t: len(t.stamp))
    ref_stamps = ref_trial.stamp

    attitude_mc_list = [crop_trial(t, t_start, t_end, ref_stamps) for t in attitude_mc_list]
    position_mc_list = [crop_trial(t, t_start, t_end, ref_stamps) for t in position_mc_list]
    
    attitude_mc_result = nav.MonteCarloResult(attitude_mc_list)
    attitude_mc_result.stamp = attitude_mc_result.stamp - attitude_mc_result.stamp[0]
    position_mc_result = nav.MonteCarloResult(position_mc_list)
    position_mc_result.stamp = position_mc_result.stamp - position_mc_result.stamp[0]
    return attitude_mc_result, position_mc_result



def plot_three_sigma(
    results_list: List[nav.GaussianResultList],
    label: str,
    **kwargs
):
    fig, axs = None, None

    for i, results in enumerate(results_list):
        fig, axs = nav.plot_error(results, label=f"{label}_{i}", axs=axs)
    
    fig: Figure = fig
    axs: List[plt.Axes] = axs

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

    for ax, label in zip(axs.ravel(), axis_labels):
        ax.set_ylabel(label)

    return fig, axs



def main(dataset, run, show_figs = False, save_figs = False):
    

    path_gt = f"/root/datasets/{dataset}/{run}/results/ifo001_ground_truth.txt"
    base = f"/root/datasets/{dataset}/{run}/results_uvio"
    # Localisation Trials
    trial_names_labels = \
        {
        #  "Localization calibrated DSTWR with bias estimation": "uvio_twotag_localisation_calib_bias_dstwr_std",
        #  "Localization calibrated DSTWR": "uvio_twotag_localisation_dstwr_std",
        #  "Localization standard TWR with bias estimation": "uvio_twotag_localisation_calib_bias_twr",
        #  "Frame aligned localization calibrated DSTWR": "uvio_twotag_localisation_dstwr_std_frame_align",
        "Localization TWR": "uvio_twotag_localisation_twr",
        # "Frame aligned localization TWR": "uvio_twotag_localisation_twr_frame_aligned",
        "Frame aligned PDOP / SQ initialized localization TWR": "uvio_twotag_localisation_twr_frame_aligned_wls_pdop_fim_sq_jac",
         }
    
    # SLAM Trials
    # trial_names_labels = \
    #     {"SLAM calibrated DSTWR with bias estimation": "uvio_twotag_slam_calib_bias_dstwr_std",
    #      "SLAM calibrated DSTWR": "uvio_twotag_slam_dstwr_std",
    #      "SLAM standard TWR with bias estimation": "uvio_twotag_slam_calib_bias_twr"}
    
    estimators = {}
    
    for label, trial_name in trial_names_labels.items():
        estimators[label] = [
            f"{base}/{trial_name}/run_{i:02d}.txt"
            for i in range(5)
        ]
    for label, trials in estimators.items():
        results_list = get_results_list(trial_paths=trials, path_gt=path_gt)
        ate_i = compute_ate_trials(results_list)
        print(f"{label} \n| ATE (rot): {ate_i[0]} \n| ATE (pos): {ate_i[1]}")

        attitude_results, position_results = get_attitude_position_results(results_list)

        print("+-------------------+-----------------+------------+-----------+")
        print("| avg rot ATE (deg) | avg pos ATE (m) |  rot aNEES | pos aNEES |")
        print("+-------------------+-----------------+------------+-----------+")
        print(
            f"| {np.mean(ate_i[0]):^17.4f} | "
            f"{np.mean(ate_i[1]):^15.4f} | "
            f"{(np.mean(attitude_results.nees) / attitude_results.dof[0]):^10.4f} | "
            f"{(np.mean(position_results.nees) / position_results.dof[0]):^9.4f} |"
        )
        print("+-------------------+-----------------+------------+-----------+")

        nees_fig, nees_axs =  plt.subplots(2,1)
        nees_fig, ax = nav.plot_nees(attitude_results, ax=nees_axs[0])
        nees_fig, ax = nav.plot_nees(position_results, ax=nees_axs[1])
        nees_axs[0].set_title("Attitude NEES")
        nees_axs[0].set_ylim(0, 10)
        nees_axs[1].set_title("Position NEES")
        nees_fig.tight_layout()


        fig, axs = plot_three_sigma(results_list, label=f"trial")
        fig.suptitle(t=label)
        fig.tight_layout()

        if save_figs:
            nees_fig.savefig(base + f"/figs/nees_{label}.pdf")
            fig.savefig(base + f"/figs/3sigma_{label}.pdf")
    
    if show_figs:
        plt.show()
    

if __name__=="__main__":
    dataset = "miluv"
    run = "1b"
    show_figs = True
    save_figs = False
    main(dataset, run, show_figs, save_figs)