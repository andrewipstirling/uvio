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

    # NOTE: stamps are deliberately left as-is (not zeroed here). Zeroing
    # per-trial at load time discards each trial's real start time relative
    # to other trials/estimators. The zero-reference shift (if any) should
    # be applied once, later, at the point where results are combined for
    # plotting -- see plot_average_three_sigma_combined -- so that a genuine
    # relative offset between estimators is preserved instead of collapsed.
    return results


def _ate(
    results: nav.GaussianResultList
) -> Tuple[np.ndarray, np.ndarray]:
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

    # Convert boolean mask -> explicit integer indices
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
        position_est_list: List[nav.StateWithCovariance] = []
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
    t_end = min(t.stamp[-1] for t in attitude_mc_list)
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


def plot_average_three_sigma_combined(
    estimator_results: dict,
    sigma_multiple: int = 3,
):
    """
    Plot the average position error and average sigma_multiple-sigma bound
    across trials, for multiple estimators overlaid on the same axes.

    Orientation components are excluded -- only position (x, y, z) is shown.

    Parameters
    ----------
    estimator_results : dict[str, List[nav.GaussianResultList]]
        Mapping from estimator label to its list of per-trial results.
    sigma_multiple : int
        Number of standard deviations to shade (default 3).

    Trials are no longer zeroed to t=0 individually at load time (see
    load_aligned_results_for_trial), so each trial retains its real
    bag/GT-referenced timestamp. Within an estimator, trials are padded out
    to the longest trial's length (nanmean) rather than truncated to the
    shortest, so the averaged curve extends to the real end of the data
    instead of stopping early at whichever trial happened to be shortest.
    Across estimators, a single shared zero-reference (the earliest start
    time among all estimators) is subtracted once at the end, so a
    genuinely later-starting estimator still appears shifted relative to
    the others instead of also being pinned to t=0.
    """
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))

    axis_labels = [r"$\delta \xi_{x}$ [m]", r"$\delta \xi_{y}$ [m]", r"$\delta \xi_{z}$ [m]"]
    pos_cols = [3, 4, 5]  # position components only (rotation excluded)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- Pass 1: compute per-estimator averages, keeping real (unzeroed)
    # timestamps so that each estimator's true relative start time survives.
    per_estimator = {}
    for label, results_list in estimator_results.items():
        # Pad every trial out to the LONGEST trial's length (instead of
        # truncating to the shortest) so the average keeps extending as far
        # as data exists -- averaging over fewer trials near the tail rather
        # than cutting everything off at whichever trial happened to be
        # shortest. This matches the un-averaged plots, where each trial is
        # drawn out to its own real length.
        lengths = [len(r.stamp) for r in results_list]
        max_len = max(lengths)
        ref_idx = int(np.argmax(lengths))  # longest trial supplies the x-axis

        errors_padded = np.full((len(results_list), max_len, 6), np.nan)
        sigmas_padded = np.full((len(results_list), max_len, 6), np.nan)
        for j, r in enumerate(results_list):
            n = len(r.stamp)
            errors_padded[j, :n, :] = r.error
            sigmas_padded[j, :n, :] = np.sqrt(
                np.array([np.diag(cov) for cov in r.covariance])
            )

        avg_error = np.nanmean(errors_padded, axis=0)  # (max_len, 6)
        avg_sigma = np.nanmean(sigmas_padded, axis=0)  # (max_len, 6)

        # Real (unzeroed) timestamps of the longest trial -- used as the
        # common x-axis reference for this estimator's averaged curve.
        stamps = results_list[ref_idx].stamp

        per_estimator[label] = (stamps, avg_error, avg_sigma)

    # --- Pass 2: apply a single, shared zero-reference across ALL estimators
    # (not one per estimator) so the true relative offset between estimators
    # is preserved -- this is the line that matters, and it only runs once,
    # after every estimator's real stamps have been gathered.
    t_min = min(stamps[0] for stamps, _, _ in per_estimator.values())

    for i, (label, (stamps, avg_error, avg_sigma)) in enumerate(per_estimator.items()):
        color = colors[i % len(colors)]
        shifted_stamps = stamps - t_min

        for ax, col, axis_label in zip(axs, pos_cols, axis_labels):
            ax.plot(shifted_stamps, avg_error[:, col], color=color, label=label)
            ax.fill_between(
                shifted_stamps,
                -sigma_multiple * avg_sigma[:, col],
                sigma_multiple * avg_sigma[:, col],
                color=color,
                alpha=0.2,
            )
            ax.set_ylabel(axis_label)

    axs[-1].set_xlabel("Time [s]")
    axs[0].legend(loc="upper left")
    # fig.suptitle(f"Average position error / average {sigma_multiple}$\\sigma$ across trials")
    fig.tight_layout()

    return fig, axs


def main(show_figs=False, save_figs=False):

    dataset = "miluv"
    run = "default_1_random3_2"

    path_gt = f"/workspace/datasets/{dataset}/{run}/results/ifo001_ground_truth.txt"
    base = f"/workspace/datasets/{dataset}/{run}/results_uvio"
    # Localisation Trials
    trial_names_labels = \
        {
        #  "Localization calibrated DSTWR with bias estimation": "uvio_twotag_localisation_calib_bias_dstwr_std",
        #  "Localization calibrated DSTWR": "uvio_twotag_localisation_dstwr_std",
        #  "Localization standard TWR with bias estimation": "uvio_twotag_localisation_calib_bias_twr",
        #  "Frame aligned localization calibrated DSTWR": "uvio_twotag_localisation_dstwr_std_frame_align",
        "Standard VIO": "standard_ov",
        # "Frame aligned PDOP / SQ URLS initialized UVIO": "uvio_local_frame_align_rangecost_yawcov_4dofjac_fix_grid_split_urls",
        # "Local alignUVIO": "uvio_frame_aligned_local_dstwr_FINAL",
        # "Global AlignUVIO": "uvio_frame_aligned_local_dstwr_FINAL",5
        # "alignUVIO Marginalized Global": "uvio_frame_aligned_local_dstwr_marg_5",
        "AlignUVIO Global": "uvio_frame_aligned_local_dstwr_marg_FINAL",
        # "Schmidt-AlignUVIO Global": "uvio_frame_aligned_local_dstwr_margslam_schmidt",
        # "SLAM Generated Schmidt AlignUVIO": "uvio_slam_FINAL",
        # "SLAM Generated Schmidt AlignUVIO 2": "uvio_frame_aligned_schmidt_slam_2",
        # "SLAM Generated Schmidt AlignUVIO 0": "uvio_frame_aligned_schmidt_slam",
        # "Schmidt-AlignUVIO Global": "uvio_frame_aligned_local_dstwr_margslam_schmidt",
        # "Frame-Aligned UVIO Considered Map": "uvio_frame_aligned_local_schmidt",

        # "Frame-Aligned UVIO UWB Extrinsic Considered Map": "uvio_frame_aligned_local_dstwr_uwbextrinsic_schmidt",

         }

    # SLAM Trials
    # trial_names_labels = \
    #     {"SLAM calibrated DSTWR with bias estimation": "uvio_twotag_slam_calib_bias_dstwr_std",
    #      "SLAM calibrated DSTWR": "uvio_twotag_slam_dstwr_std",
    #      "SLAM standard TWR with bias estimation": "uvio_twotag_slam_calib_bias_twr"}

    estimators = {}

    for label, trial_name in trial_names_labels.items():
        if "global" in label.lower():
            estimators[label] = [
                f"{base}/{trial_name}/run_{i:02d}_global.txt"
                for i in range(5)
            ]
        else:
            estimators[label] = [
                f"{base}/{trial_name}/run_{i:02d}.txt"
                for i in range(5)
            ]
    latex_table_str = {}
    estimator_results = {}
    for label, trials in estimators.items():
        results_list = get_results_list(trial_paths=trials, path_gt=path_gt)
        estimator_results[label] = results_list
        ate_i = compute_ate_trials(results_list)
        print(f"{label} \n| ATE (rot): {ate_i[0]} \n| ATE (pos): {ate_i[1]}")

        attitude_results, position_results = get_attitude_position_results(results_list)

        print("+-------------------+-----------------+------------+-----------+")
        print("| avg rot ATE (deg) | avg pos ATE (m) |  rot aNEES | pos aNEES |")
        print("+-------------------+-----------------+------------+-----------+")
        print(
            f"& {np.mean(ate_i[0]):^17.4f} & "
            f"{np.mean(ate_i[1]):^15.4f} & "
            f"{(np.mean(attitude_results.nees) / attitude_results.dof[0]):^10.4f} & "
            f"{(np.mean(position_results.nees) / position_results.dof[0]):^9.4f} \\\\"
        )
        print("+-------------------+-----------------+------------+-----------+")
        latex_table_str[label] = f"& {np.mean(ate_i[0]):.4f} & {np.mean(ate_i[1]):.4f} & {(np.mean(attitude_results.nees) / attitude_results.dof[0]):.4f} & {(np.mean(position_results.nees) / position_results.dof[0]):.4f} \\\\"

        nees_fig, nees_axs = plt.subplots(2, 1)
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

    # Combined, position-only average error / average 3-sigma plot,
    # with all estimators overlaid on the same axes.
    avg_fig, avg_axs = plot_average_three_sigma_combined(estimator_results)
    if save_figs:
        avg_fig.savefig(base + "/figs/avg_3sigma_combined.pdf")

    if show_figs:
        plt.show()

    for label, table_str in latex_table_str.items():
        print(label, table_str, "\n")


if __name__ == "__main__":
    show_figs = True
    save_figs = False
    main(show_figs, save_figs)