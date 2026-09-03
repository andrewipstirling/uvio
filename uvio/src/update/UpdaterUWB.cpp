/*
 * Copyright (C) 2020 Alessandro Fornasier, Control of Networked Systems, University of Klagenfurt, Austria.
 *
 * All rights reserved.
 *
 * You can contact the author at <alessandro.fornasier@aau.at>
 */

#include "UpdaterUWB.h"
#include "ros/UVIOROS1Visualizer.h"

using namespace uvio;

void UpdaterUWB::update(std::shared_ptr<UVioState> state, const std::shared_ptr<UwbData> &message) {

  // [Andrew] Prepare return values
  Eigen::MatrixXd H_x;
  Eigen::VectorXd res;
  std::vector<std::shared_ptr<ov_type::Type>> Hx_order;

  // Get the Jacobian for the uwb update
  // [Andrew] This now handles multiple tags
  UVioUpdaterHelper::get_uwb_jacobian_full(state, message, H_x, res, Hx_order);

  // [Andrew] Safety check, if all measurements in this batch were
  // invalid (e.g., unknown anchors), exit.
  if (res.rows() == 0) {
    return;
  }
  // Match size of R to the actual number of VALID measurements
  // Use the row count of H_x/res rather than the raw message size.
  Eigen::MatrixXd R = pow(_options.uwb_sigma_range, 2) * Eigen::MatrixXd::Identity(res.rows(), res.rows());

  // Chi2 distance check
  Eigen::MatrixXd P_marg = ov_msckf::StateHelper::get_marginal_covariance(state->_state, Hx_order);
  Eigen::MatrixXd S = H_x * P_marg * H_x.transpose() + R; // Innovation matrix
  // Mahalanobis distance
  double chi2 = res.dot(S.llt().solve(res)); // r^T(S^-1)r

  // Get our threshold (we precompute up to 500 but handle the case that it is more)
  double chi2_check = _chi_squared_table[res.rows()];

  // Check if we should update or not
  if (chi2 > _options.uwb_chi2_multipler * chi2_check) {
    // Giulio: Change to PRINT_DEBUG after
    PRINT_INFO(RED "UWB chi2 = %f > %f\n" RESET, chi2, _options.uwb_chi2_multipler * chi2_check);
    return;
  }

  // We are good! Perform measurement compression
  UVioUpdaterHelper::measurement_compress_inplace(H_x, res);
  if (H_x.rows() < 1) {
    return;
  }

  // Update the state
  ov_msckf::StateHelper::EKFUpdate(state->_state, Hx_order, H_x, res, R);
}

void UpdaterUWB::update_single(std::shared_ptr<UVioState> state, const double timestamp, const size_t tag_id, const size_t anchor_id,
                               const double range, const double std_dev) {

  // [Andrew] Assign meas_noise if std_dev is empty/0.0
  Eigen::Matrix<double, 1, 1> R;

  const double sigma = (std_dev > 0.0) ? std_dev : _options.uwb_sigma_range;

  // [Andrew] Square the measurement noise
  R(0, 0) = sigma * sigma;
  if (state->_has_marginalized_frame_alignment && !state->_options.do_calib_uwb_frame_transfrom){
    Eigen::MatrixXd P_align = state->_marg_alignment_cov;
    R(0.0) += state->_uwb_range_alignment_cov;
    R(0,0) *= state->_options.marg_frame_align_cov_inflation;
    PRINT_INFO(GREEN "Inflated UWB covariance of R = %.4f\n" RESET, R(0,0));
  }

  // Our return values (state jacobian, residual, and order of state jacobian)
  Eigen::MatrixXd H_x, H_schmidt;
  Eigen::VectorXd res;
  std::vector<std::shared_ptr<ov_type::Type>> Hx_order;
  std::vector<std::shared_ptr<ov_type::Type>> Hx_order_schmidt;

  // Get the Jacobian for the uwb update
  if (state->_options.do_schmidt_uwb_anchors && state->_has_initialized_schmidt) {
    UVioUpdaterHelper::get_uwb_active_schmidt_jacobian_single(state, timestamp, tag_id, anchor_id, range, H_x, H_schmidt, res, Hx_order);
    Hx_order_schmidt.push_back(state->_calib_GLOBALtoANCHORS.at(anchor_id));
  } else {
    UVioUpdaterHelper::get_uwb_jacobian_single(state, timestamp, tag_id, anchor_id, range, H_x, res, Hx_order);
  }

  // [Andrew] Safety check
  if (H_x.rows() == 0 || res.rows() == 0) {
    return;
  }

  // Chi2 distance check
  Eigen::MatrixXd S;
  Eigen::MatrixXd P_AA, P_AS;
  if (state->_options.do_schmidt_uwb_anchors && state->_has_initialized_schmidt) {
    Eigen::MatrixXd P_AA = ov_msckf::StateHelper::get_marginal_covariance(state->_state, Hx_order);
    if (state->_anchor_schmidt_idx_map.find(anchor_id) == state->_anchor_schmidt_idx_map.end()) {
      PRINT_ERROR(RED "Anchor ID %d not in the Schmidt inex map\n" RESET, anchor_id);
      std::exit(EXIT_FAILURE);
    }
    int anc_idx = state->_anchor_schmidt_idx_map.at(anchor_id);
    Eigen::MatrixXd P_SS = state->_Cov_schmidt.block<3, 3>(anc_idx, anc_idx);

    Eigen::MatrixXd P_AS = state->get_marginal_active_schmidt_cross_covariance(Hx_order, anchor_id);

    S = H_x * P_AA * H_x.transpose() + H_x * P_AS * H_schmidt.transpose() + H_schmidt * P_AS.transpose() * H_x.transpose() +
        H_schmidt * P_SS * H_schmidt.transpose() + R;
  } else {
    Eigen::MatrixXd P_marg = ov_msckf::StateHelper::get_marginal_covariance(state->_state, Hx_order);
    S = H_x * P_marg * H_x.transpose() + R; // Innovation matrix
  }
  double normalized_innovation = (res(0) * res(0)) / S(0, 0);
  PRINT_INFO(GREEN "[UWB] Normalized innovation = %.4f (expected ~1.0 on average)\n" RESET, normalized_innovation);
  PRINT_INFO(GREEN "[UWB] S = %.6f, R = %.6f, ratio = %.4f\n", 
           S(0,0), R(0,0), S(0,0)/R(0,0));
  double chi2 = res.dot(S.llt().solve(res)); // r^T(S^-1)r

  // Get our threshold (we precompute up to 500 but handle the case that it is more)
  double chi2_check = _chi_squared_table[res.rows()];

  // Check if we should update or not
  bool is_rejected = (chi2 > _options.uwb_chi2_multipler * chi2_check);
  // Draw in rviz
  if (_viz) {
    _viz->visualize_uwb_measurement(tag_id, anchor_id, range, is_rejected);
  }

  if (is_rejected) {
    PRINT_INFO(RED "[Updater UWB] Measurement from tag[%zu] to anchor[%zu] rejected: chi2 = %f > %f with R=%.3f\n" RESET, tag_id, anchor_id,
               chi2, _options.uwb_chi2_multipler * chi2_check, R(0, 0));
    return;
  }

  // We are good! Perform measurement compression
  if (state->_options.do_schmidt_uwb_anchors &&state->_has_initialized_schmidt) {
    schmidt_ekf_update_single(state, Hx_order, anchor_id, H_x, H_schmidt, res, R);
  }
  // Do regular update instead
  else {
    UVioUpdaterHelper::measurement_compress_inplace(H_x, res);
    if (H_x.rows() < 1) {
      return;
    }
    // Update the state
    ov_msckf::StateHelper::EKFUpdate(state->_state, Hx_order, H_x, res, R);
  }
}
// [ANDREW] TODO: Generalize this for batch of UWB measurements m
void UpdaterUWB::schmidt_ekf_update_single(std::shared_ptr<UVioState> state,  const std::vector<std::shared_ptr<ov_type::Type>> &H_order_active, const size_t anchor_id,const Eigen::MatrixXd &H_active, const Eigen::MatrixXd &H_schmidt, const Eigen::VectorXd &res, const Eigen::MatrixXd &R) {

  assert(res.rows() == R.rows());
  assert(H_active.rows() == res.rows());
  assert(H_schmidt.rows() == res.rows());
  assert(H_schmidt.cols() == 3);
  // PRINT_DEBUG(MAGENTA "Starting schmidt_ekf_update_single()\n" RESET);
  int a = state->_state->max_covariance_size();
  int size_meas = (int)res.rows();
  assert(size_meas == 1);
  // column offset for this anchor in _Cov_schmidt and _Cov_cross
  int anchor_col = state->_anchor_schmidt_idx_map.at(anchor_id);
  // Get H offsets for active variables
  int current_it = 0;
  std::vector<int> H_id;
  for (const auto &var : H_order_active) {
    H_id.push_back(current_it);
    current_it += var->size();
  }
  // current_it is now total size of active measurement variables

  // Build L_a = P^{AA} * H_a^T + P^{AS} * H_s^T
  // For each active variable x_i, the i-th block row is:
  //   L_a[i] = sum_j  P^{AA}_{ij} * H_a_j^T
  //           + P^{AS}_{i, anchor} * H_anchor^T
  // where j indexes over H_order_active
  auto &cov = ov_msckf::StateHelper::get_active_covariance(state->_state);
  Eigen::MatrixXd L_a = Eigen::MatrixXd::Zero(a, size_meas);
  for (const auto &var : ov_msckf::StateHelper::get_active_vars(state->_state)) {
    Eigen::MatrixXd L_a_i = Eigen::MatrixXd::Zero(var->size(), size_meas);
    for (size_t i = 0; i < H_order_active.size(); i++) {
      std::shared_ptr<ov_type::Type> meas_var = H_order_active.at(i);
      L_a_i.noalias() += cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
                         H_active.block(0, H_id[i], size_meas, meas_var->size()).transpose();
    }
    L_a_i.noalias() += state->_Cov_cross.block(var->id(), anchor_col, var->size(), 3) * H_schmidt.transpose();

    L_a.block(var->id(), 0, var->size(), size_meas) = L_a_i;
  }

  // Build L_s = P^{SA} * H_a^T + P^{SS} * H_s^T   (3 x m)
  //
  // P^{SA} = P^{AS}^T, so for the anchor block:
  //   L_s = P^{AS}[active_marg_vars, anchor]^T * H_a^T
  //        + P^{SS}[anchor, anchor] * H_anchor^T
  Eigen::MatrixXd L_s = Eigen::MatrixXd::Zero(3, size_meas);
  // Schmidt-active contribution: P^{SA} * H_a^T
  // = _Cov_cross[H_order_active rows, anchor_col]^T * H_a^T
  for (size_t i = 0; i < H_order_active.size(); i++) {
    std::shared_ptr<ov_type::Type> meas_var = H_order_active.at(i);
    Eigen::MatrixXd P_SA_i = state->_Cov_cross.block(meas_var->id(), anchor_col, meas_var->size(), 3).transpose();
    L_s.noalias() += P_SA_i * H_active.block(0, H_id[i], size_meas, meas_var->size()).transpose();
  }

  // Schmidt - Schmidt contribution
  L_s.noalias() += state->_Cov_schmidt.block<3,3>(anchor_col, anchor_col) * H_schmidt.transpose();

  // Build innovation covariance S using the marginal blocks.
  // Build P_small from marginal covariance functions, then stack
  // S = H_small * P_small * H_small^T + R
  // where H_small = [H_active | H_anchor]
  //       P_small = [ P^{AA}_marg   P^{AS}_marg ]
  //                 [ P^{AS}_marg^T P^{SS}_3x3  ]
  int size_small_active  = current_it;
  int size_small_schmidt = 3;
  int size_small_state   = size_small_active + size_small_schmidt;

  Eigen::MatrixXd P_small =
      Eigen::MatrixXd::Zero(size_small_state, size_small_state);

  // P^{AA} marginal block
  P_small.block(0, 0, size_small_active, size_small_active) =
      ov_msckf::StateHelper::get_marginal_covariance(state->_state, H_order_active);

  // P^{SS} block for this anchor
  P_small.block(size_small_active, size_small_active, 3, 3) =
      state->_Cov_schmidt.block<3, 3>(anchor_col, anchor_col);

  // P^{AS} cross block (size_small_active x 3)
  Eigen::MatrixXd P_AS_marg = state->get_marginal_active_schmidt_cross_covariance(
      H_order_active, anchor_id);
  P_small.block(0, size_small_active, size_small_active, 3) = P_AS_marg;
  P_small.block(size_small_active, 0, 3, size_small_active) = P_AS_marg.transpose();

  // Stack Jacobian: H_small = [H_active | H_anchor]
  Eigen::MatrixXd H_small =
      Eigen::MatrixXd::Zero(size_meas, size_small_state);
  H_small.block(0, 0,                 size_meas, size_small_active)  = H_active;
  H_small.block(0, size_small_active, size_meas, size_small_schmidt) = H_schmidt;

  Eigen::MatrixXd S(R.rows(), R.cols());
  S.triangularView<Eigen::Upper>() = H_small * P_small * H_small.transpose();
  S.triangularView<Eigen::Upper>() += R;
  Eigen::MatrixXd Sinv = Eigen::MatrixXd::Identity(R.rows(), R.rows());
  S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv);
  Sinv = Sinv.selfadjointView<Eigen::Upper>();

  // Kalman gain for the active state
  //   K = L_a * S^{-1}   (a x m)
  Eigen::MatrixXd K = L_a * Sinv;

  // P^{AA} update:  P^{AA} -= K * L_a^T
  // ----------------------------------------------------------------
  cov.triangularView<Eigen::Upper>() -= K * L_a.transpose();
  cov = cov.selfadjointView<Eigen::Upper>();
  // ov_msckf::StateHelper::set_active_covariance(state->_state, cov);
  // Positive semi-definite check
  Eigen::VectorXd diags = cov.diagonal();
  bool found_neg = false;
  for (int i = 0; i < diags.size(); i++) {
    if (diags(i) < 0.0) {
      PRINT_WARNING(RED "[UpdaterUWB] Negative diagonal in P^{AA} after Schmidt update: %.10f "
                    "at index %d\n" RESET, diags(i), i);
      found_neg = true;
    }
  }
  if (found_neg) std::exit(EXIT_FAILURE);

  // P^{AS} update:  P^{AS} -= K * L_s^T
  state->_Cov_cross.block(0, anchor_col, a, 3) -= K * L_s.transpose();

  Eigen::VectorXd dx = K * res;
  // Confirm this update is working for the state->_state reference
  for (const auto &var : ov_msckf::StateHelper::get_active_vars(state->_state)) {
    var->update(dx.block(var->id(), 0, var->size(), 1));
  }

  // Update camera intrinsic objects if calibrating online
  if (state->_state->_options.do_calib_camera_intrinsics) {
    for (auto const &calib : state->_state->_cam_intrinsics) {
      state->_state->_cam_intrinsics_cameras.at(calib.first)->set_value(
          calib.second->value());
    }
  }
  // PRINT_DEBUG(MAGENTA "Finished schmidt_ekf_update_single()\n" RESET);
}
