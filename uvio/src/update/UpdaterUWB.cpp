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

void UpdaterUWB::update(std::shared_ptr<UVioState> state, const std::shared_ptr<UwbData>& message) {

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
  Eigen::MatrixXd S = H_x*P_marg*H_x.transpose() + R; //Innovation matrix
  // Mahalanobis distance
  double chi2 = res.dot(S.llt().solve(res)); //r^T(S^-1)r

  // Get our threshold (we precompute up to 500 but handle the case that it is more)
  double chi2_check = _chi_squared_table[res.rows()];

  // Check if we should update or not
  if (chi2 > _options.uwb_chi2_multipler*chi2_check) {
    // Giulio: Change to PRINT_DEBUG after
    PRINT_INFO(RED "UWB chi2 = %f > %f\n" RESET, chi2, _options.uwb_chi2_multipler*chi2_check);
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

void UpdaterUWB::update_single(std::shared_ptr<UVioState> state, const double timestamp, const size_t tag_id, const size_t anchor_id, const double range) {

  // Measurement noise for the uwb update
  // Alessandro: fixed value
  Eigen::MatrixXd R = pow(_options.uwb_sigma_range,2)*Eigen::MatrixXd::Identity(1, 1);

  // Our return values (state jacobian, residual, and order of state jacobian)
  Eigen::MatrixXd H_x;
  Eigen::VectorXd res;
  std::vector<std::shared_ptr<ov_type::Type>> Hx_order;

  // Get the Jacobian for the uwb update
  UVioUpdaterHelper::get_uwb_jacobian_single(state, timestamp, tag_id, anchor_id, range, H_x, res, Hx_order);

  // [Andrew] Safety check
  if (H_x.rows() == 0 || res.rows() == 0) {
    return;
  }

  // Chi2 distance check
  Eigen::MatrixXd P_marg = ov_msckf::StateHelper::get_marginal_covariance(state->_state, Hx_order);
  Eigen::MatrixXd S = H_x*P_marg*H_x.transpose() + R; //Innovation matrix
  double chi2 = res.dot(S.llt().solve(res)); //r^T(S^-1)r

  // Get our threshold (we precompute up to 500 but handle the case that it is more)
  double chi2_check = _chi_squared_table[res.rows()];

  // Check if we should update or not
  bool is_rejected = (chi2 > _options.uwb_chi2_multipler*chi2_check);
  // Draw in rviz
  if (_viz) {
      _viz->visualize_uwb_measurement(tag_id, anchor_id, range, is_rejected);
    }
  
  if (is_rejected) {
    PRINT_INFO(RED "[Updater UWB] Measurement from tag[%zu] to anchor[%zu] rejected: chi2 = %f > %f\n" RESET, tag_id, anchor_id, chi2, _options.uwb_chi2_multipler*chi2_check);
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
