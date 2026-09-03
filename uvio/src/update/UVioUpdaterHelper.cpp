/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2019 Patrick Geneva
 * Copyright (C) 2019 Kevin Eckenhoff
 * Copyright (C) 2019 Guoquan Huang
 * Copyright (C) 2020 Alessandro Fornasier, Control of Networked Systems, University of Klagenfurt, Austria (alessandro.fornasier@aau.at)
 * Copyright (C) 2020 OpenVINS Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "UVioUpdaterHelper.h"

using namespace uvio;

void UVioUpdaterHelper::get_uwb_jacobian_full(
            std::shared_ptr<UVioState> state, 
            std::shared_ptr<UwbData> measurement, 
            Eigen::MatrixXd &H_x,
            Eigen::VectorXd &res, 
            std::vector<std::shared_ptr<ov_type::Type>> &x_order) {


  // Compute the size of the states involved with this feature
  int total_hx = 0;
  std::vector<size_t> state_anc_id;
  std::unordered_map<std::shared_ptr<ov_type::Type>, size_t> map_hx;

  // Add state clone
  std::shared_ptr<ov_type::PoseJPL> clone_I = state->_state->_imu->pose();
  map_hx.insert({clone_I, total_hx});
  x_order.push_back(clone_I);
  total_hx += clone_I->size();

  // [Andrew] Inserted logic for multi-tag agents
  // Find all unique tags in this measurement batch, add to Jacobian match
  // if multi tags have measurements, H_x has enough cols
  if (state->_options.do_calib_uwb_extrinsics) {
    for (const auto &it_range : measurement->uwb_ranges) {
      auto tag_var = state->get_calib_uwb(it_range.tag_id);
      if (map_hx.find(tag_var) == map_hx.end()){
        map_hx.insert({tag_var, total_hx});
        x_order.push_back(tag_var);
        total_hx += tag_var->size();
      }
    }
  }
  // Add all anchors that are not fixed 
  for (const auto &it : state->_calib_GLOBALtoANCHORS) {
    if (!it.second->fixed()) {
      state_anc_id.push_back(it.second->anchor_id());
      map_hx.insert({it.second, total_hx});
      x_order.push_back(it.second);
      total_hx += it.second->size();
    }
  }
  // [Andrew] Initialize Jacobian and Residual matrices
  res = Eigen::VectorXd::Zero(measurement->uwb_ranges.size());
  H_x = Eigen::MatrixXd::Zero(measurement->uwb_ranges.size(), total_hx);

  // [Andrew] Pre-compute IMU orientation
  Eigen::Matrix3d R_GtoI = state->_state->_imu->Rot();
  Eigen::Vector3d p_IinG = state->_state->_imu->pos();

  // [Andrew] Loop through range measurements
  int idx = 0;
  for (const auto &it_range : measurement->uwb_ranges) {
    
    size_t tag_id = it_range.tag_id;
    size_t anchor_id = it_range.anchor_id;
    double range = it_range.range;

    PRINT_DEBUG(YELLOW "[UWB] Processing measurement %d: Tag %zu, Anchor %zu\n" RESET, idx, it_range.tag_id, it_range.anchor_id);

    // Get uwb extrinsics for the specific tag
    auto tag_var = state->get_calib_uwb(tag_id);
    Eigen::Vector3d p_IinU = tag_var->value();

    // Get anchor data
    auto anchor_ptr = state->_calib_GLOBALtoANCHORS.at(anchor_id);
    if (anchor_ptr == nullptr) {
      PRINT_ERROR(RED "[CRITICAL] anchor_ptr is NULL for ID %zu\n" RESET, it_range.anchor_id);
      continue;
    }
    AnchorData anchor; 
    try
    {
      anchor = anchor_ptr->anchor();
    }
    catch(const std::out_of_range &oor)
    {
      PRINT_DEBUG(RED "[UWB Update] No anchor found for the given measurement ID %d" RESET, it_range.anchor_id);
      continue;
    }

    // Compute Residual
    // h(x) =  beta*|| (C_ab r_tz_b + r_zw_a) - r_pw_a|| + gamma
    // beta: dist-dependent bias
    // alpha: constant bias term
    Eigen::Vector3d p_UinG = p_IinG + R_GtoI.transpose() * (-p_IinU);
    double raw_dist = (p_UinG - anchor.p_AinG).norm(); 
    double beta_scale = (1 + anchor.dist_bias);
    res(idx) = range - (beta_scale * raw_dist + anchor.const_bias);
    
    // [Andrew] gamma row vector from UVIO jacobians
    Eigen::RowVector3d gamma = (p_UinG - anchor.p_AinG).transpose() / raw_dist;
    
    // [Andrew] Jacobian wrt IMU Pose
    Eigen::Matrix<double, 1, 6> H_pose;
    Eigen::Matrix3d skew_pU = ov_core::skew_x(-p_IinU);
    H_pose.block<1, 3>(0,0) = beta_scale * gamma * R_GtoI.transpose() * skew_pU;// rotation component
    H_pose.block<1, 3>(0,3) = beta_scale * gamma; 

    H_x.block(idx, map_hx[clone_I], 1, 6) = H_pose;

    // [Andrew] Jacobian wrt UWB extrinsics
    if (state->_options.do_calib_uwb_extrinsics) {
      // [Andrew] Need negative by gamma as the state variable is
      // p_ IinU = -p_UinI = -r_tz_b
      // This is different than the jacobian presented in paper
      H_x.block(idx, map_hx[tag_var], 1, 3) = beta_scale * -gamma * R_GtoI.transpose();
    }

    // [Andrew] Jacobian wrt anchor (position, beta, alpha)
    // [Andrew] TODO: Check if we need fej here?
    if (! anchor.fix) {
      size_t anchor_col = map_hx[anchor_ptr];
      H_x.block(idx, anchor_col, 1 , 3) = beta_scale * -gamma;
      H_x.block(idx, anchor_col + 3, 1, 1).setConstant(1.0);
      H_x.block(idx, anchor_col + 4, 1, 1).setConstant(raw_dist); 
    }

    idx++;
  }
  // [Andrew] Resize to actual processed measurements
  if (idx < measurement->uwb_ranges.size()) {
    H_x.conservativeResize(idx, H_x.cols());
    res.conservativeResize(idx);
  }
}

void UVioUpdaterHelper::get_uwb_jacobian_single(std::shared_ptr<UVioState> state, 
  const double timestamp, 
  const size_t tag_id,
  const size_t anchor_id, 
  const double range, 
  Eigen::MatrixXd &H_x, 
  Eigen::VectorXd &res, 
  std::vector<std::shared_ptr<ov_type::Type> > &x_order) {

    // [Andrew] Safety checks
    std::shared_ptr<UWBAnchor> anchor_ptr;
    if (state->_calib_GLOBALtoANCHORS.find(anchor_id) == state->_calib_GLOBALtoANCHORS.end()) {
      PRINT_DEBUG(RED "[UWB] No anchor found for ID %zu\n" RESET, anchor_id);
      return;
    }
    anchor_ptr = state->_calib_GLOBALtoANCHORS.at(anchor_id);

    if (state->_calib_UWBtoIMU_map.find(tag_id) == state->_calib_UWBtoIMU_map.end()) {
      PRINT_DEBUG(RED "[UWB] No extrinsic variable found for Tag ID %zu\n" RESET, tag_id);
      return;
    }
    auto tag_var = state->_calib_UWBtoIMU_map.at(tag_id);

    std::pair<size_t, size_t> tag_anc_id = {tag_id, anchor_id};
    if (state->_uwb_biases_map.find(tag_anc_id) == state->_uwb_biases_map.end()){
      PRINT_DEBUG(RED "[UWB] No bias found for Tag ID %zu and Anchor ID %zu\n" RESET, tag_id, anchor_id);
      return;
    }
    auto uwb_bias_ptr = state->_uwb_biases_map.at(tag_anc_id);
    
    // [Andrew] Compute state ordering and Jacobian size
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<ov_type::Type>, size_t> map_hx;

    // Add Pose Clone
    std::shared_ptr<ov_type::PoseJPL> clone_I = state->_state->_imu->pose();
    map_hx[clone_I] = total_hx;
    x_order.push_back(clone_I);
    total_hx += clone_I->size();


    // Add alignment variables if doing frame UWB-VIO frame transform
    std::shared_ptr<ov_type::PoseJPL> uwb_align_var;
    if (state->_options.do_calib_uwb_frame_transfrom){
      uwb_align_var = state->_calib_VIOtoUWB_frame_alignment;
      map_hx[uwb_align_var] = total_hx;
      x_order.push_back(uwb_align_var);
      total_hx += uwb_align_var->size();
    }

    // Add extrinsics
    if (state->_options.do_calib_uwb_extrinsics) {
      map_hx[tag_var] = total_hx;
      x_order.push_back(tag_var);
      total_hx += tag_var->size();
    }
    // Add Anchor (if not fixed)
    if (!anchor_ptr->fixed()) {
      map_hx[anchor_ptr] = total_hx;
      x_order.push_back(anchor_ptr);
      total_hx += anchor_ptr->size();
    }

    // Add biases if not fixed
    if (state->_options.do_calib_uwb_biases){
      map_hx[uwb_bias_ptr] = total_hx;
      x_order.push_back(uwb_bias_ptr);
      total_hx += uwb_bias_ptr->size();
    }

    double uwb_const_bias = uwb_bias_ptr->const_bias()->value()(0);
    double uwb_dist_bias = uwb_bias_ptr->dist_bias()->value()(0);

    // Initialize Matrices
    H_x = Eigen::MatrixXd::Zero(1, total_hx);
    res = Eigen::VectorXd::Zero(1);
    // Retrieve current state values
    Eigen::Matrix3d R_VtoI = state->_state->_imu->Rot(); // R_bv
    Eigen::Vector3d p_IinV = state->_state->_imu->pos(); //p_zwv_v
    Eigen::Vector3d p_UinI = tag_var->value(); // p_zt_b

    // Alignment values (Defaults to Identity if not optimizing)
    Eigen::Matrix3d R_av = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_av = Eigen::Vector3d::Zero();
    if (state->_options.do_calib_uwb_frame_transfrom) {
      R_av = state->_calib_VIOtoUWB_frame_alignment->Rot();
      t_av = state->_calib_VIOtoUWB_frame_alignment->pos();
    }
    // Compute Residual and Jacobian blocks
    AnchorData anchor = anchor_ptr->anchor();
    // p_twv_v = p_zwv_v + R_vb * p_tz_b
    // Negative is used as UVioManagerOptions loads imu relative to uwb tag
    Eigen::Vector3d p_UinV = (R_VtoI.transpose() * (p_UinI)) + p_IinV;
    // p_twa_a = R_av * p_twv_v + p_wvwa_a
    Eigen::Vector3d p_UinG = (R_av * p_UinV) + t_av;
    double raw_dist = (p_UinG - anchor.p_AinG).norm(); 
    double beta_scale = (1 + uwb_dist_bias);
    res(0) = range - (beta_scale * raw_dist + uwb_const_bias);

    // [Andrew] gamma row vector from UVIO jacobians
    Eigen::RowVector3d gamma = (p_UinG - anchor.p_AinG).transpose() / raw_dist;
    

    // [Andrew] Jacobian wrt IMU Pose
    Eigen::Matrix<double, 1, 6> H_pose;
    Eigen::Matrix3d skew_pU = ov_core::skew_x(p_UinI);
    H_pose.block<1, 3>(0,0) = beta_scale * gamma * R_av * R_VtoI.transpose() * ov_core::skew_x(p_UinI); // rotation component
    H_pose.block<1, 3>(0,3) = beta_scale * gamma * R_av; // position component
    H_x.block(0, map_hx[clone_I], 1, 6) = H_pose;

    // Jacobian wrt UWB Extrinsics (p_IinU)
    if (state->_options.do_calib_uwb_extrinsics) {
      H_x.block<1, 3>(0, map_hx[tag_var]) = beta_scale * gamma * R_av * R_VtoI.transpose();
    }

    // Jacobian wrt Anchor (Position, Constant Bias, Dist-dependent Bias)
    if (!anchor.fix) {
      PRINT_DEBUG(YELLOW "[UWB] Computing jacobian for anchor [%zu]\n" RESET, anchor_id);
      size_t anchor_col = map_hx[anchor_ptr];
      H_x.block<1, 3>(0, anchor_col) = beta_scale * -gamma; // p_AinG
    }

    if (state->_options.do_calib_uwb_biases){
      PRINT_DEBUG(YELLOW "[UWB] Computing jacobian for UWB bias between tag [%zu] and anchor [%zu]\n" RESET, tag_id, anchor_id);
      size_t bias_col = map_hx[uwb_bias_ptr];
      H_x(0, bias_col) = 1.0;  // alpha (const_bias)
      H_x(0, bias_col + 1) = raw_dist;  // beta (dist_bias)
    }

    if (state->_options.do_calib_uwb_frame_transfrom){
      Eigen::Matrix<double, 1, 6> H_align = Eigen::MatrixXd::Zero(1, 6);
      H_align.block<1, 3>(0,0) = beta_scale * -gamma * ov_core::skew_x(R_av * p_UinV);// rotation component
      H_align.block<1,3>(0, 3) = beta_scale * gamma; // position component

      // If we want the filter to only update Yaw
      H_align(0, 0) = 0.0; // Zero out Roll
      H_align(0, 1) = 0.0; // Zero out Pitch

      H_x.block<1, 6>(0, map_hx[uwb_align_var]) = H_align;
      state->_uwb_alignment_jac = H_align;
    }
    // PRINT_DEBUG(YELLOW "[UWB] Processing measurement %d: Tag %zu, Anchor %zu\n" RESET, idx, it_range.tag_id, it_range.anchor_id);
    // DEBUG
    PRINT_DEBUG(YELLOW "[UWB] Range measurement from tag %zu to anchor %zu = %lf\n" RESET, tag_id, anchor.id, range);
    PRINT_DEBUG(YELLOW "[UWB] Predicted measurement from tag %zu to anchor %zu = %lf\n" RESET, tag_id, anchor.id, (beta_scale * raw_dist) + anchor.const_bias);
    PRINT_DEBUG(YELLOW "[UWB] Residual for tag %zu to anchor %zu = %lf\n" RESET, tag_id, anchor.id, res(0));
    PRINT_DEBUG(YELLOW "[UWB] Const bias and dist bias for tag %zu to anchor %zu = [%lf, %lf]\n" RESET, tag_id, anchor.id, uwb_const_bias, uwb_dist_bias);

}

void UVioUpdaterHelper::get_uwb_active_schmidt_jacobian_single(std::shared_ptr<UVioState> state, const double timestamp, const size_t tag_id, const size_t anchor_id, const double range, Eigen::MatrixXd &H_active, Eigen::MatrixXd &H_schmidt, Eigen::VectorXd &res, std::vector<std::shared_ptr<ov_type::Type>> &x_order_active) {
  // PRINT_DEBUG(MAGENTA "Starting get_uwb_active_schmidt_jacobian_single()\n" RESET);
  // [Andrew] UWB safety checks
  if (!state->_options.do_schmidt_uwb_anchors || !state->_has_initialized_schmidt) {
    PRINT_ERROR(RED " [UVioUpdaterHelper] This function is for schmidt state support, use the orginal get_uwb_jacobian_single function");
    std::exit(EXIT_FAILURE);
  }

  if (state->_calib_GLOBALtoANCHORS.find(anchor_id) == state->_calib_GLOBALtoANCHORS.end()) {
    PRINT_DEBUG(RED "[UWB] No anchor found for ID %zu\n" RESET, anchor_id);
    return;
  }
  std::shared_ptr<UWBAnchor> anchor_ptr = state->_calib_GLOBALtoANCHORS.at(anchor_id);

  if (state->_calib_UWBtoIMU_map.find(tag_id) == state->_calib_UWBtoIMU_map.end()) {
      PRINT_DEBUG(RED "[UWB] No extrinsic variable found for Tag ID %zu\n" RESET, tag_id);
      return;
    }
  std::shared_ptr<ov_type::Vec> tag_var = state->_calib_UWBtoIMU_map.at(tag_id);

  std::pair<size_t, size_t> tag_anc_id = {tag_id, anchor_id};
  if (state->_uwb_biases_map.find(tag_anc_id) == state->_uwb_biases_map.end()){
    PRINT_DEBUG(RED "[UWB] No bias found for Tag ID %zu and Anchor ID %zu\n" RESET, tag_id, anchor_id);
    return;
  }
  auto uwb_bias_ptr = state->_uwb_biases_map.at(tag_anc_id);

  // [Andrew] Compute state orddering and Jacobian size for both and active schmidt state
  int total_hx_active = 0;
  std::unordered_map<std::shared_ptr<ov_type::Type>, size_t> map_hx_active;
  int total_hx_schmidt = 0;
  std::unordered_map<std::shared_ptr<ov_type::Type>, size_t> map_hx_schmidt;

  // Add pose clone to the active state
  std::shared_ptr<ov_type::PoseJPL> clone_I = state->_state->_imu->pose();
  map_hx_active[clone_I] = total_hx_active;
  x_order_active.push_back(clone_I);
  total_hx_active += clone_I->size();

  // Add alignment variables if doing frame UWB-VIO frame transform
  std::shared_ptr<ov_type::PoseJPL> uwb_align_var;
  if (state->_options.do_calib_uwb_frame_transfrom){
    uwb_align_var = state->_calib_VIOtoUWB_frame_alignment;
    map_hx_active[uwb_align_var] = total_hx_active;
    x_order_active.push_back(uwb_align_var);
    total_hx_active += uwb_align_var->size();
  }

  // Add extrinsics
  if (state->_options.do_calib_uwb_extrinsics) {
    map_hx_active[tag_var] = total_hx_active;
    x_order_active.push_back(tag_var);
    total_hx_active += tag_var->size();
  }
  // Add schmidt state anchor map
  if (anchor_ptr->fixed() && state->_options.do_schmidt_uwb_anchors && state->_has_initialized_schmidt){
    map_hx_schmidt[anchor_ptr] = total_hx_schmidt;
    total_hx_schmidt += anchor_ptr->size();
  }

  // Add biases if not fixed
  if (state->_options.do_calib_uwb_biases){
    map_hx_active[uwb_bias_ptr] = total_hx_active;
    x_order_active.push_back(uwb_bias_ptr);
    total_hx_active += uwb_bias_ptr->size();
  }
  double uwb_const_bias = uwb_bias_ptr->const_bias()->value()(0);
  double uwb_dist_bias = uwb_bias_ptr->dist_bias()->value()(0);

  // Initialize Matrices
  H_active = Eigen::MatrixXd::Zero(1, total_hx_active);
  H_schmidt = Eigen::MatrixXd::Zero(1, total_hx_schmidt);
  res = Eigen::VectorXd::Zero(1);

  // Retrieve current state values
  Eigen::Matrix3d R_VtoI = state->_state->_imu->Rot(); // R_bv
  Eigen::Vector3d p_IinV = state->_state->_imu->pos(); //p_zwv_v
  Eigen::Vector3d p_UinI = tag_var->value(); // p_zt_b

  // Alignment values (Defaults to Identity if not optimizing)
  Eigen::Matrix3d R_av = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_av = Eigen::Vector3d::Zero();
  if (state->_options.do_calib_uwb_frame_transfrom) {
    R_av = state->_calib_VIOtoUWB_frame_alignment->Rot();
    t_av = state->_calib_VIOtoUWB_frame_alignment->pos();
  }
  // Compute Residual and Jacobian blocks
  AnchorData anchor = anchor_ptr->anchor();
  // p_twv_v = p_zwv_v + R_vb * p_tz_b
  // Negative is used as UVioManagerOptions loads imu relative to uwb tag
  Eigen::Vector3d p_UinV = (R_VtoI.transpose() * (p_UinI)) + p_IinV;
  // p_twa_a = R_av * p_twv_v + p_wvwa_a
  Eigen::Vector3d p_UinG = (R_av * p_UinV) + t_av;
  double raw_dist = (p_UinG - anchor.p_AinG).norm(); 
  double beta_scale = (1 + uwb_dist_bias);
  res(0) = range - (beta_scale * raw_dist + uwb_const_bias);

  // [Andrew] gamma row vector from UVIO jacobians
  Eigen::RowVector3d gamma = (p_UinG - anchor.p_AinG).transpose() / raw_dist;
  
  // ACTIVE POSES
  // [Andrew] Jacobian wrt IMU Pose
  Eigen::Matrix<double, 1, 6> H_pose;
  Eigen::Matrix3d skew_pU = ov_core::skew_x(p_UinI);
  H_pose.block<1, 3>(0,0) = beta_scale * gamma * R_av * R_VtoI.transpose() * ov_core::skew_x(p_UinI); // rotation component
  H_pose.block<1, 3>(0,3) = beta_scale * gamma * R_av; // position component
  H_active.block(0, map_hx_active[clone_I], 1, 6) = H_pose;

  // Jacobian wrt UWB Extrinsics (p_IinU)
  if (state->_options.do_calib_uwb_extrinsics) {
    H_active.block<1, 3>(0, map_hx_active[tag_var]) = beta_scale * gamma * R_av * R_VtoI.transpose();
  }
  // Jacobian wrt to UWB bias terms
  if (state->_options.do_calib_uwb_biases){
    PRINT_DEBUG(YELLOW "[UWB] Computing jacobian for UWB bias between tag [%zu] and anchor [%zu]\n" RESET, tag_id, anchor_id);
    size_t bias_col = map_hx_active[uwb_bias_ptr];
    H_active(0, bias_col) = 1.0;  // alpha (const_bias)
    H_active(0, bias_col + 1) = raw_dist;  // beta (dist_bias)
  }
  // Jacobian wrt to UWB global alignment state
  if (state->_options.do_calib_uwb_frame_transfrom){
    Eigen::Matrix<double, 1, 6> H_align = Eigen::MatrixXd::Zero(1, 6);
    H_align.block<1, 3>(0,0) = beta_scale * -gamma * ov_core::skew_x(R_av * p_UinV);// rotation component
    H_align.block<1,3>(0, 3) = beta_scale * gamma; // position component

    // If we want the filter to only update Yaw
    H_align(0, 0) = 0.0; // Zero out Roll
    H_align(0, 1) = 0.0; // Zero out Pitch

    H_active.block<1, 6>(0, map_hx_active[uwb_align_var]) = H_align;
  }

  // SCHMIDT STATES
  // Jacobian wrt Anchor (Position, Constant Bias, Dist-dependent Bias)
  if (anchor_ptr->fixed() && state->_options.do_schmidt_uwb_anchors) {
    PRINT_DEBUG(YELLOW "[UWB] Computing jacobian for anchor [%zu]\n" RESET, anchor_id);
    size_t anchor_col = map_hx_schmidt[anchor_ptr];
    H_schmidt.block<1, 3>(0, anchor_col) = beta_scale * -gamma; // p_AinG
  }

  // DEBUG
  PRINT_DEBUG(YELLOW "[UWB] Range measurement from tag %zu to anchor %zu = %lf\n" RESET, tag_id, anchor.id, range);
  PRINT_DEBUG(YELLOW "[UWB] Predicted measurement from tag %zu to anchor %zu = %lf\n" RESET, tag_id, anchor.id, (beta_scale * raw_dist) + anchor.const_bias);
  PRINT_DEBUG(YELLOW "[UWB] Residual for tag %zu to anchor %zu = %lf\n" RESET, tag_id, anchor.id, res(0));
  PRINT_DEBUG(YELLOW "[UWB] Const bias and dist bias for tag %zu to anchor %zu = [%lf, %lf]\n" RESET, tag_id, anchor.id, uwb_const_bias, uwb_dist_bias);



}
