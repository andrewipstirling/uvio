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

#include "UVioPropagator.h"

using namespace uvio;

void UVioPropagator::propagate(std::shared_ptr<UVioState> state, double timestamp) {

  // If the difference between the current update time and state is zero
  // We should crash, as this means we would have two clones at the same time!!!!
  if (state->_state->_timestamp == timestamp) {
    PRINT_ERROR(RED "UVioPropagator::propagate(): Propagation called again at same timestep at last update timestep!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // We should crash if we are trying to propagate backwards
  if (state->_state->_timestamp > timestamp) {
    PRINT_ERROR(RED "UVioPropagator::propagate(): Propagation called trying to propagate backwards in time!!!!\n" RESET);
    PRINT_ERROR(RED "UVioPropagator::propagate(): desired propagation = %.4f\n" RESET, (timestamp - state->_state->_timestamp));
    std::exit(EXIT_FAILURE);
  }

  if (!have_last_prop_time_offset) {
    last_prop_time_offset = state->_state->_calib_dt_CAMtoIMU->value()(0);
    have_last_prop_time_offset = true;
  }

  // Get what our IMU-camera offset should be (t_imu = t_cam + calib_dt)
  double t_off_new = state->_state->_calib_dt_CAMtoIMU->value()(0);

  // First lets construct an IMU vector of measurements we need
  double time0 = state->_state->_timestamp + last_prop_time_offset;
  double time1 = timestamp + t_off_new;
  std::vector<ov_core::ImuData> prop_data;
  {
    std::lock_guard<std::mutex> lck(imu_data_mtx);
    prop_data = select_imu_readings(imu_data, time0, time1);
  }

  // We are going to sum up all the state transition matrices, so we can do a single large multiplication at the end
  // Phi_summed = Phi_i*Phi_summed
  // Q_summed = Phi_i*Q_summed*Phi_i^T + Q_i
  // After summing we can multiple the total phi to get the updated covariance
  // We will then add the noise to the IMU portion of the state
  Eigen::MatrixXd Phi_summed = Eigen::MatrixXd::Identity(state->_state->imu_intrinsic_size() + 15, state->_state->imu_intrinsic_size() + 15);
  Eigen::MatrixXd Qd_summed = Eigen::MatrixXd::Zero(state->_state->imu_intrinsic_size() + 15, state->_state->imu_intrinsic_size() + 15);
  double dt_summed = 0;

  // Loop through all IMU messages, and use them to move the state forward in time
  // This uses the zero'th order quat, and then constant acceleration discrete
  if (prop_data.size() > 1) {
    for (size_t i = 0; i < prop_data.size() - 1; i++) {

      // Get the next state Jacobian and noise Jacobian for this IMU reading
      Eigen::MatrixXd F, Qdi;
      predict_and_compute(state->_state, prop_data.at(i), prop_data.at(i + 1), F, Qdi);

      // Next we should propagate our IMU covariance
      // Pii' = F*Pii*F.transpose() + G*Q*G.transpose()
      // Pci' = F*Pci and Pic' = Pic*F.transpose()
      // NOTE: Here we are summing the state transition F so we can do a single mutiplication later
      // NOTE: Phi_summed = Phi_i*Phi_summed
      // NOTE: Q_summed = Phi_i*Q_summed*Phi_i^T + G*Q_i*G^T
      Phi_summed = F * Phi_summed;
      Qd_summed = F * Qd_summed * F.transpose() + Qdi;
      Qd_summed = 0.5 * (Qd_summed + Qd_summed.transpose());
      dt_summed += prop_data.at(i + 1).timestamp - prop_data.at(i).timestamp;
    }
  }
  // Crash Here if 
  double expected_dt = time1 - time0;
  double diff = std::abs(expected_dt - dt_summed);
  if (diff >= 1e-4){
    PRINT_ERROR(RED "UVioPropagator::propagate(): dt mismatch: expected=%.6f, summed=%.6f, diff=%.6f, samples=%zu\n" RESET, expected_dt, dt_summed, diff, prop_data.size());
    assert(false);
    //TODO: Figure out why we crash here
  }
  

  

  // Do the update to the covariance with our "summed" state transition and IMU noise addition...
  std::vector<std::shared_ptr<ov_type::Type>> Phi_order;
  Phi_order.push_back(state->_state->_imu);
  if (state->_state->_options.do_calib_imu_intrinsics) {
    Phi_order.push_back(state->_state->_calib_imu_dw);
    Phi_order.push_back(state->_state->_calib_imu_da);
    if (state->_state->_options.do_calib_imu_g_sensitivity) {
      Phi_order.push_back(state->_state->_calib_imu_tg);
    }
    if (state->_state->_options.imu_model == ov_msckf::StateOptions::ImuModel::KALIBR) {
      Phi_order.push_back(state->_state->_calib_imu_GYROtoIMU);
    } else {
      Phi_order.push_back(state->_state->_calib_imu_ACCtoIMU);
    }
  }
  ov_msckf::StateHelper::EKFPropagation(state->_state, Phi_order, Phi_order, Phi_summed, Qd_summed);

  // Set timestamp data
  state->_state->_timestamp = timestamp;
  last_prop_time_offset = t_off_new;

  // If we're tracking schmidt states
  if (state->_options.do_schmidt_uwb_anchors){

    // Confirm anchor schmidt ID map isn't empty
    if (state->_anchor_schmidt_idx_map.empty()){
      PRINT_ERROR(RED "UVioPropagator::propagate(): Schmidt state anchor index map is empty\n" RESET);
      std::exit(EXIT_FAILURE);
    }
    // Find where the IMU states start in the active covariance layout
    int start_id = Phi_order.at(0)->id();
    int phi_size = Phi_summed.rows();
    int total_anchors_size = 3 * state->_anchor_schmidt_idx_map.size();
    // Now perform P^{AS} cross-covariance propagation
    // P^{AS}_k = \Phi_{k-1} * P^{AS}_{k-1}
    state->_Cov_cross.block(start_id, 0, phi_size, total_anchors_size) = Phi_summed * state->_Cov_cross.block(start_id, 0, phi_size, total_anchors_size);
  }
  
}

void UVioPropagator::propagate_and_clone(std::shared_ptr<UVioState> state, double timestamp){
  // Get prop date for computing the relevant imu_data
  if (!have_last_prop_time_offset) {
    last_prop_time_offset = state->_state->_calib_dt_CAMtoIMU->value()(0);
    have_last_prop_time_offset = true;
  }

  // Get what our IMU-camera offset should be (t_imu = t_cam + calib_dt)
  double t_off_new = state->_state->_calib_dt_CAMtoIMU->value()(0);

  // First lets construct an IMU vector of measurements we need
  double time0 = state->_state->_timestamp + last_prop_time_offset;
  double time1 = timestamp + t_off_new;
  std::vector<ov_core::ImuData> prop_data;
  {
    std::lock_guard<std::mutex> lck(imu_data_mtx);
    prop_data = select_imu_readings(imu_data, time0, time1);
  }
  // Last angular velocity (used for cloning when estimating time offset)
  // Remember to correct them before we store them
  Eigen::Vector3d last_a = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_w = Eigen::Vector3d::Zero();
  if (!prop_data.empty()) {
    Eigen::Matrix3d Dw = ov_msckf::State::Dm(state->_state->_options.imu_model, state->_state->_calib_imu_dw->value());
    Eigen::Matrix3d Da = ov_msckf::State::Dm(state->_state->_options.imu_model, state->_state->_calib_imu_da->value());
    Eigen::Matrix3d Tg = ov_msckf::State::Tg(state->_state->_calib_imu_tg->value());
    last_a = state->_state->_calib_imu_ACCtoIMU->Rot() * Da * (prop_data.at(prop_data.size() - 1).am - state->_state->_imu->bias_a());
    last_w = state->_state->_calib_imu_GYROtoIMU->Rot() * Dw * (prop_data.at(prop_data.size() - 1).wm - state->_state->_imu->bias_g() - Tg * last_a);
  }
  // Call our propagate function
  propagate(state, timestamp);

  // Perform the active state clone
  ov_msckf::StateHelper::augment_clone(state->_state, last_w);

  // Now perform bookeeping of active-schmidt cross covariance after cloning
  if(state->_options.do_schmidt_uwb_anchors){
      // Confirm anchor schmidt ID map isn't empty
    if (state->_anchor_schmidt_idx_map.empty()){
      PRINT_ERROR(RED "UVioPropagator::propagate(): Schmidt state anchor index map is empty\n" RESET);
      std::exit(EXIT_FAILURE);
    }

    int n_schmidt = state->_Cov_cross.cols();
    int old_rows = state->_Cov_cross.rows();
    const auto clone = state->_state->_clones_IMU.rbegin()->second;
    int clone_size = clone->size();
    int clone_id = clone->id();
    int cur_state_id = state->_state->_imu->id();

    state->_Cov_cross.conservativeResize(old_rows + clone_size, n_schmidt);
    state->_Cov_cross.block(clone_id, 0, clone_size, n_schmidt) =
      state->_Cov_cross.block(cur_state_id, 0, clone_size, n_schmidt);
    
  }

}
