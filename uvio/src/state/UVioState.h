/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2022 Patrick Geneva
 * Copyright (C) 2018-2022 Guoquan Huang
 * Copyright (C) 2018-2022 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
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

#ifndef UVIO_STATE_H
#define UVIO_STATE_H

#include "UVioStateOptions.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/UWBAnchor.h"
#include "types/UWBBIas.h"

namespace uvio {

/**
 * @brief State of our filter
 *
 * This state has all the current estimates for the filter.
 * This system is modeled after the MSCKF filter, thus we have a sliding window of clones.
 * We additionally have more parameters for online estimation of calibration and SLAM features.
 * We also have the covariance of the system, which should be managed using the StateHelper class.
 */
struct UVioState {

  UVioState(UVioStateOptions &options, std::shared_ptr<ov_msckf::State> state) : _options(options), _state(state) {
    // Initialize frame transform
    _calib_VIOtoUWB_frame_alignment = std::make_shared<ov_type::PoseJPL>();

    // Pre-Allocate the schmidt and cross covariance matrices
    if (_options.do_schmidt_uwb_anchors) {
      // P^{SS} : (3 * num_anchors, 3 * num_anchors)
      int size_schmidt = 3 * _options.max_schmidt_anchors;
      _Cov_schmidt = Eigen::MatrixXd(size_schmidt, size_schmidt);

      // P^{AS} : (size of active state, size of schmidt state)
      // Automatically set to zero due to no cross covariance
      _Cov_cross = Eigen::MatrixXd::Zero(_state->max_covariance_size(), size_schmidt);

      // Set this to false until UWB measurements are being received
      _has_initialized_schmidt = false;
    }

    // Initialize the uwb extrinsics map
    for (size_t tag_id : _options.tag_ids) {
      // Create new 3x1 vector state for each tag
      _calib_UWBtoIMU_map[tag_id] = std::make_shared<ov_type::Vec>(3);

      PRINT_DEBUG(GREEN "[UVioState] Initialized state variable for Tag ID: %zu\n" RESET, tag_id);

      for (size_t anc_id : _options.anchor_ids) {
        std::pair<size_t, size_t> tag_anc_pair{tag_id, anc_id};
        _uwb_biases_map[tag_anc_pair] = std::make_shared<UWBBias>(0.0, 0.0);
      }
    }
  }

  ~UVioState() {}

  /// Struct containing filter options
  UVioStateOptions _options;

  /// Calibration position for the uwb sensor (p_IinU)
  /// Map from Tag Index (0, 1, 2...) -> Extrinsic Calibration Variable (p_UinI or p_tz_b)
  std::map<size_t, std::shared_ptr<ov_type::Vec>> _calib_UWBtoIMU_map;

  // For localization, frame alignment state variables
  std::shared_ptr<ov_type::PoseJPL> _calib_VIOtoUWB_frame_alignment;

  /**
   * @brief Map from <TagID, AnchorID> -> Biases (Constant & Distance)
   * This captures the unique hardware/multipath characteristics of each pair.
   */
  std::map<std::pair<size_t, size_t>, std::shared_ptr<UWBBias>> _uwb_biases_map;

  /// Positions of the uwb anchors (id, UWB_anchor)
  std::unordered_map<size_t, std::shared_ptr<UWBAnchor>> _calib_GLOBALtoANCHORS;

  // Boolean to check if we have initialized the schmidt state / covariances
  bool _has_initialized_schmidt;

  // Schmidt state covariance for UWB anchors P^{SS}
  Eigen::MatrixXd _Cov_schmidt;

  // Cross covariance between the active and Schmidt state P^{AS}
  Eigen::MatrixXd _Cov_cross;

  // Maps Anchor ID -> index in the _Cov_shmidt matrix
  std::map<size_t, int> _anchor_schmidt_idx_map;

  /// Pointer to the ov_msckf::State object (base active state)
  const std::shared_ptr<ov_msckf::State> _state;

  // Helper to get a specific tag's variable safely
  std::shared_ptr<ov_type::Vec> get_calib_uwb(size_t tag_id) {
    if (_calib_UWBtoIMU_map.find(tag_id) != _calib_UWBtoIMU_map.end()) {
      return _calib_UWBtoIMU_map.at(tag_id);
    }
    // Fallback: if tag_id doesn't exist (e.g. ROS sends a new ID),
    // return the first tag or handle error
    PRINT_WARNING("[UVioState] Tag ID %zu doesn't exist", tag_id)
    return _calib_UWBtoIMU_map.begin()->second;
  }

  std::shared_ptr<UWBBias> get_uwb_biases(size_t tag_id, size_t anchor_id) {
    std::pair<size_t, size_t> tag_anchor_key{tag_id, anchor_id};
    if (_uwb_biases_map.find(tag_anchor_key) == _uwb_biases_map.end()) {
      _uwb_biases_map[tag_anchor_key] = std::make_shared<UWBBias>(0.0, 0.0);
      PRINT_ERROR(RED "[UVIOState] Missing bias state for Tag %zu <-> Anchor %zu" RESET, tag_id, anchor_id);
      return nullptr;
    }

    return _uwb_biases_map.at(tag_anchor_key);
  }

  void register_schmidt(size_t anc_id, Eigen::Matrix3d anc_cov) {
    // Get size of map, informs the index to insert to
    size_t cur_size_map = _anchor_schmidt_idx_map.size();
    int idx_to_insert = (int)3 * cur_size_map;
    int new_schmidt_size = idx_to_insert + 3;

    _Cov_schmidt.conservativeResize(new_schmidt_size, new_schmidt_size);

    // Insert values into the schmidt covariance
    _Cov_schmidt.block<3, 3>(idx_to_insert, idx_to_insert) = anc_cov;
    _Cov_schmidt.block(0, idx_to_insert, idx_to_insert, 3).setZero();
    _Cov_schmidt.block(idx_to_insert, 0, 3, idx_to_insert).setZero();
    // Insert the anchor's prior/initial covariance matrix into the diagonal block
    _Cov_schmidt.block<3, 3>(idx_to_insert, idx_to_insert) = anc_cov;

    // Set cross covariance to zero initially
    // Use dynamic sized block function
    int size_active_cov = _state->max_covariance_size();
    _Cov_cross.conservativeResize(size_active_cov, new_schmidt_size);
    _Cov_cross.block(0, idx_to_insert, size_active_cov, 3).setZero();

    // Insert index to map
    _anchor_schmidt_idx_map[anc_id] = idx_to_insert;
  }

  void marginalize_active_schmidt(std::shared_ptr<ov_type::Type> marg) {
    // First remove the relevant rows from the cross-covariance matrix
    // between the active given state and the Schmidt state, so need to
    // marginalize out the given rows from the cross covariance
    // PRINT_DEBUG(MAGENTA "Starting UVioState::marginalize_active_schmidt\n" RESET);
    // First confirm we have a schmidt state
    if (_options.do_schmidt_uwb_anchors && _has_initialized_schmidt) {
      int marg_size = marg->size();
      int marg_id = marg->id();
      // Size of variables after the marginalization
      int x2_size = (int)_Cov_cross.rows() - marg_id - marg_size;
      // Shift covariance block to the correct location
      if (x2_size > 0) {
        _Cov_cross.block(marg_id, 0, x2_size, _Cov_cross.cols()) =
            _Cov_cross.block(marg_id + marg_size, 0, x2_size, _Cov_cross.cols()).eval();
      }
      // Remove the bottom row corresponding to the marginalized var
      _Cov_cross.conservativeResize(_Cov_cross.rows() - marg_size, _Cov_cross.cols());
    }
    // Now proceed with marginalizing the active state
    ov_msckf::StateHelper::marginalize(_state, marg);
    // PRINT_DEBUG(MAGENTA "Finished UVioState::marginalize_active_schmidt\n" RESET);
  }

  void marginalize_slam() {
    // PRINT_DEBUG(MAGENTA "Starting UVioState::marginalize_slam\n" RESET);
    // Remove SLAM features that have their marginalization flag set
    // We also check that we do not remove any aruoctag landmarks
    int ct_marginalized = 0;
    auto it0 = _state->_features_SLAM.begin();
    while (it0 != _state->_features_SLAM.end()) {
      if ((*it0).second->should_marg && (int)(*it0).first > 4 * _state->_options.max_aruco_features) {
        marginalize_active_schmidt((*it0).second);
        it0 = _state->_features_SLAM.erase(it0);
        ct_marginalized++;
      } else {
        it0++;
      }
    }
    // PRINT_DEBUG(MAGENTA "Finished UVioState::marginalize_slam\n" RESET);
  }

  void marginalize_old_clone() {
    // PRINT_DEBUG(MAGENTA "Starting UVioState::marginalize_old_clone\n" RESET);
    if ((int)_state->_clones_IMU.size() > _state->_options.max_clone_size) { // Get timestamp of oldest clone
      double marginal_time = _state->margtimestep();
      // Lock the mutex to avoid deleting any elements from _clones_IMU while accessing it from other threads
      std::lock_guard<std::mutex> lock(_state->_mutex_state);
      assert(marginal_time != INFINITY);
      marginalize_active_schmidt(_state->_clones_IMU.at(marginal_time));
      // Note that the marginalizer should have already deleted the clone
      // Thus we just need to remove the pointer to it from our state
      _state->_clones_IMU.erase(marginal_time);
    }
  }

  void apply_cross_covariance_correction(
    const Eigen::MatrixXd &P_AA_before,
    const Eigen::MatrixXd &P_AA_after,
    int n_before) {

  // PRINT_DEBUG(MAGENTA "Starting UVioState::apply_cross_covariance_correction\n" RESET);

  if (!_options.do_schmidt_uwb_anchors) return;
  if (!_has_initialized_schmidt) return;
  if (_anchor_schmidt_idx_map.empty()) return;

  // P_AA_after may be larger than P_AA_before if new variables were initialized.
  // The correction (I - KH) only applies to the n_before rows that existed
  // before the update. Newly added rows have zero cross-covariance by construction
  // and do not need correction.
  assert(P_AA_before.rows() == n_before);
  assert(P_AA_after.rows() >= n_before);
  assert(_Cov_cross.rows() >= n_before);

  int m = 3 * (int)_anchor_schmidt_idx_map.size();

  // Extract only the pre-existing rows of _Cov_cross
  Eigen::MatrixXd P_AS_old = _Cov_cross.topRows(n_before);

  // Solve P_AA_before * X = P_AS_old  =>  X = P_AA_before^{-1} * P_AS_old
  Eigen::MatrixXd X = P_AA_before.ldlt().solve(P_AS_old);

  // Apply correction only to the n_before rows
  // P^{AS}_new[0:n_before] = P_AA_after[0:n_before, 0:n_before] * X
  _Cov_cross.topRows(n_before) =
      P_AA_after.topLeftCorner(n_before, n_before) * X;

  // Rows n_before onwards (newly initialized SLAM features) remain zero —
  // already set by the conservativeResize + setZero in the caller
}
  Eigen::MatrixXd get_marginal_active_schmidt_cross_covariance(
      const std::vector<std::shared_ptr<ov_type::Type>> &vars_active,
      const size_t anchor_id) {

    if (vars_active.size() == 0) {
      PRINT_ERROR("UVioState::get_marginal_active_schmidt_cross_covariance() - One of the variable lists is empty, returning empty cross covariance\n");
    }
    int active_size = 0;
    for (const auto &var : vars_active){
      active_size += var->size();
    }

    int schmidt_size = _calib_GLOBALtoANCHORS.at(anchor_id)->size();
    assert (schmidt_size == 3);
    auto anchor_idx = _anchor_schmidt_idx_map.at(anchor_id);
    // Construct the return covariance
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(active_size, schmidt_size);

    // For each variable, copy over the other variable cross terms
    int a_idx = 0;
    // Fill into cov from each vars vector
    for (size_t a = 0; a < vars_active.size(); a++) {
      cov.block(a_idx, 0, vars_active.at(a)->size(), schmidt_size) = 
          _Cov_cross.block(vars_active.at(a)->id(), anchor_idx, vars_active.at(a)->size(), schmidt_size);
      a_idx += vars_active.at(a)->size();
    }
    return cov;
    }

  void initialize_invertible_schmidt(std::shared_ptr<ov_type::Type> new_variable, const std::vector<std::shared_ptr<ov_type::Type>> &H_order, const Eigen::MatrixXd &H_R, const Eigen::MatrixXd &H_L, const Eigen::MatrixXd &R, const Eigen::VectorXd &res) {
  // First, call the base class initialize_invertible to initialize
  // the new variable in the active state.
  ov_msckf::StateHelper::initialize_invertible(_state,new_variable, H_order, H_R, H_L, R, res);
  
  // Now, we need to update the cross-covariance between the active state and the schmidt state
  // Add a row at the bottom of the cross-covariance for this new variable.
  if (_options.do_schmidt_uwb_anchors) {
    // Get new covariance size after adding new_variable
    int new_size_a = _state->max_covariance_size();
    int old_size_a = new_size_a - new_variable->size();
    _Cov_cross.conservativeResizeLike(Eigen::MatrixXd::Zero(new_size_a, _Cov_cross.cols()));

    // Size of schmidt states
    int n = (int)_anchor_schmidt_idx_map.size() * 3; 
    // For each active variable, find its M = H * P_AS term
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(res.rows(), n);
    int current_it = 0;
    std::vector<int> H_id;
    for (const auto &meas_var : H_order) {
      H_id.push_back(current_it);
      current_it += meas_var->size();
    }

    for (size_t i = 0; i < H_order.size(); i++) {
      std::shared_ptr<ov_type::Type> meas_var = H_order.at(i);
      M.noalias() += H_R.block(0, H_id.at(i), H_R.rows(), meas_var->size()) *
                    _Cov_cross.block(meas_var->id(), 0, meas_var->size(), n);
    }

    // Now, let's add the contribution of this new variable to the cross covariance
    _Cov_cross.block(old_size_a, 0, new_variable->size(), n) = -H_L.inverse() * M;

    // Now set the schmidt state as being initialized
    _has_initialized_schmidt = true;
    }

  
    return;
  }

  
};

} // namespace uvio

#endif // UVIO_STATE_H
