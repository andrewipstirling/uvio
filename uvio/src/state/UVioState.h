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
struct UVioState{

  UVioState(UVioStateOptions &options, std::shared_ptr<ov_msckf::State> state) : _options(options), _state(state) {
    // Initialize frame transform
    _calib_VIOtoUWB_frame_alignment = std::make_shared<ov_type::PoseJPL>();

    // Pre-Allocate the schmidt and cross covariance matrices
    if (_options.do_schmidt_uwb_anchors) {
      // P^{SS} : (3 * num_anchors, 3 * num_anchors)
      int size_schmidt = 3 * _options.max_scmhidt_anchors;
      _Cov_schmidt = Eigen::MatrixXd(size_schmidt, size_schmidt);
      
      // P^{AS} : (size of active state, size of schmidt state)
      // Automatically set to zero due to no cross covariance
      _Cov_cross = Eigen::MatrixXd::Zero(_state->max_covariance_size(), size_schmidt);
    }


    // Initialize the uwb extrinsics map
    for (size_t tag_id : _options.tag_ids){
      // Create new 3x1 vector state for each tag
      _calib_UWBtoIMU_map[tag_id] = std::make_shared<ov_type::Vec>(3);

      PRINT_DEBUG(GREEN "[UVioState] Initialized state variable for Tag ID: %zu\n" RESET, tag_id);

      for (size_t anc_id : _options.anchor_ids){
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
    if (_uwb_biases_map.find(tag_anchor_key) == _uwb_biases_map.end()){
      _uwb_biases_map[tag_anchor_key] = std::make_shared<UWBBias>(0.0, 0.0);
      PRINT_ERROR(RED "[UVIOState] Missing bias state for Tag %zu <-> Anchor %zu" RESET, tag_id, anchor_id);
      return nullptr;
    }
    
    return _uwb_biases_map.at(tag_anchor_key);

  }

  void register_schmidt(size_t anc_id, Eigen::Matrix3d anc_cov){
    // Get size of map, informs the index to insert to
    size_t cur_size_map = _anchor_schmidt_idx_map.size();
    int idx_to_insert = (int) 3 * cur_size_map;

    // Insert values into the schmidt covariance
    _Cov_schmidt.block<3, 3>(idx_to_insert, idx_to_insert) = anc_cov;
    
    // Set cross covariance to zero initially
    // Use dynamic sized block function
    int size_active_cov = _state->max_covariance_size();
    _Cov_cross.block(0, idx_to_insert, size_active_cov, 3) = Eigen::MatrixXd::Zero(size_active_cov, 3);

    // Insert index to map
    _anchor_schmidt_idx_map[anc_id] = idx_to_insert;

  }

};

} // namespace uvio

#endif // UVIO_STATE_H
