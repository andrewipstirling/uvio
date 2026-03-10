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
  // TODO: Remove this, now supported by the calib map below
  std::shared_ptr<ov_type::Vec> _calib_UWBtoIMU = std::make_shared<ov_type::Vec>(3);

  /// Map from Tag Index (0, 1, 2...) -> Extrinsic Calibration Variable (p_IinU)
  std::map<size_t, std::shared_ptr<ov_type::Vec>> _calib_UWBtoIMU_map;

  /**
   * @brief Map from <TagID, AnchorID> -> Biases (Constant & Distance)
   * This captures the unique hardware/multipath characteristics of each pair.
   */
  std::map<std::pair<size_t, size_t>, std::shared_ptr<UWBBias>> _uwb_biases_map;

  /// Positions of the uwb anchors (id, UWB_anchor)
  std::unordered_map<size_t, std::shared_ptr<UWBAnchor>> _calib_GLOBALtoANCHORS;

  /// Pointer to the ov_msckf::State object (our state)
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
};

} // namespace uvio

#endif // UVIO_STATE_H
