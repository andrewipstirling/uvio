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

#ifndef UVIO_STATE_OPTIONS_H
#define UVIO_STATE_OPTIONS_H

#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

namespace uvio {

/**
 * @brief Struct which stores uvio specific options
 */
struct UVioStateOptions {

  // List of Tag IDs (e.g., [10, 11]) attached to body
  std::vector<size_t> tag_ids;

  // List of Anchor IDs that are known
  std::vector<size_t> anchor_ids;

  /// Bool to determine whether or not to calibrate imu-to-uwb module position
  bool do_calib_uwb_extrinsics = false;

  // Bool to determine whether or not to update UWB ranging biases
  bool do_calib_uwb_biases = true; 

  // Bool to determine whether or not perform alignment between UWB anchor and VIO frames in a localization setup
  bool do_calib_uwb_frame_transfrom = false;

  // Options related to marginalizing out the frame transform if we
  // don't want to keep estimating it
  bool do_marginalize_frame_transform = false;
  double min_yaw_covar_fix_frame_align = 0.001;
  double min_pos_covar_fix_frame_align = 0.01;
  double marg_frame_align_cov_inflation = 1.0;


  // Double how much to inflate the initial covariance of the uwb frame transform state
  double init_inflation_uwb_frame_align_pos = 2.0;
  double init_inflation_uwb_frame_align_ori = 2.0;


  // Bool determines if to use DSTWR w/ antenna delays & pose dep biases
  bool do_dstwr_uwb = true;

  /// Prior covariances
  double prior_uwb_imu_cov = 0.1;

  // Create schmidt state for the UWB anchor map
  bool do_schmidt_uwb_anchors = false;

  // Number of schmidt anchors
  int max_schmidt_anchors = 0;




  /// Nice print function of what parameters we have loaded
  void print_and_load(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {

    if (parser != nullptr) {

      // Read number of tags
      std::vector<int> tag_ids_int;
      parser->parse_external("config_uwb", "init", "tag_ids", tag_ids_int);
      // [Andrew] Cast the id into size_t
      tag_ids.clear();
      for (int id : tag_ids_int){
        tag_ids.push_back(static_cast<size_t>(id));
      }

      // Read number of anchors
      int num_ancs;
      parser->parse_external("config_uwb", "init", "n_known_anchors", num_ancs);
      // [Andrew] Cast the id into size_t
      anchor_ids.clear();
      for (int i = 0; i < num_ancs; i++){
        int anc_id;
        std::string anc_name = "anchor" + std::to_string(i);
        parser->parse_external("uwb_anchors", anc_name, "id", anc_id);
        anchor_ids.push_back(static_cast<size_t>(anc_id));

      }
      // UWB options
      parser->parse_external("config_uwb", "init", "calib_uwb_extrinsics", do_calib_uwb_extrinsics);
      parser->parse_external("config_uwb", "init", "calib_uwb_biases", do_calib_uwb_biases);
      parser->parse_external("config_uwb", "init", "prior_uwb_imu_cov", prior_uwb_imu_cov);
      parser->parse_external("config_uwb", "init", "do_dstwr_uwb", do_dstwr_uwb);
      
      // Frame alignment variables
      parser->parse_external("config_uwb", "init", "do_uwb_frame_align", do_calib_uwb_frame_transfrom);
      parser->parse_external("config_uwb", "init", "init_inflation_uwb_frame_align_pos", init_inflation_uwb_frame_align_pos);
      parser->parse_external("config_uwb", "init", "init_inflation_uwb_frame_align_ori", init_inflation_uwb_frame_align_ori);

      // Marginalization variables
      parser->parse_external("config_uwb", "init", "do_marginalize", do_marginalize_frame_transform);
      parser->parse_external("config_uwb", "init", "min_yaw_covar_fix_frame_align", min_yaw_covar_fix_frame_align);
      parser->parse_external("config_uwb", "init", "min_pos_covar_fix_frame_align", min_pos_covar_fix_frame_align);
      parser->parse_external("config_uwb", "init", "marg_frame_align_cov_inflation", marg_frame_align_cov_inflation);
      
      // Schmidt state options
      parser->parse_external("config_uwb", "init", "do_schmidt_uwb_anchors", do_schmidt_uwb_anchors);
    }
    PRINT_DEBUG("    - Found %zu UWB tags\n", tag_ids.size());
    PRINT_DEBUG("    - calib_uwb_extrinsics: %s\n", do_calib_uwb_extrinsics ? "true" : "false");
    PRINT_DEBUG("    - prior_uwb_imu_cov: %.4f\n", prior_uwb_imu_cov);
  }
};

} // namespace uvio

#endif // UVIO_STATE_OPTIONS_H
