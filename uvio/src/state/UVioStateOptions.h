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

  /// Bool to determine whether or not to calibrate imu-to-uwb module position
  bool do_calib_uwb_extrinsics = false;

  /// Prior covariances
  double prior_uwb_imu_cov = 0.1;

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

      parser->parse_external("config_uwb", "init", "calib_uwb_extrinsics", do_calib_uwb_extrinsics);
      parser->parse_external("config_uwb", "init", "prior_uwb_imu_cov", prior_uwb_imu_cov);
    }
    PRINT_DEBUG("    - Found %zu UWB tags\n", tag_ids.size());
    PRINT_DEBUG("    - calib_uwb_extrinsics: %s\n", do_calib_uwb_extrinsics ? "true" : "false");
    PRINT_DEBUG("    - prior_uwb_imu_cov: %.4f\n", prior_uwb_imu_cov);
  }
};

} // namespace uvio

#endif // UVIO_STATE_OPTIONS_H
