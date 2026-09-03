#ifndef UVIOMANAGEROPTIONS_H
#define UVIOMANAGEROPTIONS_H

#include "core/VioManagerOptions.h"
#include "state/UVioStateOptions.h"
#include "update/UVioUpdaterOptions.h"
#include "utils/uvio_sensor_data.h"

namespace uvio {

/**
 * @brief Struct which stores uvio specific options
 */
struct UVioManagerOptions : ov_msckf::VioManagerOptions {

  /// Number of anchors
  int n_anchors = 0;

  /// Number of fixed anchors
  int n_anchors_to_fix = 0;

  /// Minimum traveled distance to initialize anchors 
  /// Minimum traveled distance to initialize uwb frame transform
  /// [TODO] Should this be replace with its own separate param?
  double min_dist_to_use_uwb = 0.0;

  // Minimum num of uwb measurements
  // for VIO-UWB frame alignment
  int min_uwb_ranges_for_alignment = 100;

  double min_fim_eigenvalue = 0.1;

  double max_pdop = 0.15;

  /// uwb extrinsics (p_IinU).
  Eigen::Vector3d uwb_extrinsics = Eigen::Vector3d::Zero();

  /// offset between initial position of UAV and global frame (p_GinI0).
  Eigen::Vector3d offset_p0 = Eigen::Vector3d::Zero();

  /// Map of Tag ID -> UWB Extrinsics (p_IinU).
  std::map<size_t, Eigen::Vector3d> uwb_extrinsics_map;

  /// Map of Tag ID -> offset between initial position of UAV and global frame (p_GinI0).
  std::map<size_t, Eigen::Vector3d> offset_p0_map;

  /// uwb anchors references (id, p_AinG, const_bias, dist_bias, Cov).
  std::vector<AnchorData> uwb_anchors;

  /// UWB options
  UVioUpdaterOptions uwb_options;

  /// UVIO specific options
  UVioStateOptions uvio_state_options;

  /**
   * @brief This function will load the non-simulation parameters of the system and print.
   * @param parser If not null, this parser will be used to load our parameters
   */
  void print_and_load(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {

    ov_msckf::VioManagerOptions::print_and_load(parser);

    // First, load the behavior/ID list into uvio_state_options
    uvio_state_options.print_and_load(parser);

    PRINT_DEBUG("\n\nUVIO PARAMETERS:\n");

    if (parser != nullptr) {
      /// Parse number of anchors
      // "config_uwb" : param name in estimator_config.yaml
      // Points to "uwb_config.yaml"
      parser->parse_external("config_uwb", "init", "n_fixed_anchors", n_anchors_to_fix);
      parser->parse_external("config_uwb", "init", "n_known_anchors", n_anchors);
      parser->parse_external("config_uwb", "init", "min_dist_to_use_uwb", min_dist_to_use_uwb);


      // For VIO-UWB frame alignment
      parser->parse_external("config_uwb", "init", "min_uwb_ranges_for_alignment", min_uwb_ranges_for_alignment);
      parser->parse_external("config_uwb", "init", "min_fim_eigenvalue", min_fim_eigenvalue);
      parser->parse_external("config_uwb", "init", "max_pdop", max_pdop);

      // Loop through each Tag ID in state options
      for (size_t id: uvio_state_options.tag_ids) {
        std::string tag_block = "tag" + std::to_string(id);
        // 1. Load Extrinsics into the map
        std::vector<double> p_UinI = {0, 0, 0};
        parser->parse_external("config_uwb", tag_block, "p_UinI", p_UinI);
        Eigen::Vector3d r_tz_b; 
        // Load the negatives as UVioUpdaterHelper.cpp uses p_IinU
        r_tz_b << p_UinI.at(0), p_UinI.at(1), p_UinI.at(2);
        uwb_extrinsics_map[id] = r_tz_b;

        // 2. Load Global Offsets
        std::vector<double> p_IinG0 = {0, 0, 0};
        parser->parse_external("config_uwb", tag_block, "p_IinG0", p_IinG0);
        Eigen::Vector3d offset;
        // Load the negatives
        offset << -p_IinG0.at(0), -p_IinG0.at(1), -p_IinG0.at(2);
        offset_p0_map[id] = offset;
        
        PRINT_DEBUG("  - Tag %zu Extrinsics: [%.3f, %.3f, %.3f]\n", id, r_tz_b.x(), r_tz_b.y(), r_tz_b.z());
      }

      /// Calibration parameters
      std::vector<double> p_UinI = {0, 0, 0};
      parser->parse_external("config_uwb", "tag0", "p_UinI", p_UinI);
      uwb_extrinsics << p_UinI.at(0), p_UinI.at(1), p_UinI.at(2);

      /// Initial offset parameter
      std::vector<double> p_IinG0 = {0, 0, 0};
      parser->parse_external("config_uwb", "tag0", "p_IinG0", p_IinG0);
      offset_p0 << -p_IinG0.at(0), -p_IinG0.at(1), -p_IinG0.at(2);

      /// Parse anchors
      for (int i = 0; i < n_anchors; i++) {
        int anch_id;
        bool anch_fix;
        double const_b, dist_b, c, d;
        std::vector<double> p_cov;
        std::vector<double> pos = {0, 0, 0};
        parser->parse_external("uwb_anchors", "anchor" + std::to_string(i), "id", anch_id);
        parser->parse_external("uwb_anchors", "anchor" + std::to_string(i), "fix", anch_fix);
        parser->parse_external("uwb_anchors", "anchor" + std::to_string(i), "p_AinG", pos);
        parser->parse_external("uwb_anchors", "anchor" + std::to_string(i), "const_bias", const_b);
        parser->parse_external("uwb_anchors", "anchor" + std::to_string(i), "dist_bias", dist_b);
        parser->parse_external("uwb_anchors", "anchor" + std::to_string(i), "prior_const_bias_cov", c);
        parser->parse_external("uwb_anchors", "anchor" + std::to_string(i), "prior_dist_bias_cov", d);
        parser->parse_external("uwb_anchors","anchor" + std::to_string(i),"prior_p_AinG_cov", p_cov);

        if (p_cov.size() == 1) {
            p_cov = {p_cov[0], p_cov[0], p_cov[0]};
        }
        else if (p_cov.size() != 3) {
            PRINT_WARNING("prior_p_AinG_cov must have 1 or 3 values\n");
            p_cov = {0.0, 0.0, 0.0};
        }

        AnchorData anchor;
        anchor.id = anch_id;
        anchor.fix = anch_fix;
        anchor.p_AinG << pos.at(0) - p_IinG0.at(0), pos.at(1) - p_IinG0.at(1), pos.at(2) - p_IinG0.at(2);
        anchor.const_bias = const_b;
        anchor.dist_bias = dist_b;
        anchor.cov.diagonal() << p_cov[0], p_cov[1], p_cov[2];

        uwb_anchors.push_back(anchor);
      }
    }

    PRINT_DEBUG("  - n_anchors: %d\n", n_anchors);
    PRINT_DEBUG("  - n_anchors_to_fix: %d\n", n_anchors_to_fix);
    PRINT_DEBUG("  - min_dist_to_init_anchors: %.3f\n", min_dist_to_use_uwb);
    PRINT_DEBUG("  - calib_uwb_imu: [%.3f, %.3f, %.3f]\n", uwb_extrinsics(0), uwb_extrinsics(1), uwb_extrinsics(2));
    PRINT_DEBUG("  - p_GinI0: [%.3f, %.3f, %.3f]\n", offset_p0(0), offset_p0(1), offset_p0(2));

    uvio_state_options.print_and_load(parser);
    uwb_options.print_and_load(parser);

    for (size_t i = 0; i < uwb_anchors.size(); i++) {
      PRINT_DEBUG("  - anchor[%d]: fixed = %s\n", uwb_anchors.at(i).id, uwb_anchors.at(i).fix ? "true" : "false");
      PRINT_DEBUG("  - anchor[%d]: p_AinG = [%.3f, %.3f, %.3f]\n", uwb_anchors.at(i).id, uwb_anchors.at(i).p_AinG.x(),
                  uwb_anchors.at(i).p_AinG.y(), uwb_anchors.at(i).p_AinG.z());
      PRINT_DEBUG("  - anchor[%d]: const_bias = %.3f\n", uwb_anchors.at(i).id, uwb_anchors.at(i).const_bias);
      PRINT_DEBUG("  - anchor[%d]: dist_bias = %.3f\n", uwb_anchors.at(i).id, uwb_anchors.at(i).dist_bias);
      PRINT_DEBUG("  - anchor[%d]: cov.diagonal() = [%.3f, %.3f, %.3f]\n\n", uwb_anchors.at(i).id,
                  uwb_anchors.at(i).cov.diagonal()(0), uwb_anchors.at(i).cov.diagonal()(1), uwb_anchors.at(i).cov.diagonal()(2));
    }
  }
};

} // namespace uvio

#endif // UVIOMANAGEROPTIONS_H
