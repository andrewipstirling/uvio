/*
 * UVIO: Ultra Wide-Band aided Visual-Inertial Odometry
 * Copyright (C) 2020-2022 Alessandro Fornasier
 * Copyright (C) 2018-2022 UVIO Contributors
 *
 * Control of Networked Systems, University of Klagenfurt, Austria (alessandro.fornasier@aau.at)
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

#include <core/UVioManager.h>

using namespace uvio;

UVioManager::UVioManager(UVioManagerOptions &params_) : ov_msckf::VioManager::VioManager(params_), params(params_) {

  /// Initialize uvio state and downcast propagator
  state = std::make_shared<UVioState>(params.uvio_state_options, this->get_state());
  propagator = std::static_pointer_cast<UVioPropagator>(this->get_propagator());

  // Initialize p_UinI (calibration uwb-imu)
  for (size_t id : params.uvio_state_options.tag_ids) {

    // Get the variable pointer we just created in UVioState
    auto tag_var = state->_calib_UWBtoIMU_map[id];

    if (params.uvio_state_options.do_calib_uwb_extrinsics) {
      std::vector<std::shared_ptr<ov_type::Type>> H_order;
      H_order.push_back(state->_state->_imu->q()); // Relate to IMU orientation

      Eigen::Matrix3d H_R = Eigen::Matrix3d::Zero();
      Eigen::Matrix3d H_L = Eigen::Matrix3d::Identity();
      Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * params.uvio_state_options.prior_uwb_imu_cov;
      Eigen::Vector3d res = Eigen::Vector3d::Zero();

      // Formally add to EKF covariance matrix
      ov_msckf::StateHelper::initialize_invertible(state->_state, tag_var, H_order, H_R, H_L, R, res);
    }
    // Set initial physical values from map
    state->_calib_UWBtoIMU_map[id]->set_value(params.uwb_extrinsics_map[id]);
    state->_calib_UWBtoIMU_map[id]->set_value(params.uwb_extrinsics_map[id]);
  }

  // Initialize anchors (if provided in config file)
  try_to_initialize_uwb_anchors(params.uwb_anchors);

  // Make the updater!
  updaterUWB = std::make_unique<UpdaterUWB>(params.uwb_options);

  // Make initializer for localization
  // TODO: Add cauchy value to params
  uwb_alignment_initializer = std::make_unique<UVioInitializer>(0.5);
  // Set the uwb frame alignment initialization
  UWBInitStage uwb_init_stage = DONE;
  if (params.uvio_state_options.do_calib_uwb_frame_transfrom) {
    uwb_init_stage = INITIAL;
    initialization_start_time = -1.0;
  }
  Eigen::Matrix3d C_av_est = Eigen::Matrix3d::Identity();
  Eigen::Vector3d r_av_est = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 7, 1> x_alignment;
  x_alignment << ov_core::rot_2_quat(C_av_est), r_av_est;
  state->_calib_VIOtoUWB_frame_alignment->set_value(x_alignment);
}

void UVioManager::feed_measurement_uwb(const UwbData &message) {
  // PRINT_DEBUG(YELLOW "Processing UWB message\n" RESET);
  // Basic check for VIO startup and anchors have been setup
  if (!is_initialized_vio || !are_initialized_anchors || (message.timestamp < startup_time + 2.0) || (distance < 0.1))
    return;

  // Return if the uwb measurement is out of order otherwise feed our bar measuremnts
  if (state->_state->_timestamp >= message.timestamp) {
    PRINT_DEBUG(YELLOW "UWB measurements received out of order (prop dt = %3f)\n" RESET, (message.timestamp - state->_state->_timestamp));
    return;
  }
  // If in localization setup, initialize the frame transform
  if ((uwb_init_stage == INITIAL || uwb_init_stage == REFINEMENT) && state->_options.do_calib_uwb_frame_transfrom && is_initialized_vio) {
    // Get start time for debugging
    if (initialization_start_time < 0) {
      initialization_start_time = ros::Time::now().toSec();
    }

    // Store in separate buffer for batch alignment problem
    std::vector<AlignmentConstraint> alignment_measurements;

    // build constraints
    for (const auto &r : message.uwb_ranges) {
      AlignmentConstraint c;
      c.range = r.range;
      c.std_range = r.std;
      c.p_AinG = state->_calib_GLOBALtoANCHORS.at(r.anchor_id)->p_AinG()->value();

      // Transform Tag position from body frame to VIO frame
      Eigen::Vector3d p_zw_v = state->_state->_imu->pos();
      c.p_zw_v = p_zw_v;
      // Internal VIO state tracks C_bv
      Eigen::Matrix3d C_vb = state->_state->_imu->Rot().transpose();
      c.C_bv = C_vb.transpose();
      // Position of Tag relative to IMU
      Eigen::Vector3d p_tz_b = state->_calib_UWBtoIMU_map.at(r.tag_id)->value();
      c.p_TinV = C_vb * p_tz_b + p_zw_v;
      c.p_tz_b = p_tz_b;

      // Add Pose covariance
      std::vector<std::shared_ptr<ov_type::Type>> vio_vars;
      vio_vars.push_back(state->_state->_imu);
      Eigen::MatrixXd P_full = ov_msckf::StateHelper::get_marginal_covariance(state->_state, vio_vars);
      c.P_vio = P_full.block<6, 6>(0, 0);
      c.P_anc = Eigen::Matrix3d::Zero();
      if (state->_options.do_schmidt_uwb_anchors){
        int schmidt_anc_id = state->_anchor_schmidt_idx_map.at(r.anchor_id);
        c.P_anc = state->_Cov_schmidt.block<3,3>(schmidt_anc_id, schmidt_anc_id);
      }
      bool accept = false;
      // Add them to problem depending on geometry / initialization stage
      accept = uwb_alignment_initializer->accept_measurement(c);

      if (accept && c.range > params.min_dist_to_use_uwb && c.range < 10.0)
        alignment_measurements.push_back(c);
    }
    if (!alignment_measurements.empty())
      uwb_alignment_initializer->add_measurements(alignment_measurements);

    // Triggers initialization
    if (uwb_init_stage == INITIAL &&
        uwb_alignment_initializer->can_initialize(distance, params.min_dist_to_use_uwb, params.min_uwb_ranges_for_alignment,
                                                  params.max_pdop )) {
      // VIO to UWB frame rotation
      Eigen::Matrix3d C_av;
      // VIO (wv) relative to UWB (wa) frame trans
      Eigen::Vector3d r_wvwa_a;
      // Initialization covariance
      Eigen::Matrix4d init_covar;
      bool use_squared_cost = true;
      bool use_initial_guess = false;
      if (uwb_alignment_initializer->solve(C_av, r_wvwa_a, init_covar, use_initial_guess, use_squared_cost)) {
        PRINT_INFO(GREEN "[UVIO] Initial alignment success\n" RESET);
        // Set initial
        C_av_est = C_av;
        r_av_est = r_wvwa_a;

        uwb_init_stage = REFINEMENT;
        // uwb_alignment_initializer->clear();
      }
    } else if (uwb_init_stage == REFINEMENT &&
               uwb_alignment_initializer->can_refine(C_av_est, r_av_est, distance, params.min_dist_to_use_uwb,
                                                     params.min_uwb_ranges_for_alignment, params.max_pdop, params.min_fim_eigenvalue)) 
      {
      
      // Eigen::Matrix3d C_ref;
      // Eigen::Vector3d r_ref;
      // Eigen::Matrix4d init_covar;
      // RansacConfig ransac_config;
      // if (uwb_alignment_initializer->solve_robust(C_ref, r_ref, init_covar, ransac_config)){
      //   PRINT_INFO(GREEN "[UVIO] Success! VIO-UWB frames aligned after %d UWB ranges and %3fm travelled.\n" RESET,
      //              uwb_alignment_initializer->data_count(), distance);
      //   uwb_init_stage = CAN_INJECT;

      //   C_av_est = C_ref;
      //   r_av_est = r_ref;
      //   cov_av_est = init_covar;
      // }
      Eigen::Matrix3d C_ref = C_av_est;
      Eigen::Vector3d r_ref = r_av_est;
      // Initialization covariance
      Eigen::Matrix4d init_covar;
      bool use_squared_cost = false;
      bool use_initial_guess = true;
      if (uwb_alignment_initializer->solve(C_ref, r_ref, init_covar, use_initial_guess, use_squared_cost)) {
        PRINT_INFO(GREEN "[UVIO] Success! VIO-UWB frames aligned after %d UWB ranges and %3fm travelled.\n" RESET,
                   uwb_alignment_initializer->data_count(), distance);
        uwb_init_stage = CAN_INJECT;

        C_av_est = C_ref;
        r_av_est = r_ref;
        cov_av_est = init_covar;
      }
    } else
      return;
  }

  else {
    // We have initial estimate of alignment vars so can feed state
    past_measurements.insert({message.timestamp, std::make_shared<UwbData>(message)});
  }
}

const UVioManagerOptions &UVioManager::get_uvio_params() const { return params; }

void UVioManager::try_to_initialize_uwb_anchors(const std::vector<AnchorData> &anchors) {

  // Check if argument is empty
  if (anchors.empty()) {
    PRINT_INFO("UWB anchors not initialized (anchors not provided)\n");
    return;
  }

  // Check if anchors are not initialized yet and initialize them
  if (!are_initialized_anchors) {
    params.uwb_anchors = anchors;
    params.n_anchors = anchors.size();
    int n_fixed_anchors = std::count_if(anchors.begin(), anchors.end(), [](const AnchorData &a) { return a.fix; });
    params.n_anchors_to_fix = std::max(0, params.n_anchors_to_fix - n_fixed_anchors);
    initialize_uwb_anchors();
    return;
  }

  // Check if we have new anchors to initialize
  PRINT_INFO("UWB anchors already initialized. Checking for new anchors to initialize\n");
  for (const auto &it : anchors) {
    // If anchor is not in the state, initialize it
    if (state->_calib_GLOBALtoANCHORS.find(it.id) == state->_calib_GLOBALtoANCHORS.end()) {
      PRINT_INFO(GREEN "New anchor found. Initializing anchor[%d]\n" RESET, it.id);
      params.uwb_anchors.push_back(it);
      params.n_anchors++;
      if (it.fix) {
        params.n_anchors_to_fix = std::max(0, params.n_anchors_to_fix - 1);
      }
      initialize_new_uwb_anchor(it);
    }
  }

  return;
}

void UVioManager::track_image_and_update(const ov_core::CameraData &message_const) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Assert we have valid measurement data and ids
  assert(!message_const.sensor_ids.empty());
  assert(message_const.sensor_ids.size() == message_const.images.size());
  for (size_t i = 0; i < message_const.sensor_ids.size() - 1; i++) {
    assert(message_const.sensor_ids.at(i) != message_const.sensor_ids.at(i + 1));
  }

  // Downsample if we are downsampling
  ov_core::CameraData message = message_const;
  for (size_t i = 0; i < message.sensor_ids.size() && params.downsample_cameras; i++) {
    cv::Mat img = message.images.at(i);
    cv::Mat mask = message.masks.at(i);
    cv::Mat img_temp, mask_temp;
    cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
    message.images.at(i) = img_temp;
    cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
    message.masks.at(i) = mask_temp;
  }

  // Perform our feature tracking!
  trackFEATS->feed_new_camera(message);

  // If the aruco tracker is available, the also pass to it
  // NOTE: binocular tracking for aruco doesn't make sense as we by default have the ids
  // NOTE: thus we just call the stereo tracking if we are doing binocular!
  if (is_initialized_vio && trackARUCO != nullptr) {
    trackARUCO->feed_new_camera(message);
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_state->_timestamp != message.timestamp) {
      did_zupt_update = updaterZUPT->try_update(state->_state, message.timestamp);
    }
    if (did_zupt_update) {
      assert(state->_state->_timestamp == message.timestamp);
      propagator->clean_old_imu_measurements(message.timestamp + state->_state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(message.timestamp + state->_state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator->invalidate_cache();
      return;
    }
  }

  // If we do not have VIO initialization, then try to initialize
  // TODO: Or if we are trying to reset the system, then do that here!
  if (!is_initialized_vio) {
    is_initialized_vio = try_to_initialize(message);
    if (!is_initialized_vio) {
      double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
      PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
      return;
    }
  }
  // [Andrew] If doing localization and frame alignment
  // Initialization MLE problem is done, inject alignment variables to state
  if (uwb_init_stage == CAN_INJECT and state->_options.do_calib_uwb_frame_transfrom) {
    PRINT_INFO(GREEN "[UVIO] Injecting UWB alignment states into filter\n" RESET);
    // Add the alignment values to state
    // Eigen::Matrix<double, 7, 1> x_alignment;
    // x_alignment << ov_core::rot_2_quat(C_av_est), r_av_est;
    // state->_calib_VIOtoUWB_frame_alignment->set_value(x_alignment);
    state->_calib_VIOtoUWB_frame_alignment->p()->set_value(r_av_est);
    state->_calib_VIOtoUWB_frame_alignment->p()->set_fej(r_av_est);
    state->_calib_VIOtoUWB_frame_alignment->q()->set_value(ov_core::rot_2_quat(C_av_est));
    state->_calib_VIOtoUWB_frame_alignment->q()->set_fej(ov_core::rot_2_quat(C_av_est));

    // Add to msckf
    std::vector<std::shared_ptr<ov_type::Type>> H_order;
    H_order.push_back(state->_state->_imu->pose()); // Relate to IMU pose
    Eigen::MatrixXd H_R = Eigen::MatrixXd::Zero(6, 6);
    Eigen::MatrixXd H_L = Eigen::MatrixXd::Identity(6, 6);
    Eigen::MatrixXd R_init = Eigen::MatrixXd::Identity(6, 6);
    Eigen::VectorXd res = Eigen::VectorXd::Zero(6);

    // Initialize the state
    if (state->_options.do_schmidt_uwb_anchors){
      // Can now initialize the schmidt states as well now that
      // we are processing UWB measurements
      // PRINT_DEBUG(MAGENTA "Starting the initialization of the Schmidt anchors\n" RESET)
      state->initialize_invertible_schmidt(state->_calib_VIOtoUWB_frame_alignment, H_order, H_R, H_L, R_init, res);
      assert(state->_has_initialized_schmidt);
    }
    else{
      ov_msckf::StateHelper::initialize_invertible(state->_state, state->_calib_VIOtoUWB_frame_alignment, H_order, H_R, H_L, R_init, res);
    }
    
    // ov_msckf::StateHelper::initialize(state->_state, state->_calib_VIOtoUWB_frame_alignment, H_order, H_R, H_L, R_init, res, 1.0);
    PRINT_INFO(GREEN "[UVIO] UWB alignment initialized and added to state.\n" RESET);
    // Initialize the covariance
    std::vector<std::shared_ptr<ov_type::Type>> H_order_cov;
    H_order.clear();
    H_order.push_back(state->_calib_VIOtoUWB_frame_alignment);

    // Find the maximum value in your batch covariance diagonal to stay safe
    double max_var = cov_av_est.diagonal().maxCoeff();
    double max_var_pos = cov_av_est.block<3,3>(1,1).diagonal().maxCoeff();
    // Also check against your manual Pitch/Roll variance
    // max_var = std::max(max_var, std::pow(0.1 * M_PI, 2));

    // Set Pitch and Roll with yaw covariance
    Eigen::Matrix<double, 6, 6> R_final = Eigen::Matrix<double, 6, 6>::Identity() * cov_av_est(0, 0);
    // Yaw & Position
    R_final.block<4, 4>(2, 2) = cov_av_est;
    // Inflate orientation covariance
    R_final(3, 3) = max_var_pos;
    R_final(4, 4) = max_var_pos;
    R_final(5, 5) = max_var_pos;
    R_final.block<3, 3>(0,0) *= params.uvio_state_options.init_inflation_uwb_frame_align_ori;
    // Inflate position covariance
    R_final.block<3,3>(3, 3) *= params.uvio_state_options.init_inflation_uwb_frame_align_pos; 
    // PRINT_DEBUG(MAGENTA "Rows of active covariance: %d. Rows of cross covariance rows: %d\n" RESET, state->_state->max_covariance_size(), state->_Cov_cross.rows());

    ov_msckf::StateHelper::set_initial_covariance(state->_state, R_final, H_order);

    PRINT_INFO(GREEN "[UVIO] UWB alignment initialized covariance = [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f] \n" RESET, R_final(0, 0),
               R_final(1, 1), R_final(2, 2), R_final(3, 3), R_final(4, 4), R_final(5, 5));

    double time_now = ros::Time::now().toSec();
    PRINT_INFO(GREEN "[UVIO] UWB alignment initialized after %.4f seconds\n" RESET, time_now - initialization_start_time);

    uwb_alignment_initializer->clear();
    uwb_init_stage = DONE;
    is_initialized_uwb_frame_transform = true;
  }
  // Retrive other measurement from last update up to vision measurement time
  if (!past_measurements.empty()) {
    for (auto it = past_measurements.begin(); it != past_measurements.lower_bound(message.timestamp); it++) {
      if (it->first < message.timestamp && it->first > state->_state->_timestamp) {
        do_uwb_propagate_update(it->second);
      }
    }

    // Clear all the past measurement and throw away also measurement which have the same timestamp of the vision measurment even if not
    // processed
    past_measurements.erase(past_measurements.begin(), past_measurements.upper_bound(message.timestamp));
  }

  // Print our current uwb state
  if (params.uvio_state_options.do_calib_uwb_extrinsics) {
    for (const auto &id : params.uvio_state_options.tag_ids) {
      const auto tag_val = state->get_calib_uwb(id)->value();
      PRINT_INFO(YELLOW "calib_UWBtoIMU = [%.3f,%.3f,%.3f]\n" RESET, tag_val(0), tag_val(1), tag_val(2));
    }
  }
  for (const auto &it : state->_calib_GLOBALtoANCHORS) {
    if (!it.second->fixed()) {
      PRINT_INFO(YELLOW "anchor[%d]: p_AinG = [%.3f, %.3f, %.3f]\n" RESET, it.first, it.second->p_AinG()->value()(0),
                 it.second->p_AinG()->value()(1), it.second->p_AinG()->value()(2));
    }
  }

  // Call on our propagate and update function
  do_feature_propagate_update(message);
}

void UVioManager::initialize_uwb_anchors() {

  // Check if anchors are initialized
  if (are_initialized_anchors) {
    PRINT_ERROR("UWB anchors already initialized\n");
    return;
  }

  // Check if we have any anchors
  if (params.uwb_anchors.empty()) {
    PRINT_ERROR("UWB anchors not specified\n");
    return;
  }

  for (const auto &it : params.uwb_anchors) {
    std::shared_ptr<UWBAnchor> anchor = std::make_shared<UWBAnchor>(it);
    state->_calib_GLOBALtoANCHORS.insert({it.id, anchor});
    PRINT_INFO("Anchor[%d] initialized\n", it.id);

    // [Andrew] Separating the uwb bias state from the anchor state
    for (size_t tag_id : params.uvio_state_options.tag_ids) {
      std::pair<size_t, size_t> tag_anchor_id{tag_id, it.id};
      std::shared_ptr<UWBBias> bias = std::make_shared<UWBBias>(it.const_bias, it.dist_bias);
      state->_uwb_biases_map[tag_anchor_id] = bias;
      // [Andrew] If we want to estimate this, then add to state vector
      if (state->_options.do_calib_uwb_biases) {
        // [Andrew] Add bias states to EKF
        // There's no _variables->push() option need to use
        // ov_msckf::StateHelper::initialize_invertible
        // Kind of pretend to insert

        std::vector<std::shared_ptr<ov_type::Type>> H_order;
        // Need to fake it...
        H_order.push_back(state->_state->_imu->q());

        Eigen::MatrixXd H_R = Eigen::MatrixXd::Zero(2, 3);
        Eigen::MatrixXd H_L = Eigen::MatrixXd::Identity(2, 2);
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2);
        Eigen::VectorXd res = Eigen::VectorXd::Zero(2);
        ov_msckf::StateHelper::initialize_invertible(state->_state, state->_uwb_biases_map.at(tag_anchor_id), H_order, H_R, H_L, R, res);

        H_order.clear();
        H_order.push_back(state->_uwb_biases_map.at(tag_anchor_id));
        Eigen::Matrix2d bias_cov = it.cov.bottomRightCorner(2, 2);
        ov_msckf::StateHelper::set_initial_covariance(state->_state, bias_cov, H_order);
      }
    }

    // Initialize state variable if option enabled and anchor not fixed
    if (!it.fix) {
      if (state->_options.do_schmidt_uwb_anchors){
        PRINT_ERROR(RED "Anchor %d not fixed but schmidt flag set to True, setting schmidt flag to false\n" RESET, it.id);
        state->_options.do_schmidt_uwb_anchors = false;
      }

      // Initialize state variables
      std::vector<std::shared_ptr<ov_type::Type>> H_order;

      // Need to fake it...
      H_order.push_back(state->_state->_imu->q());

      Eigen::MatrixXd H_R = Eigen::MatrixXd::Zero(3, 3);
      Eigen::MatrixXd H_L = Eigen::MatrixXd::Identity(3, 3);
      Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3);
      Eigen::VectorXd res = Eigen::VectorXd::Zero(3);
      ov_msckf::StateHelper::initialize_invertible(state->_state, state->_calib_GLOBALtoANCHORS.at(it.id), H_order, H_R, H_L, R, res);

      H_order.clear();
      H_order.push_back(state->_calib_GLOBALtoANCHORS.at(it.id));

      ov_msckf::StateHelper::set_initial_covariance(state->_state, it.cov.topLeftCorner(3, 3), H_order);

      PRINT_INFO("Anchor[%d] added to state\n", it.id);
    }

    if (it.fix && params.uvio_state_options.do_schmidt_uwb_anchors) {
      // Register Schmidt states, but we won't set them as initialized yet
      state->register_schmidt(it.id, it.cov);
    }

    // Print anchor info
    PRINT_INFO("anchor[%d]: p_AinG = [%.3f, %.3f, %.3f] | const_bias = %.4f | dist_bias = %.4f\n", it.id, it.p_AinG.x(), it.p_AinG.y(),
               it.p_AinG.z(), it.const_bias, it.dist_bias);
    PRINT_INFO("cov.diagonal() = [%.3f, %.3f, %.3f]\n\n",
    it.cov.diagonal()(0), it.cov.diagonal()(1), it.cov.diagonal()(2));
    // std::cout << "cov = \n" << it.cov << "\n" << std::endl;
  }
  are_initialized_anchors = true;
  PRINT_DEBUG(MAGENTA "UWB anchors correctly initialized\n" RESET);
}

void UVioManager::initialize_new_uwb_anchor(const AnchorData &anchor) {

  // Check if anchors are initialized
  if (!are_initialized_anchors) {
    PRINT_ERROR("UWB anchors not initialized\n");
    return;
  }

  // Check if anchor is already initialized
  if (state->_calib_GLOBALtoANCHORS.find(anchor.id) != state->_calib_GLOBALtoANCHORS.end()) {
    PRINT_ERROR("Anchor[%d] already initialized\n", anchor.id);
    return;
  }

  // Initialize anchor
  std::shared_ptr<UWBAnchor> uwb_anchor = std::make_shared<UWBAnchor>(anchor);
  state->_calib_GLOBALtoANCHORS.insert({anchor.id, uwb_anchor});

  PRINT_INFO("Anchor[%d] initialized\n", anchor.id);

  // Initialize state variable if option enabled and anchor not fixed
  if (!anchor.fix) {

    // Initialize state variables
    std::vector<std::shared_ptr<ov_type::Type>> H_order;

    // Need to fake it...
    H_order.push_back(state->_state->_imu->q());

    Eigen::MatrixXd H_R = Eigen::MatrixXd::Zero(5, 3);
    Eigen::MatrixXd H_L = Eigen::MatrixXd::Identity(5, 5);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(5, 5);
    Eigen::VectorXd res = Eigen::VectorXd::Zero(5);
    ov_msckf::StateHelper::initialize_invertible(state->_state, state->_calib_GLOBALtoANCHORS.at(anchor.id), H_order, H_R, H_L, R, res);

    H_order.clear();
    H_order.push_back(state->_calib_GLOBALtoANCHORS.at(anchor.id));

    ov_msckf::StateHelper::set_initial_covariance(state->_state, anchor.cov, H_order);

    PRINT_INFO("Anchor[%d] added to state\n", anchor.id);
  }

  // Print anchor info
  PRINT_INFO("anchor[%d]: p_AinG = [%.3f, %.3f, %.3f] | const_bias = %.4f | dist_bias = %.4f\n", anchor.id, anchor.p_AinG.x(),
             anchor.p_AinG.y(), anchor.p_AinG.z(), anchor.const_bias, anchor.dist_bias);
  PRINT_INFO("cov.diagonal() = [%.3f, %.3f, %.3f]\n\n",
    anchor.cov.diagonal()(0), anchor.cov.diagonal()(1), anchor.cov.diagonal()(2));
  // std::cout << "cov = \n" << anchor.cov << "\n" << std::endl;
}

void UVioManager::do_uwb_propagate_update(const std::shared_ptr<UwbData> &message) {

  // Check if we have at least one measurement from an initialized anchor
  bool valid = false;
  for (const auto &it : message->uwb_ranges) {
    if (state->_calib_GLOBALtoANCHORS.find(it.anchor_id) != state->_calib_GLOBALtoANCHORS.end()) {
      valid = true;
      break;
    }
  }

  // Return if no matching was found
  if (!valid) {
    return;
  }

  // Propagate the state forward to the current update time
  propagator->propagate(state, message->timestamp);

  // Return if we where unable to propagate
  if (state->_state->_timestamp != message->timestamp) {
    printf(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
    printf(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message->timestamp - state->_state->_timestamp);
    return;
  }

  // Iterate through single ranges and update
  /// Giulio: this is better because it allows to filter single measurements
  /// with the chi2 test instead of discarding all of them if just one is bad
  for (const auto &it : message->uwb_ranges) {
    // Check if measurement is from initialized anchor
    if (state->_calib_GLOBALtoANCHORS.find(it.anchor_id) != state->_calib_GLOBALtoANCHORS.end()) {
      // EKF Update with single UWB measurement
      updaterUWB->update_single(state, message->timestamp, it.tag_id, it.anchor_id, it.range, it.std);
    }
  }
  // // Check if alignment covariance has converged, if so transform and marginalize
  // check_and_marginalize_alignment();
}

void UVioManager::do_feature_propagate_update(const ov_core::CameraData &message) {

  //===================================================================================
  // State propagation, and clone augmentation
  //===================================================================================

  // Return if the camera measurement is out of order
  if (state->_state->_timestamp > message.timestamp) {
    PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET,
                  (message.timestamp - state->_state->_timestamp));
    return;
  }

  // Propagate the state forward to the current update time
  // Also augment it with a new clone!
  // NOTE: if the state is already at the given time (can happen in sim)
  // NOTE: then no need to prop since we already are at the desired timestep
  if (state->_state->_timestamp != message.timestamp) {
    // PRINT_DEBUG(MAGENTA "Starting propagate_and_clone\n" RESET);
    propagator->propagate_and_clone(state, message.timestamp);
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // If we have not reached max clones, we should just return...
  // This isn't super ideal, but it keeps the logic after this easier...
  // We can start processing things when we have at least 5 clones since we can start triangulating things...
  if ((int)state->_state->_clones_IMU.size() < std::min(state->_state->_options.max_clone_size, 5)) {
    PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state->_state->_clones_IMU.size(),
                std::min(state->_state->_options.max_clone_size, 5));
    return;
  }

  // Return if we where unable to propagate
  if (state->_state->_timestamp != message.timestamp) {
    PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
    PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message.timestamp - state->_state->_timestamp);
    return;
  }
  has_moved_since_zupt = true;
  // =======================================
  // Marginalize out UWB transform if ready here
  // Check if alignment covariance has converged
  // =======================================
  if (uwb_init_stage == DONE && state->_options.do_calib_uwb_frame_transfrom && state->_options.do_marginalize_frame_transform){
    check_and_marginalize_alignment();
  }
  

  //===================================================================================
  // MSCKF features and KLT tracks that are SLAM features
  //===================================================================================

  // Now, lets get all features that should be used for an update that are lost in the newest frame
  // We explicitly request features that have not been deleted (used) in another update step
  std::vector<std::shared_ptr<ov_core::Feature>> feats_lost, feats_marg, feats_slam;
  feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(state->_state->_timestamp, false, true);

  // Don't need to get the oldest features until we reach our max number of clones
  if ((int)state->_state->_clones_IMU.size() > state->_state->_options.max_clone_size || (int)state->_state->_clones_IMU.size() > 5) {
    feats_marg = trackFEATS->get_feature_database()->features_containing(state->_state->margtimestep(), false, true);
    if (trackARUCO != nullptr && message.timestamp - startup_time >= params.dt_slam_delay) {
      feats_slam = trackARUCO->get_feature_database()->features_containing(state->_state->margtimestep(), false, true);
    }
  }

  // Remove any lost features that were from other image streams
  // E.g: if we are cam1 and cam0 has not processed yet, we don't want to try to use those in the update yet
  // E.g: thus we wait until cam0 process its newest image to remove features which were seen from that camera
  auto it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    bool found_current_message_camid = false;
    for (const auto &camuvpair : (*it1)->uvs) {
      if (std::find(message.sensor_ids.begin(), message.sensor_ids.end(), camuvpair.first) != message.sensor_ids.end()) {
        found_current_message_camid = true;
        break;
      }
    }
    if (found_current_message_camid) {
      it1++;
    } else {
      it1 = feats_lost.erase(it1);
    }
  }

  // We also need to make sure that the max tracks does not contain any lost features
  // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
  it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) != feats_marg.end()) {
      // PRINT_WARNING(YELLOW "FOUND FEATURE THAT WAS IN BOTH feats_lost and feats_marg!!!!!!\n" RESET);
      it1 = feats_lost.erase(it1);
    } else {
      it1++;
    }
  }

  // Find tracks that have reached max length, these can be made into SLAM features
  std::vector<std::shared_ptr<ov_core::Feature>> feats_maxtracks;
  auto it2 = feats_marg.begin();
  while (it2 != feats_marg.end()) {
    // See if any of our camera's reached max track
    bool reached_max = false;
    for (const auto &cams : (*it2)->timestamps) {
      if ((int)cams.second.size() > state->_state->_options.max_clone_size) {
        reached_max = true;
        break;
      }
    }
    // If max track, then add it to our possible slam feature list
    if (reached_max) {
      feats_maxtracks.push_back(*it2);
      it2 = feats_marg.erase(it2);
    } else {
      it2++;
    }
  }

  // Count how many aruco tags we have in our state
  int curr_aruco_tags = 0;
  auto it0 = state->_state->_features_SLAM.begin();
  while (it0 != state->_state->_features_SLAM.end()) {
    if ((int)(*it0).second->_featid <= 4 * state->_state->_options.max_aruco_features)
      curr_aruco_tags++;
    it0++;
  }

  // Append a new SLAM feature if we have the room to do so
  // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
  if (state->_state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay &&
      (int)state->_state->_features_SLAM.size() < state->_state->_options.max_slam_features + curr_aruco_tags) {
    // Get the total amount to add, then the max amount that we can add given our marginalize feature array
    int amount_to_add = (state->_state->_options.max_slam_features + curr_aruco_tags) - (int)state->_state->_features_SLAM.size();
    int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
    // If we have at least 1 that we can add, lets add it!
    // Note: we remove them from the feat_marg array since we don't want to reuse information...
    if (valid_amount > 0) {
      feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
      feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
    }
  }

  // Loop through current SLAM features, we have tracks of them, grab them for this update!
  // NOTE: if we have a slam feature that has lost tracking, then we should marginalize it out
  // NOTE: we only enforce this if the current camera message is where the feature was seen from
  // NOTE: if you do not use FEJ, these types of slam features *degrade* the estimator performance....
  // NOTE: we will also marginalize SLAM features if they have failed their update a couple times in a row
  for (std::pair<const size_t, std::shared_ptr<ov_type::Landmark>> &landmark : state->_state->_features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<ov_core::Feature> feat1 = trackARUCO->get_feature_database()->get_feature(landmark.second->_featid);
      if (feat1 != nullptr)
        feats_slam.push_back(feat1);
    }
    std::shared_ptr<ov_core::Feature> feat2 = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
    if (feat2 != nullptr)
      feats_slam.push_back(feat2);
    assert(landmark.second->_unique_camera_id != -1);
    bool current_unique_cam =
        std::find(message.sensor_ids.begin(), message.sensor_ids.end(), landmark.second->_unique_camera_id) != message.sensor_ids.end();
    if (feat2 == nullptr && current_unique_cam)
      landmark.second->should_marg = true;
    if (landmark.second->update_fail_count > 1)
      landmark.second->should_marg = true;
  }

  // Lets marginalize out all old SLAM features here
  // These are ones that where not successfully tracked into the current frame
  // We do *NOT* marginalize out our aruco tags landmarks
  if (state->_options.do_schmidt_uwb_anchors && state->_has_initialized_schmidt){
    state->marginalize_slam();
  }
  else{
    ov_msckf::StateHelper::marginalize_slam(state->_state);
  }

  // Separate our SLAM features into new ones, and old ones
  std::vector<std::shared_ptr<ov_core::Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
  for (size_t i = 0; i < feats_slam.size(); i++) {
    if (state->_state->_features_SLAM.find(feats_slam.at(i)->featid) != state->_state->_features_SLAM.end()) {
      feats_slam_UPDATE.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: found old feature %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    } else {
      feats_slam_DELAYED.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: new feature ready %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    }
  }

  // Concatenate our MSCKF feature arrays (i.e., ones not being used for slam updates)
  std::vector<std::shared_ptr<ov_core::Feature>> featsup_MSCKF = feats_lost;
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(), feats_marg.end());
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(), feats_maxtracks.end());

  //===================================================================================
  // Now that we have a list of features, lets do the EKF update for MSCKF and SLAM!
  //===================================================================================

  // Sort based on track length
  // TODO: we should have better selection logic here (i.e. even feature distribution in the FOV etc..)
  // TODO: right now features that are "lost" are at the front of this vector, while ones at the end are long-tracks
  auto compare_feat = [](const std::shared_ptr<ov_core::Feature> &a, const std::shared_ptr<ov_core::Feature> &b) -> bool {
    size_t asize = 0;
    size_t bsize = 0;
    for (const auto &pair : a->timestamps)
      asize += pair.second.size();
    for (const auto &pair : b->timestamps)
      bsize += pair.second.size();
    return asize < bsize;
  };
  std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);

  // Pass them to our MSCKF updater
  // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
  // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
  if ((int)featsup_MSCKF.size() > state->_state->_options.max_msckf_in_update)
    featsup_MSCKF.erase(featsup_MSCKF.begin(), featsup_MSCKF.end() - state->_state->_options.max_msckf_in_update);
  // [ANDREW] Don't want to refactor all of OpenVINS to recover Kalman gain
  // [ANDREW] Will just compute based on covariance change
  // [ANDREW] TODO: Re-write OpenVins later to better incorporate Schmidt
  if (state->_options.do_schmidt_uwb_anchors && state->_has_initialized_schmidt){
    auto P_before = ov_msckf::StateHelper::get_active_covariance(state->_state);
    int n_before = P_before.rows();
    // PRINT_DEBUG(MAGENTA "Starting updaterMSCKF->update()\n" RESET);
    updaterMSCKF->update(state->_state, featsup_MSCKF);
    propagator->invalidate_cache();
    auto P_after = ov_msckf::StateHelper::get_active_covariance(state->_state);
    state->apply_cross_covariance_correction(P_before, P_after, n_before);
    }
  else{
    updaterMSCKF->update(state->_state, featsup_MSCKF);
  }
  
  rT4 = boost::posix_time::microsec_clock::local_time();

  // [ANDREW] TODO: Implement Schmidt state covariance

  // Perform SLAM delay init and update
  // NOTE: that we provide the option here to do a *sequential* update
  // NOTE: this will be a lot faster but won't be as accurate.
  std::vector<std::shared_ptr<ov_core::Feature>> feats_slam_UPDATE_TEMP;
  while (!feats_slam_UPDATE.empty()) {
    // Get sub vector of the features we will update with
    std::vector<std::shared_ptr<ov_core::Feature>> featsup_TEMP;
    featsup_TEMP.insert(featsup_TEMP.begin(), feats_slam_UPDATE.begin(),
                        feats_slam_UPDATE.begin() + std::min(state->_state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    feats_slam_UPDATE.erase(feats_slam_UPDATE.begin(), feats_slam_UPDATE.begin() + std::min(state->_state->_options.max_slam_in_update,
                                                                                            (int)feats_slam_UPDATE.size()));
    // Do the update
    if (state->_options.do_schmidt_uwb_anchors && state->_has_initialized_schmidt) {
      auto P_before = ov_msckf::StateHelper::get_active_covariance(state->_state);
      int n_before = P_before.rows();
      // PRINT_DEBUG(MAGENTA "Starting updaterSLAM->update()\n" RESET);
      updaterSLAM->update(state->_state, featsup_TEMP);
      feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
      propagator->invalidate_cache();
      auto P_after = ov_msckf::StateHelper::get_active_covariance(state->_state);
      // If we're updating schmidt states do that now    
      state->apply_cross_covariance_correction(P_before, P_after, n_before);
    }
    else {
      updaterSLAM->update(state->_state, featsup_TEMP);
      feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
      propagator->invalidate_cache();
    }
    
  }

  feats_slam_UPDATE = feats_slam_UPDATE_TEMP;
  rT5 = boost::posix_time::microsec_clock::local_time();
  if (state->_options.do_schmidt_uwb_anchors && state->_has_initialized_schmidt){
    auto P_before = ov_msckf::StateHelper::get_active_covariance(state->_state);
    int n_before = state->_state->max_covariance_size();
    // PRINT_DEBUG(MAGENTA "Starting updaterSLAM->delayed_init()\n" RESET);
    updaterSLAM->delayed_init(state->_state, feats_slam_DELAYED);
    auto P_after = ov_msckf::StateHelper::get_active_covariance(state->_state);
    int n_after = state->_state->max_covariance_size();
      // Extend _Cov_cross for any newly initialized SLAM features
    if (n_after > n_before) {
      state->_Cov_cross.conservativeResize(n_after, state->_Cov_cross.cols());
      state->_Cov_cross.bottomRows(n_after - n_before).setZero();
    }
    state->apply_cross_covariance_correction(P_before, P_after, n_before);
  }
  else{
    updaterSLAM->delayed_init(state->_state, feats_slam_DELAYED);

  }
  rT6 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Update our visualization feature set, and clean up the old features
  //===================================================================================

  // Re-triangulate all current tracks in the current frame
  if (message.sensor_ids.at(0) == 0) {

    // Re-triangulate features
    retriangulate_active_tracks(message);

    // Clear the MSCKF features only on the base camera
    // Thus we should be able to visualize the other unique camera stream
    // MSCKF features as they will also be appended to the vector
    good_features_MSCKF.clear();
  }

  // Save all the MSCKF features used in the update
  for (auto const &feat : featsup_MSCKF) {
    good_features_MSCKF.push_back(feat->p_FinG);
    feat->to_delete = true;
  }

  //===================================================================================
  // Cleanup, marginalize out what we don't need any more...
  //===================================================================================

  // Remove features that where used for the update from our extractors at the last timestep
  // This allows for measurements to be used in the future if they failed to be used this time
  // Note we need to do this before we feed a new image, as we want all new measurements to NOT be deleted
  trackFEATS->get_feature_database()->cleanup();
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup();
  }

  // First do anchor change if we are about to lose an anchor pose
  updaterSLAM->change_anchors(state->_state);

  // Cleanup any features older than the marginalization time
  if ((int)state->_state->_clones_IMU.size() > state->_state->_options.max_clone_size) {
    trackFEATS->get_feature_database()->cleanup_measurements(state->_state->margtimestep());
    if (trackARUCO != nullptr) {
      trackARUCO->get_feature_database()->cleanup_measurements(state->_state->margtimestep());
    }
  }

  // Finally marginalize the oldest clone if needed
  // [ANDREW] UVioState's marginalize_old_clone is overwritten
  // to support marginalizing the equivalent cross covariance in the
  // schmidt state cross covariance
  if (state->_options.do_schmidt_uwb_anchors && state->_has_initialized_schmidt){
    state->marginalize_old_clone();
  }
  else{
    ov_msckf::StateHelper::marginalize_old_clone(state->_state);
  }
  

  rT7 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Debug info, and stats tracking
  //===================================================================================

  // Get timing statitics information
  double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
  double time_prop = (rT3 - rT2).total_microseconds() * 1e-6;
  double time_msckf = (rT4 - rT3).total_microseconds() * 1e-6;
  double time_slam_update = (rT5 - rT4).total_microseconds() * 1e-6;
  double time_slam_delay = (rT6 - rT5).total_microseconds() * 1e-6;
  double time_marg = (rT7 - rT6).total_microseconds() * 1e-6;
  double time_total = (rT7 - rT1).total_microseconds() * 1e-6;

  // Timing information
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for propagation\n" RESET, time_prop);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for MSCKF update (%d feats)\n" RESET, time_msckf, (int)featsup_MSCKF.size());
  if (state->_state->_options.max_slam_features > 0) {
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM update (%d feats)\n" RESET, time_slam_update,
                (int)state->_state->_features_SLAM.size());
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM delayed init (%d feats)\n" RESET, time_slam_delay, (int)feats_slam_DELAYED.size());
  }
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for re-tri & marg (%d clones in state)\n" RESET, time_marg,
              (int)state->_state->_clones_IMU.size());

  std::stringstream ss;
  ss << "[TIME]: " << std::setprecision(4) << time_total << " seconds for total (camera";
  for (const auto &id : message.sensor_ids) {
    ss << " " << id;
  }
  ss << ")" << std::endl;
  PRINT_DEBUG(BLUE "%s" RESET, ss.str().c_str());

  // Finally if we are saving stats to file, lets save it to file
  if (params.record_timing_information && of_statistics.is_open()) {
    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double t_ItoC = state->_state->_calib_dt_CAMtoIMU->value()(0);
    double timestamp_inI = state->_state->_timestamp + t_ItoC;
    // Append to the file
    of_statistics << std::fixed << std::setprecision(15) << timestamp_inI << "," << std::fixed << std::setprecision(5) << time_track << ","
                  << time_prop << "," << time_msckf << ",";
    if (state->_state->_options.max_slam_features > 0) {
      of_statistics << time_slam_update << "," << time_slam_delay << ",";
    }
    of_statistics << time_marg << "," << time_total << std::endl;
    of_statistics.flush();
  }

  // Update our distance traveled
  if (timelastupdate != -1 && state->_state->_clones_IMU.find(timelastupdate) != state->_state->_clones_IMU.end()) {
    Eigen::Matrix<double, 3, 1> dx = state->_state->_imu->pos() - state->_state->_clones_IMU.at(timelastupdate)->pos();
    distance += dx.norm();
  }
  timelastupdate = message.timestamp;

  // Debug, print our current state
  PRINT_INFO("q_GtoI = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f | dist = %.2f (meters)\n", state->_state->_imu->quat()(0),
             state->_state->_imu->quat()(1), state->_state->_imu->quat()(2), state->_state->_imu->quat()(3), state->_state->_imu->pos()(0),
             state->_state->_imu->pos()(1), state->_state->_imu->pos()(2), distance);
  PRINT_INFO("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", state->_state->_imu->bias_g()(0), state->_state->_imu->bias_g()(1),
             state->_state->_imu->bias_g()(2), state->_state->_imu->bias_a()(0), state->_state->_imu->bias_a()(1),
             state->_state->_imu->bias_a()(2));

  // Debug for camera imu offset
  if (state->_state->_options.do_calib_camera_timeoffset) {
    PRINT_INFO("camera-imu timeoffset = %.5f\n", state->_state->_calib_dt_CAMtoIMU->value()(0));
  }

  // Debug for camera intrinsics
  if (state->_state->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < state->_state->_options.num_cameras; i++) {
      std::shared_ptr<ov_type::Vec> calib = state->_state->_cam_intrinsics.at(i);
      PRINT_INFO("cam%d intrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f,%.3f\n", (int)i, calib->value()(0), calib->value()(1),
                 calib->value()(2), calib->value()(3), calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Debug for camera extrinsics
  if (state->_state->_options.do_calib_camera_pose) {
    for (int i = 0; i < state->_state->_options.num_cameras; i++) {
      std::shared_ptr<ov_type::PoseJPL> calib = state->_state->_calib_IMUtoCAM.at(i);
      PRINT_INFO("cam%d extrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f\n", (int)i, calib->quat()(0), calib->quat()(1), calib->quat()(2),
                 calib->quat()(3), calib->pos()(0), calib->pos()(1), calib->pos()(2));
    }
  }

  // Debug for imu intrinsics
  if (state->_state->_options.do_calib_imu_intrinsics && state->_state->_options.imu_model == ov_msckf::StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("q_GYROtoI = %.3f,%.3f,%.3f,%.3f\n", state->_state->_calib_imu_GYROtoIMU->value()(0),
               state->_state->_calib_imu_GYROtoIMU->value()(1), state->_state->_calib_imu_GYROtoIMU->value()(2),
               state->_state->_calib_imu_GYROtoIMU->value()(3));
  }
  if (state->_state->_options.do_calib_imu_intrinsics && state->_state->_options.imu_model == ov_msckf::StateOptions::ImuModel::RPNG) {
    PRINT_INFO("q_ACCtoI = %.3f,%.3f,%.3f,%.3f\n", state->_state->_calib_imu_ACCtoIMU->value()(0),
               state->_state->_calib_imu_ACCtoIMU->value()(1), state->_state->_calib_imu_ACCtoIMU->value()(2),
               state->_state->_calib_imu_ACCtoIMU->value()(3));
  }
  if (state->_state->_options.do_calib_imu_intrinsics && state->_state->_options.imu_model == ov_msckf::StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("Dw = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_state->_calib_imu_dw->value()(0),
               state->_state->_calib_imu_dw->value()(1), state->_state->_calib_imu_dw->value()(2), state->_state->_calib_imu_dw->value()(3),
               state->_state->_calib_imu_dw->value()(4), state->_state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_state->_calib_imu_da->value()(0),
               state->_state->_calib_imu_da->value()(1), state->_state->_calib_imu_da->value()(2), state->_state->_calib_imu_da->value()(3),
               state->_state->_calib_imu_da->value()(4), state->_state->_calib_imu_da->value()(5));
  }
  if (state->_state->_options.do_calib_imu_intrinsics && state->_state->_options.imu_model == ov_msckf::StateOptions::ImuModel::RPNG) {
    PRINT_INFO("Dw = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_state->_calib_imu_dw->value()(0),
               state->_state->_calib_imu_dw->value()(1), state->_state->_calib_imu_dw->value()(2), state->_state->_calib_imu_dw->value()(3),
               state->_state->_calib_imu_dw->value()(4), state->_state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_state->_calib_imu_da->value()(0),
               state->_state->_calib_imu_da->value()(1), state->_state->_calib_imu_da->value()(2), state->_state->_calib_imu_da->value()(3),
               state->_state->_calib_imu_da->value()(4), state->_state->_calib_imu_da->value()(5));
  }
  if (state->_state->_options.do_calib_imu_intrinsics && state->_state->_options.do_calib_imu_g_sensitivity) {
    PRINT_INFO("Tg = | %.4f,%.4f,%.4f |  %.4f,%.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_state->_calib_imu_tg->value()(0),
               state->_state->_calib_imu_tg->value()(1), state->_state->_calib_imu_tg->value()(2), state->_state->_calib_imu_tg->value()(3),
               state->_state->_calib_imu_tg->value()(4), state->_state->_calib_imu_tg->value()(5), state->_state->_calib_imu_tg->value()(6),
               state->_state->_calib_imu_tg->value()(7), state->_state->_calib_imu_tg->value()(8));
  }
}

void UVioManager::check_and_marginalize_alignment(){
  
  // if (state->_options.do_schmidt_uwb_anchors){
  //   // Doing considered map
  //   // TODO: Figure out what to do here
  //   return;
  // }

  if (!state->_options.do_calib_uwb_frame_transfrom) {
    // Already marginalized or not doing frame alignment
    return;
  }

  std::vector<std::shared_ptr<ov_type::Type>> small_vars;
  small_vars.push_back(state->_calib_VIOtoUWB_frame_alignment);
  Eigen::MatrixXd P_align = ov_msckf::StateHelper::get_marginal_covariance(state->_state, small_vars);

  double yaw_cov = P_align(2, 2);
  double pos_cov_max = P_align.block<3,3>(3,3).diagonal().maxCoeff();

  if (yaw_cov > state->_options.min_yaw_covar_fix_frame_align || pos_cov_max > state->_options.min_pos_covar_fix_frame_align){
    // Hasn't converged yet
    return;
  }
  else{
    // We've converged for the first time
    // Update the marginalization options
    state->_options.do_calib_uwb_frame_transfrom = false;
    state->_marg_alignment_cov = P_align;
    state->_has_marginalized_frame_alignment = true;
    Eigen::MatrixXd H_align = state->_uwb_alignment_jac;
    state->_uwb_range_alignment_cov = (H_align * P_align * H_align.transpose())(0,0);
  }

  PRINT_INFO(GREEN "[UVIO] Alignment converged! Transforming state to global UWB frame {A} and marginalizing...\n" RESET);
  PRINT_INFO(GREEN "[UVIO] Alignment noise R influce %.4f\n" RESET, state->_uwb_range_alignment_cov);
  Eigen::Matrix3d R_av = state->_calib_VIOtoUWB_frame_alignment->Rot();
  Eigen::Vector3d p_offset = state->_calib_VIOtoUWB_frame_alignment->pos();

  Eigen::Vector3d p_V = state->_state->_imu->pos();
  Eigen::Vector3d v_V = state->_state->_imu->vel();
  Eigen::Matrix3d R_bv = state->_state->_imu->Rot();

  Eigen::Vector3d p_a = R_av * p_V + p_offset;
  Eigen::Vector3d v_a = R_av * v_V;
  Eigen::Matrix3d R_ba = R_bv * R_av.transpose();
  Eigen::Vector4d q_ba = ov_core::rot_2_quat(R_ba);

  // Set state values
  state->_state->_imu->p()->set_value(p_a);
  state->_state->_imu->q()->set_value(q_ba);
  state->_state->_imu->v()->set_value(v_a);
  // Set fej values
  state->_state->_imu->p()->set_fej(p_a);
  state->_state->_imu->q()->set_fej(q_ba);
  state->_state->_imu->v()->set_fej(v_a);
  

  // Transform features
  for (auto& feat : state->_state->_features_SLAM){
    if (feat.second->_feat_representation == ov_type::LandmarkRepresentation::Representation::GLOBAL_3D) {
      PRINT_ERROR(RED "[ERROR]: UVIO frame alignment marginalization not implemented for SLAM features in GLOBAL_3D representation\n" RESET);
      std::exit(EXIT_FAILURE);
    
    }
  }
  int total_dim = state->_state->max_covariance_size();
  Eigen::MatrixXd Psi = Eigen::MatrixXd::Identity(total_dim, total_dim);

  int imu_rot_id = state->_state->_imu->id();
  int imu_pos_id = state->_state->_imu->id() + 3;
  int imu_vel_id = state->_state->_imu->id() + 6;
  int align_rot_id = state->_calib_VIOtoUWB_frame_alignment->id();
  int align_pos_id = align_rot_id + 3;
  // Jacobian according to Geneva
  // https://copland.udel.edu/~ghuang/papers/tr_gps-vio.pdf
  Psi.block<3,3>(imu_rot_id, imu_rot_id)   =  Eigen::Matrix3d::Identity();
  // Psi.block<3,3>(imu_rot_id, align_rot_id).setZero();
  // Psi.block<3,3>(imu_rot_id, align_rot_id).col(2) = R_ba.col(2);
  // Psi.block<3,3>(imu_rot_id, align_rot_id) = R_ba;
  Psi.block<3,3>(imu_rot_id, align_rot_id) = -R_ba;

  // IMU Position Jacobians (Eq. 163)
  Psi.block<3,3>(imu_pos_id, imu_pos_id)   = R_av;
  // Psi.block<3,3>(imu_pos_id, align_rot_id).setZero();
  // Psi.block<3,3>(imu_pos_id, align_rot_id).col(2) = (-R_av * ov_core::skew_x(p_V)).col(2);
  Psi.block<3,3>(imu_pos_id, align_rot_id) = -1.0 * ov_core::skew_x(R_av * p_V);
  Psi.block<3,3>(imu_pos_id, align_pos_id) = Eigen::Matrix3d::Identity();

  // IMU Velocity Jacobians (Eq. 163)
  Psi.block<3,3>(imu_vel_id, imu_vel_id)   = R_av;
  // Psi.block<3,3>(imu_vel_id, align_rot_id).setZero();
  // Psi.block<3,3>(imu_vel_id, align_rot_id).col(2) = (-R_av * ov_core::skew_x(v_V)).col(2);
  Psi.block<3,3>(imu_vel_id, align_rot_id) = -1.0 * ov_core::skew_x(R_av * v_V);

  // Transform clones
  for (auto& clone : state->_state->_clones_IMU){
    Eigen::Vector3d p_clone_v = clone.second->pos();
    Eigen::Vector3d p_clone_a = R_av * clone.second->pos() + p_offset;
    Eigen::Matrix3d R_clone_ba = clone.second->Rot() * R_av.transpose();
    Eigen::Vector4d q_clone_ba = ov_core::rot_2_quat(R_clone_ba);
    int clone_rot_id = clone.second->id();
    int clone_pos_id = clone.second->id() + 3;

    // Fill Clone Jacobians in Psi (Eq. 163)
    Psi.block<3,3>(clone_rot_id, clone_rot_id) = Eigen::Matrix3d::Identity();
    // Psi.block<3,3>(clone_rot_id, align_rot_id).setZero();
    // Psi.block<3,3>(clone_rot_id, align_rot_id).col(2) = R_ba.col(2);
    // Psi.block<3,3>(clone_rot_id, align_rot_id) = R_ba;
    Psi.block<3,3>(clone_rot_id, align_rot_id) = -R_ba;

    Psi.block<3,3>(clone_pos_id, clone_pos_id) = R_av;
    // Psi.block<3,3>(clone_pos_id, align_rot_id).setZero();
    // Psi.block<3,3>(clone_pos_id, align_rot_id).col(2) = (-R_av * ov_core::skew_x(p_clone_v)).col(2);
    Psi.block<3,3>(clone_pos_id, align_rot_id) = -1.0 * ov_core::skew_x(R_av * p_clone_v);
    Psi.block<3,3>(clone_pos_id, align_pos_id) = Eigen::Matrix3d::Identity();

    // Update clone state values and FEJ
    clone.second->p()->set_value(p_clone_a);
    clone.second->p()->set_fej(p_clone_a);

    clone.second->q()->set_value(q_clone_ba);
    clone.second->q()->set_fej(q_clone_ba);
  }
  // Propagate total covariance matrix with Psi
  auto &cov = ov_msckf::StateHelper::get_active_covariance(state->_state);
  int align_idx = state->_calib_VIOtoUWB_frame_alignment->id();
  
  PRINT_INFO(GREEN "[UVIO] Before Jacobian transform: active covariance trace = %.6f\n" RESET, cov.trace());
  // Eigen will optimize intermediate allocations internally
  cov = Psi * cov * Psi.transpose();

  // Inflate the alignment covariance
  // Eigen::MatrixXd D = Eigen::MatrixXd::Identity(cov.rows(), cov.cols());
  // D.block<6,6>(align_idx, align_idx) *= std::sqrt(state->_options.marg_frame_align_cov_inflation);
  // cov = D * cov * D.transpose();

  // Enforce self-adjoint symmetry 
  cov = cov.selfadjointView<Eigen::Upper>();

  // 4. Update the state covariance using the helper function
  // ov_msckf::StateHelper::set_active_covariance(state->_state, cov_transformed);
  PRINT_INFO(GREEN "[UVIO] Before marginalization: active covariance trace = %.6f\n" RESET, cov.trace());
  PRINT_INFO(GREEN "[UVIO] Before marginalization: alignment covariance trace = %.6f\n" RESET, P_align.trace());

  if (state->_options.do_schmidt_uwb_anchors){
    state->marginalize_active_schmidt(state->_calib_VIOtoUWB_frame_alignment);
  }
  else{
    ov_msckf::StateHelper::marginalize(state->_state, state->_calib_VIOtoUWB_frame_alignment);
  }

  auto &cov_after =
    ov_msckf::StateHelper::get_active_covariance(state->_state);

  PRINT_INFO(GREEN "[UVIO] After marginalization: active covariance trace = %.6f\n" RESET, cov_after.trace());
  // Can now turn off fej as system is fully observable now
  // state->_state->_options.do_fej = false;
  
  PRINT_INFO(GREEN "[UVIO] Successfully transformed state to {A} and marginalized frame alignment parameter.\n" RESET);
}
