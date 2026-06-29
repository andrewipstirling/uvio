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
  if (params.uvio_state_options.do_calib_uwb_frame_transfrom){
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

  // Basic check for VIO startup and anchors have been setup
  if ( !is_initialized_vio || !are_initialized_anchors || (message.timestamp < startup_time + 2.0) || (distance < 0.1)) return;

  // Return if the uwb measurement is out of order otherwise feed our bar measuremnts
  if (state->_state->_timestamp >= message.timestamp) {
    PRINT_DEBUG(YELLOW "UWB measurements received out of order (prop dt = %3f)\n" RESET, (message.timestamp - state->_state->_timestamp));
    return;
  }
  // If in localization setup, initialize the frame transform
  if ((uwb_init_stage == INITIAL || uwb_init_stage == REFINEMENT) && state->_options.do_calib_uwb_frame_transfrom && is_initialized_vio){
    // Get start time for debugging
    if (initialization_start_time < 0){
      initialization_start_time = ros::Time::now().toSec();
    }

    // Store in separate buffer for batch alignment problem
    std::vector<AlignmentConstraint> alignment_measurements;
    
    // build constraints
    for (const auto& r : message.uwb_ranges){
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
      Eigen::MatrixXd P_full = ov_msckf::StateHelper::get_marginal_covariance(state->_state,vio_vars);
      c.P_vio = P_full.block<6,6>(0,0);
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
        uwb_alignment_initializer->can_initialize(distance, params.min_dist_to_use_uwb*0.5, params.min_uwb_ranges_for_alignment*0.5, params.max_pdop*2)){
      // VIO to UWB frame rotation
      Eigen::Matrix3d C_av;
      // VIO (wv) relative to UWB (wa) frame trans
      Eigen::Vector3d r_wvwa_a;
      // Initialization covariance
      Eigen::Matrix4d init_covar;
      bool use_squared_cost = true;
      bool use_initial_guess = false;
      if (uwb_alignment_initializer->solve(C_av, r_wvwa_a, init_covar, use_initial_guess, use_squared_cost)){
        PRINT_INFO(GREEN "[UVIO] Initial alignment success\n" RESET);
        // Set initial 
        C_av_est = C_av;
        r_av_est = r_wvwa_a;

        uwb_init_stage = REFINEMENT;
        uwb_alignment_initializer->clear();

      }
    }
    else if (uwb_init_stage == REFINEMENT &&
      uwb_alignment_initializer->can_refine(C_av_est, r_av_est, distance, params.min_dist_to_use_uwb, params.min_uwb_ranges_for_alignment, params.max_pdop, params.min_fim_eigenvalue)){
      Eigen::Matrix3d C_ref = C_av_est;
      Eigen::Vector3d r_ref = r_av_est;
      // Initialization covariance
      Eigen::Matrix4d init_covar;
      bool use_squared_cost = false;
      bool use_initial_guess = true;
      if (uwb_alignment_initializer->solve(C_ref, r_ref, init_covar, use_initial_guess, use_squared_cost)){
        PRINT_INFO(GREEN "[UVIO] Success! VIO-UWB frames aligned after %d UWB ranges and %3fm travelled.\n" RESET, uwb_alignment_initializer->data_count(), distance);
        uwb_init_stage = CAN_INJECT;

        C_av_est = C_ref;
        r_av_est = r_ref;
        cov_av_est = init_covar;
      }
    }
    else return;
  }

  else {
    // We have initial estimate of alignment vars so can feed state
    past_measurements.insert({message.timestamp, std::make_shared<UwbData>(message)});
  }

}

const UVioManagerOptions& UVioManager::get_uvio_params() const {
  return params;
}

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
  if ( uwb_init_stage == CAN_INJECT and state->_options.do_calib_uwb_frame_transfrom) {
    PRINT_INFO(GREEN "[UVIO] Injecting UWB alignment states into filter\n" RESET);
    // Add the alignment values to state
    Eigen::Matrix<double, 7, 1> x_alignment;
    x_alignment << ov_core::rot_2_quat(C_av_est), r_av_est;
    state->_calib_VIOtoUWB_frame_alignment->set_value(x_alignment);
    
    // Add to msckf
    std::vector<std::shared_ptr<ov_type::Type>> H_order;
    H_order.push_back(state->_state->_imu->pose()); // Relate to IMU pose
    Eigen::MatrixXd H_R = Eigen::MatrixXd::Zero(6, 6);
    Eigen::MatrixXd H_L = Eigen::MatrixXd::Identity(6, 6);
    Eigen::MatrixXd R_init = Eigen::MatrixXd::Identity(6,6);
    Eigen::VectorXd res = Eigen::VectorXd::Zero(6);

    // Initialize the state
    ov_msckf::StateHelper::initialize_invertible(state->_state, state->_calib_VIOtoUWB_frame_alignment, H_order, H_R, H_L, R_init, res);
    // ov_msckf::StateHelper::initialize(state->_state, state->_calib_VIOtoUWB_frame_alignment, H_order, H_R, H_L, R_init, res, 1.0);
    PRINT_INFO(GREEN "[UVIO] UWB alignment initialized and added to state.\n" RESET);
    // Initialize the covariance
    std::vector<std::shared_ptr<ov_type::Type>> H_order_cov;
    H_order.clear();
    H_order.push_back(state->_calib_VIOtoUWB_frame_alignment);
    
    // Find the maximum value in your batch covariance diagonal to stay safe
    double max_var = cov_av_est.diagonal().maxCoeff();
    // Also check against your manual Pitch/Roll variance
    // max_var = std::max(max_var, std::pow(0.1 * M_PI, 2));

    // Set Pitch and Roll with max_var
    Eigen::Matrix<double, 6, 6> R_final = Eigen::Matrix<double, 6, 6>::Identity() * cov_av_est(0,0) * 1;
    // Yaw & Position
    R_final.block<4,4>(2,2) = cov_av_est * 1;
    R_final *= params.uvio_state_options.init_inflation_uwb_frame_align; // 10 best performance for ls ransac -> sqrangecost -> rangecost

    ov_msckf::StateHelper::set_initial_covariance(state->_state, R_final, H_order);
    
    PRINT_INFO(GREEN "[UVIO] UWB alignment initialized covariance = [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f] \n" RESET, R_final(0,0),R_final(1,1), R_final(2,2), R_final(3,3), R_final(4,4), R_final(5,5));

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
    for (const auto& id: params.uvio_state_options.tag_ids){
      const auto tag_val = state->get_calib_uwb(id)->value();
      PRINT_INFO(YELLOW "calib_UWBtoIMU = [%.3f,%.3f,%.3f]\n" RESET, tag_val(0), tag_val(1), tag_val(2));
    }
    
  }
  for (const auto &it : state->_calib_GLOBALtoANCHORS) {
    if (!it.second->fixed()) {
      PRINT_INFO(YELLOW "anchor[%d]: p_AinG = [%.3f, %.3f, %.3f]\n" RESET, it.first,
                 it.second->p_AinG()->value()(0), it.second->p_AinG()->value()(1), it.second->p_AinG()->value()(2));
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
    for (size_t tag_id : params.uvio_state_options.tag_ids){
      std::pair<size_t, size_t> tag_anchor_id{tag_id, it.id};
      std::shared_ptr<UWBBias> bias = std::make_shared<UWBBias>(it.const_bias, it.dist_bias);
      state->_uwb_biases_map[tag_anchor_id] = bias;
      // [Andrew] If we want to estimate this, then add to state vector
      if (state->_options.do_calib_uwb_biases){
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
      state->register_schmidt(it.id, it.cov);
    }

    // Print anchor info
    PRINT_INFO("anchor[%d]: p_AinG = [%.3f, %.3f, %.3f] | const_bias = %.4f | dist_bias = %.4f\n", it.id, it.p_AinG.x(), it.p_AinG.y(),
               it.p_AinG.z(), it.const_bias, it.dist_bias);
    std::cout << "cov = \n" << it.cov << "\n" << std::endl;
  }
  are_initialized_anchors = true;
  PRINT_INFO("UWB anchors correctly initialized\n");
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
  std::cout << "cov = \n" << anchor.cov << "\n" << std::endl;
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
      // TODO: Alter to include the tag_id this comes from
    }
  }
}

