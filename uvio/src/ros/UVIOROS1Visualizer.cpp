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

#include <ros/UVIOROS1Visualizer.h>
#include "core/UVioManager.h"  // [Andrew] For forward declaration
#include <utils/opencv_yaml_parse.h>
#include <utils/utils.h>
#include <utils/uvio_sensor_data.h>
#include <visualization_msgs/Marker.h>

using namespace uvio;

UVIOROS1Visualizer::UVIOROS1Visualizer(std::shared_ptr<ros::NodeHandle> nh, std::shared_ptr<UVioManager> app, std::shared_ptr<ov_msckf::Simulator> sim)
    : ov_msckf::ROS1Visualizer(nh, std::static_pointer_cast<ov_msckf::VioManager>(app), sim), _app(app) {


      std::string spline_path;
      _nh->param<std::string>("spline_config", spline_path, "");

      if (!spline_path.empty()) {
        load_spline(spline_path);
      } else {
        PRINT_WARNING(YELLOW "[Visualizer] No spline_file parameter found. UWB corrections disabled.\n" RESET);
      }
      // Initialize the UWB range visualizer
      _pub_uwb_viz = nh->advertise<visualization_msgs::Marker>("uwb_visuals/active_ranges", 2);

      // Initialize the UWB-VIO frame transform publisher
      // Publishes the global frame instead
      _pub_global_pose = nh->advertise<geometry_msgs::PoseWithCovarianceStamped>("poseimu_global", 2);
    }

void UVIOROS1Visualizer::setup_subscribers(std::shared_ptr<ov_core::YamlParser> parser) {
  ov_msckf::ROS1Visualizer::setup_subscribers(parser);

  // Re-assign imu subscriber
  std::string topic_imu;
  _nh->param<std::string>("topic_imu", topic_imu, "/imu0");
  parser->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);
  sub_imu = _nh->subscribe(topic_imu, 1000, &UVIOROS1Visualizer::callback_inertial, this);

  // Parsing uwb topic
  std::string topic_uwb;
  _nh->param<std::string>("topic_uwb", topic_uwb, "/uwb/range");
  // TODO: Change this
  parser->parse_external("config_uwb", "tag0", "rostopic", topic_uwb);
  _sub_uwb = _nh->subscribe(topic_uwb, 1, &UVIOROS1Visualizer::callback_uwb, this);
  PRINT_DEBUG("subscribing to uwb: %s\n", topic_uwb.c_str());

  // Parsing initialized anchors topic
  std::string topic_anchors_init;
  _nh->param<std::string>("topic_anchors_init", topic_anchors_init, "/uwb_init/anchors");
  parser->parse_external("config_uwb", "init", "rostopic", topic_anchors_init);
  _sub_anchors_init = _nh->subscribe(topic_anchors_init, 1, &UVIOROS1Visualizer::callback_anchors_init, this);
  PRINT_DEBUG("subscribing to anchors_init: %s\n", topic_anchors_init.c_str());
}

void UVIOROS1Visualizer::callback_inertial(const sensor_msgs::Imu::ConstPtr &msg) {

  // convert into correct format
  ov_core::ImuData message;
  message.timestamp = msg->header.stamp.toSec();
  message.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
  message.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;

  // send it to our VIO system
  _app->feed_measurement_imu(message);
  visualize_odometry(message.timestamp);

  // If the processing queue is currently active / running just return so we can keep getting measurements
  // Otherwise create a second thread to do our update in an async manor
  // The visualization of the state, images, and features will be synchronous with the update!
  if (thread_update_running)
    return;
  thread_update_running = true;
  std::thread thread([&] {
    // Lock on the queue (prevents new images from appending)
    std::lock_guard<std::mutex> lck(camera_queue_mtx);

    // Count how many unique image streams
    std::map<int, bool> unique_cam_ids;
    for (const auto &cam_msg : camera_queue) {
      unique_cam_ids[cam_msg.sensor_ids.at(0)] = true;
    }

    // If we do not have enough unique cameras then we need to wait
    // We should wait till we have one of each camera to ensure we propagate in the correct order
    auto params = _app->get_params();
    size_t num_unique_cameras = (params.state_options.num_cameras == 2) ? 1 : params.state_options.num_cameras;
    if (unique_cam_ids.size() == num_unique_cameras) {

      // Loop through our queue and see if we are able to process any of our camera measurements
      // We are able to process if we have at least one IMU measurement greater than the camera time
      double timestamp_imu_inC = message.timestamp - _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
      while (!camera_queue.empty() && camera_queue.at(0).timestamp < timestamp_imu_inC) {
        auto rT0_1 = boost::posix_time::microsec_clock::local_time();
        double update_dt = 100.0 * (timestamp_imu_inC - camera_queue.at(0).timestamp);
        _app->feed_measurement_camera(camera_queue.at(0));
        visualize();
        publish_global_pose();
        camera_queue.pop_front();
        auto rT0_2 = boost::posix_time::microsec_clock::local_time();
        double time_total = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
        PRINT_INFO(BLUE "[TIME]: %.4f seconds total (%.1f hz, %.2f ms behind)\n" RESET, time_total, 1.0 / time_total, update_dt);
      }
    }
    thread_update_running = false;
  });

  // If we are single threaded, then run single threaded
  // Otherwise detach this thread so it runs in the background!
  if (!_app->get_params().use_multi_threading_subs) {
    thread.join();
  } else {
    thread.detach();
  }
}

void UVIOROS1Visualizer::visualize_odometry(double timestamp) {

  ROS1Visualizer::visualize_odometry(timestamp);

  // If anchors are initialized, publish transforms on TF
  if (_app->get_are_initialized_anchors()) {

    // Loop through all anchors and publish their transforms
    for (const auto &anchor : _app->get_uvio_state()->_calib_GLOBALtoANCHORS) {
      std::string anchor_name = "anchor[" + std::to_string(anchor.first) + "]";
      tf::StampedTransform trans_anchor;
      trans_anchor.stamp_ = ros::Time::now();
      trans_anchor.setOrigin(
          tf::Vector3(anchor.second->p_AinG()->value()(0), anchor.second->p_AinG()->value()(1), anchor.second->p_AinG()->value()(2)));
      trans_anchor.setRotation(tf::Quaternion(0, 0, 0, 1));
      trans_anchor.frame_id_ = "global"; // TODO: make it a parameter (?)
      trans_anchor.child_frame_id_ = anchor_name;
      mTfBr->sendTransform(trans_anchor);
    }
  }

  // Publish uwb-imu calibration transform on TF
  // [Andrew] Changed for multi-tag agents
  auto uvio_state = _app->get_uvio_state();
  for (auto const& item : uvio_state->_calib_UWBtoIMU_map){
    size_t tag_id = item.first; 
    auto const& tag_var = item.second; 
    if (tag_var == nullptr) continue;

    tf::StampedTransform trans_calib_uwb;
    trans_calib_uwb.stamp_ = ros::Time::now();
    // [Andrew] UWB extrinsic state is p_IrelU
    // p_UrelI = -p_IrelU
    trans_calib_uwb.setOrigin(tf::Vector3(
      -tag_var->value()(0),
      -tag_var->value()(1),
      -tag_var->value()(2)
    ));
    // [Andrew] uwb extrinsics only models translation
    trans_calib_uwb.setRotation(tf::Quaternion(0, 0, 0, 1));
    trans_calib_uwb.frame_id_ = "imu";
    trans_calib_uwb.child_frame_id_ = "uwb_" + std::to_string(tag_id);
    mTfBr->sendTransform(trans_calib_uwb);

  }
}

#if UWB_DRIVER == EVB_DRIVER
void UVIOROS1Visualizer::callback_uwb(const evb1000_driver::TagDistanceConstPtr &msg_uwb) {

  // Convert measurement to correct format
  UwbData message;
  message.timestamp = msg_uwb->header.stamp.toSec();

  // Get the number of measurements
  int n = msg_uwb->valid.size();

  // Fill the map with only valid ranges
  for (int i = 0; i < n; ++i) {
    if (msg_uwb->valid[i]) {
      message.uwb_ranges.insert({i, msg_uwb->distance[i]});
    }
  }

  // send it to system
  _app->feed_measurement_uwb(message);
}
#elif UWB_DRIVER == MDEK_DRIVER

void UVIOROS1Visualizer::callback_uwb(const mdek_uwb_driver::UwbConstPtr &msg_uwb) {
  UwbData message;
  message.timestamp = msg_uwb->header.stamp.toSec();

  // Check if we have measurements
  if (!msg_uwb->ranges.empty()) {
    for (const auto &it : msg_uwb->ranges) {
      // Check if the id is valid (contains only numbers)
      if (!containsChar(it.id)) {
        // Convert string id to size_t
        std::stringstream sstream(it.id);
        size_t id;
        sstream >> id;
        // Populate map
        message.uwb_ranges.insert({id, it.distance});
      } else {
        ROS_WARN("Received UWB message with characters within the id field");
      }
    }
  }

  // send it to system
  _app->feed_measurement_uwb(message);
}

#elif UWB_DRIVER == UWB_ROS_DRIVER

std::pair<double,double> UVIOROS1Visualizer::process_range(const uwb_ros::RangeStamped::ConstPtr &msg_uwb) {
  double range = static_cast<double>(msg_uwb->range);
  // DS-TWR model
  double tx1 = msg_uwb->tx1 * _dwt_to_ns;
  double rx1 = msg_uwb->rx1 * _dwt_to_ns;
  double tx2 = msg_uwb->tx2 * _dwt_to_ns;
  double rx2 = msg_uwb->rx2 * _dwt_to_ns;
  double tx3 = msg_uwb->tx3 * _dwt_to_ns;
  double rx3 = msg_uwb->rx3 * _dwt_to_ns;

  // Decawave uses 40-bit or 32-bit timer
  const double max_time_ns = std::pow(2,32) * _dwt_to_ns;
  unwrap(tx1, rx2, rx3, max_time_ns);
  unwrap(rx1, tx2, tx3, max_time_ns);
  // if (tx2 < tx1) {
  //   tx2 += max_time_ns;
  //   tx3 += max_time_ns; 
  // }
  // if (tx3 < tx2) {
  //   tx3 += max_time_ns;
  // }
  
  // Get time intervals
  double Ra1 = rx2 - tx1;
  double Ra2 = rx3 - rx2;
  double Db1 = tx2 - rx1;
  double Db2 = tx3 - tx2;

  // Get antenna delays
  size_t tag_id = static_cast<size_t>(msg_uwb->from_id);  // Tag ID
  size_t anchor_id = static_cast<size_t>(msg_uwb->to_id);  // Anchor ID
  double delay_0 = _uwb_delays[tag_id];
  double delay_1 = _uwb_delays[anchor_id];

  // Correct time intervals for antenna delays
  Ra1 += delay_0;
  Db1 -= delay_1;

  double fpp1 = msg_uwb->fpp1;
  double fpp2 = msg_uwb->fpp2;
  fpp1 = std::pow(10, (fpp1 + 82) / 10);
  fpp2 = std::pow(10, (fpp2 + 82) / 10);
  double fpp_lift_avg = 0.5 * (fpp1 + fpp2);
  // PRINT_DEBUG(MAGENTA "Avg FPP lifted: %.4f" RESET, fpp_lift_avg);
  double range_bias = _bias_spline.evaluate(fpp_lift_avg);
  double std_dev = _std_spline.evaluate(fpp_lift_avg) * 1.2;

  range = 0.5 * _c / 1e9 * (Ra1 - (Ra2 / Db2) * Db1) - range_bias;
  std::pair<double, double> range_std{range, std_dev};
  return range_std;
}

void UVIOROS1Visualizer::callback_uwb(const uwb_ros::RangeStamped::ConstPtr &msg_uwb) {
  UwbData message;
  message.timestamp = msg_uwb->header.stamp.toSec();
  size_t tag_id = static_cast<size_t>(msg_uwb->from_id);  // Tag ID
  size_t anchor_id = static_cast<size_t>(msg_uwb->to_id);  // Anchor ID
  const auto& params = _app->get_uvio_params();
  bool do_dstwr_uwb = params.uvio_state_options.do_dstwr_uwb;
  double range = msg_uwb->range;
  double std_dev = params.uwb_options.uwb_sigma_range;
  if (do_dstwr_uwb){
    std::pair<double, double> range_std = process_range(msg_uwb);
    range = range_std.first;
    std_dev = range_std.second ;
  }

  // Get list of valid tags
  const std::vector<size_t>& valid_tags = params.uvio_state_options.tag_ids;

  auto it = std::find(valid_tags.begin(), valid_tags.end(), tag_id);
  bool is_valid_tag = (it != valid_tags.end());

  // Add to the vector of measurements at timestamp
  // Filter inter-tag measurements, anchors have IDs < 20
  // TODO: Use config file of valid anchor IDs
  if (anchor_id < 20 && is_valid_tag) {
    UwbMeasurement meas(tag_id, anchor_id, range, std_dev);
    message.uwb_ranges.push_back(meas);
    // Moved to UpdaterUWB to visualize rejected measurements
    // TODO: Remove the below commented
    // if (_app->initialized()){
    //   // [Andrew] TODO Change this.
    //   bool rej = false;
    //   visualize_uwb_measurement(tag_id, anchor_id, range, rej);
    // }
    
  }
  else if (!is_valid_tag) {
    PRINT_DEBUG(YELLOW "[UWB] Range from Tag [%zu] ignored (not in tag_ids config)\n" RESET, tag_id);
  }
  else {
    PRINT_DEBUG(YELLOW "[UWB] Removed UWB intertag measurement from Tag [%zu] to Tag [%zu]\n" RESET, tag_id, anchor_id);
  }

  // send it to system
  _app->feed_measurement_uwb(message);
}
#endif

void UVIOROS1Visualizer::callback_anchors_init(const UwbAnchorArrayStampedConstPtr &msg) {
  // TODO: Alter this

  PRINT_INFO(GREEN "Received callback for uwb anchors initialization at time %f\n" RESET, msg->header.stamp.toSec());

  // Vector of uwb anchors
  std::vector<AnchorData> anchors;

  for (const auto &it : msg->anchors) {
    AnchorData anchor;
    anchor.id = it.id;
    anchor.p_AinG << it.position.x, it.position.y, it.position.z;
    anchor.const_bias = it.gamma;
    anchor.dist_bias = it.beta - 1; // beta = 1 + dist_bias

    // Fill the covariance matrix
    int idx = 0;
    for (int i = 0; i < anchor.cov.rows(); i++) {
      for (int j = i; j < anchor.cov.cols(); j++) {
        anchor.cov(i, j) = it.covariance.at(idx++);
        anchor.cov(j, i) = anchor.cov(i, j);
      }
    }
    anchors.push_back(anchor);
  }

  // Sort the vector based on the determinant of the cov matrix
  std::sort(anchors.begin(), anchors.end(),
            [](const AnchorData &a, const AnchorData &b) { return a.cov.determinant() < b.cov.determinant(); });

  // If we have anchors to fix, then fix them
  if (_app->get_n_anchors_to_fix() > 0) {
    for (int i = 0; i < _app->get_n_anchors_to_fix(); i++) {
      anchors.at(i).fix = true;
    }
  }

  // Try to initialize anchors
  _app->try_to_initialize_uwb_anchors(anchors);
}

void UVIOROS1Visualizer::visualize_uwb_measurement(size_t tag_id, size_t anchor_id, double range, bool rejected){
  auto state = _app->get_uvio_state();

  // [Andrew] Safety checks
  if (!state || !state->_state || !state->_state->_imu){
    PRINT_DEBUG(MAGENTA"[UWB VIZ] Exit: State or IMU not initialized.\n" RESET)
    return;
  }

  // [Andrew] Return if anchor or tag participating in the ranging
  // don't exist in our state estimate
  if (state->_calib_GLOBALtoANCHORS.find(anchor_id) == state->_calib_GLOBALtoANCHORS.end() || 
      state->_calib_UWBtoIMU_map.find(tag_id) == state->_calib_UWBtoIMU_map.end()){
        PRINT_DEBUG(MAGENTA"[UWB VIZ] Exit: Tag or Anchor ID don't exist.\n" RESET);
        return;
      }
  auto anchor_var = state->_calib_GLOBALtoANCHORS[anchor_id];
  auto tag_var = state->_calib_UWBtoIMU_map[tag_id];

  if (!anchor_var || !tag_var){
    PRINT_DEBUG(MAGENTA"[UWB VIZ] Exit: Null pointer for anchor or tag variable\n" RESET);
    return;
    
  } 
  
  // Eigen check: ensure the vectors are actually 3x1 (prevents the resize assertion)
  if (anchor_var->p_AinG()->value().rows() != 3 || tag_var->value().rows() != 3) {
    PRINT_DEBUG(MAGENTA "[UWB VIZ] Exit: Eigen dimension mismatch (Rows: %ld, %ld)\n" RESET, 
               anchor_var->value().rows(), tag_var->value().rows());
    return;
  }
  // [Andrew] Get tag position in global frame
  // [Andrew] I like this notation way more than p_IinG #fosho
  Eigen::Vector3d p_aw_g = anchor_var->p_AinG()->value();
  Eigen::Vector3d p_ui_b = -tag_var->value();
  Eigen::Vector3d p_iw_a = state->_state->_imu->pose()->pos();
  Eigen::Matrix3d C_ba = state->_state->_imu->pose()->Rot();
  Eigen::Vector3d p_uw_a = C_ba.transpose() * p_ui_b + p_iw_a;

  // [Andrew] Creat the Line Marker object
  visualization_msgs::Marker marker;
  marker.header.frame_id = "global";
  marker.header.stamp = ros::Time::now();
  marker.ns = "active_ranges";
  marker.id = (int32_t)(anchor_id + tag_id * 100); // Unique ID per pair
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  // [Andrew] Change the color based on the rejected flag
  if (rejected) {
    marker.color.r = 1.0; // Red for rejected
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;   // Make rejections fully opaque
    marker.scale.x = 0.01;  // Make rejections THICKER to stand out
    marker.lifetime = ros::Duration(0.1); // Keep it visible for 1 second
    PRINT_DEBUG(RED "DRAWING Rejected: Tag %zu -> Anchor %zu (Range: %.2f)\n" RESET, tag_id, anchor_id, range);
    
  } else {
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 0.6;   // Keep successful ranges subtle
    marker.scale.x = 0.01;
    marker.lifetime = ros::Duration(0.1); // Keep it short for smooth animation
    PRINT_DEBUG(MAGENTA "DRAWING: Tag %zu -> Anchor %zu (Range: %.2f)\n" RESET, tag_id, anchor_id, range);
  }


  geometry_msgs::Point p_start, p_end;
  p_start.x = p_uw_a.x(); p_start.y = p_uw_a.y(); p_start.z = p_uw_a.z();
  p_end.x = p_aw_g.x(); p_end.y = p_aw_g.y(); p_end.z = p_aw_g.z();
  marker.points.push_back(p_start);
  marker.points.push_back(p_end);
  
  _pub_uwb_viz.publish(marker);

}

void UVIOROS1Visualizer::load_spline(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        PRINT_ERROR(RED "[Visualizer] Failed to open spline file: %s\n" RESET, filename.c_str());
        return;
    }

    uint32_t n;
    double x0, dx;
    in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&x0), sizeof(double));
    in.read(reinterpret_cast<char*>(&dx), sizeof(double));

    std::vector<double> bias_vec(n);
    std::vector<double> std_vec(n);
    in.read(reinterpret_cast<char*>(bias_vec.data()), n * sizeof(double));
    in.read(reinterpret_cast<char*>(std_vec.data()),  n * sizeof(double));

    // Read Antenna delays
    uint32_t nd;
    in.read(reinterpret_cast<char*>(&nd), sizeof(uint32_t));
    _uwb_delays.clear();
    for (uint32_t i = 0; i < nd; ++i) {
        int32_t k; double v;
        in.read(reinterpret_cast<char*>(&k), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&v), sizeof(double));
        _uwb_delays[static_cast<size_t>(k)] = v;
    }

    // Populate the structs
    _bias_spline.n = n; _bias_spline.x0 = x0; _bias_spline.dx = dx;
    _bias_spline.spline = std::make_unique<SplineGroup::QuadSpline>(std::move(bias_vec), x0, dx);

    _std_spline.n = n; _std_spline.x0 = x0; _std_spline.dx = dx;
    _std_spline.spline = std::make_unique<SplineGroup::QuadSpline>(std::move(std_vec), x0, dx);

    PRINT_INFO(GREEN "[Visualizer] Splines initialized. FPP: [%.2f, %.2f]\n" RESET, 
               x0, x0 + (n-1)*dx);
}

void UVIOROS1Visualizer::publish_global_pose(){
  // Check we are in localization setup with frame alignment initialization done
  if (!_app->get_is_initialized_uwb_frame_transform()) return;

  auto uvio_state = _app->get_uvio_state();
  if (!uvio_state || !uvio_state->_state || !uvio_state->_state->_imu) return;

  // 2. Timestamp Handling (Aligning with IMU clock frame like publish_state)
  double t_ItoC = uvio_state->_state->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = uvio_state->_state->_timestamp + t_ItoC;

  PRINT_DEBUG(GREEN "[UVIO] Publishing UWB Global Pose\n" RESET);
  // Extract VIO State (Local)
  Eigen::Vector3d p_vio = uvio_state->_state->_imu->pos();
  // OpenVins JPL, Rotation from VIO to body frame
  Eigen::Matrix3d R_bv = uvio_state->_state->_imu->Rot();

  // Extract Frame Transform
  Eigen::Vector3d p_uwb_vio = uvio_state->_calib_VIOtoUWB_frame_alignment->pos();
  // Rotation from vio to UWB global
  Eigen::Matrix3d R_av = uvio_state->_calib_VIOtoUWB_frame_alignment->Rot(); 

  Eigen::Vector3d p_global = R_av * p_vio + p_uwb_vio;
  Eigen::Matrix3d R_ab = R_av * R_bv.transpose(); // Rotation from body to global
  // JPL quat, [x, y, z, w]
  // Eigen::Vector4d q_global = ov_core::rot_2_quat(R_ab);
  Eigen::Quaterniond q_global(R_ab);

  // Populate ROS Message
  geometry_msgs::PoseWithCovarianceStamped msg;
  msg.header.stamp = msg.header.stamp = ros::Time(timestamp_inI);
  // msg.header.seq = poses_seq_global; // Assumes this is defined in your header
  msg.header.frame_id = "uwb_global_anchor"; // The UWB anchor coordinate system
  
  msg.pose.pose.position.x = p_global(0);
  msg.pose.pose.position.y = p_global(1);
  msg.pose.pose.position.z = p_global(2);
  
  // msg.pose.pose.orientation.x = q_global(0);
  // msg.pose.pose.orientation.y = q_global(1);
  // msg.pose.pose.orientation.z = q_global(2);
  // msg.pose.pose.orientation.w = q_global(3);

  // Eigen::Quaterniond Formulation
  msg.pose.pose.orientation.x = q_global.x();
  msg.pose.pose.orientation.y = q_global.y();
  msg.pose.pose.orientation.z = q_global.z();
  msg.pose.pose.orientation.w = q_global.w();

  // Compute Associated Covariance of new global pose 15 (IMU) + 6 (Align)
  std::vector<std::shared_ptr<ov_type::Type>> small_vars;
  small_vars.push_back(uvio_state->_state->_imu);
  small_vars.push_back(uvio_state->_calib_VIOtoUWB_frame_alignment);
  Eigen::MatrixXd P_marginal = ov_msckf::StateHelper::get_marginal_covariance(uvio_state->_state, small_vars);
  // Covariance Extraction
  Eigen::Matrix<double, 12, 12> P_full = Eigen::MatrixXd::Zero(12,12);
  // VIO Covariance
  Eigen::Matrix<double, 6, 6> P_vio = Eigen::MatrixXd::Zero(6,6);
  P_vio.block<3,3>(0,0) = P_marginal.block<3,3>(0,0); // orientation
  P_vio.block<3,3>(3,3) = P_marginal.block<3,3>(3,3); // position
  P_vio.block<3,3>(0,3) = P_marginal.block<3,3>(0,3); // cross
  P_vio.block<3,3>(3,0) = P_marginal.block<3,3>(3,0); // cross
  // Alignment Covariance
  Eigen::Matrix<double, 6, 6> P_align = P_marginal.block<6,6>(15, 15);
  PRINT_INFO(GREEN "[UVIO] Alignment Covariance = [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f] \n" RESET, P_align(0,0),P_align(1,1), P_align(2,2), P_align(3,3), P_align(4,4), P_align(5,5));
  // Cross Covar VIO x Alignment
  Eigen::Matrix<double,6,6> P_cross = P_marginal.block<6,6>(0,15);
  // Fill the P_full
  P_full.block<6,6>(0,0) = P_vio;
  P_full.block<6,6>(6,6) = P_align;
  P_full.block<6,6>(0,6) = P_cross;
  P_full.block<6,6>(6,0) = P_cross.transpose();


  // Compute Matrix Adjoint
  Eigen::Matrix<double, 6, 6> Adj_T = Eigen::MatrixXd::Zero(6,6);
  Adj_T.block<3,3>(0, 0) = R_av;
  Adj_T.block<3,3>(3, 3) = R_av;
  Adj_T.block<3,3>(3, 0) = ov_core::skew_x(p_uwb_vio) * R_av;

  // Compute Jacobian [J_vio, J_align]
  Eigen::Matrix<double, 6, 12> Jac = Eigen::MatrixXd::Zero(6,12);
  // \delta \theta_ab wrt X_vio, X_align
  Jac.block<3,3>(0, 0) = -1 * Eigen::MatrixXd::Identity(3,3);
  Jac.block<3,3>(0, 6) = R_bv;
  
  // \delta r_a wrt X_vio, X_align
  Jac.block<3,3>(3, 3) = R_av;
  Jac.block<3,3>(3, 6) = -1 * R_av * ov_core::skew_x(p_vio);
  Jac.block<3,3>(3, 9) = Eigen::MatrixXd::Identity(3,3);

  // Compute Covariance
  // Eigen::Matrix<double, 6, 6> covar = Adj_T * P_vio * Adj_T.transpose() + P_align;
  Eigen::Matrix<double, 6, 6> covar = Jac * P_full * Jac.transpose();
  // covar = 0.5 * (covar + covar.transpose());
  // Extract Covariance blocks
  Eigen::Matrix3d cov_orient = covar.block<3,3>(0,0); // R_ab
  Eigen::Matrix3d cov_orient_pos = covar.block<3,3>(0,3);
  Eigen::Matrix3d cov_pos_orient = covar.block<3,3>(3,0);
  Eigen::Matrix3d cov_pos = covar.block<3,3>(3,3); // p_global
  // ROS organizes covariance through [pos, orientation]
  Eigen::Matrix<double, 6, 6> ROS_covar;
  ROS_covar.block<3,3>(0,0) = cov_pos;
  ROS_covar.block<3,3>(0,3) = cov_pos_orient;
  ROS_covar.block<3,3>(3,0) = cov_orient_pos;
  ROS_covar.block<3,3>(3,3) = cov_orient;
  // Add covariance to ROS msg
  // Copy to ROS message (row-major)
  for(int i = 0; i < 6; i++) {
    for(int j = 0; j < 6; j++) {
      msg.pose.covariance[i*6 + j] = ROS_covar(i, j);
    }
  }

  _pub_global_pose.publish(msg);

  // Publish Path
  // geometry_msgs::PoseStamped posetemp;
  // posetemp.header = msg.header;
  // posetemp.pose = msg.pose.pose;
  // poses_global.push_back(posetemp); // Assumes std::vector<geometry_msgs::PoseStamped>

  // // Create and Publish Path with Downsampling
  // nav_msgs::Path path_msg;
  // path_msg.header.stamp = ros::Time::now();
  // path_msg.header.seq = poses_seq_global;
  // path_msg.header.frame_id = "uwb_global_anchor";

  // // Downsample logic to prevent Rviz memory lag
  // double skip = std::floor((double)poses_global.size() / 16384.0) + 1;
  // for (size_t i = 0; i < poses_global.size(); i += (size_t)skip) {
  //   path_msg.poses.push_back(poses_global.at(i));
  // }
  
  // _pub_global_path.publish(path_msg);

  // // 8. Increment Sequence
  // poses_seq_global++;
}
