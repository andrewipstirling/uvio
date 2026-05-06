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

#ifndef UVIO_ROS1VISUALIZER_H
#define UVIO_ROS1VISUALIZER_H

#define MDEK_DRIVER 1
#define EVB_DRIVER 2
#define UWB_ROS_DRIVER 3


#ifndef UWB_DRIVER
#define UWB_DRIVER UWB_ROS_DRIVER
#endif

#if UWB_DRIVER == EVB_DRIVER
#include <evb1000_driver/TagDistance.h>
#elif UWB_DRIVER == MDEK_DRIVER
#include <mdek_uwb_driver/Uwb.h>
#elif UWB_DRIVER == UWB_ROS_DRIVER
#include <uwb_ros/RangeStamped.h>
#endif
#include "uvio/UwbAnchor.h"
#include "uvio/UwbAnchorArrayStamped.h"
#include <ros/ROS1Visualizer.h>
#include <sensor_msgs/Imu.h>
#include <boost/math/interpolators/cardinal_quadratic_b_spline.hpp>

// #include "core/UVioManager.h"

namespace uvio {

// [Andrew] Forward declaring the manager
class UVioManager;

class UVIOROS1Visualizer : public ov_msckf::ROS1Visualizer {

public:
  /**
   * @brief Default constructor
   * @param nh ROS node handler
   * @param app Core estimator manager
   * @param sim Simulator if we are simulating
   */
  UVIOROS1Visualizer(std::shared_ptr<ros::NodeHandle> nh, 
                     std::shared_ptr<UVioManager> app,
                    std::shared_ptr<ov_msckf::Simulator> sim = nullptr);

  /**
   * @brief Wrapper to ov_msckf::ROS!Visualizer::setup_subscribers. Will setup ROS subscribers and callbacks
   * @param parser Configuration file parser
   */
  void setup_subscribers(std::shared_ptr<ov_core::YamlParser> parser);

  /// Callback for inertial information
  void callback_inertial(const sensor_msgs::Imu::ConstPtr &msg);

  /**
   * @brief override of ROS1Visualizer::visualize_odometry() to add anchors tf.
   * This will take the current state estimate and get the propagated pose to the desired time.
   * This can be used to get pose estimates on systems which require high frequency pose estimates.
   */
  void visualize_odometry(double timestamp);

  /// Callback for uwb information
#if UWB_DRIVER == EVB_DRIVER
  void callback_uwb(const evb1000_driver::TagDistanceConstPtr &msg_uwb);
#elif UWB_DRIVER == MDEK_DRIVER
  void callback_uwb(const mdek_uwb_driver::UwbConstPtr &msg_uwb);
#elif UWB_DRIVER == UWB_ROS_DRIVER
  std::pair<double, double> process_range(const uwb_ros::RangeStamped::ConstPtr &msg_uwb);
  void callback_uwb(const uwb_ros::RangeStamped::ConstPtr &msg_uwb);
#endif

  /// Callback for anchors initialization information
  void callback_anchors_init(const uvio::UwbAnchorArrayStampedConstPtr &msg);

  /**
   * @brief Publishes a line between a tag and anchor in Rviz
   * @param tag_id The ID of the tag on the agent
   * @param anchor_id The ID of the fixed anchor
   * @param range The raw distance measured (can be used for label/scaling)
   */
  void visualize_uwb_measurement(size_t tag_id, size_t anchor_id, double range, bool rejected);

  /**
   * @brief Publishes the Global Pose in the UWB aligned frame
   */
  void publish_global_pose();

private:
  /// UWB subscriber
  ros::Subscriber _sub_uwb;
  ros::Subscriber _sub_anchors_init;

  // [Andrew] UWB Visualization Publisher
  ros::Publisher _pub_uwb_viz;

  // [Andrew] Global Pose Publisher, unique to UWB localization example
  ros::Publisher _pub_global_pose;

  /// Core application of the filter system
  std::shared_ptr<UVioManager> _app;
  // decawave time to ns
  static constexpr double _dwt_to_ns = 1e9 * (1.0 / 499.2e6 / 128.0); 
  // speed of light
  static constexpr double _c = 299702547;

  struct SplineGroup{
    // Bias and StdDev Spline Storage
    using QuadSpline = boost::math::interpolators::cardinal_quadratic_b_spline<double>;
    std::unique_ptr<QuadSpline> spline;
    double x0 = 0.0;
    double dx = 0.0;
    size_t n = 0;

    double evaluate(double x) const {
      if(!spline) return 0.0;
      double x_max = x0 + (static_cast<double>(n) - 1.0) * dx;
      double clamped_x = std::max(x0, std::min(x, x_max));
      return spline->operator()(clamped_x);
    }

  };

  
  SplineGroup _bias_spline;
  SplineGroup _std_spline;
  // Antenna Delay storage
  std::map<size_t, double> _uwb_delays;

  void load_spline(const std::string& filename);

  
};

} // namespace uvio

#endif // UVIO_ROS1VISUALIZER_H
