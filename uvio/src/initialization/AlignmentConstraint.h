#pragma once
#include <Eigen/Dense>

namespace uvio {

struct AlignmentConstraint {
  double range; // Range measurement
  Eigen::Vector3d p_TinV; // Tag position in VIO frame (C_vb * r_pz_b + r_zw_v)
  Eigen::Vector3d p_AinG; // Anchor position in Global UWB fram (r_lw_a)
  double std_range; // Std deviation on UWB meas

  Eigen::Matrix3d C_bv; // Heading for jacobian
  Eigen::Vector3d p_tz_b; // Tag position relative to IMU in body frame (for Jacobian)
  Eigen::Vector3d p_zw_v; // IMU position for jacobian
  Eigen::Matrix<double, 6, 6> P_vio; // Pose covariance
  
  Eigen::Matrix3d P_anc; // Anchor Covariance
};

} // namespace uvio