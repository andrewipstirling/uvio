#ifndef UVIO_INITIALIZER_H
#define UVIO_INITIALIZER_H

#include "core/VioManager.h"
#include <Eigen/Dense>
#include <vector>
#include <map>

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
};

class UVioInitializer {
    public:
        UVioInitializer(double cauchy_loss_sigma = 0.5) : _cauchy_sigma(cauchy_loss_sigma) {}

        // Adds batch of constraints for a specific timestamp
        void add_measurements(const std::vector<AlignmentConstraint>& batch);

        // Calls ceres optimization
        bool solve(Eigen::Matrix3d& C_av_out, Eigen::Vector3d& r_wa_wv_out, Eigen::Matrix4d& cov_out, bool use_initial_guess, bool use_squared_cost);

        // Helpers
        size_t data_count() const {return _constraints.size();}
        void clear() {_constraints.clear();}

        bool accept_measurement(const AlignmentConstraint& c);
        
        Eigen::MatrixXd build_G(const std::vector<AlignmentConstraint>& data);  
        
        double compute_pdop(const Eigen::MatrixXd& G);

        double check_fim_observability(const Eigen::Matrix3d& C_av, const Eigen::Vector3d& r_wa_wv);

        bool can_initialize(const double distance, const double min_distance, const double min_measurements, const double min_pdop); 

        bool can_refine(const Eigen::Matrix3d& C_av, const Eigen::Vector3d& r_wa_wv, const double distance, const double min_distance, const double min_measurements, const double min_pdop, const double min_eigenvalue); 



    private:
        std::vector<AlignmentConstraint> _constraints;

        double _cauchy_sigma;

        Eigen::Vector4d compute_initial_guess();
};

} // namespace uvio
#endif