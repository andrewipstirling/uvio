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

        bool can_initialize(const double distance, const double min_distance, const double min_measurements, const double max_pdop); 

        bool can_refine(const Eigen::Matrix3d& C_av, const Eigen::Vector3d& r_wa_wv, const double distance, const double min_distance, const double min_measurements, const double min_pdop, const double min_eigenvalue); 



    private:
        std::vector<AlignmentConstraint> _constraints;

        double _cauchy_sigma;

        Eigen::Vector4d compute_initial_guess();
};

struct CoarseYawSearch {
    static std::pair<double, Eigen::Vector3d> search(
        const std::vector<AlignmentConstraint>& constraints,
        int num_yaw_samples = 36) {
        
        double best_yaw = 0.0;
        Eigen::Vector3d best_translation = Eigen::Vector3d::Zero();
        double best_cost = std::numeric_limits<double>::max();
        
        // Grid search over yaw angles
        for (int i = 0; i < num_yaw_samples; ++i) {
            // From -pi to pi
            double yaw = -M_PI + (2.0 * M_PI * i) / num_yaw_samples;
            
            // For this yaw, solve for optimal translation using linear least squares
            Eigen::Vector3d translation = solve_translation_at_yaw(constraints, yaw);
            
            // Evaluate cost at this (yaw, translation) pair
            double cost = evaluate_cost(constraints, yaw, translation);
            
            if (cost < best_cost) {
                best_cost = cost;
                best_yaw = yaw;
                best_translation = translation;
            }
        }
        
        return {best_yaw, best_translation};
    }
    
private:
    // Solve for translation given fixed yaw using linear least squares
    static Eigen::Vector3d solve_translation_at_yaw(
        const std::vector<AlignmentConstraint>& constraints,
        double yaw) {
        
        // Build rotation matrix
        Eigen::Matrix3d C_av;
        C_av << std::cos(yaw), -std::sin(yaw), 0.0,
                std::sin(yaw),  std::cos(yaw), 0.0,
                0.0,            0.0,           1.0;
        
        // For squared-range formulation, we can linearize:
        // y_k^2 = ||C_av * p_TinV_k + r_wa_wv - p_AinG_k||^2
        // Expanding: y_k^2 = ||C_av * p_TinV_k - p_AinG_k||^2 + 2*(C_av*p_TinV_k - p_AinG_k)^T * r_wa_wv + ||r_wa_wv||^2
        // This is quadratic in r_wa_wv, but we can solve iteratively or use pseudoinverse method
        
        // Alternative: Direct range approach with iterative refinement
        // For initialization, we'll use a weighted centroid approach
        
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        
        for (const auto& c : constraints) {
            Eigen::Vector3d p_TinG_rotated = C_av * c.p_TinV;
            double weight = 1.0 / (c.std_range * c.std_range);
            
            // Use squared-range linear approximation around current estimate
            // Build normal equations: A * r_wa_wv = b
            Eigen::Vector3d diff = p_TinG_rotated - c.p_AinG;
            double current_dist = diff.norm();
            
            if (current_dist > 1e-6) {
                Eigen::Vector3d direction = diff / current_dist;
                double residual = current_dist - c.range;
                
                A += weight * direction * direction.transpose();
                b += weight * residual * direction;
            }
        }
        
        // Solve A * r_wa_wv = -b (we want to reduce residual)
        Eigen::Vector3d translation = -A.ldlt().solve(b);
        
        return translation;
    }
    
    // Evaluate total cost (sum of squared range residuals)
    static double evaluate_cost(
        const std::vector<AlignmentConstraint>& constraints,
        double yaw,
        const Eigen::Vector3d& translation) {
        
        Eigen::Matrix3d C_av;
        C_av << std::cos(yaw), -std::sin(yaw), 0.0,
                std::sin(yaw),  std::cos(yaw), 0.0,
                0.0,            0.0,           1.0;
        
        double total_cost = 0.0;
        for (const auto& c : constraints) {
            Eigen::Vector3d p_TinG = C_av * c.p_TinV + translation;
            double predicted_range = (p_TinG - c.p_AinG).norm();
            double residual = (c.range - predicted_range) / c.std_range;
            total_cost += residual * residual;
        }
        
        return total_cost;
    }
};

} // namespace uvio
#endif