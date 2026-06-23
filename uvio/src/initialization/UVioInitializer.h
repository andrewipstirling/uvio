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
        // Solving the unconstrained squared-range least squares method of 
        // Beck et. al 2008
        // Build rotation matrix
        Eigen::Matrix3d C_av;
        C_av << std::cos(yaw), -std::sin(yaw), 0.0,
                std::sin(yaw),  std::cos(yaw), 0.0,
                0.0,            0.0,           1.0;
        
        int K = constraints.size();
        Eigen::MatrixXd A(K, 4);
        Eigen::VectorXd b(K);
        
        // Build A matrix (K x 4) and b vector
        // USR-LS formulation: argmin_x ||A*x - b||^2, where x = [r^T, ||r||^2]^T
        for (int k = 0; k < K; ++k) {
            const auto& c = constraints[k];
            double w = 1.0 / (c.std_range * c.std_range);
            
            // a_k = C_av * p_TinV - p_AinG
            Eigen::Vector3d a_k = C_av * c.p_TinV - c.p_AinG;
            
            // Row k of A: [2*a_k^T, 1]
            A.block<1, 3>(k, 0) = 2.0 * w * a_k.transpose();
            A(k, 3) = w;
            
            // b_k = w * (y_k^2 - ||a_k||^2)
            b(k) = w * (c.range * c.range - a_k.squaredNorm());
        }
        
        // Solve (A^T*A) * x = A^T * b
        Eigen::VectorXd x = (A.transpose() * A).ldlt().solve(A.transpose() * b);
        
        // Extract translation (first 3 elements of x)
        return x.head<3>();
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