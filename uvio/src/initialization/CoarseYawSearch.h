#include <vector>
#include <limits>
#include <cmath>
#include <Eigen/Dense>
#include "AlignmentConstraint.h"

namespace uvio {

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