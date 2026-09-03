#ifndef UVIO_INITIALIZER_H
#define UVIO_INITIALIZER_H

#include "core/VioManager.h"
#include "UwbAlignmentRansac.h"
#include "AlignmentConstraint.h"
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <random>
#include <numeric>
#include <algorithm>

namespace uvio {



class UVioInitializer {
    public:
        UVioInitializer(double cauchy_loss_sigma = 0.5) : _cauchy_sigma(cauchy_loss_sigma) {}

        // Adds batch of constraints for a specific timestamp
        void add_measurements(const std::vector<AlignmentConstraint>& batch);

        // Calls ceres optimization
        bool solve(Eigen::Matrix3d& C_av_out, Eigen::Vector3d& r_wa_wv_out, Eigen::Matrix4d& cov_out, bool use_initial_guess, bool use_squared_cost);

        // Ransac solve
        bool solve_robust(Eigen::Matrix3d& C_av_out, Eigen::Vector3d& r_wa_wv_out, Eigen::Matrix4d& cov_out, const RansacConfig& ransac_cfg);
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



} // namespace uvio
#endif