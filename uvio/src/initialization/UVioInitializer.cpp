#include "initialization/UVioInitializer.h"
#include <ceres/ceres.h>

namespace uvio {

    // Ceres for 4dof alignment
    struct UWBAlignmentError {
        
        UWBAlignmentError( const AlignmentConstraint& c) : _c(c) {}
        
        template <typename T>
        bool operator()(const T* const yaw, const T* const translation, T* residual) const {

            T cos_y = ceres::cos(yaw[0]);
            T sin_y = ceres::sin(yaw[0]);

            Eigen::Matrix<T, 3, 3> R_av;
            R_av << cos_y, -sin_y, T(0),
                    sin_y,  cos_y, T(0),
                    T(0),   T(0),  T(1);

            // Global frame
            Eigen::Matrix<T, 3, 1> p_offset = Eigen::Matrix<T, 3, 1>(translation[0], translation[1], translation[2]);
            Eigen::Matrix<T, 3, 1> p_Tg = R_av * _c.p_TinV + p_offset;

            Eigen::Matrix<T, 3, 1> diff = p_Tg - _c.p_AinG.cast<T>();
            T pred_range = diff.norm();
            T res = T(_c.range) - pred_range;

            // Weighted Least Squares Logic
            Eigen::Matrix<T, 3, 1> u = diff / pred_range; 

            // Jacobian of range w.r.t VIO Pose
            // J_rot = u^T * R_av * C_vb * skew(r_tz_b)
            // J_pos = u^T * R_av
            Eigen::Matrix<T, 1, 3> J_rot = u.transpose() * R_av * _c.C_bv.transpose().cast<T>() * ov_core::skew_x(_c.p_tz_b).cast<T>();
            Eigen::Matrix<T, 1, 3> J_pos = u.transpose() * R_av;

            Eigen::Matrix<T, 1, 6> J_vio;
            J_vio << J_rot, J_pos;

            // Covariance Projection
            T var_vio = (J_vio * _c.P_vio.cast<T>() * J_vio.transpose())(0,0);
            T var_uwb = T(_c.std_range * _c.std_range);
            T sigma_total = ceres::sqrt(var_uwb + var_vio);

            // Weighted Residual
            residual[0] = res / sigma_total;
            
            // Un-Weighted below
            // // Transform to Global Frame 
            // T p_Tg[3];
            // p_Tg[0] = cos_y * T(_c.p_TinV.x()) - sin_y * T(_c.p_TinV.y()) + translation[0];
            // p_Tg[1] = sin_y * T(_c.p_TinV.x()) + cos_y * T(_c.p_TinV.y()) + translation[1];
            // p_Tg[2] = T(_c.p_TinV.z()) + translation[2];

            // // Compute residual
            // T dx = p_Tg[0] - T(_c.p_AinG.x());
            // T dy = p_Tg[1] - T(_c.p_AinG.y());
            // T dz = p_Tg[2] - T(_c.p_AinG.z());
            // residual[0] = T(_c.range) - ceres::sqrt(dx*dx + dy*dy + dz*dz);

            return true;
        }

        static ceres::CostFunction* Create(const AlignmentConstraint& c) {
            return (new ceres::AutoDiffCostFunction<UWBAlignmentError, 1, 1, 3>(new UWBAlignmentError(c)));
        }

        AlignmentConstraint _c;
    };

    struct UWBAlignmentErrorSquared {
        UWBAlignmentErrorSquared( const AlignmentConstraint& c) : _c(c) {}

        template <typename T>
        bool operator()(const T* const yaw, const T* const translation, T* residual) const {
            T cos_y = ceres::cos(yaw[0]);
            T sin_y = ceres::sin(yaw[0]);

            Eigen::Matrix<T, 3, 3> R_av;
            R_av << cos_y, -sin_y, T(0),
                    sin_y,  cos_y, T(0),
                    T(0),   T(0),  T(1);

            // Global frame
            Eigen::Matrix<T, 3, 1> p_offset = Eigen::Matrix<T, 3, 1>(translation[0], translation[1], translation[2]);
            Eigen::Matrix<T, 3, 1> p_tag_a = R_av * _c.p_TinV + p_offset;

            Eigen::Matrix<T, 3, 1> diff = p_tag_a - _c.p_AinG.cast<T>();
            T pred_range_squared = diff.squaredNorm();
            T res = T(_c.range * _c.range) - pred_range_squared;
            // TODO: Input normalization and std_dev weighting
            // T denom = T(2.0 * _c.range * _c.std_range);
            T denom = T(_c.std_range);
            residual[0] = res / denom;
            return true;
        }

        static ceres::CostFunction* Create(const AlignmentConstraint& c) {
            return new ceres::AutoDiffCostFunction<UWBAlignmentErrorSquared, 1, 1, 3>(new UWBAlignmentErrorSquared(c));
        }
        AlignmentConstraint _c;
    };
    void UVioInitializer::add_measurements(const std::vector<AlignmentConstraint>& batch) {
        _constraints.insert(_constraints.end(), batch.begin(), batch.end());
    }

    // Set initial yaw = 0 
    // Sets initial translation = difference of centroids
    Eigen::Vector4d UVioInitializer::compute_initial_guess() {
        Eigen::Vector3d mean_vio = Eigen::Vector3d::Zero();
        Eigen::Vector3d mean_uwb = Eigen::Vector3d::Zero();
        for (const auto& c : _constraints) {
            mean_vio += c.p_TinV;
            mean_uwb += c.p_AinG;
        }
        mean_vio /= (double)_constraints.size();
        mean_uwb /= (double)_constraints.size();

        Eigen::Vector4d x0;
        x0(0) = 0.0; // Initial Yaw
        x0.tail<3>() = mean_uwb - mean_vio; // Initial Translation
        return x0;
    }

    bool UVioInitializer::solve(Eigen::Matrix3d& C_av_out, Eigen::Vector3d& r_wa_wv_out, Eigen::Matrix4d& cov_out, bool use_initial_guess, bool use_squared_cost){
        double yaw;
        double trans[3];
        if (use_initial_guess) {
            yaw = atan2(C_av_out(1,0), C_av_out(0,0));
            trans[0] = r_wa_wv_out(0);
            trans[1] = r_wa_wv_out(1);
            trans[2] = r_wa_wv_out(2);
        }
        else {
            Eigen::Vector4d x0 = compute_initial_guess();
            yaw = x0(0);
            trans[0] = x0[1], trans[1] = x0[2], trans[2] = x0[3];
        }
        
        ceres::Problem problem;
        ceres::LossFunction* loss_function = new ceres::CauchyLoss(_cauchy_sigma);
        // ceres::LossFunction* loss_function = new ceres::TrivialLoss();

        for (const auto& c : _constraints) {
            if (use_squared_cost) {
                ceres::CostFunction* cost_function = UWBAlignmentErrorSquared::Create(c);
                problem.AddResidualBlock(cost_function, loss_function, &yaw, trans);
                }

            else {
                ceres::CostFunction* cost_function = UWBAlignmentError::Create(c);
                problem.AddResidualBlock(cost_function, loss_function, &yaw, trans);
            }
            
        }

        // Setup solver options
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 100;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (summary.termination_type == ceres::CONVERGENCE || summary.termination_type == ceres::USER_SUCCESS) {
            // Compute associated Covariance blocks
            ceres::Covariance::Options cov_options;
            ceres::Covariance covariance(cov_options);

            // Want full covariance between yaw and translation
            std::vector<std::pair<const double*, const double*>> covariance_blocks;
            covariance_blocks.push_back(std::make_pair(&yaw, &yaw));
            covariance_blocks.push_back(std::make_pair(&yaw, trans));
            covariance_blocks.push_back(std::make_pair(trans, trans));
            if (covariance.Compute(covariance_blocks, &problem)) {
                // Extract the 4x4 covariance matrix (1 for yaw, 3 for translation)
                // Ceres stores them in row-major arrays
                double cov_raw[4 * 4];
                // We need a list of all parameters in order to get the full block
                std::vector<const double*> all_params = {&yaw, trans};
                covariance.GetCovarianceMatrix(all_params, cov_raw);
                
                cov_out = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(cov_raw);
            } else {
                // Fallback to a default if computation fails (e.g., non-observable)
                cov_out = Eigen::Matrix4d::Identity() * 0.1; 
            }


            C_av_out << ceres::cos(yaw), -ceres::sin(yaw), 0, 
                        ceres::sin(yaw),  ceres::cos(yaw), 0, 
                        0,                0,               1;
            r_wa_wv_out = Eigen::Vector3d(trans[0], trans[1], trans[2]);
            PRINT_INFO(GREEN "[UVIO] VIO-UWB frames aligned, yaw = [%.3f] | offset = [%.3f, %.3f, %.3f], "
            "confidence = [%.4f, %.4f, %.4f, %.4f] \n" RESET, 
                yaw, 
                r_wa_wv_out(0), r_wa_wv_out(1), r_wa_wv_out(2), 
                cov_out(0,0), cov_out(1,1), cov_out(2,2), cov_out(3,3));
            
            return true;
        }
        return false;

    }

    bool UVioInitializer::accept_measurement(const AlignmentConstraint& c){
        
        // Always accept first measurements
        if (_constraints.size() < 5) return true;

        // Build temporary dataset including most recent
        std::vector<AlignmentConstraint> temp = _constraints;
        temp.push_back(c);

        // Compute PDOP before
        Eigen::MatrixXd G_old = build_G(_constraints);
        double pdop_old = compute_pdop(G_old);

        // Compute PDOP after
        Eigen::MatrixXd G_new = build_G(temp);
        double pdop_new = compute_pdop(G_new);

        // Accept if PDOP improves
        return (pdop_new < pdop_old);
    }

    Eigen::MatrixXd UVioInitializer::build_G(const std::vector<AlignmentConstraint>& data) {
        // Build the PDOP matrix G
        int N = data.size();
        Eigen::MatrixXd G(N,3);

        // Compute Mean Position
        // Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        // for (const auto& m : data)
        //     mean += m.p_TinV;
        // mean /= N;

        for (int i = 0; i < N; i++) {
            Eigen::Vector3d diff = data[i].p_TinV - data[i].p_AinG;

            if (diff.norm() < 1e-6) diff = Eigen::Vector3d(1,0,0); // fallback

            G.row(i) = diff.normalized().transpose();
        }
        return G;
    }

    double UVioInitializer::compute_pdop(const Eigen::MatrixXd& G) {
        Eigen::Matrix3d GTG = G.transpose() * G;
        // Guard against 
        if (GTG.determinant() < 1e-9)
            return std::numeric_limits<double>::infinity();
        
        return std::sqrt(GTG.inverse().trace());
    }

    double UVioInitializer::check_fim_observability(const Eigen::Matrix3d& C_av, const Eigen::Vector3d& r_wa_wv){
        if (data_count() < 4)
            return 0.0;
        // TODO: Finish this
        Eigen::Matrix4d F = Eigen::Matrix4d::Zero();
        for (const auto& c: _constraints) {
            Eigen::Vector3d d_k = C_av * c.p_TinV + r_wa_wv - c.p_AinG;
            double norm_d = d_k.norm();
            if (norm_d < 1e-6) continue; // No need to bother

            Eigen::Vector3d u_k = d_k / norm_d;

            // Finish this !!
            Eigen::RowVector4d H_k;
            H_k.block<1,3>(0,1) = u_k.transpose();
            
            Eigen::Vector3d unit_z(0, 0, 1);
            double yaw_jac = (-1.0 * u_k.transpose() * C_av * ov_core::skew_x(c.p_TinV) * unit_z)(0);
            H_k(0) = yaw_jac;

            // TODO: Add variance from pose
            Eigen::RowVector3d J_rot = u_k.transpose() * C_av * c.C_bv.transpose() * ov_core::skew_x(c.p_tz_b);
            Eigen::RowVector3d J_pos = u_k.transpose() * C_av;
            Eigen::Matrix<double, 1, 6> J_vio;
            J_vio << J_rot, J_pos;
            double var_vio = (J_vio * c.P_vio * J_vio.transpose())(0,0);
            double var_uwb = c.std_range * c.std_range;
            F += (H_k.transpose() * H_k) / (var_uwb + var_vio);
        }
        double trace = F.trace();
        if (trace < 1e-8) return 0.0;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eig(F);
        double min_eval = eig.eigenvalues()(0);
        // Eigenvalues go up as more data is added
        double weighted_min_eval = min_eval / trace;
        // Non-weighted version
        return min_eval;
    }

    bool UVioInitializer::can_initialize(
                              const double distance,
                              const double min_distance,
                              const double min_measurements,
                              const double min_pdop){
        Eigen::MatrixXd G_cur = build_G(_constraints);
        double cur_pdop = compute_pdop(G_cur);
        PRINT_INFO(GREEN "[UVIO] Current PDOP of UWB initialization problem %3f with %d constraints\n" RESET, cur_pdop, data_count());
        if (distance > min_distance &&
            data_count() > min_measurements &&
            cur_pdop < min_pdop){
                return true;
            }
        else return false;
    }

    bool UVioInitializer::can_refine(const Eigen::Matrix3d& C_av, const Eigen::Vector3d& r_wa_wv, const double distance, const double min_distance, const double min_measurements, const double min_pdop, const double min_eigenvalue) {
        bool can_init = can_initialize(distance, min_distance, min_measurements, min_pdop);
        double cur_min_eigenvalue = check_fim_observability(C_av, r_wa_wv);
        PRINT_INFO(GREEN "[UVIO] Current FIM minimum eigenvalue of UWB initialization problem %3f with %d constraints\n" RESET, cur_min_eigenvalue, data_count());
        if (cur_min_eigenvalue > min_eigenvalue && can_init) return true;

        else return false;
    }

}