#include "UwbAlignmentRansac.h"
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace uvio {

    RansacResult UwbAlignmentRansac::run(const std::vector<AlignmentConstraint>& constraints,
                                      const RansacConfig& cfg,
                                      uint32_t seed) {
        RansacResult result;
        int N = constraints.size();
        if (N < cfg.min_sample_size) {
            PRINT_WARNING(YELLOW "[UVIOInit-RANSAC] Not enough constraints (%d) for sampling\n" RESET, N);
            return result;
        }

        std::mt19937 rng(seed);

        std::vector<int> all_idx(N);
        std::iota(all_idx.begin(), all_idx.end(), 0);

        int best_inlier_count = -1;
        double best_msac_cost = std::numeric_limits<double>::max();
        double best_yaw = 0.0;
        Eigen::Vector3d best_trans = Eigen::Vector3d::Zero();
        std::vector<int> best_inliers;

        int max_trials = cfg.max_iterations;
        int trial = 0;

        while (trial < max_trials) {
            trial++;

            // --- 1. Draw a random minimal(-ish) sample without replacement ---
            std::vector<int> sample_idx = draw_sample(all_idx, cfg.min_sample_size, rng);
            std::vector<AlignmentConstraint> sample;
            sample.reserve(sample_idx.size());
            for (int idx : sample_idx) sample.push_back(constraints[idx]);

            // --- 2. Degeneracy gate ---
            if (!is_well_conditioned(sample, cfg.condition_number_max)) {
                continue; // don't count against max_trials budget if you prefer; kept simple here
            }

            // --- 3. Hypothesis generation (cheap, closed-form) ---
            auto hyp = CoarseYawSearch::search(sample, cfg.coarse_yaw_samples);
            double yaw = hyp.first;
            Eigen::Vector3d trans = hyp.second;

            // --- 4. Consensus scoring against FULL constraint set ---
            std::vector<int> inliers;
            double msac_cost = 0.0;
            score_hypothesis(constraints, yaw, trans, cfg.inlier_threshold, inliers, msac_cost);

            // --- 5. Keep best (MSAC cost is the primary criterion; inlier count breaks tail cases) ---
            bool better = (msac_cost < best_msac_cost);
            if (better) {
                best_msac_cost = msac_cost;
                best_inlier_count = inliers.size();
                best_yaw = yaw;
                best_trans = trans;
                best_inliers = inliers;

                // --- 6. Adaptive iteration budget update ---
                double eps = 1.0 - static_cast<double>(best_inlier_count) / N; // outlier fraction estimate
                eps = std::min(std::max(eps, 0.0), 0.95); // clamp for numerical sanity
                double s = cfg.min_sample_size;
                double denom = std::log(1.0 - std::pow(1.0 - eps, s));
                if (denom < -1e-12) {
                    int adaptive_N = static_cast<int>(std::ceil(std::log(1.0 - cfg.confidence) / denom));
                    max_trials = std::min(cfg.max_iterations, std::max(adaptive_N, trial));
                }
            }
        }

        if (best_inlier_count < cfg.min_inliers_accept) {
            PRINT_WARNING(YELLOW "[UVIOInit-RANSAC] Best model only has %d/%d inliers, rejecting\n" RESET,
                          best_inlier_count, N);
            return result;
        }

        result.success = true;
        result.yaw = best_yaw;
        result.translation = best_trans;
        result.inlier_indices = best_inliers;
        result.inlier_ratio = static_cast<double>(best_inlier_count) / N;

        PRINT_INFO(GREEN "[UVIOInit-RANSAC] Converged after %d trials: %d/%d inliers (%.1f%%)\n" RESET,
                   trial, best_inlier_count, N, 100.0 * result.inlier_ratio);

        return result;
    }

    // Fisher-Yates partial shuffle for sampling without replacement
    std::vector<int> UwbAlignmentRansac::draw_sample(std::vector<int> pool, int k, std::mt19937& rng) {
        for (int i = 0; i < k; ++i) {
            std::uniform_int_distribution<int> dist(i, pool.size() - 1);
            std::swap(pool[i], pool[dist(rng)]);
        }
        pool.resize(k);
        return pool;
    }

    // Condition number of the centered anchor-position matrix in the sample.
    // Catches collinear/coplanar anchors AND collapsed tag-position baselines
    // by checking both point clouds involved in the sample.
    bool UwbAlignmentRansac::is_well_conditioned(const std::vector<AlignmentConstraint>& sample, double max_cond) {
        if (!check_spread(sample, /*use_anchor=*/true, max_cond)) return false;
        if (!check_spread(sample, /*use_anchor=*/false, max_cond)) return false;
        return true;
    }

    bool UwbAlignmentRansac::check_spread(const std::vector<AlignmentConstraint>& sample, bool use_anchor, double max_cond) {
        int K = sample.size();
        Eigen::MatrixXd pts(K, 3);
        for (int i = 0; i < K; ++i) {
            pts.row(i) = use_anchor ? sample[i].p_AinG.transpose() : sample[i].p_TinV.transpose();
        }
        pts.rowwise() -= pts.colwise().mean();

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(pts, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd sv = svd.singularValues();

        // Guard divide-by-zero: a truly degenerate (rank-deficient) cluster
        // will have a ~0 smallest singular value -> treat as infinite condition number
        if (sv(sv.size() - 1) < 1e-6) return false;

        double condition = sv(0) / sv(sv.size() - 1);
        return condition < max_cond;
    }

    // MSAC scoring: capped squared residual, summed over the full set.
    // Also returns the inlier index list (residual within threshold).
    void UwbAlignmentRansac::score_hypothesis(const std::vector<AlignmentConstraint>& constraints,
                                  double yaw, const Eigen::Vector3d& trans,
                                  double threshold, std::vector<int>& inliers, double& msac_cost) {
        Eigen::Matrix3d C_av;
        C_av << std::cos(yaw), -std::sin(yaw), 0.0,
                std::sin(yaw),  std::cos(yaw), 0.0,
                0.0,            0.0,           1.0;

        double tau2 = threshold * threshold;
        msac_cost = 0.0;
        inliers.clear();

        for (int k = 0; k < static_cast<int>(constraints.size()); ++k) {
            const auto& c = constraints[k];
            Eigen::Vector3d p_TinG = C_av * c.p_TinV + trans;
            double predicted_range = (p_TinG - c.p_AinG).norm();
            double e = (c.range - predicted_range) / c.std_range;
            double e2 = e * e;

            if (e2 < tau2) {
                inliers.push_back(k);
                msac_cost += e2;
            } else {
                msac_cost += tau2; // capped contribution for outliers (MSAC, not pure RANSAC count)
            }
        }
    }

} // namespace uvio