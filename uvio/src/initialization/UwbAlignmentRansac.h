// UwbAlignmentRansac.h
#include "core/VioManager.h"
#include <vector>
#include <Eigen/Dense>
#include "AlignmentConstraint.h"
#include "CoarseYawSearch.h"
#include <random>
#include <numeric>
#include <algorithm>

namespace uvio {

struct RansacConfig {
    int min_sample_size = 6;
    int max_iterations = 500;
    double confidence = 0.99;
    double inlier_threshold = 2.7;
    double condition_number_max = 50.0;
    int coarse_yaw_samples = 12;
    int min_inliers_accept = 8;
};

struct RansacResult {
    bool success = false;
    double yaw = 0.0;
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    std::vector<int> inlier_indices;
    double inlier_ratio = 0.0;
};

class UwbAlignmentRansac {
public:
    static RansacResult run(const std::vector<AlignmentConstraint>& constraints,
                             const RansacConfig& cfg,
                             uint32_t seed = std::random_device{}());

private:
    static std::vector<int> draw_sample(std::vector<int> pool, int k, std::mt19937& rng);
    static bool is_well_conditioned(const std::vector<AlignmentConstraint>& sample, double max_cond);
    static bool check_spread(const std::vector<AlignmentConstraint>& sample, bool use_anchor, double max_cond);
    static void score_hypothesis(const std::vector<AlignmentConstraint>& constraints,
                                  double yaw, const Eigen::Vector3d& trans,
                                  double threshold, std::vector<int>& inliers, double& msac_cost);
};

} // namespace uvio