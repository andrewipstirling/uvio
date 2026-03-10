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

#ifndef UVIO_SENSOR_DATA_H
#define UVIO_SENSOR_DATA_H

#include <unordered_map>
#include <Eigen/Eigen>

namespace uvio
{
/**
 * @brief Single UWB measurement between one anchor and one tag
 */
struct UwbMeasurement {
    size_t tag_id;     ///< Which UWB tag on the robot this measurement came from
    size_t anchor_id;  ///< Which UWB anchor in the environment this measurement came from
    double range;      ///< Measured distance

    double std; // [Andrew] StdDev of measurement from fpp spline

    UwbMeasurement() : tag_id(0), anchor_id(0), range(0.0), std(0.0) {}
    UwbMeasurement(size_t t, size_t a, double r) : tag_id(t), anchor_id(a), range(r), std(0.0) {}
    UwbMeasurement(size_t t, size_t a, double r, double sdev) : tag_id(t), anchor_id(a), range(r), std(sdev) {}
};
// TODO: Update UwbData and fix UVioManager, UVIORosVisualizer, 
// TODO: 
/**
 * @brief Struct for a single uwb measurement (time, anchor_id (0,1,2,3,...), distance measurement)
 */
// struct UwbData {

//     /// Timestamp of the reading
//     double timestamp;

//     /// anchor id, distance measurement
//     /// unordered_map is used since the IDs are not ordered in any sense
//     std::unordered_map<size_t, double> uwb_ranges;

//     /// Sort function to allow for using of STL containers
//     bool operator<(const UwbData& other) const {
//         return timestamp < other.timestamp;
//     }
// };


/**
 * @brief Struct for a set of UWB measurements at a single timestamp
 */
struct UwbData {
    double timestamp;                        ///< Timestamp of the reading
    std::vector<UwbMeasurement> uwb_ranges;  ///< All ranges at this time

    /// Sort function to allow for using STL containers (e.g., set)
    bool operator<(const UwbData& other) const {
        return timestamp < other.timestamp;
    }
};

/**
 * @brief Struct for anchors information (id, anchors)
 */
struct AnchorData {

    /// anchor id
    size_t id;

    /// Fixed anchor (no state update)
    bool fix = false;

    /// Biases (y = d + dist_bias * d + const_bias + noise)
    double const_bias = 0;
    double dist_bias = 0;

    /// p_AinG
    Eigen::Vector3d p_AinG;

    /// covariance of the estimation (5x5)
    // [Andrew] Changed for 3x3, biases now managed by UWBBias class
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(5, 5);
};

/**
 * @brief Struct describing a tag attached to the robot body
 */
struct TagData {
    
    /// Tag ID
    size_t id;

    /// Position of the tag relative to the robot body frame origin
    /// Resolved in the body frame of the robot
    // r_tz_b
    Eigen::Vector3d offset;

    // Fix tag, (no state update)
    bool fix = true;

};

}

#endif // UVIO_SENSOR_DATA_H
