#pragma once

#include "sim/Simulator.h"              // Base simulator (OpenVINS)
#include "core/UVioManager.h"           // To feed measurements to UVioManager
#include "update/UpdaterUWB.h"          // For simulating UWB updates
#include "state/UVioState.h"            // State structure for UVIO
#include "utils/uvio_sensor_data.h"     // If you need sensor data helpers
#include <memory>                        // For std::shared_ptr
#include <vector> 
#include <Eigen/Eigen>
// #include <ros/ros.h>

namespace uvio {

class UVIOSimulator: public ov_msckf::Simulator {
    public:
        /**
         * @brief Default constructor, will load all configuration variables
         * @param params_ UVioManager parameters. Should have already   been loaded from cmd.
         */
        UVIOSimulator(uvio::UVioManagerOptions& params_);
        ~UVIOSimulator();

        /**
         * OverWriting the Simulator version for UWB params
         * @brief Will get a set of perturbed parameters
         * @param gen_state Random number gen to use
         * @param params_ Parameters we will perturb
         */
        static void perturb_parameters(std::mt19937 gen_state, UVioManagerOptions &params_);

        
        /**
         * OverWriting the Simulator version for checking UWB
         * @brief Returns if we are actively simulating
         * @return True if we still have simulation data
         */
        bool ok() { return is_running; }

        /**
         * Over Writing the Simulator version that checks UWB measurements
         * @brief Gets the timestamp we have simulated up too
         * @return Timestamp
         */
        double current_timestamp() { return timestamp; }

        /**
         * @brief Gets the next UWB reading if we have one.
         * @param time_uwb Time that this measurement occured at
         * @param anchor_ids Camera ids that the corresponding vectors match
         * @param feats Noisy uv measurements and ids for the returned time
         * @return True if we have a measurement
         */
        bool get_next_uwb(double &time_cam, std::vector<int> &camids, std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats);



        

};
} // namespace uvio