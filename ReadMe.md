# LOCAL-UVIO: UWB-Aided Visual-Inertial Odometry Framework for Localization against a prior UWB map. 

This library defines a multi-sensor framework that leverages Ultra Wideband (UWB) technology and Visual-Inertial Odometry (VIO) to provide robust and low-drift localization. The whole project is built on top of OpenVINS and [academic paper], while enhancing it to explicitly support localization against prior UWB maps. 

## Usage

To use LOCAL-UVIO, you can refer to the OpenVINS documentation for everything except the UWB-related features. It requires UWB range measurements to be available to the system. This repo supports either the [MDEK1001](https://www.qorvo.com/products/p/MDEK1001) series or the [DW1000](https://forum.qorvo.com/uploads/default/original/1X/8b220e1e26fea4ebd83f0b0e5ef42eb9a251310d.pdf) APIs. Support is selected at build under the `./uvio/cmake/ROS1.cmake` file, with specific message types for the DW100 series defined in the dependency [uwb_ros](https://github.com/decargroup/miluv/tree/main/uwb_ros). 

The framework requires the position of UWB anchors to operate. You have two options for providing this information:

1. UWB Anchors Config File (uwb_anchors.yaml): you can specify the positions and confidence of the UWB anchors in the global reference frame in the 
uwb_anchors.yaml configuration file. The transform between the global UWB anchor reference frame and the local VIO frame is estimated internally.

2. You can obtain a map of the UWB anchors using either a motion capture system or a SLAM procedure as shown in [academic paper]. Alternatively, you can initialize unknown UWB anchors following the steps of [initialization library](https://github.com/aau-cns/uwb_init).

## LOCAL-UVIO Architecture Overview
The UVIO package is an extension of the OpenVINS framework, specifically designed to fuse UWB range measurements with visual and inertial data.

1. Core Manager Classes \
`core/UVioManager`
- Inherits from `ov_msckf::VioManger`. Handles data feed of the IMU, Camera, and UWB, and the main state machine. \
`core/UVioManagerOptions`
- Stores parameters loaded from `.yaml` files. 

2. State and Propagation \
`state/UVioState`
- Inherits from `ov_msckf::State`. Contains the base IMU state, and augmented camera/UWB states.
- Hold the UWB extrinsic map, and anchor map. \
`state/UVioPropagator`
- Performs IMU propagation. 

3. Update Classes \
`update/UpdaterUWB`
- Holds the measurement model, residual and outlier rejection computation is done here for UWB measurements. \
`update/UVioUpdaterHelper`
- Jacobian matrix allocation and computation. 

4. Visualization and Tools \
`ros/UVIOROS1Visualizer`
- Interfaces the internal C++ state with ROS/Rviz. \

5. UWB-VIO Alignment Initialization \
`initialization/UVioInitializer`
- Used in localization mode, performs initialization of the frame alignment transform between the local VIO frame and global UWB map. \

6. Simulation Tools \
`sim/UVIOSimulator`
- IN DEVELOPMENT: Inherits from `ov_core::BSplineSE3`. Used for simulated IMU generation, along with camera and UWB measurements. \



<!-- LINKS: -->
[academic paper]: https://arxiv.org/abs/2308.00513

