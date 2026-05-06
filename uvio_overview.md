# UVIO System Architecture Overview

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

5. Types


C_{ab} is the rotation matrix between frame F_b and frame F_a, such that vectors in F_b are transformed to F_a. The vector r^{tz}_b represents the point t relative to z resolved in F_b