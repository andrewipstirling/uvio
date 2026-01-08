cmake_minimum_required(VERSION 3.0.2)
project(py_vins)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS rospy std_msgs
)

###########
## Build ##
###########

## Install Python scripts so rosrun can find them
catkin_install_python(PROGRAMS
  scripts/run_ov.py
  scripts/run_uvio.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#############
## Install ##
#############

## Optionally install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
