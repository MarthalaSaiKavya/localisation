cmake_minimum_required(VERSION 3.0.2)
project(pf_localization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  tf
)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install Python nodes
catkin_install_python(PROGRAMS
  scripts/motion_model_node.py
  scripts/likelihood_field_node.py
  scripts/particle_filter_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
