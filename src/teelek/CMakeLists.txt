################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.8)
project(teelek)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

################################################################################
# Find ament packages and libraries for ament and system dependencies
################################################################################
find_package(ament_cmake REQUIRED)
# find_package(tf2 REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
# find_package(OpenCV REQUIRED)
# find_package(cv_bridge REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
# find_package(tf2_geometry_msgs REQUIRED)


################################################################################
# Install
################################################################################
install(
  DIRECTORY launch src config
  DESTINATION share/${PROJECT_NAME}
)

################################################################################
# Python package installation
################################################################################
# ament_python_install_package(${PROJECT_NAME})

################################################################################
# Install Python Programs
################################################################################
install(PROGRAMS
  src/xbox_control.py
  src/play_control.py
  src/robot_movement.py
  src/vision_control.py
  src/camera_april_tag_calibrate.py
  src/mission_node.py
  src/motion_node.py
  src/plant_node.py
  src/return_to_origin_node.py

  DESTINATION lib/${PROJECT_NAME}
)

# install(PROGRAMS
#   src/controller.py
#   src/utilize.py
#   DESTINATION lib/${PROJECT_NAME}/src
# )

################################################################################
# TensorFlow Lite
################################################################################

# Adjust the following to match your path (if you didn't install to /usr/local)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

################################################################################
# Install C++ Programs
################################################################################

# add_executable(apriltag_auto_aim scripts_camera/apriltag_auto_aim.cpp)

# ament_target_dependencies(apriltag_auto_aim 
#   rclcpp
#   sensor_msgs 
#   geometry_msgs 
#   cv_bridge
#   OpenCV
# )


# target_link_libraries(apriltag_auto_aim ${OpenCV_LIBS} apriltag)

install(TARGETS
#   apriltag_auto_aim
  DESTINATION lib/${PROJECT_NAME}
)


################################################################################

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

################################################################################
# Macro for ament package
################################################################################
ament_package()