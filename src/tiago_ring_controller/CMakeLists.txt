cmake_minimum_required(VERSION 2.8.3)
project(tiago_ring_controller)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  cv_bridge
  geometry_msgs
  play_motion_msgs
  rospy
  sensor_msgs
  std_msgs
  trajectory_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    actionlib
    cv_bridge
    geometry_msgs
    play_motion_msgs
    rospy
    sensor_msgs
    std_msgs
    trajectory_msgs
)

# Executables: the co-simulation scripts.  The flat research scripts under
# legacy/ are frozen and not installed; the wrist force/torque logger started by
# tiago_ring_controller.launch is the one exception.
catkin_install_python(
  PROGRAMS
    scripts/cosim_gazebo_smoke.py
    scripts/measure_legacy_tick.py
    scripts/run_cosim_trial.py
    legacy/record_experiments.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

foreach(dir launch worlds)
  install(DIRECTORY ${dir}/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach()

install(
  DIRECTORY src/config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  find_package(PythonInterp REQUIRED)

  set(unit_test_command
    "\"${CMAKE_COMMAND}\" -E env PYTHONDONTWRITEBYTECODE=1 \"${PYTHON_EXECUTABLE}\" \"${CMAKE_CURRENT_SOURCE_DIR}/test/unit_tests_runner.py\" --gtest_output=xml:${CATKIN_TEST_RESULTS_DIR}/${PROJECT_NAME}/unittest-python.xml"
  )
  catkin_run_tests_target(
    "unittest"
    "python"
    "unittest-python.xml"
    COMMAND ${unit_test_command}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_rostest(test/ros_interfaces.test)
endif()
