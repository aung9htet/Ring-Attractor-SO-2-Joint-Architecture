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

# Catkin's Python relay files do not export symbols for top-level py_modules.
# The legacy flat modules therefore remain importable in linked devel spaces by
# placing their unchanged source directory ahead of the generated relays.  The
# hook is devel-only; install spaces use the normal distutils installation.
catkin_add_env_hooks(
  50.tiago_ring_controller_pythonpath
  SHELLS sh
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks
  SKIP_INSTALL
)

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

catkin_install_python(
  PROGRAMS
    src/analysis_single_joint.py
    src/calibrate_single_joint.py
    src/compositional_fourier_decoder.py
    src/demo_graphs.py
    src/experiment_camera.py
    src/gain_modulation_analysis.py
    src/multi_ring_component.py
    src/multi_ring_sawtooth_scalar_decoder.py
    src/readout_mean_std_analysis.py
    src/record_experiments.py
    src/ring_attractor.py
    src/single_joint_data_collector.py
    src/single_joint_data_visualizer.py
    src/single_joint_pid_data_collector.py
    src/single_ring.py
    src/train_homeostasis.py
    src/train_ring_model.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(
  PROGRAMS
    src/builders_analysis/ring_attractor_analysis.py
    src/builders_analysis/single_joint_ring_component_analysis.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}/builders_analysis
)

catkin_install_python(
  PROGRAMS
    src/trainer/train_ring_model.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}/trainer
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
