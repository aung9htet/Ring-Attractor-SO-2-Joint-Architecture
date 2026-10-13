# ROS 2 Humble + MuJoCo + PAL Kangaroo simulation, all from ROS apt binaries.
# Base pinned by digest (osrf/ros:humble-desktop, built 2026-09-10).
FROM osrf/ros@sha256:fb07245b32187d74350be25323d8ad2f8ca5c25c325759911a1eff2267a49c1e

# kangaroo-simulation pulls kangaroo_mujoco, kangaroo_robot, pal_sea_arm, mujoco_ros2_control, mujoco_vendor (MuJoCo).
# kangaroo-moveit-config: PAL's launch resolves it even with moveit:=False but does not declare it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ros-humble-kangaroo-simulation ros-humble-kangaroo-moveit-config \
    && rm -rf /var/lib/apt/lists/*

COPY kangaroo_sim.launch.py /opt/kangaroo_sim.launch.py

# Every interactive shell (including `docker exec`) gets ROS sourced.
RUN echo 'source /opt/ros/humble/setup.bash' >> /etc/bash.bashrc
ENV NVIDIA_DRIVER_CAPABILITIES=all
