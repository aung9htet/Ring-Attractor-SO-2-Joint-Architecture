# Source this file in every additional container terminal.
source /opt/ros/noetic/setup.bash
source /tiago_public_ws/devel/setup.bash
source /model_repo/.runtime/catkin_ws/devel/setup.bash
source /usr/local/nest/bin/nest_vars.sh
# Preserve the legacy source spelling even when Catkin traverses its workspace
# symlink. Some research helpers intentionally derive paths from __file__.
export PYTHONPATH="/tiago_public_ws/src/tiago_ring_controller/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
