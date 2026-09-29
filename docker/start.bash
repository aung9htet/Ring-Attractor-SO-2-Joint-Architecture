#!/usr/bin/env bash
set -eo pipefail
if [[ $(id -u) == 0 && $LOCAL_USER_ID != 0 ]]; then
    getent group "$LOCAL_GROUP_ID" >/dev/null || groupadd -g "$LOCAL_GROUP_ID" model
    getent passwd "$LOCAL_USER_ID" >/dev/null || useradd -M -u "$LOCAL_USER_ID" -g "$LOCAL_GROUP_ID" -d "$HOME" -s /bin/bash model
    exec setpriv --reuid "$LOCAL_USER_ID" --regid "$LOCAL_GROUP_ID" --clear-groups /bin/bash "$0" "$@"
fi
source /opt/ros/noetic/setup.bash
source /tiago_public_ws/devel/setup.bash
source /usr/local/nest/bin/nest_vars.sh
workspace=/model_repo/.runtime/catkin_ws
mkdir -p "$workspace"
exec 9>/model_repo/.runtime/build.lock
flock 9
# Use the original source path so Catkin's generated test environment and Python
# relay modules also preserve __file__-based research paths. Only the controller
# is built; all upstream packages remain supplied by the image's underlay.
catkin config --workspace "$workspace" --source-space /tiago_public_ws/src \
    --extend /tiago_public_ws/devel --buildlist tiago_ring_controller \
    --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3 >/dev/null
if ! catkin build --workspace "$workspace" tiago_ring_controller --no-deps -j2 -p1 --no-status; then
    echo 'Controller build failed. See .runtime/catkin_ws/logs; source files have not been replaced.' >&2
    exit 1
fi
flock -u 9
exec 9>&-
source /model_repo/docker/environment.bash
cd /tiago_public_ws/src/tiago_ring_controller/src
exec "$@"
