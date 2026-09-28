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
mkdir -p "$workspace/src"
package_link=$workspace/src/tiago_ring_controller
if [[ -L $package_link ]]; then
    [[ $(readlink "$package_link") == /tiago_public_ws/src/tiago_ring_controller ]] || { echo 'Unexpected controller workspace symlink' >&2; exit 1; }
elif [[ -e $package_link ]]; then
    echo 'Expected a symlink for the controller in the runtime workspace' >&2
    exit 1
else
    ln -s /tiago_public_ws/src/tiago_ring_controller "$package_link"
fi
exec 9>/model_repo/.runtime/build.lock
flock 9
catkin config --workspace "$workspace" --extend /tiago_public_ws/devel --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3 >/dev/null
if ! catkin build --workspace "$workspace" tiago_ring_controller --no-deps -j2 -p1 --no-status; then
    echo 'Controller build failed. See .runtime/catkin_ws/logs; source files have not been replaced.' >&2
    exit 1
fi
flock -u 9
exec 9>&-
source /model_repo/docker/environment.bash
cd /tiago_public_ws/src/tiago_ring_controller/src
exec "$@"
