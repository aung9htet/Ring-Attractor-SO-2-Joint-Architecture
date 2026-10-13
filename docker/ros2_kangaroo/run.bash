#!/usr/bin/env bash
# Start the ROS 2 Humble + Kangaroo MuJoCo container: GPU, display, $PWD mounted at /work.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
image=ros2-kangaroo:humble
die() { echo "Error: $*" >&2; exit 1; }

docker image inspect "$image" >/dev/null 2>&1 || docker build -t "$image" "$here"

args=(run --rm --init --name ros2-kangaroo --network host --ipc host
      --user "$(id -u):$(id -g)" -e HOME=/tmp
      -v "$PWD:/work" -w /work)
[[ -t 0 && -t 1 ]] && args+=(-it)

# GPU: same flags as run_model_docker.bash (PRIME offload for hybrid laptops).
if docker info --format '{{json .Runtimes}}' | grep -q nvidia && nvidia-smi >/dev/null 2>&1; then
    args+=(--gpus all -e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia)
fi

# Display: MuJoCo's viewer is X11 (GLFW); on Wayland this goes through XWayland.
[[ -n ${DISPLAY:-} ]] || die 'DISPLAY is not set; run from a desktop terminal.'
command -v xauth >/dev/null || die 'Install xauth on the host.'
xauth_file=$(mktemp /tmp/ros2-kangaroo-xauth.XXXXXX); trap 'rm -f "$xauth_file"' EXIT
xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$xauth_file" nmerge -
args+=(-e DISPLAY -e XAUTHORITY=/tmp/.xauth
       -v "$xauth_file:/tmp/.xauth:ro" -v /tmp/.X11-unix:/tmp/.X11-unix:ro)

docker "${args[@]}" "$image" "${@:-bash}"
