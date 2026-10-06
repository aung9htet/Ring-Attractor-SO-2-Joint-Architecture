#!/usr/bin/env bash
# Headless containers for tests and experiments (no desktop, no interactive shell).
#
#   docker/batch.bash nest [--rw] [--out DIR] -- COMMAND     one-shot: NEST and the fake robot (unit tests, tuning, reports)
#   docker/batch.bash up NAME [--gpu] [--out DIR] [--world W]  start Gazebo with the TIAGo in the background, wait until it is ready
#   docker/batch.bash exec NAME -- COMMAND                   run a command against that simulation
#   docker/batch.bash down NAME                              stop it
#
# COMMAND is one shell string; it runs in src/tiago_ring_controller with ROS, the workspace
# and NEST sourced.  The package is mounted read-only unless --rw is given (needed only by
# commands that write into the package: installing an artifact, regenerating examples).
# --out DIR is mounted at /out.  Containers have no network, so several can run side by side,
# each with its own ROS master.  --gpu renders with NVIDIA through the desktop's X server;
# the recording camera (/recording_camera/image_raw) needs it, everything else does not.
# --world W picks the world: tiago_ring_controller (default, with the table) or
# tiago_ring_stress (no table, for scripts/stress_test.py).
#
# README.md, section "Setup", shows the usual calls.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$repo_dir/docker/image.env"
package=/tiago_public_ws/src/tiago_ring_controller
setup="source /opt/ros/noetic/setup.bash; source /tiago_public_ws/devel/setup.bash; source /usr/local/nest/bin/nest_vars.sh; export PYTHONPATH=$package/src:\$PYTHONPATH; cd $package"
banner='^\s*$|-- N E S T --|Copyright \(C\) 2004|Version: HEAD|Built: |This program is provided|NO WARRANTY|Problems or suggestions|Visit https://www.nest|Type .nest.help'
die() { echo "Error: $*" >&2; exit 1; }
usage() { sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

common=(--init --network none --shm-size 1g --user "$(id -u):$(id -g)"
        -e HOME=/tmp -e USER=model -e LOGNAME=model -e PYTHONDONTWRITEBYTECODE=1 -e MPLBACKEND=Agg
        -v "$repo_dir/docs:/model_docs:ro"
        -v "$repo_dir/docs:/model_repo_docs/docs:ro" -v "$repo_dir/README.md:/model_repo_docs/README.md:ro")

mode=${1:-}; shift || true
case "$mode" in
nest)
    access=ro; out=
    while (($#)); do
        case "$1" in
            --rw) access=rw; shift ;;
            --out) out=$2; shift 2 ;;
            --) shift; break ;;
            *) die "unknown option $1 (put -- before the command)" ;;
        esac
    done
    (($# == 1)) || die 'give one command string after --'
    mounts=(-v "$repo_dir/src/tiago_ring_controller:$package:$access")
    [[ -z $out ]] || { mkdir -p "$out"; mounts+=(-v "$(cd "$out" && pwd -P):/out"); }
    docker run --rm -i "${common[@]}" "${mounts[@]}" --entrypoint bash "$MODEL_IMAGE" -c "$setup && $1" 2>&1 \
        | grep --line-buffered -v -E "$banner" || true
    ;;
up)
    name=${1:-}; [[ -n $name ]] || die 'up needs a container name'; shift
    gpu=0; out=; world=tiago_ring_controller
    while (($#)); do
        case "$1" in
            --gpu) gpu=1; shift ;;
            --out) out=$2; shift 2 ;;
            --world) world=$2; shift 2 ;;
            *) die "unknown option $1" ;;
        esac
    done
    mounts=(-v "$repo_dir/src/tiago_ring_controller:$package:ro")
    [[ -z $out ]] || { mkdir -p "$out"; mounts+=(-v "$(cd "$out" && pwd -P):/out"); }
    while IFS= read -r relative; do
        [[ -n $relative && $relative != \#* ]] || continue
        mounts+=(-v "$repo_dir/overrides/$relative:/tiago_public_ws/src/$relative:ro")
    done < "$repo_dir/docker/overrides.list"
    if ((gpu)); then
        [[ -n ${DISPLAY:-} ]] || die '--gpu needs a desktop session (DISPLAY)'
        xauth_file=$(mktemp /tmp/tiago-ring-xauth.XXXXXX)
        xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$xauth_file" nmerge -
        chmod 644 "$xauth_file"
        graphics=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display
                  -e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia
                  -e "DISPLAY=$DISPLAY" -e XAUTHORITY=/tmp/x.xauth -e QT_X11_NO_MITSHM=1
                  -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v "$xauth_file:/tmp/x.xauth:ro")
    else
        graphics=(-e DISPLAY= -e XAUTHORITY= -e LIBGL_ALWAYS_SOFTWARE=1 -e NVIDIA_VISIBLE_DEVICES=void)
    fi
    docker run -d --rm --name "$name" "${common[@]}" "${graphics[@]}" "${mounts[@]}" \
        -e ROS_MASTER_URI=http://localhost:11311 -e ROS_HOSTNAME=localhost -e GAZEBO_MODEL_DATABASE_URI= \
        --entrypoint bash "$MODEL_IMAGE" -c "source /opt/ros/noetic/setup.bash; source /tiago_public_ws/devel/setup.bash; exec roslaunch tiago_gazebo tiago_gazebo.launch world:=$world gui:=false public_sim:=true arm:=true end_effector:=pal-hey5 ft_sensor:=schunk-ft" >/dev/null
    # ready when the arm controller runs; then the start-up motion needs about 30 s more
    docker exec -i "$name" bash -c 'source /opt/ros/noetic/setup.bash; python3 -' <<'PY'
import subprocess, time
started = time.time()
while time.time() - started < 300:
    out = subprocess.run("rosservice call /controller_manager/list_controllers 2>/dev/null", shell=True, capture_output=True, text=True).stdout
    if "arm_controller" in out and out.count("running") >= 4:
        break
    time.sleep(3)
else:
    raise SystemExit("the arm controller did not come up in 300 s")
time.sleep(30)
print("simulation ready after %.0f s" % (time.time() - started))
PY
    ;;
exec)
    name=${1:-}; [[ -n $name ]] || die 'exec needs a container name'; shift
    [[ ${1:-} == -- ]] && shift
    (($# == 1)) || die 'give one command string after --'
    docker exec -i "$name" bash -c "$setup && $1" 2>&1 | grep --line-buffered -v -E "$banner" || true
    ;;
down)
    name=${1:-}; [[ -n $name ]] || die 'down needs a container name'
    docker stop "$name" >/dev/null && echo "stopped $name"
    ;;
-h|--help|'') usage ;;
*) die "unknown mode $mode (nest, up, exec, down)" ;;
esac
