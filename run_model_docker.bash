#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$repo_dir/docker/image.env"
headless=0
software=0
container_name="tiago-ring-$(id -u)-$$"
usage() {
    cat <<'HELP'
Usage: ./run_model_docker.bash [options] [-- COMMAND ...]
  --headless       Disable desktop forwarding (Gazebo also needs gui:=false)
  --software       Use software rendering instead of NVIDIA
  --name NAME      Set a container name for additional terminals
  -h, --help       Show this help
Default: build the mounted controller incrementally and open a ready shell.
HELP
}
die() { echo "Error: $*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --headless) headless=1; shift ;;
        --software) software=1; shift ;;
        --name) [[ $# -ge 2 && -n $2 ]] || die '--name needs a value'; container_name=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        *) die "Unknown option: $1 (use -- before a container command)" ;;
    esac
done
command -v docker >/dev/null || die 'Docker Engine is required.'
docker info >/dev/null 2>&1 || die 'Cannot access Docker Engine. Check the daemon and your Docker group membership.'
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die 'This environment requires Linux amd64.'
[[ -f $repo_dir/src/tiago_ring_controller/package.xml ]] || die 'Recovered controller package is missing; startup never copies source from the image.'
[[ $repo_dir != *,* ]] || die 'Docker mount paths cannot contain commas; move the checkout to another path.'
if docker container inspect "$container_name" >/dev/null 2>&1; then
    die "Container $container_name already exists. Choose another --name or enter the existing container as documented."
fi
xauth_file=
cleanup() { [[ -z $xauth_file ]] || rm -f -- "$xauth_file"; }
trap cleanup EXIT
args=(run --rm --init --name "$container_name" --network host --shm-size 1g
      --mount "type=bind,src=$repo_dir,dst=/model_repo"
      --mount "type=bind,src=$repo_dir/src/tiago_ring_controller,dst=/tiago_public_ws/src/tiago_ring_controller"
      --mount "type=bind,src=$repo_dir/recovered/experiment_results,dst=/tiago_public_ws/src/experiment_results"
      --workdir /tiago_public_ws/src/tiago_ring_controller/src
      --entrypoint /bin/bash
      -e "LOCAL_USER_ID=$(id -u)" -e "LOCAL_GROUP_ID=$(id -g)"
      -e HOME=/model_repo/.runtime/home -e USER=model -e LOGNAME=model
      -e PYTHONDONTWRITEBYTECODE=1
      -e "ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}")
for variable in ROS_IP ROS_HOSTNAME; do
    [[ -z ${!variable:-} ]] || args+=(-e "$variable=${!variable}")
done
while IFS= read -r relative; do
    [[ -n $relative && $relative != \#* ]] || continue
    [[ -f $repo_dir/overrides/$relative ]] || die "Missing override: $relative"
    args+=(--mount "type=bind,src=$repo_dir/overrides/$relative,dst=/tiago_public_ws/src/$relative,readonly")
done < "$repo_dir/docker/overrides.list"
[[ -t 0 && -t 1 ]] && args+=(-it)
if (( ! software )) && command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 && docker info --format '{{json .Runtimes}}' | grep -q nvidia; then
    args+=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display
           -e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia)
    graphics=nvidia
else
    args+=(-e NVIDIA_VISIBLE_DEVICES=void -e LIBGL_ALWAYS_SOFTWARE=1)
    graphics=software
fi
if ((headless)); then
    args+=(-e DISPLAY= -e XAUTHORITY= -e MPLBACKEND=Agg)
else
    [[ -n ${DISPLAY:-} ]] || die 'DISPLAY is missing. Run from a desktop terminal or use --headless.'
    command -v xauth >/dev/null || die 'Install xauth on the host, or use --headless.'
    [[ -d /tmp/.X11-unix ]] || die 'X11/XWayland socket directory is missing; use --headless.'
    xauth_file=$(mktemp /tmp/tiago-ring-xauth.XXXXXX)
    cookie=$(xauth nlist "$DISPLAY" 2>/dev/null || true)
    [[ -n $cookie ]] || die 'No X11 authorization cookie found. Check DISPLAY and XAUTHORITY, or use --headless.'
    printf '%s\n' "$cookie" | sed 's/^..../ffff/' | xauth -f "$xauth_file" nmerge -
    args+=(-e "DISPLAY=$DISPLAY" -e XAUTHORITY=/tmp/tiago-ring.xauth -e QT_X11_NO_MITSHM=1
           --mount 'type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly'
           --mount "type=bind,src=$xauth_file,dst=/tmp/tiago-ring.xauth,readonly")
fi
if ! docker image inspect "$MODEL_IMAGE" >/dev/null 2>&1; then
    docker pull "$MODEL_IMAGE" || die 'Cannot pull the pinned image. Check network access and Docker Hub login.'
fi
mkdir -p "$repo_dir/.runtime/home" "$repo_dir/.runtime/catkin_ws/src"
(($#)) || set -- bash --rcfile /model_repo/docker/bashrc -i
echo "Starting $container_name ($graphics); mounted repository: $repo_dir"
docker "${args[@]}" "$MODEL_IMAGE" /model_repo/docker/start.bash "$@"
