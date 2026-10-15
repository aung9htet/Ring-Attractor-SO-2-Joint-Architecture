#!/usr/bin/env bash
# Smoke test: launch the 4-DoF Kangaroo sim and check it is usable for arm velocity control.
# From the repo root: docker/ros2_kangaroo/run.bash docker/ros2_kangaroo/smoke_test.bash
# Expected values come from PAL's 4-DoF model, not from our code. Exit code = number of failures.
set -u
launch=${LAUNCH:-ros2 launch /opt/kangaroo_sim.launch.py}
expected="arm_left_1_joint arm_left_2_joint arm_left_3_joint arm_left_4_joint arm_right_1_joint arm_right_2_joint arm_right_3_joint arm_right_4_joint"
fails=0
check() { if "${@:2}" >/dev/null 2>&1; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }

$launch >/tmp/launch.log 2>&1 &
sleep 45

arm_joints() { timeout 10 ros2 topic echo --once /joint_states --field name | grep -o 'arm_[a-z]*_[0-9]_joint' | sort | xargs; }
command_interfaces() {
    local ifaces; ifaces=$(timeout 15 ros2 control list_hardware_interfaces) || return 1
    for j in $expected; do for i in position velocity effort; do grep -q "$j/$i \[available\]" <<<"$ifaces" || return 1; done; done
}
sim_time_ns() { timeout 10 ros2 topic echo --once /clock --field clock | awk '/^ *sec:/{s=$2} /nanosec:/{n=$2} END{printf "%d%09d\n", s, n}'; }
clock_advances() { local t1 t2; t1=$(sim_time_ns); sleep 2; t2=$(sim_time_ns); ((t2 > t1)); }

check "1 /joint_states has exactly the 8 arm joints" test "$(arm_joints)" == "$expected"
check "2 arm joints accept position/velocity/effort" command_interfaces
check "3 joint_state_broadcaster active" bash -c 'timeout 15 ros2 control list_controllers | grep joint_state_broadcaster | grep -q active'
check "4 /clock advances (physics running)" clock_advances
check "5 ros2_control_node renders on NVIDIA GPU" bash -c "nvidia-smi | grep -qE ' G .*ros2_control_node'"

echo "$fails failure(s); launch log: /tmp/launch.log"
exit "$fails"
