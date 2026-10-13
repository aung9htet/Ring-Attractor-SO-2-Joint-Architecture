"""PAL's kangaroo_mujoco launch, unchanged, plus the one remap it needs on ROS 2 Humble.

On Humble, ros2_control_node reads ~/robot_description, but PAL's launch publishes
/robot_description without remapping it, so the controller manager waits forever.
Upstream's own demos add this remap on Humble only
(mujoco_ros2_control_demos/launch/01_basic_robot.launch.py).
Defaults: PAL's pregenerated 4-DoF model (floating base, no end-effector), MoveIt off.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pal_launch = PathJoinSubstitution([FindPackageShare('kangaroo_mujoco'), 'launch', 'kangaroo_mujoco.launch.py'])
    return LaunchDescription([
        DeclareLaunchArgument('arm_type', default_value='4dof'),
        DeclareLaunchArgument('moveit', default_value='False'),
        SetRemap(src='~/robot_description', dst='/robot_description'),
        IncludeLaunchDescription(pal_launch),
    ])
