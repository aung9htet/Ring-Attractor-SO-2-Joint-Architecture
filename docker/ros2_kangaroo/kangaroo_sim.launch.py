"""PAL's kangaroo_mujoco launch, unchanged, plus the one remap it needs on ROS 2 Humble.

On Humble, ros2_control_node reads ~/robot_description, but PAL's launch publishes
/robot_description without remapping it, so the controller manager waits forever.
Upstream's own demos add this remap on Humble only
(mujoco_ros2_control_demos/launch/01_basic_robot.launch.py).
Defaults: PAL's pregenerated 4-DoF model (floating base, no end-effector), MoveIt off.
Also starts PAL's own left and right arm trajectory controllers, unchanged (docs/kangaroo/DECISIONS.md D7).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import SetRemap
from launch_ros.substitutions import FindPackageShare


def pal_launch(package, name):
    return PathJoinSubstitution([FindPackageShare(package), 'launch', name])


def generate_launch_description():
    # Scoped per side so the two includes' `side` arguments do not overwrite each other.
    arm_controllers = [GroupAction([
        IncludeLaunchDescription(pal_launch('kangaroo_controller_configuration', 'arm_controller.launch.py'),
                                 launch_arguments={'side': side, 'arm_type': LaunchConfiguration('arm_type')}.items())])
        for side in ('left', 'right')]
    return LaunchDescription([
        DeclareLaunchArgument('arm_type', default_value='4dof'),
        DeclareLaunchArgument('moveit', default_value='False'),
        SetRemap(src='~/robot_description', dst='/robot_description'),
        IncludeLaunchDescription(pal_launch('kangaroo_mujoco', 'kangaroo_mujoco.launch.py')),
        *arm_controllers,
    ])
