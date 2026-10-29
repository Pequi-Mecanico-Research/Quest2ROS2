"""Full Quest teleop: ROS-TCP endpoint + right arm controller + mobile base joy bridge.

The endpoint starts first; the arm controller and the base bridge only start after
wait_for_quest receives the first Quest message (i.e. the VR is actually connected).
If the wait times out, the whole launch shuts down.

    ros2 launch q2r2_bringup quest_teleop.launch.py
    ros2 launch q2r2_bringup quest_teleop.launch.py base:=false
    ros2 launch q2r2_bringup quest_teleop.launch.py quest_timeout:=60.0
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def include(package, launch_file, **kwargs):
    path = os.path.join(get_package_share_directory(package), 'launch', launch_file)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path), **kwargs)


def generate_launch_description():
    # Quest <-> ROS bridge (Unity ROS-TCP)
    endpoint = include('ros_tcp_endpoint', 'endpoint.py')

    wait_for_quest = Node(
        package='q2r2_bringup',
        executable='wait_for_quest',
        parameters=[{
            'topic': LaunchConfiguration('quest_topic'),
            'timeout': LaunchConfiguration('quest_timeout'),
        }],
        output='screen',
    )

    teleop = [
        # Arm controller (root namespace, /servo_node/..., /tf)
        Node(
            package='q2r2_bringup',
            executable='right_arm_controller',
            output='screen',
        ),
        # Mobile base
        include(
            'q2r_base_bridge', 'q2r_joy_bridge.launch.py',
            condition=IfCondition(LaunchConfiguration('base')),
        ),
    ]

    def on_wait_exit(event, _context):
        if event.returncode == 0:
            return teleop
        return [
            LogInfo(msg='Quest não conectou; encerrando o launch.'),
            EmitEvent(event=Shutdown(reason='Quest not connected')),
        ]

    return LaunchDescription([
        DeclareLaunchArgument(
            'base',
            default_value='true',
            choices=('true', 'false'),
            description='Also start the Quest -> mobile base joy bridge.',
        ),
        DeclareLaunchArgument(
            'quest_topic',
            default_value='/q2r_right_hand_pose',
            description='Topic whose first message means the Quest is connected.',
        ),
        DeclareLaunchArgument(
            'quest_timeout',
            default_value='0.0',
            description='Seconds to wait for the Quest before giving up (<= 0 waits forever).',
        ),

        endpoint,
        wait_for_quest,
        RegisterEventHandler(OnProcessExit(target_action=wait_for_quest, on_exit=on_wait_exit)),
    ])
