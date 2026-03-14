from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    motion_node = Node(
        package='teelek',
        executable='motion_node.py',
        name='motion_node',
        output='screen'
    )

    mission_node = Node(
        package='teelek',
        executable='mission_node.py',
        name='mission_node',
        output='screen'
    )

    plant_node = Node(
        package='teelek',
        executable='plant_node.py',
        name='plant_node',
        output='screen'
    )

    scancabbage_node = Node(
        package='teelek',
        executable='scan_cabbage.py',
        name='scancabbage_node',
        output='screen'
    )

    return LaunchDescription([
        # motion_node,
        mission_node,
        plant_node,
        # scancabbage_node
    ])