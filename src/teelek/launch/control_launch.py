import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()

    package_dir = get_package_share_directory('teelek')

    # motor_config = os.path.join(
    #     get_package_share_directory('teelek'),
    #     'config',
    #     'motor_config.yaml'
    # )

    joy = Node(
        package="joy",
        executable="joy_node",
        name="Joy_Node",
        output="screen",
        # namespace="",
        # parameters=[{"autorepeat_rate": 50.0}],
        # arguments=["--dev", "/dev/input/js0"],  # replace with your joystick device path
        # remappings = [
        #     ('/joy', '/teelek/joy')
        # ]
    )   

    joystick_control = Node(
        package="teelek",
        executable="play_control.py",
        name="Joystick_Node",
        # output="screen",
        namespace="",
    )

    # movement_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(package_dir, 'launch', 'robot_movement_launch.py')
    #     )
    # )
    
    
    ld.add_action(joy)
    ld.add_action(joystick_control)
    # ld.add_action(movement_launch)

    return ld