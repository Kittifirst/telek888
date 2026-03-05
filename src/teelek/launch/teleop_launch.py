import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()

    # --- Joy Node ---
    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="Joy_Node",
        output="screen",
        # arguments=["--dev", "/dev/input/js0"],  # ใส่ถ้าใช้ joystick device
    )

    # --- Joystick control node ---
    xbox_control = Node(
        package="teelek",
        executable="xbox_control.py",
        name="Joystick_Node",
        output="screen",
    )
    
    # --- Joystick control node ---
    play_control = Node(
        package="teelek",
        executable="play_control.py",
        name="Joystick_Node",
        output="screen",
    )

    # --- Robot movement node ---
    robot_movement = Node(
        package="teelek",
        executable="robot_movement.py",
        name="rack_distance_node",
        output="screen",
    )

    # --- micro-ROS Agent nodes ---
    microros_agent_0 = Node(
        package="micro_ros_agent",
        executable="micro_ros_agent",
        output="screen",
        arguments=["serial", "--dev", "/dev/ttyACM0"],
    )
    microros_agent_1 = Node(
        package="micro_ros_agent",
        executable="micro_ros_agent",
        output="screen",
        arguments=["serial", "--dev", "/dev/ttyACM1"],
    )

    # เพิ่มทุก Node เข้า LaunchDescription
    ld.add_action(joy_node)
    # ld.add_action(xbox_control)
    ld.add_action(play_control)
    ld.add_action(robot_movement)
    ld.add_action(microros_agent_0)
    # ถ้าต้องการ micro-ROS agent อีกตัวก็เปิด:
    # ld.add_action(microros_agent_1)

    return ld
