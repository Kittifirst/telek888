from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    # Launch RackDistance node
    node_rack_distance = Node(
        package="teelek",        
        executable="robot_movement.py",  
        name="rack_distance_node",
        output="screen"
    )

    ld.add_action(node_rack_distance)

    return ld
