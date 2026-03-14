from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    motion_node = Node(
        package='teelek',
        executable='motion_node.py',
        name='motion_node',
        output='screen'
    )

    cameraapriltag_node = Node(
        package='teelek',
        executable='camera_apriltag.py',
        name='cameraapriltag_node',
        output='screen'
    )

    tag_follower = Node(
        package='teelek',
        executable='tag_follower_node.py',
        name='tag_follower',
        output='screen'
    )

    robot_plotter = Node(
        package='teelek',
        executable='robot_plotter.py',
        name='robot_plotter',
        output='screen'
    )

    go_to_pose = Node(
        package='teelek',
        executable='go_to_pose.py',
        name='PoseGoTo',
        output='screen'
    )

    mission_pose = Node(
        package='teelek',
        executable='mission_pose.py',
        name='MissionNode',
        output='screen'
    )

    main_control = Node(
        package='teelek',
        executable='main_control.py',
        name='main_control',
        output='screen'
    )

    return LaunchDescription([
        motion_node,
        cameraapriltag_node,
        tag_follower,
        robot_plotter,
        go_to_pose,
        main_control,
        # mission_pose
    ])