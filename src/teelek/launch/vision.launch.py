from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # ---------------- CAM 1 ----------------
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam1',
            output='screen',
            parameters=[{
                'video_device': '/dev/video2',
                'image_width': 320,
                'image_height': 240,
                'pixel_format': 'mjpeg2rgb',
                'framerate': 30.0,
                'io_method': 'mmap',
            }],
            remappings=[
                ('/image_raw', '/cam1/image_raw')
            ]
        ),

        # ---------------- CAM 2 ----------------
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam2',
            output='screen',
            parameters=[{
                'video_device': '/dev/video4',
                'image_width': 320,
                'image_height': 240,
                'pixel_format': 'mjpeg2rgb',
                'framerate': 30.0,
                'io_method': 'mmap',
            }],
            remappings=[
                ('/image_raw', '/cam2/image_raw')
            ]
        ),

        # ---------------- Vision Control ----------------
        Node(
            package='teelek',
            executable='vision_control.py',
            name='vision_control',
            output='screen',
            remappings=[
                ('/image_raw', '/cam1/image_raw')   # เลือกใช้ cam1
            ]
        )
    ])