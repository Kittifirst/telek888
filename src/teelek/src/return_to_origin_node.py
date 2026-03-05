#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32, Bool
import math

class ReturnToOrigin(Node):

    def __init__(self):
        super().__init__('return_to_origin')

        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

        # State machine
        self.state = 0
        self.waiting_motion = False
        self.tolerance = 0.02  # 2 cm

        # Subscribers
        self.create_subscription(Pose2D, '/teelek/robot_pose', self.pose_callback, 10)
        self.create_subscription(Bool, '/teelek/motion_done', self.motion_done_callback, 10)

        # Publishers (ไป motion node)
        self.move_pub = self.create_publisher(Float32, '/teelek/move_distance', 10)
        self.rotate_pub = self.create_publisher(Float32, '/teelek/rotate_angle', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.run_state)

        self.get_logger().info("Return To Origin Node Ready")

        # Pose Flag
        self.pose_ready = False

    def pose_callback(self, msg):
        self.robot_x = msg.x
        self.robot_y = msg.y
        self.robot_theta = msg.theta
        self.pose_ready = True

    def motion_done_callback(self, msg):
        if msg.data:
            self.waiting_motion = False
        
    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def run_state(self):

        if self.waiting_motion:
            return
        
        if not self.pose_ready:
            return

        # STATE 0 → รอข้อมูลก่อน
        if self.state == 0:
            self.get_logger().info("Start Y alignment")
            self.state = 1

        # STATE 1 → เดินแกน Y
        elif self.state == 1:
            if abs(self.robot_y) > self.tolerance:
                distance = -self.robot_y * 100
                self.send_move(distance)
            else:
                self.get_logger().info("Y aligned")
                self.state = 2

        # STATE 2 → หมุนให้หน้าแกน X (0 rad)
        elif self.state == 2:
            target_angle = 0.0
            error = self.normalize_angle(target_angle - self.robot_theta)

            if abs(error) > math.radians(3):
                self.send_rotate(error)
            else:
                self.get_logger().info("Rotation aligned")
                self.state = 3

        # STATE 3 → เดินแกน X
        elif self.state == 3:
            if abs(self.robot_x) > self.tolerance:
                distance = -self.robot_x * 100
                self.send_move(distance)
            else:
                self.get_logger().info("Arrived at Origin (0,0)")
                self.state = 4

    def send_move(self, distance):
        msg = Float32()
        msg.data = float(distance)
        self.move_pub.publish(msg)
        self.waiting_motion = True
        self.get_logger().info(f"Move {distance:.3f} m")

    def send_rotate(self, angle):
        msg = Float32()
        msg.data = float(angle)
        self.rotate_pub.publish(msg)
        self.waiting_motion = True
        self.get_logger().info(f"Rotate {math.degrees(angle):.2f} deg")


def main(args=None):
    rclpy.init(args=args)
    node = ReturnToOrigin()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()