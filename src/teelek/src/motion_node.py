#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, Bool
from geometry_msgs.msg import Twist, Pose2D


def mm_to_cm(distance: float) -> float:
    return distance / 10.0


class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")

        # ---------------- Publishers ----------------
        self.pub_robot_distance = self.create_publisher(
            Float32, "/teelek/robot_distance", 10)

        self.cmd_pub = self.create_publisher(
            Twist, '/teelek/cmd_move', 10)

        self.done_pub = self.create_publisher(
            Bool, '/teelek/motion_done', 10)
        
        self.done_pub = self.create_publisher(
            Bool, "/teelek/move_done", 10)

        # ---------------- Subscribers ----------------
        self.create_subscription(
            Float32MultiArray,
            "/teelek/debug/encoder_wheels",
            self.encoder_callback,
            10)

        self.create_subscription(
            Float32,
            "/teelek/move_distance",
            self.move_callback,
            10)

        self.create_subscription(
            Float32,
            "/teelek/rotate_angle",
            self.rotate_callback,
            10)

        self.create_subscription(
            Pose2D,
            "/teelek/robot_pose",
            self.pose_callback,
            10)

        self.timer = self.create_timer(0.1, self.control_loop)

        # ---------------- Encoder variables ----------------
        self.robot_distance = 0.0
        self.prev_wheel_tick = [0, 0, 0, 0]

        # ---------------- Move control ----------------
        self.target_distance = 0.0
        self.moving = False

        # ---------------- Rotate control (Vision) ----------------
        self.robot_theta = 0.0
        self.target_theta = 0.0
        self.rotating = False
        self.rotate_tolerance = math.radians(3)

        # ---------------- Control parameters ----------------
        self.Kp = 50.0
        self.max_pwm = 1023.0
        self.deadband = 0.5 #cm

        self.Kp_rotate = 2.5
        self.max_angular = 1.0

    # =========================================================
    # Callbacks
    # =========================================================

    def pose_callback(self, msg: Pose2D):
        self.robot_theta = msg.theta

    def move_callback(self, msg):
        self.target_distance = self.robot_distance + msg.data
        self.moving = True
        self.get_logger().info(f"Move command received: {msg.data:.3f} cm")

    def rotate_callback(self, msg):
        self.target_theta = self.normalize_angle(self.robot_theta + msg.data)
        self.rotating = True
        self.get_logger().info(
            f"Rotate command received: {math.degrees(msg.data):.2f} deg")

    def encoder_callback(self, msg: Float32MultiArray):
        tick_per_revolution = 541.0168 * 11 * 4
        diameter = 130
        circumference = math.pi * diameter

        ticks = [msg.data[0], msg.data[1], msg.data[2], msg.data[3]]
        delta_ticks = [ticks[i] - self.prev_wheel_tick[i] for i in range(4)]
        self.prev_wheel_tick = ticks

        avg_delta = sum(delta_ticks) / 4.0
        self.robot_distance += mm_to_cm(
            (avg_delta / tick_per_revolution) * circumference)

    # =========================================================
    # Control Loop
    # =========================================================

    def control_loop(self):

        # Publish current distance
        robot_msg = Float32()
        robot_msg.data = float(self.robot_distance)
        self.pub_robot_distance.publish(robot_msg)

        cmd = Twist()

        # ================= ROTATE MODE =================
        if self.rotating:
            error = self.normalize_angle(self.target_theta - self.robot_theta)

            if abs(error) < self.rotate_tolerance:
                cmd.angular.z = 0.0
                self.rotating = False
                self.publish_done()
                self.get_logger().info("Rotation complete")

            else:
                angular = self.Kp_rotate * error
                angular = max(-self.max_angular,
                              min(self.max_angular, angular))
                cmd.angular.z = angular

        # ================= MOVE MODE =================
        elif self.moving:
            error = self.target_distance - self.robot_distance

            if abs(error) <= self.deadband:
                cmd.linear.x = 0.0
                self.moving = False
                self.publish_done()
                self.get_logger().info("Reached target distance")

            else:
                pwm = self.Kp * error
                min_pwm = 550.0

                if abs(pwm) < min_pwm:
                    pwm = min_pwm if pwm > 0 else -min_pwm

                pwm = max(-self.max_pwm, min(self.max_pwm, pwm))
                cmd.linear.x = pwm

        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    # =========================================================

    def publish_done(self):
        done_msg = Bool()
        done_msg.data = True
        self.done_pub.publish(done_msg)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))


def main():
    rclpy.init()
    node = MotionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()