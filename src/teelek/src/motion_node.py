#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, Bool
from geometry_msgs.msg import Twist, Pose2D


def mm_to_m(distance: float) -> float:
    return distance / 1000.0


class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")

        # ---------------- Publishers ----------------
        self.pub_robot_distance = self.create_publisher(
            Float32, "/teelek/robot_distance", 10)

        self.cmd_pub = self.create_publisher(
            Twist, '/teelek/cmd_move', 10)

        self.motion_done_pub = self.create_publisher(
            Bool, '/teelek/motion_done', 10)
        
        self.move_done_pub = self.create_publisher(
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

        self.timer = self.create_timer(0.1, self.control_loop)

        # ---------------- Encoder variables ----------------
        self.robot_distance = 0.0
        self.prev_wheel_tick = [0, 0, 0, 0]

        # ---------------- Move control ----------------
        self.target_distance = 0.0
        self.moving = False

        # ---------------- Control parameters ----------------
        self.Kp = 50.0
        self.max_pwm = 1023.0
        self.deadband = 0.005 #m

        self.Kp_rotate = 2.5
        self.max_angular = 1.0

    # =========================================================
    # Callbacks
    # =========================================================
    def move_callback(self, msg):
        self.target_distance = self.robot_distance + msg.data
        self.moving = True
        self.get_logger().info(f"Move command received: {msg.data:.3f} m")

    def encoder_callback(self, msg: Float32MultiArray):

        tick_per_revolution = 541.0168 * 11 * 4

        diameter = 0.13  # meter
        circumference = math.pi * diameter

        ticks = [msg.data[0], msg.data[1], msg.data[2], msg.data[3]]

        delta_ticks = [ticks[i] - self.prev_wheel_tick[i] for i in range(4)]
        self.prev_wheel_tick = ticks

        avg_delta = sum(delta_ticks) / 4.0

        self.robot_distance += (
            (avg_delta / tick_per_revolution) * circumference
        )

    def move_distance_callback(self, msg):

        distance = msg.data

        target = Pose2D()

        target.x = self.internal_x + distance * math.cos(self.internal_theta)
        target.y = self.internal_y + distance * math.sin(self.internal_theta)
        target.theta = self.internal_theta

        self.move_callback(target)

        self.get_logger().info(
            f"Distance Move Command: {distance:.3f} m -> "
            f"Target XY ({target.x:.2f}, {target.y:.2f})"
        )

    # =========================================================
    # Control Loop
    # =========================================================

    def control_loop(self):

        # Publish current distance
        robot_msg = Float32()
        robot_msg.data = float(self.robot_distance)
        self.pub_robot_distance.publish(robot_msg)

        cmd = Twist()
        if self.moving:

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
            cmd.linear.y = 0.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    # =========================================================

    def publish_done(self):
        msg = Bool()
        msg.data = True
        self.move_done_pub.publish(msg)
        self.motion_done_pub.publish(msg)

def main():
    rclpy.init()
    node = MotionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()